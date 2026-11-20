"""Opt-in GPU checks for a built example. No model requests or performance assertions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def parse_sample(stdout, repetitions):
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise ValueError('expected one JSON line')
    row = json.loads(lines[0])
    if row.get('correct') is not True or row.get('elements') != 4096 or row.get('repetitions') != repetitions:
        raise ValueError('incorrect sample result')
    for name in ('direct_event_ms', 'direct_wall_ms', 'graph_event_ms', 'graph_wall_ms'):
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid timing: ' + name)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    source = Path(__file__).resolve().parent
    metadata = json.loads((build / 'build.json').read_text())
    if not re.fullmatch(r'sm_[0-9]+[af]?', metadata['arch']):
        parser.error('invalid build architecture')
    binary = build / 'cuda-runtime-check'
    for name, digest in metadata['sources'].items():
        if name not in ('kernel.cu', 'main.rs', 'build.py'):
            parser.error('unexpected source entry')
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            parser.error('source differs from build: ' + name)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != metadata['binary_sha256']:
        parser.error('binary differs from build receipt')
    rows = []
    for repetitions in (1, 17, 1000):
        for _ in range(3):
            result = subprocess.run([str(binary), str(repetitions)], check=True,
                                    capture_output=True, text=True, timeout=60)
            rows.append(parse_sample(result.stdout, repetitions))
    for argument in ('0', '16777217', 'not-an-integer'):
        result = subprocess.run([str(binary), argument], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 or result.stdout or 'Error:' not in result.stderr:
            raise RuntimeError('invalid CLI argument was not rejected')
    result = subprocess.run([str(binary), '1'], env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                            capture_output=True, text=True, timeout=30)
    if result.returncode == 0 or result.stdout or 'CUDA error' not in result.stderr:
        raise RuntimeError('no-device initialization was not reported as a Rust error')
    lifecycle = build / 'cuda-lifecycle-check'
    subprocess.run(['nvcc', '-std=c++17', '-O1', '-arch=' + metadata['arch'], '--fmad=false',
                    str(source / 'lifecycle.cu'), '-o', str(lifecycle)], check=True, timeout=120)
    result = subprocess.run([str(lifecycle)], check=True, capture_output=True, text=True, timeout=120)
    checked = json.loads(result.stdout)
    expected = dict(correct=True, lifecycle_cycles=40, correctness_checks=240,
                    injected_setup_failures=6, live_resources=0)
    if checked != expected:
        raise RuntimeError('unexpected lifecycle result')
    receipt = {'runtime_measured': True, 'arch': metadata['arch'], 'build': metadata,
               'sample_runs': rows, 'invalid_cli_cases': 3, 'no_device_error_passed': True,
               'lifecycle': checked, 'performance_claim': False,
               'validation_sources': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                      for name in ('validate.py', 'lifecycle.cu')},
               'lifecycle_binary_sha256': hashlib.sha256(lifecycle.read_bytes()).hexdigest()}
    (build / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
