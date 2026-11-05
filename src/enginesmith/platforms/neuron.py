"""Linux on AWS Trainium (Neuron devices): HBM per NeuronCore, NKI kernels compiled to NEFF, run through libnrt."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

NAME = 'linux-neuron'
DEFAULT_REFERENCE = 'nxdi'
NEURON_BIN = Path('/opt/aws/neuron/bin')
ELF = b'\x7fELF'
# Libraries an engine may link: the C/C++ runtime, the Neuron driver interface and runtime (libnrt and the
# collectives it loads). Anything that knows the model or does numerics for it (Python, PyTorch, XLA, JAX, vLLM,
# NxD) is refused.
REFUSED = ('python', 'torch', 'xla', 'jax', 'vllm', 'neuronx_distributed', 'tensorflow', 'onnx', 'ggml')


def _run(argv, timeout=20) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _neuron_ls() -> list[dict]:
    tool = shutil.which('neuron-ls') or str(NEURON_BIN / 'neuron-ls')
    out = _run([tool, '--json-output'], timeout=30)
    try:
        devices = json.loads(out) if out else []
    except json.JSONDecodeError:
        return []
    return devices if isinstance(devices, list) else []


def supported() -> bool:
    return platform.system() == 'Linux' and Path('/sys/devices/virtual/neuron_device/neuron0').exists()


def _meminfo(key: str) -> int | None:
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(key + ':'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def machine() -> dict:
    devices = _neuron_ls()
    first = devices[0] if devices else {}
    cores = sum(d.get('nc_count', 0) for d in devices) or None
    per_core = first['memory_size'] // first['nc_count'] if first.get('memory_size') and first.get('nc_count') else None
    instance = first.get('instance_type')
    os_name = _run(['sh', '-c', '. /etc/os-release && echo "$PRETTY_NAME"']) or f'Linux {platform.release()}'
    return {'platform': NAME, 'os': os_name, 'arch': platform.machine(), 'chip': instance,
            'accelerator': f'{len(devices)} x Trainium ({instance})' if devices else 'Neuron device',
            'accelerator_cores': cores,
            # An engine runs on one logical NeuronCore: its share of the device's HBM is what a model can use.
            'memory_bytes': per_core, 'host_memory_bytes': _meminfo('MemTotal'),
            'neuron_devices': len(devices), 'logical_neuroncore_config': first.get('logical_neuroncore_config'),
            'device_memory_bytes': first.get('memory_size'),
            'toolchain': {'cargo': shutil.which('cargo'), 'cxx': shutil.which('g++') or shutil.which('clang++'),
                          'neuronx_cc': shutil.which('neuronx-cc'),
                          'libnrt': str(p) if (p := Path('/opt/aws/neuron/lib/libnrt.so')).exists() else None}}


def checks(facts: dict) -> tuple[list[str], list[str]]:
    problems, notes = [], []
    tools = facts['toolchain']
    if not facts.get('neuron_devices'):
        problems.append('neuron-ls lists no Neuron device: install the Neuron driver (aws-neuronx-dkms).')
    if not tools['libnrt']:
        problems.append('libnrt is missing (/opt/aws/neuron/lib/libnrt.so): install aws-neuronx-runtime-lib.')
    if not tools['cxx'] and not tools['cargo']:
        problems.append('Neither a C++ compiler nor cargo is on PATH; an engine needs one for its host code.')
    if not tools['neuronx_cc']:
        notes.append('neuronx-cc is not on PATH; engines that compile NKI kernels at build time need it (put a '
                     'Neuron virtual environment\'s bin directory on PATH).')
    return problems, notes


def memory_budget() -> int | None:
    """90% of one logical NeuronCore's HBM: the engine and the reference each run on one core."""
    per_core = machine()['memory_bytes']
    return int(per_core * 0.9) if per_core else None


def memory_critical() -> bool:
    """Host memory almost exhausted: MemAvailable below 3% of MemTotal."""
    total, available = _meminfo('MemTotal'), _meminfo('MemAvailable')
    return bool(total and available is not None and available < total * 0.03)


def engine_environment() -> dict:
    """The NeuronCores of this run: ENGINESMITH_NEURON_CORES logical cores (default 2) from ENGINESMITH_NEURON_CORE
    (default 0). With a reference of tensor-parallel degree N the run needs 2N: the engine uses the first N, the
    reference the first N and, for judging, the next N. Runs started with different values can share the machine.
    The Neuron runtime otherwise claims every core it can see."""
    base, count = (os.environ.get(name, default) for name, default in
                   (('ENGINESMITH_NEURON_CORE', '0'), ('ENGINESMITH_NEURON_CORES', '2')))
    if not base.isdigit() or not count.isdigit() or int(count) < 1:
        raise ValueError(f'ENGINESMITH_NEURON_CORE and ENGINESMITH_NEURON_CORES must be numbers, not {base!r}, '
                         f'{count!r}')
    first, last = int(base), int(base) + int(count) - 1
    return {'NEURON_RT_VISIBLE_CORES': f'{first}-{last}' if last > first else str(first)}


def keep_awake() -> subprocess.Popen | None:
    return None  # servers do not sleep


def native_problem(path: Path) -> str | None:
    """None when path is an ELF executable that links libnrt and no Python, framework or inference runtime library."""
    if Path(path).read_bytes()[:4] != ELF:
        return 'not an ELF executable (scripts and wrappers are not allowed)'
    libraries = _run(['ldd', str(path)], timeout=60) or ''
    names = [line.strip().split(' ')[0] for line in libraries.splitlines()]
    for name in names:
        if any(word in name.lower() for word in REFUSED):
            return f'links {name}'
    if not any(name.startswith('libnrt.so') for name in names):
        return 'an executable that does not link libnrt (the model must run on the NeuronCores, not the host CPU)'
    return None
