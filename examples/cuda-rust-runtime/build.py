"""Build with explicit toolchain and architecture; no network or package registry needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help='nvcc architecture, for example sm_103')
    parser.add_argument('--out', type=Path, default=Path('build'))
    args = parser.parse_args()
    if not re.fullmatch(r'sm_[0-9]+[af]?', args.arch):
        parser.error('expected sm_<digits> with an optional architecture suffix')
    source = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    nvcc, rustc = shutil.which('nvcc'), shutil.which('rustc')
    if not nvcc or not rustc:
        parser.error('nvcc and rustc must be available on PATH')
    # Toolkit discovery follows nvcc; set CUDA_LIBRARY_DIR for nonstandard toolkit layouts.
    cuda = Path(nvcc).resolve().parent.parent
    libraries = Path(os.environ.get('CUDA_LIBRARY_DIR', str(cuda / 'lib64')))
    commands = [
        [nvcc, '-std=c++17', '-O3', '-arch=' + args.arch, '--fmad=false', '-Xcompiler=-fPIC',
         '-c', str(source / 'kernel.cu'), '-o', str(out / 'kernel.o')],
        ['ar', 'rcs', str(out / 'libcuda_example.a'), str(out / 'kernel.o')],
        [rustc, '--edition=2021', '-C', 'opt-level=3', str(source / 'main.rs'),
         '-L', str(out), '-L', str(libraries), '-l', 'static=cuda_example', '-l', 'dylib=cudart',
         '-l', 'dylib=stdc++', '-o', str(out / 'cuda-runtime-check')],
    ]
    for argv in commands:
        subprocess.run(argv, check=True)
    metadata = {'arch': args.arch, 'nvcc': subprocess.check_output([nvcc, '--version'], text=True),
                'rustc': subprocess.check_output([rustc, '--version'], text=True),
                'sources': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                            for name in ('kernel.cu', 'main.rs', 'build.py')},
                'binary_sha256': hashlib.sha256((out / 'cuda-runtime-check').read_bytes()).hexdigest(),
                'numerics': {'fast_math': False, 'fmad': False}, 'runtime_measured': False}
    (out / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
