"""Reference implementations: the trusted engines a candidate is checked and timed against.

A reference is a module in this package that runs as `python -m enginesmith.references.<module> --model MODEL_DIR`,
answers the engine protocol (docs/protocol.md: hello, generate) and two requests only references answer: judge
(teacher-forced checking with the rank of each mismatching token) and stats (memory). It runs a stock inference
library unmodified and reproduces that library's own generation path exactly. KINDS maps the name a task uses
(`[reference] kind`) to the module and the Python package whose version identifies it. CONTRIBUTING.md says how to add
one.

A kind may name a container image through an environment variable (`image_env`): when the variable is set, the
reference runs inside that image with Apptainer (`apptainer exec --rocm`), which is how SGLang is usually deployed on
AMD GPUs, and its version is read inside the image.
"""
from __future__ import annotations

import functools
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

KINDS = {
    'mlx-lm': {'module': 'enginesmith.references.mlx_lm', 'package': 'mlx-lm', 'install': 'run `uv sync --extra mlx`.'},
    'sglang': {'module': 'enginesmith.references.sglang', 'package': 'sglang', 'image_env': 'ENGINESMITH_SGLANG_IMAGE',
               'install': 'set ENGINESMITH_SGLANG_IMAGE to an SGLang ROCm image (Apptainer .sif), or install sglang.'},
}


def _image(kind: str) -> str | None:
    variable = KINDS[kind].get('image_env')
    return os.environ.get(variable) or None if variable else None


def _in_image(image: str, argv: list[str]) -> list[str]:
    """argv run with the image's Python, this package importable and the home directory mounted. HOME is the user's,
    not the scratch directory the harness gives engines, so that the library keeps its compiled kernels between runs;
    TMPDIR is /tmp, because SGLang puts its IPC sockets there and a run's scratch path exceeds the 107-byte limit of a
    Unix socket path."""
    source, home = str(Path(__file__).resolve().parents[2]), str(Path.home())
    return ['apptainer', 'exec', '--rocm', '--bind', f'{home},{source}', '--env', f'PYTHONPATH={source}',
            '--env', f'HOME={home}', '--env', 'TMPDIR=/tmp', image, 'python3', *argv]


def command(kind: str, model: Path) -> list[str]:
    argv = ['-m', KINDS[kind]['module'], '--model', str(Path(model).resolve())]
    image = _image(kind)
    return _in_image(image, argv) if image else [sys.executable, *argv]


@functools.cache
def _version_in_image(image: str, package: str) -> str | None:
    probe = f'import importlib.metadata as m; print(m.version({package!r}))'
    try:
        result = subprocess.run(_in_image(image, ['-c', probe]), capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def version(kind: str) -> str | None:
    image = _image(kind)
    if image:
        return _version_in_image(image, KINDS[kind]['package'])
    try:
        return importlib.metadata.version(KINDS[kind]['package'])
    except importlib.metadata.PackageNotFoundError:
        return None
