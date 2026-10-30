"""Reference implementations: the trusted engines a candidate is checked and timed against.

A reference is a module in this package that runs as `python -m enginesmith.references.<module> --model MODEL_DIR`,
answers the engine protocol (docs/protocol.md: hello, generate) and two requests only references answer: judge
(teacher-forced checking with the rank of each mismatching token) and stats (memory). It runs a stock inference
library unmodified and reproduces that library's own generation path exactly. KINDS maps the name a task uses
(`[reference] kind`) to the module and the Python package whose version identifies it. CONTRIBUTING.md says how to add
one.
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

KINDS = {
    'mlx-lm': {'module': 'enginesmith.references.mlx_lm', 'package': 'mlx-lm'},
}


def command(kind: str, model: Path) -> list[str]:
    return [sys.executable, '-m', KINDS[kind]['module'], '--model', str(model)]


def version(kind: str) -> str | None:
    try:
        return importlib.metadata.version(KINDS[kind]['package'])
    except importlib.metadata.PackageNotFoundError:
        return None
