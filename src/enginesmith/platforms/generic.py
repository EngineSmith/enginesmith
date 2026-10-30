"""The fallback for machines no other platform module supports: the harness and its tests work, but there is no
default reference, no memory watchdog, no sleep inhibition and no native-executable check. It is also the smallest
example of the interface (enginesmith/platforms/__init__.py); a new platform starts from a copy of it."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

NAME = 'generic'
DEFAULT_REFERENCE = None  # a task must name its reference ([reference] kind or command)


def supported() -> bool:
    return True


def machine() -> dict:
    memory = None
    if hasattr(os, 'sysconf') and 'SC_PHYS_PAGES' in os.sysconf_names:
        try:
            memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (OSError, ValueError):
            pass
    return {'platform': NAME, 'os': f'{platform.system()} {platform.release()}', 'arch': platform.machine(),
            'chip': platform.processor() or None, 'accelerator': None, 'accelerator_cores': None,
            'memory_bytes': memory, 'toolchain': {'cargo': shutil.which('cargo')}}


def checks(facts: dict) -> tuple[list[str], list[str]]:
    return ([f'EngineSmith has no platform module for {facts["os"]} on {facts["arch"]} yet: runs need one '
             '(see CONTRIBUTING.md, "Supporting another platform"). The CPU tests work.'], [])


def memory_budget() -> int | None:
    facts = machine()
    return int(facts['memory_bytes'] * 0.75) if facts['memory_bytes'] else None


def memory_critical() -> bool:
    return False


def keep_awake() -> subprocess.Popen | None:
    return None


def native_problem(path: Path) -> str | None:
    return None
