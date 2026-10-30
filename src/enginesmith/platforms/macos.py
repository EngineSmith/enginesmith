"""macOS on Apple Silicon: Metal GPU, unified memory, the mlx-lm reference."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

NAME = 'macos-metal'
DEFAULT_REFERENCE = 'mlx-lm'
MACH_O = (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xfe\xed\xfa\xcf')


def _run(argv, timeout=20) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def supported() -> bool:
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def _memory_bytes() -> int | None:
    value = _run(['/usr/sbin/sysctl', '-n', 'hw.memsize'])
    return int(value) if value and value.isdigit() else None


def machine() -> dict:
    cores = None
    profile = _run(['/usr/sbin/system_profiler', 'SPDisplaysDataType', '-json'], timeout=60)
    if profile:
        try:
            cores = int(json.loads(profile)['SPDisplaysDataType'][0].get('sppci_cores'))
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    sleep = None
    for line in (_run(['/usr/bin/pmset', '-g']) or '').splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == 'sleep':
            sleep = parts[1]
    chip = _run(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'])
    return {'platform': NAME, 'os': f'macOS {platform.mac_ver()[0]}', 'arch': platform.machine(), 'chip': chip,
            'accelerator': f'{chip} GPU (Metal)' if chip else 'Metal GPU', 'accelerator_cores': cores,
            'memory_bytes': _memory_bytes(), 'system_sleep_minutes': sleep,
            'toolchain': {'cargo': shutil.which('cargo'), 'metal_compiler': _run(['/usr/bin/xcrun', '-f', 'metal'])}}


def checks(facts: dict) -> tuple[list[str], list[str]]:
    problems, notes = [], []
    if not facts['toolchain']['cargo']:
        problems.append('cargo is not on PATH (install Rust from https://rustup.rs); the default engine build uses it.')
    if facts.get('system_sleep_minutes') not in (None, '0'):
        notes.append('System sleep is on; a run keeps the machine awake with caffeinate while it is active, but a '
                     'closed lid or a power loss still stops it (`enginesmith resume` continues).')
    if not facts['toolchain']['metal_compiler']:
        notes.append('The offline Metal compiler (xcrun metal) is not installed; engines must compile their Metal '
                     'source at run time (the usual way for Rust engines).')
    return problems, notes


def memory_budget() -> int | None:
    """MLX's recommended working set (about three quarters of unified memory), else 75% of memory."""
    try:
        import mlx.core as mx
        return mx.device_info()['max_recommended_working_set_size']
    except Exception:
        total = _memory_bytes()
        return int(total * 0.75) if total else None


def memory_critical() -> bool:
    """macOS memory pressure level 4 (critical); 1 is normal and 2 warning."""
    level = _run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], timeout=5)
    return bool(level) and level.isdigit() and int(level) >= 4


def keep_awake() -> subprocess.Popen | None:
    """Prevent idle sleep for as long as this process lives (caffeinate exits with it)."""
    try:
        return subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def native_problem(path: Path) -> str | None:
    """None when path is a Mach-O executable that links no Python or MLX library."""
    if Path(path).read_bytes()[:4] not in MACH_O:
        return 'not a Mach-O executable (scripts and wrappers are not allowed)'
    libraries = _run(['otool', '-L', str(path)], timeout=60) or ''
    for line in libraries.splitlines()[1:]:
        name = line.strip().split(' ')[0]
        if 'python' in name.lower() or 'mlx' in name.lower():
            return f'links {name}'
    return None
