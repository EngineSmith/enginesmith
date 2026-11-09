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
# Libraries a native engine may link (docs/libraries.md): the system and C/C++ runtimes, and Metal with the
# Objective-C layer it is called through. A framework is named by its framework, any other library by its file name
# up to the first dot; both must be the system's own copy.
ALLOWED = {'libSystem', 'libc++', 'libobjc', 'libiconv', 'Foundation', 'CoreFoundation', 'Metal'}
SYSTEM = ('/usr/lib/', '/System/Library/')
# Libraries refused by name, for a clearer message (everything else outside ALLOWED is refused too).
REFUSED = ('libpython', 'Python', 'libmlx', 'mlx', 'MetalPerformanceShaders', 'Accelerate', 'CoreML', 'libtorch',
           'libggml', 'libllama', 'libonnxruntime')

# Apple M5 Max: the streaming read ceiling measured in the skill backend (backends/metal-rust/README.md, "Measured
# platform numbers": about 586-590 GB/s). Apple publishes no peak matrix throughput for its GPU, so there is no
# prefill bound.
M5_MAX = {'bandwidth': 590e9, 'compute': None,
          'source': 'Apple M5 Max: 590 GB/s measured streaming read (skill backend metal-rust); no published peak '
                    'compute'}


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


def peak(cores: int = 1) -> dict | None:
    """The GPU's memory bandwidth (bytes/s) and peak compute (FLOP/s, None when unknown), on the chips with
    figures."""
    return M5_MAX if _run(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']) == 'Apple M5 Max' else None


def keep_awake() -> subprocess.Popen | None:
    """Prevent idle sleep for as long as this process lives (caffeinate exits with it)."""
    try:
        return subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _library_name(library: str) -> str:
    """The name ALLOWED and REFUSED use: a framework's name, or a library's file name up to the first dot."""
    if '.framework/' in library:
        return library.split('.framework/')[0].rsplit('/', 1)[-1]
    return library.rsplit('/', 1)[-1].split('.')[0]


def native_problem(path: Path) -> str | None:
    """None when path is a Mach-O executable that links only system libraries in ALLOWED."""
    if Path(path).read_bytes()[:4] not in MACH_O:
        return 'not a Mach-O executable (scripts and wrappers are not allowed)'
    listing = _run(['otool', '-L', str(path)], timeout=60)
    if listing is None:
        return 'otool cannot list the libraries it links'
    for line in listing.splitlines():
        if line.endswith(':'):  # the file itself, once per architecture of a universal binary
            continue
        library = line.strip().split(' (compatibility')[0]
        name = _library_name(library)
        if name.startswith(REFUSED):
            return (f'links {library}, a framework, inference runtime or precompiled numerical library: the engine '
                    'must own every computation on the model (copy the open-source kernels you need into its source '
                    'instead)')
        if name not in ALLOWED or not library.startswith(SYSTEM):
            return f'links {library}, which is not an allowed system library (docs/libraries.md)'
    return None
