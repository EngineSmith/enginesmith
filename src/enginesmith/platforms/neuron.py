"""Linux on AWS Trainium (Neuron devices): HBM per NeuronCore, NKI kernels compiled to NEFF, run through libnrt."""
from __future__ import annotations

import json
import os
import platform
import shutil
import struct
import subprocess
from pathlib import Path

NAME = 'linux-neuron'
DEFAULT_REFERENCE = 'nxdi'
NEURON_BIN = Path('/opt/aws/neuron/bin')
ELF = b'\x7fELF'
# The libraries an engine may link (docs/libraries.md, "Trainium"): the C and C++ runtimes, the loader, and libnrt,
# the Neuron runtime that loads the engine's own NEFF files onto the NeuronCores (its collectives library is loaded by
# libnrt itself, not linked). Everything else is refused.
ALLOWED = {'libc', 'libm', 'libdl', 'libpthread', 'librt', 'libutil', 'libgcc_s', 'libstdc++',
           'ld-linux-x86-64', 'ld-linux-aarch64', 'linux-vdso', 'libnrt'}
# Refused by name, for a clearer message (everything else outside ALLOWED is refused too).
REFUSED = ('libpython', 'libtorch', 'libc10', 'libneuronxla', 'libxla', 'libjax', 'libvllm', 'libtensorflow',
           'libonnxruntime', 'libggml', 'libllama')

# Trainium2 per chip, from the public Neuron documentation (https://awsdocs-neuron.readthedocs-hosted.com/en/latest/
# about-neuron/arch/neuron-hardware/trainium2.html): 2.9 TB/s of HBM bandwidth and 667 dense BF16 TFLOPS, over 8
# physical NeuronCores. A logical NeuronCore at the default LNC=2 has a quarter: 725 GB/s and about 167 TFLOPS.
TRN2_CHIP = {'bandwidth': 2.9e12, 'compute': 667e12, 'physical_cores': 8}


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


def engine_environment(engine_cores: int | None = None, reference_on_device: bool = True) -> dict:
    """The NeuronCores a process may use (the Neuron runtime otherwise claims every core it can see). The run has
    ENGINESMITH_NEURON_CORES logical cores (default 2) from ENGINESMITH_NEURON_CORE (default 0): with a reference of
    tensor-parallel degree N, 2N of them. The reference sees all of them (it generates on the first N and judges on
    the next N); an engine (engine_cores = N) sees only the first N, the cores the reference generates on. A
    correctness-only reference runs on the host (reference_on_device False), so the run then needs only N. Runs
    started with different values can share the machine."""
    base, count = (os.environ.get(name, default) for name, default in
                   (('ENGINESMITH_NEURON_CORE', '0'), ('ENGINESMITH_NEURON_CORES', '2')))
    if not base.isdigit() or not count.isdigit() or int(count) < 1:
        raise ValueError(f'ENGINESMITH_NEURON_CORE and ENGINESMITH_NEURON_CORES must be numbers, not {base!r}, '
                         f'{count!r}')
    if engine_cores and (2 if reference_on_device else 1) * engine_cores > int(count):
        # The reference judges on the N cores after the engine's; with fewer it would take another run's.
        needed = (2 if reference_on_device else 1) * engine_cores
        raise ValueError(f'the run needs {needed} NeuronCores (the reference\'s tensor-parallel degree {engine_cores}'
                         f'{", twice: the reference judges on the cores after them" if reference_on_device else ""}) '
                         f'but ENGINESMITH_NEURON_CORES is {count}')
    first = int(base)
    last = first + (engine_cores or int(count)) - 1
    return {'NEURON_RT_VISIBLE_CORES': f'{first}-{last}' if last > first else str(first)}


BUSY = (b'Failed to initialize NRT runtime', b'cores busy', b'Logical Neuron Core(s) not available')


def peak(cores: int = 1) -> dict | None:
    """Trainium2's memory bandwidth (bytes/s) and dense BF16 compute (FLOP/s) for cores logical NeuronCores."""
    first = next(iter(_neuron_ls()), {})
    if not str(first.get('instance_type') or '').startswith('trn2'):
        return None
    lnc = first.get('logical_neuroncore_config') or 2
    share = cores * lnc / TRN2_CHIP['physical_cores']
    return {'bandwidth': TRN2_CHIP['bandwidth'] * share, 'compute': TRN2_CHIP['compute'] * share,
            'source': f'Trainium2 vendor figures (Neuron documentation): 2.9 TB/s and 667 dense BF16 TFLOPS per chip '
                      f'of 8 physical NeuronCores, here {cores} logical core(s) at LNC={lnc}'}


def busy_accelerator(log_tail: bytes) -> bool:
    """A build log that shows the NeuronCores held by another process: NKI builds run each kernel once to compile it,
    so a build can meet a core another run briefly holds. The build is retried once."""
    return any(marker in log_tail for marker in BUSY)


def keep_awake() -> subprocess.Popen | None:
    return None  # servers do not sleep


def native_problem(path: Path) -> str | None:
    """None when path is an ELF executable whose direct dependencies (DT_NEEDED) are all in ALLOWED and include
    libnrt, so the model runs on the NeuronCores."""
    from enginesmith.platforms.linux_rocm import needed_libraries
    data = Path(path).read_bytes()
    if data[:4] != ELF:
        return 'not an ELF executable (scripts and wrappers are not allowed)'
    try:
        libraries = needed_libraries(data)
    except (struct.error, IndexError, ValueError, UnicodeDecodeError):
        return 'not a readable ELF executable'
    if libraries is None:
        return 'not a 64-bit little-endian ELF executable'
    stems = [os.path.basename(library).split('.so')[0] for library in libraries]
    for library, stem in zip(libraries, stems):
        if stem.startswith(REFUSED):
            return (f'links {library}, a framework, compiler or inference runtime: the engine must own every '
                    'computation on the model (docs/libraries.md)')
        if stem not in ALLOWED:
            return f'links {library}, which is not a system library or the Neuron runtime (docs/libraries.md)'
    if 'libnrt' not in stems:
        return 'an executable that does not link libnrt (the model must run on the NeuronCores, not the host CPU)'
    return None
