"""Linux with an AMD Instinct or Radeon GPU under ROCm: HIP, device VRAM, the SGLang reference.

The GPU facts come from the amdgpu driver's sysfs: the KFD topology (/sys/class/kfd/kfd/topology/nodes/*/properties:
gfx target, SIMD count, PCI location, DRM render node) and the render node's device directory (VRAM total and used).
A node the process may not use (a Slurm job's cgroup) cannot be read, and is skipped the way the ROCm runtime skips it;
ROCR_VISIBLE_DEVICES, then HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES) select among the rest. Engines are
single-GPU: the first visible GPU is the one they get, and memory_bytes is its VRAM. rocm-smi only adds the
marketing name.

A native engine owns every numerical computation on the model's tensors. It may link the C and C++ runtimes and the
GPU's driver and runtime layer (the HIP runtime, HSA, the hiprtc compiler and what they need), but not precompiled
numerical libraries (rocBLAS, hipBLASLt, MIOpen, Composable Kernel, aiter, RCCL, ...) or an inference runtime: their
kernels were tuned for other shapes and the coding sessions cannot change them for this model. Their open-source
kernels may be copied into the engine's source instead. native_problem refuses every library outside the allow-list.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path

NAME = 'linux-rocm'
DEFAULT_REFERENCE = 'sglang'
ROOT = Path('/')  # the tests point this at a fake tree
BUSY_BYTES = 2 << 30  # VRAM in use above this before the engine starts: another process holds the GPU
# Shared libraries a native engine may link (the soname up to ".so").
ALLOWED = {'libc', 'libm', 'libdl', 'libpthread', 'librt', 'libutil', 'libresolv', 'libatomic', 'libgcc_s',
           'libstdc++', 'ld-linux-x86-64', 'ld-linux-aarch64', 'linux-vdso', 'libz',
           'libamdhip64', 'libhsa-runtime64', 'libhiprtc', 'libhiprtc-builtins', 'libamd_comgr', 'libdrm',
           'libdrm_amdgpu', 'libnuma', 'libelf'}
# Libraries refused by name, for a clearer message (everything else outside ALLOWED is refused too).
REFUSED = ('libpython', 'libtorch', 'libc10', 'librocblas', 'libhipblas', 'libMIOpen', 'librocprim', 'libhipcub',
           'librccl', 'libmlx', 'libggml', 'libllama', 'libonnxruntime', 'libtriton', 'libvllm',
           'libcomposable_kernel', 'libck', 'libaiter', 'librocsolver', 'libhipsolver', 'librocfft', 'libhipfft',
           'librocsparse', 'libhipsparse', 'libmigraphx', 'librocrand', 'libhiprand')


def _run(argv, timeout=20) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _read(path: str) -> str | None:
    try:
        return (ROOT / path).read_text().strip()
    except (OSError, UnicodeDecodeError):  # a node outside the job's cgroup: PermissionError
        return None


def _int(path: str) -> int | None:
    value = _read(path)
    return int(value) if value and value.isdigit() else None


def _gpus() -> list[dict]:
    """The GPUs this process may open, in the ROCm runtime's order, before the *_VISIBLE_DEVICES selection."""
    nodes = ROOT / 'sys/class/kfd/kfd/topology/nodes'
    try:
        names = sorted((p.name for p in nodes.iterdir() if p.name.isdigit()), key=int)
    except OSError:
        return []
    gpus = []
    for name in names:
        text = _read(f'sys/class/kfd/kfd/topology/nodes/{name}/properties')
        props = dict(line.split(None, 1) for line in (text or '').splitlines() if len(line.split()) == 2)
        props = {key: int(value) for key, value in props.items() if value.isdigit()}
        if not props.get('simd_count'):  # unreadable, or a CPU node
            continue
        version, location = props.get('gfx_target_version', 0), props.get('location_id', 0)
        device = f'sys/class/drm/renderD{props.get("drm_render_minor")}/device'
        gpus.append({
            'node': int(name),
            'gfx_arch': f'gfx{version // 10000}{version // 100 % 100:x}{version % 100:x}' if version else None,
            'compute_units': props['simd_count'] // props['simd_per_cu'] if props.get('simd_per_cu') else None,
            'pci_bus': f'{props.get("domain", 0):04x}:{location >> 8:02x}:{location >> 3 & 0x1f:02x}.{location & 7}',
            'uuid': f'GPU-{props["unique_id"]:016x}' if props.get('unique_id') else None,
            'device': device, 'name': _read(f'{device}/product_name'),
            'vram_total': _int(f'{device}/mem_info_vram_total'), 'vram_used': _int(f'{device}/mem_info_vram_used')})
    return gpus


def _select(gpus: list[dict], value: str | None) -> list[dict]:
    """The devices a *_VISIBLE_DEVICES value keeps, in its order: indices or GPU- UUIDs, up to the first invalid."""
    if value is None:
        return gpus
    chosen = []
    for token in (t.strip() for t in value.split(',') if t.strip()):
        match = [g for g in gpus if g['uuid'] and g['uuid'].lower() == token.lower()] if token.startswith('GPU-') \
            else gpus[int(token):int(token) + 1] if token.isdigit() else []
        if not match:
            break
        chosen += match
    return chosen


def visible_gpus() -> list[dict]:
    gpus = _select(_gpus(), os.environ.get('ROCR_VISIBLE_DEVICES'))
    return _select(gpus, os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES')))


def supported() -> bool:
    return platform.system() == 'Linux' and (ROOT / 'dev/kfd').exists() and bool(_gpus())


def _product_names() -> dict[str, str]:
    """Marketing names by PCI bus from `rocm-smi --showproductname --showbus --json` (sysfs has the board name)."""
    text = _run(['rocm-smi', '--showproductname', '--showbus', '--json']) or ''
    try:
        cards = json.loads(text[text.find('{'):])
        return {card['PCI Bus'].lower(): card['Card Series'] for card in cards.values()
                if card.get('PCI Bus') and card.get('Card Series')}
    except (ValueError, AttributeError, TypeError):
        return {}


def _rocm_version() -> str | None:
    if version := _read('opt/rocm/.info/version'):
        return version
    match = re.search(r'HIP version: (\S+)', _run(['hipcc', '--version']) or '')
    return match and match.group(1)


def _meminfo() -> dict[str, int]:
    rows = (line.split() for line in (_read('proc/meminfo') or '').splitlines())
    return {row[0].rstrip(':'): int(row[1]) * 1024 for row in rows if len(row) >= 2 and row[1].isdigit()}


def machine() -> dict:
    gpus = visible_gpus()
    gpu = gpus[0] if gpus else {}
    name = gpu and (_product_names().get(gpu['pci_bus']) or gpu['name'] or 'AMD GPU')
    cpu = re.search(r'^model name\s*:\s*(.+)$', _read('proc/cpuinfo') or '', re.M)
    release = re.search(r'^PRETTY_NAME="?([^"\n]+)"?$', _read('etc/os-release') or '', re.M)
    cargo = shutil.which('cargo') or shutil.which('cargo', path=Path.home() / '.cargo' / 'bin')
    hipcc = shutil.which('hipcc') or shutil.which('hipcc', path=ROOT / 'opt' / 'rocm' / 'bin')
    return {'platform': NAME, 'os': f'{release.group(1) if release else "Linux"} ({platform.release()})',
            'arch': platform.machine(), 'chip': cpu and cpu.group(1).strip(),
            'accelerator': f'{name} ({gpu["gfx_arch"]})' if gpu else None,
            'gfx_arch': gpu.get('gfx_arch'), 'accelerator_cores': gpu.get('compute_units'),
            'memory_bytes': gpu.get('vram_total'), 'vram_used_bytes': gpu.get('vram_used'),
            'pci_bus': gpu.get('pci_bus'), 'visible_gpus': len(gpus),
            'toolchain': {'cargo': cargo, 'hipcc': hipcc, 'rocm': _rocm_version()}}


def checks(facts: dict) -> tuple[list[str], list[str]]:
    problems, notes = [], []
    if not facts['toolchain']['cargo']:
        problems.append('cargo is not on PATH (install Rust from https://rustup.rs); the default engine build uses it.')
    if not facts['toolchain']['hipcc']:
        problems.append('hipcc is not on PATH or in /opt/rocm/bin: install ROCm (HIP SDK) to compile GPU kernels.')
    if not facts['visible_gpus']:
        problems.append('No AMD GPU is visible to this process (check ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES; '
                        'in a Slurm job, request one with --gres=gpu:1).')
    if facts['visible_gpus'] > 1:
        notes.append(f'{facts["visible_gpus"]} GPUs are visible; engines and the reference use only the first '
                     f'({facts["pci_bus"]}). Set HIP_VISIBLE_DEVICES to choose another.')
    if (facts.get('vram_used_bytes') or 0) > BUSY_BYTES:
        notes.append(f'{facts["vram_used_bytes"] / 2**30:.1f} GiB of the GPU\'s VRAM is already in use: another '
                     'process shares the GPU, which slows timing and leaves less memory.')
    return problems, notes


def memory_budget() -> int | None:
    """85% of the first visible GPU's free VRAM."""
    gpus = visible_gpus()
    if not gpus or gpus[0]['vram_total'] is None:
        return None
    return int((gpus[0]['vram_total'] - (gpus[0]['vram_used'] or 0)) * 0.85)


def memory_critical() -> bool:
    """Host MemAvailable below 3% of MemTotal, or the first visible GPU's VRAM over 98% used."""
    memory = _meminfo()
    if memory.get('MemTotal') and memory.get('MemAvailable', memory['MemTotal']) < memory['MemTotal'] * 0.03:
        return True
    gpus = visible_gpus()
    return bool(gpus) and bool(gpus[0]['vram_total']) and (gpus[0]['vram_used'] or 0) > gpus[0]['vram_total'] * 0.98


def keep_awake() -> subprocess.Popen | None:
    return None  # servers do not sleep


def needed_libraries(data: bytes) -> list[str] | None:
    """The DT_NEEDED entries of a 64-bit little-endian ELF file (its direct shared-library dependencies), or None
    when it is not one. A static executable needs none."""
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        return None
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', data, 0x3a)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
    names = []
    for _, kind, _, _, offset, size, link, *_ in sections:
        if kind != 6:  # SHT_DYNAMIC
            continue
        strings = sections[link][4]
        for tag, value in struct.iter_unpack('<qQ', data[offset:offset + size]):
            if tag == 1:  # DT_NEEDED
                start = strings + value
                names.append(data[start:data.index(b'\0', start)].decode())
    return names


def native_problem(path: Path) -> str | None:
    """None when path is an ELF executable whose direct dependencies are system or HIP runtime libraries only."""
    data = Path(path).read_bytes()
    if data[:4] != b'\x7fELF':
        return 'not an ELF executable (scripts and wrappers are not allowed)'
    try:
        libraries = needed_libraries(data)
    except (struct.error, IndexError, ValueError, UnicodeDecodeError):
        return 'not a readable ELF executable'
    if libraries is None:
        return 'not a 64-bit little-endian ELF executable'
    for library in libraries:
        stem = os.path.basename(library).split('.so')[0]
        if stem.startswith(REFUSED):
            return (f'links {library}, a precompiled numerical or inference library: the engine must own every '
                    'computation on the model (copy the open-source kernels you need into its source instead)')
        if stem not in ALLOWED:
            return f'links {library}, which is not a system or HIP runtime library'
    return None
