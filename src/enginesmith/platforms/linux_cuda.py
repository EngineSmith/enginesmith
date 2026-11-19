"""CUDA prerequisites, without a built-in or GPU-validated reference.

Both arms see the same selected full GPU UUIDs in the same order. A declared TP
degree is a task/reference responsibility; visibility alone is not proof of TP.
Memory estimates assume balanced sharding. Native checks inspect direct ELF
dependencies, not statically embedded code or libraries opened with dlopen.
"""
from __future__ import annotations

import csv
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path

NAME = 'linux-cuda'
DEFAULT_REFERENCE = None
ROOT = Path('/')
ALLOWED = {'libc', 'libm', 'libdl', 'libpthread', 'librt', 'libutil', 'libresolv', 'libatomic', 'libgcc_s',
           'libstdc++', 'ld-linux-x86-64', 'ld-linux-aarch64', 'linux-vdso', 'libz', 'libnuma', 'libelf',
           'libcuda', 'libcudart', 'libnvrtc', 'libnvrtc-builtins', 'libnvJitLink', 'libnvfatbin',
           'libnvidia-ptxjitcompiler', 'libnvidia-fatbinaryloader', 'libnvidia-ml', 'libnccl'}
GPU_FIELDS = 'index,uuid,pci.bus_id,name,memory.total,memory.used,driver_version,compute_cap,mig.mode.current'


TELEMETRY_TIMEOUT_SECONDS = 20.0


def _run(argv, timeout=TELEMETRY_TIMEOUT_SECONDS) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def parse_gpus(text: str) -> list[dict]:
    """Parse a fixed nvidia-smi CSV query. Missing or malformed telemetry fails closed."""
    gpus = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != 9:
            raise ValueError('CUDA telemetry needs all nine requested fields')
        index, uuid, bus, name, total, used, driver, capability, mig = (x.strip() for x in row)
        if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid):
            raise ValueError('CUDA telemetry has no complete GPU UUID')
        if not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', bus):
            raise ValueError('CUDA telemetry has an invalid PCI bus identifier')
        if not index.isdigit() or not total.isdigit() or not used.isdigit() or int(total) <= 0:
            raise ValueError('CUDA telemetry has invalid device or memory values')
        if int(used) > int(total) or not re.fullmatch(r'\d+\.\d+', capability):
            raise ValueError('CUDA telemetry has invalid memory use or compute capability')
        if not driver or driver in ('[N/A]', 'N/A'):
            raise ValueError('CUDA telemetry has no NVIDIA driver version')
        if mig.lower() not in ('disabled', 'enabled', '[n/a]', 'n/a', 'not supported'):
            raise ValueError('CUDA telemetry has an unknown MIG state')
        domain, bus_number, device = bus.lower().split(':')
        gpus.append({'index': int(index), 'uuid': uuid, 'pci_bus': f'{int(domain, 16):08x}:{bus_number}:{device}',
                     'name': name, 'vram_total': int(total) << 20, 'vram_used': int(used) << 20,
                     'driver': driver, 'compute_capability': capability, 'mig_enabled': mig.lower() == 'enabled'})
    if len({g['uuid'] for g in gpus}) != len(gpus) or len({g['pci_bus'] for g in gpus}) != len(gpus):
        raise ValueError('CUDA telemetry lists duplicate devices')
    return sorted(gpus, key=lambda g: g['pci_bus'])


def _gpus() -> list[dict]:
    text = _run(['nvidia-smi', f'--query-gpu={GPU_FIELDS}', '--format=csv,noheader,nounits'])
    if text is None:
        raise ValueError('nvidia-smi cannot report the required GPU telemetry')
    return parse_gpus(text) if text else []


def _select(gpus: list[dict], value: str | None, *, uuid_only=False) -> list[dict]:
    if value is None:
        return gpus
    if not value.strip() or value.strip() == '-1':
        return []
    chosen = []
    for token in value.split(','):
        token = token.strip()
        if token.startswith('MIG-'):
            raise ValueError('MIG devices are unsupported')
        if token.isdigit() and not uuid_only:
            if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
                raise ValueError('numeric CUDA_VISIBLE_DEVICES requires CUDA_DEVICE_ORDER=PCI_BUS_ID; use full UUIDs')
            match = gpus[int(token):int(token) + 1]
        else:
            match = [g for g in gpus if g['uuid'] == token]
        if len(match) != 1 or match[0] in chosen:
            raise ValueError('GPU selection contains an unknown, abbreviated, repeated, or unavailable UUID/index')
        chosen.extend(match)
    return chosen


def visible_gpus() -> list[dict]:
    gpus = _select(_gpus(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    gpus = _select(gpus, os.environ.get('ENGINESMITH_CUDA_UUIDS'), uuid_only=True)
    if any(g['mig_enabled'] for g in gpus):
        raise ValueError('MIG is enabled on a selected GPU; MIG is unsupported')
    return gpus


def supported() -> bool:
    return platform.system() == 'Linux' and bool(shutil.which('nvidia-smi') or (ROOT / 'dev/nvidiactl').exists())


def _meminfo() -> dict[str, int]:
    try:
        rows = (ROOT / 'proc/meminfo').read_text().splitlines()
    except OSError:
        return {}
    return {parts[0].rstrip(':'): int(parts[1]) * 1024 for row in rows
            if len(parts := row.split()) >= 2 and parts[1].isdigit()}


def machine() -> dict:
    error = None
    try:
        gpus = visible_gpus()
    except ValueError as exc:
        gpus, error = [], str(exc)
    nvcc = shutil.which('nvcc')
    version = _run([nvcc, '--version']) if nvcc else None
    match = re.search(r'release (\S+),', version or '')
    return {'platform': NAME, 'os': f'Linux {platform.release()}', 'arch': platform.machine(),
            'chip': platform.processor() or None, 'accelerator': ', '.join(dict.fromkeys(g['name'] for g in gpus)) or None,
            'accelerator_cores': None, 'visible_gpus': len(gpus), 'gpu_uuids': [g['uuid'] for g in gpus], 'gpus': gpus,
            'memory_bytes': sum(g['vram_total'] for g in gpus) if gpus else None,
            'balanced_tp_memory_bytes': min(g['vram_total'] for g in gpus) * len(gpus) if gpus else None,
            'vram_used_bytes': sum(g['vram_used'] for g in gpus) if gpus else None, 'selection_error': error,
            'toolchain': {'nvcc': nvcc, 'cxx': shutil.which('c++') or shutil.which('g++'),
                          'cuda': match.group(1) if match else None}}


def checks(facts: dict) -> tuple[list[str], list[str]]:
    problems = []
    if facts.get('selection_error'):
        problems.append(facts['selection_error'])
    if not facts.get('visible_gpus'):
        problems.append('No supported NVIDIA GPU is selected.')
    if not facts['toolchain']['nvcc']:
        problems.append('nvcc is not on PATH: a compatible CUDA Toolkit is required.')
    if not facts['toolchain']['cxx']:
        problems.append('A native C++ compiler is not on PATH.')
    notes = ['CUDA prerequisite support has no built-in reference: use an externally validated reference.command.',
             'Both arms receive the same selected GPU UUIDs; the reference command must implement the declared TP degree.',
             'Balanced-TP VRAM estimates do not prove each shard, workspace, and KV cache fits.',
             'CPU tests provide no model correctness, format support, or GPU performance evidence.']
    return problems, notes


def memory_budget() -> int | None:
    """85% of the smallest selected GPU's free VRAM times the GPU count: balanced TP only."""
    try:
        gpus = visible_gpus()
    except ValueError:
        return None
    return int(min(g['vram_total'] - g['vram_used'] for g in gpus) * len(gpus) * .85) if gpus else None


def memory_critical() -> bool:
    """Host available memory below 3%, any selected GPU above 98%, or lost GPU telemetry."""
    host = _meminfo()
    if host.get('MemTotal') and host.get('MemAvailable', host['MemTotal']) < host['MemTotal'] * .03:
        return True
    try:
        gpus = visible_gpus()
    except ValueError:
        return True
    return not gpus or any(g['vram_used'] > g['vram_total'] * .98 for g in gpus)


def engine_environment(engine_cores: int | None = None) -> dict:
    gpus = visible_gpus()
    if not gpus or engine_cores is not None and engine_cores > len(gpus):
        raise ValueError('CUDA selection has fewer GPUs than the declared tensor-parallel degree')
    return {'CUDA_VISIBLE_DEVICES': ','.join(g['uuid'] for g in gpus), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}


def keep_awake() -> subprocess.Popen | None:
    return None


def needed_libraries(data: bytes) -> list[str]:
    """Read ELF program headers, including sectionless executables. Never execute the candidate."""
    if len(data) < 64 or data[:7] != b'\x7fELF\x02\x01\x01':
        raise ValueError('not a 64-bit little-endian ELF executable (scripts and wrappers are refused)')
    (_, kind, machine_id, version, entry, phoff, _, _, ehsize, phsize, phnum, _, _, _) = struct.unpack_from(
        '<16sHHIQQQIHHHHHH', data)
    expected_machine = {'x86_64': 62, 'aarch64': 183}.get(platform.machine())
    if kind not in (2, 3) or not expected_machine or machine_id != expected_machine or version != 1 or not entry:
        raise ValueError('ELF is not an executable for this host architecture')
    if ehsize != 64 or phsize != 56 or not phnum or phnum == 65535 or phoff < 64 or phoff + phsize * phnum > len(data):
        raise ValueError('ELF has an invalid program-header table')
    headers = [struct.unpack_from('<IIQQQQQQ', data, phoff + i * phsize) for i in range(phnum)]
    for _, _, offset, _, _, filesz, memsz, _ in headers:
        if offset + filesz > len(data) or filesz > memsz:
            raise ValueError('ELF has a truncated or invalid segment')
    loads = [h for h in headers if h[0] == 1]
    if not any(h[1] & 1 and h[3] <= entry < h[3] + h[6] for h in loads):
        raise ValueError('ELF entry point is outside executable load segments')
    interpreters = [h for h in headers if h[0] == 3]
    if len(interpreters) > 1:
        raise ValueError('ELF has multiple program interpreters')
    for h in interpreters:
        interpreter = data[h[2]:h[2] + h[5]]
        loader, triplet = {62: ('ld-linux-x86-64.so.2', 'x86_64-linux-gnu'),
                           183: ('ld-linux-aarch64.so.1', 'aarch64-linux-gnu')}[expected_machine]
        directories = ('/lib', '/lib64', '/usr/lib', '/usr/lib64',
                       f'/lib/{triplet}', f'/usr/lib/{triplet}')
        allowed = {(directory + '/' + loader).encode() + b'\0' for directory in directories}
        if interpreter not in allowed:
            raise ValueError('ELF program interpreter is not the system dynamic loader')
    dynamic = [h for h in headers if h[0] == 2]
    if len(dynamic) > 1:
        raise ValueError('ELF has multiple dynamic segments')
    if not dynamic:
        return []
    _, _, offset, _, _, size, _, _ = dynamic[0]
    if not size or size % 16:
        raise ValueError('ELF has an invalid dynamic segment')
    tags, terminated = {}, False
    for tag, value in struct.iter_unpack('<qQ', data[offset:offset + size]):
        if tag == 0:
            terminated = True
            break
        tags.setdefault(tag, []).append(value)
    if not terminated:
        raise ValueError('ELF dynamic segment is unterminated')
    if 0x7fffffff in tags or 0x7ffffffd in tags:
        raise ValueError('ELF dependency filters are unsupported')
    if not tags.get(1):
        return []
    if len(tags.get(5, [])) != 1 or len(tags.get(10, [])) != 1:
        raise ValueError('ELF dynamic dependencies have no unique string table')
    address, size = tags[5][0], tags[10][0]
    containing = [h for h in loads if h[3] <= address and address + size <= h[3] + h[5]]
    if len(containing) != 1 or not size:
        raise ValueError('ELF string table is outside file-backed load segments')
    strings = containing[0][2] + address - containing[0][3]
    names = []
    for value in tags[1]:
        if value >= size:
            raise ValueError('ELF dependency string offset is out of bounds')
        start = strings + value
        end = data.find(b'\0', start, strings + size)
        if end < 0 or end == start:
            raise ValueError('ELF dependency name is empty or unterminated')
        names.append(data[start:end].decode('ascii'))
    return names


def native_problem(path: Path) -> str | None:
    try:
        libraries = needed_libraries(Path(path).read_bytes())
    except (OSError, ValueError, struct.error, UnicodeDecodeError) as exc:
        return str(exc)
    for library in libraries:
        match = re.fullmatch(r'([^/]+)\.so(?:\.[0-9]+)*', library)
        if not match or match[1] not in ALLOWED:
            return f'links {library}, outside the system/CUDA runtime library allowlist (docs/libraries.md)'
    return None
