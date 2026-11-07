"""The linux-rocm platform's parsing, on a fake sysfs tree and recorded rocm-smi output (MI325X, one Slurm GPU)."""
import json
import struct
from pathlib import Path

import pytest

from enginesmith.platforms import linux_rocm

GiB = 2**30
CPU_NODE = 'cpu_cores_count 64\nsimd_count 0\nsimd_per_cu 0\ngfx_target_version 0\nlocation_id 0\ndrm_render_minor 0\n'
# Recorded on an MI325X node (the second GPU is the first with another location, render node and UUID).
GPU_NODE = ('cpu_cores_count 0\nsimd_count 1216\narray_count 32\ncu_per_simd_array 10\nsimd_per_cu 4\n'
            'gfx_target_version 90402\nvendor_id 4098\ndevice_id 29861\nlocation_id {location}\ndomain 0\n'
            'drm_render_minor {minor}\nunique_id {uid}\n')
ROCM_SMI = {'card0': {'PCI Bus': '0000:85:00.0', 'VRAM Total Memory (B)': '274861129728',
                      'VRAM Total Used Memory (B)': '297771008', 'Card Series': 'AMD Instinct MI325X',
                      'Card Model': '0x74a5', 'Card SKU': 'M3250101', 'Node ID': '7', 'GFX Version': 'gfx942'}}


def gpu(root, node, minor, location, uid, used):
    path = root / f'sys/class/kfd/kfd/topology/nodes/{node}'
    path.mkdir(parents=True)
    (path / 'properties').write_text(GPU_NODE.format(location=location, minor=minor, uid=uid))
    device = root / f'sys/class/drm/renderD{minor}/device'
    device.mkdir(parents=True)
    (device / 'product_name').write_text('AMD Instinct MI325DLC OAM\n')
    (device / 'mem_info_vram_total').write_text('274861129728\n')
    (device / 'mem_info_vram_used').write_text(f'{used}\n')


@pytest.fixture
def node(tmp_path, monkeypatch):
    """Two CPU nodes, a node outside the job's cgroup (unreadable: no properties here) and two MI325X."""
    for name, text in (('0', CPU_NODE), ('1', CPU_NODE), ('2', None)):
        path = tmp_path / f'sys/class/kfd/kfd/topology/nodes/{name}'
        path.mkdir(parents=True)
        if text:
            (path / 'properties').write_text(text)
    gpu(tmp_path, 7, 168, 34048, 1229801703532086340, 297771008)
    gpu(tmp_path, 8, 176, 0x9500, 1, 5 * GiB)
    (tmp_path / 'dev').mkdir()
    (tmp_path / 'dev/kfd').touch()
    (tmp_path / 'proc').mkdir()
    (tmp_path / 'proc/cpuinfo').write_text('processor\t: 0\nmodel name\t: AMD EPYC 9575F 64-Core Processor\n')
    (tmp_path / 'proc/meminfo').write_text('MemTotal:       100000 kB\nMemAvailable:    50000 kB\n')
    monkeypatch.setattr(linux_rocm, 'ROOT', tmp_path)
    monkeypatch.setattr(linux_rocm.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(linux_rocm, '_run', lambda argv, timeout=20: json.dumps(ROCM_SMI)
                        if argv[0] == 'rocm-smi' else None)
    for variable in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def test_the_first_visible_gpu_is_described(node):
    assert linux_rocm.supported()
    facts = linux_rocm.machine()
    assert facts['accelerator'] == 'AMD Instinct MI325X (gfx942)' and facts['gfx_arch'] == 'gfx942'
    assert facts['accelerator_cores'] == 304 and facts['memory_bytes'] == 274861129728
    assert facts['pci_bus'] == '0000:85:00.0' and facts['visible_gpus'] == 2
    assert facts['chip'] == 'AMD EPYC 9575F 64-Core Processor'
    problems, notes = linux_rocm.checks({**facts, 'toolchain': {'cargo': 'cargo', 'hipcc': 'hipcc', 'rocm': '7'}})
    assert not problems and len(notes) == 1 and 'only the first' in notes[0]


def test_without_rocm_smi_the_name_comes_from_sysfs(node, monkeypatch):
    monkeypatch.setattr(linux_rocm, '_run', lambda argv, timeout=20: None)
    assert linux_rocm.machine()['accelerator'] == 'AMD Instinct MI325DLC OAM (gfx942)'
    assert linux_rocm._gpus()[0]['uuid'] == 'GPU-1111222233334444'


def test_the_visible_devices_variables_select_the_gpu(node, monkeypatch):
    monkeypatch.setenv('ROCR_VISIBLE_DEVICES', '1')
    facts = linux_rocm.machine()
    assert facts['pci_bus'] == '0000:95:00.0' and facts['visible_gpus'] == 1
    assert facts['accelerator'] == 'AMD Instinct MI325DLC OAM (gfx942)'  # rocm-smi does not list this bus
    _, notes = linux_rocm.checks({**facts, 'toolchain': {'cargo': 'cargo', 'hipcc': 'hipcc', 'rocm': None}})
    assert len(notes) == 1 and '5.0 GiB' in notes[0]  # busy
    monkeypatch.setenv('HIP_VISIBLE_DEVICES', '0')  # an index among the devices ROCR left
    assert linux_rocm.machine()['pci_bus'] == '0000:95:00.0'
    monkeypatch.delenv('ROCR_VISIBLE_DEVICES')
    monkeypatch.setenv('HIP_VISIBLE_DEVICES', 'GPU-1111222233334444')
    assert linux_rocm.machine()['pci_bus'] == '0000:85:00.0'
    monkeypatch.delenv('HIP_VISIBLE_DEVICES')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    assert linux_rocm.machine()['pci_bus'] == '0000:95:00.0' and linux_rocm.machine()['visible_gpus'] == 2
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    facts = linux_rocm.machine()
    assert facts['accelerator'] is None and facts['memory_bytes'] is None and linux_rocm.memory_budget() is None
    problems, _ = linux_rocm.checks({**facts, 'toolchain': {'cargo': None, 'hipcc': None, 'rocm': None}})
    assert len(problems) == 3 and 'No AMD GPU' in problems[2]


def test_memory_budget_and_critical(node, monkeypatch):
    assert linux_rocm.memory_budget() == int((274861129728 - 297771008) * 0.85)
    assert not linux_rocm.memory_critical()
    (node / 'proc/meminfo').write_text('MemTotal:       100000 kB\nMemAvailable:     2000 kB\n')
    assert linux_rocm.memory_critical()
    (node / 'proc/meminfo').write_text('MemTotal:       100000 kB\nMemAvailable:    50000 kB\n')
    (node / 'sys/class/drm/renderD168/device/mem_info_vram_used').write_text(f'{274861129728 - 1000}\n')
    assert linux_rocm.memory_critical()


def test_gfx_names_and_unsupported_machines(node, monkeypatch):
    path = node / 'sys/class/kfd/kfd/topology/nodes/7/properties'
    path.write_text(path.read_text().replace('90402', '90010'))
    assert linux_rocm._gpus()[0]['gfx_arch'] == 'gfx90a'
    path.write_text(path.read_text().replace('90010', '110000'))
    assert linux_rocm._gpus()[0]['gfx_arch'] == 'gfx1100'
    (node / 'dev/kfd').unlink()
    assert not linux_rocm.supported()
    monkeypatch.setattr(linux_rocm, 'ROOT', node / 'nowhere')
    assert linux_rocm._gpus() == [] and linux_rocm.machine()['accelerator'] is None
    assert linux_rocm.memory_budget() is None and not linux_rocm.memory_critical()


def elf(needed: list[str]) -> bytes:
    """A minimal 64-bit little-endian ELF executable with a .dynstr and a .dynamic section."""
    strings = b'\0' + b''.join(name.encode() + b'\0' for name in needed)
    offsets = [1 + sum(len(n) + 1 for n in needed[:i]) for i in range(len(needed))]
    dynamic = b''.join(struct.pack('<qQ', 1, o) for o in offsets) + struct.pack('<qQ', 0, 0)
    strtab, dynoff = 64, 64 + len(strings)
    shoff = dynoff + len(dynamic)
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', header, 0x10, 3)  # ET_DYN (a PIE)
    struct.pack_into('<Q', header, 0x28, shoff)
    struct.pack_into('<HH', header, 0x3a, 64, 3)
    sections = [bytes(64), struct.pack('<IIQQQQIIQQ', 0, 3, 0, 0, strtab, len(strings), 0, 0, 1, 0),
                struct.pack('<IIQQQQIIQQ', 0, 6, 0, 0, dynoff, len(dynamic), 1, 0, 8, 16)]
    return bytes(header) + strings + dynamic + b''.join(sections)


def test_the_native_check_allows_the_hip_runtime_only(tmp_path):
    engine = tmp_path / 'engine'
    engine.write_bytes(elf(['libamdhip64.so.7', 'libhiprtc.so.7', 'libhsa-runtime64.so.1', 'libstdc++.so.6',
                            'libm.so.6', 'libgcc_s.so.1', 'libc.so.6', 'ld-linux-x86-64.so.2']))
    assert linux_rocm.native_problem(engine) is None
    engine.write_bytes(elf([]))  # static
    assert linux_rocm.native_problem(engine) is None
    for library in ('libpython3.12.so.1.0', 'librocblas.so.5', 'libhipblaslt.so.1', 'libMIOpen.so.1',
                    'libtorch_cpu.so', 'libaiter.so'):
        engine.write_bytes(elf(['libc.so.6', library]))
        assert linux_rocm.native_problem(engine).startswith(f'links {library}, a precompiled'), library
    engine.write_bytes(elf(['libc.so.6', 'libfoo.so.2']))
    assert linux_rocm.native_problem(engine) == ('links libfoo.so.2, which is not a system or HIP runtime library '
                                                 '(docs/libraries.md)')
    engine.write_text('#!/bin/sh\nexec python -m engine "$@"\n')
    assert 'not an ELF' in linux_rocm.native_problem(engine)
    engine.write_bytes(b'\x7fELF')
    assert linux_rocm.native_problem(engine) == 'not a readable ELF executable'


@pytest.mark.skipif(not Path('/bin/ls').is_file() or Path('/bin/ls').read_bytes()[:4] != b'\x7fELF'
                    or Path('/bin/ls').read_bytes()[4:6] != b'\x02\x01', reason='needs a 64-bit Linux /bin/ls')
def test_the_elf_reader_reads_a_real_executable():
    assert 'libc.so.6' in linux_rocm.needed_libraries(Path('/bin/ls').read_bytes())
