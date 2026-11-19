"""CPU tests using synthetic telemetry and ELF bytes; these are not GPU evidence."""
import struct
import subprocess
from types import SimpleNamespace

import pytest

from enginesmith import platforms, protocol
from enginesmith.platforms import linux_cuda as cuda


def uuid(n):
    return f'GPU-00000000-0000-0000-0000-{n:012x}'


def telemetry(count=8, used=None, mig='Disabled'):
    return '\n'.join(f'{7 - n}, {uuid(n)}, 00000000:{n + 1:02x}:00.0, NVIDIA Test GPU, 81920, '
                     f'{(used or {}).get(n, 1000)}, 580.00, 9.0, {mig}' for n in reversed(range(count)))


@pytest.fixture
def node(tmp_path, monkeypatch):
    (tmp_path / 'proc').mkdir()
    (tmp_path / 'proc/meminfo').write_text('MemTotal: 100000 kB\nMemAvailable: 50000 kB\n')
    monkeypatch.setattr(cuda, 'ROOT', tmp_path)
    monkeypatch.setattr(cuda, '_run', lambda argv, timeout=20: telemetry() if argv[0] == 'nvidia-smi' else None)
    monkeypatch.setattr(cuda.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cuda.platform, 'machine', lambda: 'x86_64')
    for name in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'ENGINESMITH_CUDA_UUIDS'):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_cpu_fixture_describes_all_eight_devices_and_balanced_capacity(node):
    facts = cuda.machine()
    assert facts['gpu_uuids'] == [uuid(n) for n in range(8)]
    assert facts['visible_gpus'] == 8 and facts['memory_bytes'] == 8 * 81920 * 2**20
    assert facts['balanced_tp_memory_bytes'] == facts['memory_bytes']
    assert cuda.memory_budget() == int((81920 - 1000) * 2**20 * 8 * .85)
    assert not cuda.memory_critical()


def test_both_arms_receive_identical_eight_full_uuids_in_rank_order(node, monkeypatch, tmp_path):
    order = [7, 2, 0, 1, 6, 3, 4, 5]
    chosen = ','.join(uuid(n) for n in order)
    monkeypatch.setenv('ENGINESMITH_CUDA_UUIDS', chosen)
    monkeypatch.setattr(platforms, 'current', lambda: cuda)
    reference = protocol.engine_environment(tmp_path / 'reference')
    candidate = protocol.engine_environment(tmp_path / 'candidate', engine_cores=8)
    # An external adapter may own TP without a task-level degree declaration.
    candidate_default = protocol.engine_environment(tmp_path / 'candidate-default', engine_cores=1)
    for arm in (reference, candidate, candidate_default):
        assert arm['CUDA_VISIBLE_DEVICES'] == chosen
        assert arm['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
        assert arm['HF_HUB_OFFLINE'] == '1'
    with pytest.raises(ValueError, match='fewer GPUs'):
        cuda.engine_environment(9)


def test_explicit_selection_cannot_expand_the_inherited_visibility(node, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', f'{uuid(3)},{uuid(2)}')
    monkeypatch.setenv('ENGINESMITH_CUDA_UUIDS', uuid(0))
    with pytest.raises(ValueError, match='unavailable'):
        cuda.visible_gpus()
    monkeypatch.setenv('ENGINESMITH_CUDA_UUIDS', uuid(2))
    assert cuda.engine_environment(1)['CUDA_VISIBLE_DEVICES'] == uuid(2)


def test_numeric_visibility_uses_explicit_pci_order_not_smi_indices(node, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,0')
    with pytest.raises(ValueError, match='PCI_BUS_ID'):
        cuda.visible_gpus()
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    assert [g['uuid'] for g in cuda.visible_gpus()] == [uuid(7), uuid(0)]


@pytest.mark.parametrize('selection', ['GPU-00000000', f'{uuid(1)},{uuid(1)}', '8', 'MIG-test', '0,', 'invalid'])
def test_invalid_or_ambiguous_visibility_fails_closed(node, monkeypatch, selection):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', selection)
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    assert cuda.machine()['selection_error']
    assert cuda.memory_budget() is None and cuda.memory_critical()
    with pytest.raises(ValueError):
        cuda.engine_environment()


def test_empty_selection_and_mig_are_not_supported(node, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert cuda.machine()['visible_gpus'] == 0
    assert cuda.memory_critical()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    monkeypatch.setattr(cuda, '_run', lambda *args, **kwargs: telemetry(mig='Enabled'))
    assert 'MIG' in cuda.machine()['selection_error']


def test_memory_watchdog_checks_nonfirst_gpu_and_lost_telemetry(node, monkeypatch):
    monkeypatch.setattr(cuda, '_run', lambda *args, **kwargs: telemetry(used={7: 81919}))
    assert cuda.memory_critical()
    assert cuda.memory_budget() == int(2**20 * 8 * .85)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid(0))
    assert not cuda.memory_critical()  # only selected GPUs govern this run
    (node / 'proc/meminfo').write_text('MemTotal: 100000 kB\nMemAvailable: 2000 kB\n')
    assert cuda.memory_critical()
    monkeypatch.setattr(cuda, '_run', lambda *args, **kwargs: None)
    assert cuda.memory_critical()


@pytest.mark.parametrize('text', ['bad row', telemetry().replace('81920', '[N/A]'),
                                telemetry().replace('9.0', '[N/A]'), telemetry(mig='Unknown'),
                                telemetry().replace(uuid(1), uuid(0))])
def test_malformed_telemetry_is_refused(text):
    with pytest.raises(ValueError):
        cuda.parse_gpus(text)


def elf(libraries=(), *, machine=62, kind=2, entry=0x400100, terminate=True):
    """Sectionless ELF with an executable PT_LOAD and optional PT_DYNAMIC."""
    data = bytearray(1024)
    strings = b'\0' + b''.join(name.encode() + b'\0' for name in libraries)
    dynamic = [(5, 0x400300), (10, len(strings))]
    offset = 1
    for name in libraries:
        dynamic.append((1, offset))
        offset += len(name) + 1
    if terminate:
        dynamic.append((0, 0))
    struct.pack_into('<16sHHIQQQIHHHHHH', data, 0, b'\x7fELF\x02\x01\x01' + bytes(9), kind, machine, 1,
                     entry, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    struct.pack_into('<IIQQQQQQ', data, 64, 1, 5, 0, 0x400000, 0x400000, len(data), len(data), 4096)
    struct.pack_into('<IIQQQQQQ', data, 120, 2, 4, 256, 0x400100, 0x400100, 16 * len(dynamic), 16 * len(dynamic), 8)
    for n, pair in enumerate(dynamic):
        struct.pack_into('<qQ', data, 256 + 16 * n, *pair)
    data[768:768 + len(strings)] = strings
    return bytes(data)


def test_sectionless_native_elf_checks_direct_libraries_without_running_it(node, tmp_path):
    engine = tmp_path / 'engine'
    libs = ['libc.so.6', 'libcudart.so.12', 'libstdc++.so.6', 'libnccl.so.2']
    engine.write_bytes(elf(libs))
    assert cuda.needed_libraries(engine.read_bytes()) == libs
    assert cuda.native_problem(engine) is None
    for library in ('libcublas.so.12', 'libcublasLt.so.12', 'libcudnn.so.9', 'libnvinfer.so.10',
                    'libpython3.12.so.1.0', 'libtorch.so', 'libunknown.so', '/tmp/libc.so.6'):
        engine.write_bytes(elf(libs + [library]))
        assert library in cuda.native_problem(engine)


@pytest.mark.parametrize('data', [b'#!/bin/sh\n', b'\x7fELF', elf(machine=183), elf(kind=1), elf(entry=0),
                                elf(entry=0x500000), elf(['libc.so.6'], terminate=False), elf()[:500]])
def test_non_native_wrong_architecture_and_malformed_elf_are_refused(node, tmp_path, data):
    engine = tmp_path / 'engine'
    engine.write_bytes(data)
    assert cuda.native_problem(engine)


def test_out_of_bounds_dynamic_string_table_is_refused(node, tmp_path):
    data = bytearray(elf(['libc.so.6']))
    struct.pack_into('<Q', data, 264, 0x999999)
    engine = tmp_path / 'engine'
    engine.write_bytes(data)
    assert 'string table' in cuda.native_problem(engine)


def test_elf_interpreter_wrappers_and_dependency_filters_are_refused(node, tmp_path):
    engine = tmp_path / 'engine'
    data = bytearray(elf(['libc.so.6']))
    struct.pack_into('<H', data, 56, 3)  # third program header
    interpreter = b'/usr/bin/python3\0'
    struct.pack_into('<IIQQQQQQ', data, 176, 3, 4, 900, 0x400384, 0x400384,
                     len(interpreter), len(interpreter), 1)
    data[900:900 + len(interpreter)] = interpreter
    engine.write_bytes(data)
    assert 'program interpreter' in cuda.native_problem(engine)
    data = bytearray(elf(['libc.so.6']))
    struct.pack_into('<q', data, 256, 0x7fffffff)
    engine.write_bytes(data)
    assert 'dependency filters' in cuda.native_problem(engine)


def test_native_pie_and_static_program_headers_are_accepted(node, tmp_path):
    engine = tmp_path / 'engine'
    engine.write_bytes(elf(['libc.so.6'], kind=3))
    assert cuda.native_problem(engine) is None
    data = bytearray(elf())
    struct.pack_into('<H', data, 56, 1)  # static program: only PT_LOAD
    engine.write_bytes(data)
    assert cuda.native_problem(engine) is None


def elf_with_interpreter(interpreter, *, machine=62):
    data = bytearray(elf(['libc.so.6'], machine=machine))
    struct.pack_into('<H', data, 56, 3)
    struct.pack_into('<IIQQQQQQ', data, 176, 3, 4, 900, 0x400384, 0x400384,
                     len(interpreter), len(interpreter), 1)
    data[900:900 + len(interpreter)] = interpreter
    return bytes(data)


@pytest.mark.parametrize('architecture,machine,loader', [
    ('x86_64', 62, b'/lib64/ld-linux-x86-64.so.2\0'),
    ('x86_64', 62, b'/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2\0'),
    ('aarch64', 183, b'/lib/ld-linux-aarch64.so.1\0'),
    ('aarch64', 183, b'/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1\0'),
])
def test_system_loader_locations_match_the_host_architecture(node, monkeypatch, architecture, machine, loader):
    monkeypatch.setattr(cuda.platform, 'machine', lambda: architecture)
    assert cuda.needed_libraries(elf_with_interpreter(loader, machine=machine)) == ['libc.so.6']


@pytest.mark.parametrize('loader', [
    b'/tmp/ld-linux-x86-64.so.2\0', b'ld-linux-x86-64.so.2\0',
    b'/lib64/../tmp/ld-linux-x86-64.so.2\0', b'/lib64/ld-linux-x86-64.so.2',
    b'/lib64/ld-linux-x86-64.so.2\0extra\0', b'/lib/ld-linux-aarch64.so.1\0',
])
def test_loader_basename_alone_does_not_allow_a_custom_interpreter(node, loader):
    with pytest.raises(ValueError, match='program interpreter'):
        cuda.needed_libraries(elf_with_interpreter(loader))


def test_telemetry_command_is_bounded_and_does_not_need_a_run_configuration(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=' telemetry \n')

    monkeypatch.setattr(cuda.subprocess, 'run', run)
    assert cuda._run(['nvidia-smi']) == 'telemetry'
    assert calls == [(['nvidia-smi'], {'capture_output': True, 'text': True, 'timeout': 20.0})]
    monkeypatch.setattr(cuda.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stdout='partial'))
    assert cuda._run(['nvidia-smi']) is None


@pytest.mark.parametrize('error', [OSError('unavailable'), subprocess.TimeoutExpired('nvidia-smi', 20)])
def test_unavailable_or_timed_out_telemetry_fails_closed(monkeypatch, error):
    def run(*args, **kwargs):
        raise error
    monkeypatch.setattr(cuda.subprocess, 'run', run)
    assert cuda._run(['nvidia-smi']) is None


def test_doctor_requires_native_compilers_and_reports_external_reference_limit(node, monkeypatch):
    monkeypatch.setattr(cuda.shutil, 'which', lambda executable: None)
    problems, notes = cuda.checks(cuda.machine())
    assert any('nvcc' in problem for problem in problems)
    assert any('C++' in problem for problem in problems)
    assert any('no built-in reference' in note for note in notes)
    assert cuda.DEFAULT_REFERENCE is None
