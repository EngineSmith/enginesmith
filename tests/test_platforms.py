import importlib
import inspect

import pytest

from conftest import write_task
from enginesmith import brief, platforms, references
from enginesmith.task import TaskError, load_task

INTERFACE = {'NAME': str, 'DEFAULT_REFERENCE': (str, type(None))}
FUNCTIONS = ('supported', 'machine', 'checks', 'memory_budget', 'memory_critical', 'keep_awake', 'native_problem')


@pytest.mark.parametrize('name', platforms.PLATFORMS)
def test_every_platform_implements_the_interface(name):
    module = importlib.import_module(name)
    for attribute, kind in INTERFACE.items():
        assert isinstance(getattr(module, attribute), kind), attribute
    for function in FUNCTIONS:
        assert inspect.isfunction(getattr(module, function)), function
    assert module.DEFAULT_REFERENCE in (None, *references.KINDS)


def test_the_generic_platform_works_anywhere(tmp_path):
    generic = importlib.import_module('enginesmith.platforms.generic')
    facts = generic.machine()
    assert {'platform', 'os', 'arch', 'chip', 'accelerator', 'memory_bytes'} <= set(facts)
    problems, _ = generic.checks(facts)
    assert 'CONTRIBUTING.md' in problems[0]
    assert generic.native_problem(tmp_path) is None and not generic.memory_critical()
    assert platforms.PLATFORMS[-1] == 'enginesmith.platforms.generic'


def test_the_reference_comes_from_the_task_or_the_platform(tmp_path, model_dir, monkeypatch):
    path = tmp_path / 'task.toml'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "claude"\n[reference]\n'
                    'kind = "mlx-lm"\n')
    task = load_task(path)
    assert brief.reference_argv(task)[1:] == ['-m', 'enginesmith.references.mlx_lm', '--model', str(model_dir)]
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "claude"\n')
    monkeypatch.setattr(platforms, 'current', lambda: importlib.import_module('enginesmith.platforms.generic'))
    monkeypatch.setattr(brief, 'current', platforms.current)
    with pytest.raises(ValueError, match='no default reference'):
        brief.reference_argv(load_task(path))
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "claude"\n[reference]\n'
                    'kind = "vllm"\n')
    with pytest.raises(TaskError, match='reference.kind'):
        load_task(path)
    custom = load_task(write_task(tmp_path, model_dir))
    assert brief.reference_kind(custom) is None and brief.reference_argv(custom) == list(custom.reference)


NEURON_LS = '''[{"instance_type": "trn2.48xlarge", "neuron_device": 0, "nc_count": 4, "logical_neuroncore_config": 2,
  "memory_size": 103079215104, "neuroncore_ids": [0, 1, 2, 3], "neuron_processes": []},
 {"instance_type": "trn2.48xlarge", "neuron_device": 1, "nc_count": 4, "logical_neuroncore_config": 2,
  "memory_size": 103079215104, "neuroncore_ids": [4, 5, 6, 7], "neuron_processes": []}]'''


def test_the_neuron_platform_reads_neuron_ls(monkeypatch):
    neuron = importlib.import_module('enginesmith.platforms.neuron')
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: NEURON_LS if 'neuron-ls' in argv[0] else None)
    facts = neuron.machine()
    assert facts['chip'] == 'trn2.48xlarge' and facts['accelerator_cores'] == 8 and facts['neuron_devices'] == 2
    assert facts['memory_bytes'] == 103079215104 // 4 and facts['logical_neuroncore_config'] == 2
    assert neuron.memory_budget() == int(103079215104 // 4 * 0.9)
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: None)
    problems, _ = neuron.checks(neuron.machine())
    assert any('neuron-ls lists no Neuron device' in p for p in problems)


def test_the_neuron_native_check_refuses_scripts_and_frameworks(tmp_path, monkeypatch):
    neuron = importlib.import_module('enginesmith.platforms.neuron')
    script = tmp_path / 'engine.sh'
    script.write_text('#!/bin/sh\n')
    assert 'not an ELF' in neuron.native_problem(script)
    binary = tmp_path / 'engine'
    binary.write_bytes(b'\x7fELF' + bytes(60))
    allowed = ('\tlinux-vdso.so.1 (0x0)\n\tlibnrt.so.1 => /opt/aws/neuron/lib/libnrt.so.1 (0x0)\n'
               '\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x0)\n')
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: allowed)
    assert neuron.native_problem(binary) is None
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20:
                        allowed + '\tlibpython3.12.so.1.0 => /usr/lib/libpython3.12.so.1.0 (0x0)\n')
    assert 'libpython3.12' in neuron.native_problem(binary)
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: allowed + '\tlibtorch_cpu.so => /x (0x0)\n')
    assert 'libtorch_cpu' in neuron.native_problem(binary)
    cpu_only = '\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x0)\n'
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: cpu_only)
    assert 'NeuronCores' in neuron.native_problem(binary)


def test_a_fixed_length_reference_is_told_the_longest_request(tmp_path, model_dir):
    path = tmp_path / 'task.toml'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "claude"\n[reference]\n'
                    'kind = "nxdi"\n[workload]\nlengths = [256, 1024]\noutput_tokens = 65\n')
    task = load_task(path)
    assert task.max_request_length() == 4097 + 65  # the longest control prompt
    assert brief.reference_argv(task)[1:] == ['-m', 'enginesmith.references.nxdi', '--model', str(model_dir),
                                              '--max-length', '4162']
    assert references.command('mlx-lm', model_dir, 4162)[-1] == str(model_dir)


def test_the_neuron_platform_gives_engines_the_run_cores(monkeypatch, tmp_path):
    neuron = importlib.import_module('enginesmith.platforms.neuron')
    monkeypatch.delenv('ENGINESMITH_NEURON_CORE', raising=False)
    assert neuron.engine_environment() == {'NEURON_RT_VISIBLE_CORES': '0-1'}
    monkeypatch.setenv('ENGINESMITH_NEURON_CORE', '8')
    assert neuron.engine_environment() == {'NEURON_RT_VISIBLE_CORES': '8-9'}
    monkeypatch.setattr(platforms, 'current', lambda: neuron)
    from enginesmith import protocol
    assert protocol.engine_environment(tmp_path)['NEURON_RT_VISIBLE_CORES'] == '8-9'
    monkeypatch.setenv('ENGINESMITH_NEURON_CORE', 'x')
    with pytest.raises(ValueError):
        neuron.engine_environment()


def test_a_tensor_parallel_reference_gets_its_degree_and_the_run_its_cores(tmp_path, model_dir, monkeypatch):
    path = tmp_path / 'task.toml'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "claude"\n[reference]\n'
                    'kind = "nxdi"\ntensor_parallel = 4\n[workload]\nlengths = [256]\n')
    task = load_task(path)
    assert task.reference_tp == 4 and brief.reference_argv(task)[-2:] == ['--tp', '4']
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "claude"\n[reference]\n'
                    'kind = "mlx-lm"\ntensor_parallel = 2\n')
    with pytest.raises(TaskError, match='tensor_parallel'):
        load_task(path)
    neuron = importlib.import_module('enginesmith.platforms.neuron')
    monkeypatch.setenv('ENGINESMITH_NEURON_CORE', '16')
    monkeypatch.setenv('ENGINESMITH_NEURON_CORES', '8')
    assert neuron.engine_environment() == {'NEURON_RT_VISIBLE_CORES': '16-23'}
    monkeypatch.setenv('ENGINESMITH_NEURON_CORES', '1')
    assert neuron.engine_environment() == {'NEURON_RT_VISIBLE_CORES': '16'}
