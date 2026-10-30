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
