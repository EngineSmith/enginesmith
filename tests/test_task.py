import pytest

from conftest import write_task
from enginesmith.task import DEFAULT_SKILL, Task, TaskError, load_task


def test_a_minimal_task_takes_the_defaults(tmp_path, model_dir, monkeypatch):
    monkeypatch.setenv('DSH_SESSION_ID', 'session-1')   # the user talks to DeepSeek Harness
    path = tmp_path / 'task.toml'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n')
    task = load_task(path)
    assert task.model == model_dir and task.vocab_size == 1000 and task.skill == DEFAULT_SKILL
    assert task.build == ('cargo', 'build', '--release') and task.executable == 'target/release/engine'
    assert task.native and task.agent == 'dsh' and task.lengths == (256, 1024, 4096, 16384)
    assert task.objective.prefill == task.objective.decode == 0.5
    assert task.gates.case_mismatch_percent == 20 and task.evaluation.rounds == 5
    assert task.budget.session_timeout_s == 7200 and task.budget.wall_time_s == 48 * 3600


def test_the_agent_defaults_to_the_one_the_user_talks_to(tmp_path, model_dir, monkeypatch):
    path = tmp_path / 'task.toml'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n')
    with pytest.raises(TaskError, match='agent.kind is not set'):
        load_task(path)
    monkeypatch.setenv('CLAUDECODE', '1')
    assert load_task(path).agent == 'claude'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\n[agent]\nkind = "codex"\n')
    assert load_task(path).agent == 'codex'   # a named kind wins


def test_the_frozen_json_round_trips(tmp_path, model_dir):
    task = load_task(write_task(tmp_path, model_dir, objective={'prefill': 0.2, 'decode': 0.8,
                                                                'length_weights': {'256': 3}}))
    again = Task.from_json(task.to_json())
    assert again == task and again.digest() == task.digest()
    assert task.objective.case_weight(256) == 3 and task.objective.case_weight(64) == 1


@pytest.mark.parametrize('sections, message', [
    ({'workload': {'lengths': [64, 64]}}, 'distinct'),
    ({'workload': {'lengths': [600000]}}, 'distinct integers'),
    ({'workload': {'output_tokens': 1}}, 'output_tokens'),
    ({'objective': {'prefill': 0, 'decode': 0}}, 'must not all be zero'),
    ({'objective': {'length_weights': {'128': 1}}}, 'not one of workload.lengths'),
    ({'acceptance': {'min_improvement': 1.5}}, 'min_improvement'),
    ({'gates': {'case_mismatch_percent': 101}}, 'case_mismatch_percent'),
    ({'agent': {'kind': 'claude'}}, 'agent.command'),
    ({'agent': {'kind': 'dsh', 'model': 'deepseek-flash'}}, 'own settings'),
    ({'evaluation': {'cooldown': 'soon'}}, 'duration'),
    ({'engine': {'executable': '../x'}}, 'relative path'),
    ({'budget': {'tokens': 5}}, 'unknown keys'),
    ({'agent': {'command': ['/usr/bin/env', 'DEEPSEEK_API_KEY=abc', 'pi', '{prompt}']}}, 'credential'),
    ({'agent': {'command': ['pi', '--api-key', 'sk-abcdefghijklmnopqrstu', '{prompt}']}}, 'credential'),
])
def test_invalid_tasks_are_refused(tmp_path, model_dir, sections, message):
    with pytest.raises(TaskError, match=message):
        load_task(write_task(tmp_path, model_dir, **sections))


def test_paths_resolve_next_to_the_task_and_must_exist(tmp_path, model_dir):
    path = tmp_path / 'task.toml'
    path.write_text('name = "m"\n[model]\npath = "missing"\n[agent]\nkind = "claude"\n[workload]\nvocab_size = 10\n')
    with pytest.raises(TaskError, match='is not a directory'):
        load_task(path)
    assert load_task(path, check_paths=False).model == tmp_path / 'missing'


def test_the_model_source_is_a_repository_id(tmp_path, model_dir):
    path = tmp_path / 'task.toml'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\nsource = "mlx-community/Qwen3-0.6B-4bit"\n'
               '[agent]\nkind = "claude"\n')
    assert load_task(path).model_source == 'mlx-community/Qwen3-0.6B-4bit'
    path.write_text(f'name = "m"\n[model]\npath = "{model_dir.name}"\nsource = "not a repository"\n'
               '[agent]\nkind = "claude"\n')
    with pytest.raises(TaskError, match='Hugging Face repository'):
        load_task(path)
