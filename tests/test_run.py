import json
import shutil
import sys

import pytest

from conftest import FAKE_ENGINE, write_task
from enginesmith import cli, engine_repo
from enginesmith.run import Run
from enginesmith.task import load_task


@pytest.fixture
def make_run(tmp_path, model_dir, monkeypatch):
    def make(plan, **sections):
        monkeypatch.setenv('FAKE_AGENT_PLAN', json.dumps(plan))
        task = load_task(write_task(tmp_path, model_dir, **sections))
        return Run.create(task, tmp_path / 'runs')
    return make


def outcomes(run):
    return [(r['label'], r['kind'], r['reason']) for r in run.state['rounds']]


def test_a_run_repairs_then_accepts_improvements_and_runs_the_final(make_run):
    run = make_run({'001': ['--wrong-from', '0'], '002': ['--speed', '1'], '003': ['--speed', '2']})
    state = run.drive()
    assert outcomes(run) == [('001', 'generate', 'failed'), ('002', 'repair', 'first_passing_candidate'),
                             ('003', 'optimize', 'score_improved')]
    assert state['incumbent'] == '003' and state['search_end'] == 'sessions_used' and state['phase'] == 'done'
    assert state['final']['passed'] and state['final']['round'] == '003' and state['final']['score'] > 1.4
    # The repair session saw the failure; the optimization session saw the best engine.
    repair = (run.round_dir('002') / 'brief' / 'SESSION.md').read_text()
    assert 'Round 001 (the candidate in engine/) failed' in repair and 'mismatching steps' in repair
    optimize = (run.round_dir('003') / 'brief' / 'SESSION.md').read_text()
    assert 'The current best engine: round 002' in optimize and '| p64 |' in optimize
    # Every candidate is kept in the engine repository; the notes travel with the engine.
    refs = engine_repo.git('for-each-ref', '--format=%(refname)', 'refs/rounds', cwd=run.repo).split()
    assert refs == [f'refs/rounds/{n}' for n in ('000', '001', '002', '003')]
    notes = engine_repo.git('show', f'{run.record("003")["snapshot"]}:NOTES.md', cwd=run.repo)
    assert notes.startswith('round 003')
    report = json.loads((run.dir / 'report.json').read_text())
    assert report['best']['round'] == '003' and (run.dir / 'report.html').read_text().startswith('<!doctype html>')
    # The evolution is recorded as the run goes: one row per round, then the held-out final, also as CSV.
    history = report['history']
    assert [r['round'] for r in history] == ['001', '002', '003', 'final']
    assert history[0]['failure'] and history[1]['accepted'] and history[2]['incumbent_score'] > 1.4
    assert history[1]['engine_lines'] > 0 and history[1]['dev_evals'] == 0
    assert history[2]['cases'][0]['decode_tok_s'] > 0 and history[2]['cases'][0]['reference_ttft_ms'] > 0
    assert report['milestones']['first_correct']['round'] == '002'
    assert report['milestones']['reference_matched']['round'] in ('002', '003')
    csv_lines = (run.dir / 'history.csv').read_text().splitlines()
    assert len(csv_lines) == 5 and 'p256_decode_tok_s' in csv_lines[0] and csv_lines[4].startswith('final,')
    held_out = json.loads((run.dir / 'final' / 'evaluation' / 'result.json').read_text())
    assert held_out['windows'] == 'held_out' and held_out['seed'] == json.loads(
        (run.dir / 'final' / 'final.json').read_text())['seed']


def test_an_unchanged_or_slower_candidate_is_rejected(make_run):
    run = make_run({'001': ['--speed', '2'], '002': 'same', '003': ['--speed', '1']})
    run.drive()
    assert outcomes(run) == [('001', 'generate', 'first_passing_candidate'), ('002', 'optimize', 'unchanged'),
                             ('003', 'optimize', 'insufficient_improvement')]
    assert 'evaluation' not in run.record('002')
    facts = json.loads((run.dir / 'state.json').read_text())
    assert facts['incumbent'] == '001'


def test_the_seed_is_evaluated_first(make_run, tmp_path):
    seed = tmp_path / 'seed'
    seed.mkdir()
    script = seed / 'engine'
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_ENGINE}" --vocab 1000 "$@"\n')
    script.chmod(0o755)
    run = make_run({'001': ['--speed', '3']}, engine={'seed': str(seed)}, budget={'sessions': 1})
    run.drive()
    assert outcomes(run) == [('000', 'seed', 'first_passing_candidate'), ('001', 'optimize', 'score_improved')]


def test_a_session_timeout_keeps_the_engine_directory_as_it_is(make_run):
    run = make_run({'001': 'sleep'}, agent={'session_timeout': '2s'}, budget={'sessions': 1})
    run.drive()
    record = run.record('001')
    assert record['session']['timed_out'] and record['unchanged'] and record['decision']['reason'] == 'unchanged'
    assert run.state['final'] == {'skipped': 'no candidate passed'}


def test_an_interrupted_session_is_not_restarted(make_run):
    run = make_run({'001': ['--speed', '1']}, budget={'sessions': 1})
    run.session_registries = []
    run._start_round()
    record = run.record('001')
    record['session_started'] = 1.0  # the driver died during the session
    run.save_record(record)
    (run.dir / 'work' / '001').mkdir(parents=True)
    engine_repo.checkout(run.repo, record['base'], run.dir / 'work' / '001' / 'engine')
    Run(run.dir).drive()
    run = Run(run.dir)
    assert run.record('001')['session'] == {'interrupted': True}
    assert outcomes(run) == [('001', 'generate', 'unchanged')]


def test_stop_ends_the_search_and_continue_search_resumes_it(make_run, capsys):
    run = make_run({'001': ['--speed', '1'], '002': ['--speed', '2']}, budget={'sessions': 2})
    cli.main(['stop', str(run.dir)])
    run.drive()
    assert run.state['search_end'] == 'stopped' and run.state['phase'] == 'done' and not run.state['rounds']
    run = make_run({'001': ['--speed', '1']}, budget={'sessions': 1})
    cli.main(['stop', str(run.dir)])
    cli.main(['resume', str(run.dir), '--continue-search'])
    run = Run(run.dir)
    assert outcomes(run) == [('001', 'generate', 'first_passing_candidate')] and run.state['final']['passed']
    status = cli.status(run.dir)
    assert status['best']['round'] == '001' and not status['driver_running'] and 'share' in status['next']


def test_dev_eval_inside_a_session(make_run):
    run = make_run({'001': {'flags': ['--speed', '2'], 'dev_eval': ['--no-build', '--lengths', '64']}},
                   budget={'sessions': 1}, workload={'controls': True})
    run.drive()
    output = engine_repo.git('show', f'{run.record("001")["snapshot"]}:dev-eval.txt', cwd=run.repo)
    assert output.startswith('exit 0') and 'PASSED' in output and '| p64 |' in output
    result = json.loads((run.round_dir('001') / 'brief' / 'dev-eval' / '001.json').read_text())
    assert result['passed'] and result['windows'] == 'public' and [c['id'] for c in result['cases']] == ['p64']
    assert run.record('001')['dev_evals'] == {'runs': 1, 'passed': 1}


def test_a_changed_harness_stops_the_evaluations(make_run, monkeypatch):
    run = make_run({'001': ['--speed', '1']}, budget={'sessions': 1})
    monkeypatch.setattr('enginesmith.run.harness_digest', lambda: 'changed')
    with pytest.raises(Exception, match='harness source changed'):
        run.drive()


def test_an_agent_that_cannot_run_ends_the_search(make_run, tmp_path):
    run = make_run({}, agent={'command': [sys.executable, '-c', 'import sys; sys.exit(1)']}, budget={'sessions': 5})
    run.drive()
    assert [r['reason'] for r in run.state['rounds']] == ['unchanged', 'unchanged']
    assert run.state['search_end'] == 'agent_failed' and 'exited within seconds twice' in run.state['agent_failed']
    run = make_run({}, agent={'command': ['/nonexistent/agent']}, budget={'sessions': 5})
    run.drive()
    assert run.state['search_end'] == 'agent_failed' and len(run.state['rounds']) == 1


def test_large_files_are_left_out_of_the_snapshot(make_run, monkeypatch):
    monkeypatch.setattr(engine_repo, 'MAX_FILE_BYTES', 1000)
    run = make_run({'001': ['--speed', '1']}, budget={'sessions': 1})
    original = engine_repo.snapshot

    def with_weights(repository, worktree, label):
        (worktree / 'weights.bin').write_bytes(b'0' * 5000)
        return original(repository, worktree, label)
    monkeypatch.setattr(engine_repo, 'snapshot', with_weights)
    run.drive()
    assert run.record('001')['files_left_out'] == ['weights.bin']
    assert 'weights.bin' not in engine_repo.git('ls-tree', '--name-only', run.record('001')['snapshot'], cwd=run.repo)


def test_the_session_log_keeps_the_output_without_streaming_deltas(make_run):
    script = ('import sys; print(\'{"type":"message_update","delta":"x"}\'); print(\'{"type":"message_end"}\'); '
              'print("plain line")')
    run = make_run({}, agent={'command': [sys.executable, '-c', script]}, budget={'sessions': 1})
    run.drive()
    log = (run.round_dir('001') / 'session.log').read_text()
    assert log == '{"type":"message_end"}\nplain line\n'


def test_the_agent_s_reported_usage_is_recorded(make_run):
    script = ('import json; print(json.dumps({"type": "message_end", "message": {"role": "assistant", "model": "m1", '
              '"usage": {"input": 100, "output": 20, "cost": {"total": 0.25}}}})); print(json.dumps({"type": '
              '"message_end", "message": {"role": "assistant", "usage": {"input": 50, "cost": {"total": 0.5}}}}))')
    run = make_run({}, agent={'command': [sys.executable, '-c', script]}, budget={'sessions': 1})
    run.drive()
    session = run.record('001')['session']
    assert session['cost_usd'] == 0.75 and session['model'] == 'm1'
    assert session['input_tokens'] == 150 and session['output_tokens'] == 20 and session['cache_read_tokens'] is None
    assert '$0.75' in (run.dir / 'report.html').read_text()
    assert json.loads((run.dir / 'report.json').read_text())['milestones']['total_input_tokens'] == 150
