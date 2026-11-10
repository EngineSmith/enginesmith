"""The concurrent workload mode: the request plan, the task keys, the evaluator and the acceptance rows."""
import json
from pathlib import Path

import pytest

from conftest import VOCAB, fake_engine, write_task
from enginesmith import acceptance, concurrent, workload
from enginesmith.concurrent import ConcurrentEvaluator, ConcurrentPlan, percentile
from enginesmith.evaluate import rows
from enginesmith.task import PRESETS, Acceptance, ConcurrentObjective, Gates, TaskError, load_task

OBJECTIVE = ConcurrentObjective(0.4, 0.3, 0.3, {1: 0.5, 4: 0.5})


def plan(levels=(1, 4), requests=8, prompt_range=(64, 256), output_range=(8, 16), rounds=2, controls=False,
         gates=Gates(), objective=OBJECTIVE):
    structure = workload.request_plan(b's' * 32, levels, requests, prompt_range, output_range, 0.01)
    cases = tuple(c for c in workload.cases((), 8, None, controls)) if controls else ()
    return ConcurrentPlan(structure, cases, VOCAB, rounds, 0, 0.0, gates, objective, 'nonce', b'0' * 32, b's' * 32)


def evaluate(tmp_path, *flags, reference=('--reference', '--concurrent', '--batch', '1'), **kwargs):
    return ConcurrentEvaluator(fake_engine('--reference'), fake_engine(*reference),
                               fake_engine('--concurrent', *flags), plan(**kwargs), tmp_path / 'eval',
                               reference_controls=tmp_path / 'controls.json').run()


def test_the_request_plan_is_reproducible_and_inside_its_ranges():
    a = workload.request_plan(b'x', (1, 4, 16), 64, (256, 4096), (64, 512), 0.2)
    assert a == workload.request_plan(b'x', (1, 4, 16), 64, (256, 4096), (64, 512), 0.2)
    assert a != workload.request_plan(b'y', (1, 4, 16), 64, (256, 4096), (64, 512), 0.2)
    for level, spec in a.items():
        assert len(spec['requests']) == 64 and len(spec['clients']) == level
        assert all(256 <= r.length <= 4096 and 64 <= r.output_tokens <= 512 for r in spec['requests'])
        assert sorted(len(c) for c in spec['clients']) in ([64 // level] * level,)
        assert min(spec['starts_s']) == 0 and max(spec['starts_s']) <= 0.2
    assert len({r.length for r in a[16]['requests']}) > 32  # drawn, not a few fixed shapes


def test_percentiles_are_nearest_rank():
    assert percentile(list(range(1, 101)), 90) == 90 and percentile([5.0], 99) == 5.0
    assert percentile(list(range(1, 65)), 90) == 58 and percentile([], 50) is None


def test_a_batching_candidate_beats_a_one_at_a_time_reference_where_requests_overlap(tmp_path):
    result = evaluate(tmp_path, '--batch', '4')
    assert result['passed'] and result['mode'] == 'concurrent', result['failure']
    by_level = {c['level']: c for c in result['cases']}
    assert 0.6 < by_level[1]['throughput_speedup'] < 1.6           # one client: nothing to batch
    assert by_level[4]['throughput_speedup'] > 2.0                 # four clients: four requests per step
    assert by_level[4]['ttft_speedup'] > 1.5 and by_level[4]['latency_speedup'] > 1.5
    assert result['score'] > 1.2 and set(result['segments']) == {'throughput', 'ttft', 'latency'}
    assert all(m == 0 for c in result['cases'] for m in c['mismatches'])
    # Every token's time is kept with the evaluation.
    trace = json.loads((tmp_path / 'eval' / 'traces' / 'round-0-candidate-c4.json').read_text())
    assert len(trace['sent']) == 8 and all(len(t) == len(trace['tokens'][i]) for i, t in trace['arrivals'].items())


def test_holding_tokens_back_gains_nothing(tmp_path):
    streamed = evaluate(tmp_path / 'a', '--batch', '4', levels=(4,), objective=ConcurrentObjective(0, 0.5, 0.5, {4: 1}))
    held = evaluate(tmp_path / 'b', '--batch', '4', '--burst', levels=(4,),
                    objective=ConcurrentObjective(0, 0.5, 0.5, {4: 1}))
    s, h = streamed['cases'][0], held['cases'][0]
    assert h['ttft_speedup'] < 0.6 * s['ttft_speedup']             # its first token comes with its last
    assert h['latency_speedup'] < 1.3 * s['latency_speedup']       # end-to-end time does not move
    assert held['score'] < streamed['score']


def test_a_wrong_candidate_fails_the_gates_in_round_0(tmp_path):
    result = evaluate(tmp_path, '--batch', '4', '--wrong-from', '0', rounds=3)
    assert not result['passed'] and result['failure']['kind'] == 'gates' and result['cases'] == []
    assert 'round 0' in result['failure']['message'] and result['failure']['streams'][0]['first_mismatch'][0] == 0


def test_a_wrong_candidate_fails_at_the_controls(tmp_path):
    result = evaluate(tmp_path, '--batch', '4', '--wrong-from', '0', controls=True)
    assert not result['passed'] and result['cases'] == [] and result['controls']
    assert all(not c['passed'] for c in result['controls'])


def test_a_token_for_an_unknown_request_is_a_protocol_failure(tmp_path):
    result = evaluate(tmp_path, '--batch', '4', '--bad-id')
    assert not result['passed'] and result['failure']['kind'] == 'protocol'


def test_a_concurrent_result_feeds_the_acceptance_rules(tmp_path):
    result = evaluate(tmp_path, '--batch', '4')
    table = rows(result)
    assert [r[:2] for r in table] == [('c1', 1), ('c4', 4)]
    first = acceptance.decide(None, acceptance.Result(True, result['score'], table), OBJECTIVE,
                              Acceptance())
    assert first.accepted
    faster = tuple((i, k, t * 1.05, f, l) if k == 4 else (i, k, t, f, l) for i, k, t, f, l in table)
    decision = acceptance.decide(acceptance.Result(True, result['score'], table),
                                 acceptance.Result(True, acceptance.score(faster, OBJECTIVE), faster), OBJECTIVE,
                                 Acceptance())
    # One level's throughput 5% faster: the throughput mean over the levels gains 2.5%, the other segments nothing.
    assert decision.accepted and decision.reason == 'segment_improved'
    assert decision.best_case[:2] == ('c4', 'throughput') and decision.segment_gains['ttft'] == 0


# The task file --------------------------------------------------------------------------------------------------------

def concurrent_task(tmp_path, model_dir, workload_table=None, objective=None, **sections):
    preset = PRESETS['interactive']
    w = {'mode': 'concurrent', 'preset': 'interactive', 'arrival': 'closed', 'levels': preset['levels'],
         'requests_per_level': 64, 'prompt_range': preset['prompt_range'], 'output_range': preset['output_range'],
         'stagger': '0.2s', 'lengths': None, 'output_tokens': None, **(workload_table or {})}
    o = {'throughput': 0.4, 'ttft': 0.3, 'latency': 0.3, 'level_weights': {'1': 0.5, '4': 0.5}, **(objective or {})}
    return write_task(tmp_path, model_dir, workload=w, objective=o, **sections)


def test_a_concurrent_task_states_every_value(tmp_path, model_dir):
    task = load_task(concurrent_task(tmp_path, model_dir))
    assert task.mode == 'concurrent' and task.concurrent.levels == (1, 4) and task.concurrent.stagger_s == 0.2
    assert task.objective.weights() == (0.4, 0.3, 0.3) and task.objective.case_weight(4) == 0.5
    assert all(not c.timed for c in task.cases()) and task.max_request_length() == 4096 + 512
    frozen = type(task).from_json(json.loads(json.dumps(task.to_json())))
    assert frozen == task


@pytest.mark.parametrize('workload_table, objective, message', [
    ({'arrival': 'open'}, None, 'closed'),
    ({'levels': [1, 4], 'requests_per_level': 2}, None, 'at least the highest level'),
    ({'lengths': [256]}, None, 'applies to workload.mode = "single"'),
    ({'preset': 'nightly'}, None, 'workload.preset must be one of'),
    (None, {'level_weights': {'1': 1}}, 'every level'),
    (None, {'prefill': 0.5}, 'unknown keys'),
    ({'prompt_range': [512, 256]}, None, 'prompt_range'),
])
def test_a_concurrent_task_with_a_bad_value_is_refused(tmp_path, model_dir, workload_table, objective, message):
    with pytest.raises(TaskError, match=message):
        load_task(concurrent_task(tmp_path, model_dir, workload_table, objective))


def test_a_concurrent_task_needs_every_value(tmp_path, model_dir):
    path = concurrent_task(tmp_path, model_dir)
    text = path.read_text().replace('stagger = "0.2s"\n', '')
    path.write_text(text)
    with pytest.raises(TaskError, match='needs workload.stagger'):
        load_task(path)


def test_concurrent_keys_need_the_concurrent_mode(tmp_path, model_dir):
    with pytest.raises(TaskError, match='need workload.mode = "concurrent"'):
        load_task(write_task(tmp_path, model_dir, workload={'levels': [1, 4]}))


# A run ---------------------------------------------------------------------------------------------------------------

def test_a_concurrent_run_accepts_a_batching_engine_and_runs_the_final(tmp_path, model_dir, monkeypatch):
    from enginesmith.run import Run
    monkeypatch.setenv('FAKE_AGENT_PLAN', json.dumps({
        '001': ['--concurrent', '--batch', '1'],
        '002': {'flags': ['--concurrent', '--batch', '4'], 'dev_eval': ['--no-build', '--requests', '8']}}))
    path = concurrent_task(tmp_path, model_dir, {'requests_per_level': 8, 'prompt_range': [64, 256],
                                                 'output_range': [8, 16], 'stagger': '0.01s'},
                           budget={'sessions': 2, 'wall_time': '100h'})
    run = Run.create(load_task(path), tmp_path / 'runs')
    assert run.state['structure_seed']
    state = run.drive()
    assert [(r['label'], r['reason']) for r in state['rounds']] == [('001', 'first_passing_candidate'),
                                                                    ('002', 'score_improved')]
    assert state['incumbent'] == '002' and state['final']['passed'] and state['final']['score'] > 1.2
    final = json.loads((run.dir / 'final' / 'final.json').read_text())
    assert final['structure_seed'] and final['structure_seed'] != state['structure_seed']
    # The second session saw the first engine's levels and could run dev-eval on a public plan.
    session = (run.round_dir('002') / 'brief' / 'SESSION.md').read_text()
    assert '| 4 clients, 8 requests |' in session and 'throughput mean' in session
    dev = next((run.dir / 'rounds' / '002' / 'brief' / 'dev-eval').glob('*.json'))
    assert json.loads(dev.read_text())['passed']
    contract = (run.round_dir('002') / 'brief' / 'README.md').read_text()
    assert 'protocol 2' in contract and 'closed-loop clients' in contract
    report = json.loads((run.dir / 'report.json').read_text())
    assert report['mode'] == 'concurrent' and report['best']['cases'][1]['level'] == 4
    assert 'c4_throughput_speedup' in (run.dir / 'history.csv').read_text().splitlines()[0]
    assert 'Best engine by concurrency level' in (run.dir / 'report.html').read_text()


def test_a_concurrent_engine_can_be_shared_and_verified(tmp_path, model_dir, monkeypatch):
    from enginesmith.run import Run
    from enginesmith.share import share, verify
    monkeypatch.setenv('FAKE_AGENT_PLAN', json.dumps({'001': ['--concurrent', '--batch', '4']}))
    path = concurrent_task(tmp_path, model_dir, {'requests_per_level': 8, 'prompt_range': [64, 256],
                                                 'output_range': [8, 16], 'stagger': '0.01s'}, budget={'sessions': 1})
    run = Run.create(load_task(path), tmp_path / 'runs')
    assert run.drive()['final']['passed']
    result = share(run, out=tmp_path / 'out', agent_label='Claude Opus 5.5 subagents')
    package = Path(result['package'])
    readme = (package / 'README.md').read_text()
    assert 'closed-loop clients' in readme and '| 4 | ' in readme and 'normalized latency' in readme
    assert 'Claude Opus 5.5 subagents' in readme and 'Decode probe' not in readme and 'version 2' in readme
    assert str(tmp_path) not in readme and 'c4_throughput_speedup' in (package / 'history.csv').read_text()
    info = json.loads((package / 'enginesmith.json').read_text())
    assert info['settings']['mode'] == 'concurrent' and info['settings']['concurrent']['levels'] == [1, 4]
    assert str(tmp_path) not in json.dumps(info)
    summary = verify(package, model_dir, out=tmp_path / 'verify', rounds=1, reference=fake_engine('--reference'), log=lambda m: None)
    assert summary['passed'] and [c['level'] for c in summary['cases']] == [1, 4]
