import json

import pytest

from conftest import VOCAB, fake_engine
from enginesmith import workload
from enginesmith.evaluate import Evaluator, Plan, ReferenceFailed
from enginesmith.task import Gates, Objective


def plan(lengths=(64, 1024), controls=False, rounds=2, windows='nonce', gates=Gates(), output_tokens=8):
    return Plan(tuple(workload.cases(lengths, output_tokens, controls=controls)), VOCAB, rounds, 0, 0.0, gates,
                Objective(), windows, b'0' * 32)


def evaluate(tmp_path, *flags, reference=('--reference',), **kwargs):
    return Evaluator(fake_engine(*reference), fake_engine(*flags), plan(**kwargs), tmp_path / 'eval',
                     reference_controls=tmp_path / 'controls.json').run()


def test_a_correct_faster_candidate_passes_with_its_speedup(tmp_path):
    # Requests of tens of milliseconds, so that process overhead does not dominate the ratios.
    result = evaluate(tmp_path, '--speed', '2', lengths=(1024, 4096), output_tokens=32)
    assert result['passed'] and result['complete'] and result['failure'] is None
    for case in result['cases']:
        assert case['mismatches'] == [0, 0]
        assert 1.5 < case['decode_speedup'] < 2.6
    assert 1.5 < result['score'] < 2.6
    assert result['gates']['total_mismatches'] == 0


def test_a_wrong_candidate_fails_at_the_controls_before_any_timing(tmp_path):
    result = evaluate(tmp_path, '--wrong-from', '0', controls=True)
    assert not result['passed'] and result['failure']['kind'] == 'gates' and result['cases'] == []
    assert all(not c['passed'] and c['first_mismatch'][0] == 0 for c in result['controls'])
    assert not result['complete']
    # The reference's control tokens are cached for the next evaluation.
    assert set(json.loads((tmp_path / 'controls.json').read_text())) == {c['id'] for c in result['controls']}


def test_late_mismatches_within_the_cap_pass_and_beyond_it_fail(tmp_path):
    gates = Gates(case_mismatch_percent=30, total_mismatch_percent=30)
    passed = evaluate(tmp_path / 'a', '--wrong-from', '6', gates=gates)            # 2 of 8 steps
    assert passed['passed'] and [m[0] for m in passed['cases'][0]['judged'][1]] == [6, 7]
    result = evaluate(tmp_path / 'b', '--wrong-from', '4', gates=gates)            # 4 of 8 steps
    assert not result['passed'] and result['failure']['where'] == 'p64, p1024'


def test_a_wrong_long_case_fails_after_round_0(tmp_path):
    result = evaluate(tmp_path, '--wrong-from', '0', '--wrong-length', '1024', rounds=3, controls=True)
    assert all(c['passed'] for c in result['controls'])
    assert not result['passed'] and result['failure']['where'] == 'p1024' and not result['complete']


def test_the_total_cap_applies_across_streams(tmp_path):
    result = evaluate(tmp_path, '--wrong-from', '7', gates=Gates(case_mismatch_percent=20, total_mismatch_percent=10))
    assert not result['passed'] and result['failure']['where'] == 'total'
    assert result['gates'] == {'total_steps': 32, 'total_mismatches': 4, 'allowed': 3}


def test_a_crashing_candidate_fails_with_its_stderr(tmp_path):
    result = evaluate(tmp_path, '--crash-after', '3')
    assert not result['passed'] and result['failure']['kind'] == 'crash'
    assert 'stderr_tail' in result['failure']


def test_a_reference_failure_is_not_the_candidate_s(tmp_path):
    with pytest.raises(ReferenceFailed):
        evaluate(tmp_path, reference=('--crash-after', '0'))


def test_windows_are_fresh_per_round_and_public_ones_are_offset_0():
    p = plan()
    case = p.cases[0]
    assert p.offset(case, 0) != p.offset(case, 1) and p.offset(case, 0) >= workload.NONCE[0]
    assert plan(windows='public').offset(case, 1) == 0
    assert plan(windows='held_out').offset(case, 0) < workload.HELD_OUT[1]


def test_a_low_ranked_mismatching_token_fails_however_few(tmp_path):
    gates = Gates(case_mismatch_percent=30, total_mismatch_percent=30, mismatch_rank=10)
    assert evaluate(tmp_path / 'a', '--wrong-from', '7', reference=('--reference', '--rank', '10'), gates=gates)['passed']
    result = evaluate(tmp_path / 'b', '--wrong-from', '7', reference=('--reference', '--rank', '11'), gates=gates)
    assert not result['passed'] and result['failure']['streams'][0]['worst_rank'] == 11
    assert evaluate(tmp_path / 'c', '--wrong-from', '7', reference=('--reference', '--rank', '5000'),
                    gates=Gates(30, 30, 0))['passed']
