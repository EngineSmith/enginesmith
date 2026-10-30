import math

import pytest

from enginesmith.acceptance import Result, decide, score, segment_means
from enginesmith.task import Acceptance, Objective

BALANCED = Objective()
RULES = Acceptance()


def rows(*values):
    """(prefill, decode) speedups for cases p256, p1024, ..."""
    return tuple((f'p{256 * 4 ** i}', 256 * 4 ** i, p, d) for i, (p, d) in enumerate(values))


def test_the_score_is_the_weighted_geometric_mean():
    assert score(rows((2, 8)), BALANCED) == pytest.approx(4)
    weighted = Objective(prefill=1, decode=3, length_weights={1024: 0})
    assert score(rows((2, 8), (100, 100)), weighted) == pytest.approx(math.exp((math.log(2) + 3 * math.log(8)) / 4))
    assert score(rows((2, 0)), BALANCED) is None
    assert score(rows((2, 0)), Objective(prefill=1, decode=0)) == pytest.approx(2)
    assert segment_means(rows((2, 8), (8, 2)), BALANCED) == pytest.approx({'prefill': 4, 'decode': 4})


def result(*values, objective=BALANCED, passed=True):
    r = rows(*values)
    return Result(passed, score(r, objective) if passed else None, r)


def test_the_first_passing_candidate_is_accepted_and_failures_never():
    assert decide(None, result((1, 1)), BALANCED, RULES).reason == 'first_passing_candidate'
    assert decide(None, result((1, 1), passed=False), BALANCED, RULES).reason == 'failed'


def test_the_score_rule():
    old = result((1, 1), (1, 1))
    assert decide(old, result((1.02, 1.02), (1.02, 1.02)), BALANCED, RULES).reason == 'score_improved'
    assert not decide(old, result((1.005, 1.005), (1.005, 1.005)), BALANCED, RULES).accepted


def test_the_segment_rule_needs_the_other_segment_and_the_score_to_hold():
    old = result((1, 1), (1, 1))
    assert decide(old, result((1.02, 0.995), (1.02, 0.995)), BALANCED,
                  Acceptance(min_improvement=0.05)).reason == 'segment_improved'
    assert not decide(old, result((1.02, 0.98), (1.02, 0.98)), BALANCED,
                      Acceptance(min_improvement=0.05, case_improvement=0)).accepted


def test_the_case_rule_and_unweighted_segments():
    old = result((1, 1), (1, 1))
    candidate = result((1, 1.05), (1, 0.995))
    assert decide(old, candidate, BALANCED, Acceptance(min_improvement=0.05, segment_improvement=0.05)).reason \
        == 'case_improved'
    # A prefill-only objective ignores decode gains.
    prefill_only = Objective(prefill=1, decode=0)
    old = result((1, 1), (1, 1), objective=prefill_only)
    assert not decide(old, result((1, 2), (1, 2), objective=prefill_only), prefill_only, RULES).accepted


def test_zero_turns_a_rule_off():
    old = result((1, 1))
    assert not decide(old, result((1.5, 1.5)), BALANCED, Acceptance(0, 0, 0, 0.01)).accepted
