"""The score and the acceptance rules. Pure functions: no process, clock or file access.

A row is one timed case: (case_id, length, prefill_speedup, decode_speedup), where a speedup is the median over the
evaluation's paired rounds of reference time / candidate time (higher is better).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from enginesmith.task import Acceptance, Objective

SEGMENTS = ('prefill', 'decode')
_EPS = 1e-12


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def score(rows, objective: Objective) -> float | None:
    """exp of the weighted mean of log speedups over every (case, segment); a case's weight is its length weight times
    the segment weight. None when a weighted speedup is missing or not positive."""
    total = weighted = 0.0
    for _, length, *speedups in rows:
        for segment_weight, speedup in zip((objective.prefill, objective.decode), speedups):
            weight = objective.case_weight(length) * segment_weight
            if weight == 0:
                continue
            if not _positive(speedup):
                return None
            total += weight
            weighted += weight * math.log(speedup)
    return math.exp(weighted / total) if total else None


def segment_means(rows, objective: Objective) -> dict[str, float | None]:
    """Per segment, the case-weighted geometric mean of its speedups (None when a speedup is missing)."""
    means = {}
    for index, segment in enumerate(SEGMENTS):
        total = weighted = 0.0
        for _, length, *speedups in rows:
            weight = objective.case_weight(length)
            if weight == 0:
                continue
            if not _positive(speedups[index]):
                total = 0.0
                break
            total += weight
            weighted += weight * math.log(speedups[index])
        means[segment] = math.exp(weighted / total) if total else None
    return means


@dataclass(frozen=True)
class Result:
    """What the rules need from one evaluated candidate."""
    passed: bool
    score: float | None
    rows: tuple = ()


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    score_gain: float | None = None
    segment_gains: dict | None = None
    best_case: tuple | None = None  # (case_id, segment, gain)

    def to_json(self) -> dict:
        return {'accepted': self.accepted, 'reason': self.reason, 'score_gain': self.score_gain,
                'segment_gains': self.segment_gains, 'best_case': list(self.best_case) if self.best_case else None}


def decide(incumbent: Result | None, candidate: Result, objective: Objective, rules: Acceptance) -> Decision:
    """Accept the first passing candidate; afterwards one that passes and
    - improves the score by at least min_improvement, or
    - improves the prefill or decode mean by at least segment_improvement while the other mean and the score get
      worse by at most max_regression, or
    - improves one case's prefill or decode speedup by at least case_improvement while the score gets worse by at
      most max_regression.
    Only weighted cases and segments count. Thresholds should sit above the noise of an unchanged engine (the run's
    repeat evaluations show it)."""
    if not candidate.passed or candidate.score is None:
        return Decision(False, 'failed')
    if incumbent is None:
        return Decision(True, 'first_passing_candidate')
    gain = candidate.score / incumbent.score - 1
    old, new = segment_means(incumbent.rows, objective), segment_means(candidate.rows, objective)
    weights = dict(zip(SEGMENTS, (objective.prefill, objective.decode)))
    gains = {s: new[s] / old[s] - 1 for s in SEGMENTS if weights[s] > 0 and old[s] and new[s]}
    previous = {row[0]: row for row in incumbent.rows}
    best = None
    for case_id, length, *speedups in candidate.rows:
        if case_id not in previous or objective.case_weight(length) == 0:
            continue
        for index, segment in enumerate(SEGMENTS):
            before = previous[case_id][2 + index]
            if weights[segment] > 0 and _positive(before) and _positive(speedups[index]):
                case_gain = speedups[index] / before - 1
                if best is None or case_gain > best[2]:
                    best = (case_id, segment, case_gain)
    facts = dict(score_gain=gain, segment_gains=gains, best_case=best)
    if rules.min_improvement > 0 and gain + _EPS >= rules.min_improvement:
        return Decision(True, 'score_improved', **facts)
    floor = -rules.max_regression - _EPS
    if gain >= floor:
        if rules.segment_improvement > 0 and any(
                value + _EPS >= rules.segment_improvement and all(gains[o] >= floor for o in gains if o != s)
                for s, value in gains.items()):
            return Decision(True, 'segment_improved', **facts)
        if rules.case_improvement > 0 and best is not None and best[2] + _EPS >= rules.case_improvement:
            return Decision(True, 'case_improved', **facts)
    return Decision(False, 'insufficient_improvement', **facts)
