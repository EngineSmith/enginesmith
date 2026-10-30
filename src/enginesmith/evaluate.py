"""Paired evaluation of a candidate engine against the reference (docs/design.md, "Evaluation").

1. Controls: the candidate generates every control case (short prompts at kernel boundaries). Its tokens are compared
   with the reference's and the gates are applied, so a broken engine fails in minutes, before any timing.
2. Rounds: per round, both engines run every timed case on the same prompt, the reference first in even rounds and
   the candidate first in odd ones. Each engine process loads the model, runs its warmup requests and then, per case,
   a timed prefill request generate(P, 1) and a timed decode request generate((), n - 1). Only one engine is resident
   at a time, and every process starts after a cooldown.
3. Judging: a candidate token stream identical to the reference's agrees by definition. A stream that differs is
   teacher-forced through the reference, and every step whose token is not the reference's top choice is a
   mismatch. A stream passes with at most case_mismatch_percent mismatching steps, each of whose tokens is among
   the reference's top mismatch_rank choices, and the evaluation with at most total_mismatch_percent of all judged
   steps. (On Lily prompts the model's output is repetitive, so teacher forcing alone is lenient: a stream of one
   repeated token is judged to mismatch at only a few steps, but those tokens rank in the hundreds or thousands.)

Speedups are per case and segment the median over rounds of reference time / candidate time.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from enginesmith import acceptance, workload
from enginesmith.processes import Registry
from enginesmith.protocol import Engine, EngineFailure
from enginesmith.task import Gates, Objective

HELLO_TIMEOUT = 1800.0
REFERENCE_TIMEOUT = 4 * 3600.0
CONTROL_TIMEOUT = 600.0
# A candidate request may take TIMEOUT_FACTOR times the reference's time for the same case and segment, and at
# least MIN_TIMEOUT seconds, so a slow but correct first engine can still become the baseline.
TIMEOUT_FACTOR = 50
MIN_TIMEOUT = 300.0
# The version of the judging rules, the judging zone of zones.toml. A change to that zone adds a version there and
# sets this to it; runs, reports and shared engines record it.
EVALUATION_VERSION = 1
WARMUP_LENGTH = 2048


class CandidateFailed(Exception):
    def __init__(self, failure: dict):
        super().__init__(failure['message'])
        self.failure = failure


class ReferenceFailed(Exception):
    pass


class Stopped(Exception):
    pass


@dataclass(frozen=True)
class Plan:
    cases: tuple                 # workload.Case, timed and control
    vocab: int
    rounds: int
    warmup: int
    cooldown_s: float
    gates: Gates
    objective: Objective
    windows: str = 'nonce'       # 'public' (offset 0), 'nonce' or 'held_out'
    seed: bytes = b''

    def offset(self, case: workload.Case, round_index: int) -> int:
        if self.windows == 'public' or not case.timed:
            return 0
        span = workload.NONCE if self.windows == 'nonce' else workload.HELD_OUT
        return workload.draw_offset(self.seed, f'{self.windows}/{round_index}/{case.id}', case.length, span)

    def warmup_prompt(self, index: int) -> list[int]:
        length = min(WARMUP_LENGTH, max(c.length for c in self.cases))
        offset = workload.draw_offset(self.seed or b'public', f'warmup/{index}', length, workload.NONCE)
        return workload.prompt(length, self.vocab, offset)


def allowed(steps: int, percent: int) -> int:
    return steps * percent // 100


class Evaluator:
    def __init__(self, reference_argv, candidate_argv, plan: Plan, directory: Path, *, registry: Registry | None = None,
                 log=lambda message: None, stop=lambda: False, reference_controls: Path | None = None):
        self.reference_argv, self.candidate_argv, self.plan = list(reference_argv), list(candidate_argv), plan
        self.directory, self.registry, self.log, self.stop = Path(directory), registry, log, stop
        self.reference_controls = reference_controls
        self.timed = [c for c in plan.cases if c.timed]
        self.controls = [c for c in plan.cases if not c.timed]
        self.sessions = 0
        self.load_ms = {'reference': [], 'candidate': []}
        self.started = time.time()

    # Engine sessions -------------------------------------------------------------------------------------------

    def _open(self, role: str, label: str) -> Engine:
        if self.stop():
            raise Stopped()
        if self.sessions and self.plan.cooldown_s:
            time.sleep(self.plan.cooldown_s)
        self.sessions += 1
        argv = self.reference_argv if role == 'reference' else self.candidate_argv
        engine = Engine(argv, scratch=self.directory / 'scratch' / f'{self.sessions:03d}-{role}-{label}',
                        label=role, registry=self.registry)
        try:
            self.load_ms[role].append(engine.hello(HELLO_TIMEOUT))
        except EngineFailure as exc:
            engine.close()
            self._failed(role, exc, engine, f'{label}: loading')
        return engine

    def _failed(self, role: str, exc: EngineFailure, engine: Engine, where: str):
        if role == 'reference':
            raise ReferenceFailed(f'reference {where}: {exc} {engine.stderr_tail(2000)}') from exc
        raise CandidateFailed({'kind': exc.kind, 'message': exc.message, 'where': where,
                               'stderr_tail': engine.stderr_tail()}) from exc

    def _timeout(self, role: str, reference_ms: float | None) -> float:
        if role == 'reference':
            return REFERENCE_TIMEOUT
        return CONTROL_TIMEOUT if reference_ms is None else max(MIN_TIMEOUT, TIMEOUT_FACTOR * reference_ms / 1000.0)

    def _split(self, engine: Engine, role: str, case: workload.Case, prompt, limits=(None, None)):
        """The timed request pair; returns (tokens, prefill_ms, decode_ms)."""
        try:
            first, prefill_ms = engine.generate(prompt, 1, True, self.plan.vocab, self._timeout(role, limits[0]))
            rest, decode_ms = engine.generate([], case.output_tokens - 1, False, self.plan.vocab,
                                              self._timeout(role, limits[1]))
        except EngineFailure as exc:
            self._failed(role, exc, engine, case.id)
        return first + rest, prefill_ms, decode_ms

    def _warm(self, engine: Engine, role: str):
        for index in range(self.plan.warmup):
            try:
                engine.generate(self.plan.warmup_prompt(index), 2, True, self.plan.vocab, self._timeout(role, None))
            except EngineFailure as exc:
                self._failed(role, exc, engine, 'warmup')

    # Steps -----------------------------------------------------------------------------------------------------

    def _control_tokens(self, role: str) -> dict[str, list[int]]:
        engine = self._open(role, 'controls')
        tokens = {}
        with engine:
            for case in self.controls:
                prompt = workload.prompt(case.length, self.plan.vocab)
                try:
                    tokens[case.id], _ = engine.generate(prompt, case.output_tokens, True, self.plan.vocab,
                                                         self._timeout(role, None))
                except EngineFailure as exc:
                    self._failed(role, exc, engine, case.id)
        return tokens

    def _reference_controls(self) -> dict[str, list[int]]:
        cached = self.reference_controls
        if cached and cached.is_file():
            data = json.loads(cached.read_text())
            if all(case.id in data for case in self.controls):
                return data
        tokens = self._control_tokens('reference')
        if cached:
            cached.write_text(json.dumps(tokens))
        return tokens

    def _judge(self, streams: list[tuple]) -> dict:
        """streams: (key, prompt, tokens); returns key -> mismatches ([step, margin, reference token], ...)."""
        if not streams:
            return {}
        engine = self._open('reference', 'judge')
        results = {}
        with engine:
            for key, prompt, tokens in streams:
                try:
                    reply, _ = engine.call('judge', REFERENCE_TIMEOUT, prompt=prompt, tokens=tokens)
                except EngineFailure as exc:
                    self._failed('reference', exc, engine, f'judge {key}')
                results[key] = reply['mismatches']
        return results

    def _gate(self, case: workload.Case, mismatches: list) -> dict:
        """A stream passes with at most the allowed number of mismatching steps, each of whose tokens is among the
        reference's top mismatch_rank choices."""
        limit, ranks = allowed(case.output_tokens, self.plan.gates.case_mismatch_percent), self.plan.gates.mismatch_rank
        worst = max((m[3] for m in mismatches), default=None)
        return {'mismatches': len(mismatches), 'allowed': limit, 'worst_rank': worst,
                'passed': len(mismatches) <= limit and not (ranks and worst and worst > ranks),
                'first_mismatch': mismatches[0] if mismatches else None}

    def _check_controls(self) -> list[dict]:
        if not self.controls:
            # A cheap smoke test so a candidate that cannot load or answer fails before the reference's long arm.
            engine = self._open('candidate', 'smoke')
            with engine:
                self._split(engine, 'candidate', workload.Case('smoke', 16, 2, 1.0), workload.prompt(16, self.plan.vocab))
            return []
        candidate = self._control_tokens('candidate')
        reference = self._reference_controls()
        differing = [(c.id, workload.prompt(c.length, self.plan.vocab), candidate[c.id])
                     for c in self.controls if candidate[c.id] != reference[c.id]]
        judged = self._judge(differing)
        return [{'id': c.id, 'length': c.length, 'output_tokens': c.output_tokens, 'identical': c.id not in judged,
                 'judged': judged.get(c.id, []), **self._gate(c, judged.get(c.id, []))} for c in self.controls]

    def _round(self, r: int, samples: dict) -> None:
        order = ('reference', 'candidate') if r % 2 == 0 else ('candidate', 'reference')
        for role in order:
            engine = self._open(role, f'round-{r}')
            with engine:
                self._warm(engine, role)
                for case in self.timed:
                    prompt = workload.prompt(case.length, self.plan.vocab, self.plan.offset(case, r))
                    reference = samples[(case.id, 'reference')]
                    limits = (reference[-1][1], reference[-1][2]) if reference else (None, None)
                    samples[(case.id, role)].append(self._split(engine, role, case, prompt, limits))
                    if self.stop():
                        raise Stopped()
            self.log(f'round {r + 1}/{self.plan.rounds}: {role} done')

    def _judge_rounds(self, samples: dict, rounds) -> dict:
        differing = []
        for case in self.timed:
            for r in rounds:
                tokens = samples[(case.id, 'candidate')][r][0]
                if tokens != samples[(case.id, 'reference')][r][0]:
                    prompt = workload.prompt(case.length, self.plan.vocab, self.plan.offset(case, r))
                    differing.append(((case.id, r), prompt, tokens))
        return self._judge(differing)

    # Entry point -----------------------------------------------------------------------------------------------

    def run(self) -> dict:
        """The evaluation's result. Raises ReferenceFailed (the reference itself failed) and Stopped."""
        result = {'windows': self.plan.windows, 'seed': self.plan.seed.hex() or None, 'rounds': self.plan.rounds,
                  'started': self.started, 'controls': [], 'cases': [], 'failure': None}
        samples = {(c.id, role): [] for c in self.timed for role in ('reference', 'candidate')}
        try:
            result['controls'] = self._check_controls()
            failed = [c['id'] for c in result['controls'] if not c['passed']]
            if failed:
                return self._finish(result, failure={'kind': 'gates', 'where': ', '.join(failed),
                                                     'message': f'{len(failed)} control case(s) failed the gates'})
            # Round 0 is judged before the other rounds are timed, so a wrong engine fails early.
            self._round(0, samples)
            judged = self._judge_rounds(samples, [0])
            streams = [{'id': c.id, 'round': 0, 'judged': judged.get((c.id, 0), []),
                        **self._gate(c, judged.get((c.id, 0), []))} for c in self.timed]
            failed = [stream for stream in streams if not stream['passed']]
            if failed:
                return self._finish(result, failure={
                    'kind': 'gates', 'where': ', '.join(stream['id'] for stream in failed), 'streams': failed,
                    'message': f'{len(failed)} timed case(s) failed the gates in round 0'})
            for r in range(1, self.plan.rounds):
                self._round(r, samples)
            judged.update(self._judge_rounds(samples, range(1, self.plan.rounds)))
        except CandidateFailed as exc:
            return self._finish(result, failure=exc.failure)
        for case in self.timed:
            reference, candidate = samples[(case.id, 'reference')], samples[(case.id, 'candidate')]
            streams = [self._gate(case, judged.get((case.id, r), [])) for r in range(self.plan.rounds)]
            row = {'id': case.id, 'length': case.length, 'weight': case.weight, 'output_tokens': case.output_tokens,
                   'passed': all(s['passed'] for s in streams), 'mismatches': [s['mismatches'] for s in streams],
                   'allowed': streams[0]['allowed'],
                   'worst_rank': max((s['worst_rank'] for s in streams if s['worst_rank']), default=None),
                   # Every mismatching step per round: [step, margin, the reference's token, rank].
                   'judged': [judged.get((case.id, r), []) for r in range(self.plan.rounds)],
                   'first_mismatch': next((s['first_mismatch'] for s in streams if s['first_mismatch']), None)}
            for index, segment in ((1, 'prefill'), (2, 'decode')):
                row[f'{segment}_ms'] = {'reference': [s[index] for s in reference],
                                        'candidate': [s[index] for s in candidate]}
                row[f'{segment}_speedup'] = statistics.median(a[index] / b[index]
                                                              for a, b in zip(reference, candidate))
            result['cases'].append(row)
        failed = [c['id'] for c in result['cases'] if not c['passed']]
        return self._finish(result, failure={'kind': 'gates', 'where': ', '.join(failed),
                                             'message': f'{len(failed)} timed case(s) failed the gates'}
                            if failed else None)

    def _finish(self, result: dict, failure: dict | None) -> dict:
        streams = [(c['mismatches'], c['output_tokens']) for c in result['controls']] + \
                  [(m, c['output_tokens']) for c in result['cases'] for m in c['mismatches']]
        steps, total = sum(n for _, n in streams), sum(m for m, _ in streams)
        limit = allowed(steps, self.plan.gates.total_mismatch_percent)
        result['gates'] = {'total_steps': steps, 'total_mismatches': total, 'allowed': limit}
        if failure is None and total > limit:
            failure = {'kind': 'gates', 'where': 'total',
                       'message': f'{total} of {steps} judged steps mismatch (at most {limit} allowed)'}
        complete = failure is None or failure['kind'] == 'gates' and bool(result['cases'])
        result.update(failure=failure, passed=failure is None, complete=complete, load_ms=self.load_ms,
                      duration_s=time.time() - self.started)
        rows = [(c['id'], c['length'], c['prefill_speedup'], c['decode_speedup']) for c in result['cases']]
        result['score'] = acceptance.score(rows, self.plan.objective) if failure is None else None
        result['segments'] = acceptance.segment_means(rows, self.plan.objective) if rows else None
        return result


def rows(result: dict) -> tuple:
    """The acceptance rows of an evaluation result."""
    return tuple((c['id'], c['length'], c['prefill_speedup'], c['decode_speedup']) for c in result['cases'])
