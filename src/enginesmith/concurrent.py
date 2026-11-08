"""Paired evaluation of a concurrent task (docs/design.md, "Concurrent evaluation"; protocol 2 in docs/protocol.md).

Per concurrency level c, c closed-loop clients send the level's requests: a client sends its next request as soon as
its last one has received every token. Every request is independent (a fresh prompt, no shared state), its prompt and
output lengths come from the run's request plan, its prompt window from the evaluation's seed. The harness timestamps
each request just before it writes it and every token when its line arrives.

1. Controls: the candidate generates every control case alone; its tokens are judged against the single-request
   reference exactly as in a single-request evaluation.
2. Rounds: per round both engines run every level on the same requests, the reference first in even rounds and the
   candidate first in odd ones. The reference is the reference library's own continuous-batching generator.
3. Judging: per round, a candidate stream identical to the reference's agrees; one that differs is teacher-forced
   through the single-request reference and gated like any stream.
4. Per level and round: throughput = the time from the first request to the last token (the same work for both
   engines); ttft = the 90th percentile over the requests of the time to the first token; latency = the 90th
   percentile over the requests of the end-to-end time divided by the output tokens (normalized latency). All three
   start when the harness writes the request, so an engine gains nothing by holding tokens back. A speedup is the
   median over rounds of reference / candidate.
"""
from __future__ import annotations

import json
import math
import os
import selectors
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from enginesmith import workload
from enginesmith.evaluate import (CONTROL_TIMEOUT, HELLO_TIMEOUT, MIN_TIMEOUT, REFERENCE_TIMEOUT, TIMEOUT_FACTOR,
                                  WARMUP_LENGTH, CandidateFailed, Evaluator, Stopped)
from enginesmith.protocol import MAX_REPLY_BYTES, Engine, EngineFailure
from enginesmith.task import ConcurrentObjective, Gates

PROTOCOL = 2
TTFT_PERCENTILE = 90
LATENCY_PERCENTILE = 90
# A failing level reports at most this many of its failing streams.
FAILING_STREAMS = 10


def percentile(values, p: float) -> float | None:
    """The nearest-rank percentile: the smallest value with at least p% of the values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


@dataclass(frozen=True)
class ConcurrentPlan:
    levels: dict                 # level -> {'requests', 'clients', 'starts_s'} (workload.request_plan)
    controls: tuple              # workload.Case, correctness only
    vocab: int
    rounds: int
    warmup: int
    cooldown_s: float
    gates: Gates
    objective: ConcurrentObjective
    windows: str = 'nonce'       # 'public', 'nonce' or 'held_out'
    seed: bytes = b''            # this evaluation's prompt windows
    structure_seed: bytes = b''  # the request plan (the same for every round of a run)

    @property
    def cases(self) -> tuple:
        return self.controls

    def prompt(self, request: workload.Request, round_index: int) -> list[int]:
        if self.windows == 'public':
            offset = workload.draw_offset(b'public', f'public/{request.id}', request.length, workload.PUBLIC)
        else:
            span = workload.NONCE if self.windows == 'nonce' else workload.HELD_OUT
            offset = workload.draw_offset(self.seed, f'{self.windows}/{round_index}/{request.id}', request.length, span)
        return workload.prompt(request.length, self.vocab, offset)

    def warmup_prompt(self, index: int) -> list[int]:
        longest = max(r.length for spec in self.levels.values() for r in spec['requests'])
        length = min(WARMUP_LENGTH, longest)
        offset = workload.draw_offset(self.seed or b'public', f'warmup/{index}', length, workload.NONCE)
        return workload.prompt(length, self.vocab, offset)


def plan(task, rounds: int, windows: str, seed: bytes, structure_seed: bytes,
         levels: tuple | None = None, requests: int | None = None) -> ConcurrentPlan:
    """The plan of a concurrent task's evaluation; levels and requests narrow it (dev-eval)."""
    c = task.concurrent
    chosen = levels or c.levels
    structure = workload.request_plan(structure_seed, chosen, requests or c.requests_per_level, c.prompt_range,
                                      c.output_range, c.stagger_s)
    return ConcurrentPlan(structure, tuple(x for x in task.cases() if not x.timed), task.vocab_size, rounds,
                          task.evaluation.warmup, task.evaluation.cooldown_s, task.gates, task.objective, windows,
                          seed, structure_seed)


def drive(engine: Engine, clients, starts_s, prompts: dict, vocab: int, timeout_s: float,
          stop=lambda: False) -> dict:
    """Run closed-loop clients against one engine (protocol 2). clients: per client its requests in order; starts_s:
    when each client sends its first request. Returns, per request id, the time it was sent and every token's arrival
    time (seconds from the start) and the tokens."""
    stdin, stdout = engine.process.stdin.fileno(), engine.process.stdout.fileno()
    started = time.perf_counter()
    deadline = started + timeout_s
    queues = [list(c) for c in clients]
    waiting = list(range(len(clients)))  # clients that have not sent their first request
    outgoing = bytearray()
    sent, arrivals, tokens, active = {}, {}, {}, {}
    total, finished = sum(len(q) for q in queues), 0

    def submit(client: int, now: float) -> None:
        request = queues[client].pop(0)
        outgoing.extend(json.dumps({'id': request.id, 'op': 'submit', 'prompt': prompts[request.id],
                                    'max_tokens': request.output_tokens}, separators=(',', ':')).encode() + b'\n')
        sent[request.id], arrivals[request.id], tokens[request.id] = now - started, [], []
        active[request.id] = (request, client)

    def handle(line: bytes, now: float) -> list[int]:
        """The clients whose request this line completed."""
        if not line.strip():
            return []
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raise EngineFailure('protocol', f'output is not JSON: {line[:200]!r}') from None
        if type(event) is not dict:
            raise EngineFailure('protocol', f'output is not a JSON object: {line[:200]!r}')
        if event.get('ok') is False:
            raise EngineFailure('error', f'request {event.get("id")}: ' + str(event.get('error', ''))[:500])
        pairs = event.get('tokens')
        if type(pairs) is not list or set(event) != {'tokens'}:
            raise EngineFailure('protocol', f'expected {{"tokens": [[id, token], ...]}}: {line[:200]!r}')
        done = []
        for pair in pairs:
            if type(pair) is not list or len(pair) != 2 or pair[0] not in active:
                raise EngineFailure('protocol', f'a token for a request that is not in flight: {pair!r}'[:300])
            request_id, token = pair
            if type(token) is not int or not 0 <= token < vocab:
                raise EngineFailure('protocol', f'request {request_id}: token {token!r} is not an id below {vocab}')
            tokens[request_id].append(token)
            arrivals[request_id].append(now - started)
            request, client = active[request_id]
            if len(tokens[request_id]) == request.output_tokens:
                del active[request_id]
                done.append(client)
        return done

    with selectors.DefaultSelector() as selector:
        selector.register(stdout, selectors.EVENT_READ)
        writing, checked = False, started
        while finished < total:
            now = time.perf_counter()
            if now > deadline:
                raise EngineFailure('timeout', f'{total - finished} of {total} requests unfinished after '
                                               f'{timeout_s:.0f} s')
            for client in [c for c in waiting if started + starts_s[c] <= now]:
                waiting.remove(client)
                submit(client, now)
            if outgoing and not writing:
                selector.register(stdin, selectors.EVENT_WRITE)
                writing = True
            elif not outgoing and writing:
                selector.unregister(stdin)
                writing = False
            wait = min(deadline - now, 1.0)
            if waiting:
                wait = min(wait, max(0.0, started + min(starts_s[c] for c in waiting) - now))
            for key, _ in selector.select(wait):
                if key.fd == stdin:
                    try:
                        del outgoing[:os.write(stdin, outgoing[:1 << 20])]
                    except BrokenPipeError:
                        raise EngineFailure('crash', engine._exit_note()) from None
                    continue
                chunk = os.read(stdout, 1 << 20)
                read_at = time.perf_counter()
                if not chunk:
                    raise EngineFailure('crash', engine._exit_note())
                engine._buffer += chunk
                while b'\n' in engine._buffer:
                    line, engine._buffer = engine._buffer.split(b'\n', 1)
                    for client in handle(line, read_at):
                        finished += 1
                        if queues[client]:
                            submit(client, read_at)
                if len(engine._buffer) > MAX_REPLY_BYTES:
                    raise EngineFailure('protocol', 'an output line larger than 64 MiB')
            if now - checked > 1.0:  # the stop request is a file; look at it once a second, not on every token
                checked = now
                if stop():
                    raise Stopped()
    return {'sent': sent, 'arrivals': arrivals, 'tokens': tokens}


def measure(trace: dict, requests) -> dict:
    """A level's times from one drive, in milliseconds: the makespan (first request sent to last token), the
    percentiles over the requests of the time to the first token and of the normalized latency (end-to-end time /
    output tokens), and, report-only, the time per output token after the first as the engine streamed them and the
    99th percentile of the gaps between consecutive tokens."""
    ttft, latency, tpot, gaps = [], [], [], []
    for r in requests:
        sent, times = trace['sent'][r.id], trace['arrivals'][r.id]
        ttft.append(times[0] - sent)
        latency.append((times[-1] - sent) / len(times))
        tpot.append((times[-1] - times[0]) / (len(times) - 1))
        gaps.extend(b - a for a, b in zip(times, times[1:]))
    makespan = max(trace['arrivals'][r.id][-1] for r in requests) - min(trace['sent'][r.id] for r in requests)
    ms = lambda value: round(value * 1000, 4) if value is not None else None
    output = sum(r.output_tokens for r in requests)
    return {'makespan_ms': ms(makespan), 'output_tok_s': round(output / makespan, 2) if makespan > 0 else None,
            'ttft_p50_ms': ms(percentile(ttft, 50)), 'ttft_p90_ms': ms(percentile(ttft, TTFT_PERCENTILE)),
            'latency_p50_ms': ms(percentile(latency, 50)),
            'latency_p90_ms': ms(percentile(latency, LATENCY_PERCENTILE)),
            'tpot_p50_ms': ms(percentile(tpot, 50)), 'tpot_p90_ms': ms(percentile(tpot, 90)),
            'gap_p99_ms': ms(percentile(gaps, 99))}


SEGMENT_KEYS = {'throughput': 'makespan_ms', 'ttft': 'ttft_p90_ms', 'latency': 'latency_p90_ms'}


class ConcurrentEvaluator(Evaluator):
    """The evaluation of a concurrent task; run() returns a result shaped like Evaluator.run()'s, with one case per
    level (mode 'concurrent')."""

    def __init__(self, reference_argv, reference_concurrent_argv, candidate_argv, plan: ConcurrentPlan,
                 directory: Path, **kwargs):
        super().__init__(reference_argv, candidate_argv, plan, directory, **kwargs)
        self.reference_concurrent_argv = list(reference_concurrent_argv)

    def _open(self, role: str, label: str) -> Engine:
        """The candidate always speaks protocol 2; the reference does in the timed rounds (its concurrent mode) and
        answers controls and judge requests as the single-request reference (protocol 1)."""
        if self.stop():
            raise Stopped()
        if self.sessions and self.plan.cooldown_s:
            time.sleep(self.plan.cooldown_s)
        self.sessions += 1
        timed = label.startswith('round-')
        if role == 'reference':
            argv, protocol = (self.reference_concurrent_argv, PROTOCOL) if timed else (self.reference_argv, 1)
        else:
            argv, protocol = self.candidate_argv, PROTOCOL
        engine = Engine(argv, scratch=self.directory / 'scratch' / f'{self.sessions:03d}-{role}-{label}',
                        label=role, registry=self.registry,
                        engine_cores=None if role == 'reference' else self.engine_cores)
        try:
            self.load_ms[role].append(engine.hello(HELLO_TIMEOUT, protocol))
        except EngineFailure as exc:
            engine.close()
            self._failed(role, exc, engine, f'{label}: loading')
        return engine

    def _alone(self, engine: Engine, role: str, request_id: str, prompt, output_tokens: int, where: str,
               timeout: float) -> list[int]:
        """One request on its own; its tokens."""
        request = workload.Request(request_id, 1, 0, len(prompt), output_tokens)
        try:
            trace = drive(engine, [[request]], [0.0], {request_id: prompt}, self.plan.vocab, timeout, self.stop)
        except EngineFailure as exc:
            self._failed(role, exc, engine, where)
        return trace['tokens'][request_id]

    def _control_tokens(self, role: str) -> dict[str, list[int]]:
        if role == 'reference':
            return super()._control_tokens(role)
        engine = self._open(role, 'controls')
        with engine:
            return {case.id: self._alone(engine, role, case.id, workload.prompt(case.length, self.plan.vocab),
                                         case.output_tokens, case.id, CONTROL_TIMEOUT) for case in self.controls}

    def _check_controls(self) -> list[dict]:
        if self.controls:
            return super()._check_controls()
        # A cheap smoke test so a candidate that cannot load or answer fails before the reference's long arm.
        engine = self._open('candidate', 'smoke')
        with engine:
            self._alone(engine, 'candidate', 'smoke', workload.prompt(16, self.plan.vocab), 2, 'smoke',
                        CONTROL_TIMEOUT)
        return []

    def _warm(self, engine: Engine, role: str):
        for index in range(self.plan.warmup):
            self._alone(engine, role, f'warmup-{index}', self.plan.warmup_prompt(index), 2, 'warmup',
                        REFERENCE_TIMEOUT if role == 'reference' else CONTROL_TIMEOUT)

    def _round(self, r: int, samples: dict) -> None:
        order = ('reference', 'candidate') if r % 2 == 0 else ('candidate', 'reference')
        traces = self.directory / 'traces'
        traces.mkdir(exist_ok=True)
        for role in order:
            engine = self._open(role, f'round-{r}')
            with engine:
                self._warm(engine, role)
                for level, spec in self.plan.levels.items():
                    prompts = {q.id: self.plan.prompt(q, r) for q in spec['requests']}
                    reference = samples[(level, 'reference')]
                    timeout = REFERENCE_TIMEOUT if role == 'reference' or not reference else max(
                        MIN_TIMEOUT, TIMEOUT_FACTOR * reference[-1]['metrics']['makespan_ms'] / 1000.0)
                    try:
                        trace = drive(engine, spec['clients'], spec['starts_s'], prompts, self.plan.vocab, timeout,
                                      self.stop)
                    except EngineFailure as exc:
                        self._failed(role, exc, engine, f'level {level}, round {r}')
                    # The raw times stay with the evaluation, for audits and analysis.
                    (traces / f'round-{r}-{role}-c{level}.json').write_text(json.dumps(trace))
                    samples[(level, role)].append({'tokens': trace['tokens'],
                                                   'metrics': measure(trace, spec['requests'])})
            self.log(f'round {r + 1}/{self.plan.rounds}: {role} done')

    def _judge_rounds(self, samples: dict, rounds) -> dict:
        differing = []
        for level, spec in self.plan.levels.items():
            for r in rounds:
                candidate, reference = samples[(level, 'candidate')][r]['tokens'], samples[(level, 'reference')][r]['tokens']
                differing += [((level, r, q.id), self.plan.prompt(q, r), candidate[q.id])
                              for q in spec['requests'] if candidate[q.id] != reference[q.id]]
        return self._judge(differing)

    def _streams(self, level: int, rounds, judged: dict) -> list[dict]:
        return [{'id': q.id, 'round': r, **self._gate(q, judged.get((level, r, q.id), []))}
                for r in rounds for q in self.plan.levels[level]['requests']]

    def run(self) -> dict:
        """The evaluation's result. Raises ReferenceFailed (the reference itself failed) and Stopped."""
        p = self.plan
        result = {'mode': 'concurrent', 'windows': p.windows, 'seed': p.seed.hex() or None,
                  'structure_seed': p.structure_seed.hex() or None, 'rounds': p.rounds, 'started': self.started,
                  'controls': [], 'cases': [], 'failure': None}
        samples = {(level, role): [] for level in p.levels for role in ('reference', 'candidate')}
        try:
            result['controls'] = self._check_controls()
            failed = [c['id'] for c in result['controls'] if not c['passed']]
            if failed:
                return self._finish(result, failure={'kind': 'gates', 'where': ', '.join(failed),
                                                     'message': f'{len(failed)} control case(s) failed the gates'})
            # Round 0 is judged before the other rounds are timed, so a wrong engine fails early.
            self._round(0, samples)
            judged = self._judge_rounds(samples, [0])
            failed = [s for level in p.levels for s in self._streams(level, [0], judged) if not s['passed']]
            if failed:
                return self._finish(result, failure={
                    'kind': 'gates', 'where': ', '.join(s['id'] for s in failed[:FAILING_STREAMS]),
                    'streams': failed[:FAILING_STREAMS],
                    'message': f'{len(failed)} request stream(s) failed the gates in round 0'})
            for r in range(1, p.rounds):
                self._round(r, samples)
            judged.update(self._judge_rounds(samples, range(1, p.rounds)))
        except CandidateFailed as exc:
            return self._finish(result, failure=exc.failure)
        for level, spec in p.levels.items():
            streams = self._streams(level, range(p.rounds), judged)
            reference = [s['metrics'] for s in samples[(level, 'reference')]]
            candidate = [s['metrics'] for s in samples[(level, 'candidate')]]
            failing = [s for s in streams if not s['passed']]
            row = {'id': f'c{level}', 'level': level, 'weight': p.objective.case_weight(level),
                   'requests': len(spec['requests']), 'output_tokens': sum(q.output_tokens for q in spec['requests']),
                   'passed': not failing,
                   'mismatches': [sum(s['mismatches'] for s in streams if s['round'] == r) for r in range(p.rounds)],
                   'judged_streams': [sum(1 for q in spec['requests'] if (level, r, q.id) in judged)
                                      for r in range(p.rounds)],
                   'worst_rank': max((s['worst_rank'] for s in streams if s['worst_rank']), default=None),
                   'failing_streams': failing[:FAILING_STREAMS],
                   'first_mismatch': next((s['first_mismatch'] for s in streams if s['first_mismatch']), None),
                   'reference': reference, 'candidate': candidate}
            for segment, key in SEGMENT_KEYS.items():
                pairs = [(a[key], b[key]) for a, b in zip(reference, candidate)]
                row[f'{segment}_speedup'] = (statistics.median(a / b for a, b in pairs)
                                             if pairs and all(a and b and a > 0 and b > 0 for a, b in pairs) else None)
            result['cases'].append(row)
        failed = [c['id'] for c in result['cases'] if not c['passed']]
        return self._finish(result, failure={'kind': 'gates', 'where': ', '.join(failed),
                                             'message': f'request streams of {len(failed)} level(s) failed the gates'}
                            if failed else None)

    def _finish(self, result: dict, failure: dict | None) -> dict:
        from enginesmith import acceptance
        steps = sum(c['output_tokens'] for c in result['controls']) + \
            sum(c['output_tokens'] for c in result['cases']) * self.plan.rounds
        total = sum(c['mismatches'] for c in result['controls']) + sum(sum(c['mismatches']) for c in result['cases'])
        limit = steps * self.plan.gates.total_mismatch_percent // 100
        result['gates'] = {'total_steps': steps, 'total_mismatches': total, 'allowed': limit}
        if failure is None and total > limit:
            failure = {'kind': 'gates', 'where': 'total',
                       'message': f'{total} of {steps} judged steps mismatch (at most {limit} allowed)'}
        complete = failure is None or failure['kind'] == 'gates' and bool(result['cases'])
        result.update(failure=failure, passed=failure is None, complete=complete, load_ms=self.load_ms,
                      duration_s=time.time() - self.started)
        table = rows(result)
        result['score'] = acceptance.score(table, self.plan.objective) if failure is None else None
        result['segments'] = acceptance.segment_means(table, self.plan.objective) if table else None
        return result


def rows(result: dict) -> tuple:
    """The acceptance rows of a concurrent evaluation result: (case id, level, throughput, ttft, latency speedups)."""
    return tuple((c['id'], c['level'], c['throughput_speedup'], c['ttft_speedup'], c['latency_speedup'])
                 for c in result['cases'])
