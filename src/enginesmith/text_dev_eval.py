"""Opt-in natural-text diagnostics, isolated from formal scoring and synthetic development results."""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

from enginesmith import engine_repo, workload
from enginesmith.dev_eval import gpu_lock, session_info
from enginesmith.evaluate import Evaluator, Plan, ReferenceFailed, Stopped, allowed
from enginesmith.platforms import current
from enginesmith.processes import Registry
from enginesmith.task import Task
from enginesmith.text_dataset import DatasetError, Sample, load_samples


class TextEvaluator(Evaluator):
    """Reuse controls, timing, lifecycle and gates; route timed prompts and replay through the same samples.

    Only the opt-in development command constructs this evaluator. The formal Evaluator and its Plan are unmodified.
    """
    def __init__(self, *args, text_samples: list[Sample], provenance: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_samples = {sample.id: sample for sample in text_samples}
        self.provenance = provenance
        if set(self.text_samples) != {case.id for case in self.timed}:
            raise DatasetError('timed cases and selected text samples must match exactly')

    def _round(self, r: int, samples: dict) -> None:
        for role in (('reference', 'candidate') if r % 2 == 0 else ('candidate', 'reference')):
            engine = self._open(role, f'round-{r}')
            with engine:
                self._warm(engine, role)
                for case in self.timed:
                    prompt = list(self.text_samples[case.id].tokens)
                    reference = samples[(case.id, 'reference')]
                    limits = (reference[-1][1], reference[-1][2]) if reference else (None, None)
                    samples[(case.id, role)].append(self._split(engine, role, case, prompt, limits))
                    if self.stop():
                        raise Stopped()
            self.log(f'text diagnostic round {r + 1}/{self.plan.rounds}: {role} done')

    def _judge_rounds(self, samples: dict, rounds) -> dict:
        differing = []
        for case in self.timed:
            for r in rounds:
                tokens = samples[(case.id, 'candidate')][r][0]
                if tokens != samples[(case.id, 'reference')][r][0]:
                    differing.append(((case.id, r), list(self.text_samples[case.id].tokens), tokens))
        return self._judge(differing)

    def _finish(self, result: dict, failure: dict | None) -> dict:
        streams = [(c['mismatches'], c['output_tokens']) for c in result['controls']] + \
                  [(m, c['output_tokens']) for c in result['cases'] for m in c['mismatches']]
        steps, total = sum(n for _, n in streams), sum(m for m, _ in streams)
        limit = allowed(steps, self.plan.gates.total_mismatch_percent)
        if failure is None and total > limit:
            failure = {'kind': 'gates', 'where': 'total',
                       'message': f'{total} of {steps} judged steps mismatch (at most {limit} allowed)'}
        result.update(failure=failure, passed=failure is None,
                      complete=failure is None or failure['kind'] == 'gates' and bool(result['cases']),
                      gates={'total_steps': steps, 'total_mismatches': total, 'allowed': limit},
                      load_ms=self.load_ms, duration_s=time.time() - self.started)
        # Even a passing diagnostic is never a formal score, and lives outside the agent's synthetic dev-eval log.
        result.update(workload='natural-text-development', diagnostic_only=True, score=None, segments=None,
                      score_reason='Natural-text diagnostics are excluded from formal scoring and acceptance.',
                      provenance=self.provenance, lengths=[], speculative_telemetry=None,
                      speculative_telemetry_reason='The engine protocol does not expose draft/verification counters.')
        for length in sorted({c['length'] for c in result['cases']}):
            rows = [c for c in result['cases'] if c['length'] == length]
            result['lengths'].append({'length': length, 'sample_count': len(rows), 'passed': all(c['passed'] for c in rows),
                                     **{f'{s}_speedup_median': statistics.median(c[f'{s}_speedup'] for c in rows)
                                        for s in ('prefill', 'decode')}})
        return result


def run(dataset: Path, lengths: str | None, rounds: int, build: bool, cooldown: float | None,
        samples_per_length: int = 3, seed: int = 0) -> int:
    info = session_info()
    task = Task.from_json(info['task'])
    directory = None
    try:
        if type(rounds) is not int or rounds < 1:
            raise DatasetError('--rounds must be positive')
        if cooldown is not None and (not math.isfinite(cooldown) or cooldown < 0):
            raise DatasetError('--cooldown must be finite and nonnegative')
        if lengths in (None, ''):
            chosen = [n for n in task.lengths if n <= 32768] or list(task.lengths[:1])
        elif lengths == 'all':
            chosen = list(task.lengths)
        else:
            try:
                chosen = [int(n) for n in lengths.split(',')]
            except ValueError as exc:
                raise DatasetError('--lengths must be comma-separated task lengths or all') from exc
        if not chosen or not set(chosen) <= set(task.lengths):
            raise DatasetError('--lengths must be a subset of the task lengths')
        samples, provenance = load_samples(dataset, task.model, task.vocab_size, chosen, samples_per_length, seed)
        # A natural timed case no longer covers the synthetic boundary at the same length.
        controls = tuple(workload.cases((), task.output_tokens, controls=task.controls))
        cases = controls + tuple(workload.Case(s.id, s.length, task.output_tokens, task.objective.case_weight(s.length))
                                 for s in samples)
        plan = Plan(cases, task.vocab_size, rounds, task.evaluation.warmup,
                    task.evaluation.cooldown_s if cooldown is None else cooldown, task.gates, task.objective, 'public')
        outputs = Path(info['brief']) / 'text-dev-eval'
        outputs.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='diagnostic-', dir=outputs))
        (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        engine = Path(info['engine'])
        executable = engine / task.executable
        with gpu_lock():
            if build:
                result = engine_repo._run_logged(list(task.build), engine, directory / 'build.log',
                                                 engine_repo.BUILD_TIMEOUT, dict(os.environ))
                if result['exit_code'] != 0:
                    raise DatasetError('build failed; see the diagnostic build.log')
            if not executable.is_file():
                raise DatasetError('candidate executable is missing')
            if task.native and (reason := current().native_problem(executable)):
                raise DatasetError(f'candidate executable is not permitted: {reason}')
            evaluator = TextEvaluator(info['reference'], [str(executable), '--model', str(task.model)], plan,
                                      directory, text_samples=samples, provenance=provenance,
                                      registry=Registry(info['registry']), log=lambda m: print(m, flush=True),
                                      reference_controls=directory / 'reference-controls.json',
                                      engine_cores=task.reference_tp)
            result = evaluator.run()
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'natural-text diagnostic: {"PASSED" if result["passed"] else "FAILED"}; '
              f'{samples_per_length} document(s)/length, {rounds} paired round(s); no formal score')
        for row in result['lengths']:
            print(f'{row["length"]}: prefill {row["prefill_speedup_median"]:.4f}x, '
                  f'decode {row["decode_speedup_median"]:.4f}x (median over {row["sample_count"]} samples)')
        print(f'full diagnostic result: {directory / "result.json"}')
        return 0 if result['passed'] else 1
    except (DatasetError, OSError, ReferenceFailed) as exc:
        kind = 'reference_failed' if isinstance(exc, ReferenceFailed) else 'input_or_build'
        if directory is not None:
            (directory / 'error.json').write_text(json.dumps({'kind': kind, 'message': str(exc),
                                                             'diagnostic_only': True, 'at': time.time()}, indent=2) + '\n')
        print(f'natural-text diagnostic {kind}: {exc}')
        return 2
