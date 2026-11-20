"""`enginesmith dev-eval`: the coding agent's own evaluation, inside a session.

It builds the session's engine directory and runs the formal evaluation's procedure on the public prompts (offset 0
of the Lily sequence). Its results are feedback for the agent only; the harness evaluates every candidate itself.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from enginesmith import engine_repo
from enginesmith.brief import case_table, failure_text
from enginesmith.evaluate import Evaluator, Plan, ReferenceFailed
from enginesmith.platforms import current
from enginesmith.processes import Registry
from enginesmith.task import Task

LOCK = '/tmp/engine-gpu.lock'


@contextmanager
def gpu_lock(path: str | None = None):
    """The GPU lock the skill's scripts share (GPU_LOCK_FILE, else LOCK: runs that share a machine on separate
    accelerator cores give each its own); skipped under gpu_lock.py, which already holds it."""
    path = path or os.environ.get('GPU_LOCK_FILE') or LOCK
    if os.environ.get('GPU_LOCK_HELD'):
        yield
        return
    with open(path, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'waiting for the GPU lock {path} ...', flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def session_info() -> dict:
    path = os.environ.get('ENGINESMITH_SESSION')
    if not path:
        raise SystemExit('dev-eval runs inside an EngineSmith session (ENGINESMITH_SESSION is not set)')
    return json.loads(Path(path).read_text())


def run(lengths: str | None, rounds: int, build: bool, cooldown: float | None, levels: str | None = None,
        requests: int | None = None) -> int:
    info = session_info()
    task = Task.from_json(info['task'])
    engine = Path(info['engine'])
    outputs = Path(info['brief']) / 'dev-eval'
    number = len([p for p in outputs.iterdir() if p.is_dir()]) + 1
    directory = outputs / f'{number:03d}'
    directory.mkdir(parents=True)
    if build:
        log = directory / 'build.log'
        print(f'building: {" ".join(task.build)}', flush=True)
        result = engine_repo._run_logged(list(task.build), engine, log, engine_repo.BUILD_TIMEOUT, dict(os.environ))
        if result['exit_code'] != 0:
            print(f'the build failed ({result}); log tail:\n{engine_repo.tail(log)}')
            return 1
    executable = engine / task.executable
    if not executable.is_file():
        print(f'no executable at engine/{task.executable}')
        return 1
    if task.native and (reason := current().native_problem(executable)):
        print(f'engine/{task.executable} is {reason}; the harness would reject it')
        return 1
    if task.mode == 'concurrent':
        return _concurrent(info, task, executable, directory, outputs, number, rounds, cooldown, levels, requests)
    if lengths in (None, ''):
        chosen = [n for n in task.lengths if n <= 32768] or task.lengths[:1]
    elif lengths == 'all':
        chosen = list(task.lengths)
    else:
        chosen = [int(n) for n in lengths.split(',')]
    cases = tuple(c for c in task.cases() if not c.timed or c.length in chosen)
    if not any(c.timed for c in cases):
        print(f'--lengths must name some of the task lengths: {", ".join(map(str, task.lengths))}')
        return 1
    plan = Plan(cases, task.vocab_size, rounds, task.evaluation.warmup,
                task.evaluation.cooldown_s if cooldown is None else cooldown, task.gates, task.objective, 'public')
    controls = Path(info['brief']) / 'reference-controls.json'
    evaluator = Evaluator(info['reference'], [str(executable), '--model', str(task.model)], plan,
                          directory, registry=Registry(info['registry']), log=lambda m: print(m, flush=True),
                          reference_controls=controls, engine_cores=task.reference_tp)
    started = time.time()
    with gpu_lock():
        try:
            result = evaluator.run()
        except ReferenceFailed as exc:
            print(f'the reference failed (not your engine): {exc}')
            return 2
    (outputs / f'{number:03d}.json').write_text(json.dumps(result, indent=2))
    print(f'\ndev-eval {number:03d}: {"PASSED" if result["passed"] else "FAILED"} in {time.time() - started:.0f} s '
          f'(public prompts, {rounds} round(s), lengths {", ".join(map(str, chosen))})')
    if result['failure']:
        print(failure_text({'evaluation': result}))
    if result['cases']:
        print(case_table(result))
    if result['score']:
        s = result['segments']
        print(f'score {result["score"]:.4f} (prefill {s["prefill"]:.4f}, decode {s["decode"]:.4f}) on these lengths')
    print(f'full result: {outputs / f"{number:03d}.json"}')
    return 0 if result['passed'] else 1


def _concurrent(info, task, executable, directory, outputs, number, rounds, cooldown, levels, requests) -> int:
    """dev-eval of a concurrent task: the formal procedure on a public request plan and public prompt windows."""
    from dataclasses import replace
    from enginesmith import concurrent
    from enginesmith.brief import reference_concurrent_argv
    chosen = tuple(int(n) for n in levels.split(',')) if levels else task.concurrent.levels
    if not set(chosen) <= set(task.concurrent.levels):
        print(f'--levels must name some of the task levels: {", ".join(map(str, task.concurrent.levels))}')
        return 1
    count = requests or task.concurrent.requests_per_level
    if count < max(chosen):
        print(f'--requests must be at least the highest level ({max(chosen)})')
        return 1
    plan = concurrent.plan(task, rounds, 'public', b'', b'public', chosen, count)
    if cooldown is not None:
        plan = replace(plan, cooldown_s=cooldown)
    evaluator = concurrent.ConcurrentEvaluator(
        info['reference'], reference_concurrent_argv(task), [str(executable), '--model', str(task.model)], plan,
        directory, registry=Registry(info['registry']), log=lambda m: print(m, flush=True),
        reference_controls=Path(info['brief']) / 'reference-controls.json', engine_cores=task.reference_tp)
    started = time.time()
    with gpu_lock():
        try:
            result = evaluator.run()
        except ReferenceFailed as exc:
            print(f'the reference failed (not your engine): {exc}')
            return 2
    (outputs / f'{number:03d}.json').write_text(json.dumps(result, indent=2))
    print(f'\ndev-eval {number:03d}: {"PASSED" if result["passed"] else "FAILED"} in {time.time() - started:.0f} s '
          f'(public request plan and prompts, {rounds} round(s), levels {", ".join(map(str, chosen))}, {count} '
          'requests per level)')
    if result['failure']:
        print(failure_text({'evaluation': result}))
    if result['cases']:
        print(case_table(result))
    if result['score']:
        means = ', '.join(f'{k} {v:.4f}' for k, v in result['segments'].items() if v is not None)
        print(f'score {result["score"]:.4f} ({means}) on these levels')
    print(f'full result: {outputs / f"{number:03d}.json"}')
    return 0 if result['passed'] else 1
