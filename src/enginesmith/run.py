"""A run: coding sessions, builds, evaluations and decisions, then the final evaluation (docs/design.md).

Every round is one candidate: a coding session (none for the seed round), the snapshot of its engine source, a
clean build, a formal evaluation and the acceptance decision. A round records its progress in round.json after
every step, so `enginesmith resume` continues an interrupted run where it stopped: an interrupted session is not
restarted (its engine directory is taken as it is), an interrupted evaluation is run again.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from enginesmith import acceptance, agents, brief, engine_repo, report
from enginesmith.evaluate import EVALUATION_VERSION, Evaluator, Plan, ReferenceFailed, Stopped, rows
from enginesmith.platforms import current
from enginesmith.processes import Registry, Watchdog, kill_group
from enginesmith.task import REPO, Task

PACKAGE = Path(__file__).resolve().parent
DEFAULT_EVALUATION_S = 3600.0
MARGIN_S = 600.0
QUICK_S = 30.0


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def harness_digest() -> str:
    """sha256 over the harness's own source files; the run refuses to evaluate when it changes."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob('*.py')):
        digest.update(path.relative_to(PACKAGE).as_posix().encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def tree_digest(path: Path | None) -> str | None:
    """sha256 over a file, or over a directory's files with their relative paths; None without a path."""
    if path is None:
        return None
    path = Path(path)
    root = path if path.is_dir() else path.parent
    files = sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts) if path.is_dir() \
        else [path]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode() + b'\0' + file.read_bytes())
    return digest.hexdigest()


class RunError(RuntimeError):
    pass


class Run:
    def __init__(self, directory: Path | str):
        self.dir = Path(directory).resolve()
        if not (self.dir / 'task.json').is_file():
            raise RunError(f'{self.dir} is not a run directory')
        self.task = Task.from_json(read_json(self.dir / 'task.json'))
        self.state = read_json(self.dir / 'state.json')
        self.repo = self.dir / 'engine.git'
        self.registry = Registry(self.dir / 'engines')
        self.session_registry = None

    # Creation ----------------------------------------------------------------------------------------------------

    @classmethod
    def create(cls, task: Task, root: Path | None = None) -> Run:
        root = Path(root) if root else task.path.parent / 'runs'
        root.mkdir(parents=True, exist_ok=True)
        stem = f'{task.name}-{time.strftime("%Y%m%d-%H%M%S")}'
        directory, number = root / stem, 1
        while True:
            try:
                directory.mkdir()
                break
            except FileExistsError:
                number += 1
                directory = root / f'{stem}-{number}'
        write_json(directory / 'task.json', task.to_json())
        shutil.copy2(task.path, directory / 'task.toml')
        (directory / 'cache').mkdir()
        initial = engine_repo.init(directory / 'engine.git', task.seed)
        estimate = (read_json(task.path.parent / 'prepare.json') or {}).get('estimates', {}).get('evaluation_s')
        now = time.time()
        write_json(directory / 'state.json', {
            'phase': 'search', 'created': now, 'deadline': now + task.budget.wall_time_s, 'harness': harness_digest(),
            'evaluation_version': EVALUATION_VERSION, 'task_digest': task.digest(),
            'agent_version': agents.version(task.agent),
            'inputs': {'skill': tree_digest(task.skill), 'model_card': tree_digest(task.model_card)}, 'initial_commit': initial, 'seed_pending': task.evaluate_seed,
            'sessions': 0, 'rounds': [], 'current': None, 'incumbent': None, 'latest': None,
            'evaluation_s': [estimate] if estimate else [], 'search_end': None, 'final': None, 'error': None})
        return cls(directory)

    # Bookkeeping -------------------------------------------------------------------------------------------------

    def save(self) -> None:
        write_json(self.dir / 'state.json', self.state)

    def event(self, message) -> None:
        entry = {'time': time.time(), **(message if isinstance(message, dict) else {'message': str(message)})}
        with open(self.dir / 'events.jsonl', 'a') as log:
            log.write(json.dumps(entry) + '\n')
        print(time.strftime('%H:%M:%S'), entry.get('message') or json.dumps(entry), flush=True)

    def round_dir(self, label: str) -> Path:
        return self.dir / 'rounds' / label

    def record(self, label: str) -> dict:
        return read_json(self.round_dir(label) / 'round.json')

    def save_record(self, record: dict) -> None:
        write_json(self.round_dir(record['label']) / 'round.json', record)

    def stop_mode(self) -> str | None:
        path = self.dir / 'STOP'
        return (path.read_text().strip() or 'finish') if path.exists() else None

    def evaluation(self, label: str) -> dict | None:
        return read_json(self.round_dir(label) / 'evaluation' / 'result.json')

    # Driver ------------------------------------------------------------------------------------------------------

    def drive(self) -> dict:
        lock = open(self.dir / 'driver.lock', 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError('another process is driving this run') from None
        lock.write(str(os.getpid()))
        lock.flush()
        current().keep_awake()
        registries = [self.registry]
        try:
            with Watchdog(registries, on_kill=self.event):
                self.session_registries = registries
                while self._step():
                    if self.stop_mode() == 'abort':
                        raise Stopped()
        except Stopped:
            self.event('stopped (abort); `enginesmith resume` continues the run')
        finally:
            self.registry.kill_all()
            report.write(self)
        return self.state

    def _step(self) -> bool:
        phase = self.state['phase']
        if phase == 'search':
            if self.state['current']:
                self._continue(self.record(self.state['current']))
            elif reason := self._search_over():
                self.state.update(phase='final', search_end=reason)
                self.save()
                self.event(f'search ended: {reason}')
            else:
                self._start_round()
            return True
        if phase == 'final':
            self._final()
            run_quality = self.task.quality and self.state['incumbent'] and not self.state.get('error')
            self.state['phase'] = 'quality' if run_quality else 'done'
            self.save()
            return True
        if phase == 'quality':
            self._quality()
            self.state['phase'] = 'done'
            self.save()
            return True
        return False

    def _estimates(self) -> tuple[float, float]:
        durations = self.state['evaluation_s']
        evaluation = max(durations) if durations else DEFAULT_EVALUATION_S
        e = self.task.evaluation
        return evaluation, evaluation * e.final_rounds / e.rounds

    def _search_over(self) -> str | None:
        if self.stop_mode():
            return 'stopped'
        if self.state.get('agent_failed'):
            return 'agent_failed'
        if self.state['seed_pending']:
            return None
        if self.state['sessions'] >= self.task.budget.sessions:
            return 'sessions_used'
        evaluation, final = self._estimates()
        quality = self.task.quality.time_limit_s if self.task.quality else 0.0
        needed = self.task.budget.session_timeout_s + evaluation + final + quality + MARGIN_S
        if self.state['deadline'] - time.time() < needed:
            return 'time'
        return None

    def _start_round(self) -> None:
        s = self.state
        if s['seed_pending']:
            label, kind, base, base_round = '000', 'seed', s['initial_commit'], None
        else:
            label = f'{s["sessions"] + 1:03d}'
            if s['incumbent']:
                kind, base_round = 'optimize', s['incumbent']
            elif s['latest']:
                kind, base_round = 'repair', s['latest']
            else:
                kind, base_round = 'generate', None
            base = self.record(base_round)['snapshot'] if base_round else s['initial_commit']
            s['sessions'] += 1
        self.round_dir(label).mkdir(parents=True)
        record = {'label': label, 'kind': kind, 'base': base, 'base_round': base_round, 'started': time.time(),
                  'step': 'snapshot' if kind == 'seed' else 'session'}
        self.save_record(record)
        s.update(current=label, seed_pending=False if kind == 'seed' else s['seed_pending'])
        self.save()
        self.event(f'round {label}: {kind}' + (f' from round {base_round}' if base_round else ''))

    def _continue(self, record: dict) -> None:
        step = record['step']
        if step == 'session':
            self._session(record)
            record['step'] = 'snapshot'
        elif step == 'snapshot':
            self._snapshot(record)
        elif step == 'build':
            self._build(record)
        elif step == 'evaluate':
            self._evaluate(record)
        elif step == 'decide':
            self._decide(record)
            return
        self.save_record(record)

    # Steps -------------------------------------------------------------------------------------------------------

    def _session(self, record: dict) -> None:
        label, work = record['label'], self.dir / 'work' / record['label']
        if record.get('session_started'):
            # The driver stopped during this session: do not start it again; take the engine directory as it is.
            if record.get('agent_pgid'):
                kill_group(record['agent_pgid'])
            Registry(work / 'engines').kill_all()
            record['session'] = {'interrupted': True}
            self.event(f'round {label}: the session was interrupted; its engine directory is taken as it is')
            return
        if work.exists():
            shutil.rmtree(work)
        engine_repo.checkout(self.repo, record['base'], work / 'engine')
        cache = self.dir / 'cache' / 'reference-controls.json'
        info = {'run': str(self.dir), 'round': label, 'kind': record['kind'], 'task': self.task.to_json(),
                'engine': str(work / 'engine'), 'brief': str(work / 'brief'),
                'reference': brief.reference_argv(self.task), 'registry': str(work / 'engines'),
                'reference_controls': str(cache)}
        session_file = brief.write(work, self.task, record['kind'], self._start_text(record), self._facts(record),
                                   info)
        if cache.is_file():
            shutil.copy2(cache, work / 'brief' / 'reference-controls.json')
        registry = Registry(work / 'engines')
        self.session_registries.append(registry)
        command = agents.argv(self.task.agent, self.task.agent_command, self.task.agent_model, work)
        record.update(session_started=time.time(), agent=command[0])
        self.save_record(record)

        def started(pid):
            record['agent_pgid'] = pid
            self.save_record(record)
        try:
            record['session'] = agents.run_session(
                command, cwd=work, log=self.round_dir(label) / 'session.log',
                timeout=self.task.budget.session_timeout_s, registry=registry,
                stop_requested=lambda: self.stop_mode() is not None, env=agents.environment(session_file),
                on_start=started)
        finally:
            self.session_registries.remove(registry)
        session = record['session']
        session.update(agents.usage(self.round_dir(label) / 'session.log'))
        self.event(f'round {label}: session ended after {session["duration_s"] / 60:.0f} min'
                   + (' (timeout)' if session['timed_out'] else '') + (' (stopped)' if session['stopped'] else '')
                   + (f' ({session["error"]})' if session.get('error') else ''))
        # An agent that cannot start, or fails within seconds twice in a row (usually: not logged in), would burn
        # every session; stop the run instead.
        quick = session.get('error') or (session['exit_code'] not in (0, None) and session['duration_s'] < QUICK_S
                                         and not session['stopped'])
        self.state['quick_failures'] = self.state.get('quick_failures', 0) + 1 if quick else 0
        if session.get('error') or self.state['quick_failures'] >= 2:
            reason = session.get('error') or 'it exited within seconds twice'
            self.state['agent_failed'] = (f'the coding agent failed to run ({reason}); see '
                                          f'{self.round_dir(label) / "session.log"}')
            self.event(self.state['agent_failed'])
        self.save()

    def _start_text(self, record: dict) -> str:
        if record['kind'] == 'generate':
            return 'the seed engine' if self.task.seed else 'an empty project (only a .gitignore)'
        return f'round {record["base_round"]}\'s engine'

    def _snapshot(self, record: dict) -> None:
        label, work = record['label'], self.dir / 'work' / record['label']
        if record['kind'] == 'seed':
            record['snapshot'] = record['base']
        else:
            try:
                record['snapshot'], large = engine_repo.snapshot(self.repo, work / 'engine', label)
                record['changed_files'] = engine_repo.changed_files(self.repo, record['base'], record['snapshot'])
                record['code'] = engine_repo.code_stats(self.repo, record['base'], record['snapshot'])
                if large:
                    record['files_left_out'] = large
                    self.event(f'round {label}: files over 50 MiB left out of the snapshot: {", ".join(large)}')
            except engine_repo.SnapshotError as exc:
                record.update(snapshot=None, error=f'no snapshot: {exc}')
            if (work / 'brief').exists():
                shutil.move(work / 'brief', self.round_dir(label) / 'brief')
            checks = sorted((self.round_dir(label) / 'brief' / 'dev-eval').glob('*.json'))
            record['dev_evals'] = {'runs': len(checks),
                                   'passed': sum(1 for path in checks if (read_json(path) or {}).get('passed'))}
            engine_repo.remove_worktree(self.repo, work / 'engine')
            shutil.rmtree(work, ignore_errors=True)
        unchanged = (record['snapshot'] and record['kind'] != 'seed'
                     and engine_repo.git('rev-parse', f'{record["snapshot"]}^{{tree}}', cwd=self.repo)
                     == engine_repo.git('rev-parse', f'{record["base"]}^{{tree}}', cwd=self.repo))
        if unchanged:
            record['unchanged'] = True
        record['step'] = 'build' if record['snapshot'] and not unchanged else 'decide'

    def _build(self, record: dict) -> None:
        directory = self.round_dir(record['label'])
        log = directory / 'build.log'
        self.event(f'round {record["label"]}: building {record["snapshot"][:12]}')
        result = engine_repo.build(self.repo, self.dir / 'build', record['snapshot'], self.task.build,
                                   self.task.executable, self.task.native, directory / 'engine', log)
        if not result['passed']:
            result['log_tail'] = engine_repo.tail(log)
            self.event(f'round {record["label"]}: {result["error"]}')
        record['build'] = result
        record['step'] = 'evaluate' if result['passed'] else 'decide'

    def plan(self, rounds: int, windows: str, seed: bytes) -> Plan:
        t = self.task
        return Plan(tuple(t.cases()), t.vocab_size, rounds, t.evaluation.warmup, t.evaluation.cooldown_s, t.gates,
                    t.objective, windows, seed)

    def _evaluator(self, engine: Path, plan: Plan, directory: Path, stop) -> Evaluator:
        if harness_digest() != self.state['harness']:
            raise RunError('the harness source changed during the run; evaluations stopped')
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        return Evaluator(brief.reference_argv(self.task), [str(engine), '--model', str(self.task.model)], plan,
                         directory, registry=self.registry, log=self.event, stop=stop,
                         reference_controls=self.dir / 'cache' / 'reference-controls.json')

    def _evaluate(self, record: dict) -> None:
        label = record['label']
        directory = self.round_dir(label) / 'evaluation'
        self.event(f'round {label}: evaluating')
        evaluator = self._evaluator(self.round_dir(label) / 'engine', self.plan(self.task.evaluation.rounds, 'nonce',
                                                                               os.urandom(32)),
                                    directory, stop=lambda: self.stop_mode() is not None)
        try:
            result = evaluator.run()
        except Stopped:
            record.update(stopped=True, step='decide')
            return
        except ReferenceFailed as exc:
            self._fail(f'the reference failed: {exc}')
            record.update(error=str(exc)[:2000], step='decide')
            return
        write_json(directory / 'result.json', result)
        record['evaluation'] = {k: result.get(k) for k in ('passed', 'complete', 'score', 'segments', 'failure',
                                                           'duration_s', 'gates')}
        record['evaluation']['controls'] = [c for c in result['controls'] if not c['passed']]
        record['evaluation']['cases'] = [c for c in result['cases'] if not c['passed']]
        if result['complete']:
            self.state['evaluation_s'].append(result['duration_s'])
            self.save()
        record['step'] = 'decide'

    def _fail(self, message: str) -> None:
        self.state.update(error=message, phase='done', search_end='error')
        self.save()
        self.event(message)

    def _decide(self, record: dict) -> None:
        s, label = self.state, record['label']
        result = self.evaluation(label) if (record.get('evaluation') or {}).get('passed') else None
        if record.get('unchanged'):
            decision = acceptance.Decision(False, 'unchanged')
        elif record.get('stopped'):
            decision = acceptance.Decision(False, 'stopped')
        elif result is None:
            decision = acceptance.Decision(False, 'failed')
        else:
            incumbent = self.evaluation(s['incumbent']) if s['incumbent'] else None
            decision = acceptance.decide(
                acceptance.Result(True, incumbent['score'], rows(incumbent)) if incumbent else None,
                acceptance.Result(True, result['score'], rows(result)), self.task.objective, self.task.acceptance)
        record.update(decision=decision.to_json(), step='done', ended=time.time())
        self.save_record(record)
        if decision.accepted:
            s['incumbent'] = label
        if record.get('snapshot') and not record.get('stopped'):
            s['latest'] = label
        s['rounds'].append({'label': label, 'kind': record['kind'], 'base_round': record.get('base_round'),
                            'accepted': decision.accepted, 'reason': decision.reason,
                            'score': result['score'] if result else None})
        s['current'] = None
        self.save()
        score = f', score {result["score"]:.4f}' if result else ''
        self.event(f'round {label}: {"accepted" if decision.accepted else "rejected"} ({decision.reason}{score})')
        report.write(self)

    def _facts(self, record: dict) -> str:
        s = self.state
        lines = [f'# Session facts: round {record["label"]} ({record["kind"]})', '']
        if s['incumbent']:
            best = self.evaluation(s['incumbent'])
            lines += [f'## The current best engine: round {s["incumbent"]}', '',
                      f'Score {best["score"]:.4f}; prefill mean {best["segments"]["prefill"]:.4f}, decode mean '
                      f'{best["segments"]["decode"]:.4f} (speedups over the reference).', '', brief.case_table(best),
                      '']
        if record['kind'] == 'repair':
            lines += [f'## Round {record["base_round"]} (the candidate in engine/) failed', '',
                      brief.failure_text(self.record(record['base_round'])), '']
        since = [r for r in s['rounds'] if not r['accepted'] and (not s['incumbent'] or r['label'] > s['incumbent'])]
        if s['incumbent'] and since:
            lines += ['## Candidates rejected since the current best was accepted', '']
            for r in since:
                past = self.record(r['label'])
                d = past.get('decision') or {}
                gains = d.get('segment_gains') or {}
                text = ', '.join(f'{k} {v * 100:+.2f}%' for k, v in gains.items())
                if d.get('score_gain') is not None:
                    text = f'score {d["score_gain"] * 100:+.2f}%, {text}'
                if d.get('best_case'):
                    text += f', best case {d["best_case"][0]} {d["best_case"][1]} {d["best_case"][2] * 100:+.2f}%'
                lines.append(f'- round {r["label"]}: {r["reason"]}' + (f' ({text})' if text else ''))
                if r['reason'] == 'failed':
                    lines.append('  ' + brief.failure_text(past).replace('\n', '\n  '))
            lines.append('')
        if s['rounds']:
            lines += ['## All rounds', '', '| round | kind | from | outcome | score |', '|---|---|---|---|---|']
            for r in s['rounds']:
                score = f'{r["score"]:.4f}' if r['score'] else ''
                lines.append(f'| {r["label"]} | {r["kind"]} | {r["base_round"] or ""} | '
                             f'{"accepted" if r["accepted"] else "rejected"}: {r["reason"]} | {score} |')
        else:
            lines.append('This is the first round of the run.')
        return '\n'.join(lines) + '\n'

    def quality(self, label: str, settings, directory: Path, stop=lambda: False, during_run: bool = True) -> dict:
        """Task accuracy of one round's engine against the reference (enginesmith.quality). During the run the
        harness must be unchanged; afterwards (`enginesmith quality`) the summary records the harness it ran with."""
        from enginesmith import quality
        digest = harness_digest()
        if during_run and digest != self.state['harness']:
            raise RunError('the harness source changed during the run; evaluations stopped')
        engine = [str(self.round_dir(label) / 'engine'), '--model', str(self.task.model)]
        summary = quality.evaluate(settings, self.task.model, self.task.vocab_size, engine,
                                   brief.reference_argv(self.task), directory, registry=self.registry, stop=stop,
                                   log=self.event)
        summary.update(round=label, harness=digest)
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary

    def _quality(self) -> None:
        label = self.state['incumbent']
        self.event(f'quality: MATH-500 / IFEval accuracy of round {label} and of the reference')
        try:
            summary = self.quality(label, self.task.quality, self.dir / 'quality',
                                   stop=lambda: self.stop_mode() == 'abort')
        except Exception as exc:  # report-only: a failure here never changes the run's outcome
            self.state['quality'] = {'round': label, 'error': f'{type(exc).__name__}: {exc}'[:2000]}
            self.event(f'quality: failed: {exc}')
            return
        self.state['quality'] = {'round': label, **{name: {arm: summary[name][arm]['accuracy']
                                                           for arm in ('reference', 'engine')}
                                                    for name in self.task.quality.benchmarks}}

    def _final(self) -> None:
        s = self.state
        if s.get('error'):
            return
        if not s['incumbent']:
            s['final'] = {'skipped': 'no candidate passed'}
            self.event('final: skipped, no candidate passed')
            return
        directory = self.dir / 'final'
        directory.mkdir(exist_ok=True)
        saved = read_json(directory / 'final.json')
        if saved is None:
            # The held-out windows come from a seed created now, after the last coding session.
            saved = {'round': s['incumbent'], 'seed': os.urandom(32).hex(), 'started': time.time()}
            write_json(directory / 'final.json', saved)
        self.event(f'final: evaluating round {saved["round"]} on held-out prompts')
        plan = self.plan(self.task.evaluation.final_rounds, 'held_out', bytes.fromhex(saved['seed']))
        evaluator = self._evaluator(self.round_dir(saved['round']) / 'engine', plan, directory / 'evaluation',
                                    stop=lambda: self.stop_mode() == 'abort')
        try:
            result = evaluator.run()
        except ReferenceFailed as exc:
            s['final'] = {'round': saved['round'], 'error': str(exc)[:2000]}
            return
        write_json(directory / 'evaluation' / 'result.json', result)
        s['final'] = {'round': saved['round'], 'passed': result['passed'], 'score': result['score'],
                      'segments': result['segments'], 'failure': result['failure']}
        self.event(f'final: {"passed" if result["passed"] else "failed"}'
                   + (f', score {result["score"]:.4f}' if result['score'] else ''))
