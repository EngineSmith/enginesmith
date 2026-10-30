"""The enginesmith command line. Commands that a setup agent reads print JSON."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from enginesmith import workload


def emit(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def _driving(directory: Path) -> bool:
    try:
        with open(directory / 'driver.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
    except BlockingIOError:
        return True


def _detach(directory: Path) -> None:
    with open(directory / 'run.log', 'ab') as log:
        subprocess.Popen([sys.executable, '-m', 'enginesmith.cli', 'resume', str(directory)], stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def status(directory: Path) -> dict:
    from enginesmith.run import Run
    run = Run(directory)
    s = run.state
    current = run.record(s['current']) if s['current'] else None
    best = run.evaluation(s['incumbent']) if s['incumbent'] else None
    events = (directory / 'events.jsonl').read_text().splitlines()[-5:] if (directory / 'events.jsonl').exists() else []
    return {'run': str(directory), 'driver_running': _driving(directory), 'phase': s['phase'],
            'stop_requested': run.stop_mode(), 'error': s['error'], 'search_end': s['search_end'],
            'sessions': f'{s["sessions"]} of {run.task.budget.sessions}',
            'hours_left': round((s['deadline'] - time.time()) / 3600, 2),
            'current_round': current and {'label': current['label'], 'kind': current['kind'], 'step': current['step'],
                                          'minutes': round((time.time() - current['started']) / 60, 1)},
            'best': best and {'round': s['incumbent'], 'score': best['score'], 'segments': best['segments']},
            'rounds': [f'{r["label"]} {r["kind"]}: {"accepted" if r["accepted"] else "rejected"} ({r["reason"]})'
                       for r in s['rounds'][-10:]],
            'final': s['final'], 'recent_events': [json.loads(e).get('message') for e in events],
            'report': str(directory / 'report.html')}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='enginesmith', description='Build a specialized inference engine for one '
                                     'model on this machine with a coding agent.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='machine, toolchain and agent facts (JSON)')
    p = sub.add_parser('probe', help='model facts and an estimate of the prompt lengths that fit (JSON)')
    p.add_argument('model', type=Path)
    p = sub.add_parser('validate', help='check a task file (JSON)')
    p.add_argument('task', type=Path)
    p = sub.add_parser('explain', help='write and print the plan of a run for the user (score shares, gates, '
                       'evolution, acceptance, budget); run needs it')
    p.add_argument('task', type=Path)
    p = sub.add_parser('prepare', help='run the reference at every task length and estimate evaluation time')
    p.add_argument('task', type=Path)
    p.add_argument('--lengths', help='comma-separated subset of the task lengths')
    p = sub.add_parser('run', help='start a run')
    p.add_argument('task', type=Path)
    p.add_argument('--runs-dir', type=Path, help='where to create the run directory (default: runs/ next to the task)')
    p.add_argument('--detach', action='store_true', help='run in the background; output goes to RUN/run.log')
    p = sub.add_parser('resume', help='continue a run where it stopped')
    p.add_argument('run', type=Path)
    p.add_argument('--continue-search', action='store_true',
                   help='withdraw a stop request (or an agent failure, once fixed) and keep searching')
    p.add_argument('--detach', action='store_true')
    p = sub.add_parser('status', help='where a run is (JSON)')
    p.add_argument('run', type=Path)
    p = sub.add_parser('stop', help='end the search (the final evaluation still runs); --now stops everything')
    p.add_argument('run', type=Path)
    p.add_argument('--now', action='store_true')
    p = sub.add_parser('report', help='write and print the report path')
    p.add_argument('run', type=Path)
    p = sub.add_parser('quality', help="MATH-500 / IFEval accuracy of a run's engine and of the reference")
    p.add_argument('run', type=Path)
    p.add_argument('--round', help='the round whose engine to evaluate (default: the best engine)')
    p.add_argument('--benchmarks', help='comma-separated: math500, ifeval (default: the task\'s, else both)')
    p.add_argument('--samples', type=int, help='items per benchmark')
    p.add_argument('--max-tokens', type=int, help='output tokens per item')
    p.add_argument('--time-limit', help='for all benchmarks and both arms, e.g. "2h"')
    p = sub.add_parser('share', help="package a run's engine (source, binary, results) to post in the Discussions")
    p.add_argument('run', type=Path)
    p.add_argument('--round', help='the round to share (default: the best engine)')
    p.add_argument('--out', type=Path, help='where to write the package (default: RUN/share)')
    p.add_argument('--no-binary', action='store_true', help='leave the built binary out (source only)')
    p = sub.add_parser('verify', help='build a shared engine from source and evaluate it against the local mlx-lm')
    p.add_argument('package', type=Path, help='a package directory or its zip')
    p.add_argument('--model', type=Path, required=True, help='the model directory (the same files as the package\'s)')
    p.add_argument('--lengths', help='comma-separated prompt lengths (default: the package\'s)')
    p.add_argument('--rounds', type=int, help='paired rounds (default: the package\'s)')
    p.add_argument('--binary', action='store_true', help="run the package's own binary instead of building it")
    p.add_argument('--out', type=Path, help='where to write the result (default: next to the package)')
    p = sub.add_parser('dev-eval', help='inside a session: build and evaluate the engine on public prompts')
    p.add_argument('--lengths', help='comma-separated task lengths, or "all" (default: those up to 32768)')
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--no-build', action='store_true')
    p.add_argument('--cooldown', type=float, help='seconds before each engine process (default: the task\'s)')
    p = sub.add_parser('dataset', help='write the public development prompts as JSON lines')
    p.add_argument('--vocab', type=int, required=True)
    p.add_argument('--lengths', default=','.join(map(str, workload.LENGTHS)))
    p.add_argument('--output-tokens', type=int, default=65)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    from enginesmith.task import TaskError, load_task
    try:
        if args.command == 'doctor':
            from enginesmith.setup_tools import doctor
            emit(doctor())
        elif args.command == 'probe':
            from enginesmith.setup_tools import probe
            emit(probe(args.model))
        elif args.command == 'validate':
            task = load_task(args.task)
            emit({'valid': True, 'digest': task.digest(), 'task': task.to_json(),
                  'cases': [c.__dict__ for c in task.cases()]})
        elif args.command == 'explain':
            from enginesmith.explain import write_plan
            path, text = write_plan(load_task(args.task))
            print(f'<!-- For the agent: show the user this whole plan, every section and number, in their language, '
                  f'with why each choice was made; then ask them to confirm. It is also in {path}; '
                  '`enginesmith run` refuses a task without a current plan. -->\n')
            print(text)
        elif args.command == 'prepare':
            from enginesmith.setup_tools import prepare
            lengths = [int(n) for n in args.lengths.split(',')] if args.lengths else None
            emit(prepare(args.task, lengths, log=lambda m: print(m, file=sys.stderr, flush=True)))
        elif args.command == 'run':
            from enginesmith.explain import plan_problem
            from enginesmith.run import Run
            from enginesmith.setup_tools import SANDBOXED, sandboxed
            task = load_task(args.task)
            if problem := plan_problem(task) or (sandboxed() and SANDBOXED):
                raise TaskError(problem)
            run = Run.create(task, args.runs_dir)
            if args.detach:
                _detach(run.dir)
                emit({'run': str(run.dir), 'log': str(run.dir / 'run.log'), 'detached': True})
            else:
                print(f'run directory: {run.dir}', flush=True)
                run.drive()
                emit(status(run.dir))
        elif args.command == 'resume':
            from enginesmith.run import Run
            from enginesmith.setup_tools import SANDBOXED, sandboxed
            if sandboxed():
                raise TaskError(SANDBOXED)
            run = Run(args.run)
            if args.continue_search:
                (run.dir / 'STOP').unlink(missing_ok=True)
                run.state.update(agent_failed=None, quick_failures=0)
                if (run.state['phase'] == 'final' and run.state['search_end'] in ('stopped', 'agent_failed')
                        and not run.state['final']):
                    run.state.update(phase='search', search_end=None)
                run.save()
            elif run.stop_mode() == 'abort':
                (run.dir / 'STOP').write_text('finish')
            if args.detach:
                _detach(run.dir)
                emit({'run': str(run.dir), 'log': str(run.dir / 'run.log'), 'detached': True})
            else:
                run.drive()
                emit(status(run.dir))
        elif args.command == 'status':
            emit(status(args.run.resolve()))
        elif args.command == 'stop':
            from enginesmith.run import Run
            run = Run(args.run)
            (run.dir / 'STOP').write_text('abort' if args.now else 'finish')
            emit({'run': str(run.dir), 'stop': 'abort' if args.now else 'finish',
                  'note': 'the driver notices within seconds; a running session is ended and kept as evidence'})
        elif args.command == 'report':
            from enginesmith import report
            from enginesmith.run import Run
            report.write(Run(args.run))
            print(Path(args.run).resolve() / 'report.html')
        elif args.command == 'quality':
            import dataclasses
            from enginesmith.run import Run
            from enginesmith.task import Quality, duration
            run = Run(args.run)
            label = args.round or run.state['incumbent']
            if not label:
                raise TaskError('the run has no best engine; name a round with --round')
            settings = run.task.quality or Quality()
            changes = {key: value for key, value in (
                ('benchmarks', tuple(args.benchmarks.split(',')) if args.benchmarks else None),
                ('samples', args.samples), ('max_tokens', args.max_tokens),
                ('time_limit_s', duration(args.time_limit, '--time-limit') if args.time_limit else None))
                if value is not None}
            settings = dataclasses.replace(settings, **changes)
            with contextlib.redirect_stdout(sys.stderr):  # progress lines; the summary alone goes to stdout
                summary = run.quality(label, settings, run.dir / f'quality-round-{label}', during_run=False)
            emit(summary)
        elif args.command == 'share':
            from enginesmith.run import Run
            from enginesmith.share import share
            emit(share(Run(args.run), args.round, args.out, binary=not args.no_binary))
        elif args.command == 'verify':
            from enginesmith.share import verify
            lengths = [int(n) for n in args.lengths.split(',')] if args.lengths else None
            emit(verify(args.package, args.model, out=args.out, lengths=lengths, rounds=args.rounds,
                        use_binary=args.binary, log=lambda m: print(m, file=sys.stderr, flush=True)))
        elif args.command == 'dev-eval':
            from enginesmith import dev_eval
            return dev_eval.run(args.lengths, args.rounds, not args.no_build, args.cooldown)
        elif args.command == 'dataset':
            lengths = [int(n) for n in args.lengths.split(',')]
            with open(args.out, 'x') as out:
                for case in workload.cases(lengths, args.output_tokens):
                    out.write(json.dumps({'id': case.id, 'prompt': workload.prompt(case.length, args.vocab),
                                          'output_tokens': case.output_tokens, 'weight': case.weight}) + '\n')
            emit({'written': str(args.out), 'cases': len(lengths)})
    except TaskError as exc:
        emit({'valid': False, 'error': str(exc)})
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
