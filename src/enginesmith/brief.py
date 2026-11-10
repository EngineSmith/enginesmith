"""The brief a coding session reads: the contract (README.md), this session's facts (SESSION.md), the skill, the
model card and the engine protocol. The harness writes facts only; strategy lives in the skill."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from enginesmith import references
from enginesmith.platforms import current
from enginesmith.task import REPO, Task

PROTOCOL_DOC = REPO / 'docs' / 'protocol.md'
KIND_INTRO = {
    'generate': ('There is no passing engine yet. `engine/` holds {start}. Build a first engine that passes the '
                 'correctness gates; correctness comes before speed.'),
    'repair': ('There is no passing engine yet. `engine/` holds the latest candidate, which failed (SESSION.md says '
               'how). Make it pass the correctness gates.'),
    'optimize': ('`engine/` holds the current best engine. Make it faster under the objective below while it keeps '
                 'passing the correctness gates.'),
}


def reference_kind(task: Task) -> str | None:
    """The reference kind a task uses: its own, else the platform's default (None with a custom command)."""
    if task.reference:
        return None
    return task.reference_kind or current().DEFAULT_REFERENCE


def reference_argv(task: Task) -> list[str]:
    if task.reference:
        return list(task.reference)
    kind = reference_kind(task)
    if kind is None:
        raise ValueError(f'the {current().NAME} platform has no default reference; set [reference] kind or command')
    return references.command(kind, task.model, task.max_request_length(), task.reference_tp)


def reference_concurrent_argv(task: Task) -> list[str]:
    """The reference of a concurrent task's timed rounds: the reference in its concurrent mode (protocol 2), started
    with --concurrent; a custom reference command must accept it."""
    kind = reference_kind(task)
    if not task.reference and kind != 'mlx-lm':
        raise ValueError(f'a concurrent task needs a reference with a concurrent mode; {kind} has none yet (mlx-lm '
                         'has)')
    return reference_argv(task) + ['--concurrent']


def _percent(value: float) -> str:
    return f'{value * 100:g}%'


def rules_text(task: Task) -> list[str]:
    """The acceptance rules in words, with the task's segment names."""
    a = task.acceptance
    names = ' or '.join(task.objective.SEGMENTS)
    rules = []
    if a.min_improvement:
        rules.append(f'the score improves by at least {_percent(a.min_improvement)}')
    if a.segment_improvement:
        rules.append(f'the mean of one segment ({names}) improves by at least {_percent(a.segment_improvement)}, '
                     f'while the other weighted means and the score get worse by at most {_percent(a.max_regression)}')
    if a.case_improvement:
        rules.append(f'one case\'s speedup in one segment improves by at least {_percent(a.case_improvement)}, '
                     f'while the score gets worse by at most {_percent(a.max_regression)}')
    return rules


def _judged_single(task: Task) -> tuple[str, str]:
    """The protocol paragraph and the speed and score items of a single-request task's brief."""
    o, e = task.objective, task.evaluation
    lengths = ', '.join(f'{n:,} (weight {o.case_weight(n):g})' for n in task.lengths)
    protocol = (f'`brief/protocol.md` defines it: the engine is started as `ENGINE --model MODEL_DIR` and answers '
                f'`hello` and\n`generate` requests as JSON lines. The reference implementation '
                f'({reference_kind(task) or "a custom reference"}) runs\nthe same way (`{" ".join(reference_argv(task))}`).')
    speed = f'''2. Speed: {e.rounds} paired rounds, reference and candidate alternating (AB, BA, ...), each engine a fresh process
   with {e.warmup} warmup request(s), {e.cooldown_s:g} s cooldown before each process. Per case: a prefill request
   generate(P, 1) and a decode request generate((), {task.output_tokens - 1}), each on a fresh prompt. A speedup is the
   median over rounds of reference time / candidate time. A request that takes more than 50 times the reference's
   time for it (and more than 300 s) fails the candidate.
3. The score is the weighted geometric mean of the per-case prefill and decode speedups. Prefill weight
   {o.prefill:g}, decode weight {o.decode:g}. Prompt lengths: {lengths}. Every case generates
   {task.output_tokens} tokens.'''
    return protocol, speed


def _judged_concurrent(task: Task) -> tuple[str, str]:
    """The same for a concurrent task."""
    o, e, c = task.objective, task.evaluation, task.concurrent
    levels = ', '.join(f'{n} (weight {o.case_weight(n):g})' for n in c.levels)
    protocol = (f'`brief/protocol.md` defines it; this task uses **protocol 2** (concurrent requests). The engine is '
                f'started as\n`ENGINE --model MODEL_DIR`, answers `hello` with protocol 2, and must accept `submit` '
                'requests at any time, also while it\nis computing others, and stream each request\'s tokens as '
                '`{"tokens": [[id, token], ...]}` lines. Requests are independent:\neach one starts from an empty '
                'state. How it schedules them is up to the engine. The reference\n'
                f'({reference_kind(task) or "a custom reference"}, its own continuous-batching generator) runs as '
                f'`{" ".join(reference_concurrent_argv(task))}`.')
    speed = f'''2. Speed: {e.rounds} paired rounds, reference and candidate alternating (AB, BA, ...), each engine a fresh process
   with {e.warmup} warmup request(s), {e.cooldown_s:g} s cooldown before each process. Per concurrency level C, C
   closed-loop clients send the level's {c.requests_per_level} requests: a client sends its next request as soon as
   its last one has all its tokens; the clients start within {c.stagger_s:g} s of each other. Prompt lengths are drawn
   from {c.prompt_range[0]:,}-{c.prompt_range[1]:,} tokens and output lengths from {c.output_range[0]}-{c.output_range[1]}
   tokens (the same request plan in every round of the run, fresh prompt windows every round). The harness times a
   request from just before it writes it: throughput = time from the first request to the last token; ttft = the
   90th percentile over the requests of the time to the first token; latency = the 90th percentile over the
   requests of the end-to-end time divided by the output tokens. A speedup is the median over rounds of reference /
   candidate. A level that takes more than 50 times the reference's time (and more than 300 s) fails the candidate.
3. The score is the weighted geometric mean of the per-level speedups. Segment weights: throughput
   {o.throughput:g}, ttft {o.ttft:g}, latency {o.latency:g}. Levels: {levels}.'''
    return protocol, speed


def contract(task: Task, kind: str, start: str) -> str:
    g = task.gates
    rules = rules_text(task)
    protocol, speed = (_judged_concurrent if task.mode == 'concurrent' else _judged_single)(task)
    allowed = current().ALLOWED
    native = ('\n- The engine must be native and self-contained: one executable that needs no files other than the '
              'model directory and computes the model with its own source. It links only the operating system\'s '
              'and the C/C++ runtime\'s libraries, the accelerator\'s driver and runtime layer, and utilities that do '
              'no arithmetic on the model' + (f' (here: {", ".join(sorted(allowed, key=str.lower))})' if allowed else '')
              + '; never Python, a framework, an inference runtime or a precompiled numerical library (MLX, mlx-lm, '
              'PyTorch, llama.cpp, a BLAS and the like). You may read and adapt the source of any open-source '
              'project (respect its license): copied kernels are the engine\'s own source.') if task.native else ''
    forward = ('\n- Decode computes one forward pass of the checkpoint per generated token: no draft models, and no '
               'drafted, speculated or looked-up tokens verified in batches.') if task.one_forward_per_token else ''
    card = '\n- `brief/model-card.md`: the model card written when this task was set up.' if task.model_card else ''
    quality = ('\n6. Then the best engine and the reference answer MATH-500 and IFEval with greedy decoding, and '
               'the accuracy is reported next to the reference\'s.') if task.quality else ''
    return f'''# EngineSmith session brief

You are building a specialized inference engine for one model on this machine. EngineSmith, the harness that
started this session, judges your work. This file is its contract with you; it wins over the skill and anything
else you read.

## Your task this session

{KIND_INTRO[kind].format(start=start)}

`brief/SESSION.md` has the facts from earlier rounds. Finish when this task is done; there is no reward for using
the whole time. A diagnosis or a plan for a later session is not done: the harness evaluates `engine/` as you leave
it, and the next session starts from there. The session ends at the latest after
{task.budget.session_timeout_s / 3600:g} h, and `engine/` is taken as it is then.

## The model

`{task.model}` (vocabulary size {task.vocab_size:,}). Do not modify anything in it.

## What you deliver

- The source in `engine/`. It is a git checkout; when the session ends the harness commits everything in it except
  what `.gitignore` excludes (keep `target/` ignored). Do not commit model weights or large generated files.
- The harness builds a clean checkout of that commit with `{' '.join(task.build)}` and runs `{task.executable}` from
  it. The build must produce that one executable.{native}{forward}
- Keep `engine/NOTES.md` up to date: what you changed, how you verified it, what it measured, and what you tried
  and rejected, with numbers. It travels with the engine to the next session.

## The engine protocol

{protocol}

## How a candidate is judged

1. Correctness: short control prompts first, then every timed request. A token stream identical to the
   reference's agrees. A stream that differs is replayed through the reference with your tokens forced; a step
   whose token is not the reference's top choice is a mismatch. Each stream may have at most
   {g.case_mismatch_percent}% mismatching steps, {f'each token among the reference\'s top {g.mismatch_rank} choices, ' if g.mismatch_rank else ''}and the whole evaluation at most {g.total_mismatch_percent}%.
{speed}
4. A passing candidate replaces the current best engine when {"; or when ".join(rules) or "it is the first to pass"}.
   Otherwise it is rejected, and the next session starts from the best engine again.
5. After the last session, the best engine is evaluated once more on held-out prompts drawn after the search.{quality}

## Tools

- `enginesmith dev-eval` builds `engine/` (`--no-build` to skip) and runs the same correctness checks and paired
  timing on public prompts, one round by default (`--rounds N`, `--lengths 256,4096` or `--lengths all`). Use it
  before you finish. It holds the GPU lock `/tmp/engine-gpu.lock` while it runs.
- The skill's scripts (`brief/skill/scripts/`) speak the same protocol; they take the same GPU lock.
- Only one process should use the GPU while you measure.

## Files

- `brief/SESSION.md`: this session's facts.
- `brief/skill/SKILL.md`: how to build and speed up an engine. Read it before you start.{card}
- `brief/protocol.md`: the engine protocol.

## Rules

- Work inside this session directory. Do not modify the harness, the reference, the model or anything else
  outside it (build caches such as `~/.cargo` are fine).
- The evaluation prompts are drawn fresh for every request; do not try to find or predict them.
- Stop every process you start (engines, profilers) before you finish.
- Copy or adapt code only from permissively licensed projects (MIT, BSD, Apache-2.0 and the like): shared engines
  are published under MIT by default. Keep their copyright notices, and list each in `engine/THIRD_PARTY.md` with
  the project, the files and the license.
'''


def write(session_dir: Path, task: Task, kind: str, start: str, facts: str, info: dict) -> Path:
    """Create session_dir/brief and its files; returns the session file for ENGINESMITH_SESSION."""
    brief = session_dir / 'brief'
    brief.mkdir(parents=True)
    (brief / 'README.md').write_text(contract(task, kind, start))
    (brief / 'SESSION.md').write_text(facts)
    shutil.copytree(task.skill, brief / 'skill', ignore=shutil.ignore_patterns('__pycache__'))
    if task.model_card:
        shutil.copy2(task.model_card, brief / 'model-card.md')
    shutil.copy2(PROTOCOL_DOC, brief / 'protocol.md')
    (brief / 'dev-eval').mkdir()
    session_file = brief / 'session.json'
    session_file.write_text(json.dumps(info, indent=2))
    return session_file


def _ms(values) -> str:
    return '/'.join(f'{v:.1f}' for v in values)


def case_table(result: dict) -> str:
    if result.get('mode') == 'concurrent':
        return level_table(result)
    lines = ['| case | prefill speedup | decode speedup | prefill ms (ref / cand, per round) | '
             'decode ms (ref / cand, per round) | mismatching steps per round |', '|---|---|---|---|---|---|']
    for c in result['cases']:
        lines.append(f"| {c['id']} | {c['prefill_speedup']:.3f} | {c['decode_speedup']:.3f} | "
                     f"{_ms(c['prefill_ms']['reference'])} / {_ms(c['prefill_ms']['candidate'])} | "
                     f"{_ms(c['decode_ms']['reference'])} / {_ms(c['decode_ms']['candidate'])} | "
                     f"{c['mismatches']} |")
    return '\n'.join(lines)


def _per_round(case: dict, key: str) -> str:
    return (f"{_ms(m[key] for m in case['reference'])} / {_ms(m[key] for m in case['candidate'])}")


def _speedup(value) -> str:
    return f'{value:.3f}' if isinstance(value, (int, float)) else '-'


def level_table(result: dict) -> str:
    """A concurrent result per level: the speedups, then the reference's and the candidate's times per round."""
    lines = ['| level | throughput speedup | ttft speedup | latency speedup | makespan ms (ref / cand, per round) | '
             'ttft p90 ms | latency p90 ms per token | mismatching steps per round |', '|---|---|---|---|---|---|---|---|']
    for c in result['cases']:
        lines.append(f"| {c['level']} clients, {c['requests']} requests | {_speedup(c['throughput_speedup'])} | "
                     f"{_speedup(c['ttft_speedup'])} | {_speedup(c['latency_speedup'])} | "
                     f"{_per_round(c, 'makespan_ms')} | {_per_round(c, 'ttft_p90_ms')} | "
                     f"{_per_round(c, 'latency_p90_ms')} | {c['mismatches']} |")
    return '\n'.join(lines)


def failure_text(record: dict) -> str:
    """What went wrong with a round's candidate, as facts."""
    lines = []
    build = record.get('build') or {}
    if build and not build.get('passed'):
        lines += [f"- The build failed: {build.get('error')}.", '', 'Build log (tail):', '```',
                  build.get('log_tail', ''), '```']
    evaluation = record.get('evaluation') or {}
    failure = evaluation.get('failure')
    if failure:
        lines.append(f"- The evaluation failed ({failure['kind']}) at {failure.get('where')}: {failure['message']}")
        if failure.get('stderr_tail'):
            lines += ['', 'Engine stderr (tail):', '```', failure['stderr_tail'], '```']
        streams = [s for c in evaluation.get('cases', []) for s in c.get('failing_streams', [])]
        for c in evaluation.get('controls', []) + evaluation.get('cases', []) + failure.get('streams', []) + streams:
            if not c.get('passed') and c.get('first_mismatch') and 'allowed' in c:
                step, margin, token, rank = c['first_mismatch']
                lines.append(f"- {c['id']}: {c['mismatches']} mismatching steps (at most {c['allowed']} allowed), "
                             f"worst rank {c.get('worst_rank')}; first at step {step}, where the reference's top "
                             f"token is {token} and yours is its choice number {rank}, {margin:.4g} below it in "
                             f"log-probability.")
    if record.get('error'):
        lines.append(f"- {record['error']}")
    return '\n'.join(lines) or '- No details were recorded.'
