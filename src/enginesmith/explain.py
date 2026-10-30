"""`enginesmith explain TASK`: the plan of a run, generated from the task file and prepare.json, for the user to read
before the run starts. Every number in it comes from the task, the harness's own constants or a measurement; the
setup agent translates it and adds why each value was chosen, but never restates the rules from memory.

It also writes the plan to plan.md next to the task, stamped with what it was generated from; `enginesmith run`
refuses a task whose plan is missing or stale, so no run starts before its plan was produced for the user."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from enginesmith import references, workload
from enginesmith.brief import reference_argv, reference_kind
from enginesmith.evaluate import MIN_TIMEOUT, TIMEOUT_FACTOR
from enginesmith.task import Task


def _hours(seconds: float) -> str:
    return f'{seconds / 3600:.1f} h' if seconds >= 3600 else f'{seconds / 60:.0f} min'


def _pct(value: float) -> str:
    return f'{100 * value:g}%'


def _share(value: float) -> str:
    return f'{100 * value:.1f}%'


def _architecture(task: Task) -> list[str]:
    try:
        from enginesmith.setup_tools import probe
        facts = probe(task.model)
    except Exception as exc:  # the plan still renders without the model's facts
        return [f'- Model facts unavailable: {exc}']
    def n(value) -> str:
        return f'{value:,}' if isinstance(value, int) else '?'
    kinds = ', '.join(f'{count} {kind}' for kind, count in facts['layer_kinds'].items())
    experts = (f'; {facts["experts"]} experts, {facts["experts_per_token"]} per token' if facts['experts'] else '')
    quant = facts['quantization']
    quant = f'{quant.get("bits")}-bit, group {quant.get("group_size")}' if isinstance(quant, dict) else 'none'
    return [f'- Architecture: `{facts["model_type"]}`, {n(facts["layers"])} layers ({kinds}){experts}; hidden '
            f'{n(facts["hidden_size"])}, {n(facts["attention_heads"])} query / {n(facts["kv_heads"])} KV heads of '
            f'{n(facts["head_dim"])}; vocabulary {n(facts["vocab_size"])}; context {n(facts["max_position_embeddings"])}.',
            f'- Weights: {facts["weights_bytes"] / 2**30:.2f} GiB, quantization {quant}. KV cache '
            f'{n(facts["kv_bytes_per_token_estimate"])} bytes per token of context (estimate: {facts["kv_estimate_basis"]}).']


def _prepare(task: Task) -> dict:
    path = task.path.parent / 'prepare.json'
    return json.loads(path.read_text()) if path.is_file() else {}


AGENT_NAMES = {'claude': 'Claude Code', 'codex': 'Codex CLI', 'kimi': 'Kimi Code', 'pi': 'Pi', 'dsh': 'DeepSeek Harness'}


def _stamp(task: Task) -> str:
    """The first line of plan.md: a digest of the task's settings and of prepare.json, the plan's two inputs."""
    prepare = task.path.parent / 'prepare.json'
    digest = hashlib.sha256(task.digest().encode() + (prepare.read_bytes() if prepare.is_file() else b''))
    return f'<!-- enginesmith plan {digest.hexdigest()} -->'


def write_plan(task: Task) -> tuple[Path, str]:
    text = explain(task)
    path = task.path.parent / 'plan.md'
    path.write_text(f'{_stamp(task)}\n{text}')
    return path, text


def plan_problem(task: Task) -> str | None:
    """Why a run cannot start yet: no plan.md, or one written before the task or prepare.json changed."""
    path = task.path.parent / 'plan.md'
    if not path.is_file():
        return (f'no plan for this task: run `enginesmith explain {task.path}` and show the user the whole plan '
                'before starting the run')
    if path.read_text().split('\n', 1)[0] != _stamp(task):
        return (f'the task or prepare.json changed since {path} was written: run `enginesmith explain {task.path}` '
                'again and show the user the new plan')
    return None


def explain(task: Task) -> str:
    o, a, g, e, b = task.objective, task.acceptance, task.gates, task.evaluation, task.budget
    prepared = _prepare(task)
    measured = {row['length']: row for row in prepared.get('observed', {}).get('lengths', [])}
    load_s = (prepared.get('observed', {}).get('load_ms') or 0) / 1000
    out = [f'# Plan: {task.name}', '']

    out += ['## 1. Model and reference', '',
            f'- Model: `{task.model.name}`' + (f' from `{task.model_source}`' if task.model_source else '')
            + f' (vocabulary bound for prompts: {task.vocab_size:,}).']
    out += _architecture(task)
    kind = reference_kind(task)
    described = {'mlx-lm': 'stock mlx-lm, reproducing `generate_step` token for token: prefill in chunks of '
                           'min(2048, remaining - 1) tokens, the last prompt token alone, greedy decoding pipelined '
                           'one step ahead'}
    out += [f'- Reference: {described.get(kind, kind)} (`{" ".join(reference_argv(task)[1:3])}`, '
            f'version {references.version(kind)}).' if kind else
            f'- Reference: the custom command `{" ".join(task.reference)}`.', '']

    timed = [c for c in task.cases() if c.timed]
    total_weight = sum(o.case_weight(c.length) * (o.prefill + o.decode) for c in timed)
    out += ['## 2. What is measured', '',
            f'Every timed case is one prompt length. Per case and request, the harness times a prefill request '
            f'`generate(P, 1)` (time to first token) and a decode request `generate((), {task.output_tokens - 1})` '
            f'({task.output_tokens - 1} more tokens), from just before it writes the request until the reply has '
            'arrived; the engine\'s own timings are ignored. Prompts are raw token ids from the public Lily formula, '
            'a fresh window for every request, so nothing can be cached between requests.', '',
            '| prompt tokens | case weight | share of the score: prefill | decode | reference here: prefill | '
            'decode per token | peak memory |', '|---:|---:|---:|---:|---:|---:|---:|']
    for c in timed:
        w = o.case_weight(c.length)
        row = measured.get(c.length)
        ref = (f'{row["prefill_ms"] / 1000:.2f} s | {row["decode_ms_per_token"]:.2f} ms | '
               f'{row["peak_memory_bytes"] / 2**30:.1f} GiB') if row else 'not measured | not measured | -'
        out.append(f'| {c.length:,} | {w:g} | {_share(w * o.prefill / total_weight) if total_weight else "-"} | '
                   f'{_share(w * o.decode / total_weight) if total_weight else "-"} | {ref} |')
    controls = [c.length for c in task.cases() if not c.timed]
    out += ['', (f'Correctness-only control prompts ({len(controls)}): lengths {", ".join(map(str, controls))}, '
                 f'{task.output_tokens} tokens each; they are checked before any timing.') if controls else
            'No control prompts (`workload.controls = false`).', '']
    if not measured:
        out += ['The reference has not been measured on this machine yet: run `enginesmith prepare` first.', '']

    out += ['## 3. The score', '',
            'For each timed case c and segment s (prefill, decode): speedup(c, s) = median over the paired rounds of '
            'reference time / engine time. The score is the weighted geometric mean',
            '', '    score = exp( sum_c sum_s w_c * w_s * ln speedup(c, s) / sum_c sum_s w_c * w_s )', '',
            f'with segment weights w_prefill = {o.prefill:g}, w_decode = {o.decode:g} and the case weights above; the '
            'table\'s "share" columns are each term\'s weight in it. 1.20 means the engine is 20% faster than '
            'mlx-lm on this mix. The prefill and decode means used by the acceptance rules weight the cases by '
            'their case weights only.', '']

    allowed = g.case_mismatch_percent * task.output_tokens // 100
    out += ['## 4. Correctness gates (every evaluation)', '',
            f'1. Control prompts first: a candidate that fails them is rejected within minutes, before timing.',
            '2. A candidate token stream identical to the reference\'s agrees. A stream that differs is replayed '
            'through the reference with the candidate\'s tokens forced (teacher forcing); a step whose token is not '
            'the reference\'s top choice is a mismatch.',
            f'3. A stream passes with at most {g.case_mismatch_percent}% mismatching steps ({allowed} of '
            f'{task.output_tokens})' + (f', each mismatching token among the reference\'s top {g.mismatch_rank} '
                                        'choices' if g.mismatch_rank else '') +
            f'; the whole evaluation with at most {g.total_mismatch_percent}% of all judged steps.',
            f'4. Every candidate request must answer within {TIMEOUT_FACTOR} times the reference\'s time for it '
            f'(and at least {MIN_TIMEOUT:g} s); a slower candidate fails.', '',
            'A correct engine only diverges at near-ties: its mismatching tokens are the reference\'s 2nd or 3rd '
            'choice (up to rank ~50 on MoE models, whose expert routing can flip on a one-ulp difference); a wrong '
            'one produces tokens ranked in the hundreds or thousands.', '']

    per_arm = sum(row['prefill_ms'] + row['decode_ms'] for row in measured.values()) / 1000 + load_s
    estimate = e.rounds * 2 * (per_arm + e.cooldown_s) + 2 * load_s + 60 if measured else None
    out += ['## 5. One evaluation', '',
            f'- {e.rounds} paired rounds, the reference first in even rounds and the candidate first in odd ones '
            '(AB, BA, ...), the same fresh prompt for both arms of a round.',
            f'- Each arm of each round is a fresh engine process: load, {e.warmup} untimed warmup request(s), then '
            f'every case; one engine resident at a time, {e.cooldown_s:g} s cooldown before every process.',
            '- Round 0 is judged before the other rounds are timed, so a wrong engine fails early.',
            f'- Duration: about {_hours(estimate)} (ESTIMATE: the reference\'s measured times, the candidate assumed as '
            'fast, plus loads and cooldowns).' if estimate else '- Duration: unknown until `enginesmith prepare` runs.',
            '']

    agent = (' '.join(task.agent_command) if task.agent == 'command' else
             f'{AGENT_NAMES[task.agent]} (`{task.agent}`) with '
             + (f'model `{task.agent_model}`' if task.agent_model else "the agent's default model"))
    out += ['## 6. How the search evolves', '',
            f'The run is a sequence of rounds; each round is one candidate. A round gets a fresh headless session of '
            f'{agent} (at most {_hours(b.session_timeout_s)}), whose brief holds the contract (this plan\'s rules), the '
            'facts of '
            'earlier rounds, the skill and the model card. The agent edits `engine/` and checks itself with '
            '`enginesmith dev-eval` (the same procedure on public prompts). When the session ends, the harness commits '
            '`engine/`, builds a clean checkout of that commit, runs one formal evaluation (sections 4-5) and decides.',
            '', '| round kind | when | starts from |', '|---|---|---|']
    if task.seed:
        out.append('| seed | first, with no session | the seed engine (`engine.seed`) |')
    out += ['| generate | no candidate yet | ' + ('the seed' if task.seed else 'an empty project') + ' |',
            '| repair | no candidate has passed yet | the latest candidate, with its failure as facts |',
            '| optimize | a best engine exists | the best engine, with the rejected candidates since as facts |', '',
            'A candidate whose source equals its starting point is rejected as unchanged without an evaluation. '
            'A rejected optimization is not continued: the next session starts from the best engine again, knowing '
            'what was tried and why it was rejected.', '']

    rules = []
    if a.min_improvement:
        rules.append(f'the score improves by at least {_pct(a.min_improvement)}')
    if a.segment_improvement:
        rules.append(f'the prefill or the decode mean improves by at least {_pct(a.segment_improvement)} while the '
                     f'other mean and the score get worse by at most {_pct(a.max_regression)}')
    if a.case_improvement:
        rules.append(f'one case\'s prefill or decode speedup improves by at least {_pct(a.case_improvement)} while the '
                     f'score gets worse by at most {_pct(a.max_regression)}')
    out += ['## 7. Acceptance: when a candidate replaces the best engine', '',
            '- It must pass every gate of section 4 and complete the evaluation.',
            '- The first candidate that passes becomes the best engine, however slow it is.',
            '- After that, a passing candidate replaces the best engine when ' + ('; or when '.join(rules) or
                                                                                 'never (every rule is off)') + '.',
            '- Only weighted cases and segments count. The thresholds sit above the timing noise of an unchanged '
            'engine (about 1% for the means, up to 4% for one case); lowering them accepts smaller steps and more '
            'noise. A rejected candidate stays in the run\'s history (its commit is kept).', '']

    q = task.quality
    out += ['## 8. After the search', '',
            f'- Final: the best engine is evaluated once more, {e.final_rounds} paired rounds on held-out prompts drawn '
            'from a seed created after the last session; this is the result to report.']
    out.append(f'- Task accuracy: {", ".join(q.benchmarks)}, {q.samples} items each, greedy decoding, at most '
               f'{q.max_tokens:,} output tokens, time limit {_hours(q.time_limit_s)}; the engine and mlx-lm answer the '
               'same items; reported next to each other, not gated.' if q else
               '- Task accuracy: not run (no `[quality]` table).')
    out += ['']

    evaluation_s = estimate or 3600.0
    final_s = evaluation_s * e.final_rounds / e.rounds
    out += ['## 9. Budget', '',
            f'- Up to {b.sessions} sessions of at most {_hours(b.session_timeout_s)}, within {_hours(b.wall_time_s)} '
            'in all.',
            f'- Before each session the run checks that the session, one evaluation (~{_hours(evaluation_s)}), the '
            f'final (~{_hours(final_s)})' + (f', the task accuracy ({_hours(q.time_limit_s)})' if q else '') +
            ' and 10 minutes still fit; otherwise the search ends and the final runs.',
            f'- Rough upper bound for the search: {b.sessions} x ({_hours(b.session_timeout_s)} + {_hours(evaluation_s)})'
            f' = {_hours(b.sessions * (b.session_timeout_s + evaluation_s))}; sessions usually end earlier.',
            '- The agent runs on your own login or API key; its cost is recorded per round when it reports one.', '']
    return '\n'.join(out)
