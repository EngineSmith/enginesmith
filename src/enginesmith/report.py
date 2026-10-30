"""The run report: report.json (facts) and report.html (the same facts for people)."""
from __future__ import annotations

import csv
import html
import io
import json
import statistics
import time

from enginesmith import agents


# One row per decided round, then one for the held-out final. The per-case columns follow, for every task length.
FIELDS = ('round', 'kind', 'base_round', 'started_min', 'ended_min', 'session_min', 'accepted', 'reason', 'failure',
          'passed', 'score', 'prefill', 'decode', 'incumbent_score', 'agent_model', *agents.USAGE_KEYS,
          'cumulative_cost_usd', 'dev_evals', 'dev_evals_passed', 'engine_lines', 'lines_added', 'lines_removed',
          'changed_files')
CASE_FIELDS = ('prefill_speedup', 'decode_speedup', 'ttft_ms', 'prefill_tok_s', 'decode_tok_s', 'reference_ttft_ms',
               'reference_prefill_tok_s', 'reference_decode_tok_s', 'mismatches')


def _speeds(case: dict) -> dict:
    """A case's speeds, medians over the paired rounds: the time to first token and prefill tokens per second from
    the prefill request, decode tokens per second from the decode request (output_tokens - 1 tokens); the engine's,
    then the reference's."""
    out = {}
    for arm, prefix in (('candidate', ''), ('reference', 'reference_')):
        prefill = (case.get('prefill_ms') or {}).get(arm) or []
        decode = (case.get('decode_ms') or {}).get(arm) or []
        ttft = statistics.median(prefill) if prefill else None
        step = statistics.median(decode) / (case['output_tokens'] - 1) if decode and case['output_tokens'] > 1 else None
        out.update({f'{prefix}ttft_ms': round(ttft, 3) if ttft else None,
                    f'{prefix}prefill_tok_s': round(case['length'] * 1000 / ttft, 1) if ttft else None,
                    f'{prefix}decode_tok_s': round(1000 / step, 1) if step else None})
    return out


def _cases(result: dict | None) -> list[dict]:
    return [{'length': c['length'], 'prefill_speedup': c.get('prefill_speedup'),
             'decode_speedup': c.get('decode_speedup'), 'mismatches': sum(c.get('mismatches') or []), **_speeds(c)}
            for c in (result or {}).get('cases', [])]


def history(run) -> list[dict]:
    """The run's evolution, recorded as it runs. Minutes count from the run's start; tokens and costs are what the
    agent reported, None when it does not report them."""
    s = run.state

    def minutes(t):
        return round((t - s['created']) / 60, 1) if t else None
    rows, incumbent, cost, priced = [], None, 0.0, False
    for entry in s['rounds']:
        record = run.record(entry['label']) or {}
        session, code, checks = record.get('session') or {}, record.get('code') or {}, record.get('dev_evals') or {}
        result = run.evaluation(entry['label'])
        failure = (result or {}).get('failure') or {}
        if entry['accepted']:
            incumbent = entry['score']
        if isinstance(session.get('cost_usd'), (int, float)):
            cost, priced = cost + session['cost_usd'], True
        rows.append({**dict.fromkeys(FIELDS),
                     'round': entry['label'], 'kind': entry['kind'], 'base_round': entry.get('base_round'),
                     'started_min': minutes(record.get('started')), 'ended_min': minutes(record.get('ended')),
                     'session_min': round(session['duration_s'] / 60, 1) if session.get('duration_s') else None,
                     'accepted': entry['accepted'], 'reason': entry['reason'],
                     'failure': failure.get('kind') or ('build' if (record.get('build') or {}).get('error') else None),
                     'failure_message': failure.get('message') or (record.get('build') or {}).get('error'),
                     'passed': (result or {}).get('passed'), 'score': entry.get('score'),
                     'prefill': ((result or {}).get('segments') or {}).get('prefill'),
                     'decode': ((result or {}).get('segments') or {}).get('decode'), 'incumbent_score': incumbent,
                     'agent_model': session.get('model'), **{key: session.get(key) for key in agents.USAGE_KEYS},
                     'cumulative_cost_usd': round(cost, 6) if priced else None,
                     'dev_evals': checks.get('runs'), 'dev_evals_passed': checks.get('passed'),
                     'engine_lines': code.get('lines'), 'lines_added': code.get('added'),
                     'lines_removed': code.get('removed'), 'changed_files': len(record.get('changed_files') or []),
                     'cases': _cases(result)})
    final = s.get('final') or {}
    if final.get('round'):
        path = run.dir / 'final' / 'evaluation' / 'result.json'
        result = json.loads(path.read_text()) if path.is_file() else {}
        ended = result['started'] + result['duration_s'] if result.get('started') and result.get('duration_s') else None
        rows.append({**dict.fromkeys(FIELDS), 'round': 'final', 'kind': 'final', 'base_round': final['round'],
                     'started_min': minutes(result.get('started')), 'ended_min': minutes(ended),
                     'reason': 'held-out', 'failure': (final.get('failure') or {}).get('kind'),
                     'failure_message': (final.get('failure') or {}).get('message'), 'passed': final.get('passed'),
                     'score': final.get('score'), 'prefill': (final.get('segments') or {}).get('prefill'),
                     'decode': (final.get('segments') or {}).get('decode'), 'cases': _cases(result)})
    return rows


def milestones(run, rows: list[dict]) -> dict:
    """When the search first had a correct engine and first matched the reference, and what it used in all."""
    search = [r for r in rows if r['kind'] != 'final']
    final = next((r for r in rows if r['kind'] == 'final'), None)
    correct = next((r for r in search if r['passed']), None)
    matched = next((r for r in search if r['passed'] and (r['score'] or 0) >= 1.0), None)

    def total(key):
        values = [r[key] for r in search if isinstance(r[key], (int, float))]
        return round(sum(values), 6) if values else None
    return {'first_correct': correct and {'round': correct['round'], 'minutes': correct['ended_min']},
            'reference_matched': matched and {'round': matched['round'], 'minutes': matched['ended_min']},
            'best_score': next((r['incumbent_score'] for r in reversed(search) if r['incumbent_score']), None),
            'final': final and {'passed': final['passed'], 'score': final['score']},
            'sessions': run.state['sessions'], 'search_minutes': search[-1]['ended_min'] if search else None,
            **{f'total_{key}': total(key) for key in agents.USAGE_KEYS},
            'agent': run.task.agent, 'agent_version': run.state.get('agent_version'),
            'agent_models': sorted({r['agent_model'] for r in search if r['agent_model']}),
            'inputs': run.state.get('inputs')}


def history_csv(rows: list[dict], lengths) -> str:
    def cell(value):
        return '' if value is None else value
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([*FIELDS, *(f'p{n}_{field}' for n in lengths for field in CASE_FIELDS)])
    for row in rows:
        cases = {case['length']: case for case in row['cases']}
        writer.writerow([cell(row[field]) for field in FIELDS]
                        + [cell(cases.get(n, {}).get(field)) for n in lengths for field in CASE_FIELDS])
    return out.getvalue()


def build(run) -> dict:
    s, t = run.state, run.task
    rows = history(run)
    best = run.evaluation(s['incumbent']) if s['incumbent'] else None
    return {'name': t.name, 'model': str(t.model), 'agent': t.agent, 'phase': s['phase'],
            'evaluation_version': s.get('evaluation_version'),
            'created': s['created'], 'deadline': s['deadline'], 'updated': time.time(),
            'search_end': s['search_end'], 'error': s['error'], 'sessions': s['sessions'],
            'current_round': s['current'], 'objective': {'prefill': t.objective.prefill, 'decode': t.objective.decode,
                                                          'lengths': {n: t.objective.case_weight(n) for n in t.lengths}},
            'incumbent': s['incumbent'],
            'best': {'round': s['incumbent'], 'score': best['score'], 'segments': best['segments'],
                     'cases': _cases(best)} if best else None,
            'final': s['final'], 'quality': _quality(run), 'milestones': milestones(run, rows), 'history': rows}


def _quality(run) -> dict | None:
    summary = run.dir / 'quality' / 'summary.json'
    if summary.is_file():
        return json.loads(summary.read_text())
    return run.state.get('quality')


def _time(value) -> str:
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(value)) if value else ''


def _x(value) -> str:
    return f'{value:.3f}x' if isinstance(value, (int, float)) else '-'


def _pct(value, error=None) -> str:
    if not isinstance(value, (int, float)):
        return '-'
    return f'{100 * value:.1f}%' + (f' ± {100 * error:.1f}' if isinstance(error, (int, float)) else '')


def _n(value) -> str:
    return '-' if value is None else f'{value:,.1f}' if isinstance(value, float) else f'{value:,}'


def _k(value) -> str:
    return '-' if value is None else f'{value / 1000:,.0f}k' if value >= 10000 else f'{value:,}'


def _money(value) -> str:
    return f'${value:.2f}' if isinstance(value, (int, float)) else ''


def render(data: dict) -> str:
    e = html.escape
    best = data['best']
    def outcome(r):
        if r['kind'] == 'final':
            return ('ok', 'passed') if r['passed'] else ('no', 'failed')
        return ('ok', 'accepted') if r['accepted'] else ('no', 'rejected')
    rows = ''.join(
        f'<tr><td>{e(r["round"])}</td><td>{e(r["kind"])}</td><td>{e(r["base_round"] or "")}</td>'
        f'<td>{_n(r["ended_min"])}</td><td class="{outcome(r)[0]}">{outcome(r)[1]}</td><td>{e(r["reason"] or "")}</td>'
        f'<td>{_x(r["score"])}</td><td>{_x(r["prefill"])}</td><td>{_x(r["decode"])}</td><td>{_n(r["session_min"])}</td>'
        f'<td>{_k(r["input_tokens"])} / {_k(r["output_tokens"])}</td><td>{_money(r["cost_usd"])}</td>'
        f'<td>{_n(r["dev_evals"])}</td><td>{_n(r["engine_lines"])} (+{_n(r["lines_added"])} '
        f'-{_n(r["lines_removed"])})</td><td>{e(r["failure_message"] or "")}</td></tr>' for r in data['history'])
    cases = ''.join(f'<tr><td>{c["length"]:,}</td><td>{_x(c["prefill_speedup"])}</td><td>{_x(c["decode_speedup"])}</td>'
                    f'<td>{_n(c["ttft_ms"])} / {_n(c["reference_ttft_ms"])}</td>'
                    f'<td>{_n(c["decode_tok_s"])} / {_n(c["reference_decode_tok_s"])}</td></tr>'
                    for c in (best or {}).get('cases', []))
    m = data['milestones']
    milestone_text = (
        f'First correct engine: {"round " + m["first_correct"]["round"] + " after " + _n(m["first_correct"]["minutes"]) + " min" if m["first_correct"] else "none yet"}. '
        f'Matched the reference (score 1.0): {"round " + m["reference_matched"]["round"] + " after " + _n(m["reference_matched"]["minutes"]) + " min" if m["reference_matched"] else "not yet"}. '
        f'Agent: {e(m["agent"])} {e(m["agent_version"] or "")}, {e(", ".join(m["agent_models"]) or "model not reported")}; '
        f'tokens {_k(m["total_input_tokens"])} in, {_k(m["total_output_tokens"])} out, {_k(m["total_cache_read_tokens"])} '
        f'cache reads; cost {_money(m["total_cost_usd"]) or "not reported"}.')
    quality = data.get('quality') or {}
    benchmarks = [name for name in ('math500', 'ifeval') if isinstance(quality.get(name), dict)]
    quality_rows = ''.join(
        f'<tr><td>{"MATH-500" if name == "math500" else "IFEval (prompt-level, strict)"}</td>'
        f'<td>{_pct(quality[name]["reference"]["accuracy"], quality[name]["reference"]["standard_error"])}</td>'
        f'<td>{_pct(quality[name]["engine"]["accuracy"], quality[name]["engine"]["standard_error"])}</td>'
        f'<td>{quality[name]["paired"]} of {quality[name]["items"]}</td>'
        f'<td>{quality[name].get("identical_outputs", "-")}</td></tr>' for name in benchmarks)
    quality_html = (f'<h2>Task accuracy</h2><p class="muted">Greedy decoding, at most '
                    f'{quality["settings"]["max_tokens"]} output tokens; reported, not gated.</p>'
                    f'<div class="scroll"><table><tr><th>benchmark</th><th>reference</th><th>engine</th>'
                    f'<th>items answered by both</th><th>identical outputs</th></tr>{quality_rows}</table></div>'
                    ) if benchmarks else (
                    f'<h2>Task accuracy</h2><p>{e(quality["error"])}</p>' if quality.get('error') else '')
    final = data['final'] or {}
    final_text = ('not run yet' if not final else final.get('skipped') or final.get('error')
                  or f'{"passed" if final.get("passed") else "failed"}, score {_x(final.get("score"))}')
    status = data['error'] or (f'search ended ({data["search_end"]})' if data['search_end'] else
                               f'searching, round {data["current_round"] or "-"}')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(data["name"])} report</title>
<style>
:root {{ --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --ok: #1a7f37; --no: #b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161617; --fg: #f5f5f7; --muted: #a1a1a6; --line: #3a3a3c;
  --ok: #4ac26b; --no: #ff7b72; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, system-ui, sans-serif;
  margin: 0 auto; max-width: 1100px; padding: 24px 16px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 28px 0 8px; }}
.muted {{ color: var(--muted); }} .ok {{ color: var(--ok); }} .no {{ color: var(--no); }}
.scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; }}
</style></head><body>
<h1>{e(data["name"])}</h1>
<div class="muted">{e(data["model"])} · agent {e(data["agent"])} · evaluation version
{e(str(data["evaluation_version"] or "unrecorded"))} · started {_time(data["created"])} · updated {_time(data["updated"])}</div>
<h2>Status</h2>
<p>{e(status)}. {data["sessions"]} coding session(s). Best engine: {e(data["incumbent"] or "none yet")}
{f"(score {_x(best['score'])}, prefill {_x(best['segments']['prefill'])}, decode {_x(best['segments']['decode'])})" if best else ""}.
Held-out final: {e(final_text)}.</p>
<p class="muted">Speedups are reference time / engine time (higher is better). Objective: prefill weight
{data["objective"]["prefill"]:g}, decode weight {data["objective"]["decode"]:g}.</p>
<p>{milestone_text}</p>
<h2>Best engine by case</h2>
<div class="scroll"><table><tr><th>prompt tokens</th><th>prefill speedup</th><th>decode speedup</th>
<th>time to first token, ms (engine / reference)</th><th>decode tok/s (engine / reference)</th></tr>{cases}</table></div>
{quality_html}
<h2>Rounds</h2>
<p class="muted">The same data, with every prompt length, is in history.csv next to this report.</p>
<div class="scroll"><table><tr><th>round</th><th>kind</th><th>from</th><th>ended, min</th><th>outcome</th><th>reason</th>
<th>score</th><th>prefill</th><th>decode</th><th>session min</th><th>tokens in / out</th><th>agent cost</th>
<th>dev-evals</th><th>engine lines</th><th>failure</th></tr>{rows}</table></div>
</body></html>
'''


def write(run) -> dict:
    data = build(run)
    (run.dir / 'report.json').write_text(json.dumps(data, indent=2, sort_keys=True))
    (run.dir / 'history.csv').write_text(history_csv(data['history'], run.task.lengths))
    (run.dir / 'report.html').write_text(render(data))
    return data
