"""The run report: report.json (facts) and report.html (the same facts for people)."""
from __future__ import annotations

import html
import json
import time


def build(run) -> dict:
    s, t = run.state, run.task
    rounds = []
    for entry in s['rounds']:
        record = run.record(entry['label']) or {}
        evaluation = record.get('evaluation') or {}
        rounds.append({**entry, 'started': record.get('started'), 'ended': record.get('ended'),
                       'session_min': round((record.get('session') or {}).get('duration_s', 0) / 60, 1),
                       'agent_cost_usd': (record.get('session') or {}).get('cost_usd'),
                       'changed_files': len(record.get('changed_files') or []),
                       'build_error': (record.get('build') or {}).get('error'),
                       'failure': evaluation.get('failure'), 'segments': evaluation.get('segments'),
                       'decision': record.get('decision')})
    best = run.evaluation(s['incumbent']) if s['incumbent'] else None
    return {'name': t.name, 'model': str(t.model), 'agent': t.agent, 'phase': s['phase'],
            'created': s['created'], 'deadline': s['deadline'], 'updated': time.time(),
            'search_end': s['search_end'], 'error': s['error'], 'sessions': s['sessions'],
            'current_round': s['current'], 'objective': {'prefill': t.objective.prefill, 'decode': t.objective.decode,
                                                          'lengths': {n: t.objective.case_weight(n) for n in t.lengths}},
            'incumbent': s['incumbent'],
            'best': {'round': s['incumbent'], 'score': best['score'], 'segments': best['segments'],
                     'cases': [{k: c[k] for k in ('id', 'length', 'prefill_speedup', 'decode_speedup')}
                               for c in best['cases']]} if best else None,
            'final': s['final'], 'quality': _quality(run), 'rounds': rounds}


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


def _money(value) -> str:
    return f'${value:.2f}' if isinstance(value, (int, float)) else ''


def render(data: dict) -> str:
    e = html.escape
    best = data['best']
    rows = ''.join(
        f'<tr><td>{e(r["label"])}</td><td>{e(r["kind"])}</td><td>{e(r["base_round"] or "")}</td>'
        f'<td class="{"ok" if r["accepted"] else "no"}">{"accepted" if r["accepted"] else "rejected"}</td>'
        f'<td>{e(r["reason"])}</td><td>{_x(r["score"])}</td>'
        f'<td>{_x((r["segments"] or {}).get("prefill"))}</td><td>{_x((r["segments"] or {}).get("decode"))}</td>'
        f'<td>{r["session_min"]}</td><td>{_money(r["agent_cost_usd"])}</td>'
        f'<td>{e((r["failure"] or {}).get("message") or r["build_error"] or "")}</td></tr>'
        for r in data['rounds'])
    cases = ''.join(f'<tr><td>{e(c["id"])}</td><td>{c["length"]:,}</td><td>{_x(c["prefill_speedup"])}</td>'
                    f'<td>{_x(c["decode_speedup"])}</td></tr>' for c in (best or {}).get('cases', []))
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
<div class="muted">{e(data["model"])} · agent {e(data["agent"])} · started {_time(data["created"])} ·
updated {_time(data["updated"])}</div>
<h2>Status</h2>
<p>{e(status)}. {data["sessions"]} coding session(s). Best engine: {e(data["incumbent"] or "none yet")}
{f"(score {_x(best['score'])}, prefill {_x(best['segments']['prefill'])}, decode {_x(best['segments']['decode'])})" if best else ""}.
Held-out final: {e(final_text)}.</p>
<p class="muted">Speedups are reference time / engine time (higher is better). Objective: prefill weight
{data["objective"]["prefill"]:g}, decode weight {data["objective"]["decode"]:g}.</p>
<h2>Best engine by case</h2>
<div class="scroll"><table><tr><th>case</th><th>prompt tokens</th><th>prefill</th><th>decode</th></tr>{cases}</table></div>
{quality_html}
<h2>Rounds</h2>
<div class="scroll"><table><tr><th>round</th><th>kind</th><th>from</th><th>outcome</th><th>reason</th><th>score</th>
<th>prefill</th><th>decode</th><th>session min</th><th>agent cost</th><th>failure</th></tr>{rows}</table></div>
</body></html>
'''


def write(run) -> dict:
    data = build(run)
    (run.dir / 'report.json').write_text(json.dumps(data, indent=2, sort_keys=True))
    (run.dir / 'report.html').write_text(render(data))
    return data
