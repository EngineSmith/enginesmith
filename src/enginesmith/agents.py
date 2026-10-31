"""Coding sessions: one fresh, headless agent process per session.

Built-in launch templates exist for Claude Code, Codex CLI, Kimi Code, Pi and DeepSeek Harness; `kind = "command"`
takes any argv.
Each runs without approval prompts and without the agent's own sandbox (the session must build and run GPU code;
docs/design.md, "Sessions", says what the light isolation relies on instead). Placeholders:
{prompt} (the session prompt), {workspace} (the session directory) and {model} (agent.model). The agent runs in its
own process group with the user's environment plus ENGINESMITH_SESSION, so it uses the user's own login or API key.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from enginesmith.processes import Registry, stop

TEMPLATES = {
    'claude': ['claude', '-p', '{prompt}', '--output-format', 'stream-json', '--verbose',
               '--permission-mode', 'bypassPermissions'],
    'codex': ['codex', 'exec', '--json', '--skip-git-repo-check', '--dangerously-bypass-approvals-and-sandbox',
              '--cd', '{workspace}', '{prompt}'],
    # Kimi Code's prompt mode asks for no approval (its "auto" permission policy) and refuses --yolo.
    'kimi': ['kimi', '-p', '{prompt}', '--output-format', 'stream-json'],
    'pi': ['pi', '--print', '--mode', 'json', '--no-session', '{prompt}'],
    # DeepSeek Harness's one-shot profile. DSH_PERMISSION_MODE=danger-full-access turns its sandbox off and its
    # approvals to never (the released 0.2.1 has no permission flags). It takes its model from its own settings, so
    # agent.model is refused for it.
    'dsh': ['env', 'DSH_PERMISSION_MODE=danger-full-access', 'dsh', '--profile', 'headless', '--json', '{prompt}'],
}
NAMES = {'claude': 'Claude Code', 'codex': 'Codex CLI', 'kimi': 'Kimi Code', 'pi': 'Pi', 'dsh': 'DeepSeek Harness'}
MODEL_FLAG = ['--model', '{model}']
# Streaming deltas (Pi's JSON mode emits one per few tokens); the complete message follows in message_end, so the
# session log keeps the whole transcript without them.
DROPPED = (b'{"type":"message_update"',)
PROMPT = ('You are the coding agent of an EngineSmith session in {workspace}. Read brief/README.md there first and '
          'follow it; it is your whole task. Work only inside that directory.')


def argv(kind: str, command, model: str | None, workspace: Path) -> list[str]:
    template = list(command) if kind == 'command' else list(TEMPLATES[kind])
    if model and kind != 'command':
        position = template.index('{prompt}') if template[-1] == '{prompt}' else len(template)
        template[position:position] = MODEL_FLAG
    values = {'{workspace}': str(workspace), '{model}': model or '', '{prompt}': PROMPT.format(workspace=workspace)}
    out = []
    for part in template:
        for placeholder, value in values.items():
            part = part.replace(placeholder, value)
        out.append(part)
    return out


# Variables that mark the process as running inside another agent's session. The run is usually started from the
# user's own agent session, and a session agent that inherits them can refuse to start (Claude Code's nested-session
# check) or attach to the parent's session state. Credentials are kept.
PARENT_SESSION = ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_CODE_SSE_PORT', 'PI_SESSION_FILE', 'PI_SESSION_ID',
                  'CODEX_SANDBOX', 'CODEX_SANDBOX_NETWORK_DISABLED', 'CODEX_THREAD_ID', 'DSH_SHELL', 'DSH_SESSION_ID',
                  'DSH_PROFILE', 'DSH_PROFILE_DIR', 'ENGINESMITH_SESSION')
# The variables each agent sets for the commands it runs, which tell which agent the user is talking to. Kimi Code
# sets none.
MARKERS = (('dsh', ('DSH_SESSION_ID', 'DSH_SHELL')), ('pi', ('PI_SESSION_ID', 'PI_SESSION_FILE')),
           ('codex', ('CODEX_THREAD_ID', 'CODEX_SANDBOX')), ('claude', ('CLAUDECODE',)))


def current() -> str | None:
    """The agent whose session runs this process (the user's own agent), or None when no marker says."""
    for kind, names in MARKERS:
        if any(os.environ.get(name) for name in names):
            return kind
    return None


def environment(session_file: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if key not in PARENT_SESSION}
    env['ENGINESMITH_SESSION'] = str(session_file)
    env['PATH'] = os.pathsep.join([str(Path(sys.executable).parent), env.get('PATH', '')])
    return env


def run_session(command, *, cwd: Path, log: Path, timeout: float, registry: Registry, stop_requested,
                env: dict, on_start=None) -> dict:
    """Run one agent session to completion, its timeout or a stop request; afterwards stop its process group and any
    engine it left running."""
    started = time.monotonic()
    with open(log, 'ab') as out:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, env=env, start_new_session=True)
        except OSError as exc:
            return {'exit_code': None, 'error': f'cannot start {command[0]}: {exc}', 'timed_out': False,
                    'stopped': False, 'duration_s': 0.0, 'leaked_engines': []}
        copier = threading.Thread(target=_copy, args=(process.stdout, out), daemon=True)
        copier.start()
        if on_start:
            on_start(process.pid)
        timed_out = stopped = False
        while process.poll() is None:
            if time.monotonic() - started > timeout:
                timed_out = True
                break
            if stop_requested():
                stopped = True
                break
            time.sleep(1.0)
        ended = stop(process, grace=0 if (timed_out or stopped) else 5)
        copier.join(timeout=10)
    return {'exit_code': ended['exit_code'], 'signals': ended['signals'], 'timed_out': timed_out,
            'stopped': stopped, 'duration_s': time.monotonic() - started, 'leaked_engines': registry.kill_all()}


def _copy(source, out) -> None:
    """Copy the agent's output to the session log line by line, without streaming deltas."""
    for line in iter(source.readline, b''):
        if not line.startswith(DROPPED):
            out.write(line)
            out.flush()
    source.close()


USAGE_KEYS = ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'cost_usd')


_SESSION_ID = re.compile(r'[A-Za-z0-9-]{8,80}\Z')


def _last(path: Path, marker: bytes, pick) -> dict | str | None:
    """pick(record) of the last JSON line of a file that contains marker and gives a value."""
    value = None
    try:
        with open(path, 'rb') as lines:
            for line in lines:
                if marker in line:
                    try:
                        value = pick(json.loads(line)) or value
                    except (ValueError, AttributeError):
                        pass
    except OSError:
        pass
    return value


def _codex_turn(thread: str) -> dict:
    """The settings of a Codex session's last turn (model, effort), from its own session file."""
    home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
    for path in home.glob(f'sessions/*/*/*/rollout-*-{thread}.jsonl') if _SESSION_ID.match(thread) else ():
        return _last(path, b'"turn_context"', lambda r: r.get('type') == 'turn_context' and r.get('payload')) or {}
    return {}


def _claude_effort(session: str) -> str | None:
    """The reasoning effort Claude Code recorded in the session's transcript."""
    home = Path(os.environ.get('CLAUDE_CONFIG_DIR') or Path.home() / '.claude')
    for path in home.glob(f'projects/*/{session}.jsonl') if _SESSION_ID.match(session) else ():
        effort = _last(path, b'"effort"', lambda r: r.get('effort'))
        return effort if isinstance(effort, str) else None
    return None


def usage(log: Path) -> dict:
    """What the session's agent reported about itself: the model, the reasoning effort, the tokens (input excludes
    cache reads) and the cost. A value the agent does not report is None. Read from the JSON output: Pi (each
    assistant message_end, with the model and thinking level), DeepSeek Harness (each step_end status; no model or
    effort), Claude Code (the init event's model and session; the result event's totals) and Codex (each
    turn.completed). Claude Code's effort and Codex's model and effort come from the agent's own session file, which
    the output names. Kimi Code's output is not parsed."""
    totals, found, model, effort = dict.fromkeys(USAGE_KEYS, 0), set(), None, None
    claude_session = codex_thread = None

    def add(key, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] += value
            found.add(key)
    try:
        with open(log, 'rb') as lines:
            for line in lines:
                if not line.startswith(b'{'):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get('type')
                if kind == 'message_end' and (event.get('message') or {}).get('role') == 'assistant':
                    message = event['message']
                    model = message.get('model') or model
                    effort = message.get('thinkingLevel') or effort
                    u = message.get('usage') or {}
                    add('input_tokens', u.get('input'))
                    add('output_tokens', u.get('output'))
                    add('cache_read_tokens', u.get('cacheRead'))
                    add('cache_write_tokens', u.get('cacheWrite'))
                    add('cost_usd', (u.get('cost') or {}).get('total'))
                elif kind == 'status' and event.get('phase') == 'step_end':
                    u = event.get('usage') or {}
                    add('input_tokens', u.get('inputTokens'))
                    add('output_tokens', u.get('outputTokens'))
                    add('cache_read_tokens', u.get('cacheReadTokens'))
                    add('cache_write_tokens', u.get('cacheWriteTokens'))
                elif kind == 'system' and event.get('subtype') == 'init':
                    model = event.get('model') or model
                    claude_session = event.get('session_id') or claude_session
                elif kind == 'result':
                    u = event.get('usage') or {}
                    add('input_tokens', u.get('input_tokens'))
                    add('output_tokens', u.get('output_tokens'))
                    add('cache_read_tokens', u.get('cache_read_input_tokens'))
                    add('cache_write_tokens', u.get('cache_creation_input_tokens'))
                    add('cost_usd', event.get('total_cost_usd'))
                elif kind == 'thread.started':
                    codex_thread = event.get('thread_id') or codex_thread
                elif kind == 'turn.completed':
                    u = event.get('usage') or {}
                    cached = u.get('cached_input_tokens') or 0
                    add('input_tokens', u['input_tokens'] - cached if isinstance(u.get('input_tokens'), int) else None)
                    add('cache_read_tokens', u.get('cached_input_tokens'))
                    add('output_tokens', u.get('output_tokens'))
    except OSError:
        pass
    if isinstance(codex_thread, str):
        turn = _codex_turn(codex_thread)
        model, effort = turn.get('model') or model, turn.get('effort') or effort
    if isinstance(claude_session, str):
        effort = _claude_effort(claude_session) or effort
    out = {key: (round(totals[key], 6) if key == 'cost_usd' else totals[key]) if key in found else None
           for key in USAGE_KEYS}
    return {'model': model, 'effort': effort, **out}


def version(kind: str) -> str | None:
    """The agent CLI's own version line (`CLI --version`), or None for a custom command or when it does not say."""
    if kind not in TEMPLATES:
        return None
    try:
        result = subprocess.run([kind, '--version'], capture_output=True, text=True, timeout=60,
                                stdin=subprocess.DEVNULL)  # a built-in kind is its CLI's name
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0][:200] if result.returncode == 0 and lines else None
