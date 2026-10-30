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
    # DeepSeek Harness's one-shot profile; its full-access preset has no sandbox, and it takes its model from its own
    # settings (agent.model is refused for it).
    'dsh': ['dsh', '--profile', 'headless', '--json', '--permission-mode', 'danger-full-access',
            '--approval-policy', 'never', '{prompt}'],
}
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


def cost_usd(log: Path) -> float | None:
    """The agent's own report of what the session cost, from its JSON output: the sum of Pi's per-message costs, or
    Claude Code's total_cost_usd. None when the agent's output carries no cost."""
    total, found = 0.0, False
    try:
        with open(log, 'rb') as lines:
            for line in lines:
                if b'"message_end"' not in line[:64] and b'"result"' not in line[:64]:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                cost = None
                if event.get('type') == 'message_end' and (event.get('message') or {}).get('role') == 'assistant':
                    cost = ((event['message'].get('usage') or {}).get('cost') or {}).get('total')
                elif event.get('type') == 'result':
                    cost = event.get('total_cost_usd')
                if isinstance(cost, (int, float)):
                    total, found = total + cost, True
    except OSError:
        return None
    return round(total, 6) if found else None
