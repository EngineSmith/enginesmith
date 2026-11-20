"""SSH controller for one lease-owned foreground process group. No resource allocation or scoring."""
from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shlex
import signal
import subprocess
import threading
import time
import tomllib
from pathlib import Path

from enginesmith.processes import stop
from enginesmith.remote_worker import PROTOCOL, read_messages
from enginesmith.services import _argv, _interruptions, _keys, _seconds
from enginesmith.task import TaskError


def plan(path: Path) -> dict:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f'cannot read remote manifest: {exc}') from exc
    _keys(data, ('version', 'host', 'python', 'command', 'cwd', 'env', 'inherit_env', 'lease_s', 'timeout_s'),
          'remote manifest')
    if type(data.get('version')) is not int or data['version'] != 1:
        raise TaskError('remote manifest version must be 1')
    host = data.get('host')
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', host):
        raise TaskError('host must be an SSH configuration alias; configure user, port and key in SSH config')
    python = data.get('python', 'python3')
    cwd = data.get('cwd', '.')
    for value, name in ((python, 'python'), (cwd, 'cwd')):
        if not isinstance(value, str) or not value or '\0' in value or '\n' in value:
            raise TaskError(f'{name} must be a nonempty string without newlines')
    lease = _seconds(data.get('lease_s', 15), 'lease_s')
    if not 2 <= lease <= 300:
        raise TaskError('lease_s must be between 2 and 300')
    env, inherited = data.get('env', {}), data.get('inherit_env', ['PATH', 'HOME'])
    valid_key = lambda k: isinstance(k, str) and bool(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k))
    if not isinstance(env, dict) or any(not valid_key(k) or not isinstance(v, str) or '\0' in v
                                        for k, v in env.items()):
        raise TaskError('env must map valid variable names to strings')
    if not isinstance(inherited, list) or any(not valid_key(k) for k in inherited):
        raise TaskError('inherit_env must be an array of variable names')
    if len(set(inherited)) != len(inherited) or set(inherited) & set(env):
        raise TaskError('overlapping environment names')
    return {'ssh': ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
                    '-o', 'ServerAliveCountMax=2', host, shlex.join([python, '-m', 'enginesmith.remote_worker'])],
            'start': {'protocol': PROTOCOL, 'op': 'start', 'command': _argv(data.get('command'), 'command'),
                      'cwd': cwd, 'env': env, 'inherit_env': inherited, 'lease_s': lease},
            'timeout_s': _seconds(data.get('timeout_s', 86400), 'timeout_s')}


def run(path: Path, out: Path) -> dict:
    config = plan(path)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise TaskError(f'cannot create fresh remote output directory: {exc}') from exc
    token = secrets.token_hex(32)
    result = {'schema': 1, 'kind': 'remote_diagnostic', 'configuration': config, 'success': False,
              'failure': None, 'remote_cleanup_confirmed': False, 'events': [], 'ssh_stop': None}
    messages = queue.Queue()
    process = None
    started = time.monotonic()
    def send(message):
        process.stdin.write(json.dumps({**message, 'token': token}) + '\n')
        process.stdin.flush()
    def record(message):
        if not isinstance(message, dict):
            raise ValueError('invalid remote event')
        if message.get('event') == 'started' and message.get('token') != token:
            raise ValueError('remote ownership handshake mismatch')
        sanitized = {k: v for k, v in message.items() if k != 'token'}
        result['events'].append(sanitized)
        if message.get('event') == 'stopped':
            result['remote_cleanup_confirmed'] = message.get('stop') is not None and message['stop'].get('exit_code') is not None
    with _interruptions():
        try:
            with (out / 'remote.log').open('wb') as log:
                process = subprocess.Popen(config['ssh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=log, text=True, start_new_session=True)
            threading.Thread(target=read_messages, args=(process.stdout, messages), daemon=True).start()
            send(config['start'])
            heartbeat = time.monotonic()
            while True:
                now = time.monotonic()
                if now - started > config['timeout_s']:
                    raise TimeoutError('remote command exceeded timeout_s')
                if now >= heartbeat:
                    send({'op': 'heartbeat'})
                    heartbeat = now + config['start']['lease_s'] / 3
                try:
                    message = messages.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None:
                        raise OSError('SSH exited before a terminal remote event')
                    continue
                if message is None:
                    raise OSError('SSH control channel closed without a terminal event')
                record(message)
                if message.get('event') == 'stopped':
                    result['success'] = (message.get('reason') == 'command_exited'
                                         and result['remote_cleanup_confirmed'] and message['stop']['exit_code'] == 0)
                    if not result['success']:
                        result['failure'] = message.get('reason', 'remote failure')
                    break
        except KeyboardInterrupt:
            result['failure'] = 'interrupted'
        except (OSError, ValueError, TimeoutError) as exc:
            result['failure'] = str(exc)
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if process is not None:
                if not result['remote_cleanup_confirmed']:
                    try:
                        send({'op': 'stop'})
                    except (OSError, ValueError):
                        pass
                    # Stop heartbeats; await an actual terminal event. No confirmation is not success.
                    deadline = time.monotonic() + config['start']['lease_s'] + 11
                    while time.monotonic() < deadline and not result['remote_cleanup_confirmed']:
                        try:
                            message = messages.get(timeout=0.1)
                        except queue.Empty:
                            if process.poll() is not None:
                                break
                            continue
                        if message is None:
                            break
                        try:
                            record(message)
                        except ValueError:
                            break
                result['ssh_stop'] = stop(process, grace=1)
                process.stdin.close()
                process.stdout.close()
            if not result['remote_cleanup_confirmed']:
                result['success'] = False
            result['duration_s'] = time.monotonic() - started
            temporary = out / 'result.json.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(out / 'result.json')
    return result
