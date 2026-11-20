"""SSH controller for one lease-owned foreground process group. No resource allocation or scoring."""
from __future__ import annotations

import json
import os
import queue
import re
import secrets
import select
import shlex
import signal
import subprocess
import threading
import time
import tomllib
from pathlib import Path

from enginesmith.processes import stop
from enginesmith.remote_worker import MAX_MESSAGE_BYTES, MESSAGE_QUEUE_SIZE, PROTOCOL, STOP_REASONS, read_messages
from enginesmith.services import _argv, _interruptions, _keys, _seconds
from enginesmith.task import TaskError


def plan(path: Path) -> dict:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f'cannot read remote manifest: {exc}') from exc
    _keys(data, ('version', 'host', 'python', 'command', 'cwd', 'env', 'inherit_env', 'lease_s', 'timeout_s', 'startup_timeout_s'),
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
            'timeout_s': _seconds(data.get('timeout_s', 86400), 'timeout_s'),
            'startup_timeout_s': _seconds(data.get('startup_timeout_s', 30), 'startup_timeout_s')}


class _Events:
    """Fail closed on malformed, foreign or out-of-order worker messages."""

    def __init__(self, token, result):
        self.token, self.result = token, result
        self.state = 'waiting'
        self.invalid = False

    def record(self, message):
        try:
            self._record(message)
        except ValueError:
            self.invalid = True
            self.result['remote_cleanup_confirmed'] = False
            raise

    def _record(self, message):
        if self.invalid or self.state == 'stopped':
            raise ValueError('unexpected event after protocol failure or terminal event')
        if (not isinstance(message, dict) or type(message.get('protocol')) is not int
                or message['protocol'] != PROTOCOL):
            raise ValueError('invalid remote protocol')
        if message.get('token') != self.token:
            raise ValueError('remote ownership token mismatch')
        event = message.get('event')
        if event == 'started':
            if (self.state != 'waiting' or set(message) != {'protocol', 'event', 'token', 'pid'}
                    or type(message['pid']) is not int or message['pid'] <= 0):
                raise ValueError('invalid or out-of-order started event')
            self.state = 'running'
        elif event == 'stopped':
            if set(message) != {'protocol', 'event', 'token', 'reason', 'stop'}:
                raise ValueError('invalid stopped event schema')
            reason, stopped = message['reason'], message['stop']
            if self.state == 'waiting':
                if reason != 'start_failed' or stopped is not None:
                    raise ValueError('terminal event received before ownership handshake')
            else:
                if (not isinstance(reason, str) or reason not in STOP_REASONS
                        or not isinstance(stopped, dict) or set(stopped) != {'exit_code', 'signals'}):
                    raise ValueError('invalid stopped event reason or cleanup schema')
                code, signals = stopped['exit_code'], stopped['signals']
                if (code is not None and type(code) is not int
                        or not isinstance(signals, list)
                        or any(s not in ('SIGTERM', 'SIGKILL') for s in signals)
                        or signals not in ([], ['SIGTERM'], ['SIGKILL'], ['SIGTERM', 'SIGKILL'])
                        or reason == 'command_exited' and code is None):
                    raise ValueError('invalid remote exit status or cleanup signals')
                self.result['remote_cleanup_confirmed'] = code is not None
            self.state = 'stopped'
        else:
            raise ValueError('unknown remote event')
        self.result['events'].append({k: v for k, v in message.items() if k != 'token'})


def run(path: Path, out: Path) -> dict:
    config = plan(path)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise TaskError(f'cannot create fresh remote output directory: {exc}') from exc
    token = secrets.token_hex(32)
    result = {'schema': 1, 'kind': 'remote_diagnostic', 'configuration': config, 'success': False,
              'failure': None, 'remote_cleanup_confirmed': False, 'events': [], 'ssh_stop': None}
    messages = queue.Queue(maxsize=MESSAGE_QUEUE_SIZE)
    closed = threading.Event()
    events = _Events(token, result)
    process = None
    reader = None
    started = time.monotonic()
    command_deadline = started + config['timeout_s']
    startup_deadline = min(command_deadline, started + config['startup_timeout_s'])
    terminal_deadline = None

    def send(message, deadline):
        wire = (json.dumps({**message, 'token': token}, allow_nan=False) + '\n').encode('utf-8')
        if len(wire) > MAX_MESSAGE_BYTES:
            raise ValueError('outgoing control message too large')
        # A stalled SSH process must not block the controller in a pipe write.
        remaining = memoryview(wire)
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError('remote control write deadline exceeded')
            if not select.select([], [process.stdin], [], min(timeout, 0.1))[1]:
                continue
            try:
                count = os.write(process.stdin.fileno(), remaining)
            except BlockingIOError:
                continue
            remaining = remaining[count:]
    with _interruptions():
        try:
            with (out / 'remote.log').open('wb') as log:
                process = subprocess.Popen(config['ssh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=log, start_new_session=True)
            os.set_blocking(process.stdin.fileno(), False)
            reader = threading.Thread(target=read_messages, args=(process.stdout, messages, closed), daemon=True)
            reader.start()
            send(config['start'], startup_deadline)
            heartbeat = time.monotonic() + config['start']['lease_s'] / 3
            while True:
                now = time.monotonic()
                if events.state == 'waiting' and now >= startup_deadline:
                    raise TimeoutError('remote ownership handshake exceeded startup_timeout_s')
                if terminal_deadline is not None and now >= terminal_deadline:
                    raise TimeoutError('SSH did not close after terminal remote event')
                if now >= command_deadline:
                    raise TimeoutError('remote command exceeded timeout_s')
                if events.state == 'running' and now >= heartbeat:
                    try:
                        send({'op': 'heartbeat'}, min(command_deadline, now + config['start']['lease_s'] / 3))
                        heartbeat = now + config['start']['lease_s'] / 3
                    except BrokenPipeError:
                        # The worker may have exited with its terminal event still queued.
                        heartbeat = float('inf')
                try:
                    message = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if message is None:
                    if events.state != 'stopped':
                        raise OSError('SSH control channel closed without a terminal event')
                    try:
                        status = process.wait(timeout=max(0.01, min(command_deadline, terminal_deadline) - now))
                    except subprocess.TimeoutExpired as exc:
                        raise TimeoutError('SSH did not exit after terminal remote event') from exc
                    terminal = result['events'][-1]
                    result['success'] = (terminal['reason'] == 'command_exited'
                                         and result['remote_cleanup_confirmed'] and terminal['stop']['exit_code'] == 0
                                         and status == 0)
                    if not result['success']:
                        result['failure'] = (f'SSH exited with status {status}' if status and terminal['reason'] == 'command_exited'
                                             and terminal['stop']['exit_code'] == 0 else terminal['reason'])
                    break
                events.record(message)
                if events.state == 'stopped':
                    terminal_deadline = time.monotonic() + 2
        except KeyboardInterrupt:
            result['failure'] = 'interrupted'
        except (OSError, ValueError, TimeoutError) as exc:
            result['failure'] = str(exc)
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if process is not None:
                if events.state != 'stopped':
                    try:
                        send({'op': 'stop'}, time.monotonic() + 1)
                    except (OSError, ValueError):
                        pass
                    process.stdin.close()
                    # Stop heartbeats; await an actual terminal event. No confirmation is not success.
                    # An unacknowledged/invalid peer cannot earn a fresh, long cleanup wait.
                    grace = config['start']['lease_s'] + 11 if events.state == 'running' and not events.invalid else 1
                    deadline = time.monotonic() + grace
                    while time.monotonic() < deadline and events.state != 'stopped' and not events.invalid:
                        try:
                            message = messages.get(timeout=0.1)
                        except queue.Empty:
                            if process.poll() is not None:
                                break
                            continue
                        if message is None:
                            break
                        try:
                            events.record(message)
                        except ValueError:
                            break
                closed.set()
                result['ssh_stop'] = stop(process, grace=1)
                process.stdin.close()
                if reader is not None:
                    reader.join(timeout=1)
                # Do not wait on a reader lock if an escaped descendant holds the pipe open.
                if reader is None or not reader.is_alive():
                    process.stdout.close()
            if not result['remote_cleanup_confirmed']:
                result['success'] = False
            result['duration_s'] = time.monotonic() - started
            temporary = out / 'result.json.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(out / 'result.json')
    return result
