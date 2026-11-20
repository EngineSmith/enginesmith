"""Foreground remote command owner. Losing heartbeats expires its lease and stops its process group."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from enginesmith.processes import stop

PROTOCOL = 1


def read_messages(stream, messages):
    try:
        for line in stream:
            if len(line) > 1024 * 1024:
                messages.put({'error': 'control message too large'})
                return
            try:
                messages.put(json.loads(line))
            except ValueError:
                messages.put({'error': 'invalid control JSON'})
                return
    finally:
        messages.put(None)


def validate_start(message):
    if not isinstance(message, dict) or message.get('protocol') != PROTOCOL or message.get('op') != 'start':
        raise ValueError('expected protocol-1 start')
    token = message.get('token')
    if not isinstance(token, str) or len(token) != 64 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('invalid ownership token')
    argv = message.get('command')
    if not isinstance(argv, list) or not argv or any(not isinstance(s, str) or not s or '\0' in s for s in argv):
        raise ValueError('command must be a nonempty argv array')
    cwd = message.get('cwd')
    if not isinstance(cwd, str) or not cwd or '\0' in cwd or not os.path.isdir(cwd):
        raise ValueError('cwd must be an existing remote directory')
    env, inherited = message.get('env', {}), message.get('inherit_env', ['PATH', 'HOME'])
    def key_ok(k):
        return isinstance(k, str) and bool(k) and '=' not in k and '\0' not in k
    if not isinstance(env, dict) or any(not key_ok(k) or not isinstance(v, str) or '\0' in v for k, v in env.items()):
        raise ValueError('invalid environment')
    if not isinstance(inherited, list) or any(not key_ok(k) for k in inherited):
        raise ValueError('invalid inherited environment names')
    if len(set(inherited)) != len(inherited) or set(inherited) & set(env):
        raise ValueError('overlapping environment names')
    if any(k not in os.environ for k in inherited):
        raise ValueError('missing inherited environment variable on remote host')
    lease = message.get('lease_s')
    if type(lease) not in (int, float) or not 2 <= lease <= 300:
        raise ValueError('lease_s must be between 2 and 300')
    return argv, cwd, {**{k: os.environ[k] for k in inherited}, **env}, lease


def main():
    messages = queue.Queue()
    threading.Thread(target=read_messages, args=(sys.stdin, messages), daemon=True).start()
    child = None
    result = {'protocol': PROTOCOL, 'event': 'stopped', 'reason': None, 'stop': None}
    def emit(value):
        print(json.dumps(value), flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    code = 1
    try:
        try:
            start = messages.get(timeout=15)
        except queue.Empty:
            raise ValueError('start deadline exceeded')
        argv, cwd, env, lease = validate_start(start)
        token = start['token']
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=sys.stderr, stderr=sys.stderr, start_new_session=True)
        emit({'protocol': PROTOCOL, 'event': 'started', 'token': token, 'pid': child.pid})
        deadline = time.monotonic() + lease
        while True:
            status = child.poll()
            if status is not None:
                result['reason'] = 'command_exited'
                code = 0 if status == 0 else 1
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result['reason'] = 'lease_expired'
                break
            try:
                message = messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if message is None:
                result['reason'] = 'controller_disconnected'
                break
            if not isinstance(message, dict) or message.get('token') != token:
                result['reason'] = 'invalid_control_message'
                break
            if message.get('op') == 'heartbeat':
                deadline = time.monotonic() + lease
            elif message.get('op') == 'stop':
                result['reason'] = 'controller_stop'
                code = 0
                break
            else:
                result['reason'] = 'invalid_control_message'
                break
    except KeyboardInterrupt:
        result['reason'] = 'interrupted'
    except (ValueError, OSError, BrokenPipeError) as exc:
        result['reason'] = str(exc)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None:
            result['stop'] = stop(child, grace=0)
            if result['stop']['exit_code'] is None:
                code = 1
        try:
            emit(result)
        except (OSError, BrokenPipeError):
            pass
    return code


if __name__ == '__main__':
    sys.exit(main())
