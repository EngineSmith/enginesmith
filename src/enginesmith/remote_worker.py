"""Foreground remote command owner. Losing heartbeats expires its lease and stops its process group."""
from __future__ import annotations

import json
import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time

from enginesmith.processes import stop

PROTOCOL = 1
MAX_MESSAGE_BYTES = 1024 * 1024
MESSAGE_QUEUE_SIZE = 4
START_TIMEOUT_S = 15
STOP_REASONS = frozenset(('command_exited', 'lease_expired', 'controller_disconnected',
                          'invalid_control_message', 'controller_stop', 'interrupted', 'worker_error'))


def read_messages(stream, messages, closed=None):
    """Read bounded UTF-8 JSON lines from a binary stream, with bounded queue backpressure."""
    def put(message):
        while closed is None or not closed.is_set():
            try:
                messages.put(message, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('non-finite JSON number')

    try:
        fd = stream.fileno()
    except (OSError, AttributeError):
        fd = None
    pending = bytearray()

    def readline():
        if fd is None:
            return stream.readline(MAX_MESSAGE_BYTES + 1)
        while closed is None or not closed.is_set():
            newline = pending.find(b'\n')
            if newline >= 0:
                line = bytes(pending[:newline + 1])
                del pending[:newline + 1]
                return line
            if len(pending) > MAX_MESSAGE_BYTES:
                return bytes(pending)
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            chunk = os.read(fd, min(65536, MAX_MESSAGE_BYTES + 1 - len(pending)))
            if not chunk:
                line = bytes(pending)
                pending.clear()
                return line
            pending.extend(chunk)
        return b''

    try:
        while closed is None or not closed.is_set():
            # The limit applies before allocation, including for an unterminated line.
            line = readline()
            if not line:
                return
            if len(line) > MAX_MESSAGE_BYTES:
                put({'error': 'control message too large'})
                return
            if not line.endswith(b'\n'):
                put({'error': 'unterminated control message'})
                return
            message = json.loads(line.decode('utf-8'), object_pairs_hook=object_pairs,
                                 parse_constant=invalid_constant)
            if not isinstance(message, dict):
                raise ValueError('control message must be an object')
            if not put(message):
                return
    except (ValueError, OSError, RecursionError):
        put({'error': 'invalid control JSON or stream'})
    finally:
        put(None)


def validate_start(message):
    if (not isinstance(message, dict) or type(message.get('protocol')) is not int
            or message['protocol'] != PROTOCOL or message.get('op') != 'start'
            or set(message) - {'protocol', 'op', 'token', 'command', 'cwd', 'env', 'inherit_env', 'lease_s'}):
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
    messages = queue.Queue(maxsize=MESSAGE_QUEUE_SIZE)
    closed = threading.Event()
    reader = threading.Thread(target=read_messages, args=(sys.stdin.buffer, messages, closed), daemon=True)
    reader.start()
    child = None
    result = {'protocol': PROTOCOL, 'event': 'stopped', 'token': None, 'reason': None, 'stop': None}
    def emit(value):
        print(json.dumps(value), flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    code = 1
    try:
        try:
            start = messages.get(timeout=START_TIMEOUT_S)
        except queue.Empty:
            raise ValueError('start deadline exceeded')
        if isinstance(start, dict):
            candidate = start.get('token')
            if isinstance(candidate, str) and len(candidate) == 64 and all(c in '0123456789abcdef' for c in candidate):
                result['token'] = candidate
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
            if (not isinstance(message, dict) or set(message) != {'op', 'token'}
                    or message.get('token') != token):
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
    except (ValueError, OSError) as exc:
        result['reason'] = 'start_failed' if child is None else 'worker_error'
        try:
            print(f"{result['reason']}: {exc}", file=sys.stderr, flush=True)
        except OSError:
            pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        closed.set()
        reader.join(timeout=1)
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
