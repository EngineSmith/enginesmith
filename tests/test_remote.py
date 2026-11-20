import json
import io
import os
import queue
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path

import pytest

from enginesmith.remote import _Events, plan, run
from enginesmith.remote_worker import MAX_MESSAGE_BYTES, read_messages, validate_start
from enginesmith.task import TaskError

TOKEN = 'a' * 64


def start(tmp_path, command=None, **extra):
    return {'protocol': 1, 'op': 'start', 'token': TOKEN, 'cwd': str(tmp_path),
            'command': command or [sys.executable, '-c', 'import time; time.sleep(60)'],
            'env': {}, 'inherit_env': ['PATH'], 'lease_s': 2, **extra}


def worker():
    return subprocess.Popen([sys.executable, '-m', 'enginesmith.remote_worker'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)


def send(process, message):
    process.stdin.write(json.dumps(message) + '\n')
    process.stdin.flush()


def absent(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize('mode,reason', [('eof', 'controller_disconnected'), ('silent', 'lease_expired'),
                                        ('stop', 'controller_stop'), ('bad-token', 'invalid_control_message')])
def test_worker_lease_and_ownership(tmp_path, mode, reason):
    process = worker()
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        send(process, start(tmp_path))
        ready = json.loads(process.stdout.readline())
        assert ready['event'] == 'started' and ready['token'] == TOKEN
        if mode == 'eof':
            process.stdin.close()
            process.stdin = None
        elif mode in ('stop', 'bad-token'):
            send(process, {'op': 'stop', 'token': TOKEN if mode == 'stop' else 'b' * 64})
        stdout, stderr = process.communicate(timeout=8) if mode != 'silent' else wait_without_eof(process)
        terminal = json.loads(stdout)
        assert terminal['reason'] == reason, stderr
        assert absent(ready['pid'])
        assert unrelated.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
        unrelated.terminate()
        unrelated.wait()


def wait_without_eof(process):
    process.wait(timeout=8)
    return process.stdout.read(), process.stderr.read()


def test_heartbeat_extends_lease_and_normal_command_exit_is_retained(tmp_path):
    process = worker()
    try:
        send(process, start(tmp_path, [sys.executable, '-c', 'import time; time.sleep(3); print("done")']))
        assert json.loads(process.stdout.readline())['event'] == 'started'
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            try:
                send(process, {'op': 'heartbeat', 'token': TOKEN})
            except BrokenPipeError:
                break
            time.sleep(0.3)
        process.wait(timeout=5)
        terminal = json.loads(process.stdout.read())
        assert terminal['reason'] == 'command_exited' and terminal['stop']['exit_code'] == 0
        assert process.stderr.read().strip() == 'done'
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.parametrize('change', [{'lease_s': 0}, {'lease_s': float('nan')}, {'lease_s': True},
                                  {'token': 'x'}, {'command': 'echo bad'}, {'cwd': 'missing'},
                                  {'env': {'BAD=NAME': 'x'}}])
def test_worker_rejects_bad_start_before_launch(tmp_path, change):
    with pytest.raises(ValueError):
        validate_start(start(tmp_path, **change))


def manifest(tmp_path, **changes):
    values = {'version': 1, 'host': 'worker-a', 'python': sys.executable,
              'command': [sys.executable, '-c', 'print("remote")'], 'cwd': str(tmp_path),
              'lease_s': 2, 'timeout_s': 10, **changes}
    path = tmp_path / 'remote.toml'
    def toml_value(value):
        if isinstance(value, dict):
            return '{' + ', '.join(f'{json.dumps(k)} = {json.dumps(v)}' for k, v in value.items()) + '}'
        return json.dumps(value)
    path.write_text('\n'.join(f'{k} = {toml_value(v)}' for k, v in values.items()))
    return path


def fake_ssh(tmp_path, monkeypatch, body=None):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    script = bindir / 'ssh'
    script.write_text(f'#!{sys.executable}\n' + (body or 'import os,sys,shlex\n'
                      'args=shlex.split(sys.argv[-1]); os.execv(args[0],args)\n'))
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])


def test_controller_over_a_local_fake_ssh_transport(tmp_path, monkeypatch):
    # Exercise stdin/control lifecycle without claiming real SSH or multi-node evidence.
    fake_ssh(tmp_path, monkeypatch)
    result = run(manifest(tmp_path), tmp_path / 'out')
    assert result['success'] and result['remote_cleanup_confirmed']
    assert (tmp_path / 'out' / 'remote.log').read_text().strip() == 'remote'
    assert all('token' not in e for e in result['events'])
    assert json.loads((tmp_path / 'out' / 'result.json').read_text()) == result


def test_shell_metacharacters_do_not_escape_interpreter_argument(tmp_path):
    import shlex
    value = 'python with spaces;touch should-not-exist'
    result = plan(manifest(tmp_path, python=value))
    assert shlex.split(result['ssh'][-1]) == [value, '-m', 'enginesmith.remote_worker']
    assert not (tmp_path / 'should-not-exist').exists()


@pytest.mark.parametrize('host', ['-oProxyCommand=x', 'host;echo x', 'user@host', 'host\nother'])
def test_requires_ssh_alias(tmp_path, host):
    with pytest.raises(TaskError):
        plan(manifest(tmp_path, host=host))


@pytest.mark.parametrize('value', [0, -1, True, '30'])
def test_startup_timeout_must_be_positive_seconds(tmp_path, value):
    with pytest.raises(TaskError):
        plan(manifest(tmp_path, startup_timeout_s=value))


def test_reader_limits_unterminated_line_before_allocation():
    class LimitedStream(io.BytesIO):
        def readline(self, size=-1):
            assert size == MAX_MESSAGE_BYTES + 1
            return super().readline(size)
    stream = LimitedStream(b'x' * (MAX_MESSAGE_BYTES + 100))
    messages = queue.Queue()
    read_messages(stream, messages)
    assert messages.get_nowait() == {'error': 'control message too large'}
    assert messages.get_nowait() is None
    assert stream.tell() == MAX_MESSAGE_BYTES + 1


@pytest.mark.parametrize('wire', [b'{}', b'{broken}\n', b'{"a":1,"a":2}\n', b'NaN\n',
                                  b'Infinity\n', b'"\xff"\n', b'[' * 2000 + b']' * 2000 + b'\n'])
def test_reader_rejects_malformed_frames(wire):
    messages = queue.Queue()
    read_messages(io.BytesIO(wire), messages)
    assert 'error' in messages.get_nowait()
    assert messages.get_nowait() is None


def test_reader_backpressure_can_be_cancelled():
    messages, closed = queue.Queue(maxsize=1), threading.Event()
    reader = threading.Thread(target=read_messages, args=(io.BytesIO(b'{}\n' * 20), messages, closed))
    reader.start()
    try:
        assert messages.get(timeout=2) == {}
        closed.set()
        reader.join(timeout=2)
        assert not reader.is_alive()
        assert messages.qsize() <= 1
    finally:
        closed.set()
        reader.join(timeout=2)


def test_pipe_reader_handles_fragmented_utf8_and_multiple_lines():
    read_fd, write_fd = os.pipe()
    messages, closed = queue.Queue(maxsize=4), threading.Event()
    with os.fdopen(read_fd, 'rb') as stream, os.fdopen(write_fd, 'wb', buffering=0) as writer:
        reader = threading.Thread(target=read_messages, args=(stream, messages, closed))
        reader.start()
        try:
            writer.write(b'{"message":"\xc3')
            writer.write(b'\xa9"}\n{"next":true}\n')
            assert messages.get(timeout=2) == {'message': '\u00e9'}
            assert messages.get(timeout=2) == {'next': True}
            # Keep the writer open: cancellation must not depend on receiving EOF.
            closed.set()
            reader.join(timeout=2)
            assert not reader.is_alive()
        finally:
            closed.set()
            reader.join(timeout=2)


def test_worker_initial_start_deadline_does_not_wait_for_stdin_eof():
    process = subprocess.Popen([sys.executable, '-c',
                                'import enginesmith.remote_worker as w; w.START_TIMEOUT_S=0.2; raise SystemExit(w.main())'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.wait(timeout=5)
        event = json.loads(process.stdout.read())
        assert event['reason'] == 'start_failed' and event['stop'] is None
        assert event['token'] is None and process.returncode == 1
        assert 'Fatal Python error' not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        process.stdin.close()


def ready(**changes):
    return {'protocol': 1, 'event': 'started', 'token': TOKEN, 'pid': 123, **changes}


def terminal(**changes):
    return {'protocol': 1, 'event': 'stopped', 'token': TOKEN, 'reason': 'command_exited',
            'stop': {'exit_code': 0, 'signals': []}, **changes}


@pytest.mark.parametrize('events', [
    [terminal()], [ready(protocol=True)], [ready(protocol=2)], [ready(token='b' * 64)],
    [ready(pid=True)], [ready(pid=0)], [ready(event='unknown')], [ready(extra=1)],
    [ready(), ready()], [ready(), terminal(token='b' * 64)], [ready(), terminal(protocol=2)],
    [ready(), terminal(stop=[])], [ready(), terminal(stop={})],
    [ready(), terminal(stop={'exit_code': False, 'signals': []})],
    [ready(), terminal(stop={'exit_code': '0', 'signals': []})],
    [ready(), terminal(stop={'exit_code': None, 'signals': []})],
    [ready(), terminal(stop={'exit_code': 0, 'signals': ['SIGUSR1']})],
    [ready(), terminal(stop={'exit_code': 0, 'signals': ['SIGKILL', 'SIGTERM']})],
    [ready(), terminal(reason=[])], [ready(), terminal(reason='made_up')],
    [ready(), terminal(reason='start_failed')], [ready(), terminal(extra='untrusted')],
    [ready(), terminal(), terminal()], [ready(), terminal(), ready()], [[], 1],
])
def test_events_fail_closed_for_invalid_schema_or_order(events):
    result = {'events': [], 'remote_cleanup_confirmed': False}
    protocol = _Events(TOKEN, result)
    with pytest.raises(ValueError):
        for event in events:
            protocol.record(event)
    assert not result['remote_cleanup_confirmed']
    with pytest.raises(ValueError):
        protocol.record(terminal())
    assert not result['remote_cleanup_confirmed']


def test_start_failure_is_terminal_but_not_cleanup_confirmation():
    result = {'events': [], 'remote_cleanup_confirmed': False}
    protocol = _Events(TOKEN, result)
    protocol.record(terminal(reason='start_failed', stop=None))
    assert protocol.state == 'stopped'
    assert not result['remote_cleanup_confirmed']
    assert all('token' not in event for event in result['events'])


@pytest.mark.parametrize('code,confirmed', [(-9, True), (None, False)])
def test_terminal_preserves_incomplete_cleanup(code, confirmed):
    result = {'events': [], 'remote_cleanup_confirmed': False}
    protocol = _Events(TOKEN, result)
    protocol.record(ready())
    protocol.record(terminal(reason='controller_stop', stop={'exit_code': code, 'signals': ['SIGKILL']}))
    assert result['remote_cleanup_confirmed'] is confirmed


def test_controller_preserves_nonzero_remote_exit(tmp_path, monkeypatch):
    fake_ssh(tmp_path, monkeypatch)
    result = run(manifest(tmp_path, command=[sys.executable, '-c', 'raise SystemExit(7)']), tmp_path / 'out')
    assert not result['success'] and result['remote_cleanup_confirmed']
    assert result['events'][-1]['stop']['exit_code'] == 7
    assert result['ssh_stop']['exit_code'] == 1


def test_controller_reports_failed_launch_without_claiming_cleanup(tmp_path, monkeypatch):
    fake_ssh(tmp_path, monkeypatch)
    result = run(manifest(tmp_path, command=['./missing-executable']), tmp_path / 'out')
    assert result['failure'] == 'start_failed'
    assert not result['remote_cleanup_confirmed'] and not result['success']
    assert len(result['events']) == 1


@pytest.mark.parametrize('large_start', [False, True])
def test_unresponsive_ssh_has_bounded_startup_even_with_blocked_stdin(tmp_path, monkeypatch, large_start):
    fake_ssh(tmp_path, monkeypatch, 'import time; time.sleep(60)\n')
    changes = {'env': {'LARGE_VALUE': 'x' * 900_000}} if large_start else {}
    result = run(manifest(tmp_path, startup_timeout_s=0.2, timeout_s=60, **changes), tmp_path / 'out')
    assert ('deadline' if large_start else 'startup_timeout_s') in result['failure']
    assert not result['success'] and not result['remote_cleanup_confirmed']
    assert result['duration_s'] < 6
    assert result['ssh_stop']['exit_code'] is not None


def test_command_timeout_cleans_up_owned_child(tmp_path, monkeypatch):
    fake_ssh(tmp_path, monkeypatch)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        result = run(manifest(tmp_path, timeout_s=0.6,
                              command=[sys.executable, '-c', 'import time; time.sleep(60)']), tmp_path / 'out')
        assert result['failure'] == 'remote command exceeded timeout_s'
        assert not result['success'] and result['remote_cleanup_confirmed']
        assert absent(result['events'][0]['pid'])
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize('body', [
    'print("{not-json}",flush=True)',
    'import sys; sys.stdout.buffer.write(b"x" * (1024 * 1024 + 1)); sys.stdout.flush()',
    'import sys; sys.stdout.buffer.write(b"\\xff\\n"); sys.stdout.flush()',
    'raise SystemExit(255)',
])
def test_controller_rejects_malformed_or_closed_transport(tmp_path, monkeypatch, body):
    fake_ssh(tmp_path, monkeypatch, body + '\n')
    result = run(manifest(tmp_path, startup_timeout_s=1), tmp_path / 'out')
    assert not result['success'] and not result['remote_cleanup_confirmed']
    assert result['failure']
    assert result['duration_s'] < 5
    assert json.loads((tmp_path / 'out' / 'result.json').read_text()) == result


@pytest.mark.parametrize('kind', ['early-terminal', 'foreign-terminal', 'duplicate-terminal', 'bad-stop', 'ssh-error',
                                  'ssh-hangs', 'malformed-after-terminal'])
def test_controller_validates_complete_event_stream(tmp_path, monkeypatch, kind):
    script = '''import sys,json
start = json.loads(sys.stdin.readline())
def emit(event):
    print(json.dumps(event), flush=True)
started = {'protocol': 1, 'event': 'started', 'token': start['token'], 'pid': 123}
stopped = {'protocol': 1, 'event': 'stopped', 'token': start['token'], 'reason': 'command_exited',
           'stop': {'exit_code': 0, 'signals': []}}
'''
    if kind != 'early-terminal':
        script += 'emit(started)\n'
    if kind == 'foreign-terminal':
        script += "stopped['token'] = 'foreign'\n"
    if kind == 'bad-stop':
        script += "stopped['stop'] = []\n"
    script += 'emit(stopped)\n'
    if kind == 'duplicate-terminal':
        script += 'emit(stopped)\n'
    if kind == 'ssh-error':
        script += 'raise SystemExit(255)\n'
    if kind == 'ssh-hangs':
        script += 'import time; time.sleep(60)\n'
    if kind == 'malformed-after-terminal':
        script += 'print("malformed", flush=True)\n'
    fake_ssh(tmp_path, monkeypatch, script)
    result = run(manifest(tmp_path), tmp_path / 'out')
    assert not result['success']
    assert result['remote_cleanup_confirmed'] is (kind in ('ssh-error', 'ssh-hangs'))
    assert result['failure']


def test_worker_oversized_control_stops_running_child(tmp_path):
    process = worker()
    try:
        send(process, start(tmp_path))
        started = json.loads(process.stdout.readline())
        process.stdin.write('x' * (MAX_MESSAGE_BYTES + 1))
        process.stdin.flush()
        process.wait(timeout=8)
        stopped = json.loads(process.stdout.read())
        assert stopped['token'] == TOKEN
        assert stopped['reason'] == 'invalid_control_message'
        assert absent(started['pid'])
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
