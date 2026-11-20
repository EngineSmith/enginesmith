import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from enginesmith.remote import plan, run
from enginesmith.remote_worker import validate_start
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
    path.write_text('\n'.join(f'{k} = {json.dumps(v)}' for k, v in values.items()))
    return path


def test_controller_over_a_local_fake_ssh_transport(tmp_path, monkeypatch):
    # Exercise stdin/control lifecycle without claiming real SSH or multi-node evidence.
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    script = bindir / 'ssh'
    script.write_text(f'#!{sys.executable}\nimport os,sys,shlex\n'
                      'args=shlex.split(sys.argv[-1]); os.execv(args[0],args)\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])
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
