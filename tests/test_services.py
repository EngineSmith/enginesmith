"""Real CPU subprocesses exercise supervision, failures and ownership; no model/GPU evidence is implied."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from enginesmith import cli
from enginesmith.services import check, load_manifest
from enginesmith.task import TaskError


def py(code):
    return [sys.executable, '-c', code]


def manifest(tmp_path, *, services=None, workload=None):
    services = services or [{'name': 'frontend', 'command': py('import time; time.sleep(60)'),
                             'ready': py('pass'), 'timeout_s': 3}]
    workload = workload or {'command': py('print("measured")'), 'timeout_s': 3}
    lines = ['version = 1']
    for section, entry in [('[[services]]', s) for s in services] + [('[workload]', workload)]:
        lines.append(section)
        for key, value in entry.items():
            if isinstance(value, dict):
                encoded = '{' + ', '.join(f'{k} = {json.dumps(v)}' for k, v in value.items()) + '}'
            else:
                encoded = json.dumps(value)
            lines.append(f'{key} = {encoded}')
    path = tmp_path / 'services.toml'
    path.write_text('\n'.join(lines))
    return path


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def assert_clean(result):
    for service in result['services']:
        if service['pid'] is not None:
            assert gone(service['pid'])
            assert service['stop']['exit_code'] is not None
    if result['workload'] and result['workload']['pid']:
        assert gone(result['workload']['pid'])


def test_cli_orders_services_and_retains_logs_and_safe_provenance(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('SERVICE_TEST_SECRET', 'secret-from-parent')
    monkeypatch.setenv('SERVICE_UNLISTED', 'must-not-inherit')
    first = py('import os,time; from pathlib import Path; '
               'assert os.environ["SERVICE_TEST_SECRET"]; '
               'assert "SERVICE_UNLISTED" not in os.environ; '
               'Path("first").touch(); time.sleep(60)')
    second = py('import time; from pathlib import Path; assert Path("first").exists(); '
                'Path("second").touch(); time.sleep(60)')
    ready = lambda name: py(f'from pathlib import Path; assert Path("{name}").exists()')
    path = manifest(tmp_path, services=[
        {'name': 'prefill', 'command': first, 'ready': ready('first'),
         'inherit_env': ['PATH', 'SERVICE_TEST_SECRET'], 'env': {'TEST_MODE': 'diagnostic'}},
        {'name': 'frontend', 'command': second, 'ready': ready('second')}])
    out = tmp_path / 'out'
    monkeypatch.chdir(tmp_path.parent)
    assert cli.main(['service-check', str(path), '--out', str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((out / 'result.json').read_text())
    assert result['success'] and result['status'] == 'succeeded'
    assert [s['name'] for s in result['services']] == ['prefill', 'frontend']
    assert all(s['ready_after_s'] is not None for s in result['services'])
    assert result['configuration']['services'][0]['env'] == {'TEST_MODE': 'diagnostic'}
    assert 'secret-from-parent' not in json.dumps(result)
    assert (out / 'workload.log').read_text().strip() == 'measured'
    assert_clean(result)


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(name='../escape'),
    lambda s: s.update(name='workload'),
    lambda s: s.update(command='echo bad'),
    lambda s: s.update(ready=[]),
    lambda s: s.update(timeout_s=0),
    lambda s: s.update(timeout_s=True),
    lambda s: s.update(timeout_s='20s'),
    lambda s: s.update(cwd='missing-dir'),
    lambda s: s.update(unknown_field=1),
    lambda s: s.update(inherit_env=['ABSENT_ENGINESMITH_TEST_ENV']),
    lambda s: s.update(inherit_env=['PATH', 'PATH']),
    lambda s: s.update(env={'PATH': 'x'}),
    lambda s: s.update(env={'BAD-NAME': 'x'}),
])
def test_rejects_invalid_manifest_before_launch(tmp_path, mutation):
    service = {'name': 'server', 'command': py('from pathlib import Path; Path("started").touch()'),
               'ready': py('pass')}
    mutation(service)
    path = manifest(tmp_path, services=[service])
    with pytest.raises(TaskError):
        check(path, tmp_path / 'out')
    assert not (tmp_path / 'started').exists() and not (tmp_path / 'out').exists()


def test_duplicate_services_and_existing_output_are_rejected(tmp_path):
    path = manifest(tmp_path)
    original = path.read_text()
    service = original[original.index('[[services]]'):original.index('[workload]')]
    path.write_text(original + '\n' + service)
    with pytest.raises(TaskError, match='duplicate'):
        load_manifest(path)
    path.write_text(original)
    out = tmp_path / 'out'
    out.mkdir()
    sentinel = out / 'result.json'
    sentinel.write_text('previous evidence')
    with pytest.raises(TaskError, match='fresh'):
        check(path, out)
    assert sentinel.read_text() == 'previous evidence'


@pytest.mark.parametrize('ready', [py('raise SystemExit(2)'), py('import time; time.sleep(60)')])
def test_readiness_failure_and_hung_probe_timeout(tmp_path, ready):
    path = manifest(tmp_path, services=[{'name': 'server', 'command': py('import time; time.sleep(60)'),
                                         'ready': ready, 'timeout_s': 0.3}])
    result = check(path, tmp_path / 'out')
    assert result['status'] == 'failed' and 'readiness timed out' in result['failure']
    assert result['workload'] is None
    assert result['services'][0]['ready_attempts'] >= 1
    assert_clean(result)


def test_missing_later_executable_cleans_earlier_service(tmp_path):
    path = manifest(tmp_path, services=[
        {'name': 'one', 'command': py('import time; time.sleep(60)'), 'ready': py('pass')},
        {'name': 'two', 'command': ['./does-not-exist'], 'ready': py('pass')}])
    result = check(path, tmp_path / 'out')
    assert not result['success'] and result['workload'] is None
    assert 'does-not-exist' in result['failure']
    assert_clean(result)


def test_service_exit_during_workload_is_failure_even_with_zero_exit_code(tmp_path):
    server = py('import time; from pathlib import Path; '
                'exec("while not Path(\'workload-started\').exists(): time.sleep(.01)")')
    work = py('import time; from pathlib import Path; Path("workload-started").touch(); time.sleep(60)')
    path = manifest(tmp_path, services=[{'name': 'server', 'command': server, 'ready': py('pass')}],
                    workload={'command': work, 'timeout_s': 3})
    result = check(path, tmp_path / 'out')
    assert result['failure'] == 'service server exited with code 0'
    assert_clean(result)


@pytest.mark.parametrize('workload, message', [
    ({'command': py('raise SystemExit(7)')}, 'workload exited with code 7'),
    ({'command': py('import time; time.sleep(60)'), 'timeout_s': 0.1}, 'workload timed out'),
])
def test_workload_errors_stop_services(tmp_path, workload, message):
    result = check(manifest(tmp_path, workload=workload), tmp_path / 'out')
    assert result['failure'] == message and not result['success']
    assert_clean(result)


def wait_for(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None, process.returncode
        assert time.monotonic() < deadline, f'timed out waiting for {path}'
        time.sleep(0.02)


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_interrupt_cleans_owned_group_but_leaves_unrelated_process(tmp_path, sig):
    # The parent reaps its child on TERM so the test does not depend on init's orphan/zombie handling.
    server = '''import subprocess,sys,signal,time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
def stop(sig, frame):
    child.terminate()
    child.wait(timeout=2)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while True: time.sleep(.05)
'''
    work = py('from pathlib import Path; import time; Path("running").touch(); time.sleep(60)')
    path = manifest(tmp_path, services=[{'name': 'server', 'command': py(server),
                                        'ready': py('from pathlib import Path; assert Path("child.pid").exists()')}],
                    workload={'command': work})
    out = tmp_path / 'out'
    unrelated = subprocess.Popen(py('import time; time.sleep(60)'), start_new_session=True)
    process = subprocess.Popen([sys.executable, '-m', 'enginesmith.cli', 'service-check', str(path),
                                '--out', str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        wait_for(tmp_path / 'running', process)
        process.send_signal(sig)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stderr.decode()
        result = json.loads(stdout)
        assert result['status'] == 'interrupted' and not result['success']
        assert result == json.loads((out / 'result.json').read_text())
        assert_clean(result)
        assert gone(int((tmp_path / 'child.pid').read_text()))
        assert unrelated.poll() is None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        unrelated.terminate()
        unrelated.wait()


def test_restores_signal_handlers_and_cli_reports_invalid_input(tmp_path, capsys):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    check(manifest(tmp_path), tmp_path / 'out')
    assert before == {sig: signal.getsignal(sig) for sig in before}
    assert cli.main(['service-check', str(tmp_path / 'missing'), '--out', str(tmp_path / 'bad')]) == 2
    assert not json.loads(capsys.readouterr().out)['valid']
