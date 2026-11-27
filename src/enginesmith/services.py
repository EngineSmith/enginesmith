"""Local foreground services around an external diagnostic workload; no scoring or search integration."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
import tomllib
from pathlib import Path

from enginesmith.processes import stop
from enginesmith.task import TaskError


def _keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise TaskError(f'{label}: expected a table with keys {", ".join(allowed)}')


def _argv(value, label):
    if not isinstance(value, list) or not value or any(
            not isinstance(s, str) or not s or '\0' in s for s in value):
        raise TaskError(f'{label}: expected a nonempty array of nonempty strings')
    return value


def _seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TaskError(f'{label}: expected finite positive seconds')
    return value


def load_manifest(path: Path) -> dict:
    """Validate everything before starting a process. Paths are relative to the manifest, never the caller."""
    path = path.resolve()
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f'cannot read service manifest: {exc}') from exc
    _keys(data, ('version', 'services', 'workload'), 'manifest')
    if type(data.get('version')) is not int or data['version'] != 1:
        raise TaskError('service manifest version must be 1')
    if not isinstance(data.get('services'), list):
        raise TaskError('services must be an array of tables (empty for an external endpoint)')
    names = set()

    def command(value, label, service=False):
        allowed = ('command', 'cwd', 'env', 'inherit_env', 'timeout_s')
        _keys(value, (*allowed, 'name', 'ready') if service else allowed, label)
        argv = _argv(value.get('command'), f'{label}.command')
        cwd = value.get('cwd', '.')
        if not isinstance(cwd, str) or '\0' in cwd:
            raise TaskError(f'{label}.cwd: expected a path string')
        cwd = (path.parent / cwd).resolve()
        if not cwd.is_dir():
            raise TaskError(f'{label}.cwd: directory does not exist: {cwd}')
        env = value.get('env', {})
        inherited = value.get('inherit_env', ['PATH'])
        valid_key = lambda k: isinstance(k, str) and bool(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k))
        if not isinstance(env, dict) or any(not valid_key(k) or not isinstance(v, str) or '\0' in v
                                            for k, v in env.items()):
            raise TaskError(f'{label}.env: expected string environment values with valid names')
        if not isinstance(inherited, list) or any(not valid_key(k) for k in inherited):
            raise TaskError(f'{label}.inherit_env: expected an array of environment names')
        if len(set(inherited)) != len(inherited) or set(inherited) & set(env):
            raise TaskError(f'{label}: duplicate or overlapping environment names')
        missing = [k for k in inherited if k not in os.environ]
        if missing:
            raise TaskError(f'{label}: missing inherited environment variables: {", ".join(missing)}')
        result = {'command': argv, 'cwd': str(cwd), 'env': dict(env), 'inherit_env': inherited,
                  'timeout_s': _seconds(value.get('timeout_s', 300 if service else 3600), f'{label}.timeout_s')}
        if service:
            name = value.get('name')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z_0-9-]*', name):
                raise TaskError(f'{label}.name: expected a letter followed by letters, digits, _ or -')
            if name in names or name == 'workload':
                raise TaskError(f'{label}.name: duplicate or reserved name: {name}')
            names.add(name)
            result.update(name=name, ready=_argv(value.get('ready'), f'{label}.ready'))
        return result

    services = [command(v, f'services[{i}]', True) for i, v in enumerate(data['services'])]
    workload = command(data.get('workload'), 'workload')
    return {'version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'services': services, 'workload': workload}


class ServiceFailure(Exception):
    pass


@contextlib.contextmanager
def _interruptions():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received {signal.Signals(signum).name}')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def check(manifest: Path, out: Path) -> dict:
    """Run on the main thread. Own only local process groups started here; never attach to stored PIDs."""
    config = load_manifest(manifest)
    out = out.resolve()
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise TaskError(f'cannot create fresh service output directory: {exc}') from exc
    started = time.monotonic()
    result = {'schema': 1, 'kind': 'service_check', 'status': 'starting', 'success': False,
              'configuration': config, 'started_at': time.time(), 'services': [], 'workload': None,
              'failure': None, 'cleanup_errors': [], 'duration_s': None}
    # Snapshot only explicitly named variables. Values inherited from the caller never enter the report.
    environments = {id(s): {**{k: os.environ[k] for k in s['inherit_env']}, **s['env']}
                    for s in [*config['services'], config['workload']]}
    owned = []
    services = []

    def save():
        temporary = out / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(out / 'result.json')

    def launch(spec, argv, log_name):
        with (out / log_name).open('ab') as log:
            process = subprocess.Popen(argv, cwd=spec['cwd'], env=environments[id(spec)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        owned.append(process)
        return process

    def finish(process):
        stopped = stop(process, grace=0)
        owned.remove(process)
        if stopped['exit_code'] is None:
            result['cleanup_errors'].append(f'process {process.pid} did not exit after cleanup')
        return stopped

    def healthy():
        for process, record in services:
            code = process.poll()
            if code is not None:
                raise ServiceFailure(f'service {record["name"]} exited with code {code}')

    def wait(process, deadline):
        while True:
            healthy()
            code = process.poll()
            if code is not None:
                return code
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.05, remaining))

    # This is a diagnostic supervisor, not an engine protocol. stdout/stderr belong in role logs.
    with _interruptions():
        try:
            save()
            for spec in config['services']:
                healthy()
                record = {'name': spec['name'], 'pid': None, 'status': 'starting', 'ready_attempts': 0,
                          'log': f'{spec["name"]}.log', 'ready_log': f'{spec["name"]}.ready.log',
                          'ready_after_s': None, 'stop': None}
                result['services'].append(record)
                save()
                process = launch(spec, spec['command'], record['log'])
                services.append((process, record))
                record['pid'] = process.pid
                save()
                service_start = time.monotonic()
                deadline = service_start + spec['timeout_s']
                while True:
                    healthy()
                    if time.monotonic() >= deadline:
                        raise ServiceFailure(f'service {spec["name"]} readiness timed out')
                    probe = launch(spec, spec['ready'], record['ready_log'])
                    record['ready_attempts'] += 1
                    try:
                        code = wait(probe, min(deadline, time.monotonic() + 5))
                    finally:
                        finish(probe)
                    if code == 0:
                        healthy()
                        record.update(status='ready', ready_after_s=time.monotonic() - service_start)
                        save()
                        break
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            healthy()
            result['status'] = 'running'
            spec = config['workload']
            record = {'pid': None, 'log': 'workload.log', 'exit_code': None, 'duration_s': None, 'stop': None}
            result['workload'] = record
            save()
            process = launch(spec, spec['command'], record['log'])
            record['pid'] = process.pid
            workload_start = time.monotonic()
            save()
            try:
                code = wait(process, workload_start + spec['timeout_s'])
                record['exit_code'] = code
                if code is None:
                    raise ServiceFailure('workload timed out')
                if code != 0:
                    raise ServiceFailure(f'workload exited with code {code}')
            finally:
                record['duration_s'] = time.monotonic() - workload_start
                record['stop'] = finish(process)
            result.update(status='succeeded', success=True)
        except KeyboardInterrupt as exc:
            result.update(status='interrupted', failure=str(exc) or 'interrupted')
        except (OSError, ServiceFailure) as exc:
            result.update(status='failed', failure=str(exc))
        finally:
            # A second interrupt must not strand the groups already owned by this invocation.
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            for process in reversed(owned[:]):
                stopped = finish(process)
                for service, record in services:
                    if service is process:
                        record.update(status='stopped', stop=stopped)
            if result['cleanup_errors']:
                result['success'] = False
                if result['status'] != 'interrupted':
                    result['status'] = 'failed'
                result['failure'] = result['failure'] or '; '.join(result['cleanup_errors'])
            result['duration_s'] = time.monotonic() - started
            save()
    return result
