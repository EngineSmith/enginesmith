"""Pinned external AgentX diagnostics. The client owns replay; this module never scores candidates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from enginesmith.services import _keys, _seconds, check
from enginesmith.task import TaskError

VERSION = '1.0.14'
DATASETS = ('semianalysis_cc_traces_weka_062126', 'semianalysis_cc_traces_weka_062126_256k')
METRICS = ('request_count', 'request_throughput', 'output_token_throughput', 'input_token_throughput',
           'time_to_first_token', 'inter_token_latency', 'request_latency',
           'full_response_output_token_throughput_per_user', 'request_error_rate', 'benchmark_duration',
           'total_output_tokens', 'total_usage_completion_tokens')


def _text(value, label):
    if not isinstance(value, str) or not value or '\0' in value:
        raise TaskError(f'{label}: expected a nonempty string')
    return value


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise TaskError(f'{label}: expected an integer >= {minimum}')
    return value


def _url(value):
    _text(value, 'url')
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise TaskError('expected an http(s) URL without embedded credentials')
    if parsed.query or parsed.fragment:
        raise TaskError('URL query strings and fragments are not supported')
    return value.rstrip('/')


def plan(manifest: Path, out: Path) -> dict:
    """Build argv without invoking the benchmark, loading a tokenizer or contacting an endpoint."""
    manifest = manifest.resolve()
    try:
        raw = manifest.read_bytes()
        data = tomllib.loads(raw.decode())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f'cannot read AgentX manifest: {exc}') from exc
    fields = ('version', 'python', 'url', 'model', 'tokenizer', 'dataset', 'concurrency', 'duration_s',
              'timeout_s', 'seed', 'trust_remote_code', 'session_header', 'server_metrics', 'env', 'inherit_env')
    _keys(data, fields, 'AgentX manifest')
    if type(data.get('version')) is not int or data['version'] != 1:
        raise TaskError('AgentX manifest version must be 1')
    python = _text(data.get('python'), 'python')
    # A path is relative to the manifest. Bare executable names are resolved by the client's PATH.
    if '/' in python:
        python = str((manifest.parent / python).absolute())  # preserve venv symlink identity
    model = _text(data.get('model'), 'model')
    tokenizer = _text(data.get('tokenizer'), 'tokenizer')
    url = _url(data.get('url'))
    dataset = data.get('dataset', DATASETS[0])
    if dataset not in DATASETS:
        raise TaskError(f'dataset must be one of {DATASETS}')
    concurrency = _integer(data.get('concurrency'), 'concurrency')
    duration = _seconds(data.get('duration_s', 3600), 'duration_s')
    if duration < 900:
        raise TaskError('AgentX agentx scenario requires duration_s >= 900; a shorter run is not this workload')
    timeout = _seconds(data.get('timeout_s', duration + 5400), 'timeout_s')
    if timeout <= duration:
        raise TaskError('timeout_s must exceed duration_s to allow preparation, warmup and drain')
    seed = _integer(data.get('seed', 42), 'seed', 0)
    trust = data.get('trust_remote_code', False)
    if type(trust) is not bool:
        raise TaskError('trust_remote_code must be boolean')
    affinity = data.get('session_header', 'none')
    if affinity not in ('none', 'dynamo', 'session', 'both'):
        raise TaskError('session_header must be none, dynamo, session or both')
    metrics = data.get('server_metrics', [])
    if not isinstance(metrics, list):
        raise TaskError('server_metrics must be an array of URLs')
    metrics = [_url(value) for value in metrics]
    env, inherited = data.get('env', {}), data.get('inherit_env', ['PATH', 'HOME'])
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise TaskError('env must contain string values')
    if not isinstance(inherited, list) or any(not isinstance(k, str) for k in inherited):
        raise TaskError('inherit_env must contain variable names')
    if any(k.startswith('AIPERF_') for k in [*env, *inherited]):
        raise TaskError('AIPERF overrides are owned by this adapter; use manifest settings')
    env = {**env, 'AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID': str(affinity in ('dynamo', 'both')).lower(),
           'AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID': str(affinity in ('session', 'both')).lower()}
    artifacts = out.resolve() / 'aiperf_artifacts'
    argv = [python, '-m', 'agentx_cli', 'profile', '--scenario', 'agentx', '--url', url,
            '--endpoint', '/v1/chat/completions', '--model', model, '--tokenizer', tokenizer,
            '--public-dataset', dataset, '--concurrency', str(concurrency), '--benchmark-duration', str(duration),
            '--random-seed', str(seed), '--output-artifact-dir', str(artifacts)]
    if trust:
        argv.append('--tokenizer-trust-remote-code')
    if metrics:
        argv += ['--server-metrics', *metrics]
    return {'schema': 1, 'kind': 'agentx_diagnostic', 'agentx_version': VERSION,
            'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'command': argv,
            'cwd': str(manifest.parent), 'env': env, 'inherit_env': inherited, 'timeout_s': timeout,
            'artifacts': str(artifacts), 'expected': {'model': model, 'tokenizer': tokenizer,
            'url': url, 'dataset': dataset, 'concurrency': concurrency, 'duration_s': duration, 'seed': seed,
            'trust_remote_code': trust, 'server_metrics': metrics}}


def inspect_export(path: Path, expected: dict) -> dict:
    """Retain metric units/percentiles verbatim; validation here is configuration consistency, not scoring."""
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        return {'configuration_matches': False, 'problems': [f'cannot read AgentX export: {exc}'], 'metrics': None}
    problems = []
    def require(actual, wanted, label):
        if actual != wanted:
            problems.append(f'{label}: expected {wanted!r}, got {actual!r}')
    try:
        require(data.get('aiperf_version'), VERSION, 'export version')
        config = data['input_config']
        require(config.get('scenario'), 'agentx', 'scenario')
        require([v['name'] for v in config['models']['items']], [expected['model']], 'models')
        endpoint = config['endpoint']
        require(endpoint['urls'], [expected['url']], 'endpoint URLs')
        require(endpoint.get('type'), 'chat', 'endpoint type')
        require(endpoint.get('streaming'), True, 'streaming')
        require(endpoint.get('use_server_token_count'), True, 'server token counts')
        datasets = config['datasets']
        require(len(datasets), 1, 'dataset count')
        dataset = datasets[0]
        require(dataset['dataset'], expected['dataset'], 'dataset')
        require(dataset.get('random_seed'), expected['seed'], 'seed')
        require(dataset.get('trace_idle_gap_cap_seconds'), 300, 'trace idle cap')
        phases = config['phases']
        require(len(phases), 1, 'phase count')
        phase = phases[0]
        for key, wanted in {'kind': 'profiling', 'type': 'concurrency', 'timing_mode': 'agentic_replay',
                            'concurrency': expected['concurrency'], 'duration': expected['duration_s'],
                            'warmup_requests_per_lane': 10, 'agentic_warmup_grace_period': 1800,
                            'trajectory_start_min_ratio': 0.25, 'trajectory_start_max_ratio': 0.75,
                            'system_idle_gap_cap_seconds': 10, 'grace_period': 30,
                            'failed_request_threshold': 0.1}.items():
            require(phase.get(key), wanted, f'phase.{key}')
        require(config['tokenizer']['name'], expected['tokenizer'], 'tokenizer')
        require(config['tokenizer'].get('trust_remote_code', False), expected['trust_remote_code'], 'tokenizer trust')
        if expected['server_metrics']:
            require(config['server_metrics'].get('enabled'), True, 'server metrics enabled')
            require(config['server_metrics'].get('urls'), expected['server_metrics'], 'server metrics URLs')
        require(data.get('was_cancelled'), False, 'was_cancelled')
        count = data['request_count']['avg']
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count) or count <= 0:
            problems.append('no finite positive completed request count')
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        problems.append(f'incomplete or malformed export: {exc}')
    if not isinstance(data, dict):
        data = {}
    return {'configuration_matches': not problems, 'problems': problems,
            'export_sha256': hashlib.sha256(raw).hexdigest(),
            'metrics': {key: data.get(key) for key in METRICS},
            'missing_metrics': {key: 'not present in AgentX export' for key in METRICS if key not in data},
            'error_summary': data.get('error_summary'), 'warmup_metrics': data.get('warmup_metrics'),
            'branch_stats': data.get('branch_stats'), 'run_info': data.get('run_info'),
            'quality': {'measured': False, 'reason': 'trace replay is not a live coding-task correctness evaluation'}}


def run(manifest: Path, out: Path, *, dry_run=False) -> dict:
    config = plan(manifest, out)
    # Reuse the lifecycle validator and executor, including deadline and signal cleanup.
    from enginesmith.services import load_manifest
    import tempfile
    workload = {k: config[k] for k in ('command', 'cwd', 'env', 'inherit_env', 'timeout_s')}
    def toml_value(value):
        if isinstance(value, dict):
            return '{' + ', '.join(f'{json.dumps(k)} = {json.dumps(v)}' for k, v in value.items()) + '}'
        return json.dumps(value)
    text = 'version = 1\nservices = []\n[workload]\n' + '\n'.join(
        f'{key} = {toml_value(value)}' for key, value in workload.items()) + '\n'
    with tempfile.TemporaryDirectory(prefix='enginesmith-agentx-') as temporary:
        service_manifest = Path(temporary) / 'services.toml'
        service_manifest.write_text(text)
        load_manifest(service_manifest)
        if dry_run:
            return config
        environment = {**{k: os.environ[k] for k in config['inherit_env']}, **config['env']}
        try:
            version = subprocess.run([config['command'][0], '-c',
                'from importlib.metadata import version; print(version("agentx-harness"))'],
                cwd=config['cwd'], env=environment, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise TaskError(f'cannot verify external agentx-harness package: {exc}') from exc
        if version != VERSION:
            raise TaskError(f'agentx-harness {VERSION} required, found {version!r}')
        lifecycle = check(service_manifest, out)
    report = {'schema': 1, 'kind': 'agentx_diagnostic', 'plan': config, 'installed_version': version,
              'lifecycle_success': lifecycle['success'], 'lifecycle_status': lifecycle['status'],
              'failure': lifecycle['failure'], 'quality_measured': False, 'score': None}
    report['export'] = inspect_export(Path(config['artifacts']) / 'profile_export_aiperf.json', config['expected'])
    report['completed'] = lifecycle['success'] and report['export']['configuration_matches']
    temporary = out / 'agentx-result.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(out / 'agentx-result.json')
    return report
