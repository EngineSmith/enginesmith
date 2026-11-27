import json
import sys
from pathlib import Path

import pytest

from enginesmith.agentx import DATASETS, VERSION, inspect_export, plan, run
from enginesmith.task import TaskError


def manifest(tmp_path, **overrides):
    values = {'version': 1, 'python': sys.executable, 'url': 'http://127.0.0.1:8000', 'model': 'test-model',
              'tokenizer': 'test-tokenizer', 'concurrency': 8, **overrides}
    path = tmp_path / 'agentx.toml'
    path.write_text('\n'.join(f'{key} = {json.dumps(value)}' for key, value in values.items()))
    return path


def export(expected):
    return {'aiperf_version': VERSION, 'was_cancelled': False, 'request_count': {'avg': 20, 'unit': 'requests'},
            'time_to_first_token': {'p99': 18, 'unit': 'ms'}, 'error_summary': {'total': 2},
            'input_config': {'scenario': 'agentx', 'models': {'items': [{'name': expected['model']}]},
            'endpoint': {'urls': [expected['url']], 'type': 'chat', 'streaming': True, 'use_server_token_count': True},
            'datasets': [{'dataset': expected['dataset'], 'random_seed': expected['seed'],
                          'trace_idle_gap_cap_seconds': 300}],
            'phases': [{'kind': 'profiling', 'type': 'concurrency', 'timing_mode': 'agentic_replay',
                        'concurrency': expected['concurrency'], 'duration': expected['duration_s'],
                        'warmup_requests_per_lane': 10, 'agentic_warmup_grace_period': 1800,
                        'trajectory_start_min_ratio': 0.25, 'trajectory_start_max_ratio': 0.75,
                        'system_idle_gap_cap_seconds': 10, 'grace_period': 30, 'failed_request_threshold': 0.1}],
            'tokenizer': {'name': expected['tokenizer'], 'trust_remote_code': expected['trust_remote_code']}}}


def test_dry_run_keeps_scenario_and_blocks_ambient_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv('AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID', 'wrong')
    out = tmp_path / 'out'
    result = run(manifest(tmp_path, session_header='dynamo', duration_s=900), out, dry_run=True)
    argv = result['command']
    assert argv[1:6] == ['-m', 'agentx_cli', 'profile', '--scenario', 'agentx']
    assert argv[argv.index('--concurrency') + 1] == '8'
    assert argv[argv.index('--public-dataset') + 1] == DATASETS[0]
    assert result['env']['AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID'] == 'true'
    assert not out.exists()


@pytest.mark.parametrize('change', [
    {'concurrency': 0}, {'concurrency': True}, {'duration_s': 899}, {'timeout_s': 2}, {'seed': -1},
    {'session_header': 'unknown'}, {'url': 'http://user:secret@localhost'}, {'url': 'http://localhost?key=secret'},
    {'inherit_env': ['AIPERF_SOMETHING']}, {'trust_remote_code': 'true'}, {'dataset': 'unversioned-dataset'},
    {'unknown': 1}, {'version': True},
])
def test_invalid_manifest_fails_before_launch(tmp_path, change):
    with pytest.raises(TaskError):
        run(manifest(tmp_path, **change), tmp_path / 'out', dry_run=True)


def test_relative_interpreter_keeps_virtual_environment_identity(tmp_path):
    result = plan(manifest(tmp_path, python='client/bin/python'), tmp_path / 'out')
    assert result['command'][0] == str(tmp_path / 'client' / 'bin' / 'python')


def test_export_retains_missing_data_errors_and_units_without_score(tmp_path):
    expected = plan(manifest(tmp_path), tmp_path / 'out')['expected']
    data = export(expected)
    path = tmp_path / 'export.json'
    path.write_text(json.dumps(data))
    result = inspect_export(path, expected)
    assert result['configuration_matches']
    assert result['metrics']['time_to_first_token'] == {'p99': 18, 'unit': 'ms'}
    assert result['metrics']['output_token_throughput'] is None
    assert 'output_token_throughput' in result['missing_metrics']
    assert result['error_summary'] == {'total': 2}  # configuration match does not hide request errors
    assert result['quality']['measured'] is False


@pytest.mark.parametrize('mutate', [
    lambda x: x.update(aiperf_version='other'),
    lambda x: x.update(was_cancelled=True),
    lambda x: x.update(request_count={'avg': 0}),
    lambda x: x['input_config']['phases'][0].update(concurrency=1),
    lambda x: x['input_config']['phases'][0].update(warmup_requests_per_lane=0),
    lambda x: x['input_config']['phases'][0].update(timing_mode='request_rate'),
    lambda x: x['input_config']['endpoint'].update(streaming=False),
    lambda x: x['input_config']['datasets'][0].update(random_seed=123),
    lambda x: x['input_config']['tokenizer'].update(name='different'),
])
def test_effective_configuration_drift_is_reported(tmp_path, mutate):
    expected = plan(manifest(tmp_path), tmp_path / 'out')['expected']
    data = export(expected)
    mutate(data)
    path = tmp_path / 'export.json'
    path.write_text(json.dumps(data))
    result = inspect_export(path, expected)
    assert not result['configuration_matches'] and result['problems']


@pytest.mark.parametrize('content', ['[]', '{}', 'null', 'not json'])
def test_malformed_export_is_not_success(tmp_path, content):
    path = tmp_path / 'bad.json'
    path.write_text(content)
    assert not inspect_export(path, {})['configuration_matches']


def test_external_execution_and_version_failure(tmp_path):
    # Install only local fake metadata/module, not a fake claim of real AgentX integration.
    fake = tmp_path / 'client'
    fake.mkdir()
    meta = fake / f'agentx_harness-{VERSION}.dist-info'
    meta.mkdir()
    metadata = meta / 'METADATA'
    metadata.write_text(f'Metadata-Version: 2.1\nName: agentx-harness\nVersion: {VERSION}\n')
    path = manifest(tmp_path)
    path.write_text(path.read_text() + '\n[env]\nPYTHONPATH = ' + json.dumps(str(fake)))
    expected = plan(path, tmp_path / 'out')['expected']
    (fake / 'agentx_cli.py').write_text('import sys,json\nfrom pathlib import Path\n'
        'out=Path(sys.argv[sys.argv.index("--output-artifact-dir")+1]); out.mkdir()\n'
        f'(out/"profile_export_aiperf.json").write_text({json.dumps(json.dumps(export(expected)))})\n')
    result = run(path, tmp_path / 'out')
    assert result['completed'] and result['score'] is None and not result['quality_measured']
    assert json.loads((tmp_path / 'out' / 'agentx-result.json').read_text()) == result
    metadata.write_text('Metadata-Version: 2.1\nName: agentx-harness\nVersion: 0.0.1\n')
    with pytest.raises(TaskError, match='required'):
        run(path, tmp_path / 'wrong-version')
    assert not (tmp_path / 'wrong-version').exists()
