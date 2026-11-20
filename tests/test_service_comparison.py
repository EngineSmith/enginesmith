import copy
import json

import pytest

from enginesmith import cli
from enginesmith.service_comparison import compare, validate
from enginesmith.task import TaskError


def experiment(mode='actual'):
    return {'schema': 1, 'identity': {'model_revision': 'model-commit', 'weights_sha256': 'a' * 64,
            'precision': 'checkpoint-fp8', 'kv_precision': 'fp8', 'hardware': 'test-gpu', 'gpu_count': 16,
            'topology': 'P8+D8', 'resource_budget': {'cpu_cores': 64, 'host_memory_gb': 512, 'nic_count': 8},
            'workload_sha256': 'b' * 64, 'client_version': '1.0.14', 'dataset_sha256': 'c' * 64,
            'sampling': {'temperature': 1, 'top_p': 1}, 'cache_policy': 'independent-warmup',
            'concurrency': 2048, 'duration_s': 3600},
            'speculation': {'mode': mode, 'draft_revision': None if mode == 'none' else 'draft-commit',
                            'draft_length': 0 if mode == 'none' else 5,
                            'synthetic_rule': {'kind': 'test-rule', 'mean_acceptance': 3.51}
                            if mode == 'synthetic' else None},
            'evidence': {'runtime_verified': True, 'runtime_artifact_sha256': 'd' * 64,
                         'quality': {'mode': 'actual', 'passed': True, 'artifact_sha256': 'e' * 64}},
            'measurements': [{'pair_id': str(i), 'output_tokens_per_s': 100, 'ttft_p99_ms': 1000,
                              'itl_p99_ms': 10, 'requests': 1000, 'failed_requests': 0, 'cancelled_requests': 0}
                             for i in range(3)]}


def comparison(baseline, candidate):
    return compare(baseline, candidate, max_ttft_ms=2000, max_itl_ms=20, max_error_fraction=0.01)


def test_paired_ratios_are_reported_but_never_accepted():
    baseline = experiment()
    candidate = copy.deepcopy(baseline)
    for row, ratio in zip(candidate['measurements'], [1.1, 1.2, 1.3]):
        row['output_tokens_per_s'] *= ratio
    result = comparison(baseline, candidate)
    assert result['compatible'] and result['median_ratio'] == 1.2
    assert result['min_ratio'] == 1.1 and result['max_ratio'] == 1.3
    assert result['acceptance'] is None and not result['limitations']


@pytest.mark.parametrize('field,value', [('precision', 'fp4'), ('gpu_count', 8), ('cache_policy', 'reuse'),
                                       ('dataset_sha256', 'f' * 64), ('sampling', {'temperature': 0})])
def test_incompatible_conditions_have_no_speedup(field, value):
    baseline, candidate = experiment(), experiment()
    candidate['identity'][field] = value
    result = comparison(baseline, candidate)
    assert not result['compatible'] and result['median_ratio'] is None
    assert f'identity.{field}' in result['incompatible_fields']


def test_actual_and_synthetic_cannot_mix():
    result = comparison(experiment('actual'), experiment('synthetic'))
    assert result['incompatible_fields'] == ['speculation']
    assert not result['paired_output_throughput_ratios']


def test_same_synthetic_comparison_keeps_mode_and_requires_real_quality():
    baseline, candidate = experiment('synthetic'), experiment('synthetic')
    candidate['evidence']['quality'] = None
    result = comparison(baseline, candidate)
    assert result['compatible'] and result['performance_mode'] == 'synthetic'
    assert any('quality' in reason for reason in result['limitations'])
    candidate['evidence']['quality'] = {'mode': 'synthetic', 'passed': True, 'artifact_sha256': 'e' * 64}
    with pytest.raises(TaskError, match='never synthetic'):
        validate(candidate)


def test_missing_metrics_failed_cancelled_and_runtime_evidence_are_visible():
    baseline, candidate = experiment(), experiment()
    row = candidate['measurements'][0]
    row.update(ttft_p99_ms=None, output_tokens_per_s=None, failed_requests=6, cancelled_requests=6)
    candidate['evidence'].update(runtime_verified=False, runtime_artifact_sha256=None)
    result = comparison(baseline, candidate)
    assert len(result['paired_output_throughput_ratios']) == 2
    assert any('unverified' in reason for reason in result['limitations'])
    assert any('unavailable' in reason for reason in result['limitations'])
    assert any('error/cancellation' in reason for reason in result['limitations'])
    assert result['acceptance'] is None


def test_unpaired_repeats_are_incompatible_and_noisy_range_is_not_confidence():
    baseline, candidate = experiment(), experiment()
    candidate['measurements'][0]['pair_id'] = 'other'
    result = comparison(baseline, candidate)
    assert 'pair IDs' in result['incompatible_fields'] and result['median_ratio'] is None
    assert 'not confidence intervals' in result['uncertainty']


@pytest.mark.parametrize('mutate', [
    lambda d: d['measurements'][0].update(output_tokens_per_s=float('nan')),
    lambda d: d['measurements'][0].update(failed_requests=-1),
    lambda d: d['measurements'][0].update(failed_requests=1001),
    lambda d: d['measurements'][0].update(requests=1.5),
    lambda d: d['identity'].update(gpu_count=True),
    lambda d: d['speculation'].update(synthetic_rule={'mean': 3.51}),
    lambda d: d['evidence'].update(runtime_artifact_sha256=None),
    lambda d: d['measurements'].append(d['measurements'][0]),
])
def test_invalid_or_ambiguous_inputs_fail(mutate):
    data = experiment()
    mutate(data)
    with pytest.raises(TaskError):
        validate(data)


def test_cli_outputs_diagnostic_only(tmp_path, capsys):
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(experiment()))
    assert cli.main(['service-compare', str(path), str(path), '--max-ttft-ms', '2000',
                     '--max-itl-ms', '20', '--max-error-fraction', '.01']) == 0
    assert json.loads(capsys.readouterr().out)['acceptance'] is None
