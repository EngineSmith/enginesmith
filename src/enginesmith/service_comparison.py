"""Diagnostic experiment compatibility and paired measurements. Never an EngineSmith acceptance decision."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from enginesmith.task import TaskError

IDENTITY = ('model_revision', 'weights_sha256', 'precision', 'kv_precision', 'hardware', 'gpu_count',
            'topology', 'resource_budget', 'workload_sha256', 'client_version', 'dataset_sha256',
            'sampling', 'cache_policy', 'concurrency', 'duration_s')
METRICS = ('output_tokens_per_s', 'ttft_p99_ms', 'itl_p99_ms', 'requests', 'failed_requests', 'cancelled_requests')


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TaskError(f'{label}: expected a finite number')
    if value < 0 or positive and value <= 0:
        raise TaskError(f'{label}: expected {"positive" if positive else "nonnegative"} value')
    return value


def validate(data):
    if not isinstance(data, dict) or data.get('schema') != 1 or type(data.get('schema')) is not int:
        raise TaskError('experiment must be a schema-1 object')
    if set(data) != {'schema', 'identity', 'speculation', 'evidence', 'measurements'}:
        raise TaskError('experiment requires exactly schema, identity, speculation, evidence, measurements')
    identity = data['identity']
    if not isinstance(identity, dict) or set(identity) != set(IDENTITY):
        raise TaskError('identity must explicitly name every model/resource/workload field')
    for key, value in identity.items():
        if value is None or value == '' or value == {} or value == []:
            raise TaskError(f'identity.{key}: missing value')
    for key in ('weights_sha256', 'workload_sha256', 'dataset_sha256'):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise TaskError(f'identity.{key}: expected lowercase SHA256')
    for key in ('gpu_count', 'concurrency'):
        if type(identity[key]) is not int or identity[key] <= 0:
            raise TaskError(f'identity.{key}: expected positive integer')
    _number(identity['duration_s'], 'duration_s', positive=True)
    spec = data['speculation']
    if not isinstance(spec, dict) or set(spec) != {'mode', 'draft_revision', 'draft_length', 'synthetic_rule'}:
        raise TaskError('speculation must name mode, draft_revision, draft_length and synthetic_rule')
    if spec['mode'] not in ('none', 'actual', 'synthetic'):
        raise TaskError('speculation.mode must be none, actual or synthetic')
    if type(spec['draft_length']) is not int or spec['draft_length'] < 0:
        raise TaskError('draft_length must be a nonnegative integer')
    if spec['mode'] == 'none':
        if spec['draft_length'] or spec['draft_revision'] is not None or spec['synthetic_rule'] is not None:
            raise TaskError('disabled speculation must not have draft or synthetic settings')
    else:
        if not isinstance(spec['draft_revision'], str) or not spec['draft_revision'] or not spec['draft_length']:
            raise TaskError('enabled speculation requires draft revision and positive length')
        if spec['mode'] == 'actual' and spec['synthetic_rule'] is not None:
            raise TaskError('actual verification cannot have synthetic acceptance rules')
        if spec['mode'] == 'synthetic' and (not isinstance(spec['synthetic_rule'], dict) or not spec['synthetic_rule']):
            raise TaskError('synthetic mode requires an explicit acceptance rule, not only a mode label')
    evidence = data['evidence']
    if not isinstance(evidence, dict) or set(evidence) != {'runtime_verified', 'runtime_artifact_sha256', 'quality'}:
        raise TaskError('evidence requires runtime_verified, runtime_artifact_sha256 and quality')
    if type(evidence['runtime_verified']) is not bool:
        raise TaskError('runtime_verified must be boolean')
    quality = evidence['quality']
    if quality is not None:
        if not isinstance(quality, dict) or set(quality) != {'mode', 'passed', 'artifact_sha256'}:
            raise TaskError('quality must be null or an object with mode, passed and artifact_sha256')
        if quality['mode'] not in ('none', 'actual') or type(quality['passed']) is not bool:
            raise TaskError('quality must come from real verification/generation, never synthetic acceptance')
    for label, digest in [('runtime', evidence['runtime_artifact_sha256']),
                          ('quality', quality['artifact_sha256'] if quality else None)]:
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in '0123456789abcdef' for c in digest)):
            raise TaskError(f'{label} artifact must be a SHA256 or null')
    if evidence['runtime_verified'] and evidence['runtime_artifact_sha256'] is None:
        raise TaskError('verified runtime requires an evidence artifact digest')
    measurements = data['measurements']
    if not isinstance(measurements, list) or not measurements:
        raise TaskError('measurements must be a nonempty list')
    seen = set()
    for row in measurements:
        if not isinstance(row, dict) or set(row) != {'pair_id', *METRICS}:
            raise TaskError('measurement must contain pair_id and all metric fields (null for unavailable)')
        if not isinstance(row['pair_id'], str) or not row['pair_id'] or row['pair_id'] in seen:
            raise TaskError('pair_id must be nonempty and unique per experiment')
        seen.add(row['pair_id'])
        for key in METRICS:
            if row[key] is not None:
                _number(row[key], key, positive=key in ('output_tokens_per_s', 'requests'))
                if key.endswith('requests') and type(row[key]) is not int:
                    raise TaskError(f'{key}: expected integer count')
        if all(row[k] is not None for k in ('requests', 'failed_requests', 'cancelled_requests')):
            if row['failed_requests'] + row['cancelled_requests'] > row['requests']:
                raise TaskError('failed + cancelled counts exceed total attempts')
    return data


def load(path: Path):
    try:
        return validate(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        raise TaskError(f'cannot read experiment: {exc}') from exc


def compare(baseline, candidate, *, max_ttft_ms, max_itl_ms, max_error_fraction):
    validate(baseline)
    validate(candidate)
    _number(max_ttft_ms, 'max_ttft_ms', positive=True)
    _number(max_itl_ms, 'max_itl_ms', positive=True)
    _number(max_error_fraction, 'max_error_fraction')
    if max_error_fraction > 1:
        raise TaskError('max_error_fraction cannot exceed 1')
    incompatible = [f'identity.{k}' for k in IDENTITY if baseline['identity'][k] != candidate['identity'][k]]
    if baseline['speculation'] != candidate['speculation']:
        incompatible.append('speculation')
    b = {row['pair_id']: row for row in baseline['measurements']}
    c = {row['pair_id']: row for row in candidate['measurements']}
    if set(b) != set(c):
        incompatible.append('pair IDs')
    reasons = []
    for label, data in [('baseline', baseline), ('candidate', candidate)]:
        if not data['evidence']['runtime_verified']:
            reasons.append(f'{label}: runtime configuration unverified')
        quality = data['evidence']['quality']
        if quality is None or quality['passed'] is not True or quality['artifact_sha256'] is None:
            reasons.append(f'{label}: missing or failed real quality evidence')
        if len(data['measurements']) < 3:
            reasons.append(f'{label}: fewer than three repetitions; diagnostic only')
        for row in data['measurements']:
            for metric in METRICS:
                if row[metric] is None:
                    reasons.append(f'{label}/{row["pair_id"]}: {metric} unavailable')
            if row['ttft_p99_ms'] is not None and row['ttft_p99_ms'] > max_ttft_ms:
                reasons.append(f'{label}/{row["pair_id"]}: TTFT limit exceeded')
            if row['itl_p99_ms'] is not None and row['itl_p99_ms'] > max_itl_ms:
                reasons.append(f'{label}/{row["pair_id"]}: ITL limit exceeded')
            if all(row[k] is not None for k in ('requests', 'failed_requests', 'cancelled_requests')):
                if (row['failed_requests'] + row['cancelled_requests']) / row['requests'] > max_error_fraction:
                    reasons.append(f'{label}/{row["pair_id"]}: error/cancellation fraction exceeded')
    ratios = []
    if not incompatible:
        ratios = [c[k]['output_tokens_per_s'] / b[k]['output_tokens_per_s'] for k in sorted(b)
                  if b[k]['output_tokens_per_s'] is not None and c[k]['output_tokens_per_s'] is not None]
    return {'schema': 1, 'kind': 'service_comparison_diagnostic', 'compatible': not incompatible,
            'incompatible_fields': incompatible, 'limitations': reasons,
            'paired_output_throughput_ratios': ratios,
            'median_ratio': statistics.median(ratios) if ratios else None,
            'min_ratio': min(ratios) if ratios else None, 'max_ratio': max(ratios) if ratios else None,
            'acceptance': None, 'acceptance_reason': 'diagnostic only; not integrated into EngineSmith judging',
            'uncertainty': 'ranges are observed repeats, not confidence intervals or proof of improvement',
            'evidence_scope': 'artifact digests and verification flags are declarations; evidence not opened here',
            'performance_mode': candidate['speculation']['mode']}
