import json

from conftest import fake_engine
from enginesmith import decode_probe, report


def test_one_forward_per_token_stays_near_1(tmp_path, model_dir):
    result = decode_probe.run(fake_engine() + ['--model', str(model_dir)], 1000, 64, 32, tmp_path / 'probe')
    assert result['identical'] and len(result['rounds']) == decode_probe.ROUNDS
    assert 1 <= result['ratio'] < 2
    assert 'one-token requests took' in report.probe_text(result)


def test_speculation_raises_the_ratio(tmp_path, model_dir):
    result = decode_probe.run(fake_engine('--speculate', '16') + ['--model', str(model_dir)], 1000, 64, 32, tmp_path / 'probe')
    assert result['identical'] and result['ratio'] > 4


def test_missing_results_are_stated(tmp_path):
    assert report.probe_text(None) == 'not measured' and report.probe_text({'error': 'boom'}) == 'failed (boom)'
    assert report.natural_speed(tmp_path) is None and 'not measured' in report.natural_text(None)


def test_natural_speed_is_the_median_over_items_both_sides_answered(tmp_path):
    def rows(name, *items):
        (tmp_path / name).write_text(''.join(json.dumps(item) + '\n' for item in items))
    rows('math500-reference.jsonl', {'id': 'a', 'seconds': 2.0, 'output_tokens': 10},
         {'id': 'b', 'seconds': 4.0, 'output_tokens': 10}, {'id': 'only-here', 'seconds': 9.0, 'output_tokens': 1})
    rows('math500-engine.jsonl', {'id': 'a', 'seconds': 1.0, 'output_tokens': 10},
         {'id': 'b', 'seconds': 1.0, 'output_tokens': 10}, {'id': None, 'error': 'crashed'})
    speed = report.natural_speed(tmp_path)
    assert speed == {'math500': {'speedup': 3.0, 'items': 2}}
    assert 'MATH-500 3.00x (2 items)' in report.natural_text(speed)
