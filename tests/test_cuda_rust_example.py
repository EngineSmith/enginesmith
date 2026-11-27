"""CPU checks for the GPU example's result contract; these never execute CUDA."""
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'examples/cuda-rust-runtime/validate.py'
spec = importlib.util.spec_from_file_location('cuda_example_validation', SOURCE)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def sample():
    return dict(correct=True, elements=4096, repetitions=17, direct_event_ms=1,
                direct_wall_ms=2, graph_event_ms=3, graph_wall_ms=4)


def test_sample_json_contract():
    row = sample()
    assert validation.parse_sample(json.dumps(row) + '\n', 17) == row
    # Reject escaped display text: validation consumes the executable's raw stdout.
    with pytest.raises(ValueError):
        validation.parse_sample(json.dumps(row).replace('"', '\\"'), 17)
    with pytest.raises(ValueError):
        validation.parse_sample(json.dumps(row) + '\nextra output', 17)


@pytest.mark.parametrize('field,value', [
    ('correct', False), ('elements', 1), ('repetitions', 18),
    ('direct_event_ms', float('nan')), ('direct_wall_ms', float('inf')),
    ('graph_event_ms', -1), ('graph_wall_ms', True), ('graph_wall_ms', None),
])
def test_invalid_sample_is_rejected(field, value):
    row = {**sample(), field: value}
    with pytest.raises(ValueError):
        validation.parse_sample(json.dumps(row), 17)
