import importlib
import json
import struct

import pytest

from enginesmith import bound

PEAK = {'bandwidth': 1e9, 'compute': 1e12, 'source': 'test figures'}
LAYER = {'self_attn.q_proj.weight': [64, 64], 'self_attn.k_proj.weight': [32, 64], 'self_attn.v_proj.weight': [32, 64],
         'self_attn.o_proj.weight': [64, 64], 'mlp.gate_proj.weight': [128, 64], 'mlp.up_proj.weight': [128, 64],
         'mlp.down_proj.weight': [64, 128], 'input_layernorm.weight': [64]}


def checkpoint(directory, config, tied=False):
    """A dense model of two layers: config.json and a safetensors file of bf16 tensors (zeros)."""
    shapes = {f'model.layers.{i}.{name}': shape for i in range(2) for name, shape in LAYER.items()}
    shapes['model.embed_tokens.weight'] = [100, 64]
    if not tied:
        shapes['lm_head.weight'] = [100, 64]
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 2 * shape[0] * (shape[1] if len(shape) > 1 else 1)
        header[name] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    (directory / 'model.safetensors').write_bytes(struct.pack('<Q', len(data)) + data + bytes(offset))
    (directory / 'config.json').write_text(json.dumps({'num_hidden_layers': 2, 'hidden_size': 64,
                                                        'num_attention_heads': 4, 'num_key_value_heads': 2,
                                                        'head_dim': 16, 'torch_dtype': 'bfloat16', **config}))
    return directory


def layer_parameters():
    return 2 * sum(s[0] * (s[1] if len(s) > 1 else 1) for s in LAYER.values())


def test_the_bound_of_a_dense_model(tmp_path):
    facts = bound.model_facts(checkpoint(tmp_path, {}))
    head = 100 * 64
    # Every weight but the input embedding, which decode reads one row of.
    assert facts['weight_bytes'] == 2 * (layer_parameters() + head)
    assert facts['layer_parameters'] == layer_parameters() and facts['head_parameters'] == head
    case = bound.case_bound(facts, PEAK, 10, 4)
    per_position = 2 * 2 * 16 * 2 * 2   # K and V, 2 KV heads of 16, 2 bytes, 2 layers
    read = sum(facts['weight_bytes'] + per_position * (10 + i) for i in range(1, 4))
    assert case['decode_ms'] == pytest.approx(read / 1e9 * 1000) and case['reason'] is None
    flops = 2 * layer_parameters() * 10 + 2 * head + 4 * 4 * 16 * 2 * (10 * 11 // 2)
    assert case['prefill_ms'] == pytest.approx(flops / 1e12 * 1000)


def test_a_tied_embedding_counts_once_as_the_head_and_a_window_caps_the_kv_cache(tmp_path):
    facts = bound.model_facts(checkpoint(tmp_path, {'layer_types': ['sliding_attention', 'full_attention'],
                                                    'sliding_window': 4}, tied=True))
    assert facts['weight_bytes'] == 2 * (layer_parameters() + 100 * 64) and facts['windows'] == [4, None]
    case = bound.case_bound(facts, PEAK, 10, 2)
    assert case['decode_ms'] == pytest.approx((facts['weight_bytes'] + 2 * 2 * 16 * 2 * (4 + 11)) / 1e9 * 1000)
    assert bound._pairs(4, 10) == 4 * 5 // 2 + 6 * 4 and bound._pairs(None, 10) == 55


def test_moe_and_hybrid_models_have_no_bound(tmp_path):
    for config, reason in (({'num_local_experts': 8}, 'mixture-of-experts'),
                           ({'num_experts': 64}, 'mixture-of-experts'),
                           ({'layer_types': ['linear_attention', 'full_attention']}, 'hybrid model (linear_attention')):
        case = bound.case_bound(bound.model_facts(checkpoint(tmp_path, config)), PEAK, 10, 4)
        assert case['prefill_ms'] is None and case['decode_ms'] is None and reason in case['reason']


def test_missing_figures_leave_a_segment_not_computed(tmp_path):
    facts = bound.model_facts(checkpoint(tmp_path, {}))
    assert 'no hardware figures' in bound.case_bound(facts, None, 10, 4)['reason']
    case = bound.case_bound(facts, {'bandwidth': 1e9, 'compute': None}, 10, 4)
    assert case['decode_ms'] and case['prefill_ms'] is None and case['reason'].startswith('prefill:')
    quantized = bound.model_facts(checkpoint(tmp_path, {'quantization': {'bits': 4, 'group_size': 64}}))
    case = bound.case_bound(quantized, PEAK, 10, 4)
    assert case['decode_ms'] and case['prefill_ms'] is None and 'quantized' in case['reason']


def test_the_neuron_platform_states_trainium2_figures_per_logical_core(monkeypatch):
    from test_platforms import NEURON_LS
    neuron = importlib.import_module('enginesmith.platforms.neuron')
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: NEURON_LS if 'neuron-ls' in argv[0] else None)
    one, four = neuron.peak(), neuron.peak(4)
    assert one['bandwidth'] == pytest.approx(725e9) and one['compute'] == pytest.approx(166.75e12)
    assert four['bandwidth'] == pytest.approx(2.9e12) and 'Neuron documentation' in one['source']
    monkeypatch.setattr(neuron, '_run', lambda argv, timeout=20: None)
    assert neuron.peak() is None
