"""The hardware bound of a case: the least time the machine needs for its prefill and decode requests (docs/task.md,
"The hardware bound"). An estimate from the checkpoint and the platform's published or measured figures, never a
measurement.

- Decode floor: bytes read per generated token / memory bandwidth. The bytes are the checkpoint's weight tensors (from
  the safetensors headers; the input embedding is a gather of one row, so it counts only when it is also the output
  head) plus the KV cache at that position: per layer K and V, KV heads x head_dim x the bytes of config.json's dtype,
  over every position, or at most the window of a sliding-window layer.
- Prefill floor: FLOPs / peak compute. generate(P, 1) needs 2 x parameters x prompt tokens for the layers, the output
  head for the last token only, and 4 x heads x head_dim per (query, key) pair of causal attention.

Dense models only for now. A mixture-of-experts model (which experts a token reads depends on the routing) and a
hybrid one (linear-attention, recurrent or convolution layers) get no bound, nor does a platform without figures
(platforms.current().peak); the reason is recorded instead. Where the config leaves a choice, the bound takes the
smaller time, so that it stays a floor.

A task with a correctness-only reference scores the fraction of the bound, bound time / engine time. Where the bound
is not computed it uses NOMINAL_MS instead, so the score is still a consistent relative measure; reports then say
"not computed".
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from enginesmith.platforms import current
from enginesmith.task import model_config

NOMINAL_MS = 1.0  # per prefill request and per decode token, where the bound is not computed
ATTENTION = ('full_attention', 'sliding_attention')
DTYPE_BYTES = {'bfloat16': 2, 'float16': 2, 'float32': 4}


def _tensors(model: Path) -> dict[str, tuple[str, list[int], int]]:
    """name -> (dtype, shape, bytes) from the safetensors headers; the tensor data is never read."""
    tensors = {}
    for path in sorted(model.glob('*.safetensors')):
        with open(path, 'rb') as f:
            (size,) = struct.unpack('<Q', f.read(8))
            header = json.loads(f.read(size))
        header.pop('__metadata__', None)
        tensors.update({name: (meta['dtype'], meta['shape'], meta['data_offsets'][1] - meta['data_offsets'][0])
                        for name, meta in header.items()})
    return tensors


def model_facts(model: Path) -> dict:
    """What the bound needs from a dense checkpoint, or {'reason': why there is none}."""
    model = Path(model)
    raw = json.loads((model / 'config.json').read_text()) if (model / 'config.json').is_file() else {}
    config = model_config(model) if raw else {}
    if config.get('num_experts') or config.get('num_local_experts'):
        return {'reason': 'a mixture-of-experts model (the experts a token reads depend on the routing)'}
    kinds = config.get('layer_types') or ['full_attention'] * (config.get('num_hidden_layers') or 0)
    if other := sorted(set(kinds) - set(ATTENTION)):
        return {'reason': f'a hybrid model ({", ".join(other)} layers)'}
    heads = config.get('num_attention_heads')
    hidden = config.get('hidden_size')
    head_dim = config.get('head_dim') or (hidden // heads if heads and hidden else None)
    if not kinds or not heads or not head_dim:
        return {'reason': 'config.json states no layers, attention heads or head size'}
    tensors = _tensors(model)
    if not tensors:
        return {'reason': 'no safetensors files'}
    if type(raw.get('text_config')) is dict:  # a multimodal checkpoint: the text model's tensors only
        tensors = {k: v for k, v in tensors.items() if 'language_model.' in k or k.startswith('lm_head.')} or tensors
    head = [k for k in tensors if 'lm_head.' in k] or [k for k in tensors if 'embed_tokens.' in k]
    counted = [k for k in tensors if k in head or 'embed_tokens.' not in k]
    quantized = config.get('quantization') or config.get('quantization_config')
    window = config.get('sliding_window') if type(config.get('sliding_window')) is int else None
    elements = lambda names: sum(math.prod(tensors[k][1]) for k in names)
    return {'weight_bytes': sum(tensors[k][2] for k in counted),
            # Packed integer tensors do not give the parameter count, so a quantized checkpoint has no prefill bound.
            'layer_parameters': None if quantized else elements(counted) - elements(head),
            'head_parameters': None if quantized else elements(head),
            'heads': heads, 'kv_heads': config.get('num_key_value_heads') or heads, 'head_dim': head_dim,
            # Without layer_types every layer may be windowed when the config states a window (the smaller time).
            'windows': [window if kind == 'sliding_attention' or 'layer_types' not in config else None
                        for kind in kinds],
            'kv_bytes': DTYPE_BYTES.get(config.get('torch_dtype') or config.get('dtype'), 2)}


def _pairs(window: int | None, length: int) -> int:
    """(query, key) pairs of causal attention over length tokens, each query seeing at most window keys."""
    w = length if window is None else min(window, length)
    return w * (w + 1) // 2 + (length - w) * w


def case_bound(facts: dict, peak: dict | None, length: int, output_tokens: int) -> dict:
    """The floors of one case's two timed requests, in ms: prefill_ms for generate(P, 1) and decode_ms for the
    output_tokens - 1 tokens of the continuation; None where not computed, and why."""
    if facts.get('reason') or not peak:
        return {'prefill_ms': None, 'decode_ms': None,
                'reason': facts.get('reason') or 'no hardware figures for this platform'}
    bound, reasons = {'prefill_ms': None, 'decode_ms': None}, []
    if peak.get('bandwidth'):
        # Decode step i feeds the token at position length + i - 1 and attends to length + i positions.
        per_position = 2 * facts['kv_heads'] * facts['head_dim'] * facts['kv_bytes']
        read = sum(facts['weight_bytes'] + per_position * sum(length + i if w is None else min(w, length + i)
                                                              for w in facts['windows'])
                   for i in range(1, output_tokens))
        bound['decode_ms'] = read / peak['bandwidth'] * 1000
    else:
        reasons.append('decode: the platform states no memory bandwidth')
    if not peak.get('compute'):
        reasons.append('prefill: the platform states no peak compute')
    elif facts['layer_parameters'] is None:
        reasons.append('prefill: a quantized checkpoint (its packed tensors do not give the parameter count)')
    else:
        flops = (2 * facts['layer_parameters'] * length + 2 * facts['head_parameters']
                 + 4 * facts['heads'] * facts['head_dim'] * sum(_pairs(w, length) for w in facts['windows']))
        bound['prefill_ms'] = flops / peak['compute'] * 1000
    return {**bound, 'reason': '; '.join(reasons) or None}


def bounds(model: Path, lengths, output_tokens: int, cores: int = 1) -> dict:
    """{'peak': the platform's figures and their source, 'cases': {length: case_bound}} for an engine on `cores`
    accelerator cores of this machine."""
    peak = getattr(current(), 'peak', lambda cores=1: None)(cores)
    facts = model_facts(model)
    return {'peak': peak, 'cases': {n: case_bound(facts, peak, n, output_tokens) for n in lengths}}
