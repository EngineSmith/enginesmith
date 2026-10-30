"""Facts for setting up a task: the machine (doctor), the model (probe) and the reference's actual behavior on this
machine (prepare). They report what they observed and label every estimate as such."""
from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

from enginesmith import agents, references, workload
from enginesmith.brief import reference_argv
from enginesmith.platforms import current
from enginesmith.processes import Registry, Watchdog
from enginesmith.protocol import Engine, EngineFailure
from enginesmith.task import AGENT_KINDS, load_task, model_config


SANDBOXED = ("this shell cannot write to the home directory (a coding agent's sandbox), but the sessions' agent CLI, "
             'cargo and uv must: start the run outside the sandbox (approve running this command unsandboxed, or run '
             'it in your own terminal); do not move HOME or a tool\'s home into the repository instead')


def sandboxed() -> bool:
    """True when this shell cannot write to the home directory, as inside a coding agent's sandbox (Codex and DeepSeek
    Harness confine commands to the workspace by default)."""
    try:
        handle, path = tempfile.mkstemp(dir=Path.home(), prefix='.enginesmith-write-check-')
    except OSError:
        return True
    os.close(handle)
    os.unlink(path)
    return False


def doctor() -> dict:
    """The platform's machine facts and checks, plus the reference package, git and the coding agents: which CLIs are
    on PATH, and which one the user is talking to (current_agent, the sessions' default)."""
    platform_module = current()
    facts = platform_module.machine()
    kind = platform_module.DEFAULT_REFERENCE
    clis = [name for name in AGENT_KINDS if name != 'command']
    facts.update(python=platform.python_version(), reference=kind and {'kind': kind, 'version': references.version(kind)},
                 git=shutil.which('git'), agents={name: shutil.which(name) for name in clis},
                 current_agent=agents.current(), sandboxed=sandboxed())
    problems, notes = platform_module.checks(facts)
    if kind and not facts['reference']['version']:
        problems.append(f'The {kind} reference is not installed: run `uv sync --extra mlx`.')
    if not facts['git']:
        problems.append('git is not on PATH.')
    if not any(facts['agents'].values()):
        problems.append(f'No coding agent CLI ({", ".join(clis)}) is on PATH; install and log in to one, or use '
                        'agent.kind = "command".')
    if facts['current_agent'] and not facts['agents'][facts['current_agent']]:
        notes.append(f'This shell runs inside {facts["current_agent"]}, but `{facts["current_agent"]}` is not on PATH; '
                     'the sessions start it by that name.')
    if facts['sandboxed']:
        notes.append('This shell is sandboxed (it cannot write to the home directory). The setup commands work here '
                     '(call .venv/bin/enginesmith if `uv run` cannot open its cache); `enginesmith run` cannot: '
                     + SANDBOXED + '.')
    if not facts['current_agent']:
        notes.append('No agent marker in this shell (Kimi Code sets none): the task must name agent.kind.')
    return {**facts, 'problems': problems, 'notes': notes}


def _layer_kinds(config: dict) -> dict[str, int]:
    kinds = config.get('layer_types')
    if isinstance(kinds, list) and kinds:
        counts = {}
        for kind in kinds:
            counts[str(kind)] = counts.get(str(kind), 0) + 1
        return counts
    return {'full_attention (assumed)': config.get('num_hidden_layers', 0)}


def _full_attention(kind: str) -> bool:
    kind = kind.lower()
    return 'attention' in kind and not any(word in kind for word in ('linear', 'sliding', 'local', 'chunk'))


def probe(model: Path) -> dict:
    """Facts from config.json and the safetensors files, and an ESTIMATE of the KV-cache size and the prompt
    lengths that fit in memory."""
    model = Path(model).resolve()
    config = model_config(model)
    heads = config.get('num_attention_heads')
    kv_heads = config.get('num_key_value_heads', heads)
    head_dim = config.get('head_dim') or (config.get('hidden_size') // heads if heads else None)
    kinds = _layer_kinds(config)
    full = sum(n for kind, n in kinds.items() if _full_attention(kind))
    kv_per_token = 2 * full * kv_heads * head_dim * 2 if kv_heads and head_dim else None
    weights = sum(p.stat().st_size for p in model.glob('*.safetensors'))
    max_positions = next((config[k] for k in ('max_position_embeddings', 'max_seq_len', 'seq_length')
                          if type(config.get(k)) is int), None)
    budget = current().memory_budget()
    lengths = {}
    for n in workload.LENGTHS:
        if max_positions and n + 8192 > max_positions:
            lengths[n] = 'beyond the model context (max_position_embeddings)'
        elif kv_per_token is None or budget is None:
            lengths[n] = 'unknown'
        else:
            need = weights + kv_per_token * (n + 8192) + 2 * 2048 * config.get('hidden_size', 0) * 4 * 8
            lengths[n] = 'fits (estimate)' if need < 0.9 * budget else f'too large (estimate: {need / 2**30:.0f} GiB)'
    quantization = config.get('quantization') or config.get('quantization_config')
    return {
        'model': str(model), 'model_type': config.get('model_type'), 'architectures': config.get('architectures'),
        'vocab_size': config.get('vocab_size'), 'max_position_embeddings': max_positions,
        'layers': config.get('num_hidden_layers'), 'hidden_size': config.get('hidden_size'),
        'attention_heads': heads, 'kv_heads': kv_heads, 'head_dim': head_dim, 'layer_kinds': kinds,
        'sliding_window': config.get('sliding_window'), 'experts': config.get('num_experts') or
        config.get('num_local_experts'), 'experts_per_token': config.get('num_experts_per_tok'),
        'quantization': quantization, 'dtype': config.get('torch_dtype') or config.get('dtype'),
        'weights_bytes': weights, 'tied_embeddings': config.get('tie_word_embeddings'),
        'kv_bytes_per_token_estimate': kv_per_token, 'kv_estimate_basis':
            f'{full} full-attention layers x 2 (K, V) x {kv_heads} KV heads x {head_dim} x 2 bytes; sliding, '
            'linear-attention and convolution layers counted as constant',
        'memory_budget_bytes': budget, 'lengths': lengths,
    }


def prepare(task_path: Path, lengths=None, max_request_s: float = 3600.0, log=print) -> dict:
    """Run the reference once at every workload length on this machine and record its times and peak memory, then
    estimate how long one formal evaluation takes. Writes prepare.json next to the task file."""
    task = load_task(task_path)
    lengths = sorted(lengths or task.lengths)
    scratch = task.path.parent / '.prepare'
    registry = Registry(scratch / 'engines')
    rows, stopped, load_ms = [], None, None
    killed = []
    started = time.time()
    with Watchdog([registry], on_kill=killed.append):
        engine = Engine(reference_argv(task), scratch=scratch / 'reference', label='reference', registry=registry)
        with engine:
            try:
                load_ms = engine.hello(1800.0)
                log(f'reference loaded in {load_ms / 1000:.1f} s')
                # One untimed request first, so the first measured length does not include kernel compilation.
                engine.generate(workload.prompt(min(lengths[0], 2048), task.vocab_size), 2, True, task.vocab_size,
                                max_request_s)
                for n in lengths:
                    offset = workload.draw_offset(b'prepare', f'prepare/{n}', n, workload.NONCE)
                    prompt = workload.prompt(n, task.vocab_size, offset)
                    engine.call('stats', 60.0, reset_peak=True)
                    _, prefill = engine.generate(prompt, 1, True, task.vocab_size, max_request_s)
                    _, decode = engine.generate([], task.output_tokens - 1, False, task.vocab_size, max_request_s)
                    stats, _ = engine.call('stats', 60.0)
                    rows.append({'length': n, 'prefill_ms': prefill, 'decode_ms': decode,
                                 'decode_ms_per_token': decode / (task.output_tokens - 1),
                                 'peak_memory_bytes': stats['peak_memory']})
                    log(f'{n:>7} tokens: prefill {prefill / 1000:.2f} s, decode {decode / (task.output_tokens - 1):.2f} '
                        f'ms/token, peak memory {stats["peak_memory"] / 2**30:.1f} GiB')
            except EngineFailure as exc:
                stopped = f'{exc} (memory pressure turned critical)' if killed else str(exc)
                log(f'stopped: {stopped}')
    e = task.evaluation
    per_arm = sum(r['prefill_ms'] + r['decode_ms'] for r in rows) / 1000 + (load_ms or 0) / 1000
    controls = 2 * (load_ms or 0) / 1000 + 60
    evaluation = e.rounds * 2 * (per_arm + e.cooldown_s) + controls if rows else None
    result = {'task': str(task.path), 'observed': {'load_ms': load_ms, 'lengths': rows},
              'stopped': stopped, 'duration_s': time.time() - started,
              'estimates': {'evaluation_s': evaluation, 'final_s': evaluation and evaluation * e.final_rounds / e.rounds,
                            'basis': 'the reference times above, the candidate assumed as fast as the reference, '
                                     'plus loads and cooldowns; judging of differing streams not included'}}
    (task.path.parent / 'prepare.json').write_text(json.dumps(result, indent=2))
    return result
