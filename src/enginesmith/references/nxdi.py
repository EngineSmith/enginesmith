"""The Trainium reference: stock NxD Inference (neuronx-distributed-inference) behind the engine protocol (docs/protocol.md).

Run as `python -m enginesmith.references.nxdi --model MODEL_DIR --max-length N`. It compiles the model with NxD
Inference's own model class for its `model_type` (tensor parallel degree 1, one logical NeuronCore, bf16, bucketing,
fused QKV, async mode, on-device greedy sampling) and drives it the way `HuggingFaceGenerationAdapter.generate` does: the whole prompt goes
through the context-encoding model in one call (padded to its bucket), which returns the first token; every further
token is one call of the token-generation model with the previous token at the next position. It never computes a
step past the last requested token, so no work leaks from one timed request into the next.

Compiling takes minutes; the compiled model is cached under ENGINESMITH_NXDI_CACHE (default
~/.cache/enginesmith/nxdi), keyed by the model's config, the settings and the library versions.

Besides hello and generate it answers two requests that only the reference needs:
- judge: teacher-forces a token sequence through a second compilation of the same model that returns logits instead
  of tokens, step by step along the generation path, and reports the steps where a token is not the top choice (an
  exact tie counts as agreement), with that token's rank. It runs in a worker process started on first use on the
  second NeuronCore the reference is given (NxD Inference loads a tensor-parallel-1 model on the first visible core
  whatever its start rank, so two compilations of a large model would not fit on one core);
- stats: device memory in use, from neuron-monitor (peak = active: NxD allocates everything at load).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path

PROTOCOL = 1
CONTEXT_BUCKET_FLOOR = 128
# neuronx-cc 2.27 fails on some unaligned bucket sizes (NCC_IXRO002 at 4,162 positions for Qwen3-8B): the compiled length
# is rounded up to a multiple of this.
BUCKET_ALIGN = 512
# The Neuron tools (neuronx-cc, libneuronpjrt-path) run as subprocesses and the harness gives the reference a minimal
# PATH: put this environment's bin directory first, before any Neuron library is imported.
os.environ['PATH'] = f'{Path(sys.executable).parent}{os.pathsep}{os.environ.get("PATH", "")}'
PACKAGES = ('neuronx-distributed-inference', 'neuronx-distributed', 'torch-neuronx', 'neuronx-cc')


def _cache_root() -> Path:
    """The harness sets HOME to a scratch directory, so the default cache is under the account's real home."""
    import pwd
    default = Path(pwd.getpwuid(os.getuid()).pw_dir) / '.cache' / 'enginesmith' / 'nxdi'
    return Path(os.environ.get('ENGINESMITH_NXDI_CACHE') or default)


def _buckets(limit: int) -> list[int]:
    """Powers of two from 128, then limit (a multiple of BUCKET_ALIGN)."""
    sizes, size = [], CONTEXT_BUCKET_FLOOR
    while size < limit:
        sizes.append(size)
        size *= 2
    return sizes + [limit]


class Reference:
    def __init__(self, model_dir: Path, max_length: int, cores: list[int], judge_worker: bool = False, tp: int = 1):
        """cores: 2 x tp NeuronCores, the generating model's first and the judge worker's after them. This process sees
        only its own (the runtime claims every core it can see); the judge worker, a second process, sees the others."""
        self.cores, self.judge_worker, self.tp = cores, judge_worker, tp
        own = cores[tp:2 * tp] if judge_worker else cores[:tp]
        os.environ['NEURON_RT_VISIBLE_CORES'] = ','.join(map(str, own))
        self.max_length = max_length
        import torch
        from neuronx_distributed_inference.utils.constants import MODEL_TYPES
        self.torch, self.model_dir = torch, Path(model_dir)
        config = json.loads((self.model_dir / 'config.json').read_text())
        self.model_class = MODEL_TYPES[config['model_type']]['causal-lm']
        # max_length counts the prompt and every generated token; the last generated token is never fed back.
        self.seq_len = -(-max(CONTEXT_BUCKET_FLOOR, max_length) // BUCKET_ALIGN) * BUCKET_ALIGN
        self.vocab = config['vocab_size']
        # Mixture-of-experts models: NxD Inference's default blockwise expert kernel (shard on hidden) is not in
        # this neuronx-distributed build (NotImplementedError while tracing); shard on intermediate is.
        self.extra = ({'blockwise_matmul_config': {'use_shard_on_intermediate_dynamic_while': True}}
                      if config.get('num_experts') else {})
        self.model = self._load(greedy=not judge_worker)
        self.worker = None
        self.position = None  # positions consumed so far
        self.pending = None   # the last emitted token, not yet consumed

    def _settings(self, greedy: bool) -> dict:
        return {'tp_degree': self.tp, 'batch_size': 1, 'seq_len': self.seq_len, 'max_context_length': self.seq_len,
                'enable_bucketing': True, 'context_encoding_buckets': _buckets(self.seq_len),
                'token_generation_buckets': _buckets(self.seq_len), 'greedy': greedy,
                # The fastest configuration measured for Qwen3 on trn2 (the NKI attention, QKV and MLP kernels
                # were slower for decode, the token-generation attention kernels do not trace for Qwen3).
                # Async mode breaks logits outputs (the judge's compilation), which are not timed.
                'async_mode': greedy, 'fused_qkv': True, **self.extra}

    def _load(self, greedy: bool):
        """Compile (once, cached) and load on the visible NeuronCore."""
        from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        settings = self._settings(greedy)
        key = hashlib.sha256(json.dumps({
            'config': (self.model_dir / 'config.json').read_text(), 'model': str(self.model_dir.resolve()),
            'settings': settings, 'versions': {p: _version(p) for p in PACKAGES}}, sort_keys=True).encode())
        out = _cache_root() / key.hexdigest()[:24]
        kwargs = {k: v for k, v in settings.items() if k != 'greedy'}
        neuron_config = self.model_class.get_neuron_config_cls()(**kwargs, torch_dtype=self.torch.bfloat16,
                                     on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False) if greedy else None)
        # NxD compiles in BASE_COMPILE_WORK_DIR, by default one fixed /tmp directory shared by every process.
        os.environ['BASE_COMPILE_WORK_DIR'] = f'{out / "work"}/'
        config = self.model_class.get_config_cls()(neuron_config, load_config=load_pretrained_config(str(self.model_dir)))
        model = self.model_class(str(self.model_dir), config)
        if not (out / 'done').exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f'compiling {self.model_class.__name__} into {out}', file=sys.stderr, flush=True)
            model.compile(str(out))
            (out / 'done').write_text('')
        model.load(str(out))
        return model

    def _call(self, model, tokens: list[int], start: int):
        """One forward call with the inputs HuggingFaceGenerationAdapter builds: a prompt at positions 0, 1, ... with a
        mask over it, or one token at position start with a mask of length start (the adapter does not extend the mask
        after the context-encoding call, then extends it by one per step)."""
        from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
        torch = self.torch
        end = start + len(tokens)
        return model(input_ids=torch.tensor([tokens], dtype=torch.int32),
                     attention_mask=torch.ones((1, end if start == 0 else start), dtype=torch.int32),
                     position_ids=torch.arange(start, end, dtype=torch.int32)[None],
                     seq_ids=torch.tensor([0], dtype=torch.int32), sampling_params=prepare_sampling_params(1))

    def _check_length(self, needed: int) -> None:
        if needed > self.seq_len:
            raise ValueError(f'{needed} positions exceed the compiled length {self.seq_len} (--max-length)')

    def generate(self, prompt: list[int], max_tokens: int, reset: bool) -> list[int]:
        if reset:
            if not prompt:
                raise ValueError('a reset request needs a prompt')
            self._check_length(len(prompt) + max_tokens - 1)
            token = int(self._call(self.model, prompt, 0).tokens.reshape(-1)[-1])
            self.position = len(prompt)
            tokens = [token]
        else:
            if prompt or self.pending is None:
                raise ValueError('a continuation has an empty prompt and follows a generate request')
            self._check_length(self.position + max_tokens)
            tokens = []
            token = self.pending
        while len(tokens) < max_tokens:
            token = int(self._call(self.model, [token], self.position).tokens.reshape(-1)[-1])
            self.position += 1
            tokens.append(token)
        self.pending = tokens[-1]
        return tokens

    def judge(self, prompt: list[int], tokens: list[int]) -> dict:
        """Teacher-force tokens after prompt; a step mismatches when its token's logit is below the row maximum.
        Each mismatch is [step, margin to the maximum in log-probability, the reference's token, the token's rank
        (2 = the reference's second choice)]."""
        self._check_length(len(prompt) + len(tokens) - 1)
        self.position, self.pending = None, None
        if not self.judge_worker:
            return self._ask_worker({'op': 'judge', 'prompt': prompt, 'tokens': tokens})
        torch = self.torch
        mismatches = []
        for step, token in enumerate(tokens):
            if step == 0:
                logits = self._call(self.model, prompt, 0).logits
            else:
                logits = self._call(self.model, [tokens[step - 1]], len(prompt) + step - 1).logits
            row = torch.log_softmax(logits.reshape(-1, logits.shape[-1])[-1, :self.vocab].float(), dim=-1)
            best, value = row.max(), row[token]
            if value.item() < best.item():
                rank = int((row > value).sum().item()) + 1
                mismatches.append([step, best.item() - value.item(), int(row.argmax().item()), rank])
        return {'mismatches': mismatches}

    def _ask_worker(self, request: dict) -> dict:
        """One request to the judge worker, started (and its model loaded) on first use."""
        import subprocess
        if self.worker is None:
            env = {**os.environ, 'NEURON_RT_VISIBLE_CORES': ','.join(map(str, self.cores))}
            self.worker = subprocess.Popen(
                [sys.executable, '-m', 'enginesmith.references.nxdi', '--model', str(self.model_dir),
                 '--max-length', str(self.max_length), '--tp', str(self.tp), '--judge-worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
        self.worker.stdin.write(json.dumps({'id': 0, **request}) + '\n')
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError(f'the judge worker exited with status {self.worker.wait()}')
        reply = json.loads(line)
        if not reply.get('ok'):
            raise RuntimeError(f'judge worker: {reply.get("error")}')
        return {k: v for k, v in reply.items() if k not in ('id', 'ok')}

    def stats(self, reset_peak: bool = False) -> dict:
        used = _hbm_in_use(os.getpid())
        return {'peak_memory': used, 'active_memory': used}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hbm_in_use(pid: int) -> int:
    """Device memory this process holds: neuron-monitor's neuron_runtime_used_bytes.neuron_device for its pid."""
    import subprocess
    monitor = subprocess.Popen(['/opt/aws/neuron/bin/neuron-monitor'], stdout=subprocess.PIPE, text=True)
    try:
        for _ in range(5):  # one report per period; the first may predate this process's allocations
            report = json.loads(monitor.stdout.readline())
            for runtime in report.get('neuron_runtime_data') or []:
                if runtime.get('pid') == pid:
                    return int(runtime['report']['memory_used']['neuron_runtime_used_bytes']['neuron_device'])
    finally:
        monitor.kill()
        monitor.wait()
    raise RuntimeError(f'neuron-monitor does not list process {pid}')


def serve(reference: Reference, stdin=sys.stdin.buffer, stdout=sys.stdout) -> None:
    for line in stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id, op = request.get('id'), request.get('op')
            if op == 'hello':
                reply = {'protocol': PROTOCOL}
            elif op == 'generate':
                reply = {'tokens': reference.generate(request['prompt'], request['max_tokens'], request['reset'])}
            elif op == 'judge':
                reply = reference.judge(request['prompt'], request['tokens'])
            elif op == 'stats':
                reply = reference.stats(bool(request.get('reset_peak')))
            else:
                raise ValueError(f'unknown op {op!r}')
            reply = {'id': request_id, 'ok': True, **reply}
        except Exception as exc:  # every failure is a reply; the parent decides what it means
            traceback.print_exc(file=sys.stderr)
            reply = {'id': request_id, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        stdout.write(json.dumps(reply, separators=(',', ':')) + '\n')
        stdout.flush()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=8192,
                        help='the longest prompt plus generated tokens a request may need')
    parser.add_argument('--core', type=int, default=0,
                        help='the logical NeuronCore when NEURON_RT_VISIBLE_CORES is unset (the judge uses the next one)')
    parser.add_argument('--tp', type=int, default=1, help='tensor-parallel degree (2 x tp NeuronCores in all)')
    parser.add_argument('--judge-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    visible = os.environ.get('NEURON_RT_VISIBLE_CORES')
    cores = []
    for part in (visible or f'{args.core}-{args.core + 2 * args.tp - 1}').split(','):  # "a-b", "a,b" or "a"
        low, _, high = part.partition('-')
        cores += range(int(low), int(high or low) + 1)
    while len(cores) < 2 * args.tp:  # fewer visible than needed: continue after the last
        cores.append(cores[-1] + 1)
    # Stdout carries replies only. The libraries and the compiler subprocesses they start write to file descriptor 1,
    # so replies get a private copy of it and descriptor 1 becomes stderr. The model loads (and compiles, the first
    # time) before the first request is read, so hello's reply arrives once it is ready.
    real_stdout = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    reference = Reference(args.model, args.max_length, cores, judge_worker=args.judge_worker, tp=args.tp)
    serve(reference, stdout=real_stdout)


if __name__ == '__main__':
    main()
