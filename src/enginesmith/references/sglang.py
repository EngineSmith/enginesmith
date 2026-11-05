"""The reference on AMD Instinct GPUs: stock SGLang behind the engine protocol (docs/protocol.md).

Run as `python -m enginesmith.references.sglang --model MODEL_DIR`, with SGLang installed, or, when SGLang lives in a
container image, through `enginesmith.references.command`, which starts this module inside the image named by
$ENGINESMITH_SGLANG_IMAGE. It runs SGLang's offline engine (sglang.Engine) unmodified, on one GPU, with its own
defaults for the platform (attention backend, chunked prefill, graph capture, the overlap scheduler), so every request
follows SGLang's own generation path. Only these settings are fixed:

- no tokenizer (`skip_tokenizer_init`): requests carry raw token ids and replies return them;
- greedy decoding (temperature 0) of exactly max_tokens tokens: `ignore_eos`, no stop tokens or strings;
- the context length is the model's own plus a small margin (CONTEXT_MARGIN): SGLang stops a request a couple of
  tokens before its context length, and the workload's longest prompt fills the model's context exactly with the
  generated tokens (no position past the model's own context is ever computed); the KV pool is sized by
  `mem_fraction_static`;
- custom logit processors are enabled, which changes nothing for a request that carries none (only `judge` does).

State between requests is SGLang's radix cache. A reset request flushes it first, so nothing computed for an earlier
prompt is reused (the workload's development prompts are prefixes of each other). A continuation sends the whole
history (the prompt and every emitted token); the cache holds all of it but the last emitted token, so SGLang computes
only that token and then decodes, exactly the work of consuming the last token in place.

Every request goes through SGLang's own request path (the engine's tokenizer-manager and scheduler processes, IPC
between them), which costs a fixed amount per request: about 12 ms for a 1-token request on a 1-token prompt, and a
continuation's lookup of the cached history grows with its length (measured on one MI325X, Qwen3-0.6B, SGLang
0.5.22: a 64-token continuation costs about 20% more per token than the same tokens inside one request at 4,096
tokens). A timed case sends two requests, so the reference pays this twice, where a user of SGLang would send one.

Greedy decoding on ROCm uses SGLang's own sampler (aiter's greedy kernel when SGLANG_USE_AITER is set, else
torch.argmax); its tie-breaking is the library's, and an exact tie counts as agreement in judge anyway.

Besides hello and generate it answers two requests that only the reference needs:
- judge: replays the tokens with a custom logit processor that, at every step, measures the row as the model
  computed it (log-softmax in float32; the token's rank and margin, the row's top token) and then forces the step's
  token. It sends the two requests of a timed case, generate(prompt, 1) and a continuation for the rest, so step 0
  comes from the prefill, step 1 from consuming step 0's token (SGLang computes it as a one-token extend on the
  cached prompt, which can differ from a decode step in the last bits) and later steps from decode steps. An exact
  tie counts as agreement.
- stats: the GPU's device memory in use (SGLang allocates its KV pool up front, so the peak is the largest value
  observed at any request).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

PROTOCOL = 1
MEM_FRACTION = 0.85
CONTEXT_MARGIN = 16


def _forcing_processor():
    """A CustomLogitProcessor subclass that measures each row, then forces the request's next given token. SGLang
    makes a new instance for every batch but keeps the class, so the class counts the calls per request; the overlap
    scheduler lets req.output_ids lag a step, so it cannot be the step. A chunked prefill also calls it after every
    chunk but the last (that sample is discarded), so only a request whose prompt is already cached, a continuation,
    has exactly one call per step."""
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class ForceTokens(CustomLogitProcessor):
        calls = {}

        def __call__(self, logits, custom_param_list=None):
            import torch
            for row, params in enumerate(custom_param_list):
                rid, tokens = params['__req__'].rid, params['tokens']
                step = ForceTokens.calls.get(rid, 0)
                ForceTokens.calls[rid] = step + 1
                token = tokens[min(step, len(tokens) - 1)]
                logprobs = torch.log_softmax(logits[row].float(), dim=-1)
                best, value = logprobs.max(), logprobs[token]
                record = [(best - value).item(), int(logprobs.argmax().item()), int((logprobs > value).sum().item()) + 1]
                with open(params['log'], 'a') as f:
                    f.write(json.dumps(record) + '\n')
                forced = torch.full_like(logits[row], float('-inf'))
                forced[token] = 0
                logits[row] = forced
            return logits

    return ForceTokens


def _device_memory_used() -> int | None:
    """Bytes in use on the GPU SGLang runs on (the first one this process may use), from the amdgpu driver."""
    from enginesmith.platforms import linux_rocm
    gpus = linux_rocm.visible_gpus()
    return linux_rocm._int(f'{gpus[0]["device"]}/mem_info_vram_used') if gpus else None


class Reference:
    def __init__(self, model_dir: Path):
        import sglang
        config = json.loads((model_dir / 'config.json').read_text())
        context = config.get('max_position_embeddings') or config.get('max_seq_len') or config.get('seq_length')
        os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN'] = '1'
        self.engine = sglang.Engine(model_path=str(model_dir), skip_tokenizer_init=True, tp_size=1,
                                    context_length=context + CONTEXT_MARGIN if context else None,
                                    mem_fraction_static=MEM_FRACTION, enable_custom_logit_processor=True,
                                    log_level='error')
        self.forcing = _forcing_processor().to_str()
        self.history = None
        self.peak = 0
        self._observe()

    def _observe(self) -> int | None:
        used = _device_memory_used()
        if used is not None:
            self.peak = max(self.peak, used)
        return used

    def _flush(self):
        if not self.engine.flush_cache().success:
            raise RuntimeError('SGLang refused to flush its cache')

    def _run(self, input_ids: list[int], max_tokens: int, **extra) -> list[int]:
        sampling = {'temperature': 0.0, 'max_new_tokens': max_tokens, 'ignore_eos': True, 'skip_special_tokens': False}
        sampling.update(extra.pop('sampling', {}))
        out = self.engine.generate(input_ids=input_ids, sampling_params=sampling, **extra)
        tokens = list(out['output_ids'])
        if len(tokens) != max_tokens:
            raise RuntimeError(f'SGLang returned {len(tokens)} tokens for max_new_tokens={max_tokens}')
        return tokens

    def generate(self, prompt: list[int], max_tokens: int, reset: bool) -> list[int]:
        if reset:
            if not prompt:
                raise ValueError('a reset request needs a prompt')
            self._flush()
            history = list(prompt)
        else:
            if prompt or self.history is None:
                raise ValueError('a continuation has an empty prompt and follows a generate request')
            history = self.history
        tokens = self._run(history, max_tokens)
        self.history = history + tokens
        self._observe()
        return tokens

    def _forced(self, input_ids: list[int], tokens: list[int]) -> list[list]:
        """Generate len(tokens) tokens forced to tokens; the processor's records, one per call, in call order."""
        with tempfile.NamedTemporaryFile('r', suffix='.jsonl', prefix='judge-') as log:
            out = self._run(input_ids, len(tokens), custom_logit_processor=self.forcing,
                            sampling={'custom_params': {'tokens': tokens, 'log': log.name}})
            records = [json.loads(line) for line in log]
        if out != tokens:
            raise RuntimeError('the forced replay did not follow the given tokens')
        return records

    def judge(self, prompt: list[int], tokens: list[int]) -> dict:
        """Teacher-force tokens after prompt; a step mismatches when its token's log-probability is below the row
        maximum. Each mismatch is [step, margin to the maximum, the reference's token, the token's rank (2 = the
        reference's second choice)]. The replay sends the requests a timed case sends, generate(prompt, 1) and then
        a continuation for the rest, so every step is computed as in that case."""
        self._flush()
        records = self._forced(list(prompt), list(tokens[:1]))[-1:]
        if len(tokens) > 1:
            rest = self._forced(list(prompt) + list(tokens[:1]), list(tokens[1:]))
            if len(rest) != len(tokens) - 1:
                raise RuntimeError(f'the forced continuation measured {len(rest)} steps, expected {len(tokens) - 1}')
            records += rest
        self._flush()
        self.history = None
        return {'mismatches': [[step, margin, best, rank] for step, (margin, best, rank) in enumerate(records)
                               if margin > 0]}

    def stats(self, reset_peak: bool = False) -> dict:
        active = self._observe()
        result = {'peak_memory': self.peak, 'active_memory': active}
        if reset_peak:
            self.peak = active or 0
        return result


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
    args = parser.parse_args(argv)
    # aiter compiles kernels on first use; inside a read-only image it must keep them under HOME.
    os.environ.setdefault('AITER_JIT_DIR', str(Path.home() / '.aiter' / 'jit'))
    # SGLang's subprocesses print to stdout, which carries replies only: keep the real stdout for replies and send
    # everything else written to file descriptor 1 to stderr.
    replies = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    reference = Reference(args.model)
    try:
        serve(reference, stdout=replies)
    finally:
        reference.engine.shutdown()


if __name__ == '__main__':
    main()
