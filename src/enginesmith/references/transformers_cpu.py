"""A correctness reference: stock Hugging Face transformers on the CPU behind the engine protocol (docs/protocol.md).

Run as `python -m enginesmith.references.transformers_cpu --model MODEL_DIR` (a task's `[reference] kind =
"transformers-cpu"`). It is for model families that the platform's own reference does not support (Qwen3.5's
gated-DeltaNet hybrid on Trainium): it decides what is correct, and its times are not a speed baseline, so a task
that uses it scores the fraction of the hardware bound (enginesmith.bound). It loads the checkpoint with
AutoModelForCausalLM, the weights in bfloat16 as published (a multimodal checkpoint loads its text model only), and
computes in bfloat16 on the CPU with transformers' own torch code paths (logits are converted to float32 the way
generate does before the argmax). It runs on a fixed number of threads (--threads) with deterministic algorithms
where torch has them, so that it agrees with itself.

Every request runs transformers' own `model.generate` (greedy, no stop token, no logits processors from the
checkpoint's generation config, an explicit all-ones attention mask), so the token streams are generate's by
construction: a reset request prefills the whole prompt in one forward call and decodes one token per call. The
model's own cache class (for a hybrid model, its KV cache and the recurrent state together) is kept between
requests: a continuation passes the whole history and that cache to generate, which feeds only the uncached last
emitted token. generate never computes a step past the last requested token.

Besides hello and generate it answers two requests that only the reference needs:
- judge: teacher-forces a token sequence through the same generate call, with a logits processor that records each
  row and then forces the given token, and reports the steps where a token is not the top choice (an exact tie
  counts as agreement), with that token's rank;
- stats: the process's resident memory (peak and current); the model lives in host memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROTOCOL = 1


class Reference:
    def __init__(self, model_dir: Path, threads: int):
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        from transformers import AutoModelForCausalLM, GenerationConfig
        self.torch, self.GenerationConfig = torch, GenerationConfig
        kwargs = {}
        quantization = json.loads((model_dir / 'config.json').read_text()).get('quantization_config') or {}
        if quantization.get('quant_method') == 'mxfp4':
            # gpt-oss publishes its experts in MXFP4. transformers' native MXFP4 path needs Triton kernels (and
            # accelerate); dequantize=True decodes the experts to bfloat16 at load instead, which is exact: every
            # MXFP4 value (an e2m1 element times a power-of-two block scale) is a bfloat16 value.
            from transformers import Mxfp4Config
            kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **kwargs).eval()
        # generate fills unset fields from the model's generation config (eos_token_id from config.json, sampling
        # settings from generation_config.json); an empty one leaves greedy decoding with no stop token.
        self.model.generation_config = GenerationConfig()
        self.vocab = self.model.config.get_text_config().vocab_size  # a multimodal config keeps it in text_config
        self.cache = None    # the model's cache after the last request
        self.history = None  # every token consumed, then the last emitted one (not yet consumed)

    def _generate(self, ids: list[int], max_tokens: int, cache, processor=None):
        """model.generate on ids with cache (None: a fresh one); returns (new tokens, cache)."""
        torch = self.torch
        config = self.GenerationConfig(do_sample=False, max_new_tokens=max_tokens, min_new_tokens=0,
                                       eos_token_id=None, pad_token_id=0, use_cache=True, num_beams=1)
        inputs = torch.tensor([ids], dtype=torch.long)
        kwargs = {'past_key_values': cache} if cache is not None else {}
        with torch.inference_mode():
            out = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=config,
                                      logits_processor=[processor] if processor else None,
                                      return_dict_in_generate=True, **kwargs)
        tokens = out.sequences[0, len(ids):].tolist()
        if len(tokens) != max_tokens:
            raise RuntimeError(f'generate returned {len(tokens)} tokens, not {max_tokens}')
        return tokens, out.past_key_values

    def generate(self, prompt: list[int], max_tokens: int, reset: bool) -> list[int]:
        if max_tokens < 1:
            raise ValueError('max_tokens must be at least 1')
        if reset:
            if not prompt:
                raise ValueError('a reset request needs a prompt')
            self.cache, self.history = None, None
            ids, cache = list(prompt), None
        else:
            if prompt or self.history is None:
                raise ValueError('a continuation has an empty prompt and follows a generate request')
            ids, cache = self.history, self.cache
        if any(not 0 <= t < self.vocab for t in ids):
            raise ValueError(f'token ids must be below the vocabulary size {self.vocab}')
        tokens, self.cache = self._generate(ids, max_tokens, cache)
        self.history = ids + tokens
        return tokens

    def judge(self, prompt: list[int], tokens: list[int]) -> dict:
        """Teacher-force tokens after prompt; a step mismatches when its token's log-probability is below the row
        maximum. Each mismatch is [step, margin to the maximum, the reference's token, the token's rank (2 = the
        reference's second choice)]."""
        torch = self.torch
        self.cache, self.history = None, None
        if not prompt:
            raise ValueError('judge needs a prompt')
        if not tokens:
            return {'mismatches': []}
        mismatches = []

        def force(input_ids, scores):
            step = input_ids.shape[1] - len(prompt)
            token = tokens[step]
            row = torch.log_softmax(scores[0].float(), dim=-1)
            best, value = row.max().item(), row[token].item()
            if value < best:
                rank = int((row > row[token]).sum().item()) + 1
                mismatches.append([step, best - value, int(row.argmax().item()), rank])
            forced = torch.full_like(scores, float('-inf'))
            forced[:, token] = 0.0
            return forced

        forced, _ = self._generate(list(prompt), len(tokens), None, force)
        if forced != list(tokens):
            raise RuntimeError('teacher forcing did not follow the given tokens')
        return {'mismatches': mismatches}

    def stats(self, reset_peak: bool = False) -> dict:
        status = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmHWM', 'VmRSS'):
                status[key] = int(value.split()[0]) * 1024
        result = {'peak_memory': status['VmHWM'], 'active_memory': status['VmRSS']}
        if reset_peak:
            try:  # resets VmHWM to the current RSS (Linux 4.0+)
                Path('/proc/self/clear_refs').write_text('5')
            except OSError:
                pass
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
    parser.add_argument('--threads', type=int, default=16, help='CPU threads (fixed, for determinism)')
    args = parser.parse_args(argv)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    # Stdout carries replies only; libraries that print write to file descriptor 1, which becomes stderr.
    real_stdout = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    reference = Reference(args.model, args.threads)
    serve(reference, stdout=real_stdout)


if __name__ == '__main__':
    main()
