"""The built-in reference: stock mlx-lm behind the engine protocol (docs/protocol.md).

Run as `python -m enginesmith.references.mlx_lm --model MODEL_DIR`. It loads the model with mlx_lm.utils.load_model and runs
it with mlx-lm's own model and cache classes, unmodified, reproducing mlx_lm.generate.generate_step: the prompt is
prefilled in chunks of min(2048, remaining - 1) tokens, the last prompt token alone gives the first logits, and
greedy decoding is pipelined one step ahead. Unlike generate_step it never computes a step past the last token a
request asks for, so no work leaks from one timed request into the next.

Besides hello and generate it answers two requests that only the reference needs:
- judge: teacher-forces a token sequence and reports the steps where a token is not the reference's top choice
  (an exact tie counts as agreement), with that token's rank;
- stats: peak and active memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROTOCOL = 1
PREFILL_STEP = 2048


class Reference:
    def __init__(self, model_dir: Path):
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.utils import load_model
        self.mx, self.stream, self.make_cache = mx, generation_stream, make_prompt_cache
        self.model, _ = load_model(model_dir)
        self.cache = None
        self.pending = None

    def _forward(self, tokens):
        """Log-probabilities after feeding tokens (a 1-D array) into the cache; shape (1, vocab)."""
        mx = self.mx
        with mx.stream(self.stream):
            logits = self.model(tokens[None], cache=self.cache)[:, -1, :]
            return logits - mx.logsumexp(logits, keepdims=True)

    def _prefill(self, prompt: list[int]):
        mx = self.mx
        self.cache, self.pending = self.make_cache(self.model), None
        y = mx.array(prompt)
        with mx.stream(self.stream):
            while y.size > 1:
                n = min(PREFILL_STEP, y.size - 1)
                self.model(y[:n][None], cache=self.cache)
                mx.eval([c.state for c in self.cache])
                y = y[n:]
                mx.clear_cache()
            return self._forward(y)

    def generate(self, prompt: list[int], max_tokens: int, reset: bool) -> list[int]:
        mx = self.mx
        if reset:
            if not prompt:
                raise ValueError('a reset request needs a prompt')
            y = mx.argmax(self._prefill(prompt), axis=-1)
        else:
            if prompt or self.pending is None:
                raise ValueError('a continuation has an empty prompt and follows a generate request')
            y = mx.argmax(self._forward(self.pending), axis=-1)
        mx.async_eval(y)
        tokens = []
        for i in range(max_tokens):
            if i + 1 < max_tokens:
                following = mx.argmax(self._forward(y), axis=-1)
                mx.async_eval(following)
            tokens.append(y.item())
            if i % 256 == 0:
                mx.clear_cache()
            if i + 1 < max_tokens:
                y = following
        self.pending = y
        return tokens

    def judge(self, prompt: list[int], tokens: list[int]) -> dict:
        """Teacher-force tokens after prompt; a step mismatches when its token's log-probability is below the row
        maximum. Each mismatch is [step, margin to the maximum, the reference's token, the token's rank (2 = the
        reference's second choice)]."""
        mx = self.mx
        logprobs = self._prefill(prompt)
        mismatches = []
        for step, token in enumerate(tokens):
            if step:
                logprobs = self._forward(mx.array([tokens[step - 1]]))
            row = logprobs[0]
            best, value = row.max(), row[token]
            mx.eval(best, value)
            if value.item() < best.item():
                rank = (row > value).sum().item() + 1
                mismatches.append([step, best.item() - value.item(), mx.argmax(row).item(), rank])
        self.cache, self.pending = None, None
        return {'mismatches': mismatches}

    def stats(self, reset_peak: bool = False) -> dict:
        mx = self.mx
        result = {'peak_memory': mx.get_peak_memory(), 'active_memory': mx.get_active_memory()}
        if reset_peak:
            mx.reset_peak_memory()
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
    reference = Reference(args.model)
    from mlx_lm.generate import wired_limit
    with wired_limit(reference.model, [reference.stream]):
        serve(reference)


if __name__ == '__main__':
    main()
