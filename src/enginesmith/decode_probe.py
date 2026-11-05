"""The decode probe: one engine's decode asked for in one request, then one token per request, on the same prompts.

An engine that computes one forward pass per generated token takes about the same time either way, plus a round trip
per request. Speculative decoding verifies several tokens in one forward pass, and one-token requests take that away,
so they cost several times as much; keeping verified tokens for later requests shows as requests answered in next to
no time. The result is reported, never gated (docs/design.md, "Checks on the decode speed").
"""
from __future__ import annotations

import secrets
import statistics
from pathlib import Path

from enginesmith import workload
from enginesmith.processes import Registry
from enginesmith.protocol import Engine

ROUNDS = 5
HELLO_TIMEOUT = 1800.0
REQUEST_TIMEOUT = 900.0


def run(argv, vocab: int, length: int, tokens: int, directory: Path, registry: Registry | None = None) -> dict:
    """Times `tokens` decode tokens both ways (A, B, B, A, ...) on fresh prompts of `length` tokens, after one round
    that warms the engine up and is dropped."""
    with Engine(list(argv), scratch=directory / 'scratch', label='decode-probe', registry=registry) as engine:
        def together(prompt):
            first, _ = engine.generate(prompt, 1, True, vocab, REQUEST_TIMEOUT)
            rest, elapsed = engine.generate([], tokens, False, vocab, REQUEST_TIMEOUT)
            return first + rest, elapsed

        def apart(prompt):
            stream, _ = engine.generate(prompt, 1, True, vocab, REQUEST_TIMEOUT)
            times = []
            for _ in range(tokens):
                token, elapsed = engine.generate([], 1, False, vocab, REQUEST_TIMEOUT)
                stream += token
                times.append(elapsed)
            return stream, times

        engine.hello(HELLO_TIMEOUT)
        rounds = []
        low, high = workload.NONCE
        for index in range(ROUNDS + 1):
            prompt = workload.prompt(length, vocab, low + secrets.randbelow(high - low - length))
            if index % 2:
                (split, times), (whole, elapsed) = apart(prompt), together(prompt)
            else:
                (whole, elapsed), (split, times) = together(prompt), apart(prompt)
            if index:
                rounds.append({'together_ms': elapsed, 'apart_ms': times, 'identical': split == whole,
                               'distinct_tokens': len(set(whole))})
    requests = [ms for r in rounds for ms in r['apart_ms']]
    return {'length': length, 'tokens': tokens,
            'ratio': statistics.median(sum(r['apart_ms']) / r['together_ms'] for r in rounds),
            'together_ms': statistics.median(r['together_ms'] for r in rounds),
            'apart_ms': statistics.median(sum(r['apart_ms']) for r in rounds),
            'fastest_request_ms': min(requests), 'median_request_ms': statistics.median(requests),
            'slowest_request_ms': max(requests), 'identical': all(r['identical'] for r in rounds),
            'distinct_tokens': statistics.median(r['distinct_tokens'] for r in rounds), 'rounds': rounds}
