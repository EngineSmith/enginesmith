"""Lily workloads: raw token-ID prompts from a public formula (standard library only).

Token i of the Lily sequence is ((i * 2654435761) mod 2**32) mod vocab. A prompt of length L at offset o is the
tokens at indexes o .. o+L-1. No tokenizer, BOS or chat template is involved, so one workload serves every model;
only the vocabulary size differs.

The index period [0, 2**32) is split into three disjoint ranges:
- PUBLIC [0, MAX_LENGTH): development prompts start at offset 0, so every shorter one is a prefix of every longer
  one. Coding agents may use them freely.
- HELD_OUT [MAX_LENGTH, 2**31): the final evaluation's windows, drawn from a seed that is created only after the
  last coding session has ended.
- NONCE [2**31, 2**32): a fresh window for every timed request of a formal evaluation, so an engine cannot answer a
  timed request from a cache.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

MULTIPLIER = 2654435761
LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)
MAX_LENGTH = LENGTHS[-1]
# Correctness-only prompts at the boundaries where tiled and chunked kernels usually break.
CONTROL_LENGTHS = (1, 2, 3, 5, 63, 64, 65, 127, 128, 129, 255, 257, 2047, 2048, 2049, 4095, 4097)
PUBLIC = (0, MAX_LENGTH)
HELD_OUT = (MAX_LENGTH, 2**31)
NONCE = (2**31, 2**32)


def prompt(length: int, vocab: int, offset: int = 0) -> list[int]:
    """The Lily tokens at indexes offset .. offset+length-1."""
    if type(length) is not int or length < 1:
        raise ValueError('prompt length must be a positive integer')
    if type(vocab) is not int or not 2 <= vocab <= 2**31:
        raise ValueError('vocabulary size must be an integer in [2, 2**31]')
    if type(offset) is not int or offset < 0 or offset + length > 2**32:
        raise ValueError('a Lily window must lie inside [0, 2**32)')
    return [((i * MULTIPLIER) & 0xFFFFFFFF) % vocab for i in range(offset, offset + length)]


def draw_offset(seed: bytes, label: str, length: int, span: tuple[int, int]) -> int:
    """A window start in span such that the whole window [start, start+length) lies inside it, derived from seed and
    a label that is unique per window (for example 'public/round-2/p4096')."""
    start, stop = span
    if not 0 < length <= stop - start:
        raise ValueError('the window does not fit the range')
    digest = hashlib.sha256(seed + b'\0' + label.encode()).digest()
    return start + int.from_bytes(digest, 'big') % (stop - start - length + 1)


@dataclass(frozen=True)
class Case:
    """One prompt shape. weight 0 marks a correctness-only control case (never timed)."""
    id: str
    length: int
    output_tokens: int
    weight: float

    @property
    def timed(self) -> bool:
        return self.weight > 0


def cases(lengths, output_tokens: int, weights: dict[int, float] | None = None, controls: bool = True) -> list[Case]:
    """The timed cases (one per length, weight from weights, default 1) and, when controls, the control cases."""
    timed = [Case(f'p{n}', n, output_tokens, float((weights or {}).get(n, 1.0))) for n in lengths]
    extra = [Case(f'control-p{n}', n, output_tokens, 0.0) for n in CONTROL_LENGTHS if n not in set(lengths)] \
        if controls else []
    return timed + extra
