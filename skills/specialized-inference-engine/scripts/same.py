#!/usr/bin/env python3
"""Token-for-token (bitwise) comparison of two engine binaries on a broad prompt set. Both engines are loaded at once
and the GPU lock is held for the whole run (so do not run this under gpu_lock.py).

For each prompt, on each engine: generate(P, 65); the split request generate(P, 1) + generate((), 64); and a
continuation chain generate(P, 17) + 3 x generate((), 16). Every request with a prompt starts from a fresh state. A case is "same" when
both engines return identical tokens for all three shapes, and "self_consistent" when, within each engine, the three
shapes agree on their first 65 tokens (determinism and continuation identity).

Prompts: nonce prompts at lengths around every power-of-two chunk/tile boundary up to 4097 (with --long also
8191..32768), six random-token prompts (fixed seed), and optionally your own prompts from a JSON-lines file
(one {"id": ..., "prompt": [token ids]} per line), e.g. the task's public dev cases if the contract allows it.

usage: same.py A_ENGINE B_ENGINE --out RESULT.json [--long] [--extra-prompts FILE.jsonl [--max-extra-tokens N]]
       [launch flags, see engine_client.py]
Prints {"cases", "identical", "bad", ...}; "bad": [] means every case matched."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_client import Engine, add_launch_args, gpu_lock, load_launch, nonce_prompt  # noqa: E402

LENGTHS = [1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000,
           1023, 1024, 1025, 2047, 2048, 2049, 2051, 4095, 4096, 4097]
LONG_LENGTHS = [8191, 8192, 8193, 16384, 32768]
RANDOM_LENGTHS = (1, 7, 64, 300, 1500, 3000)
RANDOM_SEED = 20261025


def build_prompts(vocab, long=False, extra=None, max_extra=None):
    lengths = LENGTHS + (LONG_LENGTHS if long else [])
    prompts = [(f'lily-p{n}', nonce_prompt(n, 12345 + 7 * n, vocab)) for n in lengths]
    rng = random.Random(RANDOM_SEED)
    for n in RANDOM_LENGTHS:
        prompts.append((f'rand-p{n}', [rng.randrange(vocab) for _ in range(n)]))
    if extra:
        for line in Path(extra).read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if max_extra is None or len(case['prompt']) <= max_extra:
                prompts.append((str(case['id']), case['prompt']))
    return prompts


def run(eng, prompt):
    full, _ = eng.generate(prompt, 65)
    split, _, _ = eng.split(prompt, 64)
    chain, _ = eng.generate(prompt, 17)
    for _ in range(3):
        more, _ = eng.generate([], 16); chain += more
    return {'full': full, 'split': split, 'chain': chain}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', help='engine A (usually the parent)')
    ap.add_argument('b', help='engine B (the candidate)')
    ap.add_argument('--out', required=True, help='result JSON (must not exist yet)')
    ap.add_argument('--long', action='store_true', help='add 8191..32768-token prompts')
    ap.add_argument('--extra-prompts', help='JSON-lines file of {"id", "prompt"} cases to add')
    ap.add_argument('--max-extra-tokens', type=int, help='skip extra prompts longer than this')
    add_launch_args(ap)
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists(), out
    launch = load_launch(args)
    prompts = build_prompts(launch.vocab, args.long, args.extra_prompts, args.max_extra_tokens)

    with gpu_lock(args.lock_file):
        t = time.time()
        a, b = Engine(args.a, launch, 'A', args.scratch_dir), Engine(args.b, launch, 'B', args.scratch_dir)
        rows, bad = [], []
        for name, prompt in prompts:
            ra, rb = run(a, prompt), run(b, prompt)
            same = ra == rb
            self_ok = all(r['full'][:65] == r['split'][:65] and r['full'][:65] == r['chain'][:65] for r in (ra, rb))
            rows.append({'case': name, 'same': same, 'self_consistent': self_ok, 'a': ra, 'b': rb})
            if not same or not self_ok:
                bad.append(name)
        rc = (a.close(), b.close())
    out.write_text(json.dumps({'a': args.a, 'b': args.b, 'seconds': time.time() - t, 'returncodes': rc, 'bad': bad,
                               'cases': rows}, indent=1))
    print(json.dumps({'cases': len(rows), 'identical': len(rows) - len(bad), 'bad': bad, 'returncodes': rc,
                      'seconds': round(time.time() - t, 1)}))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
