#!/usr/bin/env python3
"""Paired A/B timing of two engine binaries. Both are loaded at once; each round runs the split request
(generate(P, 1) timed as prefill, then generate((), DECODE) timed as decode) on a fresh nonce prompt,
the same prompt for both arms, in A,B order on even rounds and B,A on odd rounds (ABBA). Also checks that both arms
return identical tokens for every request. Holds the GPU lock for the whole run (do not run it under gpu_lock.py).

Reported per length and segment: each arm's median, and the median over rounds of the per-round ratio A/B, which is
B's speedup (> 1: B faster). Compare ratios, not absolute times: thermal state and other GPU users move absolute
numbers by several percent. Run A against A first to learn the noise floor.

usage: ab.py A_ENGINE B_ENGINE --out RESULT.json [--lengths 256,1024,4096] [--rounds 6] [--warmup 2] [--decode 64]
       [launch flags, see engine_client.py]"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_client import Engine, add_launch_args, gpu_lock, load_launch, nonce_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', help='engine A (usually the parent)')
    ap.add_argument('b', help='engine B (the candidate)')
    ap.add_argument('--lengths', default='256,1024,4096', help='prompt lengths (default 256,1024,4096)')
    ap.add_argument('--rounds', type=int, default=6, help='timed rounds (default 6)')
    ap.add_argument('--warmup', type=int, default=2, help='untimed warm-up rounds (default 2)')
    ap.add_argument('--decode', type=int, default=64, help='decode tokens per request (default 64)')
    ap.add_argument('--out', required=True, help='result JSON (must not exist yet)')
    add_launch_args(ap)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]
    out = Path(args.out)
    assert not out.exists(), out
    launch = load_launch(args)
    vocab = launch.vocab

    with gpu_lock(args.lock_file):
        started = time.time()
        a, b = Engine(args.a, launch, 'A', args.scratch_dir), Engine(args.b, launch, 'B', args.scratch_dir)
        samples, mismatch = [], []
        for w in range(args.warmup):
            for P in lengths:
                prompt = nonce_prompt(P, 7_000_000 + 100_003 * w + P, vocab)
                a.split(prompt, args.decode); b.split(prompt, args.decode)
        for r in range(args.rounds):
            order = (a, b) if r % 2 == 0 else (b, a)
            for P in lengths:
                prompt = nonce_prompt(P, 1_000_000 * (r + 1) + 37 * P, vocab)
                res = {}
                for eng in order:
                    toks, pf, dc = eng.split(prompt, args.decode)
                    res[eng.label] = toks
                    samples.append({'round': r, 'P': P, 'arm': eng.label, 'prefill_ms': pf, 'decode_ms': dc})
                if res['A'] != res['B']:
                    first = next((i for i, (x, y) in enumerate(zip(res['A'], res['B'])) if x != y),
                                 min(len(res['A']), len(res['B'])))
                    mismatch.append({'round': r, 'P': P, 'first': first})
        rc = (a.close(), b.close())

    summary = {}
    for P in lengths:
        rows = {}
        for seg in ('prefill_ms', 'decode_ms'):
            ra = [s[seg] for s in samples if s['P'] == P and s['arm'] == 'A']
            rb = [s[seg] for s in samples if s['P'] == P and s['arm'] == 'B']
            ratios = [x / y for x, y in zip(ra, rb)]
            rows[seg] = {'A_median': st.median(ra), 'B_median': st.median(rb), 'A_over_B_median_ratio': st.median(ratios)}
        rows['A_decode_tps'] = args.decode * 1000 / rows['decode_ms']['A_median']
        rows['B_decode_tps'] = args.decode * 1000 / rows['decode_ms']['B_median']
        summary[P] = rows
    result = {'a': args.a, 'b': args.b, 'lengths': lengths, 'rounds': args.rounds, 'warmup': args.warmup,
              'decode': args.decode, 'started': started, 'seconds': time.time() - started, 'returncodes': rc,
              'token_mismatches': mismatch, 'summary': summary, 'samples': samples}
    out.write_text(json.dumps(result, indent=1))
    print(f"tokens identical: {not mismatch}  ({len(mismatch)} mismatching requests)")
    for P, rows in summary.items():
        print(f"P={P:6d}  prefill A/B {rows['prefill_ms']['A_median']:8.1f}/{rows['prefill_ms']['B_median']:8.1f} ms "
              f"(B speedup {rows['prefill_ms']['A_over_B_median_ratio']:.4f})   decode A/B {rows['decode_ms']['A_median']:6.1f}/"
              f"{rows['decode_ms']['B_median']:6.1f} ms (B speedup {rows['decode_ms']['A_over_B_median_ratio']:.4f})  "
              f"TPS {rows['A_decode_tps']:.1f} -> {rows['B_decode_tps']:.1f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
