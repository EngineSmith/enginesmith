#!/usr/bin/env python3
"""A protocol-1 engine over a toy model, for the tests (standard library only).

The toy model's greedy next token after a sequence s is (31 * s[-1] + 7 * len(s) + 3) % VOCAB. Time is simulated
with sleeps: PREFILL_S per prompt token and DECODE_S per generated token, divided by --speed.

Flags: --reference (also answers judge and stats), --wrong-from K (tokens from step K on are off by one),
--wrong-length N (only for prompts of length N), --speed F, --crash-after N (exit after N generate requests),
--load-s S (sleep before answering hello), --leak (start a long-lived child process), --rank R (the rank a
reference reports for a mismatching token).
"""
import argparse
import json
import subprocess
import sys
import time

PREFILL_S = 2e-5
DECODE_S = 2e-3


def next_token(sequence, vocab):
    return (31 * sequence[-1] + 7 * len(sequence) + 3) % vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--vocab', type=int, default=1000)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--wrong-from', type=int)
    parser.add_argument('--wrong-length', type=int)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--crash-after', type=int)
    parser.add_argument('--load-s', type=float, default=0.0)
    parser.add_argument('--leak', action='store_true')
    parser.add_argument('--rank', type=int, default=2, help='the rank a reference reports for a mismatch')
    args = parser.parse_args()
    if args.leak:
        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    sequence, prompt_length, generated, requests = [], 0, 0, 0
    for line in sys.stdin:
        request = json.loads(line)
        op = request['op']
        reply = {'id': request['id'], 'ok': True}
        if op == 'hello':
            time.sleep(args.load_s)
            reply['protocol'] = 1
        elif op == 'generate':
            requests += 1
            if args.crash_after is not None and requests > args.crash_after:
                sys.exit(3)
            if request['reset']:
                sequence, prompt_length, generated = list(request['prompt']), len(request['prompt']), 0
                time.sleep(PREFILL_S * prompt_length / args.speed)
            tokens = []
            for _ in range(request['max_tokens']):
                token = next_token(sequence, args.vocab)
                wrong = args.wrong_from is not None and generated >= args.wrong_from and (
                    args.wrong_length is None or args.wrong_length == prompt_length)
                token = (token + 1) % args.vocab if wrong else token
                sequence.append(token)
                tokens.append(token)
                generated += 1
            time.sleep(DECODE_S * request['max_tokens'] / args.speed)
            reply['tokens'] = tokens
        elif op == 'judge' and args.reference:
            sequence = list(request['prompt'])
            mismatches = []
            for step, token in enumerate(request['tokens']):
                expected = next_token(sequence, args.vocab)
                if token != expected:
                    mismatches.append([step, 1.0, expected, args.rank])
                sequence.append(token)
            reply['mismatches'] = mismatches
        elif op == 'stats' and args.reference:
            reply.update(peak_memory=1, active_memory=1)
        else:
            reply = {'id': request['id'], 'ok': False, 'error': f'unknown op {op}'}
        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
