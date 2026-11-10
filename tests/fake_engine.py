#!/usr/bin/env python3
"""A protocol-1 engine over a toy model, for the tests (standard library only).

The toy model's greedy next token after a sequence s is (31 * s[-1] + 7 * len(s) + 3) % VOCAB. Time is simulated
with sleeps: PREFILL_S per prompt token and DECODE_S per generated token, divided by --speed.

Flags: --reference (also answers judge and stats), --wrong-from K (tokens from step K on are off by one),
--wrong-length N (only for prompts of length N), --speed F, --crash-after N (exit after N generate requests),
--load-s S (sleep before answering hello), --leak (start a long-lived child process), --rank R (the rank a
reference reports for a mismatching token), --speculate N (one forward pass per N tokens of a request).

--concurrent speaks protocol 2 (docs/protocol.md): submit requests, streamed tokens. --batch N decodes up to N requests
per step (a step costs DECODE_S however many requests it holds; a new request's prefill pauses decoding); --batch 1
serves one request at a time. --burst sends a request's tokens only when it has all of them. --bad-id sends a token
for a request that is not in flight.
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
    parser.add_argument('--speculate', type=int, default=1)
    parser.add_argument('--crash-after', type=int)
    parser.add_argument('--load-s', type=float, default=0.0)
    parser.add_argument('--leak', action='store_true')
    parser.add_argument('--rank', type=int, default=2, help='the rank a reference reports for a mismatch')
    parser.add_argument('--concurrent', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--burst', action='store_true')
    parser.add_argument('--bad-id', action='store_true')
    args = parser.parse_args()
    if args.concurrent:
        return concurrent(args)
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
            time.sleep(DECODE_S * (-(-request['max_tokens'] // args.speculate)) / args.speed)
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


def concurrent(args):
    import queue
    import threading
    inbox = queue.Queue()

    def reader():
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put(None)
    threading.Thread(target=reader, daemon=True).start()

    def write(event):
        sys.stdout.write(json.dumps(event) + '\n')
        sys.stdout.flush()
    waiting, active, closed = [], [], False
    while True:
        block = not waiting and not active and not closed
        try:
            while True:
                request = inbox.get(block=block)
                block = False
                if request is None:
                    closed = True
                    break
                if request['op'] == 'hello':
                    time.sleep(args.load_s)
                    write({'id': request['id'], 'ok': True, 'protocol': 2})
                elif request['op'] == 'submit':
                    waiting.append({'id': request['id'], 'sequence': list(request['prompt']), 'left': request['max_tokens'],
                                    'generated': 0, 'held': []})
                else:
                    write({'id': request.get('id'), 'ok': False, 'error': f'unknown op {request["op"]}'})
        except queue.Empty:
            pass
        if closed and not waiting and not active:
            return
        while waiting and len(active) < args.batch:
            r = waiting.pop(0)
            time.sleep(PREFILL_S * len(r['sequence']) / args.speed)
            active.append(r)
        if not active:
            continue
        time.sleep(DECODE_S / args.speed)
        out = []
        for r in active:
            token = next_token(r['sequence'], args.vocab)
            if args.wrong_from is not None and r['generated'] >= args.wrong_from:
                token = (token + 1) % args.vocab
            r['sequence'].append(token)
            r['generated'] += 1
            r['left'] -= 1
            r['held'].append(token)
            if not args.burst:
                out.append([r['id'], token])
        for r in active:
            if args.burst and r['left'] == 0:
                out += [[r['id'], t] for t in r['held']]
        if args.bad_id:
            out.append(['nobody', 1])
        if out:
            write({'tokens': out})
        active = [r for r in active if r['left'] > 0]


if __name__ == '__main__':
    main()
