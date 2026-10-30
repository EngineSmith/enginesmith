#!/usr/bin/env python3
"""Driver for an engine binary that speaks the EngineSmith engine protocol (version 1), plus nonce prompts and a GPU
lock. Python standard library only; no host paths.

The engine is started as `ENGINE --model MODEL_DIR` with a clean environment and its own scratch directory, and is
sent `hello` once. Requests are JSON lines {"id", "op", ...}: `generate` with `prompt`, `max_tokens` and `reset`.

Model directory, in order of precedence: --model  >  --launch FILE ({"model": DIR})  >  $ENGINE_LAUNCH  >  the
EngineSmith session ($ENGINESMITH_SESSION, set inside a session). The vocabulary size comes from --vocab, the session
or the model's config.json.

GPU lock: an flock on one file shared by every process that touches the GPU. Path: --lock-file, else $GPU_LOCK_FILE,
else engine-gpu.lock in /tmp (in the system temp dir, deliberately not $TMPDIR, so processes with different TMPDIR
still share one lock). Every process must agree on the path; `enginesmith dev-eval` uses the same default.

Prompts: `nonce_prompt(length, offset, vocab)` is the deterministic token sequence
((i * 2654435761) mod 2**32) mod vocab for i in offset .. offset+length-1 (the Lily sequence).

As a command (takes the GPU lock itself; do not run it under gpu_lock.py):
  engine_client.py ENGINE [--model DIR] [--generate P,COUNT] [--split P,DECODE] [--offset N]
prints the hello time and, optionally, one timed request on a nonce prompt.

As a module:
  from engine_client import Engine, gpu_lock, load_launch, nonce_prompt
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_ENV = 'GPU_LOCK_FILE'
LAUNCH_ENV = 'ENGINE_LAUNCH'
SESSION_ENV = 'ENGINESMITH_SESSION'
HELD_ENV = 'GPU_LOCK_HELD'
CLEAN_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'


def nonce_prompt(length, offset=0, vocab=None):
    """Deterministic prompt of `length` token ids below `vocab`."""
    if not vocab:
        raise ValueError("nonce_prompt needs the vocabulary size (--vocab, the session or the model's config.json)")
    return [((i * 2654435761) & 0xFFFFFFFF) % vocab for i in range(offset, offset + length)]


lily_prompt = nonce_prompt  # the name used by the original workspace tools


def default_lock_path():
    env = os.environ.get(LOCK_ENV)
    if env:
        return env
    base = '/tmp' if os.path.isdir('/tmp') else tempfile.gettempdir()
    return os.path.join(base, 'engine-gpu.lock')


@contextmanager
def gpu_lock(path=None):
    """Hold the exclusive GPU lock for the duration of the block (blocks until it is free). Refuses, instead of
    hanging, when a parent gpu_lock.py already holds the same lock ($GPU_LOCK_HELD)."""
    path = path or default_lock_path()
    if os.environ.get(HELD_ENV) == os.path.realpath(path):
        raise RuntimeError(f'the GPU lock {path} is already held by a parent gpu_lock.py; run this tool without '
                           f'gpu_lock.py (it takes the lock itself), otherwise it deadlocks')
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Launch:
    """Model directory and the vocabulary size for prompts."""

    def __init__(self, model, vocab=None):
        if not model:
            raise ValueError('a launch needs a model directory (--model, --launch FILE, $ENGINE_LAUNCH, or a '
                             'EngineSmith session)')
        self.model = str(model)
        if vocab is None:
            try:
                config = json.loads((Path(model) / 'config.json').read_text())
                vocab = (config.get('text_config') or {}).get('vocab_size') or config.get('vocab_size')
            except (OSError, ValueError):
                vocab = None
        self.vocab = int(vocab) if vocab else None


def add_launch_args(ap):
    g = ap.add_argument_group('launch')
    g.add_argument('--launch', default=os.environ.get(LAUNCH_ENV),
                   help=f'launch JSON file {{"model": DIR}} (default ${LAUNCH_ENV})')
    g.add_argument('--model', help='model directory (overrides the launch file and the session)')
    g.add_argument('--vocab', type=int, help="vocabulary size for prompts (default: the model's config.json)")
    g.add_argument('--lock-file', default=None,
                   help=f'GPU lock file (default ${LOCK_ENV}, else /tmp/engine-gpu.lock)')
    g.add_argument('--scratch-dir', default=None,
                   help='parent directory for per-engine scratch dirs (default: the temp dir)')
    return ap


def load_launch(args):
    """--model, else the launch file, else the EngineSmith session ($ENGINESMITH_SESSION)."""
    model, vocab = None, getattr(args, 'vocab', None)
    session = os.environ.get(SESSION_ENV)
    if session:
        task = json.loads(Path(session).read_text())['task']
        model, vocab = task['model'], vocab or task['vocab_size']
    if getattr(args, 'launch', None):
        model = json.loads(Path(args.launch).read_text()).get('model', model)
    if getattr(args, 'model', None):
        model = args.model
    return Launch(model, vocab)


class Engine:
    """One engine process speaking the engine protocol (version 1): started as `ENGINE --model DIR`, JSON lines with
    `id` and `op`; `hello`, then `generate` with `prompt`, `max_tokens` and `reset`."""

    def __init__(self, path, launch, label='engine', scratch_root=None):
        self.path, self.label, self.launch = str(Path(path).resolve()), label, launch
        if scratch_root:
            Path(scratch_root).mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.mkdtemp(prefix='eng-', dir=scratch_root)
        env = {'HOME': self.scratch, 'TMPDIR': self.scratch, 'PATH': CLEAN_PATH,
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        self.stderr_path = Path(self.scratch) / 'stderr.log'
        self._stderr = open(self.stderr_path, 'wb')
        self.proc = subprocess.Popen([self.path, '--model', launch.model],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                     cwd=self.scratch, env=env, bufsize=0, start_new_session=True)
        self.id = 0
        t = time.perf_counter()
        reply = self.request('hello')
        if reply.get('protocol') != 1:
            raise RuntimeError(f'{self.label}: hello must answer protocol 1, got {reply}')
        self.hello_s = time.perf_counter() - t

    def request(self, op, **fields):
        self.id += 1
        req = {'id': self.id, 'op': op, **fields}
        self.proc.stdin.write((json.dumps(req, separators=(',', ':')) + '\n').encode())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f'{self.label}: no reply to {op}; stderr {self.stderr_path}')
        reply = json.loads(line)
        if reply.get('ok') is not True or reply.get('id') != self.id:
            raise RuntimeError(f'{self.label}: {op} failed: {line[:300]!r}')
        return reply

    def reset(self):
        """Kept for older tools: a generate request with a prompt always starts from a fresh state."""

    def generate(self, prompt, count):
        """Greedy generate of exactly `count` tokens; returns (tokens, wall ms). A nonempty prompt starts fresh
        (reset); an empty prompt continues after the last emitted token."""
        prompt = list(prompt)
        t = time.perf_counter()
        reply = self.request('generate', prompt=prompt, max_tokens=count, reset=bool(prompt))
        return reply['tokens'], (time.perf_counter() - t) * 1000.0

    def split(self, prompt, decode=64):
        """The timed request pair: generate(P, 1) (prefill), then generate((), decode)."""
        first, prefill_ms = self.generate(prompt, 1)
        rest, decode_ms = self.generate([], decode)
        return first + rest, prefill_ms, decode_ms

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
            self.proc.wait()
        try:
            os.killpg(self.proc.pid, 9)  # anything the engine started
        except OSError:
            pass
        self._stderr.close()
        return self.proc.returncode


def _pair(text):
    a, b = text.split(',')
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='Takes the GPU lock itself: never run it under gpu_lock.py (deadlock).')
    ap.add_argument('engine', help='engine binary')
    ap.add_argument('--generate', type=_pair, metavar='P,COUNT', help='one generate(P, COUNT) after a reset')
    ap.add_argument('--split', type=_pair, metavar='P,DECODE', help='one split request generate(P,1)+generate((),DECODE)')
    ap.add_argument('--offset', type=int, default=0, help='nonce prompt offset (default 0)')
    add_launch_args(ap)
    args = ap.parse_args()
    launch = load_launch(args)
    out = {'engine': args.engine, 'model': launch.model}
    with gpu_lock(args.lock_file):
        eng = Engine(args.engine, launch, 'engine', args.scratch_dir)
        out['hello_s'] = round(eng.hello_s, 3)
        try:
            if args.generate:
                p, n = args.generate
                toks, ms = eng.generate(nonce_prompt(p, args.offset, launch.vocab), n)
                out['generate'] = {'P': p, 'count': n, 'ms': round(ms, 2), 'tokens': toks}
            if args.split:
                p, n = args.split
                toks, pf, dc = eng.split(nonce_prompt(p, args.offset, launch.vocab), n)
                out['split'] = {'P': p, 'decode': n, 'prefill_ms': round(pf, 2), 'decode_ms': round(dc, 2),
                                'decode_tok_s': round(n * 1000.0 / dc, 1), 'tokens': toks}
        finally:
            out['returncode'] = eng.close()
            out['stderr'] = str(eng.stderr_path)
    print(json.dumps(out))


if __name__ == '__main__':
    sys.exit(main())
