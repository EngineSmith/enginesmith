#!/usr/bin/env python3
"""Run one command that touches the GPU (an engine run, a Metal micro-benchmark, a profiler) under the shared GPU
lock, so that measurements never overlap. Blocks until the lock is free, runs the command, releases the lock and
exits with the command's exit status.

The lock is an flock on one file: --lock-file, else $GPU_LOCK_FILE, else /tmp/engine-gpu.lock. Every process that
uses the GPU must use the same path.

Never wrap a tool that takes the lock itself (engine_client.py, same.py, ab.py): the child would wait for a lock its
parent holds, and both would hang. This script sets $GPU_LOCK_HELD for the child so that those three tools stop with
an error instead; other tools that take an flock on the same file still deadlock.

usage: gpu_lock.py [--lock-file PATH] -- CMD ARGS..."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_client import HELD_ENV, default_lock_path, gpu_lock  # noqa: E402


def main():
    argv = sys.argv[1:]
    if '--' in argv:
        own, cmd = argv[:argv.index('--')], argv[argv.index('--') + 1:]
    else:
        own, cmd = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lock-file', default=None, help='lock file (default $GPU_LOCK_FILE, else /tmp/engine-gpu.lock)')
    args = ap.parse_args(own)
    if not cmd:
        ap.error('no command given after --')
    path = args.lock_file or default_lock_path()
    env = dict(os.environ, **{HELD_ENV: os.path.realpath(path)})  # lets a nested engine_client refuse, not hang
    with gpu_lock(path):
        return subprocess.call(cmd, env=env)


if __name__ == '__main__':
    sys.exit(main())
