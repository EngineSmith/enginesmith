#!/usr/bin/env python3
"""A scripted coding agent for the run tests.

FAKE_AGENT_PLAN (JSON) maps a round label to an action: a list of fake_engine.py flags (write engine/engine with
them), "same" (change nothing), "sleep" (sleep until stopped), or {"flags": [...], "dev_eval": [...]} (write the
engine, then run `enginesmith dev-eval` with those arguments and keep its output in engine/dev-eval.txt).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FAKE_ENGINE = Path(__file__).resolve().parent / 'fake_engine.py'


def main():
    info = json.loads(Path(os.environ['ENGINESMITH_SESSION']).read_text())
    action = json.loads(os.environ['FAKE_AGENT_PLAN']).get(info['round'], 'same')
    brief = Path(info['brief'])
    assert (brief / 'README.md').is_file() and (brief / 'SESSION.md').is_file()
    engine = Path(info['engine'])
    if action == 'sleep':
        time.sleep(600)
    if action == 'same':
        return
    flags = action['flags'] if isinstance(action, dict) else action
    script = engine / 'engine'
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_ENGINE}" --vocab 1000 {" ".join(flags)} "$@"\n')
    script.chmod(0o755)
    (engine / 'NOTES.md').write_text(f'round {info["round"]}: {flags}\n' + (brief / 'SESSION.md').read_text())
    if isinstance(action, dict) and 'dev_eval' in action:
        out = subprocess.run(['enginesmith', 'dev-eval', *action['dev_eval']], capture_output=True, text=True)
        (engine / 'dev-eval.txt').write_text(f'exit {out.returncode}\n{out.stdout}{out.stderr}')


if __name__ == '__main__':
    main()
