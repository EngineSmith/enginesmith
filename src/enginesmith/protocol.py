"""The engine protocol client (docs/protocol.md): one engine process, one JSON object per line on stdin and stdout.

The parent times every request from just before it writes the request until the complete reply line has arrived,
and ignores any timing the engine reports.
"""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from pathlib import Path

from enginesmith.processes import Registry, stop

PROTOCOL = 1
MAX_REPLY_BYTES = 64 * 1024 * 1024
CLEAN_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'


class EngineFailure(Exception):
    """The engine broke the protocol, crashed, timed out or answered with an error."""

    def __init__(self, kind: str, message: str):
        super().__init__(f'{kind}: {message}')
        self.kind, self.message = kind, message


def engine_environment(scratch: Path, extra: dict | None = None) -> dict:
    """A small, explicit environment: no inherited variables apart from those given."""
    env = {'PATH': CLEAN_PATH, 'HOME': str(scratch), 'TMPDIR': str(scratch), 'LANG': 'C.UTF-8',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    env.update(extra or {})
    return env


class Engine:
    """One running engine. Use as a context manager; close() stops the whole process group."""

    def __init__(self, argv, *, scratch: Path, label: str, registry: Registry | None = None, env: dict | None = None):
        scratch.mkdir(parents=True, exist_ok=True)
        self.label, self.registry, self.scratch = label, registry, scratch
        self.stderr_path = scratch / 'stderr.log'
        with open(self.stderr_path, 'ab') as stderr:
            self.process = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                            cwd=scratch, env=env or engine_environment(scratch),
                                            start_new_session=True, close_fds=True)
        if registry:
            registry.add(self.process.pid)
        self._id = 0
        self._buffer = b''
        self.closed = None
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _exchange(self, data: bytes, timeout: float) -> tuple[bytes, float]:
        """Write data and read one reply line; returns the line and the elapsed milliseconds."""
        deadline = time.monotonic() + timeout
        stdin, stdout = self.process.stdin.fileno(), self.process.stdout.fileno()
        view = memoryview(data)
        with selectors.DefaultSelector() as selector:
            selector.register(stdout, selectors.EVENT_READ)
            selector.register(stdin, selectors.EVENT_WRITE)
            writing = True
            started = time.perf_counter()
            while b'\n' not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EngineFailure('timeout', f'no reply within {timeout:.0f} s')
                for key, _ in selector.select(remaining):
                    if key.fd == stdin:
                        try:
                            view = view[os.write(stdin, view[:1 << 20]):]
                        except BrokenPipeError:
                            raise EngineFailure('crash', self._exit_note()) from None
                        if not view and writing:
                            selector.unregister(stdin)
                            writing = False
                    else:
                        chunk = os.read(stdout, 1 << 20)
                        if not chunk:
                            raise EngineFailure('crash', self._exit_note())
                        self._buffer += chunk
                        if len(self._buffer) > MAX_REPLY_BYTES:
                            raise EngineFailure('protocol', 'reply larger than 64 MiB')
            elapsed = (time.perf_counter() - started) * 1000.0
        line, self._buffer = self._buffer.split(b'\n', 1)
        if self._buffer:
            raise EngineFailure('protocol', 'output after the reply line')
        if writing:
            raise EngineFailure('protocol', 'replied before reading the whole request')
        return line, elapsed

    def _exit_note(self) -> str:
        try:
            code = self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return 'closed its output'
        return f'exited with status {code}'

    def call(self, op: str, timeout: float, **fields) -> tuple[dict, float]:
        if self.closed is not None:
            raise EngineFailure('protocol', 'engine already closed')
        self._id += 1
        request = json.dumps({'id': self._id, 'op': op, **fields}, separators=(',', ':')).encode() + b'\n'
        line, elapsed = self._exchange(request, timeout)
        try:
            reply = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raise EngineFailure('protocol', f'reply is not JSON: {line[:200]!r}') from None
        if type(reply) is not dict or reply.get('id') != self._id:
            raise EngineFailure('protocol', f'reply without the request id {self._id}')
        if reply.get('ok') is not True:
            raise EngineFailure('error', str(reply.get('error', 'ok is not true'))[:500])
        return reply, elapsed

    def hello(self, timeout: float) -> float:
        """Wait until the model is loaded; returns the load time in milliseconds."""
        reply, elapsed = self.call('hello', timeout)
        if reply.get('protocol') != PROTOCOL:
            raise EngineFailure('protocol', f'hello must answer protocol {PROTOCOL}')
        return elapsed

    def generate(self, prompt, max_tokens: int, reset: bool, vocab: int, timeout: float) -> tuple[list[int], float]:
        """Greedy tokens and the request time in milliseconds."""
        reply, elapsed = self.call('generate', timeout, prompt=prompt, max_tokens=max_tokens, reset=reset)
        tokens = reply.get('tokens')
        if (type(tokens) is not list or len(tokens) != max_tokens
                or not all(type(t) is int and 0 <= t < vocab for t in tokens)):
            raise EngineFailure('protocol', f'tokens must be a list of {max_tokens} token ids below {vocab}')
        return tokens, elapsed

    def close(self, grace: float = 10.0) -> dict:
        if self.closed is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            self.closed = stop(self.process, grace)
            self.process.stdout.close()
            if self.registry:
                self.registry.remove(self.process.pid)
        return self.closed

    def stderr_tail(self, limit: int = 4000) -> str:
        try:
            data = self.stderr_path.read_bytes()
        except OSError:
            return ''
        return data[-limit:].decode('utf-8', 'replace')
