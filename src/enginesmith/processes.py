"""Process groups, the engine registry and the memory watchdog.

Every engine and every agent session runs in its own process group, so stopping it also stops whatever it started.
Engines started on behalf of a session (by `enginesmith dev-eval`) are recorded in the session's registry directory,
so the harness can stop any that outlive the session: a leaked engine holding tens of gigabytes can push a machine
into swap or a kernel panic.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def kill_group(pgid: int, sig: int = signal.SIGKILL) -> bool:
    """Signal a process group; False when it no longer exists."""
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def stop(process: subprocess.Popen, grace: float = 10.0) -> dict:
    """Stop a process started with start_new_session=True: wait up to grace seconds for it to exit on its own, then
    SIGTERM its group, then SIGKILL. Whatever is left of the group is killed even after a clean exit."""
    signals = []
    for sig, wait in ((None, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        if sig is not None and process.poll() is None:
            kill_group(process.pid, sig)
            signals.append(sig.name)
        try:
            process.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    kill_group(process.pid)
    return {'exit_code': process.returncode, 'signals': signals}


class Registry:
    """A directory of process-group ids (one empty file per group)."""

    def __init__(self, directory: Path | str | None):
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def add(self, pgid: int) -> None:
        if self.directory:
            (self.directory / str(pgid)).touch()

    def remove(self, pgid: int) -> None:
        if self.directory:
            (self.directory / str(pgid)).unlink(missing_ok=True)

    def kill_all(self) -> list[int]:
        """Kill every registered group; returns the ids that still existed."""
        killed = []
        if self.directory and self.directory.is_dir():
            for entry in self.directory.iterdir():
                if entry.name.isdigit() and kill_group(int(entry.name)):
                    killed.append(int(entry.name))
                entry.unlink(missing_ok=True)
        return killed


class Watchdog:
    """While active, kills every registered engine group when the platform reports critical memory pressure, and
    records it."""

    def __init__(self, registries, on_kill=None, interval: float = 2.0, critical=None):
        from enginesmith.platforms import current
        self.registries, self.on_kill, self.interval = registries, on_kill, interval
        self.critical = critical or current().memory_critical
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            if self.critical():
                killed = [pgid for registry in self.registries for pgid in registry.kill_all()]
                if killed and self.on_kill:
                    self.on_kill({'event': 'memory_critical', 'killed': killed, 'time': time.time()})
