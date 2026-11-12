"""What EngineSmith needs from the operating system and the accelerator, behind one small interface.

A platform is a module in this package that defines:

    NAME: str                                  e.g. 'macos-metal'
    DEFAULT_REFERENCE: str | None              the reference kind (enginesmith.references) a task gets by default
    supported() -> bool                        whether the module applies to the running machine
    machine() -> dict                          facts: platform, os, arch, chip, accelerator, accelerator_cores,
                                               memory_bytes, plus anything platform-specific
    checks(facts) -> (problems, notes)         what blocks a run on this machine, and what the user should know
    memory_budget() -> int | None              bytes a model and its KV cache may use (unified memory or VRAM)
    memory_critical() -> bool                  memory is about to run out (the watchdog then kills engines)
    keep_awake() -> subprocess.Popen | None    keep the machine awake while this process lives
    ALLOWED: set[str] | None                   the libraries a native engine may link (docs/libraries.md), or None
    native_problem(path) -> str | None         why a file is not a self-contained native executable linking only
                                               ALLOWED, or None

and may define:

    engine_environment(engine_cores=None, reference_on_device=True) -> dict
                                               variables every engine and reference process gets (for example the
                                               accelerator cores it may use: engine_cores, the reference's
                                               tensor-parallel degree, for an engine; none for a reference); absent:
                                               none
    peak(cores=1) -> dict | None               the memory bandwidth ('bandwidth', bytes/s) and peak dense BF16
                                               compute ('compute', FLOP/s, or None) of `cores` accelerator cores,
                                               with their 'source' named, for the hardware bound
                                               (enginesmith.bound); absent: no bound
    busy_accelerator(log_tail: bytes) -> bool  whether a failed build's log shows the accelerator held by another
                                               process, an infrastructure failure the build is retried once for;
                                               absent: never

PLATFORMS lists them in order of preference; the first one whose supported() is true is used, and `generic` (which
supports everything and provides nothing platform-specific) comes last. CONTRIBUTING.md says how to add one.
"""
from __future__ import annotations

import functools
import importlib

PLATFORMS = ('enginesmith.platforms.macos', 'enginesmith.platforms.linux_rocm', 'enginesmith.platforms.neuron',
             'enginesmith.platforms.generic')


@functools.cache
def current():
    for name in PLATFORMS:
        module = importlib.import_module(name)
        if module.supported():
            return module
    raise RuntimeError('no platform module applies')  # unreachable: generic supports every machine
