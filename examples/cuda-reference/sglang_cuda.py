"""Experimental one-GPU CUDA wrapper around the existing SGLang protocol implementation.

Not registered as a built-in reference: GPU evidence and versioned judging review are outstanding.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys

from enginesmith.references.sglang import Reference as SGLangReference, serve

PINNED_VERSION = '0.5.22'


class Reference(SGLangReference):
    def __init__(self, model):
        if importlib.metadata.version('sglang') != PINNED_VERSION:
            raise RuntimeError(f'this experiment requires stock sglang=={PINNED_VERSION}')
        from enginesmith.platforms import linux_cuda
        devices = linux_cuda.visible_gpus()
        if len(devices) != 1:
            raise RuntimeError('select exactly one CUDA GPU for this reference experiment')
        self.device_uuid = devices[0]['uuid']
        # Parent owns stock Engine generation, radix-cache continuation and teacher-forced replay.
        # No modification or monkey patch of SGLang/reference modules.
        super().__init__(model)

    def _observe(self):
        from enginesmith.platforms import linux_cuda
        device = next((g for g in linux_cuda.visible_gpus() if g['uuid'] == self.device_uuid), None)
        used = device['vram_used'] if device else None
        if used is not None:
            self.peak = max(self.peak, used)
        return used

    def stats(self, reset_peak=False):
        active = self._observe()
        result = {'peak_memory': self.peak if active is not None else None, 'active_memory': active,
                  'scope': 'whole selected device; includes other processes', 'peak_scope': 'sampled after requests'}
        if reset_peak:
            self.peak = active if active is not None else 0
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    args = parser.parse_args()
    replies = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    reference = Reference(args.model)
    try:
        serve(reference, stdout=replies)
    finally:
        reference.engine.shutdown()


if __name__ == '__main__':
    main()
