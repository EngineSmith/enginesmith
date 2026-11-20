import importlib.util
import sys
from pathlib import Path

import pytest

from enginesmith.task import REPO


def module(name):
    path = REPO / 'examples' / 'cuda-reference' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_conformance_checks_detect_bad_continuation_and_self_judge(monkeypatch):
    wrapper = module('sglang_cuda')
    monkeypatch.setitem(sys.modules, 'sglang_cuda', wrapper)
    checks = module('validate')
    class Toy:
        def generate(self, prompt, count, reset):
            if reset:
                self.position = 0
            result = list(range(self.position, self.position + count))
            self.position += count
            return result
        def judge(self, prompt, tokens):
            return {'mismatches': [] if tokens[0] == 0 else [[0, 2.0, 0, 200]]}
    result, tokens = checks.check_continuations(Toy(), [1, 2], 5, 1000)
    assert result['split_matches'] and result['chain_matches']
    assert result['wrong_stream_has_early_low_rank'] and not result['self_judge_mismatches']
    class Broken(Toy):
        def generate(self, prompt, count, reset):
            return super().generate(prompt, count, True)
    result, _ = checks.check_continuations(Broken(), [1, 2], 5, 1000)
    assert not result['split_matches'] and not result['chain_matches']


def test_version_and_device_checks_run_before_loading_model(monkeypatch, tmp_path):
    wrapper = module('sglang_cuda')
    monkeypatch.setattr(wrapper.importlib.metadata, 'version', lambda name: 'wrong')
    with pytest.raises(RuntimeError, match='stock sglang'):
        wrapper.Reference(tmp_path)
    monkeypatch.setattr(wrapper.importlib.metadata, 'version', lambda name: wrapper.PINNED_VERSION)
    from enginesmith.platforms import linux_cuda
    monkeypatch.setattr(linux_cuda, 'visible_gpus', lambda: [])
    with pytest.raises(RuntimeError, match='exactly one'):
        wrapper.Reference(tmp_path)


def test_cuda_memory_observer_reports_unknown_instead_of_zero(monkeypatch):
    wrapper = module('sglang_cuda')
    from enginesmith.platforms import linux_cuda
    reference = object.__new__(wrapper.Reference)
    reference.peak, reference.device_uuid = 0, 'test-device'
    monkeypatch.setattr(linux_cuda, 'visible_gpus', lambda: [{'uuid': 'test-device', 'vram_used': 100}])
    assert reference.stats()['active_memory'] == 100
    monkeypatch.setattr(linux_cuda, 'visible_gpus', lambda: [])
    assert reference.stats()['active_memory'] is None and reference.stats()['peak_memory'] is None
