import os
import sys
import time

import pytest

from conftest import VOCAB, fake_engine
from enginesmith.processes import Registry, kill_group
from enginesmith.protocol import Engine, EngineFailure


def test_hello_generate_and_continue(tmp_path):
    with Engine(fake_engine(), scratch=tmp_path / 's', label='e') as engine:
        assert engine.hello(10) >= 0
        first, ms = engine.generate([1, 2, 3], 1, True, VOCAB, 10)
        rest, _ = engine.generate([], 4, False, VOCAB, 10)
        whole, _ = engine.generate([1, 2, 3], 5, True, VOCAB, 10)
        assert first + rest == whole and ms > 0


def test_an_error_reply_and_a_wrong_token_count_fail(tmp_path):
    with Engine(fake_engine(), scratch=tmp_path / 's', label='e') as engine:
        engine.hello(10)
        with pytest.raises(EngineFailure) as error:
            engine.call('judge', 10)
        assert error.value.kind == 'error'
        with pytest.raises(EngineFailure, match='token ids below 10'):
            engine.generate([1], 2, True, 10, 10)


def test_a_crash_and_a_timeout_are_reported(tmp_path):
    with Engine(fake_engine('--crash-after', '1'), scratch=tmp_path / 'a', label='e') as engine:
        engine.hello(10)
        engine.generate([1], 2, True, VOCAB, 10)
        with pytest.raises(EngineFailure) as error:
            engine.generate([1], 2, True, VOCAB, 10)
        assert error.value.kind == 'crash'
    with Engine(fake_engine('--load-s', '5'), scratch=tmp_path / 'b', label='e') as engine:
        started = time.monotonic()
        with pytest.raises(EngineFailure) as error:
            engine.hello(0.5)
        assert error.value.kind == 'timeout' and time.monotonic() - started < 3


def test_close_stops_the_whole_process_group_and_unregisters(tmp_path):
    registry = Registry(tmp_path / 'registry')
    engine = Engine(fake_engine('--leak'), scratch=tmp_path / 's', label='e', registry=registry)
    engine.hello(10)
    pgid = engine.process.pid
    assert (tmp_path / 'registry' / str(pgid)).exists()
    engine.close(grace=0.5)
    assert not (tmp_path / 'registry' / str(pgid)).exists()
    time.sleep(0.2)
    assert not kill_group(pgid, 0)


def test_a_large_prompt_crosses_the_pipe(tmp_path):
    with Engine(fake_engine(), scratch=tmp_path / 's', label='e') as engine:
        engine.hello(10)
        tokens, _ = engine.generate(list(range(VOCAB)) * 300, 2, True, VOCAB, 30)
        assert len(tokens) == 2


def test_the_engine_gets_a_clean_environment(tmp_path):
    os.environ['ENGINESMITH_TEST_SECRET'] = 'x'
    script = tmp_path / 'env.py'
    script.write_text('import json, os, sys\nfor line in sys.stdin:\n    r = json.loads(line)\n'
                      '    print(json.dumps({"id": r["id"], "ok": True, "protocol": 1, "env": sorted(os.environ)}),'
                      ' flush=True)\n')
    with Engine([sys.executable, str(script)], scratch=tmp_path / 's', label='e') as engine:
        reply, _ = engine.call('hello', 10)
    assert 'ENGINESMITH_TEST_SECRET' not in reply['env'] and 'HOME' in reply['env']
