import json
import sys
from pathlib import Path

import pytest

from enginesmith.agents import MARKERS

TESTS = Path(__file__).resolve().parent
FAKE_ENGINE = TESTS / 'fake_engine.py'
FAKE_AGENT = TESTS / 'fake_agent.py'
VOCAB = 1000


def fake_engine(*flags) -> list[str]:
    return [sys.executable, str(FAKE_ENGINE), '--model', 'unused', '--vocab', str(VOCAB), *flags]


@pytest.fixture(autouse=True)
def outside_any_agent(monkeypatch):
    """Every test runs as if in a plain terminal, whether it runs in an agent's shell or in CI."""
    for _, names in MARKERS:
        for name in names:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def model_dir(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'vocab_size': VOCAB, 'num_hidden_layers': 2}))
    return model


def write_task(directory: Path, model: Path, **sections) -> Path:
    """A task file for the fake engines; sections override or extend the defaults (a dict per TOML table)."""
    reference = fake_engine('--reference')
    tables = {
        'model': {'path': str(model)},
        'reference': {'command': reference},
        'engine': {'build': ['/usr/bin/true'], 'executable': 'engine', 'native': False},
        'agent': {'kind': 'command', 'command': [sys.executable, str(FAKE_AGENT)], 'session_timeout': '2m'},
        'workload': {'lengths': [64, 256], 'output_tokens': 8, 'controls': False},
        'evaluation': {'rounds': 2, 'warmup': 0, 'cooldown': 0, 'final_rounds': 2},
        'budget': {'sessions': 3, 'wall_time': '100h'},
    }
    for name, values in sections.items():
        tables[name] = {**tables.get(name, {}), **values}

    def value(v):
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return '[' + ', '.join(value(x) for x in v) + ']'
        if isinstance(v, dict):
            return '{ ' + ', '.join(f'"{k}" = {value(x)}' for k, x in v.items()) + ' }'
        raise TypeError(v)
    lines = ['name = "test-task"', '']
    for name, values in tables.items():
        lines.append(f'[{name}]')
        lines += [f'{k} = {value(v)}' for k, v in values.items()]
        lines.append('')
    path = directory / 'task.toml'
    path.write_text('\n'.join(lines))
    return path
