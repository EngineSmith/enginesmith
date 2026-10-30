import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import fake_engine, write_task
from enginesmith.run import Run
from enginesmith.share import share, verify
from enginesmith.task import load_task


@pytest.fixture
def finished_run(tmp_path, model_dir, monkeypatch):
    monkeypatch.setenv('FAKE_AGENT_PLAN', json.dumps({'001': ['--speed', '2']}))
    (model_dir / 'weights.bin').write_bytes(b'w' * 100)
    task = load_task(write_task(tmp_path, model_dir, budget={'sessions': 1}))
    run = Run.create(task, tmp_path / 'runs')
    run.drive()
    assert run.state['final']['passed']
    return run


def test_share_writes_a_package_without_host_paths(finished_run, tmp_path):
    result = share(finished_run, out=tmp_path / 'out')
    package = tmp_path / 'out' / 'test-task-round001'
    assert result['package'] == str(package) and result['fits_a_discussions_post']
    info = json.loads((package / 'enginesmith.json').read_text())
    assert info['engine']['round'] == '001' and info['final']['passed'] and info['final']['score'] > 1.4
    assert set(info['model']['files']) == {'config.json', 'weights.bin'}
    assert (package / 'source' / 'engine').is_file() and (package / 'bin' / 'engine').is_file()
    readme = (package / 'README.md').read_text()
    assert 'enginesmith verify test-task-round001.zip' in readme and '| 64 |' in readme
    for text in (readme, (package / 'enginesmith.json').read_text()):
        assert str(tmp_path) not in text and 'FAKE_AGENT' not in text and 'fake_agent' not in text
    with zipfile.ZipFile(result['zip']) as archive:
        assert 'test-task-round001/enginesmith.json' in archive.namelist()
    source_only = share(finished_run, out=tmp_path / 'source-only', binary=False)
    assert not (tmp_path / 'source-only' / 'test-task-round001' / 'bin').exists() and source_only['zip_bytes'] > 0


def test_verify_builds_the_shared_source_and_evaluates_it_here(finished_run, tmp_path, model_dir):
    shared = share(finished_run, out=tmp_path / 'out')
    result = verify(tmp_path / 'out' / 'test-task-round001.zip', model_dir, out=tmp_path / 'check',
                    reference=fake_engine('--reference'), log=lambda m: None)
    assert result['passed'] and result['built_from_source'] and 1.4 < result['score'] < 2.6
    assert result['claimed']['passed'] and (tmp_path / 'check' / 'verify.json').is_file()
    assert shared['zip'].endswith('.zip')


def test_verify_refuses_other_model_files(finished_run, tmp_path, model_dir):
    share(finished_run, out=tmp_path / 'out')
    (model_dir / 'weights.bin').write_bytes(b'x' * 100)
    with pytest.raises(ValueError, match='model files differ'):
        verify(tmp_path / 'out' / 'test-task-round001', model_dir, out=tmp_path / 'check',
               reference=fake_engine('--reference'), log=lambda m: None)
    # A published package is verified from its repository's URL.
    package = tmp_path / 'out' / 'test-task-round001'
    for command in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'p']):
        subprocess.run(['git', *command], cwd=package, check=True)
    with pytest.raises(ValueError, match='model files differ'):
        verify(f'file://{package}', model_dir, out=tmp_path / 'check-url', reference=fake_engine('--reference'),
               log=lambda m: None)
    assert (tmp_path / 'check-url' / 'test-task-round001' / 'enginesmith.json').is_file()


@pytest.fixture
def fake_gh(tmp_path, monkeypatch):
    """A gh on PATH that records its calls (tests/fake_gh.py)."""
    bin_dir = tmp_path / 'gh-bin'
    bin_dir.mkdir()
    (bin_dir / 'gh').write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).parent / "fake_gh.py"}" "$@"\n')
    (bin_dir / 'gh').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}:{os.environ["PATH"]}')
    monkeypatch.setenv('FAKE_GH_LOG', str(tmp_path / 'gh.log'))
    return tmp_path / 'gh.log'


def test_share_publishes_a_repository_and_a_post(finished_run, tmp_path, fake_gh):
    result = share(finished_run, out=tmp_path / 'out', post=True)
    assert result['posted'] == {'repository': 'https://github.com/alice/enginesmith-test-task-round001',
                                'discussion': 'https://github.com/EngineSmith/enginesmith/discussions/7'}
    calls = [json.loads(line) for line in fake_gh.read_text().splitlines()]
    assert [c[:2] for c in calls] == [['api', 'user'], ['api', 'graphql'], ['repo', 'create'], ['api', 'graphql']]
    assert calls[2][2] == 'alice/enginesmith-test-task-round001' and '--public' in calls[2]
    body = fake_gh.with_suffix('.body.md').read_text()
    assert 'https://github.com/alice/enginesmith-test-task-round001' in body and '## How the search went' in body
    assert 'enginesmith verify https://github.com/alice/enginesmith-test-task-round001' in body
    assert str(tmp_path) not in body


def test_share_creates_nothing_when_it_cannot_post(finished_run, tmp_path, fake_gh, monkeypatch):
    monkeypatch.setenv('FAKE_GH_NO_CATEGORY', '1')
    result = share(finished_run, out=tmp_path / 'out', post=True)
    assert 'no Discussions category' in result['post_error'] and 'gh auth login' in result['next']
    calls = [json.loads(line)[:2] for line in fake_gh.read_text().splitlines()]
    assert ['repo', 'create'] not in calls and (tmp_path / 'out' / 'test-task-round001' / 'README.md').is_file()
    monkeypatch.setenv('PATH', '/usr/bin:/bin')   # no gh at all
    result = share(finished_run, out=tmp_path / 'out', post=True)
    assert 'not installed' in result['post_error'] and 'posted' not in result
