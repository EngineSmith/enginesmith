import json
import zipfile

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
