"""The skills: frontmatter, documents small enough for one read, every named file present, no host paths, English
only, helper scripts on the standard library; and the default skill's tools work against a protocol-1 engine."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FAKE_ENGINE
from enginesmith.task import REPO

SKILLS = sorted(p.parent for p in (REPO / 'skills').glob('*/SKILL.md'))
DEFAULT = REPO / 'skills' / 'specialized-inference-engine'
HOST = re.compile(r'/Users/|/private/|claude-501|scratchpad')
NAMED = re.compile(r'`((?:references|backends|examples|scripts)/[A-Za-z0-9_./-]+)`')


def test_both_skills_exist():
    assert [p.name for p in SKILLS] == ['enginesmith-setup', 'specialized-inference-engine']


@pytest.mark.parametrize('skill', SKILLS, ids=lambda p: p.name)
def test_the_frontmatter_names_the_directory(skill):
    head = (skill / 'SKILL.md').read_text().split('---\n')[1]
    fields = dict(line.split(': ', 1) for line in head.strip().splitlines())
    assert fields['name'] == skill.name and len(fields['description']) > 100


@pytest.mark.parametrize('path', sorted(p for s in SKILLS for p in s.rglob('*.md')),
                         ids=lambda p: p.relative_to(REPO / 'skills').as_posix())
def test_each_document_is_readable_in_one_view_and_portable(path):
    text = path.read_text()
    assert len(text.encode()) < 51200
    assert not HOST.search(text)
    assert not re.search(r'[㐀-鿿]', text)


@pytest.mark.parametrize('skill', SKILLS, ids=lambda p: p.name)
def test_every_file_a_skill_names_exists(skill):
    named = {match for path in skill.rglob('*.md') for match in NAMED.findall(path.read_text())}
    missing = [name for name in named if not (skill / name.rstrip('/')).exists()]
    assert not missing, missing


@pytest.mark.parametrize('path', sorted((DEFAULT / 'scripts').glob('*.py')), ids=lambda p: p.name)
def test_each_script_imports_only_the_standard_library(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    local = {p.stem for p in (DEFAULT / 'scripts').glob('*.py')}
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
        for name in names:
            assert name.split('.')[0] in sys.stdlib_module_names | local, (path.name, name)


def test_same_and_ab_run_against_a_protocol_1_engine(tmp_path):
    engine = tmp_path / 'engine'
    engine.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_ENGINE}" --vocab 1000 "$@"\n')
    engine.chmod(0o755)
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model' / 'config.json').write_text(json.dumps({'vocab_size': 1000}))
    common = ['--model', str(tmp_path / 'model'), '--lock-file', str(tmp_path / 'lock')]
    scripts = DEFAULT / 'scripts'
    same = subprocess.run([sys.executable, scripts / 'same.py', engine, engine, '--out', tmp_path / 'same.json',
                           *common], capture_output=True, text=True, timeout=300)
    assert same.returncode == 0, same.stderr
    assert json.loads(same.stdout)['bad'] == []
    ab = subprocess.run([sys.executable, scripts / 'ab.py', engine, engine, '--out', tmp_path / 'ab.json', '--lengths',
                         '64', '--rounds', '2', '--warmup', '0', '--decode', '4', *common],
                        capture_output=True, text=True, timeout=300)
    assert ab.returncode == 0 and 'tokens identical: True' in ab.stdout, ab.stderr
