"""The repository's structure (CONTRIBUTING.md, "Zones"): every harness file is in one zone of zones.toml, the
judging zone is what the current evaluation version says it is, the harness imports the standard library only, and
platform-specific code stays in the platform modules."""
import ast
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import asdict

from enginesmith.evaluate import EVALUATION_VERSION
from enginesmith.task import REPO, Acceptance, Evaluation, Gates

ZONES = tomllib.loads((REPO / 'zones.toml').read_text())
NAMES = ('judging', 'core', 'open', 'vendored')
PACKAGE = REPO / 'src' / 'enginesmith'
# Packages of the optional extras are imported inside functions, and only here.
OPTIONAL_IMPORTS = ('src/enginesmith/platforms/', 'src/enginesmith/references/', 'src/enginesmith/quality.py')
PLATFORM_TOOLS = ('caffeinate', 'otool', 'system_profiler', 'pmset', 'xcrun', 'sysctl', 'nvidia-smi', 'rocm-smi')


def harness_files() -> list[str]:
    return sorted(p.relative_to(REPO).as_posix() for p in PACKAGE.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store')


def zones_of(path: str) -> list[str]:
    return [name for name in NAMES for entry in ZONES[name]['files']
            if path == entry or (entry.endswith('/') and path.startswith(entry))]


def python_files(exclude=('src/enginesmith/third_party/',)) -> list[str]:
    return [path for path in harness_files() if path.endswith('.py') and not path.startswith(exclude)]


def judging_fingerprint() -> str:
    """sha256 over the judging zone's files and task.py's judging defaults. The EVALUATION_VERSION line does not
    count, so that setting the new version does not change the fingerprint it is recorded with."""
    digest = hashlib.sha256()
    for entry in sorted(ZONES['judging']['files']):
        text = re.sub(rb'^EVALUATION_VERSION = \d+$', b'EVALUATION_VERSION', (REPO / entry).read_bytes(), flags=re.M)
        digest.update(entry.encode() + b'\0' + text + b'\0')
    defaults = {'gates': asdict(Gates()), 'acceptance': asdict(Acceptance()), 'evaluation': asdict(Evaluation())}
    digest.update(json.dumps(defaults, sort_keys=True).encode())
    return digest.hexdigest()


def test_every_harness_file_is_in_exactly_one_zone():
    misplaced = {path: zones_of(path) for path in harness_files() if len(zones_of(path)) != 1}
    assert not misplaced, f'list each file in exactly one zone of zones.toml (CONTRIBUTING.md, "Zones"): {misplaced}'
    missing = [entry for name in NAMES for entry in ZONES[name]['files'] if not (REPO / entry).exists()]
    assert not missing, f'zones.toml lists files that do not exist: {missing}'


def test_the_judging_zone_is_the_current_evaluation_version():
    versions = ZONES['versions']
    assert [v['version'] for v in versions] == list(range(1, len(versions) + 1))
    assert len({v['fingerprint'] for v in versions}) == len(versions)
    latest = versions[-1]
    assert EVALUATION_VERSION == latest['version'], 'EVALUATION_VERSION must be the last version in zones.toml'
    fingerprint = judging_fingerprint()
    assert fingerprint == latest['fingerprint'], (
        'The judging zone changed (CONTRIBUTING.md, "Zones"). That needs an issue and the maintainers\' agreement '
        f'first. Then add [[versions]] to zones.toml with version = {latest["version"] + 1}, fingerprint = '
        f'"{fingerprint}" and what changed, and set EVALUATION_VERSION in src/enginesmith/evaluate.py to it.')


def test_the_harness_imports_the_standard_library_only():
    wrong = []
    for path in python_files():
        tree = ast.parse((REPO / path).read_text())
        in_functions = {id(node) for function in ast.walk(tree)
                        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                        for node in ast.walk(function)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [node.module.split('.')[0]]
            else:
                continue
            for root in roots:
                if root in sys.stdlib_module_names or root in ('enginesmith', '__future__'):
                    continue
                if id(node) not in in_functions or not path.startswith(OPTIONAL_IMPORTS):
                    wrong.append(f'{path}:{node.lineno} imports {root}')
    assert not wrong, ('the harness uses the standard library only; optional packages are imported inside functions '
                       f'of platform modules, references and quality.py: {wrong}')


def test_platform_specific_code_stays_in_the_platform_modules():
    wrong = []
    for path in python_files(exclude=('src/enginesmith/third_party/', 'src/enginesmith/platforms/',
                                      'src/enginesmith/references/')):
        text = (REPO / path).read_text()
        if re.search(r'enginesmith\.platforms\.\w+|from enginesmith\.platforms import (?!current\b)', text):
            wrong.append(f'{path} imports a platform module directly (use enginesmith.platforms.current())')
        wrong += [f'{path} runs {tool}' for tool in PLATFORM_TOOLS if re.search(rf'\b{re.escape(tool)}\b', text)]
    assert not wrong, f'platform-specific code belongs in src/enginesmith/platforms/: {wrong}'
