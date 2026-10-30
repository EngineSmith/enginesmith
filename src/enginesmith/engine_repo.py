"""Engine source snapshots and builds.

The run keeps every engine version in a bare git repository (run/engine.git). A session works in a worktree
checked out at its starting commit; when the session ends, everything in the worktree (except what .gitignore
excludes) is committed, and that commit is the candidate's snapshot. Builds run in one persistent build worktree
(run/build) that is reset to the snapshot and cleaned (except target/, so dependencies stay compiled) before every
build, so a build only ever sees committed source.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from enginesmith.platforms import current
from enginesmith.processes import stop

GIT_IDENTITY = ['-c', 'user.name=EngineSmith', '-c', 'user.email=enginesmith@localhost', '-c', 'commit.gpgsign=false']
IGNORED = 'target/\n'
BUILD_TIMEOUT = 3600.0
LOG_LIMIT = 4 * 1024 * 1024
MAX_FILE_BYTES = 50 * 1024 * 1024


class SnapshotError(RuntimeError):
    pass


def git(*args, cwd: Path, check: bool = True) -> str:
    result = subprocess.run(['git', *GIT_IDENTITY, *args], cwd=cwd, capture_output=True, text=True, timeout=600)
    if check and result.returncode != 0:
        raise SnapshotError(f'git {" ".join(args)}: {result.stderr.strip()[-2000:]}')
    return result.stdout.strip()


def init(repository: Path, seed: Path | None) -> str:
    """Create the bare repository with its first commit: the seed's files, or an empty project. Returns the commit."""
    repository.mkdir(parents=True)
    git('init', '-q', '--bare', str(repository), cwd=repository.parent)
    staging = repository.parent / 'engine-initial'
    staging.mkdir()
    if seed is not None:
        shutil.copytree(seed, staging, dirs_exist_ok=True, symlinks=True,
                        ignore=shutil.ignore_patterns('.git', 'target'))
    if not (staging / '.gitignore').exists():
        (staging / '.gitignore').write_text(IGNORED)
    git('init', '-q', cwd=staging)
    git('add', '-A', cwd=staging)
    git('commit', '-q', '--allow-empty', '-m', 'seed' if seed else 'empty engine project', cwd=staging)
    git('push', '-q', str(repository), 'HEAD:refs/heads/main', cwd=staging)
    commit = git('rev-parse', 'HEAD', cwd=staging)
    shutil.rmtree(staging)
    git('update-ref', 'refs/rounds/000', commit, cwd=repository)
    return commit


def checkout(repository: Path, commit: str, worktree: Path) -> None:
    worktree.parent.mkdir(parents=True, exist_ok=True)
    git('worktree', 'add', '-q', '--detach', '--force', str(worktree), commit, cwd=repository)


def snapshot(repository: Path, worktree: Path, label: str) -> tuple[str, list[str]]:
    """Commit everything in the worktree and keep the commit under refs/rounds/<label>. Files larger than
    MAX_FILE_BYTES (model weights, generated data) are left out. Returns the commit and the files left out."""
    if not (worktree / '.git').exists():
        raise SnapshotError('the engine directory is no longer a git checkout')
    git('add', '-A', cwd=worktree)
    staged = git('diff', '--cached', '--name-only', '--diff-filter=AM', '-z', cwd=worktree).split('\0')
    large = [name for name in staged if name and (worktree / name).is_file()
             and (worktree / name).stat().st_size > MAX_FILE_BYTES]
    if large:
        git('rm', '-q', '--cached', '--', *large, cwd=worktree)
    git('commit', '-q', '--allow-empty', '--no-verify', '-m', f'round {label}', cwd=worktree)
    commit = git('rev-parse', 'HEAD', cwd=worktree)
    git('update-ref', f'refs/rounds/{label}', commit, cwd=repository)
    return commit, large


def remove_worktree(repository: Path, worktree: Path) -> None:
    git('worktree', 'remove', '--force', str(worktree), cwd=repository, check=False)
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    git('worktree', 'prune', cwd=repository, check=False)


def changed_files(repository: Path, base: str, commit: str) -> list[str]:
    return [line for line in git('diff', '--name-only', base, commit, cwd=repository).splitlines() if line]


def _run_logged(argv, cwd: Path, log: Path, timeout: float, env: dict) -> dict:
    started = time.monotonic()
    with open(log, 'ab') as out:
        try:
            process = subprocess.Popen(argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env=env, start_new_session=True)
        except OSError as exc:
            out.write(f'cannot start {argv[0]}: {exc}\n'.encode())
            return {'exit_code': None, 'timed_out': False, 'error': str(exc), 'duration_s': 0.0}
        try:
            process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        stop(process, grace=0)
    return {'exit_code': process.returncode, 'timed_out': timed_out, 'duration_s': time.monotonic() - started}


def build(repository: Path, build_dir: Path, commit: str, argv, executable: str, native: bool, output: Path,
          log: Path, timeout: float = BUILD_TIMEOUT) -> dict:
    """Build one snapshot; on success the executable is copied to output. Returns the facts."""
    if not build_dir.exists():
        checkout(repository, commit, build_dir)
    else:
        git('checkout', '-q', '--detach', '--force', commit, cwd=build_dir)
    git('clean', '-q', '-ffdx', '-e', 'target/', cwd=build_dir)
    env = {**os.environ, 'CARGO_TERM_COLOR': 'never'}
    result = _run_logged(list(argv), build_dir, log, timeout, env)
    result['passed'] = False
    if log.stat().st_size > LOG_LIMIT:
        data = log.read_bytes()
        log.write_bytes(b'[log truncated to its last 4 MiB]\n' + data[-LOG_LIMIT:])
    if result['exit_code'] != 0:
        result['error'] = 'build timed out' if result['timed_out'] else f'build exited with {result["exit_code"]}'
        return result
    built = build_dir / executable
    if not built.is_file() or not os.access(built, os.X_OK):
        result['error'] = f'the build produced no executable at {executable}'
        return result
    if native and (reason := current().native_problem(built)):
        result['error'] = f'{executable} is {reason}'
        return result
    shutil.copy2(built, output)
    result['passed'] = True
    return result


def tail(path: Path, limit: int = 6000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ''
    return data[-limit:].decode('utf-8', 'replace')
