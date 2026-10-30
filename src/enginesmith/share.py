"""Sharing an engine (`enginesmith share`) and checking a shared one on another machine (`enginesmith verify`).

A package is a directory (and a zip of it) that holds what another user needs to build, run and check an engine:
the engine's source at its round, optionally the binary, enginesmith.json (the model's identity and file hashes, the
machine, the harness version, the task's settings and every result) and README.md, which doubles as the text of a
post in the repository's Discussions. It holds no host path, no agent command and no credential, and no model
weights: the user downloads the same model files, which verify checks by SHA-256.

verify builds the engine from the package's source (it runs the shared binary only when asked) and evaluates it
against the local mlx-lm reference with the formal evaluation's procedure, on fresh prompts, on this machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from enginesmith import __version__, engine_repo, references, workload
from enginesmith.brief import reference_kind
from enginesmith.evaluate import EVALUATION_VERSION, Evaluator, Plan
from enginesmith.platforms import current
from enginesmith.processes import Registry, Watchdog
from enginesmith.task import Gates, Objective, model_config

FORMAT = 1
ATTACHMENT_LIMIT = 25 * 1024 * 1024  # the largest file a GitHub Discussions post accepts


def model_files(model: Path) -> dict[str, str]:
    """SHA-256 of every file of a model directory (hidden files and directories excluded)."""
    digests = {}
    for path in sorted(model.rglob('*')):
        relative = path.relative_to(model)
        if path.is_file() and not any(part.startswith('.') for part in relative.parts):
            digest = hashlib.sha256()
            with open(path, 'rb') as data:
                for block in iter(lambda: data.read(1 << 24), b''):
                    digest.update(block)
            digests[relative.as_posix()] = digest.hexdigest()
    return digests


def model_revision(model: Path) -> str | None:
    """The commit a Hugging Face download came from (its .cache metadata), if there is one."""
    metadata = model / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
    try:
        return metadata.read_text().splitlines()[0].strip() or None
    except (OSError, IndexError):
        return None


def machine() -> dict:
    facts = current().machine()
    return {key: facts.get(key) for key in ('platform', 'os', 'chip', 'accelerator', 'accelerator_cores',
                                            'memory_bytes')}


def _cases(result: dict | None) -> list[dict]:
    return [{key: case[key] for key in ('id', 'length', 'prefill_speedup', 'decode_speedup', 'mismatches')}
            for case in (result or {}).get('cases', [])]


def describe(run, label: str) -> dict:
    """enginesmith.json for one round of a run."""
    t, s = run.task, run.state
    final_result = None
    if (s.get('final') or {}).get('round') == label:
        path = run.dir / 'final' / 'evaluation' / 'result.json'
        final_result = json.loads(path.read_text()) if path.is_file() else None
    quality = None  # the run's own task accuracy, or a later `enginesmith quality RUN --round` of this round
    for path in (run.dir / 'quality' / 'summary.json', run.dir / f'quality-round-{label}' / 'summary.json'):
        if path.is_file() and json.loads(path.read_text()).get('round', (s.get('quality') or {}).get('round')) == label:
            quality = json.loads(path.read_text())
    search = run.evaluation(label)
    costs = [((run.record(r['label']) or {}).get('session') or {}).get('cost_usd') for r in s['rounds']]
    return {
        'format': FORMAT,
        'name': f'{t.name}-round{label}',
        'engine': {'round': label, 'build': list(t.build), 'executable': t.executable, 'native': t.native},
        'model': {'name': t.model.name, 'source': t.model_source, 'revision': model_revision(t.model),
                  'vocab_size': t.vocab_size, 'files': model_files(t.model)},
        'machine': machine(),
        'harness': {'version': __version__, 'digest': s['harness'], 'evaluation_version': s.get('evaluation_version'),
                    'reference': reference_kind(t) or 'custom',
                    'reference_version': references.version(reference_kind(t)) if reference_kind(t) else None},
        'settings': {'lengths': list(t.lengths), 'output_tokens': t.output_tokens, 'controls': t.controls,
                     'objective': {'prefill': t.objective.prefill, 'decode': t.objective.decode,
                                   'length_weights': {str(k): v for k, v in t.objective.length_weights.items()}},
                     'gates': vars(t.gates), 'evaluation': vars(t.evaluation), 'acceptance': vars(t.acceptance)},
        'search': {'agent': t.agent if t.agent != 'command' else 'a custom agent command', 'agent_model': t.agent_model,
                   'sessions': s['sessions'], 'rounds': len(s['rounds']),
                   'agent_cost_usd': round(sum(c for c in costs if c), 4) if any(costs) else None,
                   'evaluation': search and {'score': search['score'], 'segments': search['segments'],
                                             'cases': _cases(search)}},
        'final': final_result and {'passed': final_result['passed'], 'score': final_result['score'],
                                   'segments': final_result['segments'], 'gates': final_result['gates'],
                                   'cases': _cases(final_result)},
        'quality': quality,
    }


def _x(value) -> str:
    return f'{value:.2f}x' if isinstance(value, (int, float)) else '-'


def readme(info: dict) -> str:
    """The package's README, also the text of a post in the Discussions' Engines category."""
    m, mc, final = info['model'], info['machine'], info['final']
    results = final or info['search']['evaluation'] or {'cases': []}
    rows = '\n'.join(f'| {c["length"]:,} | {_x(c["prefill_speedup"])} | {_x(c["decode_speedup"])} |'
                     for c in results['cases'])
    source = f'[{m["source"]}](https://huggingface.co/{m["source"]})' if m['source'] else m['name']
    revision = f' at revision `{m["revision"]}`' if m['revision'] else ''
    lines = [f'# {info["name"]}', '',
             f'A specialized inference engine for {source}{revision}, built with EngineSmith {info["harness"]["version"]}.',
             '', f'- Machine: {mc["accelerator"] or mc["chip"]}'
                 + (f', {mc["accelerator_cores"]} cores' if mc['accelerator_cores'] else '')
                 + f', {(mc["memory_bytes"] or 0) / 2**30:.0f} GiB, {mc["os"]}',
             f'- Reference: {info["harness"]["reference"]} {info["harness"]["reference_version"] or ""}'.rstrip(),
             f'- Judged under evaluation version {info["harness"]["evaluation_version"] or "(unrecorded)"}',
             f'- Search: {info["search"]["rounds"]} rounds by {info["search"]["agent"]}'
             + (f', agent cost ${info["search"]["agent_cost_usd"]:.2f}' if info['search']['agent_cost_usd'] else ''),
             '', f'## Speed against mlx-lm ({"held-out final evaluation" if final else "search evaluation"})', '',
             f'Score {_x(results.get("score"))}; prefill {_x((results.get("segments") or {}).get("prefill"))}, '
             f'decode {_x((results.get("segments") or {}).get("decode"))} (reference time / engine time, '
             f'{info["settings"]["output_tokens"]} tokens per request).', '',
             '| prompt tokens | prefill | decode |', '|---:|---:|---:|', rows, '']
    if final:
        g = final['gates']
        lines += [f'Correctness: {"passed" if final["passed"] else "FAILED"}; {g["total_mismatches"]} of '
                  f'{g["total_steps"]} judged steps mismatch (at most {g["allowed"]} allowed).', '']
    quality = info['quality']
    if quality:
        lines += ['## Task accuracy (greedy decoding)', '', '| benchmark | mlx-lm | this engine |', '|---|---:|---:|']
        for name, title in (('math500', 'MATH-500'), ('ifeval', 'IFEval, prompt-level strict')):
            if isinstance(quality.get(name), dict):
                r = quality[name]
                lines.append(f'| {title} ({r["paired"]} items) | {100 * r["reference"]["accuracy"]:.1f}% | '
                             f'{100 * r["engine"]["accuracy"]:.1f}% |')
        lines.append('')
    lines += ['## Use it', '',
              'These are the author\'s measurements. Check them on your own Mac (this builds the engine from '
              '`source/` and compares it with your local mlx-lm):', '', '```sh',
              f'hf download {m["source"] or "<model repository>"}' + (f' --revision {m["revision"]}' if m['revision'] else '')
              + f' --local-dir models/{m["name"]}',
              f'uv run enginesmith verify {info["name"]}.zip --model models/{m["name"]}', '```', '',
              f'Build it yourself: `{" ".join(info["engine"]["build"])}` in `source/`, then run '
              f'`{info["engine"]["executable"]} --model MODEL_DIR`; it speaks the engine protocol of EngineSmith\'s '
              '`docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is '
              'native code written by a coding agent. `bin/engine` is the author\'s build for their machine.', '',
              '`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.', '']
    return '\n'.join(lines)


def share(run, label: str | None = None, out: Path | None = None, binary: bool = True) -> dict:
    """Write the package directory and its zip; returns their paths and size."""
    label = label or run.state['incumbent']
    if not label or not (run.round_dir(label) / 'engine').is_file():
        raise ValueError('nothing to share: the run has no built engine for that round')
    record = run.record(label)
    info = describe(run, label)
    root = Path(out) if out else run.dir / 'share'
    package = root / info['name']
    if package.exists():
        shutil.rmtree(package)
    (package / 'source').mkdir(parents=True)
    archive = subprocess.run(['git', 'archive', '--format=tar', record['snapshot']], cwd=run.repo,
                             capture_output=True, check=True).stdout
    subprocess.run(['tar', '-x', '-C', str(package / 'source')], input=archive, check=True)
    if binary:
        (package / 'bin').mkdir()
        shutil.copy2(run.round_dir(label) / 'engine', package / 'bin' / 'engine')
    (package / 'enginesmith.json').write_text(json.dumps(info, indent=2))
    (package / 'README.md').write_text(readme(info))
    zipped = Path(shutil.make_archive(str(package), 'zip', root_dir=root, base_dir=info['name']))
    size = zipped.stat().st_size
    return {'package': str(package), 'zip': str(zipped), 'zip_bytes': size,
            'fits_a_discussions_post': size <= ATTACHMENT_LIMIT, 'post_text': str(package / 'README.md')}


def _unpack(package: Path, work: Path) -> Path:
    if package.is_dir():
        return package
    with zipfile.ZipFile(package) as archive:
        names = {name.split('/')[0] for name in archive.namelist()}
        if len(names) != 1 or any(name.startswith(('/', '..')) or '/../' in name for name in archive.namelist()):
            raise ValueError('the zip must hold one package directory')
        for member in archive.infolist():
            archive.extract(member, work)
            mode = (member.external_attr >> 16) & 0o777
            if mode and not member.is_dir():
                (work / member.filename).chmod(mode)  # zipfile does not restore permissions (executables)
    return work / names.pop()


def verify(package: Path, model: Path, *, out: Path | None = None, lengths=None, rounds: int | None = None,
           use_binary: bool = False, reference=None, log=print) -> dict:
    """Build a shared engine and evaluate it against the local reference; returns and writes the result."""
    package, model = Path(package).resolve(), Path(model).resolve()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    work = Path(out) if out else package.parent / f'{package.name.removesuffix(".zip")}.verify-{stamp}'
    work.mkdir(parents=True, exist_ok=True)
    root = _unpack(package, work)
    info = json.loads((root / 'enginesmith.json').read_text())
    if info.get('format') != FORMAT:
        raise ValueError(f'unknown package format {info.get("format")}')
    log('checking the model files (SHA-256) ...')
    local = model_files(model)
    differ = sorted(name for name, digest in info['model']['files'].items() if local.get(name) != digest)
    if differ:
        raise ValueError(f'the model files differ from the package\'s ({", ".join(differ[:5])}); download '
                         f'{info["model"]["source"] or info["model"]["name"]} at revision {info["model"]["revision"]}')
    judged = info['harness'].get('evaluation_version')
    if judged != EVALUATION_VERSION:
        log(f'the package was judged under evaluation version {judged}, this harness judges under version '
            f'{EVALUATION_VERSION} (zones.toml says what changed): its claimed numbers may not compare')
    vocab = model_config(model).get('vocab_size') if (model / 'config.json').is_file() else None
    vocab = vocab or info['model']['vocab_size']
    if use_binary:
        executable = root / 'bin' / 'engine'
        log('using the shared binary (bin/engine), not a build from source')
    else:
        source = work / 'build'
        if source.exists():
            shutil.rmtree(source)
        shutil.copytree(root / 'source', source)
        log(f'building: {" ".join(info["engine"]["build"])}')
        built = engine_repo._run_logged(list(info['engine']['build']), source, work / 'build.log',
                                        engine_repo.BUILD_TIMEOUT, dict(os.environ))
        if built['exit_code'] != 0:
            raise RuntimeError(f'the build failed; see {work / "build.log"}')
        executable = source / info['engine']['executable']
        if info['engine']['native'] and (reason := current().native_problem(executable)):
            raise RuntimeError(f'{info["engine"]["executable"]} is {reason}')
    s = info['settings']
    chosen = list(lengths or s['lengths'])
    objective = Objective(s['objective']['prefill'], s['objective']['decode'],
                          {int(k): v for k, v in s['objective']['length_weights'].items()})
    cases = workload.cases(chosen, s['output_tokens'], objective.length_weights, s['controls'])
    plan = Plan(tuple(cases), vocab, rounds or s['evaluation']['rounds'], s['evaluation']['warmup'],
                s['evaluation']['cooldown_s'], Gates(**s['gates']), objective, 'nonce', os.urandom(32))
    kind = info['harness']['reference'] if info['harness']['reference'] in references.KINDS else \
        current().DEFAULT_REFERENCE
    if reference is None and kind is None:
        raise ValueError('no reference for this package on this platform')
    reference = reference or references.command(kind, model)
    registry = Registry(work / 'engines')
    with Watchdog([registry]):
        result = Evaluator(reference, [str(executable), '--model', str(model)], plan, work / 'evaluation',
                           registry=registry, log=log).run()
    summary = {'package': info['name'], 'machine': machine(), 'passed': result['passed'], 'failure': result['failure'],
               'score': result['score'], 'segments': result['segments'], 'gates': result['gates'],
               'cases': _cases(result), 'claimed': info['final'] or info['search']['evaluation'],
               'evaluation_version': {'package': judged, 'here': EVALUATION_VERSION},
               'built_from_source': not use_binary}
    (work / 'verify.json').write_text(json.dumps(summary, indent=2))
    (work / 'evaluation' / 'result.json').write_text(json.dumps(result, indent=2))
    summary['directory'] = str(work)
    return summary
