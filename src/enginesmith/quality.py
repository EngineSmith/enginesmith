"""Task accuracy: MATH-500 and IFEval answered by the engine and by the reference (docs/design.md, "Task accuracy").

Both arms answer the same items with the model's own chat template and greedy decoding (the engine protocol has
no sampling), until an end-of-sequence token or max_tokens. An engine that is correct up to rounding then gives
mostly the same outputs as the reference, so the two accuracies differ only where the outputs do. The answer is the
text after the last </think> when the output has one, otherwise the whole output. MATH-500 is scored with
math-verify against the dataset's answer, IFEval with Google's checker (enginesmith/third_party). Accuracy is
reported, never gated. Greedy scores are not the model card's (those are usually sampled).

The datasets are downloaded once, at a pinned revision, and checked against their SHA-256. Every generation is kept
(one JSON line per item and arm), and an interrupted evaluation continues where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

from enginesmith.processes import Registry
from enginesmith.protocol import Engine, EngineFailure
from enginesmith.task import REPO

CACHE = REPO / '.cache' / 'quality'
THIRD_PARTY = Path(__file__).resolve().parent / 'third_party'
DATASETS = {
    'math500': ('HuggingFaceH4/MATH-500', '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be', 'test.jsonl',
                '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'),
    'ifeval': ('google/IFEval', '966cd89545d6b6acfd7638bc708b98261ca58e84', 'ifeval_input_data.jsonl',
               '6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2'),
}
BENCHMARKS = tuple(DATASETS)
NLTK_DATA = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers'
PUNKT = {'punkt': '51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec',
         'punkt_tab': 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106'}
MATH_SUFFIX = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
SUBSET_SEED = 20261025
CHUNK = 256
REQUEST_TIMEOUT = 900.0


def dataset(name: str) -> list[dict]:
    repo, revision, filename, sha256 = DATASETS[name]
    path = CACHE / f'{name}-{revision[:12]}.jsonl'
    if not path.is_file():
        data = _download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}', sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def items(name: str, samples: int) -> list[dict]:
    """A fixed subset: MATH-500 stratified by difficulty level, IFEval uniformly; {'id', 'text', 'item'} each."""
    rows = dataset(name)
    rng = random.Random(SUBSET_SEED)
    if name == 'math500':
        rows = [dict(row, id=row.get('unique_id', str(i))) for i, row in enumerate(rows)]
        if samples < len(rows):
            levels = {}
            for row in rows:
                levels.setdefault(row['level'], []).append(row)
            # Each level's share of samples, rounded by largest remainder so the shares add up to samples.
            exact = {level: samples * len(group) / len(rows) for level, group in levels.items()}
            quota = {level: int(value) for level, value in exact.items()}
            for level in sorted(exact, key=lambda lvl: (quota[lvl] - exact[lvl], lvl))[:samples - sum(quota.values())]:
                quota[level] += 1
            rows = [row for level, group in sorted(levels.items()) for row in rng.sample(group, quota[level])]
        return [{'id': row['id'], 'text': row['problem'] + MATH_SUFFIX, 'item': row}
                for row in sorted(rows, key=lambda row: row['id'])]
    rows = [dict(row, id=str(row['key'])) for row in rows]
    if samples < len(rows):
        rows = rng.sample(rows, samples)
    return [{'id': row['id'], 'text': row['prompt'], 'item': row} for row in sorted(rows, key=lambda r: int(r['id']))]


class Tokenizer:
    """The model's own tokenizer and chat template (transformers, offline)."""

    def __init__(self, model: Path):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model))
        config = json.loads((model / 'generation_config.json').read_text()) \
            if (model / 'generation_config.json').is_file() else {}
        ids = config.get('eos_token_id')
        self.eos = set(ids if isinstance(ids, list) else [ids] if isinstance(ids, int) else [])
        if self.tokenizer.eos_token_id is not None:
            self.eos.add(self.tokenizer.eos_token_id)

    def prompt(self, text: str) -> list[int]:
        chat = self.tokenizer.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True,
                                                  tokenize=False)
        return self.tokenizer.encode(chat, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


def answer(engine: Engine, prompt: list[int], eos: set, max_tokens: int, vocab: int) -> tuple:
    """Greedy generation in chunks until an end-of-sequence token or max_tokens; returns (tokens, 'eos' or
    'length')."""
    out = []
    while len(out) < max_tokens:
        count = min(CHUNK, max_tokens - len(out))
        tokens, _ = engine.generate(prompt, count, bool(prompt), vocab, REQUEST_TIMEOUT)
        prompt = []
        stop = next((i for i, token in enumerate(tokens) if token in eos), None)
        if stop is not None:
            return out + tokens[:stop], 'eos'
        out += tokens
    return out, 'length'


def final_text(output: str) -> str:
    return output.rsplit('</think>', 1)[1].strip() if '</think>' in output else output.strip()


def run_arm(argv, chosen: list[dict], tokenizer: Tokenizer, settings, vocab: int, path: Path, deadline: float,
            registry: Registry | None = None, stop=lambda: False, log=lambda message: None) -> list[dict]:
    """Answer every item not yet in path (JSON lines); stops at the deadline or at an engine failure."""
    done = {json.loads(line)['id'] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [item for item in chosen if item['id'] not in done]
    if todo:
        with Engine(argv, scratch=path.parent / f'scratch-{path.stem}', label=path.stem, registry=registry) as engine, \
                open(path, 'a') as out:
            try:
                engine.hello(1800.0)
                for count, item in enumerate(todo):
                    if time.time() > deadline or stop():
                        break
                    prompt = tokenizer.prompt(item['text'])
                    started = time.perf_counter()
                    tokens, finish = answer(engine, prompt, tokenizer.eos, settings.max_tokens, vocab)
                    seconds = time.perf_counter() - started
                    output = tokenizer.decode(tokens)
                    out.write(json.dumps({'id': item['id'], 'prompt_tokens': len(prompt), 'output_tokens': len(tokens),
                                          'finish': finish, 'seconds': seconds, 'final': final_text(output),
                                          'output': output}, ensure_ascii=False) + '\n')
                    out.flush()
                    log(f'{path.stem}: {len(done) + count + 1}/{len(chosen)} {len(tokens)} tokens ({finish})')
            except EngineFailure as exc:
                out.write(json.dumps({'id': None, 'error': f'{exc} {engine.stderr_tail(1000)}'}) + '\n')
                log(f'{path.stem}: stopped: {exc}')
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def score_math(rows: list[dict], chosen: dict) -> dict[str, bool]:
    from math_verify import LatexExtractionConfig, parse, verify
    scores = {}
    for row in rows:
        gold = parse('$' + chosen[row['id']]['item']['answer'] + '$', extraction_config=[LatexExtractionConfig()])
        prediction = parse(row['final'], extraction_config=[LatexExtractionConfig()]) if row['final'] else []
        scores[row['id']] = bool(prediction) and bool(verify(gold, prediction))
    return scores


def _download(url: str, sha256: str) -> bytes:
    data = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f'{url} does not have the expected SHA-256')
    return data


def _ifeval():
    """Google's checker, with the NLTK sentence tokenizer data it loads (fetched once and checked by SHA-256, not
    through nltk.download)."""
    import io
    import zipfile
    import nltk
    data = CACHE / 'nltk'
    for package, sha256 in PUNKT.items():
        if not (data / 'tokenizers' / package).is_dir():
            archive = _download(f'{NLTK_DATA}/{package}.zip', sha256)
            zipfile.ZipFile(io.BytesIO(archive)).extractall(data / 'tokenizers')
    if str(data) not in nltk.data.path:
        nltk.data.path.insert(0, str(data))
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))
    from instruction_following_eval import evaluation_lib
    return evaluation_lib


def score_ifeval(rows: list[dict], chosen: dict) -> tuple[dict[str, bool], dict]:
    """Prompt-level strict accuracy per item, and the four standard IFEval metrics."""
    ev = _ifeval()
    inputs = [ev.InputExample(key=int(row['id']), instruction_id_list=chosen[row['id']]['item']['instruction_id_list'],
                              prompt=chosen[row['id']]['item']['prompt'], kwargs=chosen[row['id']]['item']['kwargs'])
              for row in rows]
    responses = {chosen[row['id']]['item']['prompt']: row['final'] for row in rows}
    metrics, per_item = {}, {}
    for name, check in (('strict', ev.test_instruction_following_strict),
                        ('loose', ev.test_instruction_following_loose)):
        outputs = [check(inp, responses) for inp in inputs]
        followed = [x for o in outputs for x in o.follow_instruction_list]
        metrics[f'prompt_{name}'] = sum(o.follow_all_instructions for o in outputs) / max(len(outputs), 1)
        metrics[f'instruction_{name}'] = sum(followed) / max(len(followed), 1)
        if name == 'strict':
            per_item = {row['id']: o.follow_all_instructions for row, o in zip(rows, outputs)}
    return per_item, metrics


def _arm_summary(rows: list[dict], correct: dict, paired: list[str]) -> dict:
    answered = [row for row in rows if row.get('id')]
    hits = sum(correct[i] for i in paired)
    p = hits / len(paired) if paired else None
    seconds = sum(row['seconds'] for row in answered)
    return {'answered': len(answered), 'correct': hits, 'accuracy': p,
            'standard_error': math.sqrt(p * (1 - p) / len(paired)) if paired else None,
            'truncated': sum(row['finish'] != 'eos' for row in answered),
            'mean_output_tokens': sum(row['output_tokens'] for row in answered) / len(answered) if answered else None,
            'tokens_per_s': sum(row['output_tokens'] for row in answered) / seconds if seconds else None,
            'errors': [row['error'] for row in rows if row.get('error')]}


def evaluate(settings, model: Path, vocab: int, engine_argv, reference_argv, directory: Path, *,
             registry: Registry | None = None, stop=lambda: False, log=lambda message: None) -> dict:
    """Run every benchmark of settings (the task's [quality]) through both arms; returns and writes the summary."""
    directory.mkdir(parents=True, exist_ok=True)
    fixed = {'samples': settings.samples, 'max_tokens': settings.max_tokens, 'decoding': 'greedy'}
    marker = directory / 'settings.json'
    if marker.is_file() and json.loads(marker.read_text()) != fixed:
        raise ValueError(f'{directory} holds answers made with other settings ({marker}); use another directory')
    marker.write_text(json.dumps(fixed, indent=2))
    tokenizer = Tokenizer(model)
    share = settings.time_limit_s / (2 * len(settings.benchmarks))
    summary = {'settings': {'benchmarks': list(settings.benchmarks), **fixed}}
    for name in settings.benchmarks:
        chosen = items(name, settings.samples)
        by_id = {item['id']: item for item in chosen}
        rows = {}
        for arm, argv in (('reference', reference_argv), ('engine', engine_argv)):
            rows[arm] = run_arm(argv, chosen, tokenizer, settings, vocab, directory / f'{name}-{arm}.jsonl',
                                time.time() + share, registry, stop, log)
        answered = {arm: {row['id'] for row in arm_rows if row.get('id')} for arm, arm_rows in rows.items()}
        paired = sorted(answered['reference'] & answered['engine'])
        outputs = {arm: {row['id']: row['output'] for row in arm_rows if row.get('id')} for arm, arm_rows in rows.items()}
        result = {'items': len(chosen), 'paired': len(paired),
                  'identical_outputs': sum(outputs['engine'][i] == outputs['reference'][i] for i in paired)}
        for arm, arm_rows in rows.items():
            valid = [row for row in arm_rows if row.get('id')]
            if name == 'math500':
                correct, metrics = score_math(valid, by_id), {}
            else:
                correct, metrics = score_ifeval(valid, by_id)
            result[arm] = {**_arm_summary(arm_rows, correct, paired), **metrics}
            result[arm]['_correct'] = correct
        if paired:
            diffs = [result['engine']['_correct'][i] - result['reference']['_correct'][i] for i in paired]
            mean = sum(diffs) / len(diffs)
            var = sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
            result.update(difference=mean, difference_standard_error=math.sqrt(var / len(diffs)))
        for arm in ('reference', 'engine'):
            del result[arm]['_correct']
        summary[name] = result
        log(f'{name}: reference {_percent(result["reference"]["accuracy"])}, engine '
            f'{_percent(result["engine"]["accuracy"])} over {len(paired)} items answered by both '
            f'({result["identical_outputs"]} with identical outputs)')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def _percent(value) -> str:
    return 'n/a' if value is None else f'{100 * value:.1f}%'
