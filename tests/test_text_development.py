import json
import shlex
import sys
from dataclasses import replace

import pytest

from conftest import FAKE_ENGINE, VOCAB, fake_engine, write_task
from enginesmith import cli, dev_eval, text_dev_eval, workload
from enginesmith.evaluate import Plan, ReferenceFailed
from enginesmith.task import Gates, Objective, load_task
from enginesmith.text_dataset import DatasetError, digest, load_samples, prepare
from enginesmith.text_dev_eval import TextEvaluator


@pytest.fixture
def corpus(tmp_path, model_dir):
    (model_dir / 'tokenizer.json').write_text('{"test":"tokenizer-v1"}')
    raw = tmp_path / 'corpus.jsonl'
    raw.write_text('\n'.join(json.dumps({'id': f'doc-{i}', 'text': f'Document {i}: ' + 'a story about trees. ' * 30})
                             for i in range(4)))
    prepared = tmp_path / 'prepared.jsonl'
    prepare(raw, prepared, model_dir, VOCAB, lambda t: [ord(c) for c in t], source='test-fixture',
            revision='v1', license='test-only', tokenizer_runtime='fake-codepoint-tokenizer')
    return raw, prepared, model_dir


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


def test_preparation_preserves_raw_text_and_declares_policy(corpus, tmp_path):
    raw, prepared, model = corpus
    rows = read_rows(prepared)
    assert rows[0]['policy'] == {'add_special_tokens': False, 'chat_template': None, 'text_preprocessing': 'none'}
    for original, row in zip(read_rows(raw), rows[1:]):
        assert ''.join(map(chr, row['tokens'])) == original['text']
    with pytest.raises(FileExistsError):
        prepare(raw, prepared, model, VOCAB, lambda t: [1], source='test', revision='v1', license='test',
                tokenizer_runtime='fake')


def test_samples_are_contiguous_distinct_and_reproducible(corpus):
    _, path, model = corpus
    samples, provenance = load_samples(path, model, VOCAB, [64, 256], 3, 9)
    assert (samples, provenance) == load_samples(path, model, VOCAB, [64, 256], 3, 9)
    docs = {r['id']: r for r in read_rows(path)[1:]}
    for length in (64, 256):
        group = [s for s in samples if s.length == length]
        assert len(group) == len({s.document_id for s in group}) == 3
        for s in group:
            assert list(s.tokens) == docs[s.document_id]['tokens'][s.offset:s.offset + length]
            assert s.metadata()['prompt_sha256'] == digest(s.tokens)
    assert samples != load_samples(path, model, VOCAB, [64, 256], 3, 10)[0]
    rows = read_rows(path)
    write_rows(path, [rows[0], *reversed(rows[1:])])
    assert samples == load_samples(path, model, VOCAB, [64, 256], 3, 9)[0]


@pytest.mark.parametrize('field,value', [('tokens', [True]), ('tokens', [-1]), ('tokens', [VOCAB]),
                                        ('tokens', []), ('text_sha256', 'invalid'), ('tokens_sha256', '0' * 64)])
def test_invalid_prepared_documents_are_rejected(corpus, field, value):
    _, path, model = corpus
    rows = read_rows(path)
    rows[1][field] = value
    write_rows(path, rows)
    with pytest.raises(DatasetError):
        load_samples(path, model, VOCAB, [4], 1, 0)


@pytest.mark.parametrize('change', ['duplicate-id', 'duplicate-content', 'bad-policy', 'vocabulary', 'provenance', 'json'])
def test_invalid_corpora_fail_before_sampling(corpus, change):
    _, path, model = corpus
    rows = read_rows(path)
    if change == 'duplicate-id':
        rows[2]['id'] = rows[1]['id']
    elif change == 'duplicate-content':
        rows[2]['text_sha256'] = rows[1]['text_sha256']
    elif change == 'bad-policy':
        rows[0]['policy']['add_special_tokens'] = True
    elif change == 'vocabulary':
        rows[0]['vocab_size'] = VOCAB + 1
    elif change == 'provenance':
        del rows[0]['license']
    write_rows(path, rows)
    if change == 'json':
        path.write_text('{truncated')
    with pytest.raises(DatasetError):
        load_samples(path, model, VOCAB, [4], 1, 0)


def test_changed_tokenizer_and_insufficient_coverage_are_rejected(corpus):
    _, path, model = corpus
    with pytest.raises(DatasetError, match='distinct sufficiently long'):
        load_samples(path, model, VOCAB, [4096], 3, 0)
    with pytest.raises(DatasetError, match='found 4'):
        load_samples(path, model, VOCAB, [4], 5, 0)
    (model / 'tokenizer.json').write_text('{"test":"changed"}')
    with pytest.raises(DatasetError, match='tokenizer artifacts differ'):
        load_samples(path, model, VOCAB, [4], 1, 0)


def test_truncating_a_complete_document_is_detected(corpus):
    _, path, model = corpus
    write_rows(path, read_rows(path)[:-1])
    with pytest.raises(DatasetError, match='document count or checksum'):
        load_samples(path, model, VOCAB, [4], 1, 0)


def test_tokenizer_change_during_preparation_leaves_no_output(corpus, tmp_path):
    raw, _, model = corpus
    out = tmp_path / 'changed.jsonl'
    def encode(text):
        (model / 'tokenizer.json').write_text('changed')
        return [1, 2]
    with pytest.raises(DatasetError, match='changed during preparation'):
        prepare(raw, out, model, VOCAB, encode, source='test', revision='v1', license='test', tokenizer_runtime='fake')
    assert not out.exists()


@pytest.mark.parametrize('rows', [[{'id': 'a', 'text': ''}], [{'id': 'a', 'text': 'one'}, {'id': 'a', 'text': 'two'}],
                                 [{'id': 'a', 'text': 'same'}, {'id': 'b', 'text': 'same'}], []])
def test_bad_raw_corpus_leaves_no_prepared_output(corpus, tmp_path, rows):
    raw, _, model = corpus
    write_rows(raw, rows)
    out = tmp_path / 'failed.jsonl'
    with pytest.raises(DatasetError):
        prepare(raw, out, model, VOCAB, lambda t: [1], source='test', revision='v1', license='test',
                tokenizer_runtime='fake')
    assert not out.exists()


class RecordingEvaluator(TextEvaluator):
    def __init__(self, *args, **kwargs):
        self.requests, self.replays, self.opens = [], [], []
        super().__init__(*args, **kwargs)

    def _open(self, role, label):
        self.opens.append((role, label))
        return super()._open(role, label)

    def _split(self, engine, role, case, prompt, limits=(None, None)):
        self.requests.append((role, case.id, list(prompt)))
        return super()._split(engine, role, case, prompt, limits)

    def _judge(self, streams):
        self.replays.extend(streams)
        return super()._judge(streams)


def evaluator(corpus, tmp_path, *, flags=(), controls=False, gates=Gates(), reference=None):
    _, path, model = corpus
    samples, provenance = load_samples(path, model, VOCAB, [64, 256], 2, 2)
    cases = tuple(workload.cases((), 8, controls=controls))
    cases += tuple(workload.Case(s.id, s.length, 8, 1.0) for s in samples)
    plan = Plan(cases, VOCAB, 2, 1, 0, gates, Objective(), 'public')
    return RecordingEvaluator(reference or fake_engine('--reference'), fake_engine(*flags), plan, tmp_path / 'eval',
                              text_samples=samples, provenance=provenance)


def test_pairing_replay_abba_and_separate_results(corpus, tmp_path):
    ev = evaluator(corpus, tmp_path, flags=('--wrong-from', '7'), gates=Gates(30, 30, 100))
    result = ev.run()
    assert result['passed'] and result['complete']
    assert result['score'] is result['segments'] is result['speculative_telemetry'] is None
    assert result['score_reason'] and result['speculative_telemetry_reason'] and result['diagnostic_only']
    assert [r for r, label in ev.opens if label.startswith('round-')] == ['reference', 'candidate', 'candidate', 'reference']
    for role, key, prompt in ev.requests:
        if key != 'smoke':
            assert prompt == list(ev.text_samples[key].tokens)
            assert prompt != workload.prompt(len(prompt), VOCAB)
    assert len(ev.replays) == 8
    for (key, round_index), prompt, tokens in ev.replays:
        assert prompt == list(ev.text_samples[key].tokens)
    assert all(c['mismatches'] == [1, 1] for c in result['cases'])
    assert [(r['length'], r['sample_count']) for r in result['lengths']] == [(64, 2), (256, 2)]
    assert all(len(c[s + '_ms']['reference']) == len(c[s + '_ms']['candidate']) == 2
               for c in result['cases'] for s in ('prefill', 'decode'))


def test_synthetic_controls_are_preserved(corpus, tmp_path):
    ev = evaluator(corpus, tmp_path, controls=True)
    result = ev.run()
    assert result['passed'] and len(result['controls']) == 17
    assert any(c['length'] == 64 for c in result['controls'])
    assert all(c['passed'] and c['identical'] for c in result['controls'])


def test_wrong_candidate_and_reference_failure_are_distinct(corpus, tmp_path):
    bad = evaluator(corpus, tmp_path / 'bad', flags=('--wrong-from', '0')).run()
    assert not bad['passed'] and bad['failure']['kind'] == 'gates' and bad['score'] is None
    with pytest.raises(ReferenceFailed):
        evaluator(corpus, tmp_path / 'ref', reference=fake_engine('--crash-after', '0')).run()


def test_total_mismatch_gate_still_applies(corpus, tmp_path):
    result = evaluator(corpus, tmp_path, flags=('--wrong-from', '7'), gates=Gates(30, 10, 100)).run()
    assert not result['passed'] and result['failure']['where'] == 'total'


def test_diagnostics_do_not_compute_a_formal_objective(corpus, tmp_path):
    ev = evaluator(corpus, tmp_path)
    ev.plan = replace(ev.plan, objective=Objective(length_weights={64: 0, 256: 0}))
    result = ev.run()
    assert result['passed'] and result['score'] is None
    assert len(result['lengths']) == 2


def test_cli_dispatch_leaves_synthetic_default_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(dev_eval, 'run', lambda *args: calls.append(('synthetic', args)) or 0)
    monkeypatch.setattr(text_dev_eval, 'run', lambda *args: calls.append(('text', args)) or 0)
    assert cli.main(['dev-eval']) == 0 and calls[-1] == ('synthetic', (None, 1, True, None))
    assert cli.main(['dev-eval', '--text-dataset', 'prepared.jsonl', '--samples-per-length', '2',
                     '--dataset-seed', '7', '--no-build']) == 0
    assert calls[-1][0] == 'text' and calls[-1][1][2:] == (1, False, None, 2, 7)
    with pytest.raises(SystemExit):
        cli.main(['dev-eval', '--dataset-seed', '7'])


@pytest.fixture
def session(corpus, tmp_path, monkeypatch):
    _, path, model = corpus
    task = load_task(write_task(tmp_path, model))
    engine, brief = tmp_path / 'engine', tmp_path / 'brief'
    engine.mkdir()
    brief.mkdir()
    binary = engine / 'engine'
    binary.write_text('#!/bin/sh\nexec ' + shlex.join([sys.executable, str(FAKE_ENGINE), '--vocab', str(VOCAB)]) + ' "$@"\n')
    binary.chmod(0o755)
    info = {'task': task.to_json(), 'engine': str(engine), 'brief': str(brief), 'reference': fake_engine('--reference'),
            'registry': str(tmp_path / 'registry.json')}
    monkeypatch.setattr(text_dev_eval, 'session_info', lambda: info)
    return info, path, task


def test_command_writes_diagnostics_without_synthetic_or_task_mutation(session):
    info, path, task = session
    original = json.dumps(info, sort_keys=True)
    assert text_dev_eval.run(path, 'all', 1, False, 0, 2, 0) == 0
    from pathlib import Path
    results = list((Path(info['brief']) / 'text-dev-eval').glob('*/result.json'))
    assert len(results) == 1
    result = json.loads(results[0].read_text())
    assert result['passed'] and result['score'] is None and len(result['provenance']['samples']) == 4
    assert not (Path(info['brief']) / 'dev-eval').exists()
    assert json.dumps(info, sort_keys=True) == original


def test_command_records_reference_failure_separately(session):
    from pathlib import Path
    info, path, _ = session
    info['reference'] = fake_engine('--crash-after', '0')
    assert text_dev_eval.run(path, 'all', 1, False, 0, 1, 0) == 2
    errors = list((Path(info['brief']) / 'text-dev-eval').glob('*/error.json'))
    assert len(errors) == 1 and json.loads(errors[0].read_text())['kind'] == 'reference_failed'
    assert not list((Path(info['brief']) / 'text-dev-eval').glob('*/result.json'))


@pytest.mark.parametrize('lengths,rounds,cooldown,samples', [('65', 1, 0, 1), ('64,64', 1, 0, 1),
                                                        ('all', 0, 0, 1), ('all', 1, -1, 1),
                                                        ('all', 1, float('nan'), 1), ('all', 1, 0, 0),
                                                        ('all', 1, 0, 20)])
def test_bad_arguments_or_coverage_launch_no_engine(session, monkeypatch, lengths, rounds, cooldown, samples):
    info, path, task = session
    def unexpected(*args, **kwargs):
        pytest.fail('validation must precede building or opening engines')
    monkeypatch.setattr(text_dev_eval.engine_repo, '_run_logged', unexpected)
    monkeypatch.setattr(TextEvaluator, '_open', unexpected)
    assert text_dev_eval.run(path, lengths, rounds, True, cooldown, samples, 0) == 2
