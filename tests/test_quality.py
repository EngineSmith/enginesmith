import json

import pytest

from conftest import VOCAB, fake_engine
from enginesmith import quality
from enginesmith.protocol import Engine
from enginesmith.task import Quality

MATH = [{'unique_id': f'test/{i}', 'problem': f'problem {i}', 'answer': str(i % 3), 'level': 1 + i % 5}
        for i in range(500)]


def test_the_math_subset_is_stratified_and_fixed(monkeypatch):
    monkeypatch.setattr(quality, 'dataset', lambda name: MATH)
    chosen = quality.items('math500', 200)
    assert len(chosen) == 200 and chosen == quality.items('math500', 200)
    assert all(c['text'].endswith(quality.MATH_SUFFIX) for c in chosen)
    assert {lvl: sum(c['item']['level'] == lvl for c in chosen) for lvl in range(1, 6)} == {lvl: 40 for lvl in range(1, 6)}
    assert len(quality.items('math500', 1000)) == 500
    assert [len(quality.items('math500', n)) for n in (1, 7, 100, 333)] == [1, 7, 100, 333]


def test_answers_stop_at_the_end_of_sequence(tmp_path):
    with Engine(fake_engine(), scratch=tmp_path / 's', label='e') as engine:
        engine.hello(10)
        whole, _ = engine.generate([5, 6], 600, True, VOCAB, 30)
        eos = {whole[300]}
        stop = whole.index(whole[300])
        tokens, finish = quality.answer(engine, [5, 6], eos, 1000, VOCAB)
        assert finish == 'eos' and tokens == whole[:stop]
        tokens, finish = quality.answer(engine, [5, 6], set(), 300, VOCAB)
        assert finish == 'length' and tokens == whole[:300]


def test_the_final_answer_is_the_text_after_thinking():
    assert quality.final_text('<think>a\n</think>\n\nThe answer is 4.') == 'The answer is 4.'
    assert quality.final_text('  plain  ') == 'plain'


def test_math_verify_scores_boxed_answers():
    pytest.importorskip('math_verify')
    chosen = {'a': {'item': {'answer': '\\frac{1}{2}'}}, 'b': {'item': {'answer': '3'}}}
    rows = [{'id': 'a', 'final': 'So it is \\boxed{0.5}.'}, {'id': 'b', 'final': 'I think \\boxed{4}'}]
    assert quality.score_math(rows, chosen) == {'a': True, 'b': False}


def test_the_ifeval_checker_scores_instructions():
    pytest.importorskip('langdetect')
    item = {'key': 1, 'prompt': 'Write a sentence without commas.', 'instruction_id_list': ['punctuation:no_comma'],
            'kwargs': [{}]}
    chosen = {'1': {'item': item}}
    per_item, metrics = quality.score_ifeval([{'id': '1', 'final': 'No commas here at all.'}], chosen)
    assert per_item == {'1': True} and metrics['prompt_strict'] == 1.0
    per_item, _ = quality.score_ifeval([{'id': '1', 'final': 'Well, a comma.'}], chosen)
    assert per_item == {'1': False}


class FakeTokenizer:
    eos = set()

    def __init__(self, model):
        pass

    def prompt(self, text):
        return [len(text) % 50 + 1, 2, 3]

    def decode(self, tokens):
        return '<think>...</think> \\boxed{1}'


def test_both_arms_answer_the_same_items_and_are_compared(tmp_path, monkeypatch):
    pytest.importorskip('math_verify')
    monkeypatch.setattr(quality, 'dataset', lambda name: MATH[:6])
    monkeypatch.setattr(quality, 'Tokenizer', FakeTokenizer)
    settings = Quality(benchmarks=('math500',), samples=6, max_tokens=8)
    summary = quality.evaluate(settings, tmp_path, VOCAB, fake_engine('--speed', '2'), fake_engine(), tmp_path / 'q')
    result = summary['math500']
    assert result['paired'] == 6 and result['reference']['accuracy'] == result['engine']['accuracy'] == 2 / 6
    assert result['difference'] == 0 and result['engine']['truncated'] == 6 and result['identical_outputs'] == 6
    rows = [json.loads(line) for line in (tmp_path / 'q' / 'math500-engine.jsonl').read_text().splitlines()]
    assert len(rows) == 6 and rows[0]['final'] == '\\boxed{1}'
    # A second call continues instead of answering again.
    again = quality.evaluate(settings, tmp_path, VOCAB, fake_engine('--crash-after', '0'), fake_engine(),
                             tmp_path / 'q')
    assert again['math500']['paired'] == 6 and not again['math500']['engine']['errors']
    with pytest.raises(ValueError, match='other settings'):
        quality.evaluate(Quality(benchmarks=('math500',), samples=6, max_tokens=9), tmp_path, VOCAB, fake_engine(),
                         fake_engine(), tmp_path / 'q')
