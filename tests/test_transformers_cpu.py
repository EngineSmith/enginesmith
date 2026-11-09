"""The CPU correctness reference: its registration, and its protocol on a tiny random model when transformers and
torch are installed (CONTRIBUTING.md, "A reference": generate equals model.generate, split and chained requests equal
one, judge reports no mismatch on its own tokens)."""
import importlib.util

import pytest

from enginesmith import references


def test_the_cpu_reference_is_registered_as_correctness_only(model_dir):
    kind = references.KINDS['transformers-cpu']
    assert kind['correctness_only'] and importlib.util.find_spec(kind['module'])
    # It compiles nothing and runs on the host: neither the longest request nor a tensor-parallel degree is passed.
    assert references.command('transformers-cpu', model_dir, 4162, 4)[1:] == [
        '-m', 'enginesmith.references.transformers_cpu', '--model', str(model_dir)]


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from enginesmith.references.transformers_cpu import Reference
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    directory = tmp_path_factory.mktemp('tiny-llama')
    model.save_pretrained(directory)
    return Reference(directory, threads=1)


def test_generate_equals_model_generate_and_a_split_request_equals_one(reference):
    torch = pytest.importorskip('torch')
    prompt = [(7 * i + 3) % 97 for i in range(40)]
    tokens = reference.generate(prompt, 17, True)
    with torch.inference_mode():
        out = reference.model.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                                       do_sample=False, max_new_tokens=17, min_new_tokens=17, eos_token_id=None,
                                       pad_token_id=0)
    assert tokens == out[0, len(prompt):].tolist()
    chained = reference.generate(prompt, 1, True) + reference.generate([], 8, False) + reference.generate([], 8, False)
    assert chained == tokens
    with pytest.raises(ValueError):
        reference.generate([97], 1, True)  # beyond the vocabulary


def test_judge_agrees_with_its_own_tokens_and_ranks_others(reference):
    prompt = [(5 * i + 1) % 97 for i in range(24)]
    tokens = reference.generate(prompt, 12, True)
    assert reference.judge(prompt, tokens) == {'mismatches': []}
    wrong = [(t + 1) % 97 for t in tokens]
    mismatches = reference.judge(prompt, wrong)['mismatches']
    assert mismatches and all(m[1] > 0 and m[3] >= 2 for m in mismatches)
    assert mismatches[0][0] == 0 and mismatches[0][2] == tokens[0]
    assert reference.stats()['peak_memory'] > 0
