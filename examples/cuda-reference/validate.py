"""CUDA reference/stock parity checks. Run serially on one idle GPU; never a mock performance result."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from sglang_cuda import PINNED_VERSION, Reference
from enginesmith.references.sglang import CONTEXT_MARGIN, MEM_FRACTION


def prompt(length, vocab):
    # Fixed public control data for conformance, not a benchmark corpus or held-out workload.
    return [((i * 7919) + 37) % vocab for i in range(length)]


def check_continuations(reference, tokens, count, vocab):
    full = reference.generate(tokens, count, True)
    split = reference.generate(tokens, 1, True) + reference.generate([], count - 1, False)
    chain = reference.generate(tokens, 1, True)
    for _ in range(count - 1):
        chain += reference.generate([], 1, False)
    self_judge = reference.judge(tokens, split)
    wrong = [(token + vocab // 2) % vocab for token in split]
    bad_judge = reference.judge(tokens, wrong)
    records = bad_judge['mismatches']
    return {'split_matches': full == split, 'chain_matches': full == chain,
            'self_judge_mismatches': self_judge['mismatches'], 'wrong_stream_mismatches': records,
            'wrong_stream_has_early_low_rank': any(step < min(8, count) and rank >= 100
                                                   for step, margin, best, rank in records)}, full


def run(model, lengths, count):
    if importlib.metadata.version('sglang') != PINNED_VERSION:
        raise RuntimeError(f'need stock sglang=={PINNED_VERSION}')
    import sglang
    config = json.loads((model / 'config.json').read_text())
    vocab = config['vocab_size']
    context = config.get('max_position_embeddings') or config.get('max_seq_len') or config.get('seq_length')
    if not context or any(length + count > context for length in lengths):
        raise ValueError('requested controls exceed or cannot establish model context')
    os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN'] = '1'
    stock = sglang.Engine(model_path=str(model), skip_tokenizer_init=True, tp_size=1,
                          context_length=context + CONTEXT_MARGIN, mem_fraction_static=MEM_FRACTION,
                          enable_custom_logit_processor=True, log_level='error')
    outputs = {}
    try:
        for length in lengths:
            if not stock.flush_cache().success:
                raise RuntimeError('stock cache flush refused')
            outputs[length] = stock.generate(input_ids=prompt(length, vocab), sampling_params={
                'temperature': 0.0, 'max_new_tokens': count, 'ignore_eos': True,
                'skip_special_tokens': False})['output_ids']
    finally:
        stock.shutdown()
    reference = Reference(model)
    results = []
    try:
        for length in lengths:
            checks, tokens = check_continuations(reference, prompt(length, vocab), count, vocab)
            checks.update(length=length, stock_matches=tokens == outputs[length])
            results.append(checks)
    finally:
        reference.engine.shutdown()
    passed = all(row['stock_matches'] and row['split_matches'] and row['chain_matches']
                 and not row['self_judge_mismatches'] and row['wrong_stream_has_early_low_rank'] for row in results)
    return {'kind': 'experimental_cuda_reference_conformance', 'sglang_version': PINNED_VERSION,
            'passed': passed, 'cases': results, 'timing_self_evaluation': None,
            'timing_reason': 'run the formal Evaluator reference-vs-reference separately before registration',
            'long_context_prepare': None, 'engine_search_evidence': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--lengths', default='1,2,63,2047,2048,2049,4097,8192')
    parser.add_argument('--tokens', type=int, default=65)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(v) for v in args.lengths.split(',')]
    if args.tokens < 2 or not lengths or any(n <= 0 for n in lengths):
        parser.error('positive lengths and at least two tokens required')
    if args.out.exists():
        parser.error('out must be a fresh file')
    # Library logs must not corrupt machine-readable output.
    replies = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    started = time.monotonic()
    result = run(args.model.resolve(), lengths, args.tokens)
    result['duration_s'] = time.monotonic() - started
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    replies.write(json.dumps(result) + '\n')
    replies.flush()
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
