"""Explicit offline tokenizer preparation; optional Transformers dependency stays outside the harness."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from enginesmith.text_dataset import DatasetError, prepare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path, help='UTF-8 JSONL records containing id and text')
    parser.add_argument('--tokenizer', type=Path, required=True, help='local tokenizer directory for the task model')
    parser.add_argument('--vocab-size', type=int, required=True, help='the task model vocabulary size')
    parser.add_argument('--source', required=True, help='corpus source or a local corpus identifier')
    parser.add_argument('--revision', required=True, help='corpus revision')
    parser.add_argument('--license', required=True, help='corpus license, or private-local for local-only material')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import transformers
    except ImportError:
        parser.error('run this explicit preparation script in an environment with Transformers installed')
    try:
        if not args.tokenizer.is_dir():
            raise DatasetError('--tokenizer must name a local directory; downloads are not supported')
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True,
                                                               trust_remote_code=False, use_fast=True)
        info = prepare(args.corpus, args.out, args.tokenizer, args.vocab_size,
                       lambda text: tokenizer.encode(text, add_special_tokens=False), source=args.source,
                       revision=args.revision, license=args.license,
                       tokenizer_runtime=f'transformers {transformers.__version__}; '
                       f'tokenizers {importlib.metadata.version("tokenizers")}; '
                       f'{type(tokenizer).__name__}; fast={tokenizer.is_fast}')
    except (DatasetError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'prepared {info["documents"]} documents; tokenizer {info["tokenizer_sha256"]}; no inference was run')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
