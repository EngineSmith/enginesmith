"""Versioned, pre-tokenized natural-text development data; no tokenizer dependency in the harness."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FORMAT = 'natural-text-development-v1'
POLICY = {'add_special_tokens': False, 'chat_template': None, 'text_preprocessing': 'none'}
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                   'vocab.json', 'vocab.txt', 'merges.txt', 'tokenizer.model', 'spiece.model')


class DatasetError(ValueError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def tokenizer_fingerprint(directory: Path) -> str:
    files = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
             for name in TOKENIZER_FILES if (directory / name).is_file()}
    if not files or not any(name in files for name in ('tokenizer.json', 'vocab.json', 'vocab.txt',
                                                       'tokenizer.model', 'spiece.model')):
        raise DatasetError('the local tokenizer directory has no supported vocabulary/tokenizer artifacts')
    return digest(files)


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _tokens(tokens, vocab: int):
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
        raise DatasetError('document tokens must be a nonempty list of integer IDs within the task vocabulary')


def prepare(corpus: Path, output: Path, tokenizer: Path, vocab: int, encode, *, source: str, revision: str,
            license: str, tokenizer_runtime: str) -> dict:
    """Tokenization runs here, before evaluation, using the explicit caller-supplied offline encoder."""
    if type(vocab) is not int or vocab <= 0 or not all(isinstance(v, str) and v.strip()
                                                     for v in (source, revision, license, tokenizer_runtime)):
        raise DatasetError('positive vocabulary size and nonempty source, revision, license and tokenizer runtime required')
    raw = corpus.read_bytes()
    metadata = {'type': 'manifest', 'format': FORMAT, 'corpus_sha256': hashlib.sha256(raw).hexdigest(),
                'source': source, 'revision': revision, 'license': license, 'vocab_size': vocab,
                'tokenizer_sha256': tokenizer_fingerprint(tokenizer), 'tokenizer_runtime': tokenizer_runtime,
                'policy': dict(POLICY)}
    documents, ids, hashes = [], set(), set()
    try:
        lines = raw.decode('utf-8').splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
                raise DatasetError(f'corpus line {number}: a nonempty string id is required')
            text = row.get('text')
            if not isinstance(text, str) or not text.strip():
                raise DatasetError(f'corpus line {number}: nonempty text is required')
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if row['id'] in ids or text_hash in hashes:
                raise DatasetError(f'corpus line {number}: duplicate document id or content')
            tokens = encode(text)
            _tokens(tokens, vocab)
            ids.add(row['id'])
            hashes.add(text_hash)
            documents.append({'type': 'document', 'id': row['id'], 'text_sha256': text_hash,
                              'tokens_sha256': digest(tokens), 'tokens': tokens})
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError('corpus must be UTF-8 JSONL with one document per line') from exc
    if not documents:
        raise DatasetError('corpus contains no documents')
    if metadata['tokenizer_sha256'] != tokenizer_fingerprint(tokenizer):
        raise DatasetError('tokenizer artifacts changed during preparation')
    metadata.update(documents_count=len(documents), documents_sha256=_document_digest(documents))
    # Refuse to overwrite an existing corpus, and never leave a partial prepared artifact on preparation errors.
    with output.open('x') as stream:
        for row in [metadata, *documents]:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    return {**metadata, 'documents': len(documents)}


def _document_digest(documents) -> str:
    return digest(sorted((d['id'], d['text_sha256'], d['tokens_sha256']) for d in documents))


@dataclass(frozen=True)
class Sample:
    id: str
    length: int
    tokens: tuple[int, ...]
    document_id: str
    document_sha256: str
    offset: int

    def metadata(self) -> dict:
        return {'id': self.id, 'length': self.length, 'document_id': self.document_id,
                'document_sha256': self.document_sha256, 'token_offset': self.offset,
                'prompt_sha256': digest(self.tokens)}


def load_samples(path: Path, tokenizer: Path, vocab: int, lengths, samples_per_length: int, seed: int):
    """Select distinct documents per length and contiguous token windows; fail before any engine is started."""
    if type(samples_per_length) is not int or samples_per_length < 1 or type(seed) is not int:
        raise DatasetError('samples per length must be positive and the dataset seed must be an integer')
    if not lengths or any(type(n) is not int or n < 1 for n in lengths) or len(set(lengths)) != len(lengths):
        raise DatasetError('lengths must be distinct positive integers')
    raw = path.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError('prepared corpus must be UTF-8 JSONL') from exc
    if not rows or not isinstance(rows[0], dict):
        raise DatasetError('prepared corpus requires a manifest')
    manifest = rows[0]
    if manifest.get('type') != 'manifest' or manifest.get('format') != FORMAT or digest(manifest.get('policy')) != digest(POLICY):
        raise DatasetError('unsupported prepared corpus format or text/tokenizer policy')
    if type(manifest.get('vocab_size')) is not int or manifest['vocab_size'] != vocab:
        raise DatasetError('prepared corpus vocabulary size differs from the task')
    if manifest.get('tokenizer_sha256') != tokenizer_fingerprint(tokenizer):
        raise DatasetError('prepared corpus tokenizer artifacts differ from the task model')
    if not _sha(manifest.get('corpus_sha256')) or not all(isinstance(manifest.get(k), str) and manifest[k].strip()
                                                       for k in ('source', 'revision', 'license', 'tokenizer_runtime')):
        raise DatasetError('prepared corpus is missing provenance')
    ids, hashes, documents = set(), set(), []
    for row in rows[1:]:
        if not isinstance(row, dict) or row.get('type') != 'document' or not isinstance(row.get('id'), str) or not row['id'].strip():
            raise DatasetError('invalid prepared document')
        if not _sha(row.get('text_sha256')) or row['id'] in ids or row['text_sha256'] in hashes:
            raise DatasetError('duplicate document id/content or invalid document hash')
        _tokens(row.get('tokens'), vocab)
        if row.get('tokens_sha256') != digest(row['tokens']):
            raise DatasetError('prepared document token checksum mismatch')
        ids.add(row['id'])
        hashes.add(row['text_sha256'])
        documents.append(row)
    if (type(manifest.get('documents_count')) is not int or manifest['documents_count'] != len(documents)
            or manifest.get('documents_sha256') != _document_digest(documents)):
        raise DatasetError('prepared corpus document count or checksum mismatch')
    samples = []
    for length in lengths:
        eligible = [d for d in documents if len(d['tokens']) >= length]
        if len(eligible) < samples_per_length:
            raise DatasetError(f'length {length} needs {samples_per_length} distinct sufficiently long documents; '
                               f'found {len(eligible)} (no padding or repetition is permitted)')
        eligible.sort(key=lambda d: digest(['document', seed, length, d['id'], d['text_sha256']]))
        for i, doc in enumerate(eligible[:samples_per_length]):
            offset = int(digest(['window', seed, length, doc['id'], doc['tokens_sha256']]), 16) % (len(doc['tokens']) - length + 1)
            samples.append(Sample(f'text-p{length}-s{i + 1}', length, tuple(doc['tokens'][offset:offset + length]),
                                  doc['id'], doc['text_sha256'], offset))
    provenance = {**manifest, 'prepared_sha256': hashlib.sha256(raw).hexdigest(), 'sampling_version': 1,
                  'seed': seed, 'samples_per_length': samples_per_length, 'samples': [s.metadata() for s in samples]}
    return samples, provenance
