#!/usr/bin/env python3
"""Bytes read per decode token, and the decode floor, from a model directory's safetensors headers and a small JSON
spec that says which tensors a decode token reads. Reads only the 8-byte length and the JSON header of each shard
(never the tensor data); Python standard library only.

Spec (JSON):
  {
    "groups": [
      {"name": "experts", "patterns": ["model.layers.*.mlp.experts.*"], "fraction": 0.0625},
      {"name": "head",    "patterns": ["lm_head.*"]},
      {"name": "skip",    "patterns": ["*.rotary_emb.*"], "fraction": 0}
    ],
    "per_position_bytes": 20480,
    "state_bytes": 245760
  }
- Each tensor is counted in the FIRST group whose fnmatch pattern matches its name ('*' also matches dots).
- "fraction" (default 1) is the part of the group's stored bytes one token reads: for routed experts stacked in one
  tensor it is top_k / n_experts; 0 for tensors decode never reads. A tied embedding that is also the output head
  is read in full by the head (fraction 1); the embedding gather of one row is negligible.
- "per_position_bytes": KV-cache bytes read per position of context (layers x K and V x kv_heads x head_dim x
  bytes per element). "state_bytes": recurrent/conv state read per token. Both optional (default 0).
- Tensors matched by no group are listed and, by default, counted in full (--unmatched).

Floor (ms/token) = bytes per token / measured streaming bandwidth. Use the bandwidth you measured on the machine (a
plain streaming read), not the vendor's peak.

usage: bytes_per_token.py MODEL_DIR SPEC.json [--bandwidth-gbs 590] [--contexts 0,256,4096] [--json]
       bytes_per_token.py --example     (prints a spec template)"""
import argparse
import fnmatch
import json
import struct
import sys
from pathlib import Path

EXAMPLE = {
    "groups": [
        {"name": "routed experts (top_k / n_experts read)", "patterns": ["model.layers.*.mlp.experts.*"],
         "fraction": 0.0625},
        {"name": "router", "patterns": ["model.layers.*.mlp.gate.*"]},
        {"name": "attention", "patterns": ["model.layers.*.self_attn.*"]},
        {"name": "dense mlp", "patterns": ["model.layers.*.mlp.*"]},
        {"name": "norms", "patterns": ["*norm*"]},
        {"name": "head", "patterns": ["lm_head.*", "model.embed_tokens.*"]},
    ],
    "per_position_bytes": 0,
    "state_bytes": 0,
}


def shard_files(model_dir):
    model_dir = Path(model_dir)
    index = model_dir / 'model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        return [model_dir / n for n in names]
    files = sorted(model_dir.glob('*.safetensors'))
    if not files:
        raise SystemExit(f'no model.safetensors.index.json and no *.safetensors in {model_dir}')
    return files


def read_header(path):
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    return header


def tensor_table(model_dir):
    """{name: (dtype, shape, bytes)} from every shard's header (bytes from data_offsets)."""
    table = {}
    for path in shard_files(model_dir):
        for name, meta in read_header(path).items():
            start, end = meta['data_offsets']
            if name in table:
                raise SystemExit(f'tensor {name} appears in more than one shard')
            table[name] = (meta['dtype'], meta['shape'], end - start)
    return table


def classify(table, spec):
    groups = [dict(g, tensors=0, stored=0) for g in spec['groups']]
    unmatched = []
    for name in sorted(table):
        nbytes = table[name][2]
        for g in groups:
            if any(fnmatch.fnmatchcase(name, p) for p in g['patterns']):
                g['tensors'] += 1
                g['stored'] += nbytes
                break
        else:
            unmatched.append((name, nbytes))
    return groups, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model_dir', nargs='?')
    ap.add_argument('spec', nargs='?')
    ap.add_argument('--bandwidth-gbs', type=float, default=None,
                    help='measured streaming read bandwidth in GB/s (1e9 B/s); prints the floor when given')
    ap.add_argument('--contexts', default='0', help='context lengths for the KV term (default 0)')
    ap.add_argument('--unmatched', choices=('include', 'exclude', 'error'), default='include',
                    help='what to do with tensors no group matches (default include, counted in full)')
    ap.add_argument('--json', action='store_true', help='print the result as JSON')
    ap.add_argument('--example', action='store_true', help='print a spec template and exit')
    args = ap.parse_args()
    if args.example:
        print(json.dumps(EXAMPLE, indent=1))
        return 0
    if not args.model_dir or not args.spec:
        ap.error('MODEL_DIR and SPEC are required (or --example)')
    spec = json.loads(Path(args.spec).read_text())
    table = tensor_table(args.model_dir)
    groups, unmatched = classify(table, spec)
    if unmatched and args.unmatched == 'error':
        raise SystemExit(f'{len(unmatched)} tensors match no group, e.g. {unmatched[0][0]}')

    rows = []
    for g in groups:
        frac = float(g.get('fraction', 1.0))
        rows.append({'group': g['name'], 'tensors': g['tensors'], 'stored_bytes': g['stored'], 'fraction': frac,
                     'bytes_per_token': g['stored'] * frac})
    if unmatched:
        stored = sum(b for _, b in unmatched)
        frac = 1.0 if args.unmatched == 'include' else 0.0
        rows.append({'group': '(unmatched)', 'tensors': len(unmatched), 'stored_bytes': stored, 'fraction': frac,
                     'bytes_per_token': stored * frac})
    weights = sum(r['bytes_per_token'] for r in rows)
    total_stored = sum(b for _, _, b in table.values())
    per_pos = float(spec.get('per_position_bytes', 0))
    state = float(spec.get('state_bytes', 0))
    contexts = [int(c) for c in args.contexts.split(',') if c != '']
    floors = []
    for c in contexts:
        b = weights + state + per_pos * c
        row = {'context': c, 'bytes_per_token': b}
        if args.bandwidth_gbs:
            ms = b / (args.bandwidth_gbs * 1e9) * 1e3
            row.update(floor_ms=ms, ceiling_tok_s=1e3 / ms)
        floors.append(row)
    result = {'model_dir': str(args.model_dir), 'tensors': len(table), 'stored_bytes': total_stored,
              'groups': rows, 'weight_bytes_per_token': weights, 'state_bytes': state,
              'per_position_bytes': per_pos, 'bandwidth_gbs': args.bandwidth_gbs, 'floors': floors,
              'unmatched': [n for n, _ in unmatched]}
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print(f'{len(table)} tensors, {total_stored:,} stored bytes')
    print(f"{'group':44s} {'tensors':>7s} {'stored B':>16s} {'fraction':>9s} {'B/token':>16s} {'share':>6s}")
    for r in rows:
        share = r['bytes_per_token'] / weights * 100 if weights else 0.0
        print(f"{r['group'][:44]:44s} {r['tensors']:7d} {r['stored_bytes']:16,d} {r['fraction']:9.4f} "
              f"{r['bytes_per_token']:16,.0f} {share:5.1f}%")
    print(f"{'weights read per token':44s} {'':7s} {'':16s} {'':9s} {weights:16,.0f}")
    if unmatched:
        print(f'{len(unmatched)} unmatched tensors ({args.unmatched}), first: {unmatched[0][0]}')
    for f in floors:
        line = f"context {f['context']:>7d}: {f['bytes_per_token']:,.0f} B/token"
        if 'floor_ms' in f:
            line += (f"  floor {f['floor_ms']:.3f} ms/token at {args.bandwidth_gbs:g} GB/s"
                     f"  (ceiling {f['ceiling_tok_s']:.1f} tok/s)")
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
