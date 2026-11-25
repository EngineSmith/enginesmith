# Natural-text development diagnostics

Synthetic token contexts can yield repetitive continuations and unusually high speculative acceptance. A speedup on
them need not carry over to natural text. The optional text mode of `dev-eval` measures natural-text prompts alongside
the existing synthetic controls. It produces development diagnostics only: no formal score, no incumbent decision,
and no automatic change to a task, run or held-out workload.

## Prepare a local corpus before timing

Create a UTF-8 JSONL file with a stable string `id` and nonempty `text` on each line. Documents must be distinct in
both ID and content. Supply the source, revision and redistribution license of the material; use `private-local`
only for data that must stay local. No corpus is bundled or downloaded by this feature.

Run the explicit preparation script in an environment where this checkout and Transformers are installed:

```sh
python scripts/prepare_text_dataset.py corpus.jsonl \
  --tokenizer ./model --vocab-size 32000 \
  --source local-development-corpus --revision v1 --license private-local \
  --out prepared-text.jsonl
```

Use the task's actual vocabulary size, not the illustrative value above. `--tokenizer` is a local model/tokenizer
directory. Loading is offline (`local_files_only=True`) and remote tokenizer code is disabled. There is no model
weight load, inference or automatic installation. The harness itself retains standard-library-only imports.

The script passes the unmodified text to the tokenizer with `add_special_tokens=False`, without a chat template or
additional text preprocessing. The tokenizer's own normalization remains part of its behavior. The prepared artifact
records the raw corpus SHA-256, source/revision/license, vocabulary size, tokenizer artifact fingerprint, tokenizer
implementation/library versions, policy, each document's text hash, token IDs and token hash, plus the prepared
document count and collection checksum. No absolute corpus/model path
is needed in this metadata. Tokenizer artifacts are fingerprinted again against the task model before evaluation.
Provenance is an integrity and reproducibility record, not an authentication proof for an untrusted prepared file.

## Run inside a coding session

```sh
enginesmith dev-eval --text-dataset prepared-text.jsonl \
  --samples-per-length 3 --dataset-seed 0 --lengths all --rounds 2
```

The ordinary `dev-eval` command without `--text-dataset` is unchanged. Like synthetic development evaluation, the
text command requires the current session's frozen task, reference command and engine checkout. Default lengths
are task lengths up to 32K, or the smallest task length if none qualify. An explicit list must be a subset of task
lengths. `--no-build` and `--cooldown` work as in synthetic mode.

Each length selects the requested number of distinct eligible documents by a deterministic hash order. A second
hash selects one contiguous token window per document. Samples stay fixed across paired repetitions; repetitions
measure timing variability, not new document diversity. The seed changes sampling without accessing formal nonce
or held-out prompts. Selected document hashes, token offsets and exact prompt-token hashes are saved before timing.

If three samples of 512K are requested, at least three documents must individually contain at least 524,288 tokens.
Short text is never repeated, concatenated with another document, randomly padded or silently evaluated at a shorter
length. Coverage, provenance and token-ID validation fail before building or launching an engine.

## Procedure and results

Text mode inherits synthetic boundary controls, warmup, engine isolation/cooldown, protocol timing, continuation and
reset behavior, candidate timeouts, teacher-forced gates and total mismatch accounting from the existing evaluator.
When controls are enabled, every synthetic boundary is checked, including boundaries whose lengths are also selected
for natural-text timing. Its reference-control cache is local to the diagnostic directory, leaving the session's
synthetic cache untouched.
Only timed sample selection and the exact prompt used for timed replay differ. Reference and candidate run in
alternating AB/BA order, with one engine resident at a time. Prefill includes the first output token; Decode measures
the remaining `output_tokens - 1` tokens. Tokenization and sampling are outside all timed engine requests.

Results go under the session brief's **`text-dev-eval/diagnostic-*/`**, separate from **`dev-eval/`**. The directory
contains provenance, build/process logs and a `result.json` (or an `error.json` for input/build/reference failures).
The result includes raw per-repetition reference/candidate milliseconds, each sample's median paired ratios,
correctness outcomes, and per-length medians across document samples. `score` and `segments` are null with a reason:
these are not the synthetic objective or a new formal score. A failure in the reference is distinguished from a
candidate gate/protocol failure. Missing speculative counters are null with a reason, never inferred from throughput.

The command uses the session device lock and process registry. It runs within the current coding-session deadline;
it does not add time, refund a slot, start a new search, or run automatically. A text diagnostic must be requested
explicitly with a corpus made available to the coding session. Sharing a synthetic engine package does not silently
include this local corpus or its diagnostic output.

## Scope and validation

This is a core-only diagnostic feature. The formal workload, evaluator, protocol, gates, objective, acceptance and
evaluation version are unmodified. CPU tests use fake engines and an injected fake tokenizer to test exact prompt
pairing/replay, synthetic controls, AB/BA ordering, malformed inputs, insufficient coverage and result isolation.
They establish behavior, not natural-text GPU performance; no such performance claim accompanies this feature.

Draft acceptance/FIFO telemetry, conversational templates, corpus distribution and a formal natural-text objective
are separate proposals. Any formal judging change still requires the repository's issue/version/fingerprint process.
