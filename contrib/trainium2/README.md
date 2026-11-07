# Trainium2 review tools and results

The scripts the reviewer used for the numbers in the top-level README, as they ran on one trn2.48xlarge (paths and
NeuronCore numbers are that machine's).

- `clean_heldout.sh <tree> <task> <round> <tag>`: rebuilds a run's round from its engine repository with no cache, then
  runs `heldout.py`.
- `heldout.py`: an independent held-out evaluation (5 paired rounds, a fresh seed) of an engine against NxD Inference,
  using the harness's own `Evaluator`.
- `quality_driver*.py`: task accuracy (MATH-500, IFEval) of an engine against NxD Inference through
  `enginesmith.quality`, for 0.6B, 8B, 14B (TP 2) and 30B-A3B (TP 4).
- `results/`: every decided harness round (`rounds.json`), every held-out check (`heldout.json`), per-length timings
  (`cases.json`), every real-text run (`realtext.json`) and the findings (`findings.json`).
