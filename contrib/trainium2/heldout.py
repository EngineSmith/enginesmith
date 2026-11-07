"""Independent held-out check of a run engine: 5 paired rounds on held-out windows from a fresh seed."""
import json, os, sys
from pathlib import Path
from enginesmith.task import load_task
from enginesmith.evaluate import Evaluator, Plan
task_path, engine, out, ref_cores, eng_cores = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
task = load_task(Path(task_path))
ref = ["/usr/bin/env", f"NEURON_RT_VISIBLE_CORES={ref_cores}", sys.executable, "-m", "enginesmith.references.nxdi",
       "--model", str(task.model), "--max-length", str(task.max_request_length())]
if getattr(task, "reference_tp", 1) > 1:
    ref += ["--tp", str(task.reference_tp)]
cand = ["/usr/bin/env", f"NEURON_RT_VISIBLE_CORES={eng_cores}", engine, "--model", str(task.model)]
seed = os.urandom(32)
plan = Plan(tuple(task.cases()), task.vocab_size, 5, task.evaluation.warmup, task.evaluation.cooldown_s, task.gates,
            task.objective, "held_out", seed)
out.mkdir(parents=True, exist_ok=True)
r = Evaluator(ref, cand, plan, out, log=lambda m: print(m, flush=True)).run()
med = lambda xs: sorted(xs)[len(xs) // 2]
print("HELDOUT", json.dumps({"seed": seed.hex(), "passed": r.get("passed"), "score": r.get("score"), "segments": r.get("segments"), "failure": r.get("failure")}, default=str))
for c in r.get("cases", []):
    if c.get("weight"):
        print("CASE", c["length"], "prefill ref/eng", round(med(c["prefill_ms"]["reference"]), 1), round(med(c["prefill_ms"]["candidate"]), 1),
              "decode/tok ref/eng", round(med(c["decode_ms"]["reference"]) / 64, 2), round(med(c["decode_ms"]["candidate"]) / 64, 2), "worst", c.get("worst_rank"))
