"""Task-accuracy check (MATH-500, IFEval) of an engine vs NxD Inference, using enginesmith.quality unmodified.
run ARM ENGINE_PATH OUTDIR : answer both benchmarks with one arm (reference: cores 44-45, engine: cores 46-47)
score OUTDIR                : score and print the summary (same logic as quality.evaluate)"""
import json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
from enginesmith import quality as q
q.CACHE = Path.home() / "quality-check" / "cache"
MODEL = Path.home() / "models" / "Qwen3-30B-A3B"
SETTINGS = SimpleNamespace(benchmarks=("ifeval",), samples=100, max_tokens=3840)
VOCAB = json.loads((MODEL / "config.json").read_text())["vocab_size"]
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

def argv(arm, engine):
    if arm == "reference":
        return ["/usr/bin/env", "NEURON_RT_VISIBLE_CORES=" + __import__("os").environ.get("REF_CORES", "8-11") + "", str(Path.home() / "es-pr3/.venv/bin/python"), "-m",
                "enginesmith.references.nxdi", "--model", str(MODEL), "--max-length", "4600", "--tp", "4"]
    return ["/usr/bin/env", "NEURON_RT_VISIBLE_CORES=" + __import__("os").environ.get("ENG_CORES", "46-47"), engine, "--model", str(MODEL)]

def run(arm, engine, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps({"samples": SETTINGS.samples, "max_tokens": SETTINGS.max_tokens,
        "decoding": "greedy", "reference_max_length": 4600, "engine": engine}, indent=2))
    tok = q.Tokenizer(MODEL)
    for name in SETTINGS.benchmarks:
        chosen = q.items(name, SETTINGS.samples)
        t0 = time.time()
        rows = q.run_arm(argv(arm, engine), chosen, tok, SETTINGS, VOCAB, out / f"{name}-{arm}.jsonl", time.time() + 6 * 3600, log=log)
        wall = time.time() - t0
        with open(out / f"wall-{name}-{arm}.json", "a") as f:
            f.write(json.dumps({"wall_s": wall, "answered": sum(1 for r in rows if r.get("id"))}) + "\n")
        log(f"{name} {arm} done in {wall:.0f}s")

def score(out):
    summary = {"settings": json.loads((out / "settings.json").read_text())}
    for name in SETTINGS.benchmarks:
        chosen = q.items(name, SETTINGS.samples); by_id = {i["id"]: i for i in chosen}
        rows = {arm: [json.loads(l) for l in (out / f"{name}-{arm}.jsonl").read_text().splitlines()] for arm in ("reference", "engine")}
        answered = {a: {r["id"] for r in rs if r.get("id")} for a, rs in rows.items()}
        paired = sorted(answered["reference"] & answered["engine"])
        outs = {a: {r["id"]: r for r in rs if r.get("id")} for a, rs in rows.items()}
        res = {"items": len(chosen), "paired": len(paired),
               "identical_outputs": sum(outs["engine"][i]["output"] == outs["reference"][i]["output"] for i in paired)}
        # length of the common token prefix proxy: common character prefix fraction for differing items
        diverge = []
        for i in paired:
            a, b = outs["engine"][i]["output"], outs["reference"][i]["output"]
            if a != b:
                n = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                diverge.append(n)
        res["differing_first_divergence_chars"] = sorted(diverge)
        corr = {}
        for arm, rs in rows.items():
            valid = [r for r in rs if r.get("id")]
            c, m = (q.score_math(valid, by_id), {}) if name == "math500" else q.score_ifeval(valid, by_id)
            corr[arm] = c
            res[arm] = {**q._arm_summary(rs, c, paired), **m}
            wf = out / f"wall-{name}-{arm}.json"
            res[arm]["wall_s"] = sum(json.loads(l)["wall_s"] for l in wf.read_text().splitlines()) if wf.exists() else None
        if paired:
            d = [corr["engine"][i] - corr["reference"][i] for i in paired]
            mu = sum(d) / len(d); var = sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1)
            res.update(difference=mu, difference_standard_error=math.sqrt(var / len(d)),
                       discordant={"engine_only": sum(x == 1 for x in d), "reference_only": sum(x == -1 for x in d)})
        summary[name] = res
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else: score(Path(sys.argv[2]))
