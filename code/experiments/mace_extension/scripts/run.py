"""Dispatch one extension package. Resumable, cap-checked, one provenance row per job.

  python -m experiments.mace_extension.scripts.run <PKG> [--model ID] [--limit N] [--sample N]
  python -m experiments.mace_extension.scripts.run identity        # zero-cost: Sonnet jobs must all return from cache

Reuses mace10.run for captions, the stop logic and the evidence builders. Captions always run on the frozen caption model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agmem import config as C, data, kb as KB, llm, prompts
from mace10 import run as M
from . import jobs as X

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_results"
HARD_STOP = 872.82      # established limit: USD 400 additional beyond ledger 472.82
SOFT_STOP = 822.82      # established limit: no optional work past this ledger total


def manifest_sha() -> str:
    p = data.C.DATA / "LOCK.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def tree_state() -> dict:
    g = lambda *a: subprocess.run(["git", *a], cwd=C.ROOT, capture_output=True, text=True).stdout
    return {"git": g("rev-parse", "HEAD").strip(), "dirty_diff_sha": hashlib.sha256(g("diff").encode()).hexdigest()}


def one(model, crop, classes, kbd, caps, j):
    with llm.use_model(model):
        rec = M.diagnose(crop, classes, kbd, caps, j)
    return {**rec, "experiment": j["pkg"], "diagnosis_model": model or C.MODEL, "memory_bank_ids": [m["image_id"] for m in (j["memory"] or [])], "t": time.time()}


def run(pkg, model=None, limit=None, sample=None, tag="", cap=SOFT_STOP):
    name = pkg + (f"_{tag}" if tag else ""); d = RAW / name; d.mkdir(parents=True, exist_ok=True)
    pred, fail = d / "predictions.jsonl", d / "failures.jsonl"
    start = M.spent(); llm.set_budget(cap)
    J = X.build(pkg); total = len(J)
    if sample:
        J = random.Random(0).sample(J, sample)
    if limit:
        J = J[:limit]
    seen = {json.loads(l)["job_id"] for l in pred.open()} if pred.exists() else set()
    todo = [j for j in J if j["job_id"] not in seen]
    with (d / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({"pkg": name, "jobs_total": total, "selected": len(J), "done_before": len(seen), "diagnosis_model": model or C.MODEL, "caption_model": C.MODEL,
                            "ledger_start": start, "cap_ledger": cap, **tree_state(), "data_rev": C.HF_REV, "dataset_lock_sha": manifest_sha(),
                            "diag_system_sha": llm.sha(prompts.DIAG_SYS), "caption_system_sha": llm.sha(prompts.CAPTION_SYS), "claude_cli": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip(),
                            "parallel": C.PARALLEL, "started": time.time()}) + "\n")
    print(f"{name}: {len(todo)} to do of {total}, ledger {start:.2f}, model {model or C.MODEL}", flush=True)
    caps, cstats, stop = M.captions(todo, fail)
    print(f"captions {cstats['calls']} ({cstats['cache_hits']} cached, {cstats['empty']} empty) ${cstats['usd_micro'] / 1e6:.2f}", flush=True)
    recs = []
    if not stop:
        kbs = {c: (data.classes(c), KB.build(c, data.classes(c))) for c in {j["crop"] for j in todo}}
        with pred.open("a") as fh, ThreadPoolExecutor(C.PARALLEL) as ex:
            def keep(r):
                fh.write(json.dumps(r) + "\n"); fh.flush(); recs.append(r)
                if len(recs) % 200 == 0:
                    print(f"{len(recs)}/{len(todo)} ledger {M.spent():.2f}", flush=True)
            flag = threading.Event()
            futs = {ex.submit(M.guarded, flag, one, model, j["crop"], *kbs[j["crop"]], caps, j): j["job_id"] for j in todo}
            stop = M.drain(futs, flag, keep, fail)
    end = M.spent(); new = [r for r in recs if not r["cache_hit"]]
    st = {"pkg": name, "jobs_total": total, "selected": len(J), "rows": len(seen) + len(recs), "complete": len(seen) + len(recs) == total, "stop": stop, "incurred_usd": round(end - start, 4),
          "ledger_total": round(end, 4), "new_calls": len(new), "cache_hits": len(recs) - len(new), "mean_new_call_usd": round(sum(r["cost_usd"] or 0 for r in new) / max(1, len(new)), 5),
          "unknown_predictions": sum(r.get("prediction") == "UNKNOWN" for r in recs), "model_ids": sorted({m for r in new for m in (r.get("model_ids") or [])}), "finished": time.time()}
    (d / "status.json").write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1), flush=True)
    return st, recs


def identity():
    """Rebuild the Sonnet requests through this code path and compare with the saved rows. No new call is allowed."""
    llm.set_budget(M.spent())    # any uncached call would raise BudgetExceeded
    from mace10 import results as R
    old = {}
    for pkg in ("P1", "P4labels", "P8", "P5"):
        D = R.load(pkg)
        for r in D.itertuples():
            old[(r.condition, r.image_id)] = r.prediction
    J = [j for j in X.build("E1_x")] + [j for j in X.build("E4other") if (j["condition"], j["image_id"]) in old]
    J = random.Random(1).sample(J, 400)
    caps = json.loads((M.OUT / "captions.json").read_text()); same = hit = 0
    for j in J:
        c, k = data.classes(j["crop"]), KB.build(j["crop"], data.classes(j["crop"]))
        r = one(None, j["crop"], c, k, caps, j)
        hit += bool(r["cache_hit"]); same += r["prediction"] == old[(j["condition"], j["image_id"])]
    out = {"jobs": len(J), "cache_hits": hit, "same_prediction": same}
    (RAW / "identity_check.json").write_text(json.dumps(out, indent=1)); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pkg"); ap.add_argument("--model"); ap.add_argument("--limit", type=int); ap.add_argument("--sample", type=int); ap.add_argument("--tag", default=""); ap.add_argument("--cap", type=float, default=SOFT_STOP)
    a = ap.parse_args()
    if a.pkg == "identity":
        identity()
    else:
        assert a.cap <= HARD_STOP
        run(a.pkg, a.model, a.limit, a.sample, a.tag, a.cap)
