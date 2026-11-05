"""Dispatch one package of the ten-crop study (see PLAN.md). Resumable; stops the queue on a usage-limit or budget stop.

  python -m mace10.run <PKG> --cap-additional 400 [--limit N]

PKG: P7dev P1 P4labels P2 P3 P4rest P5 P6 P7, and P8 P9 from the 2026-11-05 amendment. Captions run first, then the diagnoses. Every finished job is appended to
out/<PKG>/predictions.jsonl at once, so a stop loses nothing and a rerun skips the job_ids already written. Raw replies stay
in the content-addressed cache and can be exported with mace10.export_traces.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agmem import agentic, config as C, data, kb as KB, llm, prompts, run as R
from . import amend, conditions, jobs as JB

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LEDGER_CSV = HERE / "BUDGET_LEDGER.csv"
LEDGER_START = 472.82
OPTIONAL_STOP = LEDGER_START + 350
LEDGER_HEADER = "status,package,condition,calls,cache_hits,new_calls,rate_usd,projected_usd,optional,incurred_usd,reserved_usd\n"
MAX_TRANSIENT = 25            # transient CLI failures tolerated per invocation before the queue stops
SELECT = {
    "P7dev": lambda j: j["condition"] == "imageref_dev",
    "P1": lambda j: j["pkg"] == "P1",
    "P4labels": lambda j: j["pkg"] == "P4" and j["variant"] == "labels",
    "P2": lambda j: j["pkg"] == "P2",
    "P3": lambda j: j["pkg"] == "P3",
    "P4rest": lambda j: j["pkg"] == "P4" and j["variant"] != "labels",
    "P5": lambda j: j["pkg"] == "P5",
    "P6": lambda j: j["pkg"] == "P6",
    "P7": lambda j: j["condition"] == "imageref",
    "P8": lambda j: j["pkg"] == "P8",
    "P9": lambda j: j["pkg"] == "P9",
}


def spent() -> float:
    with llm._lock:
        return llm.spent()


def with_shown_labels(j: dict) -> dict:
    memory = None if j["memory"] is None else [{"image_id": m["image_id"], "label_shown": m["label"], "label_true": m["label"]} for m in j["memory"]]
    return {**j, "memory": memory}


def variant_diagnosis(crop, classes, kbd, caps, j) -> dict:
    """Content controls and image references: same system prompt, classes, KB and schema; only the memory part differs."""
    memory = [{"id": "m_" + m["image_id"], "label_shown": m["label_shown"], "label_true": m["label_true"], "caption": caps[m["image_id"]]} for m in j["memory"]]
    text, det, refs = conditions.build(j["variant"], crop, kbd, caps[j["image_id"]], memory, j["image_id"], classes, j.get("retrieved_ids"))
    content = [llm.image_block(data.path(j["image_id"]))] + [llm.image_block(data.path(i)) for i in refs]
    content.append({"type": "text", "text": prompts.diagnosis_user(crop, classes, text)})
    r = llm.call(prompts.DIAG_SYS, content, stage=f"diag:{j['condition']}", tag=j["job_id"], reserve_usd=0.25 if refs else 0.04)
    pred = prompts.clean_prediction(r.get("parsed"), classes)
    return {**{k: v for k, v in j.items() if k != "memory"}, **pred, **det, "memory_pool_size": len(memory), "correct": pred["prediction"] == j["label"],
            "cost_usd": 0.0 if r["cache_hit"] else r.get("cost_usd"), "cache_hit": r["cache_hit"], "model_ids": r.get("model_ids"),
            "elapsed_s": r.get("elapsed_s"), "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens")}


def diagnose(crop, classes, kbd, caps, j) -> dict:
    j = with_shown_labels(j)
    if j["condition"] == "agentic":
        return agentic.diagnose(crop, classes, kbd, caps, j, k=agentic.K_READS, max_turns=6)
    if j["variant"] == "mace":
        return R._diagnose(crop, classes, kbd, caps, j)
    return variant_diagnosis(crop, classes, kbd, caps, j)


class Skipped(Exception):
    """The queue was stopped before this job started."""


def guarded(flag: threading.Event, fn, *args):
    if flag.is_set():
        raise Skipped
    return fn(*args)


def drain(futs, flag, on_result, fail_path) -> str | None:
    """Collect futures. A usage-limit or budget stop sets `flag`, so jobs not yet started finish at once as Skipped.
    (shutdown(cancel_futures=True) is not usable here: as_completed is never told about those futures and hangs.)"""
    stop, transient = None, 0
    for f in as_completed(futs):
        try:
            on_result(f.result())
        except Skipped:
            continue
        except (llm.CallFailed, llm.BudgetExceeded) as e:
            stop = stop or f"{type(e).__name__}: {str(e)[:200]}"
            flag.set()
        except Exception as e:  # transient failure: not written as a result, so a rerun picks the job up again
            transient += 1
            with fail_path.open("a") as fh:
                fh.write(json.dumps({"job": futs[f], "error": str(e)[:300], "t": time.time()}) + "\n")
            if transient >= MAX_TRANSIENT and not stop:
                stop = f"{transient} transient failures; last: {str(e)[:200]}"
                flag.set()
    return stop


def captions(selected, fail_path):
    need = {}
    for j in selected:
        for i in [j["image_id"]] + [m["image_id"] for m in (j["memory"] or [])]:
            need[i] = j["crop"]
    caps, stats = {}, Counter()

    def keep(res):
        i, c, r = res
        caps[i] = c
        stats["calls"] += 1; stats["cache_hits"] += bool(r["cache_hit"]); stats["empty"] += not c
        stats["usd_micro"] += 0 if r["cache_hit"] else round((r.get("cost_usd") or 0) * 1e6)

    flag = threading.Event()
    with ThreadPoolExecutor(C.PARALLEL) as ex:
        futs = {ex.submit(guarded, flag, R._caption, crop, i): i for i, crop in sorted(need.items())}
        stop = drain(futs, flag, keep, fail_path)
    if len(caps) < len(need) and not stop:
        stop = f"captions incomplete: {len(caps)}/{len(need)}"
    return caps, stats, stop


def save_captions(caps):
    p = OUT / "captions.json"
    allc = json.loads(p.read_text()) if p.exists() else {}
    allc.update(caps)
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(allc)); tmp.replace(p)


def ledger_rows(pkg, recs, cap_stats, other):
    rows = []
    for cond, n in sorted(Counter(r["condition"] for r in recs).items()):
        g = [r for r in recs if r["condition"] == cond]; hits = sum(bool(r["cache_hit"]) for r in g); usd = sum(r["cost_usd"] or 0 for r in g)
        rows.append(["actual", pkg, cond, n, hits, n - hits, round(usd / max(1, n - hits), 4), "", pkg == "P5", round(usd, 4), 0.0])
    if cap_stats["calls"]:
        new = cap_stats["calls"] - cap_stats["cache_hits"]; usd = cap_stats["usd_micro"] / 1e6
        rows.append(["actual", pkg, "caption", cap_stats["calls"], cap_stats["cache_hits"], new, round(usd / max(1, new), 4), "", pkg == "P5", round(usd, 4), 0.0])
    if abs(other) >= 0.0001:
        rows.append(["actual", pkg, "failed_or_uncounted", "", "", "", "", "", pkg == "P5", round(other, 4), 0.0])
    with LEDGER_CSV.open("a", newline="") as f:
        csv.writer(f).writerows(rows)


def remaining_forecast(pkg) -> float:
    """Planned USD of the packages that have no complete status yet."""
    done = {p.parent.name for p in OUT.glob("*/status.json") if json.loads(p.read_text()).get("complete")}
    if not LEDGER_CSV.exists():
        return 0.0
    total = 0.0
    with LEDGER_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["status"] != "planned":
                continue
            p = r["package"]
            if p == "P4":
                p = "P4labels" if r["condition"] == "labels" else "P4rest"
            elif p == "P7":
                p = "P7dev" if r["condition"] == "imageref_dev" else "P7"
            if p not in done:
                total += float(r["projected_usd"] or 0)
    return round(total, 2)


def run(pkg: str, cap_additional: float, limit: int | None = None) -> dict:
    d = OUT / pkg; d.mkdir(parents=True, exist_ok=True)
    pred_path, fail_path = d / "predictions.jsonl", d / "failures.jsonl"
    start = spent(); llm.set_budget(LEDGER_START + cap_additional)
    if pkg == "P5" and start >= OPTIONAL_STOP:
        print(f"P5 skipped: ledger {start:.2f} >= {OPTIONAL_STOP:.2f}"); return {"pkg": pkg, "skipped": True, "complete": False}
    selected = [j for j in (amend.build() if pkg in amend.PKGS else JB.build()) if SELECT[pkg](j)]
    total = len(selected)
    if limit:
        selected = selected[:limit]
    seen = {json.loads(l)["job_id"] for l in pred_path.open()} if pred_path.exists() else set()
    todo = [j for j in selected if j["job_id"] not in seen]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    with (d / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({"pkg": pkg, "jobs_total": total, "jobs_selected": len(selected), "already_done": len(seen), "limit": limit, "cap_additional": cap_additional,
                            "ledger_at_start": start, "budget": LEDGER_START + cap_additional, "git": git, "model_alias": C.MODEL, "data_rev": C.HF_REV,
                            "parallel": C.PARALLEL, "diag_system_sha": llm.sha(prompts.DIAG_SYS), "caption_system_sha": llm.sha(prompts.CAPTION_SYS),
                            "started": time.time()}) + "\n")
    print(f"{pkg}: {len(todo)} to do of {total}, ledger {start:.2f}", flush=True)
    caps, cap_stats, stop = captions(todo, fail_path)
    save_captions(caps)
    cap_usd = cap_stats["usd_micro"] / 1e6
    print(f"captions {cap_stats['calls']} ({cap_stats['cache_hits']} cached, {cap_stats['empty']} empty) ${cap_usd:.2f}", flush=True)
    recs = []
    if not stop:
        kbs = {c: (data.classes(c), KB.build(c, data.classes(c))) for c in {j["crop"] for j in todo}}
        with pred_path.open("a") as fh, ThreadPoolExecutor(C.PARALLEL) as ex:
            def keep(rec):
                fh.write(json.dumps(rec) + "\n"); fh.flush(); recs.append(rec)
                if len(recs) % 200 == 0:
                    print(f"{len(recs)}/{len(todo)} ledger {spent():.2f}", flush=True)
            flag = threading.Event()
            futs = {ex.submit(guarded, flag, diagnose, j["crop"], *kbs[j["crop"]], caps, j): j["job_id"] for j in todo}
            stop = drain(futs, flag, keep, fail_path)
    end = spent(); incurred = end - start
    ledger_rows(pkg, recs, cap_stats, incurred - cap_usd - sum(r["cost_usd"] or 0 for r in recs))
    rows_now = len(seen) + len(recs)
    new = [r for r in recs if not r["cache_hit"]]
    status = {"pkg": pkg, "jobs_total": total, "rows": rows_now, "complete": rows_now == total, "stop": stop, "incurred_usd": round(incurred, 4), "ledger_total": round(end, 4),
              "reserved_usd": 0.0, "new_calls": len(new), "mean_new_call_usd": round(sum(r["cost_usd"] or 0 for r in new) / max(1, len(new)), 4),
              "unknown_predictions": sum(r.get("prediction") == "UNKNOWN" for r in recs), "empty_captions": cap_stats["empty"],
              "model_ids": dict(Counter(m for r in new for m in (r.get("model_ids") or []))), "finished": time.time()}
    (d / "status.json").write_text(json.dumps(status, indent=1))
    status["remaining_forecast_usd"] = remaining_forecast(pkg)
    (d / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1), flush=True)
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pkg", choices=list(SELECT)); ap.add_argument("--cap-additional", type=float, required=True); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.pkg, a.cap_additional, a.limit)
