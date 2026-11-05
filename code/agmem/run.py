"""Run Ag-MEM and its variants (plus the SAGE KB-only baseline) on one crop of the frozen benchmark.

  python -m agmem.run plan --crop Soybean --seed 0 --n 15
  python -m agmem.run run  --crop Soybean --seed 0 --n 15 --cap 700

Conditions (identical model, system prompt, class list, KB and schema; only the memory differs):
  sage          KB only, no memory (memory-independent, so each test image is diagnosed once)
  agmem         KB + 4 captions retrieved from the mixed memory (K per disease, every K in KS)
  same_source   memory only from the test image's own collection   (paired IID items)
  other_source  memory with the test image's collection removed    (paired IID items)
  shuffled      mixed memory with permuted labels                  (IID and OOD in-scope items)
Order of execution puts the headline conditions first, so a budget stop never leaves them half done.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as C, data, kb as KB, llm, packet, prompts

DIAG_COST, CAPTION_COST = 0.0162, 0.0041   # measured means (USD per call) from earlier soybean runs
AGENTIC_COST = 0.123                       # measured mean, 4-image calibration 2026-11-03


def out_dir(crop, seed, n, tag=""):
    seeds = "-".join(map(str, seed)) if isinstance(seed, (list, tuple)) else seed
    return C.RUNS / f"{data.slug(crop)}_seed{seeds}_n{n or 'all'}{('_' + tag) if tag else ''}"


def build_jobs(crop: str, seed, n: int | None, ks=None, conditions=None, splits=None) -> list[dict]:
    """seed: int or list of ints. ks / conditions / splits restrict the job set (None = all)."""
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]
    ks = tuple(ks or C.KS); conditions = set(conditions or ("sage", "agmem", "same_source", "other_source", "shuffled"))
    jobs = []
    for sd in seeds:
        jobs += _jobs_one_seed(crop, sd, n, ks, conditions, splits)
    # SAGE is memory-independent: keep one job per image across seeds
    seen, out = set(), []
    for j in jobs:
        if j["condition"] == "sage":
            if j["image_id"] in seen: continue
            seen.add(j["image_id"])
        out.append(j)
    return out


def _jobs_one_seed(crop, seed, n, ks, conditions, splits):
    T = data.test_items(crop, seed, n)
    if splits:
        T = T[T.split.isin(splits)]
    P = data.paired(); P = P[(P.crop == crop) & (P.seed == seed)]
    pe = set(zip(P.run, P.image_id))
    base = lambda t: {"seed": seed, "run": int(t.run), "image_id": t.image_id, "split": t.split, "label": t.disease, "collection": t.collection}
    jobs = [{"condition": "sage", "seed": None, "K": None, "run": None, "image_id": i, "split": None, "label": data.images().loc[i, "class"],
             "collection": data.images().loc[i, "collection"], "memory": None} for i in sorted(T.image_id.unique())] if "sage" in conditions else []
    mem = lambda df, shown="disease": [{"image_id": r.image_id, "label_shown": getattr(r, shown), "label_true": r.disease} for r in df.itertuples()]
    order = {"agmem": [], "agentic": [], "same_source": [], "other_source": [], "shuffled": []}
    for K in ks:
        for run in sorted(T.run.unique()):
            mixed = data.memory_set(crop, seed, run, K)
            for t in T[T.run == run].itertuples():
                if "agmem" in conditions:
                    order["agmem"].append({**base(t), "condition": "agmem", "K": K, "memory": mem(mixed)})
                if "agentic" in conditions:
                    order["agentic"].append({**base(t), "condition": "agentic", "K": K, "memory": mem(mixed)})
                if t.split == "test_iid" and (run, t.image_id) in pe:
                    for v in [x for x in ("same_source", "other_source") if x in conditions]:
                        order[v].append({**base(t), "condition": v, "K": K, "memory": mem(data.memory_set(crop, seed, run, K, v, t.collection))})
                if "shuffled" in conditions and t.split in ("test_iid", "test_ood_in_scope"):
                    order["shuffled"].append({**base(t), "condition": "shuffled", "K": K, "memory": mem(mixed, "shuffled_class")})
    for v in order.values():
        jobs += v
    for j in jobs:
        j["job_id"] = f"{j['condition']}|s{j['seed']}|K{j['K']}|r{j['run']}|{j['image_id']}"
    return jobs


def caption_jobs(jobs) -> list[str]:
    ids = {j["image_id"] for j in jobs}
    for j in jobs:
        ids |= {m["image_id"] for m in (j["memory"] or [])}
    return sorted(ids)


def _cached(system, content) -> bool:
    return (C.CACHE / "llm" / f"{llm.sha({'m': C.MODEL, 's': system, 'c': content})}.json").exists()


def plan(crop, seed, n, write=True, ks=None, conditions=None, splits=None, tag=""):
    jobs = build_jobs(crop, seed, n, ks, conditions, splits); caps = caption_jobs(jobs)
    from collections import Counter
    byc = Counter(j["condition"] for j in jobs)
    est = len(caps) * CAPTION_COST + sum(AGENTIC_COST if j["condition"] == "agentic" else DIAG_COST for j in jobs)
    p = {"crop": crop, "seed": seed, "n_per_disease_per_run": n, "test_rows": sum(1 for j in jobs if j["condition"] == "agmem") // max(1, len(ks or C.KS)),
         "conditions": sorted(set(conditions or [])) or "all", "splits": splits or "all",
         "distinct_test_images": len({j["image_id"] for j in jobs if j["condition"] == "sage"}), "captions": len(caps),
         "diagnoses_by_condition": dict(byc), "diagnoses_total": len(jobs), "Ks": list(ks or C.KS),
         "estimate_usd_upper_bound_no_cache": round(est, 2), "data": str(C.DATA), "data_rev": C.HF_REV}
    if write:
        d = out_dir(crop, seed, n, tag); d.mkdir(parents=True, exist_ok=True); (d / "plan.json").write_text(json.dumps(p, indent=1))
    return p, jobs, caps


def _caption(crop, image_id):
    r = llm.call(prompts.CAPTION_SYS, [llm.image_block(data.path(image_id)), {"type": "text", "text": prompts.caption_text(crop)}],
                 stage="caption", tag=image_id)
    return image_id, (r["parsed"] if isinstance(r.get("parsed"), dict) else {}), r


def _diagnose(crop, classes, kbd, caps, j):
    memory = None
    if j["memory"] is not None:
        memory = [{"id": "m_" + m["image_id"], "label_shown": m["label_shown"], "label_true": m["label_true"], "caption": caps[m["image_id"]]}
                  for m in j["memory"]]
    text, det = packet.build(crop, kbd, caps[j["image_id"]], memory)
    content = [llm.image_block(data.path(j["image_id"])), {"type": "text", "text": prompts.diagnosis_user(crop, classes, text)}]
    r = llm.call(prompts.DIAG_SYS, content, stage=f"diag:{j['condition']}", tag=j["job_id"])
    pred = prompts.clean_prediction(r.get("parsed"), classes)
    return {**{k: v for k, v in j.items() if k != "memory"}, **pred, **det, "memory_pool_size": len(j["memory"] or []),
            "correct": pred["prediction"] == j["label"], "cost_usd": 0.0 if r["cache_hit"] else r.get("cost_usd"),
            "cache_hit": r["cache_hit"], "model_ids": r.get("model_ids")}


def run(crop, seed, n, cap, ks=None, conditions=None, splits=None, tag=""):
    p, jobs, cap_ids = plan(crop, seed, n, True, ks, conditions, splits, tag)
    d = out_dir(crop, seed, n, tag); start = llm.spent(); llm.set_budget(start + cap)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    (d / "manifest.json").write_text(json.dumps({**p, "cap_usd": cap, "ledger_start": start, "git": git, "started": time.time(),
                                                 "diag_system_sha": llm.sha(prompts.DIAG_SYS), "caption_system_sha": llm.sha(prompts.CAPTION_SYS)}, indent=1))
    print(json.dumps(p, indent=1), flush=True)
    classes = data.classes(crop); kbd = KB.build(crop, classes)
    (d / "kb_used.json").write_text(json.dumps({c: {"kb_source": v["kb_source"], "n_claims": len(v["claims"])} for c, v in kbd.items()}, indent=1))
    caps = {}
    with ThreadPoolExecutor(C.PARALLEL) as ex:
        for f in as_completed([ex.submit(_caption, crop, i) for i in cap_ids]):
            i, c, _ = f.result(); caps[i] = c
    (d / "captions.json").write_text(json.dumps(caps))
    print(f"captions {len(caps)}  spent {llm.spent() - start:.2f}", flush=True)
    done = 0; stopped = None
    with (d / "predictions.jsonl").open("w") as fh, ThreadPoolExecutor(C.PARALLEL) as ex:
        from . import agentic
        futs = [ex.submit(agentic.diagnose if j["condition"] == "agentic" else _diagnose, crop, classes, kbd, caps, j) for j in jobs]
        for f in as_completed(futs):
            try:
                rec = f.result()
            except llm.BudgetExceeded as e:
                stopped = str(e); continue
            fh.write(json.dumps(rec) + "\n"); done += 1
            if done % 500 == 0:
                fh.flush(); print(f"{done}/{len(jobs)} diagnoses  spent {llm.spent() - start:.2f}", flush=True)
    summary = {"done": done, "jobs": len(jobs), "spent_usd": round(llm.spent() - start, 4), "budget_stop": stopped, "finished": time.time()}
    (d / "run_status.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plan", "run"]); ap.add_argument("--crop", required=True)
    ap.add_argument("--seed", default="0", help="one seed or a comma list, e.g. 0,1,2"); ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--cap", type=float, default=0.0); ap.add_argument("--ks", default=None, help="e.g. 1,4,16")
    ap.add_argument("--conditions", default=None, help="e.g. sage,agmem"); ap.add_argument("--splits", default=None, help="e.g. test_iid,test_ood_in_scope")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seed = [int(x) for x in a.seed.split(",")]; seed = seed[0] if len(seed) == 1 else seed
    kw = dict(ks=[int(x) for x in a.ks.split(",")] if a.ks else None, conditions=a.conditions.split(",") if a.conditions else None,
              splits=a.splits.split(",") if a.splits else None, tag=a.tag)
    if a.cmd == "plan":
        print(json.dumps(plan(a.crop, seed, a.n, True, **kw)[0], indent=1))
    else:
        assert a.cap > 0, "--cap is required for a paid run"
        run(a.crop, seed, a.n, a.cap, **kw)
