"""Job lists for every package of the ten-crop study, and a dry run that counts calls, cache hits and incremental cost
without calling any model.

  python -m mace10.jobs dryrun        writes mace10/out/dryrun.json and BUDGET_LEDGER.csv (planned rows)

Packages (rates are the user-confirmed all-inclusive planning rates; captions are counted separately at the measured
pilot mean, because the retrieval rate of $0.020 already includes one test caption):
  P1 hero primary      SAGE + MACE retrieval, seed 0, K=4, 1,600 queries
  P2 extra draws       MACE retrieval, seeds 1 and 2, K=4
  P3 memory size       MACE retrieval K=1 and K=16, seed 0, ablation panel (400)
  P4 content controls  labels / masked / shuffled / random, K=4, ablation panel
  P5 source (optional) same-source and other-source memory, seed 0, K=4, source panel
  P6 agentic           SAGE agent loop over caption files, k=4 reads, max 6 turns, efficiency panel (200)
  P7 image reference   the four retrieved references as images, efficiency panel (200) + development calibration (40)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from agmem import config as C, data, kb as KB, llm, prompts
from . import conditions

HERE = Path(__file__).resolve().parent; LOCK = HERE / "locked"; OUT = HERE / "out"
RATE = {"sage": 0.016, "retrieval_diag": 0.016, "caption": 0.0044, "agentic": 0.131, "imageref": 0.160}
# retrieval all-inclusive 0.020 = diagnosis 0.016 + test caption 0.0044 (pilot means); memory captions are preparation.
PACKAGES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
OPTIONAL = {"P5"}


def queries() -> pd.DataFrame:
    return pd.read_csv(LOCK / "queries.csv")


def mem(crop, seed, run, K, variant="mixed", test_collection=None):
    m = data.memory_set(crop, seed, run, K, variant, test_collection)
    return [{"image_id": r.image_id, "label": r.disease} for r in m.itertuples()]


def build() -> list[dict]:
    Q = queries(); J = []
    def add(pkg, cond, r, seed=None, K=None, memory=None, variant="mace", split=None):
        run = getattr(r, f"run_s{seed}") if seed is not None else None
        J.append({"pkg": pkg, "condition": cond, "variant": variant, "crop": r.crop, "image_id": r.image_id, "split": r.split, "label": r.label,
                  "collection": r.collection, "seed": seed, "run": run, "K": K, "memory": memory,
                  "job_id": f"{pkg}|{cond}|s{seed}|r{run}|K{K}|{r.image_id}"})
    for r in Q[Q.hero].itertuples():
        add("P1", "sage", r)
        add("P1", "mace", r, 0, 4, mem(r.crop, 0, r.run_s0, 4))
        for s in (1, 2):
            add("P2", "mace", r, s, 4, mem(r.crop, s, getattr(r, f"run_s{s}"), 4))
    for r in Q[Q.ablation].itertuples():
        for K in (1, 16):
            add("P3", "mace", r, 0, K, mem(r.crop, 0, r.run_s0, K))
        for v in ("labels", "masked", "shuffled", "random"):
            add("P4", v, r, 0, 4, mem(r.crop, 0, r.run_s0, 4), variant=v)
    for r in Q[Q.source].itertuples():
        for v in ("same_source", "other_source"):
            add("P5", v, r, 0, 4, mem(r.crop, 0, r.run_s0, 4, v, r.collection))
    for r in Q[Q.efficiency].itertuples():
        add("P6", "agentic", r, 0, 4, mem(r.crop, 0, r.run_s0, 4))
        add("P7", "imageref", r, 0, 4, mem(r.crop, 0, r.run_s0, 4), variant="imageref")
    for r in Q[Q.dev].itertuples():
        add("P7", "imageref_dev", r, 0, 4, mem(r.crop, 0, r.run_s0, 4), variant="imageref")
    return J


def caption_key(image_id: str, crop: str) -> str:
    content = [llm.image_block(data.path(image_id)), {"type": "text", "text": prompts.caption_text(crop)}]
    return llm.sha({"m": C.MODEL, "s": prompts.CAPTION_SYS, "c": content})


def cached(key: str) -> bool:
    p = C.CACHE / "llm" / f"{key}.json"
    return p.exists() and not llm.looks_failed(json.loads(p.read_text()).get("text"), json.loads(p.read_text()).get("is_error"))


def diag_key(j, caps, kbd, classes):
    if j["image_id"] not in caps or any(m["image_id"] not in caps for m in (j["memory"] or [])):
        return None
    memory = None if j["condition"] == "sage" else [{"id": "m_" + m["image_id"], "label_shown": m["label"], "label_true": m["label"], "caption": caps[m["image_id"]]}
                                                     for m in j["memory"]]
    if j["condition"] == "sage":
        from agmem import packet
        text, _ = packet.build(j["crop"], kbd, caps[j["image_id"]], None)
    else:
        text, _, refs = conditions.build(j["variant"], j["crop"], kbd, caps[j["image_id"]], memory, j["image_id"], classes)
        if refs:
            return None
    content = [llm.image_block(data.path(j["image_id"])), {"type": "text", "text": prompts.diagnosis_user(j["crop"], classes, text)}]
    return llm.sha({"m": C.MODEL, "s": prompts.DIAG_SYS, "c": content})


def dryrun():
    J = build(); start = llm.spent()
    need = defaultdict(set)
    for j in J:
        need[j["crop"]].add(j["image_id"])
        for m in j["memory"] or []:
            need[j["crop"]].add(m["image_id"])
    caps, cap_miss, cap_by_pkg = {}, 0, Counter()
    first_pkg = {}
    for j in J:
        for i in [j["image_id"]] + [m["image_id"] for m in (j["memory"] or [])]:
            first_pkg.setdefault((j["crop"], i), j["pkg"])
    for crop, ids in need.items():
        for i in ids:
            k = caption_key(i, crop); p = C.CACHE / "llm" / f"{k}.json"
            if cached(k):
                caps[i] = llm.parse_json(json.loads(p.read_text())["text"]) or {}
            else:
                cap_miss += 1; cap_by_pkg[first_pkg[(crop, i)]] += 1
    kbs = {c: (KB.build(c, data.classes(c)), data.classes(c)) for c in need}
    calls, hits = Counter(), Counter()
    for j in J:
        calls[(j["pkg"], j["condition"])] += 1
        if j["condition"] in ("agentic",):
            continue
        k = diag_key(j, caps, *kbs[j["crop"]])
        if k and cached(k):
            hits[(j["pkg"], j["condition"])] += 1
    rows = []
    for (pkg, cond), n in sorted(calls.items()):
        miss = n - hits[(pkg, cond)]
        rate = RATE["agentic"] if cond == "agentic" else RATE["imageref"] if cond.startswith("imageref") else RATE["sage"] if cond == "sage" else RATE["retrieval_diag"]
        rows.append({"package": pkg, "condition": cond, "calls": n, "cache_hits": hits[(pkg, cond)], "new_calls": miss, "rate_usd": rate,
                     "projected_usd": round(miss * rate, 2), "optional": pkg in OPTIONAL})
    for pkg, n in sorted(cap_by_pkg.items()):
        rows.append({"package": pkg, "condition": "caption", "calls": n, "cache_hits": 0, "new_calls": n, "rate_usd": RATE["caption"],
                     "projected_usd": round(n * RATE["caption"], 2), "optional": pkg in OPTIONAL})
    D = pd.DataFrame(rows)
    tot = D.projected_usd.sum(); opt = D[D.optional].projected_usd.sum()
    summary = {"ledger_start_usd": round(start, 2), "cap_additional_usd": 400.0, "hard_stop_ledger_total": round(start + 400, 2),
               "optional_stop_ledger_total": round(start + 350, 2), "projected_additional_usd": round(tot, 2),
               "projected_required_usd": round(tot - opt, 2), "projected_optional_usd": round(opt, 2),
               "captions_needed": sum(len(v) for v in need.values()), "captions_cached": len(caps), "captions_new": cap_miss,
               "note": "Image-reference rate 0.160 is provisional until the 40-item development calibration; agentic and image-reference "
                       "cache hits are not checked (no cached runs exist for this protocol)."}
    D.to_csv(OUT / "dryrun_packages.csv", index=False); (OUT / "dryrun.json").write_text(json.dumps(summary, indent=1))
    print(D.to_string(index=False)); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"dryrun": dryrun}[sys.argv[1]]()
