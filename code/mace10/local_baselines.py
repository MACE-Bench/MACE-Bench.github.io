"""No-model baselines on the exact memory bank MACE sees (zero paid calls).

  python -m mace10.local_baselines index     write the image list for DINOv2 (queries + every bank image)
  <torch python> mace10/embed_dinov2.py      DINOv2 ViT-S/14 embeddings (pinned hub commit, as in the pre-study)
  python -m mace10.local_baselines run       1-NN labels (OpenCLIP from the release, DINOv2 local) per query, seed, K

Bank for a query = data.memory_set(crop, seed, run_s{seed}, K) (mixed memory, K images per disease). The retrieved-label
votes need captions (paid) and are computed by mace10.analyze once captions exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agmem import data

HERE = Path(__file__).resolve().parent
LOCK = HERE / "locked"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
KS_HERO, KS_ABL = (4,), (1, 4, 16)
PILOT = HERE.parent / "runs/soybean_seed0_n5_pilot/predictions.jsonl"


def banks():
    """Every (crop, seed, run, K) bank used by any panel, plus the soybean pilot banks."""
    Q = pd.read_csv(LOCK / "queries.csv"); B = set()
    for r in Q[Q.hero].itertuples():
        for s in (0, 1, 2):
            B.add((r.crop, s, getattr(r, f"run_s{s}"), 4))
        if r.ablation:
            for K in KS_ABL:
                B.add((r.crop, 0, r.run_s0, K))
    for run in range(5):
        for K in (1, 4, 16):
            B.add(("Soybean", 0, run, K))
    return sorted(B)


def pilot_items():
    P = pd.DataFrame(map(json.loads, PILOT.open()))
    return P[P.condition == "agmem"][["run", "K", "image_id", "split", "label"]].drop_duplicates()


def index():
    Q = pd.read_csv(LOCK / "queries.csv"); ids = set(Q.image_id) | set(pilot_items().image_id)
    for b in banks():
        ids |= set(data.memory_set(*b).image_id)
    ids = sorted(ids)
    rows = [{"image_id": i, "path": str(data.path(i))} for i in ids]
    (OUT / "dinov2_index.json").write_text(json.dumps(rows)); print(len(rows), "images;", len(banks()), "banks")


def nn_label(E, idx, q, bank_ids, bank_lab):
    M = E[[idx[i] for i in bank_ids]]; s = M @ E[idx[q]]; j = int(np.argmax(s))
    return bank_lab[j], float(s[j])


def run():
    z = np.load(data.C.DATA / "audit/clip_vitb32_embeddings.npz"); Ec = z["E"].astype(np.float32); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    ic = {k: i for i, k in enumerate(z["ids"])}
    d = np.load(OUT / "dinov2_embeddings.npz"); Ed = d["E"].astype(np.float32); idd = {k: i for i, k in enumerate(d["ids"])}
    Q = pd.read_csv(LOCK / "queries.csv"); rows = []
    cache = {}
    def bank(b):
        if b not in cache:
            m = data.memory_set(*b); cache[b] = (m.image_id.tolist(), m.disease.values)
        return cache[b]
    for r in Q[Q.hero | Q.dev].itertuples():
        for s in (0, 1, 2):
            for K in (KS_ABL if (s == 0 and r.ablation) else KS_HERO):
                ids, lab = bank((r.crop, s, getattr(r, f"run_s{s}"), K))
                lc, cc = nn_label(Ec, ic, r.image_id, ids, lab); ld, cd = nn_label(Ed, idd, r.image_id, ids, lab)
                rows.append({"panel": "hero", "crop": r.crop, "image_id": r.image_id, "split": r.split, "label": r.label, "seed": s, "K": K,
                             "run": getattr(r, f"run_s{s}"), "bank_size": len(ids), "clip_pred": lc, "clip_cos": cc, "dino_pred": ld, "dino_cos": cd,
                             "hero": r.hero, "ablation": r.ablation, "efficiency": r.efficiency, "dev": r.dev})
    for r in pilot_items().itertuples():
        ids, lab = bank(("Soybean", 0, int(r.run), int(r.K)))
        lc, cc = nn_label(Ec, ic, r.image_id, ids, lab); ld, cd = nn_label(Ed, idd, r.image_id, ids, lab)
        rows.append({"panel": "pilot", "crop": "Soybean", "image_id": r.image_id, "split": "iid" if r.split == "test_iid" else "ood", "label": r.label,
                     "seed": 0, "K": int(r.K), "run": int(r.run), "bank_size": len(ids), "clip_pred": lc, "clip_cos": cc, "dino_pred": ld, "dino_cos": cd})
    D = pd.DataFrame(rows); D["clip_correct"] = D.clip_pred == D.label; D["dino_correct"] = D.dino_pred == D.label
    D.to_csv(OUT / "local_baselines.csv", index=False)
    h = D[(D.panel == "hero") & D.hero & (D.seed == 0) & (D.K == 4)]
    print(h.groupby("split")[["clip_correct", "dino_correct"]].mean().round(3))
    p = D[D.panel == "pilot"]; print(p.groupby(["K", "split"])[["clip_correct", "dino_correct"]].mean().round(3))


if __name__ == "__main__":
    {"index": index, "run": run}[sys.argv[1]]()
