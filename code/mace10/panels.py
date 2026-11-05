"""Lock the ten-crop cohort and every query panel from metadata only (no model calls, no accuracy).

  python -m mace10.panels        writes mace10/locked/{cohort.json, queries.csv, panels.json}

Rules (fixed before any comparative accuracy is read):
- Cohort: Soybean, Corn, Tomato, plus seven crops from those with a held-out collection, eligible if they have
  >= 60 held-out in-scope images, >= 4 IID diseases and >= 2 IID collections; ranked by held-out disease count, then
  held-out image count.
- Hero panel: per crop up to 100 unique IID and 60 unique held-out (OOD) images, round-robin over diseases in the
  release's seeded budget_rank order (near-balanced; sparse diseases run out naturally; no padding).
- Memory selections: release seeds 0 (primary), 1, 2. For each seed, an IID query uses run 0 if it is a test image of
  run 0, otherwise run 2 (runs 0 and 2 together test every fold), so memory never contains the query. OOD queries use
  run 0. Memory = the release's nested facility-location set for that (seed, run, K).
- Nested panels: ablation = first 20 IID + 20 OOD per crop of the hero order (400); efficiency = the ablation items of
  five crops (Soybean, Corn, Tomato, Mango, Rice; 200); development calibration = 4 IID + 4 OOD per efficiency crop
  taken after the hero items (40, disjoint); source panel = first 20 IID per crop that are paired-eligible for
  same- vs other-source memory under seed 0 (optional).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agmem import data

OUT = Path(__file__).resolve().parent / "locked"
REQUIRED = ["Soybean", "Corn", "Tomato"]
EFFICIENCY = ["Soybean", "Corn", "Tomato", "Mango", "Rice"]
N_IID, N_OOD, N_ABL = 100, 60, 20
SEEDS = (0, 1, 2)


def cohort() -> tuple[list[str], pd.DataFrame]:
    I = data.images(); rows = []
    for crop in sorted(I[I.role == "ood_in_scope"].crop.unique()):
        c = I[I.crop == crop]; iid = c[c.role == "iid"]; ood = c[c.role == "ood_in_scope"]
        rows.append({"crop": crop, "iid": len(iid), "iid_diseases": iid["class"].nunique(), "iid_collections": iid.collection.nunique(),
                     "ood": len(ood), "ood_diseases": ood["class"].nunique(), "ood_collection": ood.collection.iloc[0],
                     "candidates": c["class"].nunique()})
    T = pd.DataFrame(rows)
    T["eligible"] = (T.ood >= 60) & (T.iid_diseases >= 4) & (T.iid_collections >= 2)
    rest = T[T.eligible & ~T.crop.isin(REQUIRED)].sort_values(["ood_diseases", "ood"], ascending=False)
    chosen = REQUIRED + rest.crop.head(7).tolist()
    T["selected"] = T.crop.isin(chosen)
    return chosen, T


def round_robin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.reset_index(drop=True).sort_values(["class", "budget_rank", "image_id"]).copy()
    df["k"] = df.groupby("class").cumcount()
    return df.sort_values(["k", "class"]).head(n).drop(columns="k")


def run_for(seed: int, crop: str, image_id: str, split: str) -> int:
    if split == "ood":
        return 0
    A = data.assignments(); a = A[(A.crop == crop) & (A.seed == seed) & (A.image_id == image_id) & (A.split == "test_iid")]
    runs = set(a.run)
    assert runs, (crop, seed, image_id)
    return 0 if 0 in runs else 2 if 2 in runs else min(runs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops, T = cohort(); I = data.images()
    P = data.paired(); rows = []
    for crop in crops:
        c = I[I.crop == crop]
        for split, role, n in (("iid", "iid", N_IID), ("ood", "ood_in_scope", N_OOD)):
            pool = c[c.role == role]; picked = round_robin(pool, n + (4 if crop in EFFICIENCY else 0))
            for pos, r in enumerate(picked.itertuples()):
                rec = {"crop": crop, "image_id": r.image_id, "split": split, "label": I.loc[r.image_id, "class"], "collection": r.collection, "order": pos}
                rec["hero"] = pos < n
                rec["ablation"] = pos < N_ABL
                rec["efficiency"] = rec["ablation"] and crop in EFFICIENCY
                rec["dev"] = (pos >= n) and crop in EFFICIENCY
                for s in SEEDS:
                    rec[f"run_s{s}"] = run_for(s, crop, r.image_id, split)
                rows.append(rec)
    Q = pd.DataFrame(rows)
    # optional source panel: first 20 hero IID per crop that are paired-eligible under seed 0 in their primary run
    pe = set(zip(P.crop, P.run, P.image_id))
    Q["source"] = False
    for crop in crops:
        m = (Q.crop == crop) & (Q.split == "iid") & Q.hero
        ok = Q[m][[ (crop, r, i) in pe for r, i in zip(Q[m].run_s0, Q[m].image_id)]].sort_values("order").head(20).index
        Q.loc[ok, "source"] = True
    Q.to_csv(OUT / "queries.csv", index=False)
    T.to_csv(OUT / "cohort_audit.csv", index=False)
    summ = {"crops": crops, "rule": __doc__.split("Rules")[1].strip(),
            "counts": {p: int(Q[p].sum()) for p in ("hero", "ablation", "efficiency", "dev", "source")},
            "per_crop": Q[Q.hero].groupby(["crop", "split"]).size().unstack().to_dict(orient="index"),
            "per_crop_diseases": Q[Q.hero].groupby(["crop", "split"]).label.nunique().unstack().to_dict(orient="index")}
    (OUT / "panels.json").write_text(json.dumps(summ, indent=1, default=int))
    print(json.dumps({k: summ[k] for k in ("crops", "counts", "per_crop")}, indent=1, default=int))


if __name__ == "__main__":
    main()
