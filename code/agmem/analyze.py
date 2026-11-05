"""Summarise one run directory: accuracy per (condition, K, test set) as mean +- SD over the runs of the seed, and
paired comparisons on identical items with exact McNemar tests (Holm-corrected within each family).

  python -m agmem.analyze --crop Soybean --seed 0 --n 15
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C, data
from .run import out_dir
from .stats import holm, mcnemar_exact

SETS = {"IID": ["test_iid"], "OOD in-scope": ["test_ood_in_scope"], "OOD out-of-scope": ["test_ood_out_of_scope"], "Rare-disease probes": ["test_probe"]}


def load(crop, seed, n, tag="") -> pd.DataFrame:
    """One row per (condition, K, seed, run, image). SAGE (memory-independent) is expanded onto every test row."""
    d = out_dir(crop, seed, n, tag)
    P = pd.DataFrame(map(json.loads, (d / "predictions.jsonl").open()))
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]
    rest = P[P.condition != "sage"].copy(); rest["K"] = rest.K.astype(int)
    keys = rest[["seed", "run", "image_id", "split", "label"]].drop_duplicates()
    sage = keys.merge(P[P.condition == "sage"][["image_id", "prediction", "correct"]], on="image_id")
    sage["condition"], sage["K"] = "sage", 0
    cols = ["seed", "run", "image_id", "split", "label", "prediction", "correct", "condition", "K"]
    D = pd.concat([sage[cols], rest[cols]], ignore_index=True); D["rid"] = D.seed.astype(int) * 10 + D.run.astype(int)
    return D


def acc_table(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, K), g in D.groupby(["condition", "K"]):
        for name, splits in SETS.items():
            h = g[g.split.isin(splits)]
            if h.empty:
                continue
            per = h.groupby("rid").correct.mean() * 100
            rows.append({"condition": cond, "K": int(K), "set": name, "mean": per.mean(), "sd": per.std(ddof=1) if len(per) > 1 else 0.0,
                         "n_per_run": int(h.groupby("rid").size().mean()), "runs": len(per)})
    return pd.DataFrame(rows)


def paired(D, a_cond, a_K, b_cond, b_K, splits, items=None):
    A = D[(D.condition == a_cond) & (D.K == a_K) & D.split.isin(splits)].set_index(["rid", "image_id"]).correct
    B = D[(D.condition == b_cond) & (D.K == b_K) & D.split.isin(splits)].set_index(["rid", "image_id"]).correct
    idx = A.index.intersection(B.index)
    if items is not None:
        idx = idx.intersection(items)
    a, b = A.loc[idx].astype(bool), B.loc[idx].astype(bool)
    fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
    per = (b.groupby(level=0).mean() - a.groupby(level=0).mean()) * 100
    return {"n_pairs": len(idx), "a_acc": 100 * a.mean(), "b_acc": 100 * b.mean(), "diff_pp": 100 * (b.mean() - a.mean()),
            "diff_sd_over_runs": per.std(ddof=1) if len(per) > 1 else 0.0, "runs_b_better": int((per > 0).sum()), "runs": int(len(per)), "fixed": fixed, "broken": broken, "p": mcnemar_exact(fixed, broken)}


def main(crop, seed, n, tag=""):
    D = load(crop, seed, n, tag); T = acc_table(D); out = {"accuracy": T.to_dict("records"), "comparisons": {}}
    Ks = sorted(int(k) for k in D[D.condition == "agmem"].K.unique()); sets = [s for s in SETS if D.split.isin(SETS[s]).any()]
    fam = []
    for K in Ks:
        for name in sets:
            k = f"agmem K={K} vs sage | {name}"; out["comparisons"][k] = paired(D, "sage", 0, "agmem", K, SETS[name]); fam.append(k)
    for k1, k2 in zip(Ks, Ks[1:]):
        for name in sets:
            k = f"agmem K={k2} vs K={k1} | {name}"; out["comparisons"][k] = paired(D, "agmem", k1, "agmem", k2, SETS[name]); fam.append(k)
    if (D.condition == "same_source").any():
        pe = D[D.condition == "same_source"].set_index(["rid", "image_id"]).index.unique()
        for K in Ks:
            for v in ("same_source", "other_source"):
                out["comparisons"][f"{v} vs agmem K={K} | paired IID"] = paired(D, "agmem", K, v, K, ["test_iid"], pe)
    if (D.condition == "shuffled").any():
        for K in Ks:
            out["comparisons"][f"shuffled vs agmem K={K} | IID"] = paired(D, "agmem", K, "shuffled", K, ["test_iid"])
    for k, p in zip(fam, holm([out["comparisons"][k]["p"] for k in fam])):
        out["comparisons"][k]["p_holm"] = p
    d = out_dir(crop, seed, n, tag); (d / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    md = markdown(T, out, crop, Ks, sets); (d / "summary.md").write_text(md); print(md)


def markdown(T, out, crop, Ks, sets) -> str:
    def cell(cond, K, st):
        r = T[(T.condition == cond) & (T.K == K) & (T.set == st)]
        return f"{r['mean'].iloc[0]:.1f} ± {r['sd'].iloc[0]:.1f}" if len(r) else "n/a"
    runs = int(T.runs.max()); n = {st: int(T[(T.set == st) & (T.condition == "sage")].n_per_run.iloc[0]) for st in sets}
    L = [f"## {crop}: accuracy (%), mean ± SD over {runs} runs\n", "| Condition | K | " + " | ".join(f"{st} ({n[st]}/run)" for st in sets) + " |",
         "|---|---|" + "---|" * len(sets), "| SAGE (KB only) | - | " + " | ".join(cell("sage", 0, st) for st in sets) + " |"]
    L += [f"| Ag-MEM | {K} | " + " | ".join(cell("agmem", K, st) for st in sets) + " |" for K in Ks]
    if (T.condition == "shuffled").any():
        L += [f"| Ag-MEM, shuffled labels | {K} | " + " | ".join(cell("shuffled", K, st) for st in sets) + " |" for K in Ks]
    L += ["\n## Paired differences (pooled over runs; exact McNemar, Holm over all rows)\n",
          "| Comparison | set | A | B | B - A (pp) | SD over runs | runs B > A | fixed / broken | p (Holm) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["comparisons"].items():
        if "p_holm" not in r: continue
        comp, st = k.split(" | ")
        L.append(f"| {comp} | {st} | {r['a_acc']:.1f} | {r['b_acc']:.1f} | {r['diff_pp']:+.1f} | {r['diff_sd_over_runs']:.1f} | {r['runs_b_better']}/{r['runs']} | {r['fixed']} / {r['broken']} | {r['p_holm']:.2g} |")
    if any("same_source" in k for k in out["comparisons"]):
        L += ["\n## Same vs other source (paired IID items)\n", "| K | pairs | mixed | same source | other source |", "|---|---|---|---|---|"]
        for K in Ks:
            a_ = out["comparisons"][f"same_source vs agmem K={K} | paired IID"]; b_ = out["comparisons"][f"other_source vs agmem K={K} | paired IID"]
            L.append(f"| {K} | {a_['n_pairs']} | {a_['a_acc']:.1f} | {a_['b_acc']:.1f} | {b_['b_acc']:.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", required=True); ap.add_argument("--seed", default="0"); ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); sd = [int(x) for x in a.seed.split(",")]
    main(a.crop, sd[0] if len(sd) == 1 else sd, a.n, a.tag)
