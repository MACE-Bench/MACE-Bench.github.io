"""Locked analysis of the ten-crop study (LOCKED_ANALYSIS.md). Reads out/<PKG>/predictions.jsonl, makes no model call.

  python -m mace10.results      writes out/results_wide.csv (one row per query, one column per method) and out/results.json

Accuracy: equal-crop mean. Intervals: paired 95% bootstrap over unique query images, stratified by crop, 2,000 resamples,
seed 0. Holm across the three primary contrasts x two splits. A column exists only if its package produced every row of it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; OUT = HERE / "out"; LOCK = HERE / "locked"; RUNS = HERE.parent / "runs"
B = 2000
T0 = 1793860000       # ledger time of the first ten-crop call


def load(pkg):
    p = OUT / pkg / "predictions.jsonl"
    return pd.DataFrame(map(json.loads, p.open())) if p.exists() else pd.DataFrame()


def vote(labels, weights=None):
    sc = {}
    for i, l in enumerate(labels or []):
        sc[l] = sc.get(l, 0) + (weights[i] if weights else 1)
    return max(sorted(sc), key=lambda k: sc[k]) if sc else None


def wide() -> pd.DataFrame:
    Q = pd.read_csv(LOCK / "queries.csv"); Q = Q[~Q.dev].copy().set_index("image_id")
    L = pd.read_csv(OUT / "local_baselines.csv"); L = L[L.panel == "hero"]
    for s in (0, 1, 2):
        g = L[(L.seed == s) & (L.K == 4)].set_index("image_id")
        Q[f"dino_s{s}"] = g.dino_correct.astype(float); Q[f"clip_s{s}"] = g.clip_correct.astype(float)
    for K in (1, 16):
        g = L[(L.seed == 0) & (L.K == K)].set_index("image_id")
        Q[f"dino_K{K}"] = g.dino_correct.astype(float); Q[f"clip_K{K}"] = g.clip_correct.astype(float)

    def put(name, g):
        if len(g):
            Q[name] = g.drop_duplicates("image_id").set_index("image_id").correct.astype(float)
    P1 = load("P1")
    if len(P1):
        put("sage", P1[P1.condition == "sage"])
        M = P1[P1.condition == "mace"]; put("mace", M); M = M.set_index("image_id")
        Q["hit"] = pd.Series({i: float(r.label in (r.memory_labels_shown or [])) for i, r in M.iterrows()})
        Q["majority"] = pd.Series({i: float(vote(r.memory_labels_shown) == r.label) for i, r in M.iterrows()})
        Q["wvote"] = pd.Series({i: float(vote(r.memory_labels_shown, r.memory_relevance) == r.label) for i, r in M.iterrows()})
    P2 = load("P2")
    for s in (1, 2):
        if len(P2):
            put(f"mace_s{s}", P2[P2.seed == s])
    P3 = load("P3")
    for K in (1, 16):
        if len(P3):
            put(f"mace_K{K}", P3[P3.K == K])
    for pkg, conds in (("P4labels", ["labels"]), ("P4rest", ["masked", "shuffled", "random"]), ("P5", ["same_source", "other_source"]), ("P6", ["agentic"]), ("P7", ["imageref"])):
        D = load(pkg)
        for c in conds:
            if len(D):
                put(c, D[D.condition == c])
    P8, P9 = load("P8"), load("P9")
    if len(P8) and "labels" in Q:
        Q["labels_all"] = Q.labels.fillna(P8.set_index("image_id").correct.astype(float))
    if len(P9):
        put("dino_labels", P9)
        M9 = P9.set_index("image_id")
        Q["hit_dino"] = pd.Series({i: float(r.label in (r.memory_labels_shown or [])) for i, r in M9.iterrows()})
    return Q.reset_index()


def eq_mean(df, col):
    return float(np.mean([g[col].mean() for _, g in df.groupby("crop")]) * 100)


def boot(df, fn, seed=0):
    """fn(sample) -> value; resamples rows within crop."""
    rng = np.random.default_rng(seed); groups = [g for _, g in df.groupby("crop")]
    return np.array([fn(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(B)])


def fast_boot(df, cols, seed=0):
    """Resample images within crop once per replicate for all columns; returns {col: replicate equal-crop means (%)}."""
    rng = np.random.default_rng(seed); groups = [g[cols].to_numpy(float) for _, g in df.groupby("crop")]
    out = np.zeros((B, len(cols)))
    for b in range(B):
        out[b] = np.mean([g[rng.integers(0, len(g), len(g))].mean(0) for g in groups], 0) * 100
    return {c: out[:, i] for i, c in enumerate(cols)}


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def holm(ps):
    order = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def macro(df, col):
    return float(df.groupby("label")[col].mean().mean() * 100)


def summarize(W) -> dict:
    R = {"accuracy": {}, "contrasts": {}, "per_crop": {}, "hit_miss": {}, "selection": {}, "efficiency": {}}
    panels = {"hero": W[W.hero], "ablation": W[W.ablation], "efficiency": W[W.efficiency], "source": W[W.source]}
    methods = [c for c in ("sage", "majority", "wvote", "clip_s0", "dino_s0", "mace", "mace_s1", "mace_s2", "clip_s1", "dino_s1", "clip_s2", "dino_s2", "mace_K1", "mace_K16", "dino_K1", "dino_K16", "clip_K1", "clip_K16",
                                   "labels", "labels_all", "dino_labels", "masked", "shuffled", "random", "same_source", "other_source", "agentic", "imageref") if c in W]
    for pn, D in panels.items():
        R["accuracy"][pn] = {}
        for sp in ("iid", "ood"):
            d = D[D.split == sp]
            if not len(d):
                continue
            cols = [c for c in methods if d[c].notna().all()]
            bs = fast_boot(d, cols)
            R["accuracy"][pn][sp] = {c: {"acc": eq_mean(d, c), "ci": ci(bs[c]), "macro": macro(d, c), "n": int(len(d))} for c in cols}
    pv, keys = [], []
    for name, a, b, pn in (("mace_minus_sage", "mace", "sage", "hero"), ("mace_minus_dino", "mace", "dino_s0", "hero"), ("labels_minus_mace", "labels", "mace", "ablation")):
        if a not in W or b not in W:
            continue
        for sp in ("iid", "ood"):
            d = panels[pn][panels[pn].split == sp]
            bs = fast_boot(d, [a, b]); diff = bs[a] - bs[b]
            p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / B)
            R["contrasts"][f"{name}_{sp}"] = {"est": eq_mean(d, a) - eq_mean(d, b), "ci": ci(diff), "p_boot": float(p), "n": int(len(d))}
            pv.append(p); keys.append(f"{name}_{sp}")
    for k, h in zip(keys, holm(pv) if pv else []):
        R["contrasts"][k]["p_holm"] = float(h)
    for extra, a, b, pn in (("masked_minus_labels", "masked", "labels", "ablation"), ("mace_minus_masked", "mace", "masked", "ablation"), ("mace_minus_shuffled", "mace", "shuffled", "ablation"),
                            ("mace_minus_random", "mace", "random", "ablation"), ("same_minus_other", "same_source", "other_source", "source"), ("imageref_minus_mace", "imageref", "mace", "efficiency"),
                            ("agentic_minus_mace", "agentic", "mace", "efficiency")):
        if a in W and b in W:
            for sp in ("iid", "ood"):
                d = panels[pn][panels[pn].split == sp]
                if len(d):
                    bs = fast_boot(d, [a, b]); R["contrasts"][f"{extra}_{sp}"] = {"est": eq_mean(d, a) - eq_mean(d, b), "ci": ci(bs[a] - bs[b]), "n": int(len(d)), "secondary": True}
    amendment(W, panels["hero"], R)
    if "mace" in W:
        H = panels["hero"]
        for crop, g in H.groupby("crop", sort=False):
            R["per_crop"][crop] = {sp: {"n": int((g.split == sp).sum()), **{c: float(g[g.split == sp][c].mean() * 100) for c in ("sage", "dino_s0", "clip_s0", "mace", "hit")}} for sp in ("iid", "ood")}
        for sp in ("iid", "ood"):
            d = H[H.split == sp]
            R["hit_miss"][sp] = {"hit_rate": eq_mean(d, "hit"), "n": int(len(d))}
            for nm, sub in (("hit", d[d.hit == 1]), ("miss", d[d.hit == 0])):   # pooled over queries: not every crop has both
                R["hit_miss"][sp][nm] = {"n": int(len(sub)), "mace": float(sub.mace.mean() * 100), "sage": float(sub.sage.mean() * 100)}
    if "mace_s1" in W:
        H = panels["hero"]
        for sp in ("iid", "ood"):
            d = H[H.split == sp]
            R["selection"][sp] = {}
            for nm, cols in (("mace", ["mace", "mace_s1", "mace_s2"]), ("clip", ["clip_s0", "clip_s1", "clip_s2"]), ("dino", ["dino_s0", "dino_s1", "dino_s2"])):
                v = [eq_mean(d, c) for c in cols]; R["selection"][sp][nm] = {"values": v, "sd": float(np.std(v, ddof=1))}
            ds = d.assign(mean3=d[["mace", "mace_s1", "mace_s2"]].mean(1), dmean3=d[["dino_s0", "dino_s1", "dino_s2"]].mean(1), sage0=d.sage)
            bs = fast_boot(ds, ["mean3", "dmean3", "sage0"])
            R["selection"][sp]["mean_minus_sage"] = {"est": eq_mean(ds, "mean3") - eq_mean(ds, "sage0"), "ci": ci(bs["mean3"] - bs["sage0"])}
            R["selection"][sp]["mean_minus_dino"] = {"est": eq_mean(ds, "mean3") - eq_mean(ds, "dmean3"), "ci": ci(bs["mean3"] - bs["dmean3"])}
    R["efficiency"] = efficiency(W)
    return R


def amendment(W, H, R):
    """Family A of AMENDMENT_2026-11-05.md: post hoc, Holm across its four contrasts, separate from the locked family."""
    if not {"labels_all", "dino_labels"} <= set(W) or H.labels_all.isna().any() or H.dino_labels.isna().any():
        return
    pv, keys = [], []; R["amendment"] = {}
    for name, a, b in (("labels_all_minus_mace", "labels_all", "mace"), ("dino_labels_minus_labels_all", "dino_labels", "labels_all"), ("dino_labels_minus_dino", "dino_labels", "dino_s0")):
        for sp in ("iid", "ood"):
            d = H[H.split == sp]; bs = fast_boot(d, [a, b]); diff = bs[a] - bs[b]
            p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / B)
            R["amendment"][f"{name}_{sp}"] = {"est": eq_mean(d, a) - eq_mean(d, b), "ci": ci(diff), "p_boot": float(p), "n": int(len(d))}
            if name != "dino_labels_minus_dino":
                pv.append(p); keys.append(f"{name}_{sp}")
    for k, h in zip(keys, holm(pv)):
        R["amendment"][k]["p_holm"] = float(h)
    R["amendment"]["hit_rates"] = {sp: {"bm25": eq_mean(H[H.split == sp], "hit"), "dino": eq_mean(H[H.split == sp], "hit_dino")} for sp in ("iid", "ood")}


def efficiency(W) -> dict:
    """Five-crop matched panel. USD and latency from the ledger's own new calls; every method pays for the test caption, which
    ranks its KB claims. Completion = parseable answer. Common-completed accuracy uses the items every method completed."""
    led = [json.loads(l) for l in (RUNS / "cost_ledger.jsonl").open()]
    led = [r for r in led if r.get("t", 0) > T0 and not r["cache_hit"] and not r.get("failed")]
    stat = lambda st: (float(np.mean([r["cost_usd"] for r in led if r["stage"] == st])) if any(r["stage"] == st for r in led) else None,
                       float(np.median([r["elapsed_s"] for r in led if r["stage"] == st and r.get("elapsed_s")])) if any(r["stage"] == st and r.get("elapsed_s") for r in led) else None)
    cap_usd, cap_s = stat("caption")
    E = W[W.efficiency]; ids = set(E.image_id); out = {"n": int(len(E)), "methods": {}}
    src = {"sage": ("P1", "sage"), "labels": ("P4labels", "labels"), "mace": ("P1", "mace"), "agentic": ("P6", "agentic"), "imageref": ("P7", "imageref")}
    done = {}
    for m, (pkg, cond) in src.items():
        D = load(pkg)
        if not len(D) or m not in E:
            continue
        D = D[(D.condition == cond) & D.image_id.isin(ids)]
        done[m] = set(D[D.prediction != "UNKNOWN"].image_id)
        u, s = stat(f"diag:{cond}")
        d = {"acc_all": eq_mean(E, m), "completion": float(100 * (D.prediction != "UNKNOWN").mean()), "usd": (u or 0) + (cap_usd or 0), "latency_s": (s or 0) + (cap_s or 0)}
        if m == "agentic":
            d |= {"reads_mean": float(D.caption_reads.mean()), "turn_bound_hits": int(D.hit_turn_bound.sum()), "over_budget": int(D.over_budget.sum()), "non_caption_reads": int(D.non_caption_reads.map(len).sum())}
        out["methods"][m] = d
    if done:
        common = set.intersection(*done.values()); C = E[E.image_id.isin(common)]
        out["common_completed_n"] = len(common)
        for m in done:
            out["methods"][m]["acc_common"] = eq_mean(C, m)
    return out


def main():
    W = wide(); W.to_csv(OUT / "results_wide.csv", index=False)
    R = summarize(W); (OUT / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R["contrasts"], indent=0)); print(json.dumps(R["efficiency"], indent=0))


if __name__ == "__main__":
    main()
