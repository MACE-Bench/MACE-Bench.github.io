"""Sonnet and Haiku side by side on saved rows; no model call. Exploratory, chosen after the extension results were seen.

  python -m experiments.mace_extension.scripts.haiku_all    # writes statistics/haiku_all.json

Haiku ran SAGE, MACE and labels only on the 1,600 hero queries. Everything here derives from those rows, so conditions
Haiku did not run (controls, agent, image references, K, selections, DINOv2 retrieval, other-collection memory) are absent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .analyze import CFG, PR, ST, contrast, eq, family, rows, table
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def side(d, m, s):
    return {"acc": eq(d, m), "sage": eq(d, s)}


def main():
    H = table(); S = {}
    P = {"sonnet": ("mace", "sage", "labels_all"), "haiku": ("h_mace", "h_sage", "h_labels")}
    # n_true from the Haiku MACE rows; memory is identical to Sonnet's
    E = pd.DataFrame(map(json.loads, (ROOT / "raw_results/E1_haiku/predictions.jsonl").open()))
    M = E[E.condition == "mace"].set_index("image_id")
    H = H.copy(); H["n_true"] = [sum(l == lab for l in M.loc[i, "memory_labels_shown"]) for i, lab in zip(H.image_id, H.label)]
    S["price_hero"] = {c: float(E[E.condition == c].cost_usd.mean()) for c in E.condition.unique()}
    S["cached_rows"] = int(E.cache_hit.sum())
    for sp in ("iid", "ood"):
        d = H[H.split == sp]; o = S.setdefault(sp, {})
        o["vs_dino"] = family([contrast(d, [m], ["dino_s0"], f"{k}_mace_minus_dino") for k, (m, _, _) in P.items()])
        o["labels_share"] = {k: (eq(d, l) - eq(d, s)) / (eq(d, m) - eq(d, s)) for k, (m, s, l) in P.items()}
        g = d.groupby("crop")[["mace", "sage", "h_mace", "h_sage", "dino_s0", "h_labels"]].mean().mul(100)
        o["per_crop"] = g.round(1).to_dict("index")
        o["crops_vs_dino"] = {k: {"up": int((g[m] > g.dino_s0).sum()), "tied": int((g[m] == g.dino_s0).sum()), "down": int((g[m] < g.dino_s0).sum())} for k, (m, _, _) in P.items()}
        o["crops_vs_sage"] = {k: {"up": int((g[m] > g[s]).sum()), "tied": int((g[m] == g[s]).sum()), "down": int((g[m] < g[s]).sum())} for k, (m, s, _) in P.items() if k in ("sonnet", "haiku") for s in [s]}
        o["hit_rate"] = eq(d, "hit")
        for name, sub in (("hit", d[d.hit == 1]), ("miss", d[d.hit == 0])):
            o[name] = {"n": int(len(sub)), **{k: {"mace": float(sub[m].mean() * 100), "sage": float(sub[s].mean() * 100), "mace_eq": eq(sub, m), "sage_eq": eq(sub, s),
                                                  "contrast": contrast(sub, [m], [s], f"{k}_{name}")} for k, (m, s, _) in P.items()}}
        o["by_n_true"] = {int(n): {"n": int(len(g2)), "mace": float(g2.mace.mean() * 100), "sage": float(g2.sage.mean() * 100), "h_mace": float(g2.h_mace.mean() * 100), "h_sage": float(g2.h_sage.mean() * 100)} for n, g2 in d.groupby("n_true")}
        c = d.groupby("crop")[["mace", "sage", "h_mace", "h_sage", "hit"]].mean().mul(100)
    g = H.groupby(["crop", "split"])[["mace", "sage", "h_mace", "h_sage", "hit"]].mean().mul(100)
    S["pearson_gain_hit"] = {"sonnet": float(np.corrcoef(g.hit, g.mace - g.sage)[0, 1]), "haiku": float(np.corrcoef(g.hit, g.h_mace - g.h_sage)[0, 1])}
    # ablation 400 and efficiency 200 subsets
    for name, col in (("ablation", "ablation"), ("efficiency", "efficiency")):
        o = S.setdefault(name, {})
        for sp in ("iid", "ood", "all"):
            d = H[H[col]] if sp == "all" else H[H[col] & (H.split == sp)]
            o[sp] = {"n": int(len(d)), "raw": {c: float(d[c].mean() * 100) for c in ("sage", "mace", "labels", "h_sage", "h_mace", "h_labels", "majority")},
                     "labels_minus_mace": {k: contrast(d, [l], [m], k) for k, (m, l) in {"haiku": ("h_mace", "h_labels"), "sonnet": ("mace", "labels")}.items()},
                     "labels_minus_majority": {k: float((d[l].mean() - d.majority.mean()) * 100) for k, l in {"haiku": "h_labels", "sonnet": "labels"}.items()}}
    ids = set(H[H.efficiency].image_id); Ee = E[E.image_id.isin(ids)]
    S["efficiency"]["price"] = {c: float(Ee[Ee.condition == c].cost_usd.mean()) for c in Ee.condition.unique()}
    S["efficiency"]["usd_per_correct"] = {c: float(Ee[Ee.condition == c].cost_usd.mean() / Ee[Ee.condition == c].correct.astype(float).mean()) for c in Ee.condition.unique()}
    led = [json.loads(l) for l in open(ROOT.parents[1] / "runs/cost_ledger.jsonl")]
    S["efficiency"]["diag_latency_median"] = {c: float(np.median([r["elapsed_s"] for r in led if r.get("tag", "").startswith(f"E1_haiku|{c}|") and r["tag"].split("|")[-1] in ids and not r.get("cache_hit")])) for c in ("sage", "mace", "labels")}
    # frozen fallback rule on Haiku rows
    F = pd.read_csv(PR / "hero_fallback.csv").set_index("image_id"); T = json.load(open(ROOT / "configurations/fallback_rule.json"))["rule"]["thresholds"]
    H["use_mace"] = [F.loc[i, "fallback"] == 0 for i in H.image_id]
    H["h_fallback"] = np.where(H.use_mace, H.h_mace, H.h_sage); H["s_fallback_check"] = np.where(H.use_mace, H.mace, H.sage)
    H["h_oracle"] = np.where(H.hit == 1, H.h_mace, H.h_sage)
    S["fallback"] = {"sonnet_replay_matches_saved": bool((H.set_index("image_id").s_fallback_check.reindex(F.index).values == F.adaptive.values).all())}
    for sp in ("iid", "ood"):
        d = H[H.split == sp]
        S["fallback"][sp] = {"rate": float(1 - d.use_mace.mean()), "h_fallback_minus_mace": contrast(d, ["h_fallback"], ["h_mace"], "h_fb"), "h_oracle_minus_mace": contrast(d, ["h_oracle"], ["h_mace"], "h_or")}
    (ST / "haiku_all.json").write_text(json.dumps(S, indent=1, default=float))
    H[["image_id", "crop", "split", "n_true"]].to_csv(PR / "hero_n_true.csv", index=False)
    print(json.dumps({k: v for k, v in S.items() if k not in ("iid", "ood")}, indent=0, default=float)[:3500])
    for sp in ("iid", "ood"):
        o = S[sp]; print(sp, json.dumps({k: o[k] for k in ("vs_dino", "labels_share", "crops_vs_dino", "crops_vs_sage", "hit_rate", "by_n_true")}, default=float))
        for n in ("hit", "miss"):
            print(n, o[n]["n"], {k: (round(o[n][k]["mace_eq"], 1), round(o[n][k]["sage_eq"], 1), round(o[n][k]["mace"], 1), round(o[n][k]["sage"], 1), round(o[n][k]["contrast"]["est"], 1), [round(x, 1) for x in o[n][k]["contrast"]["ci"]]) for k in P})
        print(pd.DataFrame(o["per_crop"]).T.assign(gs=lambda x: x.mace - x.sage, gh=lambda x: x.h_mace - x.h_sage).round(1).to_string())


if __name__ == "__main__":
    main()
