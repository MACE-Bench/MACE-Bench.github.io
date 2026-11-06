"""Post hoc robustness checks on saved rows; no model call. Every result here is EXPLORATORY: these checks were
chosen after the extension results were seen, so nothing here enters a confirmatory family.

  python -m experiments.mace_extension.scripts.robustness   # writes statistics/robustness_posthoc.json
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from agmem import data, packet
from mace10 import jobs as JB
from . import jobs as X
from .analyze import CFG, PR, ST, TB, bm25_raw, cap, contrast, dino, eq, family, rows, table


def subgroups(H):
    o = H[H.split == "ood"]; coll = o.groupby("crop").collection.first()
    bug = list(coll[coll == "Bugwood"].index); rest = list(coll[coll != "Bugwood"].index)
    G = {"all_10": list(coll.index), "bugwood_7": bug, "non_bugwood_3": rest, "excl_mango_rice_8": [c for c in coll.index if c not in ("Mango", "Rice")]}
    C = {"sonnet_mace_minus_sage": (["mace"], ["sage"]), "sonnet_mace_minus_labels": (["mace"], ["labels_all"]), "haiku_mace_minus_sage": (["h_mace"], ["h_sage"]), "haiku_mace_minus_labels": (["h_mace"], ["h_labels"])}
    out = {"groups": G, "ood": {}, "iid_same_crops": {}, "leave_one_crop_out": {}}
    for g, cs in G.items():
        for key, sp in (("ood", "ood"), ("iid_same_crops", "iid")):
            d = H[(H.split == sp) & H.crop.isin(cs)]
            out[key][g] = {n: contrast(d, a, b, n) for n, (a, b) in C.items()}
    for c in coll.index:
        d = o[o.crop != c]
        out["leave_one_crop_out"][c] = {n: eq(d, a[0]) - eq(d, b[0]) for n, (a, b) in C.items()}
    return out


def nearest(H):
    """Nearest standard-bank image for each query, by crop and split. Near-copy screen, not proof of shared origin."""
    z = np.load(data.C.DATA / "audit/clip_vitb32_embeddings.npz"); Ec = z["E"].astype(np.float32); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    ic = {k: i for i, k in enumerate(z["ids"])}; Ed, idd = dino(); I = data.images(); Q = JB.queries().set_index("image_id"); r = []
    for q in H.itertuples():
        ids = X.bank_ids(q.crop, int(Q.loc[q.image_id, "run_s0"])); lab = np.array([I.loc[i, "class"] for i in ids]); same = lab == q.label
        sc = Ec[[ic[i] for i in ids]] @ Ec[ic[q.image_id]]; sd = Ed[[idd[i] for i in ids]] @ Ed[idd[q.image_id]]
        r.append({"crop": q.crop, "split": q.split, "clip": float(sc.max()), "clip_same": float(sc[same].max()) if same.any() else np.nan, "dino": float(sd.max()), "nn_coll": I.loc[ids[int(np.argmax(sd))], "collection"]})
    D = pd.DataFrame(r); out = {}
    for (c, s), g in D.groupby(["crop", "split"]):
        out.setdefault(c, {})[s] = {"clip_mean": float(g["clip"].mean()), "clip_same_ge_090_pct": float((g.clip_same >= .90).mean() * 100), "clip_same_ge_095_pct": float((g.clip_same >= .95).mean() * 100), "dino_mean": float(g.dino.mean()), "nn_collections": g.nn_coll.value_counts().head(3).to_dict()}
    bank = {}
    for c in H.crop.unique():
        run = int(Q[Q.crop == c].run_s0.iloc[0]) if "crop" in Q else 0
        bank[c] = I.loc[list(X.bank_ids(c, run)), "collection"].value_counts().to_dict()
    return {"by_crop": out, "standard_bank_collections": bank}


def family_cov(H):
    I = data.images(); O = rows("E4other").set_index("image_id"); F = rows("E4family").set_index("image_id")
    d = pd.read_csv(PR / "hero_iid_cross_collection.csv").set_index("image_id")
    d["fam_covered"] = [float(d.loc[q, "label"] in {I.loc[m, "class"] for m in (F.loc[q, "memory_bank_ids"] if q in F.index else O.loc[q, "memory_bank_ids"])}) for q in d.index]
    d = d.reset_index(); kept = d[(d.covered == 1) & (d.fam_covered == 1)]; lost = d[(d.covered == 1) & (d.fam_covered == 0)]; ch = d[d.image_id.isin(F.index)]
    return {"covered_other": int(d.covered.sum()), "covered_family": int(d.fam_covered.sum()), "lost_coverage": int(len(lost)), "changed_rows": int(len(ch)), "changed_by_crop": ch.crop.value_counts().to_dict(),
            "family_minus_other_coverage_kept": contrast(kept, ["family"], ["other_full"], "family_minus_other_kept"), "lost_rows_acc": {"other": float(lost.other_full.mean() * 100), "family": float(lost.family.mean() * 100)} if len(lost) else None,
            "changed_kept_raw": {"n": int(len(ch[ch.fam_covered == 1])), "other": float(ch[ch.fam_covered == 1].other_full.mean() * 100), "family": float(ch[ch.fam_covered == 1].family.mean() * 100)}}


def size_control(H, reps=20):
    """Offline: does shrinking the standard bank to the other-collection size alone lower the retrieval hit rate?"""
    O = rows("E4other").set_index("image_id"); Q = JB.queries().set_index("image_id"); rng = random.Random(0); r = []
    for q in H[H.split == "iid"].itertuples():
        b = JB.mem(q.crop, 0, int(Q.loc[q.image_id, "run_s0"]), 4); lab = {m["image_id"]: m["label"] for m in b}; n = len(O.loc[q.image_id, "memory_bank_ids"])
        qc = cap(q.image_id, q.crop); tq = packet.tok(packet.caption_text(qc)) + packet.tok(str(qc.get("organ", ""))) * 3
        docs = {i: packet.tok(packet.caption_text(cap(i, q.crop))) for i in lab}; h = c = 0
        for _ in range(reps):
            ks = rng.sample(sorted(lab), min(n, len(lab))); s = bm25_raw(tq, {k: docs[k] for k in ks}); top = sorted(s, key=lambda k: (-s[k], k))[:4]
            h += q.label in {lab[k] for k in top}; c += q.label in {lab[k] for k in ks}
        r.append({"crop": q.crop, "std_n": len(lab), "other_n": n, "hit_sub": h / reps, "cov_sub": c / reps, "hit_full": q.hit})
    D = pd.DataFrame(r); e = lambda x: float(D.groupby("crop")[x].mean().mean())
    other = pd.read_csv(PR / "hero_iid_cross_collection.csv")
    cv = other[other.covered == 1]; ce = lambda k: float(cv.groupby("crop")[k].mean().mean() * 100)
    return {"hit_standard_on_covered": ce("hit"), "hit_other_on_covered": ce("other_hit"), "std_bank_mean": float(D.std_n.mean()), "other_bank_mean": float(D.other_n.mean()), "hit_full": e("hit_full") * 100, "hit_size_matched_random": e("hit_sub") * 100, "coverage_size_matched_random": e("cov_sub") * 100,
            "hit_other_actual": float(other.groupby("crop").other_hit.mean().mean() * 100), "coverage_other_actual": float(other.groupby("crop").covered.mean().mean() * 100)}


def calibration(H):
    D = rows("E3cal"); cal = set(D.image_id); hero = set(H.image_id); Q = JB.queries().set_index("image_id"); bank = set(); out = {}
    for c in H.crop.unique():
        for run in sorted(set(int(x) for x in Q[Q.index.isin(H[H.crop == c].image_id)].run_s0)):
            bank |= set(X.bank_ids(c, run))
    m = D.drop_duplicates("image_id")
    out = {"n": len(cal), "overlap_hero": len(cal & hero), "in_a_hero_run_bank": len(cal & bank), "in_own_run_bank": int(sum(r.image_id in set(X.bank_ids(r.crop, int(r.run))) for r in m.itertuples())),
           "self_retrieved": int(sum(("m_" + r.image_id) in (r.memory_ids or []) for r in D[D.condition != "sage"].itertuples()))}
    rule = json.loads((CFG / "fallback_rule.json").read_text())["rule"]; th = rule["thresholds"]; C = pd.read_csv(PR / "calibration_signals.csv")
    C["fb"] = C[rule["signal"]] < C.crop.map(th)
    out["fallback_by_cal_bank_pct"] = C.groupby("bank").fb.mean().mul(100).round(1).to_dict()
    out["cal_mem_minus_sage_on_fallback"] = {b: float((g[g.fb].mem - g[g.fb].sage).mean() * 100) for b, g in C.groupby("bank")}
    G = pd.read_csv(TB / "fallback_calibration_grid.csv"); out["candidates"] = int(len(G)); out["candidates_beating_always_mace"] = int((G.cal_score > 50.8333334).sum())
    return out


def crop_level(H):
    """Treat each crop as one observation: t interval over the ten per-crop gains, and an exact sign-flip test."""
    from itertools import product
    from scipy import stats
    C = {"sonnet_mace_minus_sage": ("mace", "sage"), "sonnet_mace_minus_labels": ("mace", "labels_all"), "haiku_mace_minus_sage": ("h_mace", "h_sage"), "haiku_mace_minus_labels": ("h_mace", "h_labels")}; out = {}
    for sp in ("iid", "ood"):
        for n, (a, b) in C.items():
            g = (H[H.split == sp].groupby("crop")[a].mean() - H[H.split == sp].groupby("crop")[b].mean()).values * 100
            m, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g)); t = stats.t.ppf(.975, len(g) - 1)
            flips = [abs(np.mean(g * np.array(s))) for s in product((1, -1), repeat=len(g))]
            out[f"{n}_{sp}"] = {"est": float(m), "ci": [float(m - t * se), float(m + t * se)], "p_signflip": float(np.mean(np.array(flips) >= abs(m) - 1e-9))}
    return out


def matched():
    """Family F5 of the protocol: size-matched control for E4. The only section of this file with a fixed Holm family."""
    M = rows("E4matched").drop_duplicates("image_id").set_index("image_id"); d = pd.read_csv(PR / "hero_iid_cross_collection.csv").set_index("image_id")
    I = data.images(); d["matched"] = M.correct.astype(float)
    d["m_hit"] = [float(d.loc[q, "label"] in (M.loc[q, "memory_labels_shown"] or [])) for q in d.index]
    d["m_top_same"] = [float(np.mean([I.loc[x[2:], "collection"] == d.loc[q, "collection"] for x in M.loc[q, "memory_ids"]])) if M.loc[q, "memory_ids"] else 0.0 for q in d.index]
    d = d.reset_index(); c = d[d.covered == 1]; u = d[d.covered == 0]
    F = family([contrast(c, ["matched"], ["other_full"], "matched_minus_other_covered"), contrast(d, ["matched"], ["other_full"], "matched_minus_other_all"), contrast(c, ["mace"], ["matched"], "standard_minus_matched_covered")])
    pc = {k: {"matched": float(g.matched.mean() * 100), "other": float(g.other_full.mean() * 100), "standard": float(g.mace.mean() * 100), "n": int(len(g))} for k, g in c.groupby("crop")}
    gain = (c.groupby("crop").matched.mean() - c.groupby("crop").other_full.mean()).values * 100; m, se = gain.mean(), gain.std(ddof=1) / np.sqrt(len(gain))
    from scipy import stats
    t = stats.t.ppf(.975, len(gain) - 1)
    return {"rows": int(d.matched.notna().sum()), "family": F, "acc": {n: {k: eq(g, k) for k in ("sage", "mace", "matched", "other_full")} for n, g in (("all", d), ("covered", c), ("not_covered", u))}, "hit": {n: {k: eq(g, k) for k in ("m_hit", "other_hit", "hit")} for n, g in (("all", d), ("covered", c))},
            "matched_minus_sage_covered": contrast(c, ["matched"], ["sage"], "matched_minus_sage_covered"), "top4_same_collection_pct": float(d.m_top_same.mean() * 100), "per_crop_covered": pc, "crop_level_matched_minus_other_covered": {"est": float(m), "ci": [float(m - t * se), float(m + t * se)]}, "n_unchanged_top4_note": "rows with cache_hit true reused an identical prompt", "cache_hits": int(rows("E4matched").cache_hit.sum())}


def main():
    H = table()
    S = {"matched": matched(), "crop_level": crop_level(H), "subgroups": subgroups(H), "nearest": nearest(H), "family": family_cov(H), "size": size_control(H), "calibration": calibration(H)}
    (ST / "robustness_posthoc.json").write_text(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
