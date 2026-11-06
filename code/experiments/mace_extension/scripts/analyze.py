"""All extension statistics from raw rows; no model call. Protocol: EXPERIMENT_PROTOCOL.md.

  python -m experiments.mace_extension.scripts.analyze select     # calibration only: writes configurations/fallback_rule.json
  python -m experiments.mace_extension.scripts.analyze all        # needs the frozen rule; writes statistics/ and tables/
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from agmem import data, llm, packet
from mace10 import jobs as JB, results as R
from . import jobs as X

ROOT = Path(__file__).resolve().parent.parent
RAW, ST, TB, CFG, PR = ROOT / "raw_results", ROOT / "statistics", ROOT / "tables", ROOT / "configurations", ROOT / "processed_results"
SIGS_Q = ["bm25_top1", "bm25_margin", "bm25_conc", "dino_top1", "dino_margin"]
SIGS_G = {"bm25_agree": (0.5, 0.75, 1.0), "dino_agree": (0.5, 0.75, 1.0), "cross_agree": (0.25, 0.5, 0.75, 1.0)}
QS = [round(0.05 * i, 2) for i in range(1, 11)]
FOCUS = ["Apple", "Grape", "Potato", "Wheat"]


def rows(name):
    p = RAW / name / "predictions.jsonl"
    return pd.DataFrame(map(json.loads, p.open())) if p.exists() else pd.DataFrame()


def caption(i, crop):
    p = llm.C.CACHE / "llm" / f"{JB.caption_key(i, crop)}.json"
    r = json.loads(p.read_text()); c = llm.parse_json(r["text"])
    return c if isinstance(c, dict) else {}


_caps = {}
def cap(i, crop):
    if i not in _caps:
        _caps[i] = caption(i, crop)
    return _caps[i]


def dino():
    a = np.load(JB.OUT / "dinov2_embeddings.npz"); b = np.load(PR / "dinov2_extra.npz")
    E = np.concatenate([a["E"], b["E"]]).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E, {k: i for i, k in enumerate(list(a["ids"]) + list(b["ids"]))}


def bm25_raw(query, docs):
    """packet.bm25 before its division by the maximum."""
    N = len(docs) or 1
    df = Counter(t for d in docs.values() for t in set(d)); avg = sum(len(d) for d in docs.values()) / N
    q = Counter(query); out = {}
    for k, d in docs.items():
        tf = Counter(d); s = 0.0
        for t in q:
            if t in tf:
                s += math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        out[k] = s
    return out


def modal_share(labs):
    return max(Counter(labs).values()) / len(labs) if labs else 0.0


def signals(qid, crop, bank, E, idx):
    """bank: list of {image_id, label}. Uses no query label."""
    qc = cap(qid, crop); q = packet.tok(packet.caption_text(qc)) + packet.tok(str(qc.get("organ", ""))) * 3
    lab = {m["image_id"]: m["label"] for m in bank}
    raw = bm25_raw(q, {i: packet.tok(packet.caption_text(cap(i, crop))) for i in lab})
    top = sorted(raw, key=lambda k: (-raw[k], k))[:4]; v = [raw[k] for k in top] + [0, 0]
    ids = sorted(lab); s = E[[idx[i] for i in ids]] @ E[idx[qid]]
    o = sorted(range(len(ids)), key=lambda n: (-s[n], ids[n]))[:4]; dtop = [ids[n] for n in o]; c = [float(s[n]) for n in o] + [0, 0]
    bl, dl = [lab[i] for i in top], [lab[i] for i in dtop]
    return {"bm25_top1": v[0], "bm25_margin": v[0] - v[1], "bm25_conc": v[0] / (sum(v[:4]) or 1), "bm25_agree": modal_share(bl), "dino_top1": c[0], "dino_margin": c[0] - c[1],
            "dino_agree": modal_share(dl), "cross_agree": sum(x in set(dl) for x in bl) / max(1, len(bl)), "bm25_ids": top, "bm25_labels": bl, "dino_ids": dtop, "dino_labels": dl}


def eq(df, col):
    return float(df.groupby("crop")[col].mean().mean() * 100)


def contrast(d, plus, minus=(), name=""):
    """Equal-crop linear contrast with a paired bootstrap. plus/minus: column names."""
    plus, minus = list(plus), list(minus); cols = list(dict.fromkeys(plus + minus))
    bs = R.fast_boot(d, cols); v = sum(bs[c] for c in plus) - sum((bs[c] for c in minus), np.zeros(R.B))
    est = sum(eq(d, c) for c in plus) - sum(eq(d, c) for c in minus)
    p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / R.B)
    return {"name": name, "est": est, "ci": R.ci(v), "p_boot": float(p), "n": int(len(d))}


def family(items):
    for x, h in zip(items, R.holm([i["p_boot"] for i in items])):
        x["p_holm"] = float(h)
    return items


def reading(c):
    lo, hi = c["ci"]
    return "positive" if lo > 0 and hi > 3 else "negative" if hi < 0 and lo < -3 else "within 3 points" if lo >= -3 and hi <= 3 else "inconclusive"


def select():
    """Choose one fallback rule on calibration rows only."""
    D = rows("E3cal"); assert len(D) == 1800, len(D)
    E, idx = dino(); recs = []
    W = D.pivot_table(index="image_id", columns="condition", values="correct", aggfunc="first").astype(float)
    meta = D.drop_duplicates("image_id").set_index("image_id")
    for i, r in meta.iterrows():
        for bankname, variant in (("mixed", "mixed"), ("other", "other_source")):
            b = JB.mem(r.crop, 0, int(r.run), 4, variant, r.collection if variant != "mixed" else None)
            s = signals(i, r.crop, b, E, idx)
            recs.append({"image_id": i, "crop": r.crop, "bank": bankname, "sage": W.loc[i, "sage"], "mem": W.loc[i, "mace" if bankname == "mixed" else "mace_other"], **{k: v for k, v in s.items() if not isinstance(v, list)}})
    C = pd.DataFrame(recs); C.to_csv(PR / "calibration_signals.csv", index=False)

    def score(use):
        x = C.assign(a=np.where(use, C.mem, C.sage))
        return float(np.mean([x[x.bank == b].groupby("crop").a.mean().mean() * 100 for b in ("mixed", "other")]))
    grid = []
    for s in SIGS_Q:
        for q in QS:
            t = C.groupby("crop")[s].quantile(q)
            use = C[s] >= C.crop.map(t)
            grid.append({"signal": s, "kind": "crop_quantile", "value": q, "cal_score": score(use), "cal_fallback": float(1 - use.mean()), "thresholds": t.to_dict()})
    for s, ts in SIGS_G.items():
        for t in ts:
            use = C[s] >= t
            grid.append({"signal": s, "kind": "global", "value": t, "cal_score": score(use), "cal_fallback": float(1 - use.mean()), "thresholds": None})
    always = score(np.ones(len(C), bool)); never = score(np.zeros(len(C), bool))
    G = pd.DataFrame(grid).sort_values(["cal_score", "cal_fallback"], ascending=[False, True]); best = G.iloc[0].to_dict()
    G.drop(columns="thresholds").to_csv(TB / "fallback_calibration_grid.csv", index=False)
    rule = {"chosen_on": "calibration rows only", "always_mace_cal": always, "always_sage_cal": never, "rule": best, "beats_always_mace": bool(best["cal_score"] > always),
            "by_bank": {b: {"mem": eq(C[C.bank == b], "mem"), "sage": eq(C[C.bank == b], "sage")} for b in ("mixed", "other")}}
    (CFG / "fallback_rule.json").write_text(json.dumps(rule, indent=1, default=float))
    print(json.dumps({k: v for k, v in rule.items() if k != "rule"}, indent=1), {k: v for k, v in best.items() if k != "thresholds"})
    print(G.drop(columns="thresholds").head(12).round(3).to_string()); print(G.groupby("signal").cal_score.max().round(2).to_string())


def table():
    W = R.wide(); H = W[W.hero].copy().set_index("image_id")
    for name, col, cond in (("E2", "dino_mace", "dino_mace"), ("E1_haiku", "h_sage", "sage"), ("E1_haiku", "h_mace", "mace"), ("E1_haiku", "h_labels", "labels"), ("E4other", "other_full", "other_source"), ("E4family", "family", "family_disjoint")):
        D = rows(name)
        if len(D):
            H[col] = D[D.condition == cond].drop_duplicates("image_id").set_index("image_id").correct.astype(float)
    if "family" in H:
        H["family"] = H.family.fillna(H.other_full)
    return H.reset_index()


def flips(d, a, b):
    """a is the reference (SAGE), b the memory method. Rates in % of queries, equal-crop."""
    g = d.groupby("crop")
    return {"corrected": float(g.apply(lambda x: ((x[a] == 0) & (x[b] == 1)).mean()).mean() * 100), "introduced": float(g.apply(lambda x: ((x[a] == 1) & (x[b] == 0)).mean()).mean() * 100)}


def ledger():
    L = pd.DataFrame(map(json.loads, (R.RUNS / "cost_ledger.jsonl").open()))
    L = L[~L.cache_hit.astype(bool)]
    return L[L.failed.fillna(False) == False] if "failed" in L else L  # noqa: E712


def use_stats(L, prefix):
    g = L[L.tag.fillna("").str.startswith(prefix)]
    return {"calls": int(len(g)), "usd": float(g.cost_usd.sum()), "mean_usd": float(g.cost_usd.mean()) if len(g) else None, "median_s": float(g.elapsed_s.median()) if len(g) else None,
            "mean_in": float(g.input_tokens.mean()) if len(g) else None, "mean_out": float(g.output_tokens.mean()) if len(g) else None}


def e1(H, L):
    out = {"accuracy": {}, "per_crop": {}, "contrasts": [], "flips": {}, "usage": {}}
    M = {"Sonnet": ("sage", "mace", "labels_all"), "Haiku": ("h_sage", "h_mace", "h_labels")}
    for sp in ("iid", "ood"):
        d = H[H.split == sp]; out["accuracy"][sp] = {}
        for m, cols in M.items():
            bs = R.fast_boot(d, list(cols))
            out["accuracy"][sp][m] = {c: {"acc": eq(d, c), "ci": R.ci(bs[c])} for c in cols}
            out["flips"][f"{m}_{sp}"] = {"mace_vs_sage": flips(d, cols[0], cols[1]), "mace_vs_labels": flips(d, cols[2], cols[1])}
        out["per_crop"][sp] = {c: {k: float(g[k].mean() * 100) for k in M["Sonnet"] + M["Haiku"]} for c, g in d.groupby("crop")}
    F = []
    for sp in ("iid", "ood"):
        d = H[H.split == sp]
        F += [contrast(d, ["h_mace"], ["h_sage"], f"haiku_mace_minus_sage_{sp}"), contrast(d, ["h_mace"], ["h_labels"], f"haiku_mace_minus_labels_{sp}")]
    out["contrasts"] = family(F)
    out["sonnet_reference"] = [contrast(H[H.split == sp], ["mace"], [b], f"sonnet_mace_minus_{b}_{sp}") for sp in ("iid", "ood") for b in ("sage", "labels_all")]
    out["model_gap"] = [contrast(H[H.split == sp], ["mace", "h_sage"], ["sage", "h_mace"], f"sonnet_gain_minus_haiku_gain_{sp}") for sp in ("iid", "ood")]
    for c in ("sage", "mace", "labels"):
        out["usage"][c] = use_stats(L, f"E1_haiku|{c}|")
    out["crops_gain"] = {sp: {m: {"up": int(sum(v[M[m][1]] > v[M[m][0]] for v in out["per_crop"][sp].values())), "tied": int(sum(v[M[m][1]] == v[M[m][0]] for v in out["per_crop"][sp].values())),
                              "down": int(sum(v[M[m][1]] < v[M[m][0]] for v in out["per_crop"][sp].values()))} for m in M} for sp in ("iid", "ood")}
    out["unknown"] = {c: int((rows("E1_haiku").query("condition==@c").prediction == "UNKNOWN").sum()) for c in ("sage", "mace", "labels")}
    return out


def e2(H, L):
    out = {"cells": {}, "per_crop": {}}; cols = ["labels_all", "mace", "dino_labels", "dino_mace"]; F = []
    for sp in ("iid", "ood"):
        d = H[H.split == sp]; bs = R.fast_boot(d, cols)
        out["cells"][sp] = {c: {"acc": eq(d, c), "ci": R.ci(bs[c])} for c in cols}
        F += [contrast(d, ["mace"], ["labels_all"], f"caption_under_bm25_{sp}"), contrast(d, ["dino_mace"], ["dino_labels"], f"caption_under_dino_{sp}"),
              contrast(d, ["dino_mace"], ["mace"], f"dino_minus_bm25_with_captions_{sp}"), contrast(d, ["dino_mace", "labels_all"], ["dino_labels", "mace"], f"interaction_{sp}")]
        out["per_crop"][sp] = {c: {"cap_bm25": float((g.mace - g.labels_all).mean() * 100), "cap_dino": float((g.dino_mace - g.dino_labels).mean() * 100), "dino_minus_bm25_caps": float((g.dino_mace - g.mace).mean() * 100)} for c, g in d.groupby("crop")}
        out[f"ret_no_caps_{sp}"] = contrast(d, ["dino_labels"], ["labels_all"], f"dino_minus_bm25_labels_{sp}")
        out[f"main_caption_{sp}"] = contrast(d, ["mace", "dino_mace"], ["labels_all", "dino_labels"], f"main_caption_x2_{sp}")
        out[f"main_retrieval_{sp}"] = contrast(d, ["dino_mace", "dino_labels"], ["mace", "labels_all"], f"main_retrieval_x2_{sp}")
    out["contrasts"] = family(F)
    for c in F:
        c["reading"] = reading(c)
    out["crops_positive"] = {sp: {k: {"up": sum(v[k] > 0 for v in out["per_crop"][sp].values()), "down": sum(v[k] < 0 for v in out["per_crop"][sp].values())} for k in ("cap_bm25", "cap_dino")} for sp in ("iid", "ood")}
    out["usage"] = use_stats(L, "E2|")
    return out


def e3(H, L):
    rule = json.loads((CFG / "fallback_rule.json").read_text()); b = rule["rule"]; E, idx = dino()
    S = []
    Q = JB.queries().set_index("image_id")
    for r in H.itertuples():
        s = signals(r.image_id, r.crop, JB.mem(r.crop, 0, int(Q.loc[r.image_id, "run_s0"]), 4), E, idx)
        S.append({"image_id": r.image_id, **{k: v for k, v in s.items() if not isinstance(v, list)}, "bm25_ids": s["bm25_ids"]})
    S = pd.DataFrame(S).set_index("image_id"); P1 = R.load("P1"); M = P1[P1.condition == "mace"].set_index("image_id")
    check = float(np.mean([["m_" + i for i in S.loc[q, "bm25_ids"]] == M.loc[q, "memory_ids"] for q in S.index]))
    D = H.set_index("image_id").join(S.drop(columns="bm25_ids"))
    th = b["thresholds"]
    use = D[b["signal"]] >= (D.crop.map(th) if b["kind"] == "crop_quantile" else b["value"])
    D["adaptive"] = np.where(use, D.mace, D.sage); D["oracle"] = np.where(D.hit == 1, D.mace, D.sage); D["fallback"] = (~use).astype(float); D = D.reset_index()
    out = {"bm25_rank_reproduced": check, "rule": {k: v for k, v in b.items() if k != "thresholds"}, "acc": {}, "per_crop": {}, "avoid": {}}; F = []
    for sp in ("iid", "ood"):
        d = D[D.split == sp]; bs = R.fast_boot(d, ["sage", "mace", "adaptive", "oracle"])
        out["acc"][sp] = {c: {"acc": eq(d, c), "ci": R.ci(bs[c])} for c in ("sage", "mace", "adaptive", "oracle")}
        F.append(contrast(d, ["adaptive"], ["mace"], f"adaptive_minus_mace_{sp}"))
        fb = d[d.fallback == 1]
        out["avoid"][sp] = {"fallback_rate": float(d.fallback.mean() * 100), "n_fallback": int(len(fb)), "sage_right_mace_wrong_total": int(((d.sage == 1) & (d.mace == 0)).sum()), "avoided": int(((fb.sage == 1) & (fb.mace == 0)).sum()),
                            "introduced": int(((fb.sage == 0) & (fb.mace == 1)).sum()), "fallback_on_miss": float((fb.hit == 0).mean() * 100) if len(fb) else None, "miss_rate": float((d.hit == 0).mean() * 100)}
        for nm, g in (("hit", d[d.hit == 1]), ("miss", d[d.hit == 0])):
            out["avoid"][sp][nm] = {c: float(g[c].mean() * 100) for c in ("sage", "mace", "adaptive")} | {"n": int(len(g))}
        out["per_crop"][sp] = {c: {k: float(g[k].mean() * 100) for k in ("sage", "mace", "adaptive", "fallback")} for c, g in d.groupby("crop")}
    out["contrasts"] = family(F)
    # how well each signal separates retrieval misses and harmful memory, post hoc
    H2 = D.assign(harm=((D.sage == 1) & (D.mace == 0)).astype(float)); auc = {}
    for s in SIGS_Q + list(SIGS_G):
        auc[s] = {}
        for sp in ("iid", "ood"):
            g = H2[H2.split == sp]; r = g.groupby("crop")[s].rank(pct=True)
            for tgt, y in (("miss", g.hit == 0), ("harm", g.harm == 1)):
                a, c = r[y], r[~y]
                auc[s][f"{tgt}_{sp}"] = float((a.values[:, None] < c.values[None, :]).mean() + 0.5 * (a.values[:, None] == c.values[None, :]).mean()) if len(a) else None
    out["auc_low_signal_predicts"] = auc
    out["usage_per_call"] = {c: use_stats(L, f"P1|{c}|") for c in ("sage", "mace")}
    D[["image_id", "crop", "split", "fallback", "adaptive", "oracle", "sage", "mace", "hit"] + SIGS_Q + list(SIGS_G)].to_csv(PR / "hero_fallback.csv", index=False)
    return out


def e4(H, L):
    I = data.images(); E, idx = dino(); O = rows("E4other").set_index("image_id"); Fm = rows("E4family").set_index("image_id")
    iid = H[H.split == "iid"].copy().set_index("image_id")
    bank = {q: list(zip(O.loc[q, "memory_bank_ids"], [I.loc[m, "class"] for m in O.loc[q, "memory_bank_ids"]])) for q in iid.index}
    iid["covered"] = [float(iid.loc[q, "label"] in {l for _, l in bank[q]}) for q in iid.index]
    iid["other_hit"] = [float(iid.loc[q, "label"] in (O.loc[q, "memory_labels_shown"] or [])) for q in iid.index]
    def nn(q):
        ids = [i for i, _ in bank[q]]; s = E[[idx[i] for i in ids]] @ E[idx[q]]; return bank[q][int(np.argmax(s))][1]
    iid["dino_other"] = [float(nn(q) == iid.loc[q, "label"]) for q in iid.index]
    iid["bank_n_other"] = [len(bank[q]) for q in iid.index]
    iid["fam_n"] = [len(Fm.loc[q, "memory_bank_ids"]) if q in Fm.index else len(bank[q]) for q in iid.index]
    d = iid.reset_index(); cov = d[d.covered == 1]; unc = d[d.covered == 0]
    out = {"n": {"iid": len(d), "covered": len(cov), "not_covered": len(unc)}, "coverage_by_crop": d.groupby("crop").covered.mean().mul(100).round(1).to_dict(), "acc": {}}
    cols = ["sage", "mace", "other_full", "family", "dino_s0", "dino_other"]
    for nm, g in (("all", d), ("covered", cov), ("not_covered", unc)):
        cc = [c for c in cols]; bs = R.fast_boot(g, cc)
        out["acc"][nm] = {c: {"acc": eq(g, c), "ci": R.ci(bs[c])} for c in cc} | {"crops": int(g.crop.nunique()), "n": len(g)}
    F = [contrast(d, ["other_full"], ["mace"], "other_minus_standard_all"), contrast(cov, ["other_full"], ["mace"], "other_minus_standard_covered"), contrast(cov, ["other_full"], ["sage"], "other_minus_sage_covered")]
    out["contrasts"] = family(F)
    out["secondary"] = [contrast(d, ["family"], ["other_full"], "family_minus_other_all"), contrast(cov, ["family"], ["other_full"], "family_minus_other_covered"), contrast(unc, ["other_full"], ["sage"], "other_minus_sage_not_covered"), contrast(d, ["other_full"], ["sage"], "other_minus_sage_all")]
    # P5 reconstruction
    src = H[H.source & (H.split == "iid")]
    out["p5_reconstruction"] = contrast(src, ["same_source"], ["other_source"], "same_minus_other_199")
    # per crop source table
    T = []
    for crop, g in H.groupby("crop"):
        a, b = g[g.split == "iid"], g[g.split == "ood"]; x = d[d.crop == crop]; xc = x[x.covered == 1]
        I_c = I[I.crop == crop]; p = I_c[(I_c.role == "iid") & I_c["class"].isin(b.label.unique())]["class"].value_counts(normalize=True); q = b.label.value_counts(normalize=True)
        ks = sorted(set(p.index) | set(q.index)); pv, qv = p.reindex(ks).fillna(0).values, q.reindex(ks).fillna(0).values; m = (pv + qv) / 2
        kl = lambda u, v: float(np.sum(np.where(u > 0, u * np.log2(np.where(u > 0, u, 1) / np.where(v > 0, v, 1)), 0)))
        cosI = [float(np.max(E[[idx[i] for i in X.bank_ids(crop, int(r.run_s0))]] @ E[idx[r.image_id]])) for r in a.itertuples()]
        cosO = [float(np.max(E[[idx[i] for i in X.bank_ids(crop, int(r.run_s0))]] @ E[idx[r.image_id]])) for r in b.itertuples()]
        T.append({"crop": crop, "ood_collection": b.collection.iloc[0], "sage_iid": a.sage.mean() * 100, "mace_iid": a.mace.mean() * 100, "sage_ood": b.sage.mean() * 100, "mace_ood": b.mace.mean() * 100,
                  "gain_iid": (a.mace - a.sage).mean() * 100, "gain_ood": (b.mace - b.sage).mean() * 100, "hit_iid": a.hit.mean() * 100, "hit_ood": b.hit.mean() * 100,
                  "other_acc": x.other_full.mean() * 100, "other_hit": x.other_hit.mean() * 100, "covered": x.covered.mean() * 100, "std_cov_stratum": xc.mace.mean() * 100, "other_cov_stratum": xc.other_full.mean() * 100,
                  "js_bits": 0.5 * kl(pv, m) + 0.5 * kl(qv, m), "dino_nn_iid": float(np.mean(cosI)), "dino_nn_ood": float(np.mean(cosO))})
    T = pd.DataFrame(T); T["dino_gap"] = T.dino_nn_iid - T.dino_nn_ood; T["hit_drop"] = T.hit_iid - T.hit_ood; T["gain_drop"] = T.gain_iid - T.gain_ood
    T.round(2).to_csv(TB / "table_D_collection_shift.csv", index=False)
    sp = lambda a, b: float(T[[a, b]].corr(method="spearman").iloc[0, 1])
    out["crop_correlations_spearman_n10"] = {"dino_gap_vs_hit_drop": sp("dino_gap", "hit_drop"), "dino_gap_vs_gain_ood": sp("dino_gap", "gain_ood"), "js_vs_hit_drop": sp("js_bits", "hit_drop"), "js_vs_gain_ood": sp("js_bits", "gain_ood"), "hit_ood_vs_gain_ood": sp("hit_ood", "gain_ood")}
    # failure categories on held-out, from saved MACE rows
    P1 = R.load("P1"); Mc = P1[P1.condition == "mace"].set_index("image_id"); fc = []
    for r in H[H.split == "ood"].itertuples():
        m = Mc.loc[r.image_id]; labs = m.memory_labels_shown; pred = m.prediction
        cat = "correct" if r.mace == 1 else ("miss_followed_memory" if r.label not in labs and pred in labs else "miss_other" if r.label not in labs else "hit_overruled_to_memory" if pred in labs else "hit_overruled_elsewhere")
        fc.append({"crop": r.crop, "category": cat, "harm": r.sage == 1 and r.mace == 0, "rescued": r.sage == 0 and r.mace == 1})
    Fc = pd.DataFrame(fc); ct = (Fc.groupby(["crop", "category"]).size().unstack(fill_value=0)); ct["harm"] = Fc.groupby("crop").harm.sum(); ct["rescued"] = Fc.groupby("crop").rescued.sum()
    ct.to_csv(TB / "failure_categories_heldout.csv"); out["failure_categories"] = ct.to_dict(orient="index")
    # retrieved collection vs query collection, IID standard bank (association only)
    rr = []
    for r in H[H.split == "iid"].itertuples():
        ids = [x[2:] for x in Mc.loc[r.image_id, "memory_ids"]]; rr.append({"crop": r.crop, "same_share": float(np.mean([I.loc[i, "collection"] == r.collection for i in ids])), "top1_same": float(I.loc[ids[0], "collection"] == r.collection), "hit": r.hit, "mace": r.mace, "sage": r.sage,
                   "bank_same": float(np.mean([I.loc[i, "collection"] == r.collection for i in X.bank_ids(r.crop, int(r.run_s0))]))})
    Rr = pd.DataFrame(rr)
    out["retrieved_collection"] = {"retrieved_same_share": float(Rr.same_share.mean() * 100), "bank_same_share": float(Rr.bank_same.mean() * 100), "by_top1_same": {str(k): {"n": int(len(g)), "hit": float(g.hit.mean() * 100), "mace": float(g.mace.mean() * 100), "sage": float(g.sage.mean() * 100)} for k, g in Rr.groupby("top1_same")}}
    out["usage"] = use_stats(L, "E4other|")
    out["bank_size"] = {"other_mean": float(d.bank_n_other.mean()), "family_mean": float(d.fam_n.mean()), "family_rows_changed": int(len(Fm))}
    d.to_csv(PR / "hero_iid_cross_collection.csv", index=False)
    return out


def dedup(H):
    z = np.load(data.C.DATA / "audit/clip_vitb32_embeddings.npz"); E = z["E"].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True); idx = {k: i for i, k in enumerate(z["ids"])}
    I = data.images(); Q = JB.queries().set_index("image_id"); v = []
    for r in H.itertuples():
        ids = X.bank_ids(r.crop, int(Q.loc[r.image_id, "run_s0"])); s = E[[idx[i] for i in ids]] @ E[idx[r.image_id]]; lab = np.array([I.loc[i, "class"] for i in ids])
        v.append(float(s[lab != r.label].max()) if (lab != r.label).any() else -1.0)
    H = H.assign(max_diff=v); out = {}
    for t in (None, 0.90, 0.85):
        g = H if t is None else H[H.max_diff < t]; key = "all" if t is None else f"drop_ge_{t}"; out[key] = {}
        for sp in ("iid", "ood"):
            d = g[g.split == sp]
            out[key][sp] = {"n": int(len(d)), "removed": int((H.split == sp).sum() - len(d)), "sage": eq(d, "sage"), "mace": eq(d, "mace"), "dino_s0": eq(d, "dino_s0"), "mace_minus_sage": contrast(d, ["mace"], ["sage"]), "mace_minus_labels": contrast(d, ["mace"], ["labels_all"])}
    for sp in ("iid", "ood"):
        d = H[(H.split == sp) & (H.max_diff >= 0.90)]
        out[f"flagged_{sp}"] = {"n": int(len(d)), "mace": float(d.mace.mean() * 100), "sage": float(d.sage.mean() * 100), "dino_s0": float(d.dino_s0.mean() * 100)}
    return out


def main():
    ST.mkdir(exist_ok=True); TB.mkdir(exist_ok=True)
    if sys.argv[1] == "select":
        return select()
    H = table(); L = ledger()
    H.to_csv(PR / "hero_all_methods.csv", index=False)
    counts = {c: int(H[c].notna().sum()) for c in ("sage", "mace", "labels_all", "dino_labels", "dino_mace", "h_sage", "h_mace", "h_labels", "other_full", "family")}
    S = {"row_counts": counts, "E1": e1(H, L), "E2": e2(H, L), "E3": e3(H, L), "E4": e4(H, L), "dedup": dedup(H)}
    (ST / "extension_results.json").write_text(json.dumps(S, indent=1, default=float)); print("written", counts)


if __name__ == "__main__":
    main()
