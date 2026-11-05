"""Results-first tables for the MACE paper: every traceable measured value is filled in; every other cell is marked.

  python -m mace10.paper            writes <paper>/tables/t{1..4}_*.tex, <paper>/numbers_mace10.tex,
                                    mace10/out/table_cells.csv (cell -> status, n, source) and RESULTS_FIRST_DRAFT.md

Cell status (rendered in the PDF):
  value                measured on this protocol (source recorded per cell)
  NR                   NEEDS RUN (incremental calls and USD from the dry run listed in the caption and table_specs.yaml)
  NA                   NOT APPLICABLE
  pilot panel          PILOT / DIFFERENT PROTOCOL (soybean, seed 0, runs 0-4, 5 images per disease and run); never merged
                       into ten-crop rows
Intervals: 95% bootstrap over unique query images, stratified by crop (2,000 resamples, seed 0); equal-crop means.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; OUT = HERE / "out"; LOCK = HERE / "locked"
PAPER = HERE / "paper_src"   # editable multi-file source; mace10/build_submission.py writes the single-file Overleaf paper
PILOT = HERE.parent / "runs"
NR = "\\nr{}"; NA = "\\na{}"
cells: list[dict] = []; mac: dict[str, str] = {}
RNG = np.random.default_rng(0); B = 2000


def cell(table, row, col, value=None, status="measured", n=None, source=""):
    cells.append({"table": table, "row": row, "col": col, "status": status, "value": value, "n": n, "source": source})
    return value if status == "measured" else NR if status == "needs_run" else NA


def eqcrop_ci(df, col):
    """Equal-crop mean accuracy (%) and stratified bootstrap 95% CI over unique images."""
    groups = [g[col].to_numpy(float) for _, g in df.groupby("crop")]
    est = np.mean([g.mean() for g in groups]) * 100
    bs = [np.mean([RNG.choice(g, len(g)).mean() for g in groups]) * 100 for _ in range(B)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(e, lo=None, hi=None):
    return f"{e:.1f}" if lo is None else f"{e:.1f} {{\\tiny[{lo:.1f}, {hi:.1f}]}}"


RES: dict = {}


def minus(t):
    """True minus signs for negative numbers in generated text."""
    import re
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", t)
METHOD_MACRO = {"sage": "Sage", "majority": "Maj", "wvote": "Vote", "clip_s0": "Clip", "dino_s0": "Dino", "mace": "Mace", "mace_K1": "MaceKone", "mace_K16": "MaceKsixteen",
                "labels": "Labels", "labels_all": "LabelsAll", "dino_labels": "DinoLabels", "masked": "Masked", "shuffled": "Shuffled", "random": "Random", "same_source": "SameSrc", "other_source": "OtherSrc", "agentic": "Agentic", "imageref": "ImageRef"}


def acc(panel, sp, m, interval=True, key="acc"):
    """Measured value as text, or the NEEDS RUN mark when the package has not produced it."""
    d = RES.get("accuracy", {}).get(panel, {}).get(sp, {}).get(m)
    if d is None:
        return NR
    return fmt(d[key], *d["ci"]) if interval and key == "acc" else fmt(d[key])


def macros_from_results():
    for panel, tag in (("hero", "Hero"), ("ablation", "Abl"), ("efficiency", "Eff"), ("source", "Src")):
        for sp, d in RES["accuracy"][panel].items():
            for m, v in d.items():
                if m in METHOD_MACRO:
                    mac[f"tc{tag}{METHOD_MACRO[m]}{sp.upper()}"] = f"{v['acc']:.1f}"
    words = {"mace_minus_sage": "MaceSage", "mace_minus_dino": "MaceDino", "labels_minus_mace": "LabelsMace", "masked_minus_labels": "MaskedLabels", "mace_minus_masked": "MaceMasked",
             "mace_minus_shuffled": "MaceShuffled", "mace_minus_random": "MaceRandom", "same_minus_other": "SameOther", "imageref_minus_mace": "ImageRefMace", "agentic_minus_mace": "AgenticMace"}
    for k, v in RES["contrasts"].items():
        name, sp = k.rsplit("_", 1); w = "tcD" + words[name] + sp.upper()
        mac[w] = f"{v['est']:+.1f}"; mac[w + "lo"] = f"{v['ci'][0]:+.1f}"; mac[w + "hi"] = f"{v['ci'][1]:+.1f}"
        if "p_holm" in v:
            mac[w + "p"] = "<0.001" if v["p_holm"] < 0.001 else f"{v['p_holm']:.3f}"
    for k, v in RES.get("amendment", {}).items():
        if k == "hit_rates":
            for sp, h in v.items():
                mac[f"tcHitBm{sp.upper()}"] = f"{h['bm25']:.1f}"; mac[f"tcHitDino{sp.upper()}"] = f"{h['dino']:.1f}"
            continue
        name, sp = k.rsplit("_", 1); w = "tcA" + {"labels_all_minus_mace": "LabelsMace", "dino_labels_minus_labels_all": "DinoBm", "dino_labels_minus_dino": "DinoNN"}[name] + sp.upper()
        mac[w] = f"{v['est']:+.1f}"; mac[w + "lo"] = f"{v['ci'][0]:+.1f}"; mac[w + "hi"] = f"{v['ci'][1]:+.1f}"
    W = pd.read_csv(OUT / "results_wide.csv"); H = W[W.hero]; g = H.groupby(["crop", "split"])[["mace", "sage", "hit"]].mean().mul(100); g["gain"] = g.mace - g.sage
    mac["tcHitR"] = f"{np.corrcoef(g.hit, g.gain)[0, 1]:.2f}"; pot = g.loc[("Potato", "ood")]; mac["tcPotatoHit"] = f"{pot.hit:.1f}"; mac["tcPotatoLoss"] = f"{-pot.gain:.1f}"
    Lb = pd.read_csv(OUT / "local_baselines.csv"); Lb = Lb[(Lb.panel == "hero") & (Lb.seed == 0) & (Lb.K == 4)].groupby("crop").bank_size.mean()
    mac["tcBankMean"] = f"{Lb.mean():.0f}"; mac["tcBankMin"] = f"{Lb.min():.0f}"; mac["tcBankMax"] = f"{Lb.max():.0f}"; mac["tcBankUsd"] = f"{Lb.mean() * 0.0043:.2f}"   # 0.0043: measured USD per caption
    for m, v in RES["efficiency"]["methods"].items():
        mac[f"tcEff{METHOD_MACRO[m]}PerCorrect"] = f"{v['usd'] / (v['acc_all'] / 100):.3f}"
    P1 = pd.DataFrame(map(json.loads, (OUT / "P1/predictions.jsonl").open())); Mq = P1[P1.condition == "mace"].copy(); Mq["sage"] = Mq.image_id.map(P1[P1.condition == "sage"].set_index("image_id").correct).astype(bool); Mq["correct"] = Mq.correct.astype(bool)
    Mq["nt"] = [sum(l == r.label for l in r.memory_labels_shown) for r in Mq.itertuples()]
    for sp, S_ in (("iid", "IID"), ("ood", "OOD")):
        g_ = Mq[Mq.split == sp]; mac[f"tcFix{S_}"] = f"{100 * (g_.correct & ~g_.sage).mean():.1f}"; mac[f"tcBreak{S_}"] = f"{100 * (~g_.correct & g_.sage).mean():.1f}"
        mac[f"tcDoseNone{S_}"] = f"{100 * g_[g_.nt == 0].correct.mean():.0f}"; mac[f"tcDoseAll{S_}"] = f"{100 * g_[g_.nt == 4].correct.mean():.0f}"; mac[f"tcDoseAllN{S_}"] = f"{(g_.nt == 4).sum()}"
    E = RES["efficiency"]["methods"]; mac["tcEffGainPts"] = f"{E['mace']['acc_all'] - E['sage']['acc_all']:.1f}"; mac["tcEffGainUsd"] = f"{E['mace']['usd'] - E['sage']['usd']:.4f}"
    for sp, S in (("iid", "IID"), ("ood", "OOD")):
        h = RES["hit_miss"][sp]; ab = RES["accuracy"]["ablation"][sp]
        mac[f"tcHitGain{S}"] = f"{h['hit']['mace'] - h['hit']['sage']:.1f}"; mac[f"tcMissLoss{S}"] = f"{h['miss']['sage'] - h['miss']['mace']:.1f}"
        mac[f"tcLabelsVote{S}"] = f"{ab['labels']['acc'] - ab['majority']['acc']:.1f}"
        mac[f"tcCapGain{S}"] = f"{-RES['amendment'][f'labels_all_minus_mace_{sp}']['est']:.1f}"
    for sp, v in RES["hit_miss"].items():
        S = sp.upper(); mac[f"tcHitRate{S}"] = f"{v['hit_rate']:.1f}"
        for nm in ("hit", "miss"):
            mac[f"tc{nm.title()}Mace{S}"] = f"{v[nm]['mace']:.1f}"; mac[f"tc{nm.title()}Sage{S}"] = f"{v[nm]['sage']:.1f}"; mac[f"tc{nm.title()}N{S}"] = f"{v[nm]['n']}"
    for sp, v in RES["selection"].items():
        S = sp.upper()
        for m in ("mace", "clip", "dino"):
            mac[f"tcSD{m.title()}{S}"] = f"{v[m]['sd']:.1f}"
        for k, w in (("mean_minus_sage", "Sage"), ("mean_minus_dino", "Dino")):
            mac[f"tcMean{w}{S}"] = f"{v[k]['est']:+.1f}"
    for m, v in RES["efficiency"]["methods"].items():
        w = METHOD_MACRO[m]; mac[f"tcEff{w}Acc"] = f"{v['acc_all']:.1f}"; mac[f"tcEff{w}Usd"] = f"{v['usd']:.3f}"; mac[f"tcEff{w}Sec"] = f"{v['latency_s']:.0f}"; mac[f"tcEff{w}Comp"] = f"{v['completion']:.1f}"


def dry():
    D = pd.read_csv(OUT / "dryrun_packages.csv"); S = json.loads((OUT / "dryrun.json").read_text())
    return D, S


def usd(D, pkg, cond=None):
    d = D[(D.package == pkg) & ((D.condition == cond) if cond else True)]
    return d.new_calls.sum(), d.projected_usd.sum()


def pilot():
    P = pd.DataFrame(map(json.loads, (PILOT / "soybean_seed0_n5_pilot/predictions.jsonl").open()))
    G = pd.DataFrame(map(json.loads, (PILOT / "soybean_seed0_n5_pilot_agentic/predictions.jsonl").open()))
    bad = (G.caption_reads == 0) & (G.num_turns == 1) & (G.prediction == "UNKNOWN"); G = G[~bad]
    return P, G


def t1(L, D):
    """Ten-crop hero. Panel A: equal-crop means per method. Panel B: per crop. Panel C: soybean pilot."""
    h = L[(L.panel == "hero") & L.hero & (L.seed == 0) & (L.K == 4)]
    rowsA = []
    calls, cost = usd(D, "P1")
    for name, m in (("SAGE (KB only)", "sage"), ("Retrieved-label majority", "majority"), ("Score-weighted label vote", "wvote"),
                    ("OpenCLIP 1-NN", "clip_s0"), ("DINOv2 1-NN", "dino_s0"), ("Labels only$^\\dagger$", "labels_all"), ("DINOv2-retrieved labels$^\\dagger$", "dino_labels"),
                    ("MACE retrieval ($K{=}4$)", "mace")):
        out = [name]
        for sp in ("iid", "ood"):
            v = acc("hero", sp, m); out.append(cell("T1", name, sp, v, status="needs_run" if v == NR else "measured", n=int((h.split == sp).sum()), source="mace10/out/results.json"))
        for sp in ("iid", "ood"):
            v = acc("hero", sp, m, key="macro"); out.append(cell("T1", name, f"macro_{sp}", v, status="needs_run" if v == NR else "measured", source="mace10/out/results.json"))
        rowsA.append(out)
        if m in ("clip_s0", "dino_s0"):
            for sp in ("iid", "ood"):
                mac[f"hero{'Clip' if 'clip' in m else 'Dino'}{sp.upper()}"] = f"{RES['accuracy']['hero'][sp][m]['acc']:.1f}"
    rowsB = []
    for crop, g in h.groupby("crop", sort=False):
        r = [crop]
        for sp in ("iid", "ood"):
            gg = g[g.split == sp]; c = RES.get("per_crop", {}).get(crop, {}).get(sp)
            r += [str(len(gg)), f"{c['sage']:.1f}", f"{100 * gg.dino_correct.mean():.1f}", f"{c['mace']:.1f}", f"{c['mace'] - c['sage']:+.1f}"] if c else [str(len(gg)), NR, f"{100 * gg.dino_correct.mean():.1f}", NR, NR]
        rowsB.append(r)
    # pilot panel
    P, G = pilot(); sage = P[P.condition == "sage"].set_index("image_id").correct
    R = P[P.condition == "agmem"]; Lp = L[L.panel == "pilot"]
    pv = []
    for sp, spp in (("iid", "test_iid"), ("ood", "test_ood_in_scope")):
        r4 = R[(R.K == 4) & (R.split == spp)]; lp = Lp[(Lp.K == 4) & (Lp.split == sp)]
        maj = np.mean([bool(s) and pd.Series(s).value_counts().index[0] == l for l, s in zip(r4.label, r4.memory_labels_shown)]) * 100
        def wvote(labs, rel):
            sc = {}
            for l_, w in zip(labs or [], rel or []):
                sc[l_] = sc.get(l_, 0) + w
            return max(sorted(sc), key=lambda k: sc[k]) if sc else None
        wv = np.mean([wvote(s_, w_) == l for l, s_, w_ in zip(r4.label, r4.memory_labels_shown, r4.memory_relevance)]) * 100
        pv.append((100 * r4.image_id.map(sage).mean(), maj, 100 * lp.clip_correct.mean(), 100 * lp.dino_correct.mean(), 100 * r4.correct.mean(), len(r4), wv))
        mac[f"pilotDino{sp.upper()}"] = f"{100 * lp.dino_correct.mean():.1f}"
    pilot_cols = {"SAGE (KB only)": 0, "Retrieved-label majority": 1, "Score-weighted label vote": 6, "OpenCLIP 1-NN": 2, "DINOv2 1-NN": 3, "MACE retrieval ($K{=}4$)": 4}
    for r in rowsA:
        i = pilot_cols.get(r[0])
        r += [""] + ([f"{pv[0][i]:.1f}", f"{pv[1][i]:.1f}"] if i is not None else ["--", "--"])
    body = "\n".join(" & ".join(r) + " \\\\" for r in rowsA)
    crop_lines = "\n".join(" & ".join(r) + " \\\\" for r in rowsB)
    n1, c1 = calls, cost
    tex = rf"""\begin{{table*}}[t]
\centering\scriptsize\setlength{{\tabcolsep}}{{5pt}}
\caption{{\textbf{{Ten-crop results}} (primary memory selection: $K{{=}}4$ images per disease, four retrieved captions). Accuracy (\%), equal-crop mean over 10 crops ({int((h.split == 'iid').sum()):,} IID and {int((h.split == 'ood').sum()):,} held-out-collection queries) with a 95\% bootstrap interval over unique images, stratified by crop; macro: mean per-disease recall. Label votes and 1-NN use the exact memory bank MACE sees and call no model. $^\dagger$Post hoc, added after the 400-query labels-only control was read. Right: the earlier soybean pilot under a different protocol (5 runs, 5 images per disease and run; {pv[0][5]} IID and {pv[1][5]} held-out item-runs), not pooled with the ten-crop columns.}}
\label{{tab:hero}}
\begin{{tabular}}{{lccccc@{{\hspace{{10pt}}}}cc}}
\toprule
 & \multicolumn{{4}}{{c}}{{Ten crops}} & & \multicolumn{{2}}{{c}}{{\pilottag{{}} soybean, different protocol}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{7-8}}
Method & IID & Held-out & Macro IID & Macro held-out & & IID & Held-out \\
\midrule
{body.replace(" \\\\", " \\\\").replace(" & ", " & ", 0)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (PAPER / "tables/t1_hero.tex").write_text(tex)
    crop_tex = ("% Panel B (per crop) for the supporting results\n\\begin{tabular}{lcccccccccc}\n\\toprule\n & \\multicolumn{5}{c}{IID} & \\multicolumn{5}{c}{Held-out}\\\\\n"
                "Crop & $n$ & SAGE & DINOv2 & MACE & $\\Delta$ & $n$ & SAGE & DINOv2 & MACE & $\\Delta$ \\\\\n\\midrule\n" + crop_lines + "\n\\bottomrule\n\\end{tabular}\n")
    (PAPER / "tables/t1b_per_crop.tex").write_text(crop_tex)


def macro_recall(df, key):
    return 100 * df.groupby("label")[key].mean().mean()


CONTROLS = [("MACE (captions + labels)", "mace"), ("Labels only", "labels"), ("Captions only (masked)", "masked"), ("Shuffled labels", "shuffled"), ("Random references", "random")]


def delta(m):
    """Difference from MACE on the ablation panel, IID / held-out."""
    A = RES.get("accuracy", {}).get("ablation", {})
    if m == "mace" or m not in A.get("iid", {}):
        return "--"
    return " / ".join(f"{A[sp][m]['acc'] - A[sp]['mace']['acc']:+.1f}" for sp in ("iid", "ood"))


def t2(L, D):
    a = L[(L.panel == "hero") & L.ablation & (L.seed == 0)]
    c3, u3 = usd(D, "P3"); c4, u4 = usd(D, "P4")
    rows = []
    for K in (1, 4, 16):
        r = [f"$K{{=}}{K}$"]
        mk = "mace" if K == 4 else f"mace_K{K}"
        r.append(f"{acc('ablation', 'iid', mk, False)} / {acc('ablation', 'ood', mk, False)}")
        for sp in ("iid", "ood"):
            g = a[(a.K == K) & (a.split == sp)]; e, lo, hi = eqcrop_ci(g, "dino_correct")
            r.append(fmt(e)); mac[f"ablDinoK{ {1: 'one', 4: 'four', 16: 'sixteen'}[K]}{sp.upper()}"] = f"{e:.1f}"
        rows.append(r)
    P, G = pilot(); R = P[P.condition == "agmem"]; sage = P[P.condition == "sage"].set_index("image_id").correct
    pil = []
    for K in (1, 4, 16):
        v = [100 * R[(R.K == K) & (R.split == s)].correct.mean() for s in ("test_iid", "test_ood_in_scope")]
        pil.append(f"$K{{=}}{K}$ & {v[0]:.1f} / {v[1]:.1f} & & \\\\")
    body_a = "\n".join(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\" for r in rows)
    body_b = "\n".join(f"{n_} & {acc('ablation', 'iid', m_, False)} & {acc('ablation', 'ood', m_, False)} & {delta(m_)} \\\\" for n_, m_ in CONTROLS)
    tex = rf"""\begin{{table}}[t]
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\caption{{\textbf{{What the gain is made of}} (400 nested queries, 40 per crop, 20 IID + 20 held-out, seed 0). Accuracy (\%), equal-crop mean. A: memory size, nested banks, always four retrieved captions; DINOv2 1-NN on the same banks for reference (measured, no model call). B: content controls at $K{{=}}4$ with identical queries, KB claims and candidates. $K{{=}}16$ banks hold fewer than 16 images for diseases with few training images.}}
\label{{tab:ablation}}
\begin{{tabular}}{{lccc}}
\toprule
\multicolumn{{4}}{{l}}{{\textit{{A. Memory size}}}} \\
Bank & MACE (IID / held-out) & DINOv2 IID & DINOv2 held-out \\
\midrule
{body_a}
\midrule
\multicolumn{{4}}{{l}}{{\textit{{B. Content controls, $K{{=}}4$}}}} \\
Evidence shown & IID & Held-out & $\Delta$ vs MACE \\
\midrule
{body_b}
\midrule
\multicolumn{{4}}{{l}}{{\textit{{\pilottag{{}} Soybean, different protocol: MACE (IID / held-out)}}}} \\
{chr(10).join(pil)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (PAPER / "tables/t2_ablation.tex").write_text(tex)


def t3(L, D):
    h = L[(L.panel == "hero") & L.hero & (L.K == 4)]
    c2, u2 = usd(D, "P2"); c5, u5 = usd(D, "P5")
    lines = []
    for name, key in (("MACE retrieval", None), ("OpenCLIP 1-NN", "clip_correct"), ("DINOv2 1-NN", "dino_correct")):
        r = [name]; vals = {"iid": [], "ood": []}
        for s in (0, 1, 2):
            for sp in ("iid", "ood"):
                if key:
                    e, _, _ = eqcrop_ci(h[(h.seed == s) & (h.split == sp)], key); vals[sp].append(e); r.append(f"{e:.1f}")
                else:
                    r.append(acc("hero", sp, "mace" if s == 0 else f"mace_s{s}", False))
        for sp in ("iid", "ood"):
            sel = RES.get("selection", {}).get(sp)
            r.append(f"{np.std(vals[sp], ddof=1):.1f}" if key else f"{sel['mace']['sd']:.1f}" if sel else NR)
        lines.append(" & ".join(r) + " \\\\")
        if key:
            mac[f"rel{'Clip' if 'clip' in key else 'Dino'}SD"] = f"{np.std(vals['iid'], ddof=1):.1f}"
    # pilot hit/miss (measured, different protocol)
    P, G = pilot(); sage = P[P.condition == "sage"].set_index("image_id").correct
    R = P[(P.condition == "agmem") & (P.K == 4)].copy(); R["hit"] = [l in (s or []) for l, s in zip(R.label, R.memory_labels_shown)]
    R["sage"] = R.image_id.map(sage).astype(bool)
    pl = []
    for sp, nm in (("test_iid", "IID"), ("test_ood_in_scope", "Held-out")):
        g = R[R.split == sp]; hh, mm = g[g.hit], g[~g.hit]
        pl.append(f"{nm} & {100 * g.hit.mean():.1f} & {100 * hh.correct.mean():.1f} / {100 * hh.sage.mean():.1f} & {100 * mm.correct.mean():.1f} / {100 * mm.sage.mean():.1f} \\\\")
    hm = [f"Ten crops, {nm} & {v['hit_rate']:.1f} & {v['hit']['mace']:.1f} / {v['hit']['sage']:.1f} & {v['miss']['mace']:.1f} / {v['miss']['sage']:.1f} \\\\" for nm, v in
          (("IID", RES["hit_miss"].get("iid")), ("held-out", RES["hit_miss"].get("ood"))) if v]
    srcn = RES.get("accuracy", {}).get("source", {}).get("iid", {}).get("sage", {}).get("n", "--")
    tex = rf"""\begin{{table}}[t]
\centering\scriptsize\setlength{{\tabcolsep}}{{2.6pt}}
\caption{{\textbf{{Reliability.}} A: accuracy (\%, equal-crop mean, 1,600 queries) under three locked memory selections (release seeds 0--2; each query keeps its fold eligibility); SD across selections is memory-selection sensitivity, not an image-sampling interval. B: hit = true disease among the four retrieved captions; MACE / SAGE accuracy on identical hit and miss items; associative, and hit status is oracle information. C: same- vs other-source memory on {srcn} paired IID queries.}}
\label{{tab:reliability}}
\begin{{tabular}}{{lcccccccc}}
\toprule
\multicolumn{{9}}{{l}}{{\textit{{A. Memory selection (IID, held-out per selection)}}}} \\
 & \multicolumn{{2}}{{c}}{{Seed 0}} & \multicolumn{{2}}{{c}}{{Seed 1}} & \multicolumn{{2}}{{c}}{{Seed 2}} & \multicolumn{{2}}{{c}}{{SD}} \\
Method & IID & H-o & IID & H-o & IID & H-o & IID & H-o \\
\midrule
{chr(10).join(lines)}
\midrule
\multicolumn{{9}}{{l}}{{\textit{{B. Retrieval hit and miss: ten crops; \pilottag{{}} soybean, different protocol, $K{{=}}4$}}}} \\
Set & Hit & \multicolumn{{3}}{{c}}{{MACE / SAGE on hits}} & \multicolumn{{4}}{{c}}{{MACE / SAGE on misses}} \\
\midrule
{chr(10).join(_hm_rows(hm + pl))}
\midrule
\multicolumn{{9}}{{l}}{{\textit{{C. Same- vs other-source memory (IID): same {acc('source', 'iid', 'same_source', False)}, other {acc('source', 'iid', 'other_source', False)}}}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (PAPER / "tables/t3_reliability.tex").write_text(tex)


def _hm_rows(pl):
    out = []
    for l in pl:
        a, b, c, d = [x.strip() for x in l.rstrip(" \\").split("&")]
        out.append(f"{a} & {b} & \\multicolumn{{3}}{{c}}{{{c}}} & \\multicolumn{{4}}{{c}}{{{d}}} \\\\")
    return out


def t4(D):
    c6, u6 = usd(D, "P6"); c7, u7 = usd(D, "P7")
    P, G = pilot()
    led = [json.loads(l) for l in (PILOT / "cost_ledger.jsonl").open()]
    el = lambda st: np.median([r["elapsed_s"] for r in led if r.get("stage") == st and not r["cache_hit"] and r.get("elapsed_s")])
    cost = lambda st: np.mean([r["cost_usd"] for r in led if r.get("stage") == st and not r["cache_hit"]])
    cap = cost("caption")
    sage = P[P.condition == "sage"].set_index("image_id").correct
    R4 = P[(P.condition == "agmem") & (P.K == 4)]; G4 = G[G.K == 4]
    pil = [("SAGE (KB only)", 100 * R4.image_id.map(sage).mean(), "100", el("diag:sage"), cost("diag:sage")),
           ("MACE retrieval", 100 * R4.correct.mean(), "100", el("diag:agmem") + el("caption"), cost("diag:agmem") + cap),
           ("MACE agentic", 100 * G4.correct.mean(), f"{100 * (G4.prediction != 'UNKNOWN').mean():.1f}", np.median(G4.elapsed_s.dropna()) if "elapsed_s" in G4 else el("diag:agentic"), cost("diag:agentic") + cap)]
    Em = RES.get("efficiency", {}).get("methods", {})
    rows = "\n".join((f"{n} & {Em[m]['acc_all']:.1f} & {Em[m]['completion']:.1f} & {Em[m]['latency_s']:.0f} & {Em[m]['usd']:.3f} \\\\" if m in Em else f"{n} & {NR} & {NR} & {NR} & {NR} \\\\")
                      for n, m in (("SAGE (KB only)", "sage"), ("Labels only", "labels"), ("MACE retrieval", "mace"), ("MACE agentic", "agentic"), ("Image references", "imageref")))
    prow = "\n".join(f"{n} & {a:.1f} & {c} & {l:.0f} & {u:.3f} \\\\" for n, a, c, l, u in pil)
    tex = rf"""\begin{{table}}[t]
\centering\scriptsize\setlength{{\tabcolsep}}{{3.5pt}}
\caption{{\textbf{{Accuracy against evidence-access effort}} (200 matched queries, 5 crops, 20 IID + 20 held-out each). Accuracy counts failures as wrong; completion = parseable answer; latency = median wall-clock seconds per query; USD = list-price estimate per query including the test caption. Agentic: four reads, at most six turns. Image references: the same four retrieved items shown as images. \pilottag{{}} rows: soybean pilot, different protocol (agentic without a turn bound).}}
\label{{tab:efficiency}}
\begin{{tabular}}{{lcccc}}
\toprule
Method & Acc. & Compl.\ (\%) & Latency (s) & USD \\
\midrule
{rows}
\midrule
\multicolumn{{5}}{{l}}{{\textit{{\pilottag{{}} Soybean, different protocol (IID and held-out pooled, $K{{=}}4$)}}}} \\
{prow}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (PAPER / "tables/t4_efficiency.tex").write_text(tex)


def t2c(D):
    """Q2 content controls (all pending) and the matched-efficiency package, in one compact table."""
    c3, u3 = usd(D, "P3"); c4, u4 = usd(D, "P4"); c6, u6 = usd(D, "P6"); c7, u7 = usd(D, "P7")
    tests = {"mace": "reference", "labels": "lookup alone", "masked": "descriptions", "shuffled": "label following", "random": "retrieval"}
    body = "\n".join(f"{n_} & {acc('ablation', 'iid', m_)} & {acc('ablation', 'ood', m_)} & {tests[m_]} \\\\" for n_, m_ in CONTROLS)
    tex = rf"""\begin{{table}}[t]
\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\caption{{\textbf{{Content controls}} at $K{{=}}4$ on the 400-query ablation panel (40 per crop, 20 IID + 20 held-out, nested in Table~\ref{{tab:hero}}): identical queries, KB claims, candidates and retrieved items; only the memory text differs. Accuracy (\%), equal-crop mean, with a 95\% bootstrap interval.}}
\label{{tab:controls}}
\begin{{tabular}}{{lccl}}
\toprule
Evidence shown & IID & Held-out & Tests \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (PAPER / "tables/t2_controls.tex").write_text(tex)
    mac["costPsix"] = f"{u6:.0f}"; mac["callsPsix"] = f"{c6:,}"; mac["costPseven"] = f"{u7:.0f}"; mac["callsPseven"] = f"{c7:,}"
    c2, u2 = usd(D, "P2"); mac["costPtwo"] = f"{u2:.0f}"; mac["callsPtwo"] = f"{c2:,}"
    c1, u1 = usd(D, "P1"); mac["costPone"] = f"{u1:.0f}"; mac["callsPone"] = f"{c1:,}"


def t_main():
    """Table 1: grouped hero results; the best locked cell per column is green; post hoc arms sit in their own block."""
    A = RES["accuracy"]["hero"]
    groups = [("No model call", [("Label majority vote", "majority"), ("Score-weighted vote", "wvote"), ("OpenCLIP 1-NN", "clip_s0"), ("DINOv2 1-NN", "dino_s0")]),
              ("Frozen VLM", [("SAGE (KB only)", "sage"), (r"\macelogo\ MACE retrieval", "mace")])]
    locked = [m for _, g in groups for _, m in g]
    best = {(sp, k): max(locked, key=lambda m: A[sp][m][k]) for sp in ("iid", "ood") for k in ("acc", "macro")}

    def c(sp, m, k):
        v = A[sp][m]; t = fmt(v["acc"], *v["ci"]) if k == "acc" else fmt(v[k])
        return ("\\cellcolor{bestgreen}" + t) if best.get((sp, k)) == m else t

    def row(n, m):
        return f"{n} & {c('iid', m, 'acc')} & {c('ood', m, 'acc')} & {c('iid', m, 'macro')} & {c('ood', m, 'macro')} \\\\"
    body = []
    for g, rows in groups:
        body += [f"\\multicolumn{{5}}{{l}}{{\\textit{{{g}}}}} \\\\"] + [row(n, m) for n, m in rows] + ["\\midrule"]
    body += ["\\multicolumn{5}{l}{\\textit{Post hoc tests, frozen VLM}} \\\\", row("Labels only", "labels_all"), row("DINOv2-retrieved labels", "dino_labels")]
    n1, n2 = A["iid"]["mace"]["n"], A["ood"]["mace"]["n"]
    tex = "\n".join([r"\begin{table}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
        r"\caption{\textbf{Ten-crop results} ($K{=}4$): accuracy (\%), 95\% intervals. Green: best planned method.}",
        r"\label{tab:hero}", r"\begin{tabular}{lcccc}", r"\toprule", r"Method & IID & Held-out & Macro IID & Macro H-o \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (PAPER / "tables/t_main.tex").write_text(minus(tex))


def t_info():
    """Table 2: caption-versus-label evidence. A: locked matched cohort. B: expanded post hoc panel."""
    A = RES["accuracy"]["ablation"]; H = RES["accuracy"]["hero"]; C = RES["amendment"]
    f = lambda d, m: fmt(d[m]["acc"], *d[m]["ci"])
    ci = lambda k: f"{C[k]['est']:+.1f} {{\\tiny[{C[k]['ci'][0]:+.1f}, {C[k]['ci'][1]:+.1f}]}}"
    best = {sp: max(("sage", "mace", "labels", "masked", "shuffled", "random"), key=lambda m: A[sp][m]["acc"]) for sp in ("iid", "ood")}
    g = lambda sp, m: ("\\cellcolor{bestgreen}" if best[sp] == m else "") + f(A[sp], m)
    d = lambda m: "--" if m == "mace" else " / ".join(f"{A[sp][m]['acc'] - A[sp]['mace']['acc']:+.1f}" for sp in ("iid", "ood"))
    rowsA = [("SAGE (no memory)", "sage"), ("MACE (captions + labels)", "mace"), ("Labels only", "labels"), ("Captions only (masked)", "masked"), ("Random references", "random"), ("Shuffled labels", "shuffled")]
    a = [f"{n} & {g('iid', m)} & {g('ood', m)} & {d(m)} \\\\" for n, m in rowsA]
    b = [f"MACE (captions + labels) & {f(H['iid'], 'mace')} & {f(H['ood'], 'mace')} & -- \\\\",
         f"Labels only & {f(H['iid'], 'labels_all')} & {f(H['ood'], 'labels_all')} & {C['labels_all_minus_mace_iid']['est']:+.1f} / {C['labels_all_minus_mace_ood']['est']:+.1f} \\\\",
         f"DINOv2-retrieved labels & {f(H['iid'], 'dino_labels')} & {f(H['ood'], 'dino_labels')} & {H['iid']['dino_labels']['acc'] - H['iid']['mace']['acc']:+.1f} / {H['ood']['dino_labels']['acc'] - H['ood']['mace']['acc']:+.1f} \\\\",
         f"\\textit{{Labels $-$ MACE}} & {ci('labels_all_minus_mace_iid')} & {ci('labels_all_minus_mace_ood')} & Text adds / unclear \\\\",
         f"\\textit{{DINOv2 $-$ BM25 labels}} & {ci('dino_labels_minus_labels_all_iid')} & {ci('dino_labels_minus_labels_all_ood')} & Matched / unclear \\\\"]
    tex = "\n".join([r"\begin{table}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{1.9pt}",
        r"\caption{\textbf{Post hoc tests}, all 1,600 queries: accuracy (\%), 95\% intervals.}",
        r"\label{tab:controls}", r"\begin{tabular}{lccc}", r"\toprule", r"Evidence shown & IID & Held-out & vs MACE \\\\", r"\midrule", *b, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (PAPER / "tables/t_info.tex").write_text(minus(tex))


def main():
    D, S = dry(); L = pd.read_csv(OUT / "local_baselines.csv")
    RES.update(json.loads((OUT / "results.json").read_text()))
    t1(L, D); t2(L, D); t3(L, D); t4(D); t2c(D); t_main(); t_info()
    macros_from_results()
    mac["dryTotal"] = f"{S['projected_additional_usd']:.0f}"; mac["dryRequired"] = f"{S['projected_required_usd']:.0f}"
    mac["dryCapsNew"] = f"{S['captions_new']:,}"; mac["nHero"] = "1,600"
    (PAPER / "numbers_mace10.tex").write_text("% generated by code/agmem-bench/mace10/paper.py; do not edit\n" +
                                              "\n".join(f"\\newcommand{{\\{k}}}{{{minus(v) if k.startswith('tc') else v}}}" for k, v in sorted(mac.items())) + "\n")
    pd.DataFrame(cells).to_csv(OUT / "table_cells.csv", index=False)
    print(json.dumps(mac, indent=0))


if __name__ == "__main__":
    main()
