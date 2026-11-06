"""Tables A-D and figures 1-4 from saved statistics only; no model call.

  python -m experiments.mace_extension.scripts.build_outputs
"""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analyze import PR, ROOT, ST, TB  # noqa: E402

FG = ROOT / "figures"; FG.mkdir(exist_ok=True)
S = json.loads((ST / "extension_results.json").read_text()); B = json.loads((ST / "robustness_posthoc.json").read_text())
SP = (("iid", "IID"), ("ood", "Held-out"))
OI = {"blue": "#0072B2", "green": "#009E73", "verm": "#D55E00", "gold": "#E69F00", "grey": "#8a8a8a", "ink": "#333333"}
plt.rcParams["path.effects"] = [pe.Normal()]   # outlines: AAAI bans Identity-H and Type 3 fonts
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def f(x):
    return f"{x:.1f}"


def ci(c):
    return f"{f(c['ci'][0])}, {f(c['ci'][1])}"


def pm(c):
    return f"{c['est']:+.1f} [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]"


def tex(rows, cols, name, caption, align, best=None):
    """SAGE table format: caption above, booktabs, best cell shaded bestgreen. Needs \\definecolor{bestgreen}."""
    L = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        L.append(" & ".join(rf"\cellcolor{{bestgreen}}{v}" if best and (i, j) in best else str(v) for j, v in enumerate(r)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TB / f"{name}.tex").write_text(re.sub(r"(?<!\\)%", r"\\%", "\n".join(L)) + "\n")
    pd.DataFrame(rows, columns=cols).to_csv(TB / f"{name}.csv", index=False)


def table_a():
    e = S["E1"]; R = []; C = {c["name"]: c for c in e["contrasts"] + e["sonnet_reference"]}
    for m, cols, tag in (("Sonnet 5.5", ("sage", "labels_all", "mace"), "Sonnet"), ("Haiku 5.5", ("h_sage", "h_labels", "h_mace"), "Haiku")):
        for c, nm in zip(cols, ("SAGE", "Labels only", "MACE")):
            r = [m, nm]
            for sp, _ in SP:
                a = e["accuracy"][sp][tag][c]; r.append(f"{f(a['acc'])} [{ci(a)}]")
            R.append(r)
    best = {(2, 2), (2, 3)}
    tex(R, ["Model", "Method", "IID", "Held-out"], "table_A_cross_model", "Diagnosis accuracy by Claude model, equal-crop mean with 95\\% CI, 1,600 queries per method. Both models are from one family. Captions, retrieval and prompts are unchanged, and Sonnet wrote all captions.", "llcc", best)
    G = []
    for lab, s, h in (("MACE minus SAGE", "sonnet_mace_minus_sage", "haiku_mace_minus_sage"), ("MACE minus labels", "sonnet_mace_minus_labels_all", "haiku_mace_minus_labels")):
        for sp, spn in SP:
            fl = "mace_vs_sage" if "SAGE" in lab else "mace_vs_labels"; hc = C[f"{h}_{sp}"]; sc = C[f"{s}_{sp}"]
            G.append([lab, spn, pm(sc), pm(hc), f"{hc['p_holm']:.3f}", f(e["flips"][f"Sonnet_{sp}"][fl]["corrected"]) + " / " + f(e["flips"][f"Sonnet_{sp}"][fl]["introduced"]), f(e["flips"][f"Haiku_{sp}"][fl]["corrected"]) + " / " + f(e["flips"][f"Haiku_{sp}"][fl]["introduced"])])
    pd.DataFrame(G, columns=["Contrast", "Split", "Sonnet (reference)", "Haiku (new)", "Haiku Holm p", "Sonnet fixed/broken %", "Haiku fixed/broken %"]).to_csv(TB / "table_A_gains.csv", index=False)


def table_b():
    e = S["E2"]; lab = {"labels_all": ("BM25", "Labels"), "mace": ("BM25", "Labels + captions"), "dino_labels": ("DINOv2", "Labels"), "dino_mace": ("DINOv2", "Labels + captions")}
    R = [[*lab[c], *[f"{f(e['cells'][sp][c]['acc'])} [{ci(e['cells'][sp][c])}]" for sp, _ in SP]] for c in lab]
    tex(R, ["Retrieval", "Evidence", "IID", "Held-out"], "table_B_factorial", "Retrieval by evidence factorial, Sonnet 5.5, equal-crop accuracy with 95\\% CI. The DINOv2 + captions cell is new.", "llcc", {(1, 2), (1, 3)})
    pd.DataFrame([[c["name"], f"{c['est']:+.1f}", f"{c['ci'][0]:+.1f}", f"{c['ci'][1]:+.1f}", f"{c['p_holm']:.3f}", c["reading"]] for c in e["contrasts"]], columns=["contrast", "est", "lo", "hi", "p_holm", "reading"]).to_csv(TB / "table_B_contrasts.csv", index=False)


def table_c():
    e = S["E3"]; nm = {"sage": "SAGE", "mace": "MACE", "adaptive": "Fallback rule", "oracle": "Labeled oracle (bound)"}
    R = [[nm[c], *[f"{f(e['acc'][sp][c]['acc'])} [{ci(e['acc'][sp][c])}]" for sp, _ in SP]] for c in nm]
    for sp, spn in SP:
        a = e["avoid"][sp]; R.append([f"{spn}: fallback rate; harms avoided / gains lost", f"{f(a['fallback_rate'])}%", f"{a['avoided']} / {a['introduced']} of {a['sage_right_mace_wrong_total']} harms"])
    tex(R, ["Method", "IID", "Held-out"], "table_C_fallback", "Retrieval-confidence fallback. The rule was frozen on 600 unused IID calibration images and scored once. The oracle uses ground truth and is an upper bound, not a method.", "lcc")


def table_d():
    T = pd.read_csv(TB / "table_D_collection_shift.csv"); c = ["crop", "ood_collection", "gain_iid", "gain_ood", "hit_iid", "hit_ood", "covered", "mace_iid", "other_acc", "dino_gap"]
    R = [[r.crop, r.ood_collection.replace("_", r"\_"), f"{r.gain_iid:+.1f}", f"{r.gain_ood:+.1f}", f(r.hit_iid), f(r.hit_ood), f(r.covered), f(r.mace_iid), f(r.other_acc), f"{r.dino_gap:.2f}"] for r in T.sort_values("gain_ood").itertuples()]
    tex(R, ["Crop", "Held-out", "Gain IID", "Gain HO", "Hit IID", "Hit HO", "Cov.", "Std", "Other", "Gap"], "table_D_summary", "Collection shift by crop (Sonnet 5.5, percent). Gain is MACE minus SAGE; Hit is true disease among four retrieved items; Cov. is the share of IID queries whose disease remains in the other-collection bank; Std and Other are IID accuracy with standard and other-collection banks; Gap is mean DINOv2 nearest-bank similarity, IID minus held-out.", "llrrrrrrrr")


def fig1():
    e = S["E1"]; fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ms = (("Sonnet", ("sage", "labels_all", "mace"), OI["blue"], "o", 0.13), ("Haiku", ("h_sage", "h_labels", "h_mace"), OI["verm"], "s", -0.13))
    for a, (sp, t) in zip(ax, SP):
        for m, cols, col, mk, off in ms:
            x = [e["accuracy"][sp][m][c]["acc"] for c in cols]; lo = [e["accuracy"][sp][m][c]["ci"][0] for c in cols]; hi = [e["accuracy"][sp][m][c]["ci"][1] for c in cols]
            y = np.arange(3) + off
            a.errorbar(x, y, xerr=[np.array(x) - lo, np.array(hi) - x], fmt=mk, ms=4, color=col, lw=0.8, capsize=0, label=m + " 5.5")
        a.set_yticks(range(3)); a.set_yticklabels(["SAGE", "Labels", "MACE"]); a.set_xlim(30, 70); a.set_title(t, fontsize=7.6, loc="left"); a.set_xlabel("Equal-crop accuracy (%)"); a.grid(axis="x", lw=0.3, color="#ddd"); a.set_axisbelow(True)
    ax[0].legend(loc="lower right", fontsize=6.4, handletextpad=0.2)
    a = ax[2]; L = {"iid": ("o", OI["ink"], "IID"), "ood": ("D", OI["gold"], "Held-out")}
    for sp, (mk, col, lb) in L.items():
        P = e["per_crop"][sp]; a.scatter([v["mace"] - v["sage"] for v in P.values()], [v["h_mace"] - v["h_sage"] for v in P.values()], s=13, marker=mk, c=col, lw=0, label=lb)
    a.plot([-10, 45], [-10, 45], lw=0.5, ls="--", color=OI["grey"]); a.axhline(0, lw=0.4, color="#bbb"); a.axvline(0, lw=0.4, color="#bbb")
    a.set_xlabel("Sonnet gain over SAGE (pts)"); a.set_ylabel("Haiku gain over SAGE (pts)"); a.set_title("Per-crop gain", fontsize=7.6, loc="left"); a.legend(fontsize=6.4, loc="upper left", handletextpad=0.1)
    fig.tight_layout(w_pad=0.6); save(fig, "fig1_model_robustness")


def fig2():
    e = S["E2"]; fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for a, (sp, t) in zip(ax, SP):
        for ret, cols, col, mk, off in (("BM25", ("labels_all", "mace"), OI["blue"], "o", -0.02), ("DINOv2", ("dino_labels", "dino_mace"), OI["green"], "s", 0.02)):
            y = [e["cells"][sp][c]["acc"] for c in cols]; lo = [e["cells"][sp][c]["ci"][0] for c in cols]; hi = [e["cells"][sp][c]["ci"][1] for c in cols]
            a.errorbar(np.arange(2) + off, y, yerr=[np.array(y) - lo, np.array(hi) - y], marker=mk, ms=4, color=col, lw=1.2, label=ret)
        a.set_xticks([0, 1]); a.set_xticklabels(["Labels", "Labels +\ncaptions"]); a.set_xlim(-0.3, 1.3); a.set_ylim(48, 68); a.set_title(t, fontsize=7.6, loc="left"); a.grid(axis="y", lw=0.3, color="#ddd")
    ax[0].set_ylabel("Equal-crop accuracy (%)"); ax[0].legend(fontsize=6.4, loc="upper left")
    P = e["per_crop"]; crops = sorted(P["iid"]); M = np.array([[P["iid"][c]["cap_bm25"], P["iid"][c]["cap_dino"], P["ood"][c]["cap_bm25"], P["ood"][c]["cap_dino"]] for c in crops])
    a = ax[2]; im = a.imshow(M, cmap="RdBu", vmin=-22, vmax=22, aspect="auto"); a.set_xticks(range(4)); a.set_xticklabels(["IID\nBM25", "IID\nDINOv2", "HO\nBM25", "HO\nDINOv2"], fontsize=6.2); a.set_yticks(range(10)); a.set_yticklabels(crops, fontsize=6.2)
    for i in range(10):
        for j in range(4):
            a.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=5.6, color="white" if abs(M[i, j]) > 13 else OI["ink"])
    a.set_title("Caption effect by crop (pts)", fontsize=7.6, loc="left"); a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
    fig.tight_layout(w_pad=0.6); save(fig, "fig2_retrieval_caption")


def fig3():
    G = pd.read_csv(TB / "fallback_calibration_grid.csv"); rule = json.loads((ROOT / "configurations/fallback_rule.json").read_text()); e = S["E3"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.3, 1]}); a = ax[0]
    for s, g in G[G.kind == "crop_quantile"].groupby("signal"):
        g = g.sort_values("cal_fallback"); chosen = s == rule["rule"]["signal"]
        a.plot(g.cal_fallback * 100, g.cal_score, lw=1.4 if chosen else 0.7, color=OI["blue"] if chosen else "#c4c4c4", zorder=3 if chosen else 1, label="dino_top1 (selected)" if chosen else None)
    a.plot([], [], lw=0.7, color="#c4c4c4", label="Other continuous signals")
    a.axhline(rule["always_mace_cal"], lw=0.8, ls="--", color=OI["verm"], label="Always MACE"); a.scatter([rule["rule"]["cal_fallback"] * 100], [rule["rule"]["cal_score"]], s=18, color=OI["blue"], zorder=4)
    a.set_xlabel("Fallback rate on calibration rows (%)"); a.set_ylabel("Calibration accuracy (%)"); a.set_title("Threshold sensitivity (calibration only)", fontsize=7.6, loc="left"); a.legend(fontsize=6.4, loc="lower left")
    a = ax[1]; nm = (("sage", "SAGE"), ("mace", "MACE"), ("adaptive", "Fallback"), ("oracle", "Oracle bound"))
    for sp, col, mk, off, lb in (("iid", OI["ink"], "o", 0.13, "IID"), ("ood", OI["gold"], "D", -0.13, "Held-out")):
        x = np.array([e["acc"][sp][c]["acc"] for c, _ in nm]); lo = np.array([e["acc"][sp][c]["ci"][0] for c, _ in nm]); hi = np.array([e["acc"][sp][c]["ci"][1] for c, _ in nm])
        a.errorbar(x, np.arange(4) + off, xerr=[x - lo, hi - x], fmt=mk, ms=3.6, color=col, lw=0.8, label=lb)
    a.set_yticks(range(4)); a.set_yticklabels([n for _, n in nm]); a.set_xlabel("Equal-crop accuracy (%)"); a.set_title("Scored once on hero queries", fontsize=7.6, loc="left"); a.legend(fontsize=6.4, loc="lower right"); a.grid(axis="x", lw=0.3, color="#ddd")
    fig.tight_layout(w_pad=0.8); save(fig, "fig3_fallback")


def fig4():
    T = pd.read_csv(TB / "table_D_collection_shift.csv").sort_values("covered"); fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.15, 1]}); a = ax[0]; y = np.arange(len(T))
    a.hlines(y, T.other_acc, T.mace_iid, lw=1.0, color="#c4c4c4"); a.scatter(T.sage_iid, y, s=13, marker="|", color=OI["ink"], label="SAGE", zorder=3); a.scatter(T.other_acc, y, s=15, color=OI["verm"], label="Other-collection bank", zorder=3); a.scatter(T.mace_iid, y, s=15, color=OI["blue"], label="Standard bank", zorder=3)
    a.set_yticks(y); a.set_yticklabels([f"{r.crop} ({r.covered:.0f}%)" for r in T.itertuples()], fontsize=6.4); a.set_xlabel("IID accuracy (%)"); a.set_title("IID, crops by disease coverage in other bank", fontsize=7.6, loc="left"); a.legend(fontsize=6.2, loc="upper left", handletextpad=0.1, borderaxespad=0.1); a.grid(axis="x", lw=0.3, color="#ddd")
    a = ax[1]; bw = T.ood_collection == "Bugwood"
    a.scatter(T[bw].dino_gap, T[bw].gain_ood, s=18, marker="o", color=OI["verm"], lw=0, label="Held-out = Bugwood"); a.scatter(T[~bw].dino_gap, T[~bw].gain_ood, s=20, marker="s", color=OI["blue"], lw=0, label="Other held-out collection"); a.axhline(0, lw=0.5, color=OI["grey"])
    off = {"Corn": (4, 4), "Soybean": (4, -7), "Tomato": (4, 3), "Rice": (4, 0), "Mango": (4, 0), "Cucumber": (3, 3), "Potato": (-27, -2), "Wheat": (3, 3), "Grape": (-7, 4), "Apple": (-6, -8)}
    for r in T.itertuples():
        a.annotate(r.crop, (r.dino_gap, r.gain_ood), xytext=off[r.crop], textcoords="offset points", fontsize=5.8)
    a.set_xlabel("DINOv2 nearest-bank similarity gap, IID minus held-out"); a.set_ylabel("Held-out MACE gain over SAGE (pts)"); a.set_title("Held-out gain and image shift (n = 10, post hoc)", fontsize=7.6, loc="left"); a.legend(fontsize=6.2, loc="upper right", handletextpad=0.1)
    fig.tight_layout(w_pad=0.8); save(fig, "fig4_collection_shift")


def save(fig, name):
    fig.savefig(FG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, dpi=600); fig.savefig(FG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=600); plt.close(fig)


if __name__ == "__main__":
    for fn in (table_a, table_b, table_c, table_d, fig1, fig2, fig3, fig4):
        fn()
