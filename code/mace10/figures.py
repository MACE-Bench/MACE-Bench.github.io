"""Result figures for the results-first MACE draft (no model calls). Measured series only; pending series are drawn as
dashed empty slots and named "needs run" in the legend, never as values.

  python -m mace10.figures      writes <paper>/figures/{crops,memsize,reliability}.pdf
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .paper import OUT, PAPER, pilot  # noqa: E402

RES = json.loads((OUT / "results.json").read_text())
A = RES["accuracy"]

OI = {"blue": "#0072B2", "green": "#009E73", "verm": "#D55E00", "pink": "#CC79A7", "gold": "#E69F00", "sky": "#56B4E9", "grey": "#8a8a8a"}
import matplotlib.patheffects as _pe  # noqa: E402
plt.rcParams["path.effects"] = [_pe.Normal()]   # draw text as outlines: AAAI bans Identity-H and Type 3 fonts
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def bars(ax, x, vals, ns, w, col, label):
    lo, hi = zip(*[wilson(round(v * n / 100), n) for v, n in zip(vals, ns)])
    ax.bar(x, vals, w * 0.9, color=col, lw=0, label=label)
    ax.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor="#333", lw=0.5, capsize=0.8)


def pending(ax, x, w, top):
    for xx in np.atleast_1d(x):
        ax.add_patch(Rectangle((xx - w * 0.45, 0), w * 0.9, top, fill=False, ls=(0, (2, 1.5)), lw=0.5, ec="#b0b0b0"))


def save(fig, name):
    fig.savefig(PAPER / f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.02, dpi=700); plt.close(fig)


def crops(L):
    h = L[(L.panel == "hero") & L.hero & (L.seed == 0) & (L.K == 4)]
    order = list(dict.fromkeys(h.crop))
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 2.6), sharex=True)
    w = 0.2; x = np.arange(len(order))
    for ax, sp, title in zip(axs, ("iid", "ood"), ("IID", "Held-out collection")):
        g = h[h.split == sp].groupby("crop", sort=False)
        n = g.size().reindex(order).values
        pc = RES["per_crop"]; nn = [pc[c][sp]["n"] for c in order]
        bars(ax, x - 1.5 * w, [pc[c][sp]["sage"] for c in order], nn, w, OI["verm"], "SAGE")
        bars(ax, x + 1.5 * w, [pc[c][sp]["mace"] for c in order], nn, w, OI["blue"], "MACE retrieval")
        bars(ax, x - 0.5 * w, (100 * g.clip_correct.mean()).reindex(order).values, n, w, OI["sky"], "OpenCLIP 1-NN")
        bars(ax, x + 0.5 * w, (100 * g.dino_correct.mean()).reindex(order).values, n, w, OI["grey"], "DINOv2 1-NN")
        ax.set_ylim(0, 100); ax.set_ylabel("Accuracy (%)"); ax.grid(axis="y", color="#ececec", lw=0.5); ax.set_axisbelow(True)
        ax.text(0.003, 0.97, title, transform=ax.transAxes, fontweight="bold", va="top", fontsize=7.5)
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"{c}" for c in order])
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=6.8)
    fig.tight_layout(pad=0.2, h_pad=0.4); fig.subplots_adjust(top=0.9); save(fig, "crops")


def memsize(L):
    """a: soybean pilot (different protocol). b: ten-crop ablation panel. Solid IID, dashed held-out."""
    a_ = L[(L.panel == "hero") & L.ablation & (L.seed == 0)]
    P, _ = pilot(); R = P[P.condition == "agmem"]; sage = P[P.condition == "sage"].set_index("image_id").correct
    Lp = L[L.panel == "pilot"]
    Ks = [1, 4, 16]; xs = np.log2(Ks)
    fig, (pa, pb) = plt.subplots(1, 2, figsize=(3.4, 1.85), sharey=True)
    for sp, spp, ls in (("iid", "test_iid", "-"), ("ood", "test_ood_in_scope", "--")):
        pa.plot(xs, [100 * R[(R.K == K) & (R.split == spp)].correct.mean() for K in Ks], ls, marker="s", color=OI["blue"], ms=3, lw=1.2)
        pa.plot(xs, [100 * Lp[(Lp.K == K) & (Lp.split == sp)].dino_correct.mean() for K in Ks], ls, marker="o", color=OI["grey"], ms=3, lw=1.0)
        pa.plot(xs, [100 * R[(R.K == 4) & (R.split == spp)].image_id.map(sage).mean()] * 3, ls, color=OI["verm"], lw=0.9)
        for key, col in (("clip_correct", OI["sky"]), ("dino_correct", OI["grey"])):
            pb.plot(xs, [100 * a_[(a_.K == K) & (a_.split == sp)].groupby("crop")[key].mean().mean() for K in Ks], ls, marker="o", color=col, ms=3, lw=1.0)
    ab = lambda sp, m: A["ablation"][sp][m]["acc"]
    for sp, ls in (("iid", "-"), ("ood", "--")):
        pb.plot(xs, [ab(sp, "mace_K1"), ab(sp, "mace"), ab(sp, "mace_K16")], ls, marker="s", color=OI["blue"], ms=3, lw=1.2)
        pb.plot(xs, [ab(sp, "sage")] * 3, ls, color=OI["verm"], lw=0.9)
    pa.set_title("a  Soybean pilot", loc="left", fontsize=7.2, fontweight="bold", pad=2)
    pb.set_title("b  Ten crops (400 queries)", loc="left", fontsize=7.2, fontweight="bold", pad=2)
    for ax in (pa, pb):
        ax.set_xticks(xs); ax.set_xticklabels(Ks); ax.grid(axis="y", color="#ececec", lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0, 70)
    pa.set_ylabel("Accuracy (%)"); fig.supxlabel("Memory images per disease ($K$)", fontsize=7.2, y=0.0)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=OI["blue"], marker="s", ms=3, lw=1.2), Line2D([], [], color=OI["verm"], lw=0.9),
          Line2D([], [], color=OI["sky"], marker="o", ms=3), Line2D([], [], color=OI["grey"], marker="o", ms=3),
          Line2D([], [], color="k", lw=0.9), Line2D([], [], color="k", lw=0.9, ls="--")]
    ls_ = ["MACE retrieval", "SAGE", "OpenCLIP 1-NN", "DINOv2 1-NN", "IID", "Held-out"]
    fig.legend(hs, ls_, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.17), fontsize=5.6, columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(pad=0.2, w_pad=0.5); save(fig, "memsize")


def reliability(L):
    h = L[(L.panel == "hero") & L.hero & (L.K == 4)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.8), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (key, col, lab) in enumerate((("clip_correct", OI["sky"], "OpenCLIP 1-NN"), ("dino_correct", OI["grey"], "DINOv2 1-NN"))):
        for j, sp in enumerate(("iid", "ood")):
            v = [100 * h[(h.seed == s) & (h.split == sp)].groupby("crop")[key].mean().mean() for s in (0, 1, 2)]
            xx = j * 2.2 + i * 0.7 + np.array([-0.18, 0, 0.18])
            a.plot(xx, v, "o", color=col, ms=3.2, label=lab if j == 0 else None)
            a.plot([xx[0] - 0.05, xx[-1] + 0.05], [np.mean(v)] * 2, color=col, lw=1.0)
    for j, sp in enumerate(("iid", "ood")):
        v = RES["selection"][sp]["mace"]["values"]; xx = j * 2.2 + 1.4 + np.array([-0.18, 0, 0.18])
        a.plot(xx, v, "s", color=OI["blue"], ms=3.2, label="MACE retrieval" if j == 0 else None)
        a.plot([xx[0] - 0.05, xx[-1] + 0.05], [np.mean(v)] * 2, color=OI["blue"], lw=1.0)
    a.set_xticks([0.7, 2.9]); a.set_xticklabels(["IID", "Held-out"]); a.set_ylim(0, 70); a.set_ylabel("Accuracy (%)")
    a.set_title("a  Memory selection", loc="left", fontsize=7.2, fontweight="bold", pad=2)
    a.legend(fontsize=5.6, loc="lower left", ncol=1)
    w = 0.2; x = np.arange(2); H = RES["hit_miss"]
    for k, (sub, key, col, lab) in enumerate((("hit", "sage", "#e8a77f", "SAGE, hit"), ("hit", "mace", OI["blue"], "MACE, hit"),
                                               ("miss", "sage", OI["verm"], "SAGE, miss"), ("miss", "mace", "#9db9d5", "MACE, miss"))):
        bars(b, x + (k - 1.5) * w, [H[sp][sub][key] for sp in ("iid", "ood")], [H[sp][sub]["n"] for sp in ("iid", "ood")], w, col, lab)
    b.set_xticks(x); b.set_xticklabels(["IID", "Held-out"]); b.set_ylim(0, 100)
    b.set_title("b  Hit / miss (ten crops)", loc="left", fontsize=7.2, fontweight="bold", pad=2)
    b.legend(fontsize=5.4, ncol=2, loc="upper left", columnspacing=0.6, handlelength=1.0)
    for ax in (a, b):
        ax.grid(axis="y", color="#ececec", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2, w_pad=0.6); save(fig, "reliability")


def cost():
    """Five-crop matched panel: accuracy against USD per query, log x."""
    M = RES["efficiency"]["methods"]; names = {"sage": "SAGE", "labels": "Labels only", "mace": "MACE retrieval", "imageref": "Image references", "agentic": "MACE agentic"}
    cols = {"sage": OI["verm"], "labels": OI["gold"], "mace": OI["blue"], "imageref": OI["pink"], "agentic": OI["green"]}
    n = RES["efficiency"]["n"]; fig, ax = plt.subplots(figsize=(3.3, 1.75))
    for m, v in M.items():
        lo, hi = wilson(round(v["acc_all"] * n / 100), n)
        ax.errorbar(v["usd"], v["acc_all"], yerr=[[v["acc_all"] - lo], [hi - v["acc_all"]]], fmt="o", color=cols[m], ms=4, lw=0.7, capsize=1.2)
        dx, ha = {"labels": (-5, "right"), "sage": (-5, "right")}.get(m, (5, "left"))
        ax.annotate(names[m], (v["usd"], v["acc_all"]), xytext=(dx, 0), textcoords="offset points", fontsize=6.2, color=cols[m], ha=ha, va="center")
    ax.set_xscale("log"); ax.set_xlim(0.0095, 0.14); ax.set_xticks([0.01, 0.02, 0.05, 0.1]); ax.set_xticklabels(["0.01", "0.02", "0.05", "0.10"]); ax.minorticks_off()
    ax.set_xlabel("USD per query (list-price estimate)"); ax.set_ylabel("Accuracy (%)"); ax.grid(color="#ececec", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2); save(fig, "cost10")


# One light palette for every figure: gray baseline, amber lookup and labels, blue captions, green MACE.
C_SAGE, C_DINO, C_MACE, C_CLIP, C_CAP, C_IMG, C_AGENT = "#d4d4d4", "#f6cf8c", "#8ed4b5", "#fae8c6", "#b5daf2", "#f0c4dc", "#a9c9ea"
EDGE = "#3a3a3a"


EXT = OUT.parent.parent / "experiments/mace_extension"
C_HSAGE, C_HMACE = "#6e6e6e", "#1f7a5c"   # Haiku: darker step of the Sonnet hue


def both():
    """Hero rows with the Sonnet and Haiku columns; Haiku ran SAGE, MACE and labels only."""
    return pd.read_csv(EXT / "processed_results/hero_all_methods.csv")


def frame(ax, axis="y"):
    """bioMoR axes: full thin frame, dotted grid."""
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    ax.grid(axis=axis, color="#cccccc", lw=0.4, ls=":"); ax.set_axisbelow(True)


def title(ax, t):
    ax.set_title(t, fontsize=7.2, pad=3)


def bar(ax, x, v, w, col, label=None, err=None, text=False, fs=5.6):
    ax.bar(x, v, w, color=col, edgecolor=EDGE, linewidth=0.5, label=label)
    if err is not None:
        ax.errorbar(x, v, yerr=err, fmt="none", ecolor=EDGE, elinewidth=0.55, capsize=1.5, capthick=0.55)
    if text:
        top = np.array(v, float) + (np.array(err[1], float) if err is not None else 0)
        for xx, tt, vv in zip(np.atleast_1d(x), np.atleast_1d(top), np.atleast_1d(v)):
            ax.text(xx, tt + 1.2, f"{vv:.0f}", ha="center", va="bottom", fontsize=fs)


def mace_rows():
    P = pd.DataFrame(map(json.loads, (OUT / "P1/predictions.jsonl").open())); M = P[P.condition == "mace"].copy()
    M["sage"] = M.image_id.map(P[P.condition == "sage"].set_index("image_id").correct).astype(bool); M["correct"] = M.correct.astype(bool)
    M["n_true"] = [sum(l == r.label for l in r.memory_labels_shown) for r in M.itertuples()]
    return M


def crop_bars():
    """Figure 2: MACE gain over SAGE per crop for both VLMs, grouped by the held-out source; one real image per crop."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.image as mpimg
    H = both(); o = H[H.split == "ood"]; coll = o.groupby("crop").collection.first()
    gain = (o.groupby("crop").mace.mean() - o.groupby("crop").sage.mean()).mul(100)
    bug = gain[coll == "Bugwood"].sort_values(ascending=False).index.tolist(); rest = gain[coll != "Bugwood"].sort_values().index.tolist(); order = bug + rest
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 2.1), sharex=True); x = np.arange(len(order)); w = 0.36
    for ax, sp, t in zip(axs, ("iid", "ood"), ("(a) In distribution (IID)", "(b) Held-out collection")):
        for k, (m, b, c, lab) in enumerate((("mace", "sage", C_MACE, "Sonnet 5.5"), ("h_mace", "h_sage", C_HMACE, "Haiku 5.5"))):
            d = H[H.split == sp].assign(g=lambda q: (q[m] - q[b]) * 100).groupby("crop").g
            v = d.mean().reindex(order).values; e = 1.96 * (d.std() / np.sqrt(d.size())).reindex(order).values
            ax.bar(x + (k - 0.5) * w, v, w, color=c, edgecolor=EDGE, linewidth=0.5, label=lab)
            ax.errorbar(x + (k - 0.5) * w, v, yerr=e, fmt="none", ecolor=EDGE, elinewidth=0.55, capsize=1.3, capthick=0.55)
            for xx, vv, ee in zip(x + (k - 0.5) * w, v, e):
                ax.text(xx, vv + ee + 1 if vv >= 0 else vv - ee - 1, f"{vv:+.0f}" if abs(vv) >= 0.5 else "0", ha="center", va="bottom" if vv >= 0 else "top", fontsize=5.2)
        ax.axhline(0, color=EDGE, lw=0.6); ax.axvline(len(bug) - 0.5, color=EDGE, lw=0.7, ls="--"); ax.set_ylim(-32, 68); ax.set_yticks([-25, 0, 25, 50]); ax.set_ylabel("Gain (points)"); frame(ax); title(ax, t)
    axs[1].text(len(bug) / 2 - 0.5, 60, "Seven crops that hold out Bugwood", ha="center", va="top", fontsize=6.2)
    axs[1].text(len(bug), 60, "Other sources", ha="center", va="top", fontsize=6.2)
    axs[1].set_xticks(x); axs[1].set_xticklabels(order); axs[1].set_xlim(-0.55, len(order) - 0.45); axs[1].tick_params(axis="x", pad=31, length=0)
    for xx, crop in zip(x, order):
        im = OffsetImage(mpimg.imread(PAPER / f"figures/img/crop_{crop}.jpg"), zoom=0.046, resample=True)
        axs[1].add_artist(AnnotationBbox(im, (xx, 0), xybox=(0, -14.5), xycoords=("data", "axes fraction"), boxcoords="offset points", frameon=False, pad=0, annotation_clip=False))
    axs[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.36, 1.0), fontsize=6.2, handlelength=1.2, columnspacing=1.0, borderaxespad=0.25)
    fig.tight_layout(pad=0.2, h_pad=0.5); save(fig, "fig2_crops")


def evidence_bars():
    """Figure 3: which evidence matters; the bioMoR ablation layout on the locked 400 queries, all six conditions."""
    fig, ax = plt.subplots(figsize=(3.35, 1.45)); Ab = A["ablation"]; x = np.arange(2); w = 0.135
    conds = (("shuffled", "#e9e9e9", "Shuffled labels"), ("sage", C_SAGE, "No memory (SAGE)"), ("random", "#ececdc", "Random references"), ("masked", C_CAP, "Captions only"), ("labels", C_DINO, "Labels only"), ("mace", C_MACE, "Both (MACE)"))
    for k, (m, col, lab) in enumerate(conds):
        v = [Ab[sp][m]["acc"] for sp in ("iid", "ood")]; lo = [Ab[sp][m]["acc"] - Ab[sp][m]["ci"][0] for sp in ("iid", "ood")]; hi = [Ab[sp][m]["ci"][1] - Ab[sp][m]["acc"] for sp in ("iid", "ood")]
        bar(ax, x + (k - 2.5) * w, v, w, col, lab, [lo, hi], text=True, fs=5.4)
    ax.set_xticks(x); ax.set_xticklabels(["IID", "Held-out"]); ax.set_ylim(0, 104); ax.set_ylabel("Accuracy (%)"); frame(ax); title(ax, "(a) Evidence controls, Sonnet")
    ax.legend(ncol=3, fontsize=5.3, loc="upper center", handlelength=1.1, columnspacing=0.7, borderaxespad=0.2, labelspacing=0.2, handletextpad=0.3)
    fig.tight_layout(pad=0.2); save(fig, "fig_evidence")


def heat():
    """Per-crop post hoc contrasts on all 1,600 queries, in points: caption text (MACE minus labels) and caption retrieval (BM25 minus DINOv2 labels)."""
    from matplotlib.colors import LinearSegmentedColormap
    H = both()
    g = H.groupby(["crop", "split"])[["mace", "labels_all", "dino_labels", "h_mace", "h_labels"]].mean().mul(100)
    g["text"] = g.mace - g.labels_all; g["retr"] = g.labels_all - g.dino_labels; g["htext"] = g.h_mace - g.h_labels
    T = pd.DataFrame({"Text, IID": g.text.xs("iid", level=1), "Text, held-out": g.text.xs("ood", level=1), "Haiku, IID": g.htext.xs("iid", level=1), "Haiku, held-out": g.htext.xs("ood", level=1), "Retrieval, IID": g.retr.xs("iid", level=1), "Retrieval, held-out": g.retr.xs("ood", level=1)}).sort_values("Text, IID", ascending=False)
    cm = LinearSegmentedColormap.from_list("mace", ["#f6b9a0", "#ffffff", "#8ed4b5"])
    fig, ax = plt.subplots(figsize=(3.35, 1.45)); ax.imshow(T.values, cmap=cm, vmin=-22, vmax=22, aspect="auto", interpolation="nearest")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax.text(j, i, f"{T.values[i, j]:+.0f}" if abs(T.values[i, j]) >= 0.5 else "0", ha="center", va="center", fontsize=5.8)
    ax.set_xticks(range(6)); ax.set_xticklabels(["IID", "Held-out"] * 3, fontsize=5.6); ax.set_yticks(range(T.shape[0])); ax.set_yticklabels(T.index, fontsize=6.4); ax.tick_params(length=0)
    ax.xaxis.tick_top(); ax.axvline(1.5, color=EDGE, lw=0.8); ax.axvline(3.5, color=EDGE, lw=0.8)
    for xx, t in ((0.5, "Text, Sonnet"), (2.5, "Text, Haiku"), (4.5, "Retrieval, Sonnet")):
        ax.text(xx, -1.72, t, ha="center", fontsize=6.2)
    ax.text(-1.05, -1.72, "(b)", ha="center", fontsize=7.2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    fig.tight_layout(pad=0.2); save(fig, "fig_heat")


def mechanism_panels():
    """Figure 4 c, d: what memory did to single answers, and accuracy by how many retrieved items carry the true disease."""
    M = mace_rows(); B = both().set_index("image_id"); M["h_correct"] = M.image_id.map(B.h_mace).astype(bool); M["h_sage"] = M.image_id.map(B.h_sage).astype(bool)
    fig, (c, d, e) = plt.subplots(1, 3, figsize=(7.0, 1.38), gridspec_kw={"width_ratios": [1.3, 1, 1]}, sharey=False)
    parts = (("Both right", lambda a, b: a & b, "#d9efe4"), ("SAGE wrong, MACE right", lambda a, b: a & ~b, C_MACE), ("SAGE right, MACE wrong", lambda a, b: ~a & b, "#f6b9a0"), ("Both wrong", lambda a, b: ~a & ~b, C_SAGE))
    rowsC = (("ood", "h_correct", "h_sage", "Haiku, held-out"), ("iid", "h_correct", "h_sage", "Haiku, IID"), ("ood", "correct", "sage", "Sonnet, held-out"), ("iid", "correct", "sage", "Sonnet, IID"))
    for y, (sp, a_, b_, lab) in enumerate(rowsC):
        g = M[M.split == sp]; left = 0
        for name, f, col in parts:
            v = 100 * f(g[a_], g[b_]).mean(); c.barh(y, v, 0.72, left=left, color=col, edgecolor=EDGE, linewidth=0.5, label=name if y == 0 else None)
            c.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=5.4); left += v
    c.set_yticks(range(4)); c.set_yticklabels([r[3] for r in rowsC], fontsize=5.8); c.set_xlim(0, 100); c.set_ylim(-0.5, 3.5); frame(c, "x"); title(c, "(c) Answer outcomes (% of queries)")
    c.legend(ncol=2, fontsize=5.0, loc="upper center", bbox_to_anchor=(0.42, -0.2), handlelength=1.0, columnspacing=0.7, handletextpad=0.3, labelspacing=0.15, frameon=False)
    for ax, a_, b_, t in ((d, "correct", "sage", "(d) Accuracy by evidence, Sonnet"), (e, "h_correct", "h_sage", "(e) Accuracy by evidence, Haiku")):
        for sp, ls, tag in (("iid", "-", "IID"), ("ood", "--", "held-out")):
            g = M[M.split == sp].groupby("n_true")[[a_, b_]].mean().mul(100)
            ax.plot(g.index, g[a_], ls, marker="s", ms=3, lw=1.2, color="#3fae83", label=f"MACE, {tag}"); ax.plot(g.index, g[b_], ls, marker="o", ms=2.6, lw=1.0, color="#8a8a8a", label=f"SAGE, {tag}")
        ax.set_xticks(range(5)); ax.set_xlabel("Retrieved items with the true disease", labelpad=1, fontsize=6.2); ax.set_ylim(0, 104); frame(ax); title(ax, t)
    d.set_ylabel("Accuracy (%)"); e.legend(fontsize=4.6, loc="upper left", ncol=1, handlelength=1.3, labelspacing=0.1, borderaxespad=0.2)
    fig.tight_layout(pad=0.2, w_pad=0.7); save(fig, "fig3_panels")


def robustness():
    """Figure 5: memory size, memory selection, memory source, and retrieval against gain."""
    fig, (m, a, c, b) = plt.subplots(1, 4, figsize=(7.0, 1.3), gridspec_kw={"width_ratios": [1.15, 1.15, 0.72, 1.05]})
    Ks = [1, 4, 16]; xs = np.log2(Ks); ab = lambda sp, k: A["ablation"][sp][k]["acc"]
    for sp, ls, tag in (("iid", "-", "IID"), ("ood", "--", "held-out")):
        m.plot(xs, [ab(sp, "mace_K1"), ab(sp, "mace"), ab(sp, "mace_K16")], ls, marker="s", color="#3fae83", ms=3, lw=1.2, label=f"MACE, {tag}")
        m.plot(xs, [ab(sp, "dino_K1"), ab(sp, "dino_s0"), ab(sp, "dino_K16")], ls, marker="o", color="#e0a93f", ms=2.6, lw=1.0, label=f"DINOv2, {tag}")
    m.set_xticks(xs); m.set_xticklabels(Ks); m.set_xlabel("$K$ per disease", labelpad=1); m.set_ylabel("Accuracy (%)"); m.set_ylim(0, 82); frame(m); title(m, "(a) Memory size, Sonnet")
    m.legend(fontsize=4.6, loc="lower right", ncol=2, handlelength=1.5, labelspacing=0.1, columnspacing=0.5, borderaxespad=0.2)
    S = RES["selection"]; x = np.arange(2); w = 0.26
    for k, (mm, col, lab) in enumerate((("clip", C_CLIP, "OpenCLIP"), ("dino", C_DINO, "DINOv2"), ("mace", C_MACE, "MACE"))):
        v = [np.mean(S[sp][mm]["values"]) for sp in ("iid", "ood")]; e = [S[sp][mm]["sd"] for sp in ("iid", "ood")]
        bar(a, x + (k - 1) * w, v, w, col, lab, [e, e], text=True, fs=5.6)
    a.set_xticks(x); a.set_xticklabels(["IID", "Held-out"]); a.set_ylim(0, 92); frame(a); a.legend(fontsize=5.2, loc="upper right", ncol=3, handlelength=0.9, columnspacing=0.5, handletextpad=0.25, borderaxespad=0.15); title(a, "(b) Memory selection, Sonnet")
    Sr = A["source"]["iid"]; ms = (("sage", "SAGE", C_SAGE), ("other_source", "Other", C_CAP), ("same_source", "Same", C_MACE))
    for i, (k, lab, col) in enumerate(ms):
        v = Sr[k]; bar(c, [i], [v["acc"]], 0.62, col, None, [[v["acc"] - v["ci"][0]], [v["ci"][1] - v["acc"]]], text=True, fs=5.6)
    c.set_xticks(range(3)); c.set_xticklabels([l for _, l, _ in ms], fontsize=6); c.set_ylim(0, 92); frame(c); title(c, "(c) Source, Sonnet")
    g = both().groupby(["crop", "split"])[["mace", "sage", "h_mace", "h_sage", "hit"]].mean().mul(100).reset_index(); g["gain"] = g.mace - g.sage; g["hgain"] = g.h_mace - g.h_sage
    for sp, col, mk, lab in (("iid", C_MACE, "o", "IID"), ("ood", C_DINO, "D", "Held-out")):
        d = g[g.split == sp]; b.scatter(d.hit, d.gain, s=14, color=col, marker=mk, edgecolor=EDGE, linewidth=0.45, label=f"Sonnet, {lab}", zorder=3)
        b.scatter(d.hit, d.hgain, s=14, facecolor="none", marker=mk, edgecolor=col, linewidth=0.8, label=f"Haiku, {lab}", zorder=3)
    xx = np.array([33, 102]); b.plot(xx, np.polyval(np.polyfit(g.hit, g.gain, 1), xx), color="#777", lw=0.8, ls="--"); b.plot(xx, np.polyval(np.polyfit(g.hit, g.hgain, 1), xx), color="#777", lw=0.8, ls=":"); b.axhline(0, color=EDGE, lw=0.5)
    b.set_xlim(30, 104); b.set_xlabel("True disease retrieved (%)", labelpad=1); b.set_ylabel("Gain (points)", labelpad=1); frame(b, "both"); b.set_ylim(-14, 58); b.legend(fontsize=4.6, loc="upper left", ncol=2, handletextpad=0.05, borderaxespad=0.15, columnspacing=0.3, labelspacing=0.1); title(b, "(d) Retrieval and gain over SAGE")
    fig.tight_layout(pad=0.2, w_pad=0.6); save(fig, "fig4_stability_cost")


def cost_fig():
    """Figure 6: accuracy and cost on 200 matched queries."""
    M = dict(RES["efficiency"]["methods"]); order = ("sage", "labels", "mace", "imageref", "agentic", "h_sage", "h_labels", "h_mace")
    Hk = json.loads((EXT / "statistics/haiku_all.json").read_text())
    led = [json.loads(l) for l in open(OUT.parent.parent / "runs/cost_ledger.jsonl")]; caps = [r for r in led if r["stage"] == "caption" and not r.get("cache_hit")]
    cap_usd = float(np.mean([r["cost_usd"] for r in caps])); cap_s = float(np.median([r["elapsed_s"] for r in caps if r.get("elapsed_s")]))
    for m in ("sage", "labels", "mace"):   # Haiku diagnosis plus the Sonnet caption it reads, the same definition as the Sonnet rows
        M["h_" + m] = {"acc_all": Hk["efficiency"]["all"]["raw"]["h_" + m], "usd": Hk["price_hero"][m] + cap_usd, "latency_s": Hk["efficiency"]["diag_latency_median"][m] + cap_s}
    names = {"sage": "SAGE", "labels": "Labels only", "mace": "MACE retrieval", "imageref": "Image references", "agentic": "MACE agentic", "h_sage": "Haiku SAGE", "h_labels": "Haiku labels only", "h_mace": "Haiku MACE"}
    short = {"sage": "SAGE", "labels": "Labels", "mace": "MACE", "imageref": "Images", "agentic": "Agent", "h_sage": "H-SAGE", "h_labels": "H-Labels", "h_mace": "H-MACE"}
    cols = {"sage": C_SAGE, "labels": C_DINO, "mace": C_MACE, "imageref": C_IMG, "agentic": C_AGENT, "h_sage": C_HSAGE, "h_labels": "#b98a2f", "h_mace": C_HMACE}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 1.28), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fr = [M[k] for k in ("h_mace", "labels", "mace")]
    a.plot([v["usd"] for v in fr], [v["acc_all"] for v in fr], color="#3fae83", lw=1.0, zorder=2)
    for k in order:
        v = M[k]; a.scatter(v["usd"], v["acc_all"], s=44, color=cols[k], edgecolor=EDGE, linewidth=0.5, zorder=3)
        dx, dy, ha = {"labels": (-6, 6, "right"), "sage": (6, -1, "left"), "imageref": (6, -5, "left"), "agentic": (-6, 6, "right"), "h_sage": (6, 0, "left"), "h_labels": (6, 0, "left"), "h_mace": (6, -6, "left")}.get(k, (-6, 4, "right"))
        a.annotate(names[k], (v["usd"], v["acc_all"]), xytext=(dx, dy), textcoords="offset points", fontsize=6, ha=ha, va="center")
    a.set_xscale("log"); a.set_xlim(0.0042, 0.13); a.set_xticks([0.005, 0.01, 0.03, 0.1]); a.set_xticklabels(["0.005", "0.01", "0.03", "0.10"]); a.minorticks_off(); a.set_ylim(28, 66); frame(a, "both")
    a.set_xlabel("USD per query (list price)", labelpad=1); a.set_ylabel("Accuracy (%)"); title(a, "(a) Accuracy against price")
    for ax, f, yl, t, ff in ((b, lambda v: v["usd"] / (v["acc_all"] / 100), "USD", "(b) USD per correct diagnosis", "{:.3f}"), (c, lambda v: v["latency_s"], "Seconds", "(c) Median latency per query", "{:.0f}")):
        vals = [f(M[k]) for k in order]; ax.bar(range(8), vals, 0.72, color=[cols[k] for k in order], edgecolor=EDGE, linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, ff.format(v).lstrip("0"), ha="center", va="bottom", fontsize=4.8)
        ax.set_xticks(range(8)); ax.set_xticklabels([short[k] for k in order], fontsize=5.2, rotation=40, ha="right", rotation_mode="anchor"); ax.set_ylim(0, max(vals) * 1.17); ax.set_ylabel(yl); frame(ax); title(ax, t)
    fig.tight_layout(pad=0.2, w_pad=0.8); save(fig, "fig_cost")


def main():
    crop_bars(); evidence_bars(); heat(); mechanism_panels(); robustness(); cost_fig(); print("figures written")


if __name__ == "__main__":
    main()
