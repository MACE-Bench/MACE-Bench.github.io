"""Recompute the extension's reported contrasts from extension_wide.csv and check them against the saved statistics.

  python results/verify_extension.py        needs numpy and pandas; no images and no model calls
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent / "extension"
W = pd.read_csv(HERE / "extension_wide.csv")
S = json.loads((HERE / "statistics/extension_results.json").read_text()); B = json.loads((HERE / "statistics/robustness_posthoc.json").read_text())


def eq(d, c):
    return float(d.groupby("crop")[c].mean().mean() * 100)


def contrast(d, plus, minus):
    return sum(eq(d, c) for c in plus) - sum(eq(d, c) for c in minus)


def boot(d, plus, minus, n=2000, seed=0):
    """Paired bootstrap within crops, equal-crop mean. Row order differs from the paper's code, so the last digit can differ."""
    r = np.random.default_rng(seed); G = [g[list(dict.fromkeys(plus + minus))] for _, g in d.groupby("crop")]; out = np.zeros(n)
    for b in range(n):
        v = 0.0
        for g in G:
            m = g.iloc[r.integers(0, len(g), len(g))].mean()
            v += sum(m[c] for c in plus) - sum(m[c] for c in minus)
        out[b] = v / len(G) * 100
    return np.percentile(out, [2.5, 97.5])


iid, ood = W[W.split == "iid"], W[W.split == "ood"]; cov = iid[iid.covered == 1]
C = {c["name"]: c for k in ("E1", "E2", "E3", "E4") for c in S[k]["contrasts"]} | {c["name"]: c for c in B["matched"]["family"]}
T = [("Haiku MACE minus SAGE, IID", iid, ["h_mace"], ["h_sage"], "haiku_mace_minus_sage_iid"), ("Haiku MACE minus SAGE, held-out", ood, ["h_mace"], ["h_sage"], "haiku_mace_minus_sage_ood"),
     ("Haiku MACE minus labels, IID", iid, ["h_mace"], ["h_labels"], "haiku_mace_minus_labels_iid"), ("Haiku MACE minus labels, held-out", ood, ["h_mace"], ["h_labels"], "haiku_mace_minus_labels_ood"),
     ("Captions under DINOv2, IID", iid, ["dino_mace"], ["dino_labels"], "caption_under_dino_iid"), ("Captions under DINOv2, held-out", ood, ["dino_mace"], ["dino_labels"], "caption_under_dino_ood"),
     ("Fallback minus MACE, IID", iid, ["adaptive"], ["mace"], "adaptive_minus_mace_iid"), ("Fallback minus MACE, held-out", ood, ["adaptive"], ["mace"], "adaptive_minus_mace_ood"),
     ("Other-collection minus standard, all IID", iid, ["other_full"], ["mace"], "other_minus_standard_all"), ("Other-collection minus standard, covered", cov, ["other_full"], ["mace"], "other_minus_standard_covered"),
     ("Other-collection minus SAGE, covered", cov, ["other_full"], ["sage"], "other_minus_sage_covered"), ("Matched minus other-collection, covered", cov, ["matched"], ["other_full"], "matched_minus_other_covered"),
     ("Matched minus other-collection, all IID", iid, ["matched"], ["other_full"], "matched_minus_other_all"), ("Standard minus matched, covered", cov, ["mace"], ["matched"], "standard_minus_matched_covered")]
bad = 0
print(f"{'Contrast':44s}{'Points':>8s}   95% interval")
for lab, d, p, m, k in T:
    v = contrast(d, p, m); lo, hi = boot(d, p, m); bad += abs(v - C[k]["est"]) > 1e-9
    print(f"{lab:44s}{v:+8.1f}   [{lo:+.1f}, {hi:+.1f}]")
bw = ood[ood.collection == "Bugwood"]; v = contrast(bw, ["mace"], ["sage"]); bad += abs(v - B["subgroups"]["ood"]["bugwood_7"]["sonnet_mace_minus_sage"]["est"]) > 1e-9
print(f"{'Post hoc: Sonnet MACE minus SAGE, Bugwood crops':44s}{v:+8.1f}   [{boot(bw, ['mace'], ['sage'])[0]:+.1f}, {boot(bw, ['mace'], ['sage'])[1]:+.1f}]")
print("\nAll point estimates match the saved statistics." if not bad else f"\n{bad} point estimates do not match.")
raise SystemExit(bool(bad))
