"""Recompute the headline numbers of the paper from results_wide.csv and check them against results.json.

  python results/verify.py        needs numpy and pandas; no images and no model calls
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = pd.read_csv(HERE / "results_wide.csv"); R = json.loads((HERE / "results.json").read_text())


def eq(df, col):
    return float(np.mean([g[col].mean() for _, g in df.groupby("crop")]) * 100)


H = W[W.hero]; bad = 0
print(f"{'Method':26s}{'IID':>8s}{'Held-out':>10s}   ({len(H)} queries, 10 crops)")
for name, col in (("SAGE", "sage"), ("DINOv2 1-NN", "dino_s0"), ("OpenCLIP 1-NN", "clip_s0"), ("Labels only (post hoc)", "labels_all"), ("DINOv2 labels (post hoc)", "dino_labels"), ("MACE retrieval", "mace")):
    a, b = eq(H[H.split == "iid"], col), eq(H[H.split == "ood"], col)
    print(f"{name:26s}{a:8.1f}{b:10.1f}")
    for sp, v in (("iid", a), ("ood", b)):
        bad += abs(v - R["accuracy"]["hero"][sp][col]["acc"]) > 1e-9
print()
for sp, lab in (("iid", "IID"), ("ood", "Held-out")):
    d = H[H.split == sp]
    print(f"MACE minus SAGE, {lab}: {eq(d, 'mace') - eq(d, 'sage'):+.1f}   MACE minus DINOv2, {lab}: {eq(d, 'mace') - eq(d, 'dino_s0'):+.1f}")
print()
print("All values match results.json." if not bad else f"{bad} values differ from results.json.")
raise SystemExit(1 if bad else 0)
