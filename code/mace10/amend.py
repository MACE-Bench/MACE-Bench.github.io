"""Jobs of the 2026-11-05 amendment (AMENDMENT_2026-11-05.md): P8 labels on the rest of the hero panel, P9 DINOv2-retrieved labels."""
from __future__ import annotations

import numpy as np

from agmem import config as C
from . import jobs as JB

PKGS = ("P8", "P9")


def top_k(E, idx, q, memory, k=C.K_RETRIEVE):
    ids = [m["image_id"] for m in memory]
    s = E[[idx[i] for i in ids]] @ E[idx[q]]
    order = sorted(range(len(ids)), key=lambda n: (-s[n], ids[n]))[:k]
    return [ids[n] for n in order], [round(float(s[n]), 6) for n in order]


def build() -> list[dict]:
    d = np.load(JB.OUT / "dinov2_embeddings.npz"); E = d["E"].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    idx = {k: i for i, k in enumerate(d["ids"])}
    Q = JB.queries(); J = []
    for r in Q[Q.hero].itertuples():
        memory = JB.mem(r.crop, 0, r.run_s0, 4)
        base = {"crop": r.crop, "image_id": r.image_id, "split": r.split, "label": r.label, "collection": r.collection, "seed": 0, "run": r.run_s0, "K": 4, "memory": memory}
        if not r.ablation:
            J.append({**base, "pkg": "P8", "condition": "labels", "variant": "labels", "job_id": f"P8|labels|s0|r{r.run_s0}|K4|{r.image_id}"})
        ids, cos = top_k(E, idx, r.image_id, memory)
        J.append({**base, "pkg": "P9", "condition": "dino_labels", "variant": "dino_labels", "retrieved_ids": ids, "retrieved_cos": cos,
                  "job_id": f"P9|dino_labels|s0|r{r.run_s0}|K4|{r.image_id}"})
    return J
