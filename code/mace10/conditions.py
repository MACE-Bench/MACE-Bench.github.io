"""Evidence variants for the ten-crop study. Every variant uses the same model, system prompt, class list, KB and output
schema as SAGE and MACE; only the memory part of the packet (or the attached reference images) differs.

  mace        packet.build unchanged (identical requests to the pilot, so cached pilot calls are reused)
  labels      the same four retrieved items, labels only (captions withheld)
  masked      the same four retrieved captions in the same order, labels withheld
  shuffled    the same four retrieved captions, labels replaced by a locked class-level derangement
  random      four locked random captions from the same bank, true labels
  imageref    the same four retrieved items as images (label shown), captions withheld
  dino_labels labels only, of the four memory images nearest by DINOv2 instead of BM25 over captions (amendment 2026-11-05)
KB claims for every variant are the ones selected for the MACE packet of the same query, so the KB part is identical
across variants; total prompt length is not matched exactly (labels-only is shorter).
"""
from __future__ import annotations

import random

from agmem import config as C, packet

HEADERS = {
    "labels": "### Labelled reference observations (labels of the most similar memory images; descriptions withheld)",
    "dino_labels": "### Labelled reference observations (labels of the most similar memory images; descriptions withheld)",
    "masked": "### Reference observations (text descriptions of memory images; labels withheld; not source claims)",
    "shuffled": "### Labelled reference observations (text descriptions of memory images; not source claims)",
    "random": "### Labelled reference observations (text descriptions of memory images; not source claims)",
    "imageref": "### Labelled reference images (attached after the test image, in this order; not source claims)",
}


def derangement(classes: list[str], seed: str) -> dict[str, str]:
    """Locked class-level derangement: no class maps to itself."""
    cl = sorted(classes); rng = random.Random(seed)
    while True:
        p = cl[:]; rng.shuffle(p)
        if all(a != b for a, b in zip(cl, p)):
            return dict(zip(cl, p))


def build(variant: str, crop: str, kb: dict, query_cap: dict, memory: list[dict], query_id: str, classes: list[str], picked_ids: list[str] | None = None):
    """Returns (packet_text, detail, ref_image_ids). memory: [{"id","label_shown","label_true","caption"}]."""
    text, det = packet.build(crop, kb, query_cap, memory)
    if variant == "mace":
        return text, det, []
    kb_part = text.split("### Labelled reference observations")[0].rstrip("\n")
    by = {m["id"]: m for m in memory}
    if variant == "random":
        rng = random.Random(f"random|{crop}|{query_id}")
        picked = rng.sample(sorted(by), min(C.K_RETRIEVE, len(by)))
    elif picked_ids is not None:
        picked = ["m_" + i for i in picked_ids]
    else:
        picked = det["memory_ids"]
    perm = derangement(classes, f"shuffle|{crop}") if variant == "shuffled" else None
    lines = [HEADERS[variant]]; shown = []
    for n, mid in enumerate(picked, 1):
        m = by[mid]; lab = perm.get(m["label_true"], m["label_true"]) if perm else m["label_true"]
        cap = packet.caption_text(m["caption"])
        if variant in ("labels", "dino_labels"):
            lines.append(f"[{mid}] (label: {lab})"); shown.append(lab)
        elif variant == "masked":
            lines.append(f"[{mid}] {cap}"); shown.append(None)
        elif variant == "imageref":
            lines.append(f"[{mid}] reference image {n} (label: {lab})"); shown.append(lab)
        else:
            lines.append(f"[{mid}] (label: {lab}) {cap}"); shown.append(lab)
    det = {**det, "variant": variant, "memory_ids": list(picked), "memory_labels_shown": shown,
           "memory_labels_true": [by[m]["label_true"] for m in picked]}
    refs = [mid[2:] for mid in picked] if variant == "imageref" else []
    return kb_part + "\n" + "\n".join(lines), det, refs
