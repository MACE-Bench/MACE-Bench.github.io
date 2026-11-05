"""Evidence packet: KB claims ranked by BM25 relevance to the test image's caption, plus (Ag-MEM only) the
K_RETRIEVE most relevant labelled memory captions. Token budget CAP_TOKENS; every disease keeps MIN_CLAIMS_PER_CLASS."""
from __future__ import annotations

import math
import re
from collections import Counter

from . import config as C

FIELDS = ("organ", "location", "lesion_shape_margin", "color", "texture", "distribution", "visible_discriminators")


def tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", str(s).lower()) if len(w) > 2]


def ntoks(s: str) -> int:
    return max(1, len(s) // 4)


def caption_text(cap: dict) -> str:
    return "; ".join(f"{k}: {cap.get(k)}" for k in FIELDS if cap.get(k))


def bm25(query: list[str], docs: dict[str, list[str]]) -> dict[str, float]:
    N = len(docs) or 1
    df = Counter(t for d in docs.values() for t in set(d))
    avg = sum(len(d) for d in docs.values()) / N if docs else 1
    q = Counter(query); out = {}
    for k, d in docs.items():
        tf = Counter(d); s = 0.0
        for t in q:
            if t in tf:
                s += math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        out[k] = s
    m = max(out.values(), default=0) or 1.0
    return {k: v / m for k, v in out.items()}


def build(crop: str, kb: dict, query_cap: dict, memory: list[dict] | None) -> tuple[str, dict]:
    """memory: list of {"id", "label_shown", "caption"}; None or [] means KB only (SAGE)."""
    claims = {x["id"]: x for c in kb.values() for x in c["claims"]}
    q = tok(caption_text(query_cap)) + tok(str(query_cap.get("organ", ""))) * 3
    rel = bm25(q, {k: tok(x["text"] + " " + (x["organ"] or "")) for k, x in claims.items()})
    order = sorted(claims, key=lambda k: (-rel[k], k))
    chosen = []
    for d in sorted(kb):
        chosen += [k for k in order if claims[k]["disease"] == d][:C.MIN_CLAIMS_PER_CLASS]
    used = sum(ntoks(claims[k]["text"]) + 8 for k in chosen)
    picked = []
    if memory:
        mrel = bm25(q, {m["id"]: tok(caption_text(m["caption"])) for m in memory})
        by = {m["id"]: m for m in memory}
        for mid in sorted(mrel, key=lambda k: (-mrel[k], k))[:C.K_RETRIEVE]:
            picked.append((by[mid], mrel[mid])); used += ntoks(caption_text(by[mid]["caption"])) + 12
    for k in order:
        if k not in chosen and used + ntoks(claims[k]["text"]) + 8 <= C.CAP_TOKENS:
            chosen.append(k); used += ntoks(claims[k]["text"]) + 8
    lines = ["### Anatomy (crop -> organ -> disease)"] + [f"[{k}] {claims[k]['text']}" for k in chosen if claims[k]["kind"] == "anatomy"]
    lines += ["### Symptom knowledge (source-supported)"] + [f"[{k}] ({claims[k]['disease']}) {claims[k]['text']}" for k in chosen if claims[k]["kind"] == "symptom"]
    if picked:
        lines.append("### Labelled reference observations (text descriptions of memory images; not source claims)")
        lines += [f"[{m['id']}] (label: {m['label_shown']}) {caption_text(m['caption'])}" for m, _ in picked]
    detail = {"n_claims": len(chosen), "est_tokens": used, "memory_ids": [m["id"] for m, _ in picked],
              "memory_labels_shown": [m["label_shown"] for m, _ in picked], "memory_labels_true": [m.get("label_true") for m, _ in picked],
              "memory_relevance": [round(r, 4) for _, r in picked]}
    return "\n".join(lines), detail
