"""Prompts, frozen from the runs that produced the earlier Ag-MEM results (prompts_frozen.json).
SAGE (KB only) and Ag-MEM use the identical system prompt, class list and output schema; only the evidence differs."""
from __future__ import annotations

import json
from pathlib import Path

_P = json.loads((Path(__file__).with_name("prompts_frozen.json")).read_text())
CAPTION_SYS = _P["caption_sys"]
DIAG_SYS = _P["diag_sys"]


def caption_text(crop: str) -> str:
    return _P["caption_txt"].replace("soybean", crop.lower())


def diagnosis_user(crop: str, classes: list[str], packet: str) -> str:
    return (f"Crop: {crop.lower()} (supplied).\n\n## Possible Classes\n" + "\n".join(f"- {c}" for c in sorted(classes))
            + "\n\n## Evidence packet\n" + packet + "\n\nClassify the image.")


def clean_prediction(parsed, classes: list[str]) -> dict:
    p = parsed if isinstance(parsed, dict) else {}
    pred = p.get("prediction")
    if pred not in classes:
        pred = {c.lower(): c for c in classes}.get(str(pred).lower(), "UNKNOWN")
    return {"prediction": pred, "ranked": [c for c in (p.get("ranked_candidates") or []) if c in classes],
            "cited": [str(e).strip("[]") for e in (p.get("used_evidence_ids") or [])], "confidence": p.get("confidence"),
            "organ": p.get("organ"), "explanation": p.get("explanation")}
