"""Knowledge base per crop: anatomy claims (disease affects organ) and symptom claims, with their source.

Priority per disease: corrected KB v1 (source-verified claims; soybean, corn, tomato), then the SAGE WACV KB
(internet.xlsx visual description), then SAGE row-level visual symptoms, then pathogen only.
"""
from __future__ import annotations

import json
import re

from . import config as C
from .data import kb_dir, sage_kb


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", str(text or "")) if len(s.strip()) > 25]


def _organs(field) -> list[str]:
    low = str(field or "").lower()
    return sorted({v for k, v in C.ORGANS.items() if k in low})


def build(crop: str, classes: list[str]) -> dict[str, dict]:
    d = kb_dir(crop)
    v1 = json.loads((d / "kb_corrected_v1.json").read_text())["classes"] if (d / "kb_corrected_v1.json").exists() else {}
    xl = {}
    if (d / "sage_wacv_internet.xlsx").exists():
        import openpyxl
        rows = list(openpyxl.load_workbook(d / "sage_wacv_internet.xlsx", read_only=True).active.iter_rows(values_only=True))
        xl = {_norm(r[0]): dict(zip(rows[0], r)) for r in rows[1:] if r[0]}
    sk = sage_kb(crop)["classes"]
    out = {}
    for c in classes:
        var = [v for v in ((sk.get(c) or {}).get("variants") or []) if v]
        if c in v1 and v1[c].get("claims"):
            e = v1[c]; organs = _organs(e.get("affected_parts_meta"))
            sym = [(x["text"], x.get("url", "")) for x in e["claims"]]; src = "kb_corrected_v1"
        elif _norm(c) in xl and xl[_norm(c)].get("Visual Description"):
            e = xl[_norm(c)]; organs = _organs(e.get("Affected Parts"))
            sym = [(s, "SAGE WACV KB internet.xlsx") for s in _sentences(e["Visual Description"])]; src = "sage_wacv_kb"
        elif any(v.get("visual_symptoms") for v in var):
            v = next(v for v in var if v.get("visual_symptoms")); organs = _organs(v.get("plant_organ"))
            sym = [(s, v.get("symptom_source") or "") for s in _sentences(v["visual_symptoms"])] or [(v["visual_symptoms"], v.get("symptom_source") or "")]
            src = "sage_rows"
        else:
            v = var[0] if var else {}; organs = _organs(v.get("plant_organ"))
            sym = [(f"{c} is caused by {v['pathogen']}.", v.get("symptom_source") or "")] if v.get("pathogen") else []
            src = "pathogen_only" if sym else "none"
        claims = [{"kind": "anatomy", "disease": c, "organ": o, "text": f"{c} affects the {o.replace('_', ' ')} of {crop.lower()}.", "source": "affected-parts field"} for o in organs]
        claims += [{"kind": "symptom", "disease": c, "organ": None, "text": t, "source": u} for t, u in sym]
        for x in claims:
            x["id"] = "c_" + hashlib_id(x["kind"], c, x["organ"], x["text"])
        out[c] = {"kb_source": src, "claims": claims}
    return out


def hashlib_id(*parts) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:10]
