"""RUN_MANIFEST.json and an extension budget ledger from saved run files; no model call, no secrets.

  python -m experiments.mace_extension.scripts.manifest
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .analyze import CFG, RAW, ROOT, ST

REPO = ROOT.parent.parent
FINAL = {"E1_haiku": "confirmatory", "E2": "confirmatory", "E3cal": "calibration", "E4other": "confirmatory", "E4family": "confirmatory", "E4matched": "confirmatory, added after E4 results"}
SMOKE = ("E1_haiku_smoke", "E1_opus_smoke", "E2_smoke")
START_LEDGER, SOFT, HARD = 657.35, 822.82, 872.82


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def main():
    runs = []
    for name in list(FINAL) + list(SMOKE):
        d = RAW / name; s = json.loads((d / "status.json").read_text()); rows = [json.loads(l) for l in (d / "predictions.jsonl").open()]
        t = [r["t"] for r in rows if r.get("t")]
        runs.append({"name": name, "use": FINAL.get(name, "smoke, excluded from all results"), "rows": len(rows), "jobs_selected": s["selected"], "complete": s["complete"], "stop": s["stop"], "failures_file": (d / "failures.jsonl").exists(),
                     "model_ids": s["model_ids"], "conditions": sorted({r["condition"] for r in rows}), "new_calls": s["new_calls"], "cache_hits": s["cache_hits"], "incurred_usd": s["incurred_usd"], "ledger_after_usd": s["ledger_total"], "unknown_predictions": s["unknown_predictions"],
                     "first_row_utc": datetime.fromtimestamp(min(t), timezone.utc).isoformat() if t else None, "last_row_utc": datetime.fromtimestamp(max(t), timezone.utc).isoformat() if t else None, "predictions_sha256": sha(d / "predictions.jsonl")})
    spent = round(sum(r["incurred_usd"] for r in runs), 4)
    M = {"generated_utc": datetime.now(timezone.utc).isoformat(), "code_base_commit": git("rev-parse", "HEAD"), "uncommitted_paths": git("status", "--porcelain").splitlines(), "python": sys.version.split()[0], "platform": platform.platform(),
         "access": "Claude Code CLI headless login; no API key stored or logged", "fixed": {"seed": 0, "K": 4, "retrieved_items": 4, "bootstrap_B": 2000, "benchmark": "frozen release v2.1", "captions": "frozen Sonnet 5.5 captions", "sampling": "CLI does not expose temperature"},
         "budget": {"ledger_start_usd": START_LEDGER, "ledger_end_usd": max(r["ledger_after_usd"] for r in runs), "extension_usd": spent, "soft_stop_usd": SOFT, "hard_stop_usd": HARD, "cost_basis": "CLI list-price estimates"},
         "frozen_rule": {"file": "configurations/fallback_rule.json", "sha256": sha(CFG / "fallback_rule.json"), "note": "hash recorded before hero scoring in configurations/fallback_rule.sha256"},
         "runs": runs, "statistics": {p.name: sha(p) for p in sorted(ST.glob("*.json"))}, "scripts": {p.name: sha(p) for p in sorted((ROOT / "scripts").glob("*.*")) if p.suffix in (".py", ".sh")}}
    (ROOT / "RUN_MANIFEST.json").write_text(json.dumps(M, indent=1))
    with (ROOT / "BUDGET_LEDGER.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["package", "use", "model", "new_calls", "cache_hits", "usd", "ledger_after_usd"])
        for r in runs:
            w.writerow([r["name"], r["use"], "+".join(r["model_ids"]), r["new_calls"], r["cache_hits"], r["incurred_usd"], r["ledger_after_usd"]])
    print(spent, M["budget"]["ledger_end_usd"], M["code_base_commit"][:8], len(M["uncommitted_paths"]))


if __name__ == "__main__":
    main()
