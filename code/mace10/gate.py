"""Exit 0 only if a package finished cleanly: every row present, no stop, only the pilot's model, no job left in failures."""
import json
import sys
from pathlib import Path

d = Path(__file__).resolve().parent / "out" / sys.argv[1]
s = json.loads((d / "status.json").read_text())
rows = [json.loads(l) for l in (d / "predictions.jsonl").open()]
models = {m for r in rows if not r["cache_hit"] for m in (r.get("model_ids") or [])}
ids = {r["job_id"] for r in rows}
fails = [json.loads(l) for l in (d / "failures.jsonl").open()] if (d / "failures.jsonl").exists() else []
open_fails = [f for f in fails if isinstance(f["job"], str) and "|" in f["job"] and f["job"] not in ids]
problems = []
if not s["complete"] or s["stop"]: problems.append(f"incomplete or stopped: {s['rows']}/{s['jobs_total']} stop={s['stop']}")
if len(ids) != len(rows): problems.append("duplicate job ids")
if models - {"claude-sonnet-5-5"}: problems.append(f"unexpected models {models}")
if open_fails: problems.append(f"{len(open_fails)} failed jobs not resolved")
print(sys.argv[1], "OK" if not problems else problems, f"rows={len(rows)} ledger={s['ledger_total']}")
sys.exit(1 if problems else 0)
