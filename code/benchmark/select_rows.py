"""Locked row selection from SAGE metadata (no images). .venv/bin/python -m memlock.select_rows
Rules (v2, fixed 2026-11-02 before any model run; v1 used IN_MIN 30, IN_CAP 100, OOS_MIN 8, OOS_CAP 20):
  crop name    = crop.strip().title(); class = canonical_disease (fallback: disease)
  collection   = filename with extension and trailing index removed (e.g. 'Banana_Leaf_738.jpg' -> 'Banana_Leaf')
  in-scope     class with >= IN_MIN raw rows; sample up to IN_CAP rows, seeded, round-robin over collections
  candidate    crop with >= 1 in-scope class (single-class crops serve the same-disease-other-crop protocol) (final eligibility is decided after near-duplicate grouping)
  out-of-scope class of a candidate crop with OOS_MIN..IN_MIN-1 rows; up to OOS_CAP rows (open-set probes)"""
import hashlib, json, re
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"
SEED = 20261102; IN_MIN, IN_CAP, OOS_MIN, OOS_CAP = 10, 300, 5, 30

def collection(f):
    s = re.sub(r"\.[A-Za-z0-9]+$", "", f or ""); return re.sub(r"[_\- ]*\(?\d+\)?$", "", s)

def rank(r): return hashlib.sha256(f"{SEED}|{r['shard']}|{r['row']}".encode()).hexdigest()

def round_robin(rows, cap):
    by = defaultdict(list)
    for r in sorted(rows, key=rank): by[r["collection"]].append(r)
    order = sorted(by, key=lambda s: hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest()); out = []
    while len(out) < cap and any(by.values()):
        for s in order:
            if by[s] and len(out) < cap: out.append(by[s].pop(0))
    return out

def main():
    R = json.loads((DATA / "sage_meta_all.json").read_text())["rows"]
    cls = defaultdict(lambda: defaultdict(list))
    for r in R:
        r["crop_n"] = r["crop"].strip().title(); r["class"] = (r.get("canonical_disease") or r["disease"]).strip()
        r["collection"] = collection(r["filename"]); cls[r["crop_n"]][r["class"]].append(r)
    sel, summary = [], {}
    for crop in sorted(cls):
        ins = {c: v for c, v in cls[crop].items() if len(v) >= IN_MIN}
        if len(ins) < 1: continue
        oos = {c: v for c, v in cls[crop].items() if OOS_MIN <= len(v) < IN_MIN}
        for c, v in ins.items():
            for r in round_robin(v, IN_CAP): sel.append({**{k: r[k] for k in ("shard", "row", "crop_n", "class", "collection", "filename", "raw_label")}, "role": "in_scope"})
        for c, v in oos.items():
            for r in round_robin(v, OOS_CAP): sel.append({**{k: r[k] for k in ("shard", "row", "crop_n", "class", "collection", "filename", "raw_label")}, "role": "out_of_scope"})
        summary[crop] = {"in_scope": {c: len(v) for c, v in ins.items()}, "out_of_scope": {c: len(v) for c, v in oos.items()}}
    (DATA / "selection.json").write_text(json.dumps({"rules": __doc__, "seed": SEED, "summary": summary, "rows": sel}, indent=0))
    print(len(summary), "candidate crops;", sum(r["role"] == "in_scope" for r in sel), "in-scope rows;", sum(r["role"] == "out_of_scope" for r in sel), "oos rows")

if __name__ == "__main__": main()
