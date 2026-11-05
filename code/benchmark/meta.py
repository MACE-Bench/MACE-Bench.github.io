"""Metadata-only inventory of HF tirtho149/SAGE (no image bytes). .venv/bin/python -m memlock.meta"""
import json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "data"; OUT.mkdir(exist_ok=True)
for l in open(os.path.expanduser("~/.config/secrets/.env")):
    l = l.strip()
    if l.startswith(("HF_TOKEN=", "export HF_TOKEN=")):
        os.environ["HF_TOKEN"] = l.split("=", 1)[1].strip().strip("\"'")
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
files = sorted(f for f in fs.ls("datasets/tirtho149/SAGE/data", detail=False) if f.endswith(".parquet"))
COLS = ["crop", "disease", "canonical_disease", "plant_organ", "raw_label", "filename", "symptom_source"]
rows = []
for s, f in enumerate(files):
    with fs.open(f, "rb", block_size=1 << 22) as fh:
        t = pq.ParquetFile(fh).read(columns=COLS).to_pylist()
    for j, r in enumerate(t):
        r["shard"], r["row"] = s, j
        r["source"] = (r["filename"] or "")[:12]
        rows.append(r)
    print(s, len(files), len(rows), flush=True)
(OUT / "sage_meta_all.json").write_text(json.dumps({"files": files, "rows": rows}))
