"""Stream SAGE shards once: save selected images (longest side 512, JPEG q90) + raw sha256 + pHash, and collect the full
per-(crop, class) KB text from every row. Resumable per shard. .venv/bin/python -m memlock.fetch"""
import hashlib, io, json, os
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; IMG = DATA / "img"; PART = DATA / "fetch_parts_v2"
IMG.mkdir(parents=True, exist_ok=True); PART.mkdir(exist_ok=True)
for l in open(os.path.expanduser("~/.config/secrets/.env")):
    l = l.strip()
    if l.startswith(("HF_TOKEN=", "export HF_TOKEN=")): os.environ["HF_TOKEN"] = l.split("=", 1)[1].strip().strip("\"'")
import imagehash, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image
KBCOLS = ["crop", "disease", "canonical_disease", "plant_organ", "visual_symptoms", "pathogen", "disease_type", "symptom_source", "symptom_quote", "source"]

def main():
    S = json.loads((DATA / "selection.json").read_text())["rows"]; files = json.loads((DATA / "sage_meta_all.json").read_text())["files"]
    want = defaultdict(dict)
    for r in S: want[r["shard"]][r["row"]] = r
    fs = HfFileSystem()
    for s, f in enumerate(files):
        out = PART / f"shard_{s:02d}.json"
        if out.exists(): continue
        with fs.open(f, "rb", block_size=1 << 23) as fh:
            pf = pq.ParquetFile(fh); names = pf.schema_arrow.names
            T = pf.read(columns=[c for c in KBCOLS if c in names] + (["image"] if s in want else []))
        kb = {}; cols = {c: T.column(c).to_pylist() for c in KBCOLS if c in names}
        for j in range(T.num_rows):
            crop = (cols["crop"][j] or "").strip().title(); cl = ((cols.get("canonical_disease") or cols["disease"])[j] or cols["disease"][j] or "").strip()
            key = f"{crop}||{cl}"; e = kb.setdefault(key, {"crop": crop, "class": cl, "n_rows": 0, "variants": {}})
            e["n_rows"] += 1
            v = {c: cols[c][j] for c in ("disease", "plant_organ", "visual_symptoms", "pathogen", "disease_type", "symptom_source", "symptom_quote") if c in cols}
            vk = hashlib.sha1(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:12]
            e["variants"].setdefault(vk, {**v, "n_rows": 0})["n_rows"] += 1
        recs = []
        if s in want:
            bcol = T.column("image")
            for j, r in want[s].items():
                rec = {**r, "parquet_source": cols.get("source", [None] * T.num_rows)[j]}
                try:
                    cell = bcol[j].as_py(); raw = cell.get("bytes") if isinstance(cell, dict) else cell; im = Image.open(io.BytesIO(raw)); im.load(); im = im.convert("RGB")
                    rec.update(sha256=hashlib.sha256(raw).hexdigest(), phash=str(imagehash.phash(im)), orig_size=list(im.size))
                    key = f"s{s}_r{j}"; p = IMG / f"{key}.jpg"
                    if not p.exists(): im.thumbnail((512, 512)); im.save(p, "JPEG", quality=90)
                    rec.update(key=key, file=f"img/{key}.jpg", readable=True)
                except Exception as ex:
                    rec.update(readable=False, error=f"{type(ex).__name__}: {str(ex)[:100]}")
                recs.append(rec)
        out.write_text(json.dumps({"kb": kb, "images": recs}))
        del T
        print(s, len(files), len(recs), flush=True)

if __name__ == "__main__": main()
