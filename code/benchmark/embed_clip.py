"""OpenCLIP ViT-B-32 (laion2b_s34b_b79k) image embeddings for every fetched image, L2-normalised, resumable.
/usr/bin/python3 memlock/embed_clip.py"""
import json, sys
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; OUT = DATA / "clip"; OUT.mkdir(exist_ok=True)
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
torch.set_grad_enabled(False); torch.manual_seed(0)
model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED); model.eval()
recs = [r for p in sorted((DATA / "fetch_parts_v2").glob("shard_*.json")) for r in json.loads(p.read_text())["images"] if r.get("readable")]
recs.sort(key=lambda r: r["key"]); keys = [r["key"] for r in recs]
done = {}
for f in OUT.glob("chunk_*.npz"):
    z = np.load(f); done.update(zip(z["keys"].tolist(), z["E"]))
todo = [k for k in keys if k not in done]; B = 64
for b in range(0, len(todo), B):
    ks = todo[b:b + B]
    x = torch.stack([pre(Image.open(DATA / "img" / f"{k}.jpg").convert("RGB")) for k in ks])
    e = model.encode_image(x).float(); e = e / e.norm(dim=-1, keepdim=True)
    np.savez(OUT / f"chunk_{len(list(OUT.glob('chunk_*.npz'))):05d}.npz", keys=np.array(ks), E=e.numpy().astype(np.float32))
    if b % 1280 == 0: print(b + len(ks), len(todo), flush=True)
for f in OUT.glob("chunk_*.npz"):
    z = np.load(f); done.update(zip(z["keys"].tolist(), z["E"]))
np.savez_compressed(DATA / "clip_embeddings.npz", keys=np.array(keys), E=np.stack([done[k] for k in keys]))
json.dump({"model": MODEL, "pretrained": PRETRAINED, "open_clip": open_clip.__version__, "torch": torch.__version__, "n": len(keys)}, open(DATA / "clip_meta.json", "w"))
print("done", len(keys))
