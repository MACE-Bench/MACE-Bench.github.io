"""DINOv2 ViT-S/14 CLS embeddings for mace10/out/dinov2_index.json, with the pre-study's pinned hub commit and transform
(code/sage-seqground/visual_audit/extract_dinov2.py). CPU, resumable in chunks. Runs with code/audio-jepa-icl/.venv (torch 2.8).
  ../audio-jepa-icl/.venv/bin/python mace10/embed_dinov2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "sage-seqground/visual_audit"))
from extract_dinov2 import HUB_COMMIT, preprocess  # noqa: E402

OUT = HERE / "out"; CH = OUT / "dinov2_chunks"; CH.mkdir(parents=True, exist_ok=True)
BATCH = 32


def main():
    torch.set_grad_enabled(False)
    rows = json.loads((OUT / "dinov2_index.json").read_text())
    model = torch.hub.load(f"facebookresearch/dinov2:{HUB_COMMIT}", "dinov2_vits14", trust_repo=True, skip_validation=True).eval()
    for b in range(0, len(rows), BATCH):
        f = CH / f"chunk_{b // BATCH:05d}.npy"
        if f.exists():
            continue
        x = torch.from_numpy(np.stack([preprocess(r["path"]) for r in rows[b:b + BATCH]]))
        e = model(x).numpy().astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True); np.save(f, e)
        if (b // BATCH) % 20 == 0:
            print(f"{b + len(x)}/{len(rows)}", flush=True)
    E = np.concatenate([np.load(f) for f in sorted(CH.glob("chunk_*.npy"))])
    assert E.shape[0] == len(rows)
    np.savez_compressed(OUT / "dinov2_embeddings.npz", E=E, ids=np.array([r["image_id"] for r in rows]))
    print("done", E.shape)


if __name__ == "__main__":
    main()
