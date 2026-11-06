"""DINOv2 ViT-S/14 for extension images missing from mace10/out/dinov2_embeddings.npz. Same pinned hub commit and transform.
Never overwrites the original file. Run with ../audio-jepa-icl/.venv/bin/python (torch)."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent / "sage-seqground/visual_audit"))
from extract_dinov2 import HUB_COMMIT, preprocess
rows = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
torch.set_grad_enabled(False)
m = torch.hub.load(f"facebookresearch/dinov2:{HUB_COMMIT}", "dinov2_vits14", trust_repo=True, skip_validation=True).eval()
E = []
for b in range(0, len(rows), 32):
    x = torch.from_numpy(np.stack([preprocess(r["path"]) for r in rows[b:b + 32]])); e = m(x).numpy().astype(np.float32)
    E.append(e / np.linalg.norm(e, axis=1, keepdims=True)); print(b, flush=True)
np.savez_compressed(out, E=np.concatenate(E), ids=np.array([r["image_id"] for r in rows])); print("done", len(rows))
