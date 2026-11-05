"""Zero-shot OpenCLIP screen for images that are documents, slides, text overlays or multi-photo collages.
/usr/bin/python3 memlock/text_flag.py  -> memlock/data/text_flags.json (a hint for manual review, not a filter)"""
import json
from pathlib import Path
import numpy as np, torch, open_clip
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); tok = open_clip.get_tokenizer("ViT-B-32")
BAD = ["a screenshot of a document with text", "a presentation slide with a title and text", "a photo with a large text caption written on it",
       "a collage of several photos with labels", "a diagram or illustration with labels", "a page from a book or website"]
GOOD = ["a photo of a plant leaf", "a close-up photo of a diseased plant", "a photo of a crop in a field", "a photo of a fruit or vegetable",
        "a photo of a plant stem or root", "a microscope image of fungal spores"]
with torch.no_grad():
    T = model.encode_text(tok(BAD + GOOD)).float(); T = (T / T.norm(dim=-1, keepdim=True)).numpy()
z = np.load(DATA / "clip_embeddings.npz"); E = z["E"].astype(np.float32); keys = z["keys"].tolist()
L = 100 * E @ T.T; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
pbad = P[:, :len(BAD)].sum(1)
out = {k: round(float(p), 3) for k, p in zip(keys, pbad)}
(DATA / "text_flags.json").write_text(json.dumps(out))
print("flagged >0.5:", int((pbad > 0.5).sum()), "of", len(keys))
