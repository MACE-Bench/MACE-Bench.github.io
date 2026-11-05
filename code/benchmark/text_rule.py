"""Decide which images are excluded for text (run after memlock.ocr_text and memlock/text_flag.py).
Excluded: OCR text containing any disease / pathogen / symptom / crop word, or >= 3 words outside the watermark
vocabulary (captions, slides, documents), or OpenCLIP slide/collage score > 0.5.
Kept but flagged has_watermark: text made only of watermark, credit or ID tokens (alamy, shutterstock, UGA1234567, www...)."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"
WATERMARK = set("""alamy shutterstock dreamstime getty gettyimages istock istockphoto depositphotos adobe stock photo photos image images
credit credits copyright rights reserved all www com org net edu gov uga bugwood bugwoodorg forestryimages ipmimages
university extension service usda ars inc ltd the and for by via from courtesy license licensed""".split())
SYMPTOM = set("""disease diseases infected infection symptom symptoms lesion lesions spot spots blight rot rots mildew rust smut scab mold mould
virus viral mosaic wilt canker cankers blotch necrosis necrotic chlorosis chlorotic fungus fungal fungi bacteria bacterial
pathogen anthracnose leaf leaves fruit stem root pest healthy causal organism control spray""".split())

def main():
    O = json.loads((DATA / "ocr_text.json").read_text())["text"]; C = json.loads((DATA / "text_flags.json").read_text())
    meta = json.loads((DATA / "sage_meta_all.json").read_text())["rows"]
    vocab = set(SYMPTOM)
    for r in meta:
        for f in ("crop", "canonical_disease", "disease"):
            vocab |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", r.get(f) or "")}
    excl, wm, why = [], [], {}
    for k, t in O.items():
        words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", " ".join(t))]
        hit = [w for w in words if w in vocab]; other = [w for w in words if w not in WATERMARK and not re.fullmatch(r"uga\d*", w)]
        if hit or len(other) >= 3: excl.append(k); why[k] = "label_or_caption_text: " + " ".join(t)[:120]
        elif words: wm.append(k)
    for k, p in C.items():
        if p > 0.5 and k in O and k not in why: excl.append(k); why[k] = f"clip_slide_or_collage {p}"
    (DATA / "text_exclusions.json").write_text(json.dumps({"excluded": sorted(set(excl)), "why": why, "watermark_only": sorted(wm)}, indent=0))
    print("excluded", len(set(excl)), "watermark-only kept", len(wm))

if __name__ == "__main__":
    main()
