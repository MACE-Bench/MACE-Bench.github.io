"""Apple Vision OCR (accurate) on every released image; images with any word of >= 3 letters are excluded
(disease names, captions, slide text, watermarks). .venv/bin/python -m memlock.ocr_text -> memlock/data/ocr_text.json"""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent; REL = HERE / "release"; DATA = HERE / "data"

def ocr(args):
    k, path = args
    import Vision
    from Foundation import NSURL
    req = Vision.VNRecognizeTextRequest.alloc().init(); req.setRecognitionLevel_(0); req.setUsesLanguageCorrection_(False)
    h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None); h.performRequests_error_([req], None)
    return k, [o.topCandidates_(1)[0].string() for o in (req.results() or [])]

if __name__ == "__main__":
    import pandas as pd
    M = pd.read_parquet(REL / "metadata/images.parquet")
    prev = json.loads((DATA / "ocr_text.json").read_text())["text"] if (DATA / "ocr_text.json").exists() else {}
    jobs = [(r.image_id, DATA / "img" / f"{r.image_id}.jpg") for r in M.itertuples() if r.image_id not in prev]
    out = dict(prev); print("new images to OCR:", len(jobs))
    with Pool(8) as p:
        for n, (k, t) in enumerate(p.imap_unordered(ocr, jobs, chunksize=32)):
            out[k] = t
            if n % 4000 == 0: print(n, flush=True)
    words = {k: re.findall(r"[A-Za-z]{3,}", " ".join(t)) for k, t in out.items()}
    flagged = sorted(k for k, w in words.items() if w)
    (DATA / "ocr_text.json").write_text(json.dumps({"text": out, "excluded": flagged}, indent=0))
    print("excluded", len(flagged), "of", len(out))
