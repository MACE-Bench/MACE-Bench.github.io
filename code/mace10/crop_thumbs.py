"""One real image per crop for the figures, chosen by eye from correctly diagnosed hero queries. Decorative: no result depends on them.

  python -m mace10.crop_thumbs      writes <paper>/figures/img/crop_<Crop>.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from agmem import data
from .paper import PAPER

PICK = json.loads((Path(__file__).parent / "crop_thumbs.json").read_text())


def main():
    for crop, image_id in PICK.items():
        im = Image.open(data.path(image_id)).convert("RGB"); w, h = im.size; m = int(min(w, h) * 0.92)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((520, 520), Image.LANCZOS)
        im.save(PAPER / f"figures/img/crop_{crop}.jpg", quality=92, dpi=(600, 600))
    print(len(PICK), "crop images")


if __name__ == "__main__":
    main()
