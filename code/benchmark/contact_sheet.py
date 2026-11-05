"""One contact-sheet PDF for manual review: one page per crop, one block per disease, every released image with its id.
Border: blue IID, orange OOD (held-out collection), gray rare-disease probe; red = OpenCLIP text/slide/collage hint (>0.3).
.venv/bin/python -m memlock.contact_sheet OUT.pdf"""
import json, sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent; R = HERE / "release"; DATA = HERE / "data"
T, PAD, LAB, W = 96, 6, 13, 2600
COL = {"iid": (42, 120, 214), "ood_in_scope": (235, 104, 52), "ood_out_of_scope": (235, 104, 52), "probe_out_of_scope": (150, 150, 150)}
FONT = "/System/Library/Fonts/Helvetica.ttc"
f_id, f_h, f_t = ImageFont.truetype(FONT, 10), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 34)
M = pd.read_parquet(R / "metadata/images.parquet"); flag = json.load(open(DATA / "text_flags.json")); M["p"] = M.image_id.map(flag).fillna(0)
order = {"iid": 0, "ood_in_scope": 1, "ood_out_of_scope": 2, "probe_out_of_scope": 3}
M["o"] = M.role.map(order); per_row = (W - 2 * PAD) // (T + PAD)
pages = []
for crop, g in sorted(M.groupby("crop"), key=lambda kv: -len(kv[1])):
    blocks = []
    for d, gd in sorted(g.groupby("class"), key=lambda kv: (kv[1].o.min(), kv[0])):
        gd = gd.sort_values(["o", "image_id"]); rows = -(-len(gd) // per_row)
        blocks.append((d, gd, rows))
    H = 120 + sum(40 + r * (T + LAB + PAD) for _, _, r in blocks) + 20
    pg = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(pg)
    n = len(g); fl = int((g.p > 0.3).sum())
    dr.text((PAD * 3, 22), f"{crop}", font=f_t, fill=(11, 11, 11))
    dr.text((PAD * 3, 72), f"{n} images, {g['class'].nunique()} diseases. Border: blue IID, orange OOD held-out collection, gray rare-disease probe, red = possible text/slide/collage ({fl}).",
            font=f_h, fill=(82, 81, 78))
    y = 120
    for d, gd, rows in blocks:
        rc = gd.role.value_counts().to_dict()
        dr.text((PAD * 3, y + 8), f"{d}   ({', '.join(f'{k.replace('_out_of_scope', ' out').replace('_in_scope', ' in')}: {v}' for k, v in rc.items())})", font=f_h, fill=(11, 11, 11)); y += 40
        for j, r in enumerate(gd.itertuples()):
            x = PAD + (j % per_row) * (T + PAD); yy = y + (j // per_row) * (T + LAB + PAD)
            im = Image.open(R / r.file).convert("RGB"); w, h = im.size; s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((T - 6, T - 6))
            c = (227, 73, 72) if r.p > 0.3 else COL[r.role]
            dr.rectangle([x, yy, x + T - 1, yy + T - 1], fill=c); pg.paste(im, (x + 3, yy + 3))
            dr.text((x + 1, yy + T + 1), r.image_id, font=f_id, fill=(120, 120, 120))
        y += rows * (T + LAB + PAD)
    pages.append(pg)
out = Path(sys.argv[1])
pages[0].save(out, "PDF", resolution=150, save_all=True, append_images=pages[1:], quality=80)
print("pages", len(pages), "->", out, round(out.stat().st_size / 1e6, 1), "MB")
