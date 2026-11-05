"""Two recorded cases for the evidence figure, chosen by a fixed rule (first by image id) among images with clear reuse terms:
  success  test image and all four references reusable; SAGE wrong, labels only wrong, MACE right, true disease retrieved
  failure  test image and at least three references reusable; true disease not retrieved, SAGE right, MACE wrong
A reference without clear reuse terms is shown as a withheld tile, never as the image. Illustrative, not representative. Every string comes from out/P1 traces and out/captions.json; no model call.

  python -m mace10.case        writes <paper>/figures/img/c_*.jpg and <paper>/case_data.tex
"""
from __future__ import annotations

import gzip
import json
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from agmem import data as _d  # noqa: F401

from agmem import data
from .paper import OUT, PAPER

CLEAR = {"LeafNet", "PlantDoc", "Cucumber_Zen", "Soybean_PNAS", "PlantVillage", "VQAPlantDise", "Bean_Leaf", "Lettuce", "SugarLeaf_ID", "Cauliflower", "Pumpkin_Leaf", "Rose_Leaf", "Durian_Leaf", "Eggplant_Dis", "Vanilla_Dise"}


def clear(i):
    r = data.images().loc[i]
    return r["collection"] in CLEAR and not (r["collection"] == "LeafNet" and r["crop"] == "Mango")   # LeafNet mango copies a CC BY-NC set


def pick():
    W = pd.read_csv(OUT / "results_wide.csv").set_index("image_id")
    P = pd.DataFrame(map(json.loads, (OUT / "P1/predictions.jsonl").open())); M = P[P.condition == "mace"].sort_values("image_id")
    good = lambda r: sum(clear(m[2:]) for m in r.memory_ids)
    s = f = None
    for r in M.itertuples():
        w = W.loc[r.image_id]
        if not clear(r.image_id):
            continue
        if s is None and good(r) == 4 and w.sage == 0 and w.labels_all == 0 and w.mace == 1 and w.hit == 1:
            s = r.image_id
        if f is None and good(r) >= 3 and w.hit == 0 and w.sage == 1 and w.mace == 0:
            f = r.image_id
    return s, f


def traces():
    T = {}
    for l in gzip.open(OUT / "P1/traces.jsonl.gz", "rt"):
        r = json.loads(l); T[(r["result"]["condition"], r["result"]["image_id"])] = r
    return T


def esc(t):
    t = str(t).replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("~", "about ").replace("µ", "$\\mu$")
    return re.sub(r"\s+", " ", t).strip()


def thumb(image_id, name, side=520):
    im = Image.open(data.path(image_id)).convert("RGB"); w, h = im.size; m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((side, side), Image.LANCZOS); im.save(PAPER / f"figures/img/{name}.jpg", quality=92, dpi=(600, 600))


def withheld(name, side=520):
    im = Image.new("RGB", (side, side), "#e4e4e4"); d = ImageDraw.Draw(im); f = ImageFont.load_default(size=58)
    for y, t in ((190, "Image"), (262, "withheld")):
        d.text((side / 2, y), t, fill="#555555", font=f, anchor="mm")
    im.save(PAPER / f"figures/img/{name}.jpg", quality=92, dpi=(600, 600))


def caption_line(c):
    return "; ".join(str(v) for k, v in c.items() if k != "organ" and v and str(v).lower() not in ("none", "not visible", "n/a"))


KEY = {"S": (3, 3), "F": (2, 1)}


def clause(line, key):
    c = [x for x in line.split("; ") if x.startswith(key)]
    assert len(c) == 1, (key, line)
    return c[0]


BOX = r"""\\begin{tcolorbox}[enhanced, colback=gray!3, colframe=gray!55, coltitle=white, colbacktitle=gray!62, boxrule=0.4pt, arc=2pt, left=3pt, right=3pt, top=2pt, bottom=2pt, toptitle=0.5pt, bottomtitle=0.5pt,
  fonttitle=\\scriptsize\\bfseries, width=\\linewidth, equal height group=mace3, title={%(panel)s \\ Retrieval Trace: \\mdseries \\c%(t)sCrop, \\c%(t)sSplit, \\c%(t)sColl{}, \\c%(t)sId}]
\\fontsize{5.9}{7.1}\\selectfont
\\textbf{Model:} Sonnet 5.5 \\ \\textbf{KB:} SAGE \\ \\textbf{$K$:} 4 \\ \\textbf{Ground truth:} \\c%(t)sTruth \\ \\textbf{Prediction:} \\c%(t)sMace\\\\
\\textbf{Outcome:} \\textcolor{%(oc)s}{\\textbf{%(out)s}} \\ \\textbf{Confidence:} \\c%(t)sConf \\ \\textbf{SAGE:} \\c%(t)sSage \\ \\textbf{Labels only:} \\c%(t)sLabels\\par\\vspace{1pt}\\hrule\\vspace{2pt}
\\begin{minipage}[t]{0.635\\linewidth}\\vspace{0pt}\\raggedright
\\tstep{1}{\\texttt{Caption test image}}
\\tstep{2}{``\\c%(t)sExTest.''}
\\tstep{3}{\\texttt{Retrieve four captions (BM25)}}
\\tstep{4}{\\c%(t)sMemLabel: ``\\c%(t)sExMem.''}
\\tstep{5}{\\texttt{Read image and packet}}
\\tstep{6}{Organ: \\c%(t)sOrgan. Ranked: \\c%(t)sRanked.}
\\tstep{7}{Cited claim \\c%(t)sKbId: ``\\c%(t)sKbText''}
\\tstep{8}{``\\c%(t)sWhy''}
\\end{minipage}\\hfill
\\begin{minipage}[t]{0.345\\linewidth}\\vspace{0pt}\\centering
\\includegraphics[width=0.62\\linewidth]{figures/img/c_%(t)s_q.jpg}\\\\[-1pt]{\\fontsize{5.4}{6}\\selectfont\\textit{Test image}}\\\\[2pt]
{\\fontsize{5.6}{6.4}\\selectfont\\textbf{Retrieved labeled references}}\\\\[1pt]
%(grid)s
\\end{minipage}
\\end{tcolorbox}"""


def traces_tex():
    """SAGE-style trace boxes; every string is a macro from case_data.tex."""
    W = pd.read_csv(OUT / "results_wide.csv"); recs = {}
    T = traces(); ids = dict(zip("SF", pick()))
    head = ["% generated by code/agmem-bench/mace10/case.py; layout after the SAGE reasoning-trace box",
            r"\\newcommand{\\tstep}[2]{\\par\\noindent\\hangindent=3.4em\\hangafter=1\\makebox[3.4em][l]{\\textbf{Step #1:}}#2\\par\\vspace{0.6pt}}",
            r"\\newcommand{\\refcell}[3]{\\parbox[t]{0.47\\linewidth}{\\centering\\setlength{\\fboxsep}{0.5pt}\\setlength{\\fboxrule}{#3}\\fcolorbox{#2}{white}{\\includegraphics[width=0.9\\linewidth]{figures/img/#1.jpg}}}}", r"\\definecolor{trOK}{RGB}{0,158,115}\\definecolor{trNo}{RGB}{190,190,190}\\definecolor{trBad}{RGB}{213,94,0}",
            r"\\noindent\\begin{minipage}[t]{0.493\\textwidth}\\vspace{0pt}"]
    body = []
    for n, (t, panel) in enumerate((("S", "a"), ("F", "b"))):
        m = T[("mace", ids[t])]["result"]; cells = []
        for k, lab in enumerate(m["memory_labels_shown"], 1):
            ok = lab == m["label"]; w = "abcd"[k - 1]
            cells.append((rf"\\refcell{{c_{t}_{k}}}{{{'trOK' if ok else 'trNo'}}}{{{'1pt' if ok else '0.3pt'}}}", rf"\\parbox[t]{{0.47\\linewidth}}{{\\centering\\fontsize{{5}}{{5.6}}\\selectfont\\c{t}L{w}}}"))
        grid = cells[0][0] + r"\\hfill" + cells[1][0] + "\\\\[-1pt]" + cells[0][1] + r"\\hfill" + cells[1][1] + "\\\\[1.5pt]" + cells[2][0] + r"\\hfill" + cells[3][0] + "\\\\[-1pt]" + cells[2][1] + r"\\hfill" + cells[3][1]
        good = m["prediction"] == m["label"]
        body.append(BOX % {"t": t, "panel": panel, "oc": "trOK" if good else "trBad", "out": "Correct" if good else "Incorrect", "grid": grid})
    tex = "\n".join(head) + "\n" + body[0] + "\n\\end{minipage}\\hfill\\begin{minipage}[t]{0.493\\textwidth}\\vspace{0pt}\n" + body[1] + "\n\\end{minipage}\n"
    tex = tex.replace(2 * chr(92), chr(92)); tex = re.sub(r"(?<!\\)\\\[", lambda m: 2 * chr(92) + "[", tex)
    (PAPER / "figures/fig3_traces.tex").write_text(tex)


def main():
    T = traces(); caps = json.loads((OUT / "captions.json").read_text()); P8 = pd.concat([pd.DataFrame(map(json.loads, (OUT / p / "predictions.jsonl").open())) for p in ("P8", "P4labels")]).set_index("image_id")
    out = ["% generated by code/agmem-bench/mace10/case.py; do not edit"]; dump = {}
    for tag, q in zip(("S", "F"), pick()):
        m = T[("mace", q)]["result"]; s = T[("sage", q)]["result"]; thumb(q, f"c_{tag}_q"); short = {}
        mac = {"Truth": m["label"], "Sage": s["prediction"], "Labels": P8.loc[q].prediction, "Mace": m["prediction"], "Why": m["explanation"], "Conf": f"{m['confidence']:.2f}", "Id": q, "Coll": m["collection"].replace("_", " "), "Crop": m["crop"], "Split": "IID" if m["split"] == "iid" else "held-out", "Cap": caption_line(caps[q])}
        for n, (mid, lab) in enumerate(zip(m["memory_ids"], m["memory_labels_shown"]), 1):
            w = "abcd"[n - 1]; (thumb(mid[2:], f"c_{tag}_{n}") if clear(mid[2:]) else withheld(f"c_{tag}_{n}")); mac[f"L{w}"] = lab; short[lab] = caption_line(caps[mid[2:]])
        first = m["memory_labels_shown"].index(m["label"]) if m["label"] in m["memory_labels_shown"] else 0
        mac["MemLabel"] = m["memory_labels_shown"][first]; mac["MemCap"] = caption_line(caps[m["memory_ids"][first][2:]])
        rep = json.loads(T[("mace", q)]["reply"]["text"][T[("mace", q)]["reply"]["text"].index("{"):T[("mace", q)]["reply"]["text"].rindex("}") + 1])
        mac["Organ"] = rep.get("organ"); mac["Ranked"] = ", ".join(rep.get("ranked_candidates", [])); text = T[("mace", q)]["text"][0]
        kb = [x for x in m["cited"] if x.startswith("c_")]; lines = [l for l in text.split("\n") if l.startswith(f"[{kb[0]}]")]
        assert lines, kb; mac["KbId"] = kb[0]; mac["KbText"] = lines[0].split("] ", 1)[1]
        mac["ExTest"] = mac["Cap"].split("; ")[KEY[tag][0]]; mac["ExMem"] = mac["MemCap"].split("; ")[KEY[tag][1]]
        for k, v in mac.items():
            out.append(f"\\newcommand{{\\c{tag}{k}}}{{{esc(v)}}}")
        dump[tag] = {**mac, "memory_ids": m["memory_ids"], "shown": m["memory_labels_shown"], "cited": m.get("cited"), "job_id": T[("mace", q)]["job_id"]}
    (PAPER / "case_data.tex").write_text("\n".join(out) + "\n"); traces_tex(); (OUT / "case_records.json").write_text(json.dumps(dump, indent=1)); print(json.dumps(dump, indent=1)[:3500])


if __name__ == "__main__":
    main()
