"""Write the AAAI-compliant single-file paper from mace10/paper_src and render the TikZ figures to PDF.

  python -m mace10.build_submission <overleaf_clone>      needs pdflatex on PATH

AAAI kit rules applied: one .tex file (no \\input), no embedded links or bookmarks, no \\tiny or \\setlength, no float changes,
plain title block, no Identity-H or Type 3 fonts, only used graphics in the project.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "paper_src"
FIGS = ["fig2_crops", "fig_evidence", "fig_heat", "fig3_panels", "fig4_stability_cost", "fig_cost"]
PRE = r"""\documentclass[letterpaper]{article}
\usepackage[preprint]{aaai2027}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}
\usepackage{caption}
\frenchspacing
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{colortbl}
\usepackage{pifont}
\definecolor{bestgreen}{RGB}{200,240,205}
\newcommand{\cm}{\ding{51}}
\newcommand{\dm}{--}
\newcommand{\mcite}[2]{\shortstack[l]{#1\\{\scriptsize\citet{#2}}}}
\setcounter{secnumdepth}{1}

\title{MACE: Memory-Augmented Crop-Disease Evaluation\\of Frozen Vision-Language Models}
"""


def macros():
    m = {}
    for f in ("numbers.tex", "numbers_analysis.tex", "numbers_mace10.tex", "case_data.tex"):
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", (SRC / f).read_text(), flags=re.M):
            m[k] = v
    return m


def render(name, body, M, opts="border=0pt,varwidth=7in"):
    """Render a TikZ or tcolorbox figure to a stand-alone PDF. Figure text is part of the graphic."""
    work = SRC / "_render"; work.mkdir(exist_ok=True)
    defs = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items())
    tex = ("\\documentclass[" + opts + "]{standalone}\n" r"\usepackage{newtxtext,helvet,graphicx,xcolor,tikz,fontawesome5}\usepackage[skins]{tcolorbox}\usepackage{pifont}\usepackage{amsmath}"
           "\n" r"\usetikzlibrary{positioning,calc,arrows.meta}" "\n" + defs + "\n" r"\setlength{\textwidth}{7in}" "\n" r"\begin{document}" "\n" + body + "\n" r"\end{document}" "\n")
    (work / f"{name}.tex").write_text(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"-output-directory={work}", str(work / f"{name}.tex")], cwd=SRC, capture_output=True, text=True)
    if not (work / f"{name}.pdf").exists() or "\n! " in r.stdout:
        raise SystemExit(r.stdout[-1500:])
    return work / f"{name}.pdf"


def main(out):
    out = Path(out); (out / "figures").mkdir(parents=True, exist_ok=True); M = macros()
    t = (SRC / "main.tex").read_text()
    shutil.copy(render("overview", (SRC / "figures/overview_tikz.tex").read_text(), M, "border=0pt"), out / "figures/overview.pdf")
    shutil.copy(render("fig3_traces", (SRC / "figures/fig3_traces.tex").read_text(), M), out / "figures/fig3_traces.pdf")
    for f in FIGS:
        shutil.copy(SRC / f"figures/{f}.pdf", out / f"figures/{f}.pdf")
    body = t[t.index(r"\author{"):]
    body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("%"))
    prior = re.sub(r"^%.*\n", "", (SRC / "tables/t_prior.tex").read_text(), flags=re.M)
    prior = re.sub(r"\\newcommand.*\n", "", prior).replace(r"\setlength{\tabcolsep}{2.4pt}", "").replace(r"\macelogo\ ", "").replace(r"\scriptsize", r"\footnotesize", 1)
    prior = prior.replace(r"\begin{tabular}{@{}lccccc l@{}}", r"\begin{tabular}{@{}lccccc@{}}")
    prior = re.sub(r" & \\textbf\{Evidence at\} \\\\", r" \\\\", prior); prior = re.sub(r" & \\textbf\{test time\} \\\\", r" \\\\", prior)
    prior = re.sub(r"(\\mcite\{[^\n]*?|\\textbf\{MACE \(ours\)\}[^\n]*?) & (?:Image only|Feature cache|Example images|KB \+ images|\\textbf\{KB \+ labeled captions\}) \\\\", r"\1 \\\\", prior)
    tm = (SRC / "tables/t_main.tex").read_text().replace(r"\macelogo\ ", "")
    tm = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", tm).replace(r"{\tiny[", r"{\scriptsize[").replace(r"\begin{tabular}{lcccc}", r"\begin{tabular}{lcc}").replace(r"\multicolumn{5}", r"\multicolumn{3}")
    rows = []
    for l in tm.split("\n"):
        if l.count("&") == 4:
            l = " & ".join(x.strip() for x in l.replace(r" \\", "").split("&")[:3]) + r" \\"
        rows.append(l)
    tm = "\n".join(rows).replace(r"\scriptsize", r"\footnotesize", 1).replace(" Green: best planned method.", " Green: best planned method.")
    body = body.replace(r"\input{tables/t_prior.tex}", prior).replace(r"\input{tables/t_main.tex}", tm)
    body = body.replace(r"\resizebox{0.98\textwidth}{!}{\input{figures/overview_tikz.tex}}", r"\includegraphics[width=\textwidth]{figures/overview.pdf}")
    body = re.sub(r"\\input\{figures/fig3_traces\.tex\}\s*\\vspace\{3pt\}\s*", r"\\includegraphics[width=\\textwidth]{figures/fig3_traces.pdf}\n\n", body)
    body = body.replace("tirtho@iastate.edu, soumiks@iastate.edu\\\\[2pt]\n\\weblink{https://mace-bench.github.io/}{\\faGlobe\\ mace-bench.github.io}\n", "tirtho@iastate.edu, soumiks@iastate.edu\n")
    body = body.replace("\\end{abstract}\n", "\\end{abstract}\n\n\\begin{links}\n  \\link{Website}{https://mace-bench.github.io/}\n  \\link{Code}{https://github.com/MACE-Bench/MACE-Bench.github.io}\n\\end{links}\n", 1)
    body = body.replace(r"\weblink{https://mace-bench.github.io/#license}{per-image credits are online}", r"per-image credits are on the website")
    for k in sorted(M, key=len, reverse=True):
        body = re.sub(r"\\" + k + r"(\{\})?(?![A-Za-z])", lambda _: M[k], body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    (out / "main.tex").write_text(PRE + "\n" + body)
    bib = (SRC / "references.bib").read_text()
    while r"\weblink{" in bib:
        bib = re.sub(r"\\weblink\{[^{}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", bib)
    (out / "references.bib").write_text(bib)
    for f in ("aaai2027.sty", "aaai2027.bst"):
        shutil.copy(SRC / f, out / f)
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1])
