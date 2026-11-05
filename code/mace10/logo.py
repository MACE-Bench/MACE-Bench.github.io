"""Original MACE logo: a nutmeg seed wrapped in the lacy red aril that is the spice mace. Vector, drawn from code.

  python -m mace10.logo      writes <paper>/figures/mace_logo.pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from .paper import PAPER

A, B = 0.74, 1.0


def ribbon(ax, phi, top, w0, col, wob=0.2, seed=0.0, z=3):
    t = np.linspace(-1.0, top, 160); r = np.sqrt(np.clip(1 - t ** 2, 0, 1))
    ph = np.radians(phi) + wob * np.sin(5.2 * t + seed)
    x = A * r * np.sin(ph); y = B * t
    w = w0 * (0.28 + 0.72 * np.sin(np.pi * np.clip((t + 1) / (top + 1), 0, 1)) ** 0.8) * (1 - ((t + 1) / (top + 1)) ** 5)
    ax.fill(np.r_[x - w, (x + w)[::-1]], np.r_[y, y[::-1]], color=col, lw=0, zorder=z)
    return t, x, y


def icon(ax):
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.32, 1.22); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Ellipse((0, 0), 2 * A, 2 * B, color="#5b3a23", lw=0, zorder=1))
    ax.add_patch(Ellipse((0.2, 0.34), 0.62, 0.92, color="#7b5233", lw=0, zorder=1.5, angle=-14))
    ax.add_patch(Ellipse((0.27, 0.52), 0.2, 0.36, color="#a07653", lw=0, zorder=1.6, angle=-14))
    red, deep, hot = "#c8321f", "#94200f", "#ef6a3a"
    for i, (phi, top, w0) in enumerate(((-74, 0.78, 0.085), (-46, 0.96, 0.1), (-16, 0.72, 0.095), (14, 0.93, 0.1), (44, 0.7, 0.09), (72, 0.86, 0.08))):
        ribbon(ax, phi, top, w0 * 1.28, deep, seed=i * 1.3, z=2.5); ribbon(ax, phi, top, w0, red, seed=i * 1.3, z=3)
        ribbon(ax, phi + 15, top * 0.62, w0 * 0.55, red, wob=0.28, seed=i * 2.1 + 1, z=3); ribbon(ax, phi - 12, top * 0.42, w0 * 0.45, hot, wob=0.3, seed=i * 1.7 + 2, z=3.2)
    t = np.linspace(0, np.pi, 80)
    ax.fill(np.r_[0.74 * np.cos(t), 0.74 * np.cos(t[::-1])], np.r_[-0.62 - 0.38 * np.sin(t) * 1.0, -0.62 + 0.0 * t - 0.09 * np.sin(t[::-1] * 4) ** 2], color=red, lw=0, zorder=3.4)
    ax.fill(np.r_[0.5 * np.cos(t), 0.5 * np.cos(t[::-1])], np.r_[-0.78 - 0.17 * np.sin(t), -0.78 + 0 * t], color=hot, lw=0, zorder=3.5)


def wordmark():
    """Icon plus a MACE wordmark in the spice colors: deep red shadow over a warm vertical gradient."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    fig = plt.figure(figsize=(5.4, 1.6)); a = fig.add_axes([0, 0, 0.255, 1]); icon(a)
    b = fig.add_axes([0.25, 0, 0.75, 1]); b.set_xlim(0, 5.0); b.set_ylim(0, 1.48); b.axis("off")
    p = TextPath((0, 0), "MACE", size=1, prop=FontProperties(family="DejaVu Serif", weight="bold")); e = p.get_extents()
    tr = Affine2D().translate(-e.x0, -e.y0).scale(4.72 / e.width, 1.12 / e.height).translate(0.12, 0.2)
    b.add_patch(PathPatch(tr.transform_path(p), fc="#5b1a10", ec="#5b1a10", lw=1.2, transform=Affine2D().translate(0.05, -0.05) + b.transData))
    top = PathPatch(tr.transform_path(p), fc="none", ec="#7a1f10", lw=0.8, transform=b.transData); b.add_patch(top)
    im = b.imshow(np.linspace(0, 1, 256)[:, None], extent=(0, 5, 0.1, 1.42), origin="upper", aspect="auto", cmap=LinearSegmentedColormap.from_list("m", ["#f6a13a", "#e8542a", "#b5271a"]), zorder=1, interpolation="bilinear")
    im.set_clip_path(top); b.set_xlim(0, 5.0); b.set_ylim(0, 1.48)
    fig.savefig(PAPER / "figures/mace_wordmark.pdf", bbox_inches="tight", pad_inches=0, transparent=True, dpi=1200); fig.savefig(PAPER / "figures/mace_wordmark.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(fig)


def main():
    fig, ax = plt.subplots(figsize=(2, 2.5)); icon(ax)
    fig.savefig(PAPER / "figures/mace_logo.pdf", bbox_inches="tight", pad_inches=0, transparent=True); fig.savefig(PAPER / "figures/mace_logo.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(fig)
    wordmark()


if __name__ == "__main__":
    main()
