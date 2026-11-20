"""Schematic of the event-token forecaster (study 218 configuration).

Draws architecture.{pdf,png}: the context table and the lead-time / phase
features on the left; the event-token transformer in the middle; the
color-anchored output head on the right. The head reads only the transformer
readout and the Dt features (no conditioner summary, event_token_no_summary).
No model or data needed; the sizes on the figure are the 218 settings
(train_LodeRunner_ddp.py, bomberman.py).

    python plot_architecture.py --outdir figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from pub_style import add_style_args, pub_band_colors, pub_fs, use_pub_style

BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
HOUSE_BAND_COLORS = (
    "#2A9D8F", "#E63946", "#F4A261", "#457B9D", "#1B9E77",
    "#D62828", "#E9C46A", "#8338EC", "#264653",
)
# Fills: transformer path, output head, inputs.
FILL = dict(vit="#dbe8f5", head="#fae3cf", inp="white")
EDGE = "0.25"
# Example context: 8 valid detections of 12 slots (band index per slot).
EX_BANDS = (0, 4, 1, 5, 8, 6, 4, 2)
N_SLOTS = 12

# Canvas in data units (1 unit = W_IN / X_MAX inches, equal aspect).
X_MAX, Y_MAX, W_IN = 160.0, 88.0, 7.0


def get_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", default=".")
    p.add_argument("--name", default="architecture")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    add_style_args(p)
    return p.parse_args()


class Canvas:
    """Thin drawing helpers on one full-figure axes in data units."""

    def __init__(self, ax, fs):
        self.ax, self.fs = ax, fs

    def box(self, x, y, w, h, text=None, fc="white", size=1.0, weight="normal",
            ls="-", z=2, va="center", dy=0.0):
        """Rounded box with centered text; returns its (x, y, w, h)."""
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=EDGE,
            lw=0.8, ls=ls, zorder=z))
        if text:
            ty = y + h / 2 + dy if va == "center" else y + h - 1.2 + dy
            self.ax.text(x + w / 2, ty, text, ha="center", va=va,
                         fontsize=self.fs * size, weight=weight, zorder=z + 1,
                         linespacing=1.15)
        return x, y, w, h

    def text(self, x, y, s, size=1.0, **kw):
        """Text in data units at a fraction ``size`` of the base font size."""
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        self.ax.text(x, y, s, fontsize=self.fs * size, zorder=5, linespacing=1.15,
                     **kw)

    def arrow(self, p0, p1, color=EDGE, rad=0.0, lw=0.9, ls="-"):
        """Arrow from p0 to p1 (data units), optionally curved."""
        self.ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
            shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=4))

    def elbow(self, pts, color=EDGE, lw=0.9):
        """Polyline through ``pts`` with an arrow head on the last segment."""
        xs, ys = zip(*pts)
        self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=4,
                     solid_capstyle="butt")
        self.arrow(pts[-2], pts[-1], color=color, lw=lw)


def draw(c, colors):
    """The whole schematic."""
    fs_s = 0.8  # small text
    mid = lambda b: (b[0] + b[2] / 2, b[1] + b[3] / 2)  # noqa: E731
    right = lambda b, f=0.5: (b[0] + b[2], b[1] + f * b[3])  # noqa: E731
    left = lambda b, f=0.5: (b[0], b[1] + f * b[3])  # noqa: E731
    top = lambda b, f=0.5: (b[0] + f * b[2], b[1] + b[3])  # noqa: E731
    bottom = lambda b, f=0.5: (b[0] + f * b[2], b[1])  # noqa: E731

    # ---------------- Inputs ----------------
    ctx = c.box(1, 30, 25, 52, fc=FILL["inp"])
    c.text(ctx[0] + ctx[2] / 2, 79, "context table", weight="bold")
    c.text(ctx[0] + ctx[2] / 2, 75.4, r"${\leq}\,12$ detections," "\nlast 2 d", size=fs_s)
    # Mini table: columns m, tau, valid, filter.
    tx0, ty0, cw, rh = 3.2, 35.0, 5.1, 2.6
    heads = (r"$m$", r"$\tau$", "valid", "filter")
    for j, hname in enumerate(heads):
        c.text(tx0 + (j + 0.5) * cw, ty0 + N_SLOTS * rh + 1.6, hname, size=0.72)
    rng = np.random.default_rng(3)
    for r in range(N_SLOTS):
        y = ty0 + (N_SLOTS - 1 - r) * rh
        ok = r < len(EX_BANDS)
        for j in range(4):
            fc = "white" if ok else "0.85"
            if ok and j == 3:
                fc = colors[EX_BANDS[r]]
            elif ok and j == 0:
                fc = plt.cm.viridis_r(0.25 + 0.5 * rng.random())
            elif ok and j == 1:
                fc = plt.cm.Greys(0.1 + 0.5 * r / len(EX_BANDS))
            c.ax.add_patch(Rectangle((tx0 + j * cw, y), cw, rh, fc=fc, ec="0.55",
                                     lw=0.35, zorder=3))
        if ok:
            c.text(tx0 + 2.5 * cw, y + rh / 2, "1", size=0.6)
    c.text(tx0 + 2 * cw, ty0 - 2.0, "gray: empty slot", size=0.68, color="0.35")

    gl = c.box(1, 4, 25, 20, fc=FILL["inp"])
    c.text(gl[0] + gl[2] / 2, 19.2, "query time", weight="bold")
    c.text(gl[0] + gl[2] / 2, 11.0,
           "lead time $\\Delta t$ (17)\ntime since\ndiscovery (21)\nFourier features",
           size=fs_s)

    # ---------------- Transformer lane ----------------
    emb = c.box(36, 66, 19, 13, fc=FILL["vit"])
    c.text(*mid(emb), "event\nembedding\n2-layer MLP", size=fs_s)
    qry = c.box(36, 35, 19, 17, fc=FILL["vit"])
    c.text(*mid(qry), "filter queries\nlearned vector\nper filter\n+ linear map of\n"
           "query time", size=fs_s)
    c.arrow((ctx[0] + ctx[2], 72.5), left(emb))
    c.elbow([(gl[0] + gl[2], 20.0), (29.0, 20.0), (29.0, 41.5), (qry[0], 41.5)])

    # Token column: events then queries, 2304 features each.
    tw, th, gap = 3.2, 2.15, 0.45
    tx = 60.5
    n_ev = len(EX_BANDS)
    ys_ev = [81.0 - (k + 1) * (th + gap) for k in range(n_ev)]
    ys_q = [ys_ev[-1] - 2.2 - (k + 1) * (th + gap) for k in range(9)]
    for k, y in enumerate(ys_ev):
        c.ax.add_patch(Rectangle((tx, y), tw, th, fc=colors[EX_BANDS[k]], ec="k",
                                 lw=0.4, zorder=3))
    for k, y in enumerate(ys_q):
        c.ax.add_patch(Rectangle((tx, y), tw, th, fc="white", ec=colors[k], lw=1.1,
                                 hatch="//////", zorder=3))
    c.text(tx + tw + 1.0, (ys_ev[0] + ys_ev[-1] + th) / 2, "event\ntokens\n(max 12)",
           size=0.72, ha="left")
    c.text(tx + tw + 1.0, (ys_q[0] + ys_q[-1] + th) / 2, "query\ntokens\n(9)",
           size=0.72, ha="left")
    c.text(tx + tw / 2, ys_q[-1] - 2.4, "2304 features\neach", size=0.68,
           va="top")
    c.arrow(right(emb), (tx - 0.6, right(emb)[1]))
    c.arrow(right(qry), (tx - 0.6, right(qry)[1]))

    # RoPE positions above the blocks.
    rope = c.box(76, 74, 24, 12, fc="white", ls="--")
    c.text(*mid(rope), "RoPE positions\ncolumn: filter\n"
           r"row: time, $\tau$ (events)" "\n" r"or $\Delta t$ (queries)", size=0.68)

    # Six stacked transformer blocks.
    bx, by, bw, bh = 76, 32, 24, 36
    for k in range(5, -1, -1):
        c.box(bx + 0.9 * k, by - 0.9 * k, bw, bh, fc=FILL["vit"], z=2 + (5 - k) * 0.01)
    c.text(bx + bw / 2, by + bh - 4.5, "6 transformer\nblocks", weight="bold")
    c.text(bx + bw / 2, by + bh / 2 - 3.5,
           "self-attention\nand MLP in parallel\n12 heads\n\nempty slots\nmasked",
           size=fs_s)
    c.arrow(bottom(rope), (bx + bw / 2, by + bh))
    brk = tx + tw + 10.0
    c.ax.plot([brk - 1.0, brk, brk, brk - 1.0], [ys_ev[0] + th, ys_ev[0] + th,
              ys_q[-1], ys_q[-1]], color=EDGE, lw=0.9, zorder=4)
    c.arrow((brk, 56.0), (bx, 56.0))

    rd = c.box(108, 48, 19, 16, fc=FILL["vit"])
    c.text(*mid(rd), "readout\nquery outputs\nonly, 8 features\neach (72)", size=fs_s)
    c.arrow((bx + bw + 4.5, 56.0), left(rd))

    # ---------------- Concatenation ----------------
    cat = c.box(108, 8, 19, 22, fc="white")
    c.text(*mid(cat), "concatenate\nreadout (72)\n+ $\\Delta t$ (17)", size=fs_s)
    c.arrow(bottom(rd), top(cat))
    # The head reads the Dt features again directly (head_in, bomberman.py).
    c.arrow((gl[0] + gl[2], 12.0), (cat[0], 12.0))
    c.text((gl[0] + gl[2] + cat[0]) / 2, 14.2, r"$\Delta t$ features", size=0.72)

    # ---------------- Output head ----------------
    hx, hw = 133, 26
    nw = hw - 4.5  # network boxes leave a corridor on the right
    lvl = c.box(hx, 61, nw, 17, fc=FILL["head"])
    c.text(*mid(lvl), "level network\nshared by all\nfilters, one per\npercentile",
           size=fs_s)
    code = c.box(hx, 37, nw, 17, fc=FILL["head"])
    c.text(*mid(code), "color network\n2-number code\n× filter response\n(9 × 2)",
           size=fs_s)
    out = c.box(hx, 4, hw, 26, fc=FILL["head"])
    c.text(out[0] + out[2] / 2, 26.0, "forecast", weight="bold")

    # 9 x 3 output grid: filters x percentiles.
    gx0, gy0, gcw, grh = hx + 2.4, 6.0, 2.45, 2.0
    for b in range(9):
        for q in range(3):
            c.ax.add_patch(Rectangle((gx0 + b * gcw, gy0 + q * grh), gcw, grh,
                                     fc=colors[b], alpha=(0.45, 1.0, 0.45)[q],
                                     ec="w", lw=0.4, zorder=3))
    c.text(hx + hw / 2, gy0 + 3 * grh + 5.2, "9 filters × 3 percentiles\n"
           r"$\mu_b + \sigma_b\,\times$ (level + color$_b$)", size=0.64)
    c.elbow([right(cat), (129.0, right(cat)[1]), (129.0, mid(lvl)[1]),
             left(lvl)])
    c.arrow((129.0, mid(code)[1]), left(code))
    c.elbow([right(lvl), (hx + hw - 1.8, right(lvl)[1]),
             (hx + hw - 1.8, out[1] + out[3])])
    c.arrow(bottom(code), (code[0] + code[2] / 2, out[1] + out[3]))


def main() -> None:
    """Render the schematic."""
    args = get_args()
    use_pub_style(args.style, args.font_dir)
    colors = pub_band_colors(HOUSE_BAND_COLORS)
    fs = 0.9 * pub_fs(8, "xtick.labelsize")
    fig = plt.figure(figsize=(W_IN, W_IN * Y_MAX / X_MAX), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    draw(Canvas(ax, fs), colors)
    os.makedirs(args.outdir, exist_ok=True)
    for fmt in args.formats:
        path = os.path.join(args.outdir, f"{args.name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
