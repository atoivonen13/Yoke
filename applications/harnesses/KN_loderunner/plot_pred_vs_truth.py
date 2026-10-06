"""Scatter of the predicted median vs the uniform truth, points colored by day.

Reads ``latetime_uniform_scored_points.csv`` written by
``eval_dense_latetime_9band.py`` (no model / GPU / torch). Per study it writes

  * ``*_bands.png`` -- one axis per band (3x3),
  * ``*_all.png``   -- every band's points on one axis,

with every day on the same axis, colored by 1-day bin of light-curve day
(phase since first detection; ``--x lead`` for lead time). Each axis shows:

  * the points (randomly thinned to ``--max_points`` per axis and drawn in
    random order, so no day is painted over the others; the numbers use every
    point),
  * the y = x diagonal,
  * per day bin, the median prediction in 0.5-mag bins of truth (line in that
    day's color),
  * the band's dense depth (dotted; 99th percentile of the dense scored truth,
    as the eval) -- to its right the truth is fainter than the survey ever
    observed, so a flat median there is the beyond-depth plateau.

The median-given-truth lines have slope < 1 even for a perfect forecaster
(regression to the mean when conditioning on truth), so compare them across
days / studies, not against the diagonal. RMSE and bias = mean(pred - true)
(< 0 = predicted too bright) per band x day are printed.

    python plot_pred_vs_truth.py --studies 138
    python plot_pred_vs_truth.py --studies 138 140 141 --x lead
    # -> runs/study_138/dense_latetime_eval_9band/pred_vs_truth/
    #    study138_pred_vs_truth_by_phase_bands.png, ..._all.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from plot_interval_coverage import (  # noqa: E402
    BAND_NAMES,
    MIN_BIN_POINTS,
    _save,
    dense_depths,
    load_points,
)
from plot_metrics_by_day import X_COLUMNS, day_bins  # noqa: E402

TRUTH_BIN_MAG = 0.5
MIN_TRUTH_BIN_POINTS = 20
CMAP = plt.get_cmap("viridis")


def _limits(df) -> tuple:
    vals = np.concatenate([df["true"].to_numpy(), df["med"].to_numpy()])
    lo, hi = np.percentile(vals, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _median_given_truth(true, pred, lims) -> np.ndarray:
    """Rows ``(truth bin center, median pred)``, truth bins with enough points."""
    edges = np.arange(np.floor(lims[0]), lims[1] + TRUTH_BIN_MAG, TRUTH_BIN_MAG)
    idx = np.digitize(true, edges) - 1
    rows = [(0.5 * (edges[k] + edges[k + 1]), np.median(pred[idx == k]))
            for k in range(len(edges) - 1)
            if (idx == k).sum() >= MIN_TRUTH_BIN_POINTS]
    return np.array(rows).reshape(-1, 2)


def day_colors(edges) -> tuple:
    """Discrete viridis over the day bins: ``(norm, per-bin colors)``."""
    norm = BoundaryNorm(edges, CMAP.N)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return norm, CMAP(norm(centers))


def scatter_by_day(ax, df, idx, edges, colors, lims, depth, max_points,
                   rng) -> list:
    """All day bins on ``ax``; returns rows ``(bin center, n, rmse, bias)``."""
    true, pred = df["true"].to_numpy(), df["med"].to_numpy()
    ax.plot(lims, lims, color="k", lw=0.8, zorder=1)
    if depth is not None:
        ax.axvline(depth, color="0.3", ls=":", lw=1.0, zorder=1)
    inwin = np.flatnonzero(idx >= 0)
    draw = (rng.choice(inwin, max_points, replace=False)
            if inwin.size > max_points else rng.permutation(inwin))
    ax.scatter(true[draw], pred[draw], s=3, c=colors[idx[draw]], alpha=0.4,
               lw=0, rasterized=True, zorder=2)
    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() < MIN_BIN_POINTS:
            continue
        resid = pred[m] - true[m]
        rows.append((0.5 * (edges[k] + edges[k + 1]), int(m.sum()),
                     float(np.sqrt(np.mean(resid ** 2))), float(resid.mean())))
        med = _median_given_truth(true[m], pred[m], lims)
        if med.size:
            # White underlay keeps the day's line readable over its own points.
            ax.plot(med[:, 0], med[:, 1], color="white", lw=3.2, zorder=3)
            ax.plot(med[:, 0], med[:, 1], color=colors[k], lw=1.8, zorder=4)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    return rows


def _colorbar(fig, axes, norm, edges, label) -> None:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02, ticks=edges)
    cb.set_label(label)


def run_study(eval_dir, label, args) -> None:
    """The per-band grid + the pooled single-axis figure for one eval dir."""
    path = os.path.join(eval_dir, "latetime_uniform_scored_points.csv")
    if not os.path.exists(path):
        print(f"no {path}; skipping")
        return
    x_col, xlabel, lo, hi = X_COLUMNS[args.x]
    df = load_points(path, extra_cols=(x_col,) if x_col != "lead_time_days"
                     else ())
    x = df["lead" if x_col == "lead_time_days" else x_col].to_numpy()
    edges = np.round(np.arange(lo, hi + 0.5 * args.bin_days, args.bin_days), 6)
    idx = day_bins(x, edges)
    print(f"{path}: {len(df)} points, {df['stem'].nunique()} objects")
    dense_csv = os.path.join(eval_dir, "latetime_scored_points.csv")
    depths = (dense_depths(load_points(dense_csv)) if os.path.exists(dense_csv)
              else {})
    outdir = args.outdir or os.path.join(eval_dir, "pred_vs_truth")
    os.makedirs(outdir, exist_ok=True)
    prefix = f"study{label}_" if label else ""
    tag = f"study {label}: " if label else ""
    rng = np.random.default_rng(args.seed)
    norm, colors = day_colors(edges)
    lines = "lines = median prediction per 0.5-mag truth bin, per day"

    stats = {}
    # Constrained layout: leaves room for the shared colorbar.
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12), sharex=True, sharey=True,
                             layout="constrained")
    fig.suptitle(f"{tag}predicted median vs uniform truth, colored by {xlabel};"
                 f"\n{lines}; dotted = dense depth")
    axes = axes.ravel()
    lims = _limits(df)
    for b, name in enumerate(BAND_NAMES):
        ax = axes[b]
        bm = (df["band"] == name).to_numpy()
        ax.set_title(name)
        if not bm.any():
            continue
        stats[name] = scatter_by_day(
            ax, df[bm].reset_index(drop=True), idx[bm], edges, colors, lims,
            depths.get(name), args.max_points, rng)
    for ax in axes[6:]:
        ax.set_xlabel("uniform truth (mag)")
    for ax in axes[::3]:
        ax.set_ylabel("predicted median (mag)")
    _colorbar(fig, axes, norm, edges, xlabel)
    out = os.path.join(outdir, f"{prefix}pred_vs_truth_by_{args.x}_bands.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    fig.suptitle(f"{tag}all bands, predicted median vs uniform truth, colored "
                 f"by {xlabel}\n{lines}")
    stats["all"] = scatter_by_day(ax, df, idx, edges, colors, lims, None,
                                  4 * args.max_points, rng)
    ax.set_xlabel("uniform truth (mag)")
    ax.set_ylabel("predicted median (mag)")
    _colorbar(fig, ax, norm, edges, xlabel)
    _save(fig, os.path.join(outdir, f"{prefix}pred_vs_truth_by_{args.x}_all.png"))
    print_stats(stats, edges, label)


def print_stats(stats, edges, label) -> None:
    """RMSE and bias per band x day bin (every point, not just the drawn ones)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    for j, what in ((2, "RMSE"), (3, "bias = mean(pred - true)")):
        print(f"\n[{label or 'eval'}] uniform truth {what}, by day-bin center")
        print(f"  {'band':<6}" + "".join(f"{c:>7.1f}" for c in centers))
        for name, rows in stats.items():
            by_c = {r[0]: r[j] for r in rows}
            fmt = "{:>+7.3f}" if j == 3 else "{:>7.3f}"
            print(f"  {name:<6}" + "".join(
                fmt.format(by_c[c]) if c in by_c else f"{'-':>7}"
                for c in centers))


def main() -> None:
    """Write the per-band and pooled pred-vs-truth scatters for each study."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", type=int, nargs="+", default=None,
                   help="Study numbers (eval dirs <runs_dir>/study_NNN/"
                   "dense_latetime_eval_9band); each gets its own figures.")
    p.add_argument("--eval_dir", default=None,
                   help="A single eval dir instead of --studies.")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--x", choices=tuple(X_COLUMNS), default="phase",
                   help="Color by light-curve day (phase) or by lead time.")
    p.add_argument("--bin_days", type=float, default=1.0)
    p.add_argument("--max_points", type=int, default=6000,
                   help="Points drawn per band axis (4x on the pooled axis); "
                   "stats use all points.")
    p.add_argument("--outdir", default=None,
                   help="Default: <eval_dir>/pred_vs_truth.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.eval_dir is not None:
        run_study(args.eval_dir, "", args)
    elif args.studies:
        for s in args.studies:
            run_study(os.path.join(args.runs_dir, f"study_{s:03d}",
                                   "dense_latetime_eval_9band"), f"{s:03d}", args)
    else:
        p.error("give --studies or --eval_dir")


if __name__ == "__main__":
    main()
