"""Scatter of the predicted median vs the uniform truth, one panel per day bin.

Reads ``latetime_uniform_scored_points.csv`` written by
``eval_dense_latetime_9band.py`` (no model / GPU / torch). Per study it writes
one figure per band plus one with every band pooled; each figure has one
panel per 1-day bin of light-curve day (phase since first detection, titled
"day a-b d"; ``--x lead`` for lead time, "lead a-b d"). Each panel shows:

  * the points (randomly thinned to ``--max_points`` for drawing; the numbers
    use every point),
  * the y = x diagonal,
  * the median prediction in 0.5-mag bins of truth, with its 5-95% spread
    (black line + grey band),
  * the band's dense depth (dotted; 99th percentile of the dense scored truth,
    as the eval) -- to its right the truth is fainter than the survey ever
    observed, so a flat median there is the beyond-depth plateau,
  * n, RMSE and bias = mean(pred - true) (< 0 = predicted too bright).

The median-given-truth line has slope < 1 even for a perfect forecaster
(regression to the mean when conditioning on truth), so judge it against the
other days / studies, not against the diagonal.

    python plot_pred_vs_truth.py --studies 138
    python plot_pred_vs_truth.py --studies 138 140 141 --x lead
    # -> runs/study_138/dense_latetime_eval_9band/pred_vs_truth/
    #    study138_pred_vs_truth_by_phase_ztfg.png, ..._all.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plot_interval_coverage import (  # noqa: E402
    BAND_COLORS,
    BAND_NAMES,
    MIN_BIN_POINTS,
    dense_depths,
    load_points,
)
from plot_metrics_by_day import X_COLUMNS, day_bins  # noqa: E402

TRUTH_BIN_MAG = 0.5


def _grid(n_panels: int, title: str):
    ncols = min(n_panels, 4 if n_panels <= 8 else 5)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(title)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes[:n_panels], ncols


def _limits(df) -> tuple:
    vals = np.concatenate([df["true"].to_numpy(), df["med"].to_numpy()])
    lo, hi = np.percentile(vals, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _median_given_truth(true, pred, lims) -> np.ndarray:
    """Rows ``(truth bin center, median pred, p5, p95)``, bins >= 20 points."""
    edges = np.arange(np.floor(lims[0]), lims[1] + TRUTH_BIN_MAG, TRUTH_BIN_MAG)
    idx = np.digitize(true, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        p = pred[idx == k]
        if p.size >= 20:
            rows.append((0.5 * (edges[k] + edges[k + 1]),
                         *np.percentile(p, [50, 5, 95])))
    return np.array(rows).reshape(-1, 4)


def plot_by_day(df, idx, edges, colors, depth, lims, title, panel_label, path,
                max_points, rng, band_legend=False) -> list:
    """One panel per day bin; ``colors`` = per-point colors. Returns stat rows."""
    n_bins = len(edges) - 1
    fig, axes, ncols = _grid(n_bins, title)
    rows = []
    true_all, pred_all = df["true"].to_numpy(), df["med"].to_numpy()
    for k, ax in enumerate(axes):
        m = idx == k
        ax.set_title(f"{panel_label} {edges[k]:g}-{edges[k + 1]:g} d",
                     fontsize=10)
        ax.plot(lims, lims, color="k", lw=0.8)
        if depth is not None:
            ax.axvline(depth, color="0.3", ls=":", lw=1.0)
        if m.sum() < MIN_BIN_POINTS:
            ax.text(0.5, 0.5, f"n={int(m.sum())}", transform=ax.transAxes,
                    ha="center")
            continue
        true, pred = true_all[m], pred_all[m]
        draw = np.flatnonzero(m)
        if draw.size > max_points:
            draw = rng.choice(draw, max_points, replace=False)
        ax.scatter(true_all[draw], pred_all[draw], s=3, c=colors[draw],
                   alpha=0.35, lw=0, rasterized=True)
        med = _median_given_truth(true, pred, lims)
        if med.size:
            ax.fill_between(med[:, 0], med[:, 2], med[:, 3], color="0.5",
                            alpha=0.25, lw=0)
            ax.plot(med[:, 0], med[:, 1], color="k", lw=1.6)
        resid = pred - true
        rmse, bias = float(np.sqrt(np.mean(resid ** 2))), float(resid.mean())
        rows.append((0.5 * (edges[k] + edges[k + 1]), int(m.sum()), rmse, bias))
        ax.text(0.03, 0.97, f"n={int(m.sum())}\nRMSE {rmse:.3f}\nbias {bias:+.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    for ax in axes:
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect("equal")
    for i, ax in enumerate(axes):
        if i + ncols >= len(axes):
            ax.set_xlabel("uniform truth (mag)")
            ax.xaxis.set_tick_params(labelbottom=True)
        if i % ncols == 0:
            ax.set_ylabel("predicted median (mag)")
    if band_legend:
        # ZTF hues are close to g / r; the per-band figures separate them.
        handles = [Line2D([], [], ls="", marker="o", ms=5, color=c, label=n)
                   for n, c in zip(BAND_NAMES, BAND_COLORS)]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False)
    # Not plot_interval_coverage._save: leave room at the bottom for the legend.
    fig.tight_layout(rect=(0, 0.04 if band_legend else 0, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")
    return rows


def run_study(eval_dir, label, args) -> None:
    """Every band's figure + the pooled one for one eval dir."""
    path = os.path.join(eval_dir, "latetime_uniform_scored_points.csv")
    if not os.path.exists(path):
        print(f"no {path}; skipping")
        return
    x_col, _, lo, hi = X_COLUMNS[args.x]
    panel_label = "day" if args.x == "phase" else "lead"
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
    rng = np.random.default_rng(args.seed)
    color_of = dict(zip(BAND_NAMES, BAND_COLORS))
    tag = f"study {label}, " if label else ""

    stats = {}
    for name in BAND_NAMES:
        bm = (df["band"] == name).to_numpy()
        if not bm.any():
            continue
        sub = df[bm].reset_index(drop=True)
        depth = depths.get(name)
        note = (f"dotted = dense depth {depth:.2f}" if depth is not None
                else "no dense CSV -> no depth line")
        stats[name] = plot_by_day(
            sub, idx[bm], edges, np.full(len(sub), color_of[name]), depth,
            _limits(sub), f"{name}: predicted median vs uniform truth by "
            f"{panel_label} ({tag}{note}; black = median prediction per "
            "0.5-mag truth bin)", panel_label,
            os.path.join(outdir, f"{prefix}pred_vs_truth_by_{args.x}_{name}.png"),
            args.max_points, rng)
    colors = df["band"].map(color_of).to_numpy()
    stats["all"] = plot_by_day(
        df, idx, edges, colors, None, _limits(df),
        f"All bands: predicted median vs uniform truth by "
        f"{panel_label} ({tag}"
        "black = median prediction per 0.5-mag truth bin)", panel_label,
        os.path.join(outdir, f"{prefix}pred_vs_truth_by_{args.x}_all.png"),
        args.max_points, rng, band_legend=True)
    print_stats(stats, edges, label)


def print_stats(stats, edges, label) -> None:
    """RMSE and bias per band x day bin (same numbers as the panel boxes)."""
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
                   help="Bin by light-curve day (phase) or by lead time.")
    p.add_argument("--bin_days", type=float, default=1.0)
    p.add_argument("--max_points", type=int, default=4000,
                   help="Points drawn per panel (stats use all points).")
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
