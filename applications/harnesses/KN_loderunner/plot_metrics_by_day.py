r"""Per-day forecast MSE, pinball loss and 90% coverage, per band and pooled.

Reads the scored-points CSVs written by ``eval_dense_latetime_9band.py`` (no
model / GPU / torch) for one or more studies and plots, in 1-day bins of
light-curve day (phase since first detection; ``--x lead`` for lead time):

  * MSE of the median forecast (mag^2),
  * mean pinball loss over the 0.05 / 0.5 / 0.95 quantiles (mag; the training
    loss, unweighted) -- scores the whole predicted distribution, not just the
    median,
  * coverage of the 90% interval (target 0.90).

One panel per band plus ``all`` (every band's points pooled, like the eval's
overall RMSE: sqrt of the ``all`` MSE over every day is the eval's number).
The interval is the calibrated one when the eval dir has
``interval_calibration.json`` (``--band raw`` forces the raw band). Calibration
only rescales the outer quantiles, so MSE is the same either way.

Shading = object-bootstrap 90% CI (points of one object are correlated, so
objects are resampled). It is test-sample noise only, NOT seed noise (~0.05
RMSE). With several studies the bootstrap weights are shared, so the delta
tables are paired.

    python plot_metrics_by_day.py --studies 138
    python plot_metrics_by_day.py --studies 138 140 141 --truth uniform
    # -> runs/metrics_by_day_compare/study138_140_141_mse_by_phase_uniform.png, ...
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_interval_coverage import (  # noqa: E402
    BAND_COLORS,
    BAND_NAMES,
    MIN_BIN_POINTS,
    STUDY_COLORS,
    ObjectBootstrap,
    _plot_series,
    _save,
    add_variants,
    load_points,
    load_scales,
    point_scores,
)

GROUPS = BAND_NAMES + ("all",)
# name -> (point_scores key, axis label, reference line, table format)
METRICS = {
    "mse": ("sq_resid", "MSE of the median (mag$^2$)", None, "{:>7.3f}"),
    "pinball": ("pinball", "mean pinball loss, q = .05/.5/.95 (mag)", None,
                "{:>7.4f}"),
    "coverage": ("covered", "90% interval coverage", 0.90, "{:>7.3f}"),
}
CSV_NAMES = {"dense": "latetime_scored_points.csv",
             "uniform": "latetime_uniform_scored_points.csv"}
X_COLUMNS = {"phase": ("phase_days", "day since first detection", 2.0, 10.0),
             "lead": ("lead_time_days", "lead time (d)", 0.0, 10.0)}


def day_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per point (-1 = outside); the last bin is closed on the right."""
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = len(edges) - 2
    idx[(x < edges[0]) | (x > edges[-1])] = -1
    return idx


def group_mask(df, g: str) -> np.ndarray:
    """Points of band ``g``, or every point for ``all``."""
    if g == "all":
        return np.ones(len(df), dtype=bool)
    return (df["band"] == g).to_numpy()


def load_entry(eval_dir, truth, band, x_col, edges) -> dict | None:
    """One study's scored points for ``truth`` with per-point scores and bins."""
    path = os.path.join(eval_dir, CSV_NAMES[truth])
    if not os.path.exists(path):
        print(f"  no {path}; skipping this study for {truth} truth")
        return None
    df = load_points(path, extra_cols=(x_col,) if x_col != "lead_time_days"
                     else ())
    x = df["lead" if x_col == "lead_time_days" else x_col].to_numpy()
    scales = (load_scales(os.path.join(eval_dir, "interval_calibration.json"))
              if band == "cal" else None)
    v = add_variants(df, scales)[-1]
    print(f"  {path}: {len(df)} points, {df['stem'].nunique()} objects, "
          f"{v} band")
    return {"df": df, "v": v, "sc": point_scores(df, [v])[v],
            "idx": day_bins(x, edges)}


def by_day(e, edges) -> tuple:
    """Per group and metric: per-bin rows and the all-days mean.

    Returns:
        tuple: ``res[(group, metric)]`` = rows ``(bin center, mean, ci_lo,
        ci_hi, n)`` for bins with >= ``MIN_BIN_POINTS`` points, and
        ``overall[(group, metric)]`` = mean over every in-window point.
    """
    res, overall = {}, {}
    for g in GROUPS:
        gm = group_mask(e["df"], g)
        for metric, (key, _, _, _) in METRICS.items():
            val = e["sc"][key]
            rows = []
            for k in range(len(edges) - 1):
                m = gm & (e["idx"] == k)
                if m.sum() < MIN_BIN_POINTS:
                    continue
                rows.append((0.5 * (edges[k] + edges[k + 1]),
                             *e["boot"].mean_ci(m, val), int(m.sum())))
            res[(g, metric)] = np.array(rows).reshape(-1, 5)
            inwin = gm & (e["idx"] >= 0)
            overall[(g, metric)] = (float(val[inwin].mean()) if inwin.any()
                                    else np.nan)
    return res, overall


def _grid(title: str, sharey: bool):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4), sharex=True, sharey=sharey)
    fig.suptitle(title)
    return fig, axes.ravel()


def plot_metric(entries, metric, truth, xlabel, path) -> None:
    """One panel per band + pooled; one line per study (band color if one)."""
    _, ylabel, ref, _ = METRICS[metric]
    # Coverage shares one y-scale; MSE / pinball differ ~4x across bands.
    fig, axes = _grid(f"{ylabel} by day ({truth} truth); shading = "
                      "object-bootstrap 90% CI", sharey=ref is not None)
    for i, g in enumerate(GROUPS):
        ax = axes[i]
        if ref is not None:
            ax.axhline(ref, color="0.4", lw=0.8)
        for e in entries:
            if len(entries) > 1:
                color = e["color"]
            else:
                color = "k" if g == "all" else BAND_COLORS[i]
            _plot_series(ax, e["res"][(g, metric)], color, "-", e["label"],
                         marker="o")
        if ref is None:
            ax.set_ylim(bottom=0.0)
        ax.set_title("all bands (pooled)" if g == "all" else g)
    for ax in axes[5:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::5]:
        ax.set_ylabel(ylabel)
    if len(entries) > 1:
        axes[0].legend(loc="best", fontsize=8)
    _save(fig, path)


def plot_overview(e, truth, xlabel, path) -> None:
    """One study: the three metrics side by side, one line per band + pooled."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(f"Study {e['label']} by day ({truth} truth); black = all "
                 "bands pooled, dashed = ZTF")
    for ax, (metric, (_, ylabel, ref, _)) in zip(axes, METRICS.items()):
        if ref is not None:
            ax.axhline(ref, color="0.4", lw=0.8)
        for i, g in enumerate(GROUPS):
            pts = e["res"][(g, metric)]
            if pts.size == 0:
                continue
            pooled = g == "all"
            # ZTF dashed: ztfg/ztfr share near-identical hues with g/r.
            ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3,
                    color="k" if pooled else BAND_COLORS[i],
                    ls="--" if g.startswith("ztf") else "-",
                    lw=2.4 if pooled else 1.3, label="all" if pooled else g)
        if ref is None:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    _save(fig, path)


def _plain(label: str) -> str:
    """Axis label without matplotlib mathtext, for the printed tables."""
    return label.replace("$^2$", "^2")


def print_tables(entries, truth, edges) -> None:
    """Per metric and study: band x day-bin means, plus the all-days mean."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    for metric, (_, ylabel, _, fmt) in METRICS.items():
        for e in entries:
            print(f"\n[{truth}] {e['label']}: {_plain(ylabel)}, by day-bin center "
                  f"(bins with >= {MIN_BIN_POINTS} points)")
            print(f"  {'band':<6}" + "".join(f"{c:>7.1f}" for c in centers)
                  + f"{'all d':>8}")
            for g in GROUPS:
                pts = e["res"][(g, metric)]
                by_c = dict(zip(pts[:, 0], pts[:, 1]))
                line = f"  {g:<6}" + "".join(
                    fmt.format(by_c[c]) if c in by_c else f"{'-':>7}"
                    for c in centers)
                print(line + " " + fmt.format(e["overall"][(g, metric)]))
            if metric == "mse":
                print(f"  (sqrt of the all / all-days MSE = RMSE "
                      f"{np.sqrt(e['overall'][('all', 'mse')]):.4f})")


def print_deltas(entries, truth, edges) -> None:
    """Pooled (all-bands) delta vs the first study per day bin, paired CI."""
    base = entries[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    for metric, (key, ylabel, _, _) in METRICS.items():
        print(f"\n[{truth}] all bands pooled, {_plain(ylabel)}: study - "
              f"{base['label']} per day bin (* = paired 90% CI excludes 0)")
        print(f"  {'study':<8}" + "".join(f"{c:>9.1f}" for c in centers))
        for e in entries[1:]:
            line = f"  {e['label']:<8}"
            for k in range(len(edges) - 1):
                m0 = base["idx"] == k
                m = e["idx"] == k
                if m0.sum() < MIN_BIN_POINTS or m.sum() < MIN_BIN_POINTS:
                    line += f"{'-':>9}"
                    continue
                r0, b0 = base["boot"].boot_means(m0, base["sc"][key])
                r, bt = e["boot"].boot_means(m, e["sc"][key])
                d = (bt - b0)[np.isfinite(bt - b0)]
                lo, hi = np.percentile(d, [5, 95])
                star = "*" if lo > 0 or hi < 0 else " "
                line += f"{r - r0:>+8.3f}{star}"
            print(line)


def main() -> None:
    """Load each study's scored points and write the by-day plots + tables."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", type=int, nargs="+", default=None,
                   help=f"1-{len(STUDY_COLORS)} study numbers (eval dirs "
                   "<runs_dir>/study_NNN/dense_latetime_eval_9band); the first "
                   "is the delta baseline.")
    p.add_argument("--eval_dir", default=None,
                   help="A single eval dir instead of --studies.")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--truth", choices=("dense", "uniform", "both"), default="both")
    p.add_argument("--x", choices=tuple(X_COLUMNS), default="phase",
                   help="Bin by light-curve day (phase) or by lead time.")
    p.add_argument("--bin_days", type=float, default=1.0)
    p.add_argument("--band", choices=("cal", "raw"), default="cal",
                   help="Interval for pinball / coverage (cal falls back to "
                   "raw without a calibration JSON).")
    p.add_argument("--outdir", default=None,
                   help="Default: <eval_dir>/metrics_by_day for one study, "
                   "<runs_dir>/metrics_by_day_compare for several.")
    p.add_argument("--n_boot", type=int, default=200,
                   help="Object-bootstrap resamples for the CIs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.eval_dir is not None:
        dirs, labels = [args.eval_dir], ["eval"]
    elif args.studies:
        if len(args.studies) > len(STUDY_COLORS):
            p.error(f"--studies takes at most {len(STUDY_COLORS)} studies")
        dirs = [os.path.join(args.runs_dir, f"study_{s:03d}",
                             "dense_latetime_eval_9band") for s in args.studies]
        labels = [f"{s:03d}" for s in args.studies]
    else:
        p.error("give --studies or --eval_dir")
    if args.outdir:
        outdir = args.outdir
    elif len(dirs) == 1:
        outdir = os.path.join(dirs[0], "metrics_by_day")
    else:
        outdir = os.path.join(args.runs_dir, "metrics_by_day_compare")
    os.makedirs(outdir, exist_ok=True)
    prefix = ("study" + "_".join(labels) + "_") if args.studies else ""

    x_col, xlabel, lo, hi = X_COLUMNS[args.x]
    edges = np.round(np.arange(lo, hi + 0.5 * args.bin_days, args.bin_days), 6)
    for truth in [t for t in ("dense", "uniform") if args.truth in (t, "both")]:
        print(f"\n=== {truth} truth ===")
        entries = []
        for i, (d, label) in enumerate(zip(dirs, labels)):
            e = load_entry(d, truth, args.band, x_col, edges)
            if e is not None:
                e.update(label=label, color=STUDY_COLORS[i])
                entries.append(e)
        if not entries:
            continue
        # One object universe + seed -> the same bootstrap weights per study.
        universe = np.unique(np.concatenate(
            [e["df"]["stem"].to_numpy() for e in entries]))
        for e in entries:
            e["boot"] = ObjectBootstrap(e["df"]["stem"].to_numpy(), args.n_boot,
                                        args.seed, universe)
            e["res"], e["overall"] = by_day(e, edges)
        for metric in METRICS:
            plot_metric(entries, metric, truth, xlabel, os.path.join(
                outdir, f"{prefix}{metric}_by_{args.x}_{truth}.png"))
        for e in entries:
            own = f"study{e['label']}_" if args.studies else ""
            plot_overview(e, truth, xlabel, os.path.join(
                outdir, f"{own}overview_by_{args.x}_{truth}.png"))
        print_tables(entries, truth, edges)
        if len(entries) > 1:
            print_deltas(entries, truth, edges)


if __name__ == "__main__":
    main()
