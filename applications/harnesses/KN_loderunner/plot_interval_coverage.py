"""Calibration diagnostics for the quantile-head forecast interval.

Reads the scored-points CSVs written by ``eval_dense_latetime_9band.py`` (no
model / GPU / torch) and plots, per band, raw vs calibrated:

  1. PP plot -- nominal quantile level tau vs the empirical fraction of truths
     at or below the predicted tau-quantile. The head learns only three
     quantiles (0.05, 0.5, 0.95), so only those three points are exact (drawn as
     markers with object-bootstrap 90% CIs). The continuous curve interpolates
     with a split Gaussian through the three learned quantiles:

         F(y) = Phi((y - med) / s_lo)  for y <  med,   s_lo = (med - low)  / z95
         F(y) = Phi((y - med) / s_hi)  for y >= med,   s_hi = (high - med) / z95

     so F(low) = 0.05, F(med) = 0.5, F(high) = 0.95 exactly; between them the
     curve depends on that Gaussian assumption. Calibration multiplies s_lo /
     s_hi by the per-band k_lo / k_hi, which is exactly the calibrated band.
  2. PIT histogram -- F(true) in 20 bins (flat = calibrated). The first and last
     bins ([0, 0.05], [0.95, 1]) are the exact miss fractions; a U-shape means
     the interval is too narrow, a hump too wide, a tilt a median bias.
  3. 90% coverage vs lead time, with object-bootstrap 90% CIs.
  4. Miss rate by side (truth fainter / brighter than the band) vs lead time;
     each side should be ~0.05.
  5. (uniform truth) coverage vs truth magnitude minus the band's dense depth --
     where the survey-limited calibration can't see.

Points of one object are correlated, so every CI resamples OBJECTS, not points.
Magnitudes: larger = fainter, so "truth <= q_tau" means truth brighter than q.

    python plot_interval_coverage.py --study 141
    # or point at an eval dir directly
    python plot_interval_coverage.py \\
        --eval_dir runs/study_141/dense_latetime_eval_9band

Calibration scales are read from ``<eval_dir>/interval_calibration.json`` (the
eval writes it there by default); without it only the raw band is plotted.
"""

import argparse
import json
import os
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402

BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
BAND_COLORS = (
    "#2A9D8F", "#E63946", "#F4A261", "#457B9D", "#1B9E77",
    "#D62828", "#E9C46A", "#8338EC", "#264653",
)
# Quantile levels the head learns (study 131 on); the outer pair is the band.
Q_LO, Q_MED, Q_HI = 0.05, 0.5, 0.95
Z_HI = NormalDist().inv_cdf(Q_HI)
# Floor on a predicted half-width (mag), matching calibrate_intervals.py.
MIN_HALF_WIDTH = 1e-3
# Bins with fewer points than this are not plotted (too noisy to read).
MIN_BIN_POINTS = 50


def load_points(csv_path: str) -> pd.DataFrame:
    """Load a scored-points CSV (dense or uniform) with the columns used here."""
    cols = ["stem", "band", "lead_time_days", "pred_mag", "pred_low",
            "pred_high", "true_mag"]
    df = pd.read_csv(csv_path, usecols=cols)
    return df.rename(columns={"lead_time_days": "lead", "pred_mag": "med",
                              "pred_low": "low", "pred_high": "high",
                              "true_mag": "true"})


def load_scales(path: str) -> dict | None:
    """Per-band ``(k_lo, k_hi)`` from a calibration JSON, or None if absent."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        cal = json.load(fh)
    return {b: (v["k_lo"], v["k_hi"]) for b, v in cal["bands"].items()}


def add_variants(df: pd.DataFrame, scales: dict | None) -> list:
    """Add per-point split-Gaussian sigmas for raw and calibrated bands.

    Returns:
        list: Variant names present, ``["raw"]`` or ``["raw", "cal"]``.
    """
    df["s_lo_raw"] = np.maximum(df["med"] - df["low"], MIN_HALF_WIDTH) / Z_HI
    df["s_hi_raw"] = np.maximum(df["high"] - df["med"], MIN_HALF_WIDTH) / Z_HI
    if scales is None:
        return ["raw"]
    k_lo = df["band"].map(lambda b: scales.get(b, (1.0, 1.0))[0]).to_numpy()
    k_hi = df["band"].map(lambda b: scales.get(b, (1.0, 1.0))[1]).to_numpy()
    df["s_lo_cal"] = np.maximum(df["s_lo_raw"] * k_lo, MIN_HALF_WIDTH / Z_HI)
    df["s_hi_cal"] = np.maximum(df["s_hi_raw"] * k_hi, MIN_HALF_WIDTH / Z_HI)
    return ["raw", "cal"]


def pit(df: pd.DataFrame, variant: str) -> np.ndarray:
    """PIT value F(true) under the split Gaussian of ``variant``."""
    d = (df["true"] - df["med"]).to_numpy()
    s = np.where(d < 0, df[f"s_lo_{variant}"], df[f"s_hi_{variant}"])
    return ndtr(d / s)


class ObjectBootstrap:
    """Object-resampled CIs for per-point rates (points within an object correlate).

    One fixed set of multinomial object weights is shared by every call, so
    CIs across bins / bands / variants come from the same resamples.
    """

    def __init__(self, stems: np.ndarray, n_boot: int, seed: int) -> None:
        self.codes, uniq = pd.factorize(stems)
        self.n_obj = len(uniq)
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(self.n_obj, np.full(self.n_obj, 1 / self.n_obj),
                                 size=n_boot).astype(np.float64)

    def rate_ci(self, mask: np.ndarray, hit: np.ndarray) -> tuple:
        """``(rate, ci_lo, ci_hi)`` of ``hit`` over points in ``mask``."""
        codes = self.codes[mask]
        h = hit[mask].astype(np.float64)
        n_pts = np.bincount(codes, minlength=self.n_obj).astype(np.float64)
        n_hit = np.bincount(codes, weights=h, minlength=self.n_obj)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.w @ n_hit) / (self.w @ n_pts)
        boot = boot[np.isfinite(boot)]
        return (float(h.mean()), float(np.percentile(boot, 5)),
                float(np.percentile(boot, 95)))


def band_bounds(df: pd.DataFrame, variant: str) -> tuple:
    """The 0.05 / 0.95 bounds of ``variant`` (calibrated = rescaled half-widths)."""
    if variant == "raw":
        return df["low"].to_numpy(), df["high"].to_numpy()
    return ((df["med"] - Z_HI * df["s_lo_cal"]).to_numpy(),
            (df["med"] + Z_HI * df["s_hi_cal"]).to_numpy())


def _band_grid(title: str):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5), sharex=True, sharey=True)
    fig.suptitle(title)
    return fig, axes.ravel()


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_pp(df, variants, boot, truth, out_path) -> list:
    """PP plot per band; returns rows of the exact-quantile table."""
    taus = np.linspace(0.0, 1.0, 101)
    rows = []
    fig, axes = _band_grid(f"PP plot ({truth} truth): exact at 0.05 / 0.5 / 0.95; "
                           "curve = split-Gaussian interpolation")
    for b, name in enumerate(BAND_NAMES):
        ax = axes[b]
        m = (df["band"] == name).to_numpy()
        ax.plot([0, 1], [0, 1], color="k", lw=0.8)
        if not m.any():
            ax.set_title(f"{name} (no points)")
            continue
        sub = df[m]
        for v in variants:
            style = dict(color=BAND_COLORS[b]) if v == "cal" else dict(
                color="0.5", ls="--")
            p = np.sort(pit(sub, v))
            ax.plot(taus, np.searchsorted(p, taus, side="right") / p.size,
                    lw=1.4, label=v, **style)
            lo, hi = band_bounds(sub, v)
            row = {"truth": truth, "band": name, "variant": v, "n": int(m.sum())}
            for tau, q in ((Q_LO, lo), (Q_MED, sub["med"].to_numpy()), (Q_HI, hi)):
                hit = np.zeros(len(df), dtype=bool)
                hit[m] = sub["true"].to_numpy() <= q
                r, c_lo, c_hi = boot.rate_ci(m, hit)
                ax.errorbar(tau, r, yerr=[[r - c_lo], [c_hi - r]], fmt="o",
                            ms=4, capsize=2, **style)
                row[f"F({tau})"] = r
            rows.append(row)
        ax.set_title(f"{name} (n={int(m.sum())})")
        ax.set_aspect("equal")
    for ax in axes[6:]:
        ax.set_xlabel("nominal quantile level")
    for ax in axes[::3]:
        ax.set_ylabel("fraction truth <= predicted quantile")
    axes[0].legend(loc="upper left", fontsize=8)
    _save(fig, out_path)
    return rows


def plot_pit(df, variants, truth, out_path) -> None:
    """PIT histogram per band (density; flat line at 1 = calibrated)."""
    edges = np.linspace(0.0, 1.0, 21)
    fig, axes = _band_grid(f"PIT histogram ({truth} truth): outer bins exact, "
                           "inner bins split-Gaussian")
    for b, name in enumerate(BAND_NAMES):
        ax = axes[b]
        sub = df[df["band"] == name]
        ax.axhline(1.0, color="k", lw=0.8)
        if sub.empty:
            continue
        for v in variants:
            style = dict(color=BAND_COLORS[b], lw=1.6) if v == "cal" else dict(
                color="0.5", lw=1.0, ls="--")
            h, _ = np.histogram(pit(sub, v), bins=edges, density=True)
            ax.stairs(h, edges, label=v, **style)
        ax.set_title(name)
    for ax in axes[6:]:
        ax.set_xlabel("PIT = F(true)")
    axes[0].legend(loc="upper center", fontsize=8)
    _save(fig, out_path)


def _binned(df, boot, key, edges, variants, fn):
    """Per band / variant / bin: ``fn(sub_mask, variant)`` -> dict of hit arrays."""
    out = {}
    x = df[key].to_numpy()
    for name in BAND_NAMES:
        bm = (df["band"] == name).to_numpy()
        for v in variants:
            hits = fn(v)
            for series, hit in hits.items():
                pts = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = bm & (x >= lo) & (x < hi)
                    if m.sum() < MIN_BIN_POINTS:
                        continue
                    pts.append((0.5 * (lo + hi), *boot.rate_ci(m, hit),
                                int(m.sum())))
                out[(name, v, series)] = np.array(pts).reshape(-1, 5)
    return out


def _hit_arrays(df, variants):
    """Per-variant covered / missed-fainter / missed-brighter boolean arrays."""
    t = df["true"].to_numpy()
    cache = {}
    for v in variants:
        lo, hi = band_bounds(df, v)
        cache[v] = {"covered": (t >= lo) & (t <= hi), "fainter": t > hi,
                    "brighter": t < lo}
    return cache


def _plot_series(ax, pts, color, ls, label, lw=1.5):
    if pts.size == 0:
        return
    ax.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=color, alpha=0.15,
                    lw=0)


def plot_coverage_vs(df, variants, boot, truth, key, edges, xlabel, out_path,
                     title_extra="", hits=None):
    """90% coverage vs ``key`` per band, raw (dashed) and calibrated (solid)."""
    hits = hits or _hit_arrays(df, variants)
    res = _binned(df, boot, key, edges, variants,
                  lambda v: {"covered": hits[v]["covered"]})
    fig, axes = _band_grid(f"90% interval coverage vs {xlabel} ({truth} truth)"
                           f"{title_extra}; shading = object-bootstrap 90% CI")
    for b, name in enumerate(BAND_NAMES):
        ax = axes[b]
        ax.axhline(0.90, color="k", lw=0.8)
        for v in variants:
            color = BAND_COLORS[b] if v == "cal" else "0.5"
            _plot_series(ax, res[(name, v, "covered")], color,
                         "-" if v == "cal" else "--", v)
        ax.set_title(name)
        ax.set_ylim(0.4, 1.02)
    for ax in axes[6:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::3]:
        ax.set_ylabel("coverage")
    axes[0].legend(loc="lower left", fontsize=8)
    _save(fig, out_path)
    return res


def plot_misses_vs_lead(df, variants, boot, truth, out_path, hits) -> None:
    """Miss rate by side vs lead; each side's target is 0.05."""
    edges = np.arange(0.0, 11.0, 1.0)
    res = _binned(df, boot, "lead", edges, variants,
                  lambda v: {k: hits[v][k] for k in ("fainter", "brighter")})
    fig, axes = _band_grid(f"Miss rate by side vs lead ({truth} truth): "
                           "fainter = truth below the band (target 0.05 each)")
    side_color = {"fainter": "#1f4e9c", "brighter": "#c0392b"}
    for b, name in enumerate(BAND_NAMES):
        ax = axes[b]
        ax.axhline(0.05, color="k", lw=0.8)
        for v in variants:
            for side in ("fainter", "brighter"):
                _plot_series(ax, res[(name, v, side)], side_color[side],
                             "-" if v == "cal" else "--", f"{side} ({v})",
                             lw=1.5 if v == "cal" else 1.0)
        ax.set_title(name)
        ax.set_ylim(0.0, 0.3)
    for ax in axes[6:]:
        ax.set_xlabel("lead time (d)")
    for ax in axes[::3]:
        ax.set_ylabel("miss rate")
    axes[0].legend(loc="upper left", fontsize=7)
    _save(fig, out_path)


def dense_depths(dense: pd.DataFrame) -> dict:
    """Per-band dense depth = 99th percentile of scored dense truth (as the eval)."""
    return {name: float(np.percentile(g["true"], 99))
            for name, g in dense.groupby("band")}


def print_tables(pp_rows, lead_res, truth, variants) -> None:
    """Exact-quantile PP table and a compact coverage-vs-lead table."""
    print(f"\n[{truth}] Exact PP points: fraction truth <= predicted quantile "
          "(targets 0.05 / 0.50 / 0.95)")
    print(f"  {'band':<6}{'var':<5}{'n':>9}{'F(.05)':>8}{'F(.5)':>8}{'F(.95)':>8}"
          f"{'cov':>7}")
    for r in pp_rows:
        print(f"  {r['band']:<6}{r['variant']:<5}{r['n']:>9d}"
              f"{r['F(0.05)']:>8.3f}{r['F(0.5)']:>8.3f}{r['F(0.95)']:>8.3f}"
              f"{r['F(0.95)'] - r['F(0.05)']:>7.3f}")
    v = variants[-1]
    print(f"\n[{truth}] {v} 90% coverage by lead bin (d), bins with "
          f">= {MIN_BIN_POINTS} points")
    centers = sorted({c for (n, vv, s), pts in lead_res.items() if vv == v
                      for c in pts[:, 0]})
    print("  " + f"{'band':<6}" + "".join(f"{c:>6.1f}" for c in centers))
    for name in BAND_NAMES:
        pts = lead_res[(name, v, "covered")]
        by_c = dict(zip(pts[:, 0], pts[:, 1]))
        print("  " + f"{name:<6}" + "".join(
            f"{by_c[c]:>6.2f}" if c in by_c else f"{'-':>6}" for c in centers))


def run_truth(df, truth, variants, boot, outdir, depths=None) -> None:
    """Every plot + table for one truth set."""
    pp_rows = plot_pp(df, variants, boot, truth,
                      os.path.join(outdir, f"interval_pp_{truth}.png"))
    plot_pit(df, variants, truth, os.path.join(outdir, f"interval_pit_{truth}.png"))
    hits = _hit_arrays(df, variants)
    lead_res = plot_coverage_vs(
        df, variants, boot, truth, "lead", np.arange(0.0, 11.0, 1.0),
        "lead time (d)", os.path.join(outdir, f"interval_cov_lead_{truth}.png"),
        hits=hits)
    plot_misses_vs_lead(df, variants, boot, truth,
                        os.path.join(outdir, f"interval_miss_lead_{truth}.png"),
                        hits)
    if depths is not None:
        df["dmag_depth"] = df["true"] - df["band"].map(depths)
        plot_coverage_vs(
            df, variants, boot, truth, "dmag_depth", np.arange(-6.0, 9.0, 1.0),
            "truth mag - dense depth", os.path.join(
                outdir, f"interval_cov_depth_{truth}.png"),
            title_extra=" (> 0 = beyond dense depth)", hits=hits)
    print_tables(pp_rows, lead_res, truth, variants)


def main() -> None:
    """Load the eval CSVs and write the interval diagnostics."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=int, default=None,
                   help="Study number -> runs/study_NNN/dense_latetime_eval_9band.")
    p.add_argument("--eval_dir", default=None,
                   help="Eval output dir (overrides --study).")
    p.add_argument("--calibration", default=None,
                   help="Calibration JSON (default <eval_dir>/interval_"
                   "calibration.json; raw band only if missing).")
    p.add_argument("--truth", choices=("dense", "uniform", "both"), default="both")
    p.add_argument("--outdir", default=None,
                   help="Plot dir (default <eval_dir>/interval_diagnostics).")
    p.add_argument("--n_boot", type=int, default=200,
                   help="Object-bootstrap resamples for the CIs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.eval_dir is None:
        if args.study is None:
            p.error("give --study or --eval_dir")
        args.eval_dir = f"runs/study_{args.study:03d}/dense_latetime_eval_9band"
    outdir = args.outdir or os.path.join(args.eval_dir, "interval_diagnostics")
    os.makedirs(outdir, exist_ok=True)

    cal_path = args.calibration or os.path.join(args.eval_dir,
                                                "interval_calibration.json")
    scales = load_scales(cal_path)
    print(f"Calibration: {cal_path if scales else 'none found -> raw band only'}")

    dense_csv = os.path.join(args.eval_dir, "latetime_scored_points.csv")
    uni_csv = os.path.join(args.eval_dir, "latetime_uniform_scored_points.csv")
    dense = load_points(dense_csv)
    variants = add_variants(dense, scales)
    print(f"Dense: {len(dense)} points, {dense['stem'].nunique()} objects")

    if args.truth in ("dense", "both"):
        boot = ObjectBootstrap(dense["stem"].to_numpy(), args.n_boot, args.seed)
        run_truth(dense, "dense", variants, boot, outdir)
    if args.truth in ("uniform", "both"):
        if not os.path.exists(uni_csv):
            print(f"No {uni_csv}; skipping uniform truth.")
            return
        uni = load_points(uni_csv)
        add_variants(uni, scales)
        print(f"Uniform: {len(uni)} points, {uni['stem'].nunique()} objects")
        boot = ObjectBootstrap(uni["stem"].to_numpy(), args.n_boot, args.seed)
        run_truth(uni, "uniform", variants, boot, outdir,
                  depths=dense_depths(dense))


if __name__ == "__main__":
    main()
