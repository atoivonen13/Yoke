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
  6. Interval score vs lead -- the proper score for a central 90% interval,

         IS = (high - low) + (2 / alpha) * (distance of truth outside the band),

     alpha = 0.1, lower = better. Coverage alone rewards an over-wide band; IS
     trades width (dotted) against miss distance, so it ranks studies' intervals.
     The table adds the mean pinball loss over the three quantiles (the training
     loss, unweighted, in mag).
  7. Spread-skill -- points binned by predicted half-width (per-band deciles):
     coverage per decile, and RMS residual vs the implied sigma (half-width /
     z95; on the diagonal = right size). Flat coverage = the per-point widths
     carry information; narrow deciles under-covering and wide over-covering =
     the band is right only on average (per-band scales can't fix that).
  8. Per-object coverage -- histogram over objects of the fraction of that
     object's points inside the band, vs the binomial reference if points missed
     independently at the band's mean coverage. Excess mass near 0 (overdispersion
     > 1 in the table) = misses cluster in whole objects (level offsets), not
     point scatter.

Points of one object are correlated, so every CI resamples OBJECTS, not points.
CIs are test-sample noise only; seed noise (~0.05 RMSE) is not in them.
Magnitudes: larger = fainter, so "truth <= q_tau" means truth brighter than q.

    python plot_interval_coverage.py --study 141
    # -> runs/study_141/.../interval_diagnostics/study141_interval_pp_uniform.png, ...
    # or point at an eval dir directly
    python plot_interval_coverage.py \\
        --eval_dir runs/study_141/dense_latetime_eval_9band
    # overlay up to 6 studies (calibrated band; coverage / IS vs lead,
    # spread-skill, coverage vs depth) + side-by-side tables
    python plot_interval_coverage.py --studies 138 140 141
    # -> runs/interval_compare/study138_140_141_compare_*.png

Calibration scales are read from ``<eval_dir>/interval_calibration.json`` (the
eval writes it there by default); without it only the raw band is plotted.
"""

import argparse
import json
import os
import re
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.stats import binom  # noqa: E402

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
# Miss rate of the central interval (0.05 below + 0.05 above).
ALPHA = Q_LO + (1.0 - Q_HI)
LEAD_EDGES = np.arange(0.0, 11.0, 1.0)
DEPTH_EDGES = np.arange(-6.0, 9.0, 1.0)
OBJ_COV_EDGES = np.linspace(0.0, 1.0, 11)
# Objects with fewer points than this in a band are left out of per-object coverage.
MIN_OBJ_POINTS = 5
# One color per study in --studies mode, in this fixed order.
STUDY_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


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

    def __init__(self, stems: np.ndarray, n_boot: int, seed: int,
                 universe: np.ndarray | None = None) -> None:
        # With a shared ``universe`` (all stems across studies) and seed, every
        # study gets the same object weights -> paired study differences.
        if universe is None:
            self.codes, uniq = pd.factorize(stems)
            self.n_obj = len(uniq)
        else:
            self.codes = pd.Index(universe).get_indexer(stems)
            self.n_obj = len(universe)
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(self.n_obj, np.full(self.n_obj, 1 / self.n_obj),
                                 size=n_boot).astype(np.float64)

    def boot_means(self, mask: np.ndarray, value: np.ndarray) -> tuple:
        """``(mean, per-resample means)`` of ``value`` over points in ``mask``."""
        codes = self.codes[mask]
        h = value[mask].astype(np.float64)
        n_pts = np.bincount(codes, minlength=self.n_obj).astype(np.float64)
        n_hit = np.bincount(codes, weights=h, minlength=self.n_obj)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.w @ n_hit) / (self.w @ n_pts)
        return float(h.mean()), boot

    def mean_ci(self, mask: np.ndarray, value: np.ndarray) -> tuple:
        """``(mean, ci_lo, ci_hi)`` of ``value`` over points in ``mask``.

        ``value`` may be boolean (a rate) or any per-point float (e.g. a score).
        """
        mean, boot = self.boot_means(mask, value)
        boot = boot[np.isfinite(boot)]
        return (mean, float(np.percentile(boot, 5)),
                float(np.percentile(boot, 95)))


def band_bounds(df: pd.DataFrame, variant: str) -> tuple:
    """The 0.05 / 0.95 bounds of ``variant`` (calibrated = rescaled half-widths)."""
    if variant == "raw":
        return df["low"].to_numpy(), df["high"].to_numpy()
    return ((df["med"] - Z_HI * df["s_lo_cal"]).to_numpy(),
            (df["med"] + Z_HI * df["s_hi_cal"]).to_numpy())


def _band_grid(title: str, sharey: bool = True, sharex: bool = True):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5), sharex=sharex, sharey=sharey)
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
                r, c_lo, c_hi = boot.mean_ci(m, hit)
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
    """Per band / variant / bin: ``fn(variant)`` -> dict of per-point arrays."""
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
                    pts.append((0.5 * (lo + hi), *boot.mean_ci(m, hit),
                                int(m.sum())))
                out[(name, v, series)] = np.array(pts).reshape(-1, 5)
    return out


def point_scores(df, variants) -> dict:
    """Per-variant per-point arrays (mag where not boolean).

    covered / fainter / brighter: inside the band, or missed on that side.
    is: interval score; width: band width; half_width: half of it;
    pinball: mean pinball loss over the 0.05 / 0.5 / 0.95 quantiles;
    sq_resid: squared median residual.
    """
    t = df["true"].to_numpy()
    med = df["med"].to_numpy()
    out = {}
    for v in variants:
        lo, hi = band_bounds(df, v)
        pin = np.zeros_like(t)
        for tau, q in ((Q_LO, lo), (Q_MED, med), (Q_HI, hi)):
            d = t - q
            pin += np.maximum(tau * d, (tau - 1.0) * d)
        out[v] = {"covered": (t >= lo) & (t <= hi), "fainter": t > hi,
                  "brighter": t < lo, "width": hi - lo,
                  "half_width": 0.5 * (hi - lo),
                  "is": (hi - lo) + (2.0 / ALPHA) * (np.maximum(lo - t, 0.0)
                                                     + np.maximum(t - hi, 0.0)),
                  "pinball": pin / 3.0, "sq_resid": (t - med) ** 2}
    return out


def _plot_series(ax, pts, color, ls, label, lw=1.5, marker=None):
    if pts.size == 0:
        return
    ax.plot(pts[:, 0], pts[:, 1], color=color, ls=ls, lw=lw, label=label,
            marker=marker, ms=3)
    ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=color, alpha=0.15,
                    lw=0)


def plot_coverage_vs(df, variants, boot, truth, key, edges, xlabel, out_path,
                     title_extra="", hits=None):
    """90% coverage vs ``key`` per band, raw (dashed) and calibrated (solid)."""
    hits = hits or point_scores(df, variants)
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
    res = _binned(df, boot, "lead", LEAD_EDGES, variants,
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


def plot_score_vs_lead(df, variants, boot, truth, out_path, sc) -> None:
    """Interval score (solid / dashed) and band width (dotted) vs lead."""
    res = _binned(df, boot, "lead", LEAD_EDGES, variants,
                  lambda v: {k: sc[v][k] for k in ("is", "width")})
    fig, axes = _band_grid(f"90% interval score vs lead ({truth} truth; lower = "
                           "better): IS with object-bootstrap 90% CI, dotted = "
                           "band width", sharey=False)
    for b, name in enumerate(BAND_NAMES):
        ax = axes[b]
        for v in variants:
            color = BAND_COLORS[b] if v == "cal" else "0.5"
            _plot_series(ax, res[(name, v, "is")], color,
                         "-" if v == "cal" else "--", f"IS ({v})")
            w = res[(name, v, "width")]
            if w.size:
                ax.plot(w[:, 0], w[:, 1], color=color, ls=":", lw=1.2,
                        label=f"width ({v})")
        ax.set_title(name)
        ax.set_ylim(bottom=0.0)
    for ax in axes[6:]:
        ax.set_xlabel("lead time (d)")
    for ax in axes[::3]:
        ax.set_ylabel("mag")
    axes[0].legend(loc="upper left", fontsize=7)
    _save(fig, out_path)


def spread_skill(df, boot, sc_v, n_bins: int = 10) -> dict:
    """Per band, points binned by predicted half-width (per-band quantiles).

    Returns:
        dict: band -> array of rows ``(mean half-width, coverage, cov_ci_lo,
        cov_ci_hi, RMS residual, rms_ci_lo, rms_ci_hi)``.
    """
    hw = sc_v["half_width"]
    out = {}
    for name in BAND_NAMES:
        bm = (df["band"] == name).to_numpy()
        rows = []
        if bm.any():
            q = np.quantile(hw[bm], np.linspace(0.0, 1.0, n_bins + 1))
            idx = np.clip(np.searchsorted(q, hw, side="right") - 1, 0, n_bins - 1)
            for k in range(n_bins):
                m = bm & (idx == k)
                if m.sum() < MIN_BIN_POINTS:
                    continue
                cov = boot.mean_ci(m, sc_v["covered"])
                msq, bt = boot.boot_means(m, sc_v["sq_resid"])
                bt = np.sqrt(bt[np.isfinite(bt)])
                rows.append((hw[m].mean(), *cov, np.sqrt(msq),
                             np.percentile(bt, 5), np.percentile(bt, 95)))
        out[name] = np.array(rows).reshape(-1, 7)
    return out


def plot_spread_skill(df, variants, boot, truth, sc, cov_path, rms_path) -> dict:
    """Coverage and RMS residual per predicted-half-width decile."""
    ss = {v: spread_skill(df, boot, sc[v]) for v in variants}
    fig_c, ax_c = _band_grid(f"Spread-skill ({truth} truth): coverage per decile "
                             "of predicted half-width (flat at 0.9 = widths carry "
                             "information)", sharex=False)
    fig_r, ax_r = _band_grid(f"Spread-skill ({truth} truth): RMS median residual "
                             "per half-width decile; black = half-width / z95 "
                             "(right size if Gaussian)", sharex=False, sharey=False)
    for b, name in enumerate(BAND_NAMES):
        ax_c[b].axhline(0.90, color="k", lw=0.8)
        for v in variants:
            pts = ss[v][name]
            color = BAND_COLORS[b] if v == "cal" else "0.5"
            ls = "-" if v == "cal" else "--"
            _plot_series(ax_c[b], pts[:, :4], color, ls, v, marker="o")
            _plot_series(ax_r[b], pts[:, [0, 4, 5, 6]], color, ls, v, marker="o")
        xs = np.concatenate([ss[v][name][:, 0] for v in variants])
        if xs.size:
            x = np.array([xs.min() * 0.95, xs.max() * 1.05])
            ax_r[b].plot(x, x / Z_HI, color="k", lw=0.8, label="half-width / z95")
        ax_c[b].set_ylim(0.4, 1.02)
        for axes in (ax_c, ax_r):
            axes[b].set_title(name)
    for axes, ylabel in ((ax_c, "coverage"), (ax_r, "RMS residual (mag)")):
        for ax in axes[6:]:
            ax.set_xlabel("predicted 90% half-width (mag)")
        for ax in axes[::3]:
            ax.set_ylabel(ylabel)
    ax_c[0].legend(loc="lower right", fontsize=8)
    ax_r[0].legend(loc="upper left", fontsize=8)
    _save(fig_c, cov_path)
    _save(fig_r, rms_path)
    return ss


def object_fracs(codes, n_obj, mask, covered) -> tuple:
    """Per-object covered fraction and point count over points in ``mask``.

    Objects with fewer than ``MIN_OBJ_POINTS`` points in ``mask`` are dropped.
    """
    n = np.bincount(codes[mask], minlength=n_obj)
    c = np.bincount(codes[mask], weights=covered[mask].astype(np.float64),
                    minlength=n_obj)
    keep = n >= MIN_OBJ_POINTS
    return c[keep] / n[keep], n[keep]


def object_stats(frac, n) -> dict:
    """Pooled coverage, overdispersion vs binomial, and P(object coverage < 0.5).

    Overdispersion = observed variance of the per-object fraction / the variance
    if every point missed independently at the pooled rate (1 = independent).
    """
    if n.size == 0:
        return {"n_obj": 0, "p": np.nan, "phi": np.nan, "lt50": np.nan,
                "lt50_binom": np.nan}
    p = float((frac * n).sum() / n.sum())
    expect_var = np.mean(p * (1.0 - p) / n)
    # k / n < 0.5  <=>  k <= ceil(n / 2) - 1
    lt50_binom = binom.cdf(np.ceil(n / 2.0) - 1, n, p).mean()
    return {"n_obj": int(n.size), "p": p,
            "phi": float(np.mean((frac - p) ** 2) / expect_var)
            if expect_var > 0 else np.nan,
            "lt50": float(np.mean(frac < 0.5)), "lt50_binom": float(lt50_binom)}


def binomial_reference(n, p, edges) -> np.ndarray:
    """Expected per-object coverage histogram if points miss independently.

    Fraction of objects per bin at miss rate 1 - p, for the observed point
    counts ``n``.
    """
    hist = np.zeros(len(edges) - 1)
    for n_k, count in zip(*np.unique(n, return_counts=True)):
        k = np.arange(n_k + 1)
        h, _ = np.histogram(k / n_k, bins=edges, weights=count * binom.pmf(k, n_k, p))
        hist += h
    return hist / max(len(n), 1)


def plot_object_coverage(df, variants, truth, sc, out_path) -> list:
    """Histogram over objects of per-object coverage (final variant) vs binomial.

    Returns:
        list: Table rows (every band and variant, plus ``all`` bands pooled).
    """
    codes, uniq = pd.factorize(df["stem"])
    n_obj = len(uniq)
    v_plot = variants[-1]
    rows = []
    fig, axes = _band_grid(f"Per-object 90% coverage ({truth} truth, {v_plot}; "
                           f"objects with >= {MIN_OBJ_POINTS} points in the band): "
                           "black = binomial (independent misses)")
    for b, name in enumerate(BAND_NAMES + ("all",)):
        bm = (np.ones(len(df), dtype=bool) if name == "all"
              else (df["band"] == name).to_numpy())
        for v in variants:
            frac, n = object_fracs(codes, n_obj, bm, sc[v]["covered"])
            st = object_stats(frac, n)
            rows.append({"band": name, "variant": v, **st})
            if name == "all" or v != v_plot or n.size == 0:
                continue
            ax = axes[b]
            h, _ = np.histogram(frac, bins=OBJ_COV_EDGES)
            ax.stairs(h / n.size, OBJ_COV_EDGES, fill=True, color=BAND_COLORS[b],
                      alpha=0.6, label="observed")
            ax.stairs(binomial_reference(n, st["p"], OBJ_COV_EDGES), OBJ_COV_EDGES,
                      color="k", lw=1.2, label="binomial")
            ax.set_title(f"{name} (n_obj={n.size}, overdisp={st['phi']:.2f})")
    for ax in axes[6:]:
        ax.set_xlabel("fraction of the object's points inside the band")
    for ax in axes[::3]:
        ax.set_ylabel("fraction of objects")
    axes[0].legend(loc="upper left", fontsize=8)
    _save(fig, out_path)
    return rows


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


def _band_masks(df):
    """``(name, mask)`` for every band present, then ``all``."""
    for name in BAND_NAMES:
        m = (df["band"] == name).to_numpy()
        if m.any():
            yield name, m
    yield "all", np.ones(len(df), dtype=bool)


def print_scores(df, variants, boot, sc, truth) -> None:
    """Coverage, width, interval score (with CI) and pinball per band."""
    print(f"\n[{truth}] Interval scores (mag; lower = better; IS CI = "
          "object-bootstrap 90%)")
    print(f"  {'band':<6}{'var':<5}{'cov':>7}{'width':>8}{'IS':>8}"
          f"{'IS CI':>16}{'pinball':>9}")
    for name, m in _band_masks(df):
        for v in variants:
            r, lo, hi = boot.mean_ci(m, sc[v]["is"])
            print(f"  {name:<6}{v:<5}{sc[v]['covered'][m].mean():>7.3f}"
                  f"{sc[v]['width'][m].mean():>8.3f}{r:>8.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>16}{sc[v]['pinball'][m].mean():>9.4f}")


def print_spread(ss, truth, variants) -> None:
    """Narrowest vs widest half-width decile: coverage and RMS / implied sigma."""
    print(f"\n[{truth}] Spread-skill, narrowest vs widest half-width decile "
          "(coverage; RMS residual / (half-width / z95), 1 = right size)")
    print(f"  {'band':<6}{'var':<5}{'cov lo':>8}{'cov hi':>8}{'ratio lo':>10}"
          f"{'ratio hi':>10}")
    for name in BAND_NAMES:
        for v in variants:
            pts = ss[v][name]
            if pts.shape[0] < 2:
                continue
            ratio = pts[:, 4] / (pts[:, 0] / Z_HI)
            print(f"  {name:<6}{v:<5}{pts[0, 1]:>8.3f}{pts[-1, 1]:>8.3f}"
                  f"{ratio[0]:>10.2f}{ratio[-1]:>10.2f}")


def print_objects(rows, truth) -> None:
    """Per-object coverage: overdispersion and P(object coverage < 0.5)."""
    print(f"\n[{truth}] Per-object coverage (objects with >= {MIN_OBJ_POINTS} "
          "points; overdisp 1 = independent misses, > 1 = misses cluster in "
          "objects)")
    print(f"  {'band':<6}{'var':<5}{'n_obj':>7}{'cov':>7}{'overdisp':>10}"
          f"{'P(<0.5)':>9}{'binom':>8}")
    for r in rows:
        if r["n_obj"] == 0:
            continue
        print(f"  {r['band']:<6}{r['variant']:<5}{r['n_obj']:>7d}{r['p']:>7.3f}"
              f"{r['phi']:>10.2f}{r['lt50']:>9.3f}{r['lt50_binom']:>8.3f}")


def run_truth(df, truth, variants, boot, outdir, prefix, depths=None) -> None:
    """Every plot + table for one truth set; filenames start with ``prefix``."""
    def path(kind):
        return os.path.join(outdir, f"{prefix}interval_{kind}_{truth}.png")

    pp_rows = plot_pp(df, variants, boot, truth, path("pp"))
    plot_pit(df, variants, truth, path("pit"))
    sc = point_scores(df, variants)
    lead_res = plot_coverage_vs(df, variants, boot, truth, "lead", LEAD_EDGES,
                                "lead time (d)", path("cov_lead"), hits=sc)
    plot_misses_vs_lead(df, variants, boot, truth, path("miss_lead"), sc)
    plot_score_vs_lead(df, variants, boot, truth, path("score_lead"), sc)
    ss = plot_spread_skill(df, variants, boot, truth, sc, path("spread_cov"),
                           path("spread_rms"))
    obj_rows = plot_object_coverage(df, variants, truth, sc, path("obj_cov"))
    if depths is not None:
        df["dmag_depth"] = df["true"] - df["band"].map(depths)
        plot_coverage_vs(df, variants, boot, truth, "dmag_depth", DEPTH_EDGES,
                         "truth mag - dense depth", path("cov_depth"),
                         title_extra=" (> 0 = beyond dense depth)", hits=sc)
    print_tables(pp_rows, lead_res, truth, variants)
    print_scores(df, variants, boot, sc, truth)
    print_spread(ss, truth, variants)
    print_objects(obj_rows, truth)


# ---------------------------------------------------------------- --studies


def _compare_grid(entries, title, path, ylabel, xlabel, series_fn, ref=None,
                  ylim=None, sharey=True, sharex=True, marker=None) -> None:
    """One line per study per band; ``series_fn(entry)`` -> {band: pts}."""
    fig, axes = _band_grid(title, sharey=sharey, sharex=sharex)
    for e in entries:
        pts_by_band = series_fn(e)
        for b, name in enumerate(BAND_NAMES):
            _plot_series(axes[b], pts_by_band[name], e["color"], "-", e["label"],
                         marker=marker)
    for b, name in enumerate(BAND_NAMES):
        if ref is not None:
            axes[b].axhline(ref, color="k", lw=0.8)
        if ylim is not None:
            axes[b].set_ylim(*ylim)
        axes[b].set_title(name)
    for ax in axes[6:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::3]:
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=8)
    _save(fig, path)


def _binned_one(e, key, edges, series):
    res = _binned(e["df"], e["boot"], key, edges, [e["v"]],
                  lambda v: {series: e["sc"][series]})
    return {name: res[(name, e["v"], series)] for name in BAND_NAMES}


def _print_delta_table(entries, truth, key, label) -> None:
    """Per band: ``key`` per study, and the paired-bootstrap delta vs the first."""
    base = entries[0]
    print(f"\n[{truth}] {label} (mag; lower = better); delta = study - "
          f"{base['study']} with paired object-bootstrap 90% CI")
    head = f"  {'band':<6}{base['study']:>8}"
    for e in entries[1:]:
        head += f"{e['study']:>8}{'delta':>9}{'90% CI':>18}"
    print(head)
    base_masks = dict(_band_masks(base["df"]))
    for name, m0 in base_masks.items():
        r0, b0 = base["boot"].boot_means(m0, base["sc"][key])
        line = f"  {name:<6}{r0:>8.3f}"
        for e in entries[1:]:
            m = dict(_band_masks(e["df"])).get(name)
            if m is None:
                line += f"{'-':>8}{'':>27}"
                continue
            r, bt = e["boot"].boot_means(m, e["sc"][key])
            d = bt - b0
            d = d[np.isfinite(d)]
            ci = f"[{np.percentile(d, 5):+.3f}, {np.percentile(d, 95):+.3f}]"
            line += f"{r:>8.3f}{r - r0:>+9.3f}{ci:>18}"
        print(line)


def print_compare(entries, truth) -> None:
    """Coverage / width / overdispersion side by side, then IS and pinball deltas."""
    print(f"\n[{truth}] Studies " + " / ".join(e["label"] for e in entries)
          + ": coverage, mean width (mag), per-object overdispersion")
    print(f"  {'band':<6}" + "".join(f"{e['study']:>21}" for e in entries))
    print(f"  {'':<6}" + "".join(f"{'cov':>7}{'width':>7}{'odisp':>7}"
                                 for e in entries))
    stats = []
    for e in entries:
        codes, uniq = pd.factorize(e["df"]["stem"])
        per = {}
        for name, m in _band_masks(e["df"]):
            st = object_stats(*object_fracs(codes, len(uniq), m, e["sc"]["covered"]))
            per[name] = (e["sc"]["covered"][m].mean(), e["sc"]["width"][m].mean(),
                         st["phi"])
        stats.append(per)
    for name in BAND_NAMES + ("all",):
        line = f"  {name:<6}"
        for per in stats:
            line += ("".join(f"{x:>7.3f}" if i < 2 else f"{x:>7.2f}"
                             for i, x in enumerate(per[name]))
                     if name in per else f"{'-':>21}")
        print(line)
    _print_delta_table(entries, truth, "is", "Interval score")
    _print_delta_table(entries, truth, "pinball", "Mean pinball loss")


def run_compare(studies, data, truth, outdir, prefix, n_boot, seed,
                depths=None) -> None:
    """Overlay the studies' final-variant band for one truth set."""
    items = [(s, d[truth], d["variant"]) for s, d in zip(studies, data)
             if d[truth] is not None]
    if len(items) < 2:
        print(f"[{truth}] fewer than 2 studies have this truth set; skipping.")
        return
    # One object universe + seed -> identical bootstrap weights per study, so
    # study differences are paired (the test objects are shared).
    universe = np.unique(np.concatenate([df["stem"].to_numpy() for _, df, _ in items]))
    entries = []
    for i, (s, df, v) in enumerate(items):
        entries.append({
            "study": f"{s:03d}", "label": f"{s:03d} ({v})", "df": df, "v": v,
            "color": STUDY_COLORS[i], "sc": point_scores(df, [v])[v],
            "boot": ObjectBootstrap(df["stem"].to_numpy(), n_boot, seed, universe)})

    def path(kind):
        return os.path.join(outdir, f"{prefix}{kind}_{truth}.png")

    ci = "shading = object-bootstrap 90% CI"
    _compare_grid(entries, f"90% coverage vs lead ({truth} truth); {ci}",
                  path("cov_lead"), "coverage", "lead time (d)",
                  lambda e: _binned_one(e, "lead", LEAD_EDGES, "covered"),
                  ref=0.90, ylim=(0.4, 1.02))
    _compare_grid(entries, f"90% interval score vs lead ({truth} truth; lower = "
                  f"better); {ci}", path("score_lead"), "interval score (mag)",
                  "lead time (d)",
                  lambda e: _binned_one(e, "lead", LEAD_EDGES, "is"), sharey=False)
    _compare_grid(entries, f"Spread-skill ({truth} truth): coverage per decile of "
                  f"predicted half-width; {ci}", path("spread_cov"), "coverage",
                  "predicted 90% half-width (mag)",
                  lambda e: {n: p[:, :4] for n, p in
                             spread_skill(e["df"], e["boot"], e["sc"]).items()},
                  ref=0.90, ylim=(0.4, 1.02), sharex=False, marker="o")
    if depths is not None:
        for e in entries:
            e["df"]["dmag_depth"] = e["df"]["true"] - e["df"]["band"].map(depths)
        _compare_grid(entries, f"90% coverage vs truth mag - dense depth ({truth} "
                      f"truth; > 0 = beyond dense depth); {ci}", path("cov_depth"),
                      "coverage", "truth mag - dense depth",
                      lambda e: _binned_one(e, "dmag_depth", DEPTH_EDGES, "covered"),
                      ref=0.90, ylim=(0.4, 1.02))
    print_compare(entries, truth)


def load_study(eval_dir, cal_path=None) -> dict:
    """Dense + (if present) uniform scored points of one eval dir, with variants."""
    cal_path = cal_path or os.path.join(eval_dir, "interval_calibration.json")
    scales = load_scales(cal_path)
    print(f"{eval_dir}: calibration "
          f"{cal_path if scales else 'none found -> raw band only'}")
    dense = load_points(os.path.join(eval_dir, "latetime_scored_points.csv"))
    variants = add_variants(dense, scales)
    print(f"  dense: {len(dense)} points, {dense['stem'].nunique()} objects")
    uni_csv = os.path.join(eval_dir, "latetime_uniform_scored_points.csv")
    uni = None
    if os.path.exists(uni_csv):
        uni = load_points(uni_csv)
        add_variants(uni, scales)
        print(f"  uniform: {len(uni)} points, {uni['stem'].nunique()} objects")
    else:
        print(f"  no {uni_csv}")
    return {"dense": dense, "uniform": uni, "variants": variants,
            "variant": variants[-1]}


def main() -> None:
    """Load the eval CSVs and write the interval diagnostics."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=int, default=None,
                   help="Study number -> runs/study_NNN/dense_latetime_eval_9band.")
    p.add_argument("--eval_dir", default=None,
                   help="Eval output dir (overrides --study).")
    p.add_argument("--studies", type=int, nargs="+", default=None,
                   help=f"Overlay 2-{len(STUDY_COLORS)} studies (default eval dirs, "
                   "each with its own calibration JSON); the first is the delta "
                   "baseline. Ignores --study / --eval_dir / --calibration.")
    p.add_argument("--calibration", default=None,
                   help="Calibration JSON (default <eval_dir>/interval_"
                   "calibration.json; raw band only if missing).")
    p.add_argument("--truth", choices=("dense", "uniform", "both"), default="both")
    p.add_argument("--outdir", default=None,
                   help="Plot dir (default <eval_dir>/interval_diagnostics; "
                   "runs/interval_compare with --studies).")
    p.add_argument("--n_boot", type=int, default=200,
                   help="Object-bootstrap resamples for the CIs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    truths = [t for t in ("dense", "uniform") if args.truth in (t, "both")]

    if args.studies:
        if not 2 <= len(args.studies) <= len(STUDY_COLORS):
            p.error(f"--studies takes 2-{len(STUDY_COLORS)} study numbers")
        data = [load_study(f"runs/study_{s:03d}/dense_latetime_eval_9band")
                for s in args.studies]
        outdir = args.outdir or os.path.join("runs", "interval_compare")
        os.makedirs(outdir, exist_ok=True)
        prefix = "study" + "_".join(f"{s:03d}" for s in args.studies) + "_compare_"
        for truth in truths:
            run_compare(args.studies, data, truth, outdir, prefix, args.n_boot,
                        args.seed, depths=dense_depths(data[0]["dense"])
                        if truth == "uniform" else None)
        return

    if args.eval_dir is None:
        if args.study is None:
            p.error("give --study, --eval_dir or --studies")
        args.eval_dir = f"runs/study_{args.study:03d}/dense_latetime_eval_9band"
    outdir = args.outdir or os.path.join(args.eval_dir, "interval_diagnostics")
    # Study number for the filenames: --study, else the runs/study_NNN in the path.
    study = args.study
    if study is None:
        hit = re.search(r"study_?(\d+)", os.path.abspath(args.eval_dir))
        study = int(hit.group(1)) if hit else None
    prefix = f"study{study:03d}_" if study is not None else ""
    os.makedirs(outdir, exist_ok=True)

    d = load_study(args.eval_dir, args.calibration)
    for truth in truths:
        df = d[truth]
        if df is None:
            print(f"No {truth} scored points; skipping.")
            continue
        boot = ObjectBootstrap(df["stem"].to_numpy(), args.n_boot, args.seed)
        run_truth(df, truth, d["variants"], boot, outdir, prefix,
                  depths=dense_depths(d["dense"]) if truth == "uniform" else None)


if __name__ == "__main__":
    main()
