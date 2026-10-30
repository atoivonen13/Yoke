"""Seed-averaged comparison of several configs (no model / GPU / torch).

Each config's replicas are studies ``config * 100 + seed`` (``make_seed_sweep.py``
for the LodeRunner configs, ``unet_baseline_9band.py --study 910<seed>`` for the
U-Net). Every evaluated replica is loaded as in ``plot_seed_replicas.py`` and all
runs are scored on the objects COMMON to every run, so the configs differ only in
model and seed, never in test sample. A config is drawn as its center over seeds
(``--center`` mean or median) with a +-1 sigma bar (sample std ACROSS SEEDS, the
noise that matters between configs -- not a test-sample CI); single seeds are
small dots.

Figures (PDF + PNG) in ``--outdir``:

  0. ``rmse_vs_config`` ``--metric`` (default uniform-truth RMSE, all bands) vs
                       config: seed dots + center +- 1 sigma;
  1. ``overview``      (a) uniform-truth RMSE, (b) dense-truth RMSE, (c) bias,
                       all bands pooled, one row per config;
  2. ``by_band``       (a) uniform RMSE, (b) bias, (c) calibrated 90% coverage
                       per band, configs side by side;
  3. ``rmse_vs_lead``  uniform RMSE vs lead time per band, center line +-1 sigma;

plus ``configs_runs.csv`` (one row per run), ``configs_summary.csv`` (config x
band x metric: mean, median, std, sem, min, max), ``configs_summary.tex`` and a
printed table with each config's difference from ``--reference`` (Welch t-test
over seeds). Configs with no evaluated replica are left out with a warning.

    python plot_seed_configs.py                  # 213 211 217 212 910, seeds 0-4
    python plot_seed_configs.py --center median
    python plot_seed_configs.py --configs 213 910 --labels "ViT" "U-Net"
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_compare_runs import GRID_SEP, common_objects  # noqa: E402
from plot_run_report import (  # noqa: E402
    AXIS,
    FULL_W,
    GROUP_NAMES,
    HALF_W,
    INK,
    INK_2,
    MUTED,
    ObjectBootstrap,
    band_axis,
    dodge,
    fmt,
    lead_edges,
    lead_grid,
    lead_panels,
    metrics_table,
    panel_letter,
    restrict_to,
    save,
    series_style,
    set_style,
    top_legend,
    write_tex,
)
from plot_seed_replicas import band_separators, load_available  # noqa: E402

DEFAULT_CONFIGS = (213, 211, 217, 212, 910)
# Concise legend / axis names (see STUDY_LOG.md for the full recipes). All
# LodeRunner configs: event-token ViT recipe, 300 epochs, redshift off.
CONFIG_LABELS = {
    213: "ViT, random init",
    211: "ViT, pretrained",
    217: "ViT, pretrained, last 2 blocks",
    212: "No ViT (head only)",
    910: "1D U-Net",
}
# Two-line tick labels of the publication figure (used while the label is the
# default one above; a --labels override is wrapped automatically instead).
PUB_LABELS = {
    213: "ViT\nrandom init",
    211: "ViT\npretrained",
    217: "ViT, pretrained\nlast 2 blocks",
    212: "No ViT\n(head only)",
    910: "1D U-Net\nbaseline",
}
PUB_YLABELS = {
    "uniform_rmse": "RMSE (mag)",
    "dense_rmse": "RMSE vs. dense truth (mag)",
    "uniform_bias": "Mean residual (mag)",
    "beyond_bias": "Mean residual beyond depth (mag)",
    "cov_cal": "Coverage of 90% interval",
}
METRIC_NAMES = {
    "uniform_rmse": "Uniform-truth RMSE (mag)",
    "dense_rmse": "Dense-truth RMSE (mag)",
    "uniform_bias": "Bias (mag; + = too faint)",
    "beyond_bias": "Beyond-depth bias (mag)",
    "cov_cal": "Calibrated coverage",
}
SIGNED = ("uniform_bias", "beyond_bias")
# Metrics in the tables and printout, in column order.
TABLE_METRICS = ("uniform_rmse", "dense_rmse", "uniform_bias", "beyond_bias",
                 "cov_cal")


def center_of(v: np.ndarray, how: str) -> np.ndarray:
    """Mean or median over seeds (rows), NaN-safe."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        return np.nanmedian(v, axis=0) if how == "median" else np.nanmean(v, axis=0)


def sigma_of(v: np.ndarray) -> np.ndarray:
    """Sample std over seeds (rows); NaN with fewer than 2 seeds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(v, axis=0, ddof=1)


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value of two seed samples (NaN if not testable)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        return np.nan
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def load_configs(configs: list, labels: list, seeds: list, runs_root: str,
                 config_run: str) -> list:
    """Evaluated replicas per config. Each config keeps the series slot of its
    position in ``configs``, so leaving one out never repaints the others.

    ``config_run``: ``fallback`` uses the original study ``config`` (one run, no
    seed sigma) when no replica is evaluated yet; ``always`` adds it as one more
    seed; ``never`` uses replicas only.
    """
    out = []
    for slot, (config, label) in enumerate(zip(configs, labels)):
        studies = [config * 100 + s for s in seeds]
        runs = load_available(studies, runs_root,
                              [f"{label}, seed {s % 100}" for s in studies])
        n_rep = len(runs)
        if config_run == "always" or (config_run == "fallback" and not runs):
            runs += load_available([config], runs_root, [f"{label}, study {config}"])
        if not runs:
            print(f"WARNING: config {config}: no evaluated replica or study "
                  f"{config}; left out.")
            continue
        out.append({"config": config, "label": label, "slot": slot, "runs": runs,
                    "n_rep": n_rep})
    print("\nRuns per config:")
    for c in out:
        note = "" if c["n_rep"] == len(c["runs"]) else \
            f"  ({c['n_rep']} replicas + original study {c['config']})"
        if c["n_rep"] == 0:
            note = f"  (NO replica evaluated: original study {c['config']} only, no σ)"
        print(f"  {c['config']:>4} {c['label']:<32} "
              f"{[r['study'] for r in c['runs']]}{note}")
    return out


def n_runs_label(c: dict) -> str:
    """Tick-label suffix: seed count, or that only the single original run exists."""
    if c["n_rep"] == 0:
        return "(1 run, no σ)"
    return f"({len(c['runs'])} seeds)"


def add_stats(cfgs: list, boot: ObjectBootstrap, how: str) -> None:
    """Per config: ``stack[metric]`` = [n_seeds, groups], its center and sigma."""
    for c in cfgs:
        tables = [metrics_table(r, boot) for r in c["runs"]]
        c["stack"] = {m: np.stack([t[m].to_numpy(dtype=float) if m in t
                                   else np.full(len(GROUP_NAMES), np.nan)
                                   for t in tables])
                      for m in METRIC_NAMES}
        c["center"] = {m: center_of(v, how) for m, v in c["stack"].items()}
        c["sd"] = {m: sigma_of(v) for m, v in c["stack"].items()}
        c["tables"] = tables


def config_handles(cfgs: list, lines: bool = False) -> tuple:
    """Legend proxies: one per config (its color, marker and dash)."""
    handles = []
    for c in cfgs:
        st = series_style(c["slot"])
        handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"],
                                  ls=st["ls"] if lines else "-", lw=1.0,
                                  markersize=4))
    return handles, [c["label"] for c in cfgs]


# --------------------------------------------------------------------- figures


def fig_overview(cfgs: list, ref: dict | None, how: str) -> plt.Figure:
    """Pooled uniform RMSE, dense RMSE and bias: seed dots + center +- 1 sigma."""
    panels = [m for m in ("uniform_rmse", "dense_rmse", "uniform_bias")
              if any(np.isfinite(c["stack"][m][:, -1]).any() for c in cfgs)]
    n = len(cfgs)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(FULL_W, 0.5 * n + 1.1), sharey=True,
        gridspec_kw={"width_ratios": [1.5 if m == "uniform_rmse" else 1.0
                                      for m in panels]})
    axes = np.atleast_1d(axes)
    y = np.arange(n, dtype=float)[::-1]
    for ax, m, letter in zip(axes, panels, "abc"):
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        signed = m in SIGNED
        if signed:
            ax.axvline(0.0, color=AXIS, lw=0.6)
        if ref is not None:
            ax.axvline(ref["center"][m][-1], color=MUTED, lw=0.7, ls=":")
        # Numbers only on the headline panel (a): selective direct labels.
        annotate = m == "uniform_rmse"
        xs = [0.0] if signed else []
        for i, c in enumerate(cfgs):
            st = series_style(c["slot"])
            v = c["stack"][m][:, -1]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            cv, sd = c["center"][m][-1], c["sd"][m][-1]
            ax.plot(v, np.full(v.size, y[i] - 0.2), ls="none", marker=st["marker"],
                    color=st["color"], alpha=0.55, markersize=3, mew=0)
            ax.errorbar([cv], [y[i] + 0.1],
                        xerr=[[sd], [sd]] if np.isfinite(sd) else None,
                        fmt=st["marker"], color=st["color"], markersize=5.5,
                        elinewidth=1.2, capsize=2.5, mec="white", mew=0.6)
            right = max(v.max(), cv + (sd if np.isfinite(sd) else 0.0))
            xs += [v.min(), v.max(), right]
            if np.isfinite(sd):
                xs.append(cv - sd)
            if annotate:
                txt = f"{cv:.3f}" + (f" ± {sd:.3f}" if np.isfinite(sd) else "")
                ax.annotate(txt, (right, y[i] + 0.1), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=7,
                            color=INK)
        lo, hi = min(xs), max(xs)
        pad = 0.06 * (hi - lo) if hi > lo else 0.01
        ax.set_xlim(lo - pad, hi + (9.0 if annotate else 1.5) * pad)
        ax.set_xlabel(METRIC_NAMES[m] + (", all bands" if not signed else ""))
        panel_letter(ax, letter)
    axes[0].set_yticks(y, [f"{c['label']}\n{c['config']} · {n_runs_label(c)[1:-1]}"
                           for c in cfgs])
    handles = [
        plt.Line2D([], [], color=MUTED, marker="o", ls="none", alpha=0.55,
                   markersize=3, mew=0),
        plt.Line2D([], [], color=MUTED, marker="o", ls="-", lw=1.2, markersize=5),
    ]
    names = ["One seed", f"{how.capitalize()} ± 1σ across seeds"]
    if ref is not None:
        handles.append(plt.Line2D([], [], color=MUTED, lw=0.7, ls=":"))
        names.append(f"{ref['label']} ({how})")
    top_legend(fig, handles, names, ncol=len(names))
    return fig


def tick_label(c: dict) -> str:
    """Two-line config name (PUB_LABELS, else wrapped label) + run count."""
    name = PUB_LABELS.get(c["config"]) if c["label"] == CONFIG_LABELS.get(
        c["config"]) else None
    name = name or textwrap.fill(c["label"], 15)
    n = len(c["runs"])
    return f"{name}\n$n={n}$" + (" (no σ)" if n < 2 else "")


def fig_rmse_vs_config(cfgs: list, ref: dict | None, how: str,
                       metric: str) -> tuple:
    """Publication figure: one metric (all bands pooled) per config.

    Hollow markers = individual seeds (evenly jittered), horizontal bar = seed
    center, whisker = +-1 sigma across seeds; ``ref``'s center +- sigma is a light
    horizontal band so every config reads against it. Returns (figure, caption).
    """
    n = len(cfgs)
    fig, ax = plt.subplots(figsize=(HALF_W if n <= 5 else FULL_W, 2.5))
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7, labelcolor=INK)
    ax.spines["bottom"].set_visible(False)
    x = np.arange(n, dtype=float)
    if metric in SIGNED:
        ax.axhline(0.0, color=AXIS, lw=0.6, zorder=1)
    has_ref = ref is not None and np.isfinite(ref["center"][metric][-1])
    if has_ref:
        rc, rs = ref["center"][metric][-1], ref["sd"][metric][-1]
        if np.isfinite(rs):
            ax.axhspan(rc - rs, rc + rs, color=GRID_SEP, lw=0, zorder=0)
        ax.axhline(rc, color=AXIS, lw=0.6, ls=(0, (2, 2)), zorder=1)
    half = 0.17  # half-width of the center bar (x units)
    lows, highs = [], []
    for i, c in enumerate(cfgs):
        st = series_style(c["slot"])
        v = c["stack"][metric][:, -1]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        cv, sd = c["center"][metric][-1], c["sd"][metric][-1]
        jit = np.linspace(-0.11, 0.11, v.size) if v.size > 1 else np.zeros(1)
        ax.plot(x[i] + jit, v, ls="none", marker=st["marker"], mfc="white",
                mec=st["color"], mew=0.8, markersize=3.6, zorder=3)
        if v.size > 1:
            ax.errorbar([x[i]], [cv], yerr=[[sd], [sd]], fmt="none",
                        ecolor=st["color"], elinewidth=0.9, capsize=3,
                        capthick=0.9, zorder=4)
            ax.plot([x[i] - half, x[i] + half], [cv, cv], color=st["color"],
                    lw=1.8, solid_capstyle="butt", zorder=4)
        top = max(v.max(), cv + (sd if np.isfinite(sd) else 0.0))
        lows.append(min(v.min(), cv - (sd if np.isfinite(sd) else 0.0)))
        highs.append(top)
        text = (f"{cv:.3f}" if v.size < 2 else f"{cv:.3f}\n±{sd:.3f}")
        ax.annotate(text, (x[i], top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=INK_2,
                    linespacing=1.0)
    lo, hi = min(lows), max(highs)
    if has_ref:
        lo, hi = min(lo, rc - (rs if np.isfinite(rs) else 0)), max(hi, rc)
    span = (hi - lo) or 0.05
    ax.set_ylim(lo - 0.08 * span, hi + 0.30 * span)  # headroom for the labels
    ax.set_xticks(x, [tick_label(c) for c in cfgs])
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylabel(PUB_YLABELS[metric])

    k = sorted({len(c["runs"]) for c in cfgs})
    seeds = f"{k[0]}" if len(k) == 1 else f"{k[0]}–{k[-1]}"
    n_obj = int(cfgs[0]["runs"][0]["uniform"]["stem"].nunique())
    truth = ("dense truth" if metric == "dense_rmse" else
             "the uniformly sampled true light curve")
    caption = (
        f"{PUB_YLABELS[metric].split(' (')[0]} of the late-time forecast against "
        f"{truth}, all bands pooled, on {n_obj} common test objects, for each model "
        f"configuration. Hollow markers: independent training seeds ({seeds} per "
        f"configuration); bar and whisker: {how} ± 1 standard deviation across "
        "seeds; numbers give the same."
        + (f" Shaded band and dashed line: {ref['label']} ({how} ± 1σ)."
           if has_ref else ""))
    return fig, caption


def fig_by_band(cfgs: list) -> plt.Figure:
    """Per-band uniform RMSE, bias and coverage: center +- 1 sigma per config."""
    rows = [m for m in ("uniform_rmse", "uniform_bias", "cov_cal")
            if any(np.isfinite(c["stack"][m]).any() for c in cfgs)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(FULL_W, 1.9 * len(rows) + 0.6),
                             sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(GROUP_NAMES))
    n = len(cfgs)
    nominal = cfgs[0]["runs"][0]["interval"]
    for ax, m, letter in zip(axes, rows, "abc"):
        if m in SIGNED:
            ax.axhline(0.0, color=AXIS, lw=0.6)
        if m == "cov_cal":
            ax.axhline(nominal, color=INK, lw=0.6, ls=":")
        for i, c in enumerate(cfgs):
            st = series_style(c["slot"])
            cv, sd = c["center"][m], c["sd"][m]
            ok = np.isfinite(cv)
            ax.errorbar(x[ok] + dodge(n, i, 0.72), cv[ok],
                        yerr=np.where(np.isfinite(sd), sd, 0.0)[ok],
                        fmt=st["marker"], color=st["color"], markersize=3.8,
                        elinewidth=1.0, capsize=1.8, mec="white", mew=0.4)
        band_separators(ax)
        label = METRIC_NAMES[m]
        if m == "cov_cal":
            label = f"Coverage of calibrated\n{nominal:.0%} band (uniform)"
        elif m == "uniform_bias":
            label = "Bias (mag)\n+ = too faint"
        ax.set_ylabel(label)
        panel_letter(ax, letter)
    band_axis(axes[-1])
    handles, labels = config_handles(cfgs)
    top_legend(fig, handles, labels, ncol=min(n, 3))
    return fig


def fig_vs_lead(cfgs: list, edges: np.ndarray, boot: ObjectBootstrap,
                how: str) -> plt.Figure:
    """Uniform RMSE vs lead time per band: center line +- 1 sigma fill per config."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = lead_panels()
    for c in cfgs:
        st = series_style(c["slot"])
        grids = np.stack([lead_grid(boot, r["uniform"], "rmse", edges)["point"]
                          for r in c["runs"]])
        cv, sd = center_of(grids, how), sigma_of(grids)
        for g, ax in enumerate(axes):
            if np.isfinite(sd[g]).any():
                ax.fill_between(centers, cv[g] - sd[g], cv[g] + sd[g],
                                color=st["color"], alpha=0.15, lw=0)
            ax.plot(centers, cv[g], color=st["color"], ls=st["ls"], lw=1.1)
    fig.supxlabel("Lead time (days)")
    fig.supylabel("Uniform RMSE (mag)")
    handles, labels = config_handles(cfgs, lines=True)
    for h in handles:
        h.set_marker("none")
    top_legend(fig, handles, labels, ncol=min(len(cfgs), 3),
               title="Line = seed center, band = ±1σ across seeds")
    return fig


# ---------------------------------------------------------------------- tables


def write_tables(cfgs: list, ref: dict | None, how: str, n_obj: int,
                 outdir: str) -> None:
    """Per-run CSV, per-config summary CSV + LaTeX, and the printed comparison."""
    runs_rows, summary = [], []
    for c in cfgs:
        for r, t in zip(c["runs"], c["tables"]):
            pooled = t.iloc[-1]
            runs_rows.append({"config": c["config"], "label": c["label"],
                              "study": r["study"], "seed": r["study"] % 100,
                              "epoch": r["epoch"]}
                             | {m: pooled.get(m, np.nan) for m in TABLE_METRICS})
        k = len(c["runs"])
        for g, group in enumerate(GROUP_NAMES):
            for m in TABLE_METRICS:
                v = c["stack"][m][:, g]
                if not np.isfinite(v).any():
                    continue
                sd = c["sd"][m][g]
                summary.append({"config": c["config"], "label": c["label"],
                                "n_seeds": k, "group": group, "metric": m,
                                "mean": np.nanmean(v), "median": np.nanmedian(v),
                                "std": sd, "sem": sd / np.sqrt(k),
                                "min": np.nanmin(v), "max": np.nanmax(v)})
    for name, rows in (("configs_runs.csv", runs_rows),
                       ("configs_summary.csv", summary)):
        path = os.path.join(outdir, name)
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.5f")
        print(f"Wrote {path}")

    def delta(c: dict) -> tuple:
        """(center diff, combined sem, Welch p) of pooled uniform RMSE vs ref."""
        if ref is None or c is ref:
            return np.nan, np.nan, np.nan
        a, b = c["stack"]["uniform_rmse"][:, -1], ref["stack"]["uniform_rmse"][:, -1]
        se = np.sqrt(np.nanvar(a, ddof=1) / a.size + np.nanvar(b, ddof=1) / b.size) \
            if a.size > 1 and b.size > 1 else np.nan
        return (c["center"]["uniform_rmse"][-1] - ref["center"]["uniform_rmse"][-1],
                se, welch_p(a, b))

    cols = [m for m in TABLE_METRICS if any(np.isfinite(c["stack"][m][:, -1]).any()
                                            for c in cfgs)]
    tex_rows = []
    for c in cfgs:
        cells = [c["label"], str(len(c["runs"]))]
        for m in cols:
            cells.append(f"${fmt(c['center'][m][-1], sign=m in SIGNED)} \\pm "
                         f"{fmt(c['sd'][m][-1])}$")
        d, se, p = delta(c)
        if ref is not None:
            cells.append("--" if not np.isfinite(d) else
                         f"${fmt(d, sign=True)} \\pm {fmt(se)}$"
                         + (f" ($p={p:.2g}$)" if np.isfinite(p) else ""))
        tex_rows.append(cells)
    header = ["Model", "Seeds"] + [METRIC_NAMES[m].split(" (")[0] for m in cols]
    if ref is not None:
        header.append(f"$\\Delta$ RMSE vs {ref['label']}")
    write_tex(os.path.join(outdir, "configs_summary.tex"), header, tex_rows,
              f"Late-time forecast error, all bands pooled, on {n_obj} common test "
              f"objects: {how} $\\pm$ standard deviation across training seeds "
              "(mag; residual = predicted $-$ true, $+$ = too faint). $\\Delta$ = "
              f"difference of the {how}s $\\pm$ combined standard error, $p$ from "
              "a Welch $t$-test over seeds.")

    print(f"\nAll bands, {n_obj} common objects; {how} ± 1σ across seeds")
    print(f"  {'config':<36}{'n':>3}{'uniform RMSE':>18}{'dense RMSE':>18}"
          f"{'bias':>16}{'coverage':>16}" + (f"{'Δ uniform':>20}{'p':>8}"
                                              if ref is not None else ""))
    for c in cfgs:
        line = f"  {(str(c['config']) + ' ' + c['label'])[:35]:<36}{len(c['runs']):>3}"
        for m, w in (("uniform_rmse", 18), ("dense_rmse", 18), ("uniform_bias", 16),
                     ("cov_cal", 16)):
            cell = f"{fmt(c['center'][m][-1], sign=m in SIGNED)} ± {fmt(c['sd'][m][-1])}"
            line += f"{cell:>{w}}"
        if ref is not None:
            d, se, p = delta(c)
            cell = "ref" if c is ref else f"{fmt(d, 4, True)} ± {fmt(se, 4)}"
            line += f"{cell:>20}{fmt(p, 3):>8}"
        print(line)
    print("  (σ = sample std over seeds; Δ ± combined sem; p = Welch t-test. "
          "With 5 seeds, σ itself is uncertain by ~±35%.)")


# ------------------------------------------------------------------------ main


def main() -> None:
    """Load every config's evaluated replicas and write the comparison."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", type=int, nargs="+", default=list(DEFAULT_CONFIGS),
                   help="Config studies; replicas are config*100 + seed.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="One per config (default: built-in short names, else "
                   "'Study N').")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--reference", type=int, default=None,
                   help="Config the differences are taken against (default: the "
                   "first; -1 for none).")
    p.add_argument("--center", choices=("mean", "median"), default="mean")
    p.add_argument("--metric", choices=tuple(METRIC_NAMES), default="uniform_rmse",
                   help="Metric of the rmse_vs_config figure.")
    p.add_argument("--config_run", choices=("fallback", "always", "never"),
                   default="fallback",
                   help="Original study <config> itself: use it when no replica "
                   "is evaluated yet (fallback), add it as an extra seed (always), "
                   "or ignore it (never).")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--outdir", default=None,
                   help="Figure dir (default runs/seed_configs_<configs>).")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--n_boot", type=int, default=100,
                   help="Object resamples (only the point estimates are used).")
    args = p.parse_args()
    if len(set(args.configs)) != len(args.configs):
        p.error("duplicate --configs")
    if args.labels is not None and len(args.labels) != len(args.configs):
        p.error("--labels needs one label per config")
    if any(not 0 <= s < 100 for s in args.seeds):
        p.error("--seeds must be in 0-99")
    labels = args.labels or [CONFIG_LABELS.get(c, f"Study {c}") for c in args.configs]

    set_style()
    cfgs = load_configs(args.configs, labels, args.seeds, args.runs_root,
                        args.config_run)
    if not cfgs:
        raise SystemExit("No config has an evaluated replica.")
    runs = [r for c in cfgs for r in c["runs"]]
    common = common_objects(runs)
    n_each = [r["uniform"]["stem"].nunique() for r in runs]
    print(f"\n{len(common)} test objects common to all {len(runs)} runs "
          f"(per run {min(n_each)}-{max(n_each)}).")
    for r in runs:
        restrict_to(r, common)
    boot = ObjectBootstrap(list(common), args.n_boot, 0)
    for r in runs:
        boot.attach(r)
    add_stats(cfgs, boot, args.center)
    ref_id = args.configs[0] if args.reference is None else args.reference
    ref = next((c for c in cfgs if c["config"] == ref_id), None)
    if ref is None and ref_id != -1:
        print(f"WARNING: reference config {ref_id} not loaded; no differences.")

    outdir = args.outdir or os.path.join(
        args.runs_root, "seed_configs_" + "_".join(str(c["config"]) for c in cfgs))
    os.makedirs(outdir, exist_ok=True)
    fmts = tuple(args.formats)
    fig, caption = fig_rmse_vs_config(cfgs, ref, args.center, args.metric)
    save(fig, outdir, "rmse_vs_config", fmts)
    path = os.path.join(outdir, "rmse_vs_config_caption.txt")
    with open(path, "w") as fh:
        fh.write(caption + "\n")
    print(f"Wrote {path}")
    for name, fig in (("overview", fig_overview(cfgs, ref, args.center)),
                      ("by_band", fig_by_band(cfgs)),
                      ("rmse_vs_lead", fig_vs_lead(cfgs, lead_edges(runs), boot,
                                                  args.center))):
        save(fig, outdir, name, fmts)
    write_tables(cfgs, ref, args.center, boot.n_obj, outdir)


if __name__ == "__main__":
    main()
