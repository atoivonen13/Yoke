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
                       config: seed center +- 1 sigma;
     ``rmse_bias_vs_config``  (a) uniform-truth RMSE, (b) bias vs config, the
                       same markers, config axis shared;
  1. ``overview``      (a) uniform-truth RMSE, (b) dense-truth RMSE, (c) bias,
                       all bands pooled, one row per config;
  2. ``by_band``       (a) uniform RMSE, (b) bias, (c) calibrated 90% coverage
                       per band, configs side by side;
  3. ``rmse_vs_lead``  uniform RMSE vs lead time per band, center line +-1 sigma;

all in ``--style`` (default ``publication.mplstyle``; ``none`` = the house
style), each with a ``<name>_caption.txt``; plus ``configs_runs.csv`` (one row
per run), ``configs_summary.csv`` (config x band x metric: mean, median, std,
sem, min, max), ``configs_summary.tex`` and a printed table with each config's
difference from ``--reference`` (Welch t-test over seeds). Configs with no
evaluated replica are left out with a warning.

``--debias val``: post-hoc per-band level correction, the same for every run.
Each run's forecast (median and interval) is shifted per band by minus its
mean residual on the VALIDATION split (the RMSE-optimal constant shift), read
from ``<eval_dir>/calibration_split/latetime_uniform_scored_points.csv``,
which the eval and the U-Net both write. The interval scales are then re-fit on
the shifted validation points (``calibrate_intervals.fit_scales``, as the eval)
so coverage stays calibrated. Test is never used for the fit. Output goes to a
``_debiased`` dir, so the uncorrected figures are kept for comparison.

    python plot_seed_configs.py                  # 213 217 211 910 212, seeds 0-4
    python plot_seed_configs.py --center median
    python plot_seed_configs.py --debias val     # -> runs/seed_configs_..._debiased
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
from matplotlib.ticker import MaxNLocator, NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_intervals import fit_scales  # noqa: E402
from plot_compare_runs import common_objects  # noqa: E402
from plot_run_report import (  # noqa: E402
    AXIS,
    BAND_INDEX,
    BAND_NAMES,
    FULL_W,
    GROUP_NAMES,
    HALF_W,
    INK,
    INK_2,
    MUTED,
    N_BANDS,
    SERIES,
    UNIFORM_CSV,
    ObjectBootstrap,
    _add_intervals,
    _read_points,
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
from pub_style import (  # noqa: E402
    DEFAULT_FONT_DIR,
    DEFAULT_STYLE,
    load_pub_style,
    pub_context,
)

DEFAULT_CONFIGS = (213, 217, 211, 910, 212)
# Concise legend / axis names (see STUDY_LOG.md for the full recipes). All
# LodeRunner configs: event-token ViT recipe, 300 epochs, redshift off.
CONFIG_LABELS = {
    213: "ViT, fresh weights",
    217: "ViT, decoder fine-tune",  # pretrained, last 2 blocks unfrozen
    211: "ViT, full fine-tune",  # pretrained, all blocks unfrozen
    910: "U-Net",
    212: "Task-specific head only",  # ViT bypassed
}
# Two-line tick labels of the publication figure (used while the label is the
# default one above; a --labels override is wrapped automatically instead).
PUB_LABELS = {
    213: "ViT\nfresh weights",
    217: "ViT\ndecoder fine-tune",
    211: "ViT\nfull fine-tune",
    910: "U-Net",
    212: "Task-specific\nhead only",
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
# Two-line x labels of the (narrow) styled overview panels.
OVERVIEW_XLABELS = {
    "uniform_rmse": "RMSE (mag)",
    "dense_rmse": "RMSE vs.\ndense truth (mag)",
    "uniform_bias": "Mean\nresidual (mag)",
}
# Caption phrases of the overview / by-band panels.
OVERVIEW_PANELS = {
    "uniform_rmse": "RMSE against the uniformly sampled true light curve",
    "dense_rmse": "RMSE against the dense truth",
    "uniform_bias": "mean residual (predicted minus true; positive = forecast "
                    "too faint)",
}
BY_BAND_PANELS = {
    "uniform_rmse": "RMSE",
    "uniform_bias": "mean residual (predicted minus true; positive = forecast "
                    "too faint)",
    "cov_cal": "coverage of the calibrated {nominal:.0%} prediction interval "
               "(dotted: nominal)",
}
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


def debias_run(run: dict) -> np.ndarray:
    """Shift one run's test forecast per band by minus its mean VALIDATION
    residual and re-fit the interval scales on the shifted validation points.

    In place on ``run["uniform"]`` / ``run["dense"]``: median, raw and
    calibrated band and hits. Returns the per-band offsets (mag, + = the
    forecast was too faint); bands without validation points get 0.
    """
    path = os.path.join(run["eval_dir"], "calibration_split", UNIFORM_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    val = _read_points(path)
    val = val[np.isfinite(val["r"])]
    offset = (val.groupby("b")["r"].mean().reindex(range(N_BANDS))
              .fillna(0.0).to_numpy())
    shift = offset[val["b"].to_numpy()]
    scales = fit_scales({"band": val["band"].to_numpy(dtype=object),
                         "med": val["med"].to_numpy() - shift,
                         "low": val["low"].to_numpy() - shift,
                         "high": val["high"].to_numpy() - shift,
                         "true": val["true"].to_numpy()}, run["interval"])
    k_lo, k_hi = np.ones(N_BANDS), np.ones(N_BANDS)
    for name, s in scales.items():
        k_lo[BAND_INDEX[name]], k_hi[BAND_INDEX[name]] = s["k_lo"], s["k_hi"]
    cal = {"k_lo": k_lo, "k_hi": k_hi}
    for key in ("uniform", "dense"):
        df = run[key]
        if df is None:
            continue
        d = offset[df["b"].to_numpy()]
        for col in ("med", "low", "high"):
            df[col] = df[col] - d
        df["r"] = df["med"] - df["true"]
        _add_intervals(df, cal)
    run["calibrated"] = True
    return offset


def debias_configs(cfgs: list) -> None:
    """``debias_run`` every run; prints each config's offsets (median over its
    seeds) per band. Stops if any run lacks its validation scored points, so
    the configs are never compared half-corrected."""
    missing = []
    for c in cfgs:
        offs = []
        for r in c["runs"]:
            try:
                offs.append(debias_run(r))
            except FileNotFoundError as err:
                missing.append(f"  study {r['study']}: no {err}")
        c["offsets"] = np.stack(offs) if offs else None
    if missing:
        raise SystemExit("--debias val needs every run's validation scored "
                         "points:\n" + "\n".join(missing))
    print("\nValidation-fit level offsets removed (mag, + = was too faint; "
          "median over seeds):")
    print(f"  {'config':<36}" + "".join(f"{b:>7}" for b in BAND_NAMES))
    for c in cfgs:
        med = np.median(c["offsets"], axis=0)
        print(f"  {(str(c['config']) + ' ' + c['label'])[:35]:<36}"
              + "".join(f"{v:>+7.3f}" for v in med))


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


def config_style(c: dict, look: dict) -> dict:
    """Color (the look's cycle, by slot), marker and dash of one config."""
    st = series_style(c["slot"])
    st["color"] = look["cycle"][c["slot"] % len(look["cycle"])]
    return st


def config_handles(cfgs: list, look: dict, lines: bool = False) -> tuple:
    """Legend proxies: one per config (its color, marker and dash)."""
    handles = []
    for c in cfgs:
        st = config_style(c, look)
        handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"],
                                  ls=st["ls"] if lines else "-", lw=look["lw"],
                                  markersize=look["ms"]))
    return handles, [c["label"] for c in cfgs]


def category_axis(ax: plt.Axes, axis: str, look: dict) -> None:
    """A category axis: no ticks (major or minor), only the labels."""
    if look["styled"]:
        (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(NullLocator())
    ax.tick_params(axis=axis, which="both", length=0)


def band_lines(ax: plt.Axes, look: dict) -> None:
    """Hairlines between the band categories, a darker one before 'All'."""
    if not look["styled"]:
        band_separators(ax)
        return
    for gx in np.arange(len(GROUP_NAMES) - 2) + 0.5:
        ax.axvline(gx, color=look["sep"], lw=0.5, zorder=0)
    ax.axvline(N_BANDS - 0.5, color=look["ref"], lw=0.6, zorder=0)


def letter_title(ax: plt.Axes, letter: str, look: dict) -> None:
    """Panel letter above the top left corner (the style's title font)."""
    if look["styled"]:
        ax.set_title(f"({letter})", loc="left")
    else:
        panel_letter(ax, letter)


# --------------------------------------------------------------------- figures


def figure_legend(fig: plt.Figure, handles: list, labels: list, ncol: int,
                  look: dict, title: str | None = None) -> None:
    """One legend above the panels: the style's legend font when styled."""
    if not look["styled"]:
        top_legend(fig, handles, labels, ncol=ncol, title=title)
        return
    try:
        fig.legend(handles, labels, loc="outside upper center", ncol=ncol,
                   title=title)
    except ValueError:  # matplotlib < 3.7 has no 'outside' locations
        fig.legend(handles, labels, loc="upper center", ncol=ncol, title=title,
                   bbox_to_anchor=(0.5, 1.06))


def fig_overview(cfgs: list, ref: dict | None, how: str,
                 styled: bool = False) -> tuple:
    """Pooled uniform RMSE, dense RMSE and bias: seed dots + center +- 1 sigma.
    Returns (figure, caption)."""
    look = config_axes_look(styled)
    panels = [m for m in ("uniform_rmse", "dense_rmse", "uniform_bias")
              if any(np.isfinite(c["stack"][m][:, -1]).any() for c in cfgs)]
    n = len(cfgs)
    width = 2 * plt.rcParams["figure.figsize"][0] if styled else FULL_W
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, (0.6 if styled else 0.5) * n + 1.3),
        sharey=True, gridspec_kw={"width_ratios": [1.5 if m == "uniform_rmse"
                                                   else 1.0 for m in panels]})
    axes = np.atleast_1d(axes)
    y = np.arange(n, dtype=float)[::-1]
    for ax, m, letter in zip(axes, panels, "abc"):
        if not styled:
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
        category_axis(ax, "y", look)
        signed = m in SIGNED
        if signed:
            ax.axvline(0.0, color=look["ref"], lw=0.6)
        if ref is not None:
            ax.axvline(ref["center"][m][-1], color=look["muted"], lw=0.7, ls=":")
        # Numbers only on the headline panel (a): selective direct labels.
        annotate = m == "uniform_rmse"
        xs = [0.0] if signed else []
        for i, c in enumerate(cfgs):
            st = config_style(c, look)
            v = c["stack"][m][:, -1]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            cv, sd = c["center"][m][-1], c["sd"][m][-1]
            ax.plot(v, np.full(v.size, y[i] - 0.2), ls="none", marker=st["marker"],
                    color=st["color"], alpha=0.55, markersize=0.8 * look["ms"],
                    mew=0)
            ax.errorbar([cv], [y[i] + 0.1],
                        xerr=[[sd], [sd]] if np.isfinite(sd) else None,
                        fmt=st["marker"], color=st["color"],
                        markersize=1.5 * look["ms"], elinewidth=0.7 * look["lw"],
                        capsize=look["cap"], mec="white", mew=0.6)
            right = max(v.max(), cv + (sd if np.isfinite(sd) else 0.0))
            xs += [v.min(), v.max(), right]
            if np.isfinite(sd):
                xs.append(cv - sd)
            if annotate:
                txt = f"{cv:.3f}" + (f" ± {sd:.3f}" if np.isfinite(sd) else "")
                ax.annotate(txt, (right, y[i] + 0.1), xytext=(5, 0),
                            textcoords="offset points", va="center",
                            fontsize=look["note_size"] if styled else 7,
                            color=look["ink"])
        lo, hi = min(xs), max(xs)
        pad = 0.06 * (hi - lo) if hi > lo else 0.01
        ax.set_xlim(lo - pad, hi + ((16.0 if styled else 9.0) if annotate
                                    else 1.5) * pad)
        if styled:
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.set_xlabel(OVERVIEW_XLABELS[m])
        else:
            ax.set_xlabel(METRIC_NAMES[m] + (", all bands" if not signed else ""))
        letter_title(ax, letter, look)
    axes[0].set_yticks(y, [f"{c['label']}\n{c['config']} · {n_runs_label(c)[1:-1]}"
                           for c in cfgs])
    handles = [
        plt.Line2D([], [], color=look["muted"], marker="o", ls="none", alpha=0.55,
                   markersize=0.8 * look["ms"], mew=0),
        plt.Line2D([], [], color=look["muted"], marker="o", ls="-",
                   lw=0.7 * look["lw"], markersize=1.5 * look["ms"]),
    ]
    names = ["One seed", f"{how.capitalize()} ± 1σ across seeds"]
    if ref is not None:
        handles.append(plt.Line2D([], [], color=look["muted"], lw=0.7, ls=":"))
        names.append(f"{ref['label']} ({how})")
    figure_legend(fig, handles, names, len(names), look)
    caption = (
        f"Late-time forecast error, all bands pooled, on {n_objects(cfgs)} common "
        "test objects, for each model configuration: "
        + "; ".join(f"({lt}) {OVERVIEW_PANELS[m]}" for lt, m in zip("abc", panels))
        + ". Small markers: individual training seeds; large markers: the "
        f"{how} over seeds with ±1 standard deviation."
        + (f" Dotted line: the {how} of {ref['label']}." if ref is not None else ""))
    return fig, caption


def tick_label(c: dict) -> str:
    """Two-line config name (PUB_LABELS, else wrapped label) + run count."""
    name = PUB_LABELS.get(c["config"]) if c["label"] == CONFIG_LABELS.get(
        c["config"]) else None
    name = name or textwrap.fill(c["label"], 15)
    n = len(c["runs"])
    # sigma in mathtext: the style's text font may lack Greek, its math has stix.
    return f"{name}\n$n={n}$" + (" (no $\\sigma$)" if n < 2 else "")


def config_axes_look(styled: bool) -> dict:
    """Marker / text sizes and colors of the per-config panels: from the
    publication style (``styled``), else the house values."""
    prm = plt.rcParams
    if styled:
        return {"styled": True,
                "cycle": prm["axes.prop_cycle"].by_key().get("color", list(SERIES)),
                "note_size": prm["xtick.labelsize"], "note_color": prm["text.color"],
                "lw": prm["lines.linewidth"], "ms": prm["lines.markersize"],
                "mew": prm["lines.markeredgewidth"],
                "cap": prm["errorbar.capsize"] + 1,
                # Reference lines (zero, nominal, 'All' separator) in mid gray,
                # band separators in the style's grid gray.
                "ink": prm["text.color"], "ref": "0.55", "muted": "0.55",
                "sep": prm["grid.color"]}
    return {"styled": False, "cycle": list(SERIES), "note_size": 6.5,
            "note_color": INK_2, "lw": 1.8, "ms": 3.6, "mew": 0.8, "cap": 3,
            "ink": INK, "ref": AXIS, "muted": MUTED, "sep": AXIS}


def config_figure(n: int, n_rows: int, styled: bool) -> tuple:
    """Figure + axes (one column, ``n_rows`` panels sharing the config axis).

    Styled: the style's single-column width fits ~3 two-line category labels at
    its tick size; more configs go to two-column width (2x, ~7 in). Each panel
    keeps the style's height.
    """
    if styled:
        w, h = plt.rcParams["figure.figsize"]
        size = (w if n <= 3 else 2 * w, h * (1 if n_rows == 1 else 0.8 * n_rows))
    else:
        size = (HALF_W if n <= 5 else FULL_W, 2.5 if n_rows == 1 else 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=size, sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax in axes:
        if styled:
            category_axis(ax, "x", {"styled": True})
        else:
            ax.tick_params(axis="x", length=0, pad=3)
            ax.tick_params(axis="x", labelsize=7, labelcolor=INK)
            ax.spines["bottom"].set_visible(False)
    return fig, axes


def draw_config_metric(ax: plt.Axes, cfgs: list, metric: str, look: dict) -> None:
    """One metric (all bands pooled) per config: a filled marker at the seed
    center with a +-1 sigma (across seeds) error bar, the value written above.
    Signed metrics get a zero line, kept inside the y range."""
    x = np.arange(len(cfgs), dtype=float)
    signed = metric in SIGNED
    if signed:
        ax.axhline(0.0, color=look["ref"], lw=0.6, zorder=1)
    lows, highs = ([0.0], [0.0]) if signed else ([], [])
    for i, c in enumerate(cfgs):
        st = config_style(c, look)  # color follows the slot
        marker, color = st["marker"], st["color"]
        if not np.isfinite(c["stack"][metric][:, -1]).any():
            continue
        cv, sd = c["center"][metric][-1], c["sd"][metric][-1]
        sd = sd if np.isfinite(sd) else 0.0  # one run: marker only, no bar
        ax.errorbar([x[i]], [cv], yerr=[[sd], [sd]] if sd > 0 else None,
                    fmt=marker, color=color, markersize=1.5 * look["ms"],
                    mew=look["mew"], elinewidth=look["lw"], capsize=look["cap"] + 1,
                    capthick=look["lw"], zorder=4)
        lows.append(cv - sd)
        highs.append(cv + sd)
        # Typographic minus, as on the tick labels.
        text = (f"{cv:+.3f}".replace("-", "−") if signed else f"{cv:.3f}") \
            + (f" ± {sd:.3f}" if sd > 0 else "")
        ax.annotate(text, (x[i], cv + sd), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=look["note_size"],
                    color=look["note_color"])
    lo, hi = min(lows), max(highs)
    span = (hi - lo) or 0.05
    ax.set_ylim(lo - 0.10 * span, hi + 0.22 * span)  # headroom for the labels
    ax.set_xticks(x, [tick_label(c) for c in cfgs])
    ax.set_xlim(-0.55, len(cfgs) - 0.45)
    ax.set_ylabel(PUB_YLABELS[metric])


def seed_caption(cfgs: list, how: str) -> str:
    """Shared caption tail: what the markers, bars and numbers are."""
    k = sorted({len(c["runs"]) for c in cfgs})
    seeds = f"{k[0]}" if len(k) == 1 else f"{k[0]}–{k[-1]}"
    return (f"Markers: the {how} over {seeds} independent training seeds; error "
            "bars: ±1 standard deviation across seeds; numbers give the same.")


def n_objects(cfgs: list) -> int:
    """Common test objects (every run is restricted to the same set)."""
    return int(cfgs[0]["runs"][0]["uniform"]["stem"].nunique())


def fig_rmse_vs_config(cfgs: list, ref: dict | None, how: str,
                       metric: str, styled: bool = False) -> tuple:
    """Publication figure: one metric (all bands pooled) per config.

    One filled marker per config at the seed center with a +-1 sigma (across
    seeds) error bar, the value written above it. Returns (figure, caption).

    ``styled``: drawn inside a publication style (``pub_context``) -- fonts,
    sizes, ticks, spines and figure width come from it; only the category axis
    is adjusted (no ticks / minor ticks between configs).
    """
    fig, (ax,) = config_figure(len(cfgs), 1, styled)
    draw_config_metric(ax, cfgs, metric, config_axes_look(styled))
    truth = ("dense truth" if metric == "dense_rmse" else
             "the uniformly sampled true light curve")
    caption = (
        f"{PUB_YLABELS[metric].split(' (')[0]} of the late-time forecast against "
        f"{truth}, all bands pooled, on {n_objects(cfgs)} common test objects, for "
        f"each model configuration. {seed_caption(cfgs, how)}")
    return fig, caption


def fig_rmse_bias_vs_config(cfgs: list, how: str, styled: bool = False) -> tuple:
    """Publication figure: (a) uniform-truth RMSE and (b) bias per config, all
    bands pooled, the config axis shared. Returns (figure, caption)."""
    fig, axes = config_figure(len(cfgs), 2, styled)
    look = config_axes_look(styled)
    for ax, metric, letter in zip(axes, ("uniform_rmse", "uniform_bias"), "ab"):
        draw_config_metric(ax, cfgs, metric, look)
        ax.text(0.01, 0.95, f"({letter})", transform=ax.transAxes, ha="left",
                va="top", fontsize=look["note_size"], color=look["note_color"])
    axes[0].tick_params(axis="x", labelbottom=False)
    caption = (
        "Late-time forecast error against the uniformly sampled true light curve, "
        f"all bands pooled, on {n_objects(cfgs)} common test objects, for each "
        "model configuration: (a) RMSE; (b) mean residual (predicted minus true; "
        f"positive = forecast too faint). {seed_caption(cfgs, how)}")
    return fig, caption


def fig_by_band(cfgs: list, how: str, styled: bool = False) -> tuple:
    """Per-band uniform RMSE, bias and coverage: center +- 1 sigma per config.
    Returns (figure, caption)."""
    look = config_axes_look(styled)
    rows = [m for m in ("uniform_rmse", "uniform_bias", "cov_cal")
            if any(np.isfinite(c["stack"][m]).any() for c in cfgs)]
    if styled:
        w, h = plt.rcParams["figure.figsize"]
        size = (2 * w, 0.75 * h * len(rows) + 0.6)
    else:
        size = (FULL_W, 1.9 * len(rows) + 0.6)
    fig, axes = plt.subplots(len(rows), 1, figsize=size, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(GROUP_NAMES))
    n = len(cfgs)
    nominal = cfgs[0]["runs"][0]["interval"]
    for ax, m, letter in zip(axes, rows, "abc"):
        if m in SIGNED:
            ax.axhline(0.0, color=look["ref"], lw=0.6)
        if m == "cov_cal":
            ax.axhline(nominal, color=look["ink"], lw=0.6, ls=":")
        for i, c in enumerate(cfgs):
            st = config_style(c, look)
            cv, sd = c["center"][m], c["sd"][m]
            ok = np.isfinite(cv)
            ax.errorbar(x[ok] + dodge(n, i, 0.72), cv[ok],
                        yerr=np.where(np.isfinite(sd), sd, 0.0)[ok],
                        fmt=st["marker"], color=st["color"],
                        markersize=look["ms"] if styled else 3.8,
                        elinewidth=0.7 * look["lw"] if styled else 1.0,
                        capsize=look["cap"] - 1 if styled else 1.8,
                        mec="white", mew=0.4)
        band_lines(ax, look)
        if styled:
            label = PUB_YLABELS[m]
            if m == "cov_cal":
                label = f"Coverage of\n{nominal:.0%} interval"
        else:
            label = METRIC_NAMES[m]
            if m == "cov_cal":
                label = f"Coverage of calibrated\n{nominal:.0%} band (uniform)"
            elif m == "uniform_bias":
                label = "Bias (mag)\n+ = too faint"
        ax.set_ylabel(label)
        letter_title(ax, letter, look)
    band_axis(axes[-1])
    for ax in axes:
        category_axis(ax, "x", look)
    handles, labels = config_handles(cfgs, look)
    figure_legend(fig, handles, labels, min(n, 3), look)
    caption = (
        "Late-time forecast error against the uniformly sampled true light curve "
        f"per band, on {n_objects(cfgs)} common test objects, for each model "
        "configuration: "
        + "; ".join(f"({lt}) {BY_BAND_PANELS[m].format(nominal=nominal)}"
                    for lt, m in zip("abc", rows))
        + f". 'All' pools the bands. {seed_caption(cfgs, how).split('; numbers')[0]}.")
    return fig, caption


def fig_vs_lead(cfgs: list, edges: np.ndarray, boot: ObjectBootstrap,
                how: str, styled: bool = False) -> tuple:
    """Uniform RMSE vs lead time per band: center line +- 1 sigma fill per config.
    Returns (figure, caption)."""
    look = config_axes_look(styled)
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = lead_panels(4.2 if styled else 3.3)
    for c in cfgs:
        st = config_style(c, look)
        grids = np.stack([lead_grid(boot, r["uniform"], "rmse", edges)["point"]
                          for r in c["runs"]])
        cv, sd = center_of(grids, how), sigma_of(grids)
        for g, ax in enumerate(axes):
            if np.isfinite(sd[g]).any():
                ax.fill_between(centers, cv[g] - sd[g], cv[g] + sd[g],
                                color=st["color"], alpha=0.15, lw=0)
            ax.plot(centers, cv[g], color=st["color"], ls=st["ls"],
                    lw=look["lw"] if styled else 1.1)
    size = {"fontsize": plt.rcParams["axes.labelsize"]} if styled else {}
    fig.supxlabel("Lead time (days)", **size)
    fig.supylabel("RMSE (mag)" if styled else "Uniform RMSE (mag)", **size)
    handles, labels = config_handles(cfgs, look, lines=True)
    for h in handles:
        h.set_marker("none")
    figure_legend(fig, handles, labels, min(len(cfgs), 3), look,
                  title=None if styled else
                  "Line = seed center, band = ±1σ across seeds")
    caption = (
        "Late-time forecast RMSE against the uniformly sampled true light curve vs. "
        f"lead time, per band and pooled ('All bands'), on {n_objects(cfgs)} common "
        f"test objects. Lines: the {how} over training seeds; shading: ±1 standard "
        "deviation across seeds.")
    return fig, caption


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
    p.add_argument("--center", choices=("mean", "median"), default="median")
    p.add_argument("--debias", choices=("none", "val"), default="none",
                   help="'val': remove each run's per-band mean validation "
                   "residual from its test forecast (see the docstring).")
    p.add_argument("--metric", choices=tuple(METRIC_NAMES), default="uniform_rmse",
                   help="Metric of the rmse_vs_config figure.")
    p.add_argument("--config_run", choices=("fallback", "always", "never"),
                   default="fallback",
                   help="Original study <config> itself: use it when no replica "
                   "is evaluated yet (fallback), add it as an extra seed (always), "
                   "or ignore it (never).")
    p.add_argument("--style", default=DEFAULT_STYLE,
                   help="Matplotlib style of every figure (path, or a name "
                   "searched in the cwd, this dir and its parents); 'none' for "
                   "the house style.")
    p.add_argument("--font_dir", default=DEFAULT_FONT_DIR,
                   help="Extra .otf/.ttf fonts to register (the style's font).")
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

    cfgs = load_configs(args.configs, labels, args.seeds, args.runs_root,
                        args.config_run)
    if not cfgs:
        raise SystemExit("No config has an evaluated replica.")
    if args.debias == "val":
        debias_configs(cfgs)
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
        args.runs_root, "seed_configs_" + "_".join(str(c["config"]) for c in cfgs)
        + ("_debiased" if args.debias == "val" else ""))
    os.makedirs(outdir, exist_ok=True)
    fmts = tuple(args.formats)
    style = None if args.style == "none" else load_pub_style(args.style,
                                                             args.font_dir)
    # Drawn AND saved inside the style: savefig.* (dpi, padding) and the fonts
    # are read at save time.
    styled = style is not None
    if not styled:
        set_style()
    with pub_context(style):
        pub_figs = (
            ("rmse_vs_config", fig_rmse_vs_config(cfgs, ref, args.center,
                                                  args.metric, styled=styled)),
            ("rmse_bias_vs_config", fig_rmse_bias_vs_config(
                cfgs, args.center, styled=styled)),
            ("overview", fig_overview(cfgs, ref, args.center, styled=styled)),
            ("by_band", fig_by_band(cfgs, args.center, styled=styled)),
            ("rmse_vs_lead", fig_vs_lead(cfgs, lead_edges(runs), boot,
                                         args.center, styled=styled)),
        )
        for name, (fig, _) in pub_figs:
            save(fig, outdir, name, fmts)
    for name, (_, caption) in pub_figs:
        path = os.path.join(outdir, f"{name}_caption.txt")
        if args.debias == "val":
            caption += (" Each run's forecast is shifted per band by minus its"
                        " mean residual on the validation split, and its"
                        " interval scales re-fit there.")
        with open(path, "w") as fh:
            fh.write(caption + "\n")
        print(f"Wrote {path}")
    write_tables(cfgs, ref, args.center, boot.n_obj, outdir)


if __name__ == "__main__":
    main()
