"""Seed-replica spread of ONE config (no model / GPU / torch).

Reads the eval outputs and loss records of the seed replicas written by
``make_seed_sweep.py`` (replica study = config * 100 + seed) exactly as
``plot_run_report.py`` does, and shows how much the seeds differ. Like
``plot_compare_runs.py``, every run is scored on the objects COMMON to all of
them with one shared set of object-bootstrap resamples, so the seed-to-seed
differences are not test-sample noise: the test set is fixed, only the seed
changes. The original config study (e.g. 213) is drawn in gray for reference.

Figures (PDF + PNG, 7.0 in two-column width) in ``--outdir``:

  1. ``overview``       (a) overall uniform RMSE per seed (90% CI whisker, dense
                        RMSE overlaid), (b) pooled bias; the seed mean and +-1
                        sigma across seeds shaded;
  2. ``band_spread``    (a) uniform RMSE and (b) bias per band, one marker per
                        seed + the seed mean; (c, d) the seed sigma per band next
                        to the test-sample sigma (bootstrap) of a single run;
  3. ``rmse_vs_lead``   uniform RMSE vs lead time per band, one line per seed
                        + the seed mean;
  4. ``bias_vs_lead``   the same for bias;
  5. ``coverage``       calibrated 90% coverage per band, one marker per seed;
  6. ``loss``           validation loss vs epoch per seed, (b) zoomed on the
                        last third; evaluated checkpoint starred;

plus ``seed_runs.csv`` (one row per run), ``seed_bands.csv`` (per band and
run), ``seed_summary.csv`` / ``seed_summary.tex`` (per band: mean, std, min,
max across seeds) and a printed summary with the difference two configs
need to show to be resolved at this seed count.

Evaluate every replica first (replicas still training are skipped):

    python eval_dense_latetime_9band.py --study 21300 --sweep_epochs 200:300:10
    ...
    python plot_seed_replicas.py --config 213 --seeds 0 1 2 3 4
    # -> runs/seeds_213/overview.pdf, ...
"""

from __future__ import annotations

import argparse
import os
import sys
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
    BAND_NAMES,
    FULL_W,
    GROUP_LABELS,
    GROUP_NAMES,
    INK,
    INK_2,
    MUTED,
    N_BANDS,
    SERIES,
    ObjectBootstrap,
    band_axis,
    dodge,
    errorbar,
    fmt,
    lead_edges,
    lead_grid,
    lead_panels,
    load_run,
    metrics_table,
    panel_letter,
    restrict_to,
    save,
    series_style,
    set_style,
    summarize,
    top_legend,
    write_tex,
)

# Reference (original) studies: gray, hollow, never a categorical slot.
REF_MARKERS = ("s", "D", "^", "v")
# Columns of metrics_table summarized across seeds (those present are used).
SUMMARY_METRICS = ("uniform_rmse", "uniform_bias", "within_rmse", "beyond_rmse",
                   "beyond_bias", "dense_rmse", "dense_bias", "cov_cal",
                   "width_cal")


def seed_style(i: int, n: int) -> dict:
    """Seed ``i`` of ``n``: its own slot up to 8 seeds, one shared slot past that."""
    if n <= len(SERIES):
        return series_style(i)
    return {"color": SERIES[0], "marker": "o", "ls": "-"}


def ref_style(j: int) -> dict:
    """Reference study ``j``: hollow gray marker, dashed gray line."""
    return {"color": INK_2, "marker": REF_MARKERS[j % len(REF_MARKERS)], "ls": "--",
            "mfc": "white"}


def legend_handles(seeds: list, refs: list, lines: bool = True,
                   mean: bool = True) -> tuple:
    """Proxy artists: each seed (or one entry past 8), the seed mean, the refs."""
    n = len(seeds)
    handles, labels = [], []
    shown = seeds if n <= len(SERIES) else seeds[:1]
    for i, r in enumerate(shown):
        st = seed_style(i, n)
        handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"],
                                  ls=st["ls"] if lines else "none", lw=1.0,
                                  markersize=4))
        labels.append(r["label"] if n <= len(SERIES) else f"{n} seeds")
    if mean:
        handles.append(plt.Line2D([], [], color=INK, lw=1.4))
        labels.append(f"Mean of {n} seeds")
    for j, r in enumerate(refs):
        st = ref_style(j)
        handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"],
                                  mfc="white", ls=st["ls"] if lines else "none",
                                  lw=0.9, markersize=4))
        labels.append(r["label"])
    return handles, labels


def points(runs: list, boot: ObjectBootstrap, kind: str,
           key: str = "uniform") -> np.ndarray:
    """Per-band + pooled point estimates, ``[n_runs, 10]`` (NaN rows if no data)."""
    out = np.full((len(runs), len(GROUP_NAMES)), np.nan)
    for i, r in enumerate(runs):
        if r[key] is not None:
            out[i] = summarize(boot, r[key], kind)["point"]
    return out


def test_sd(runs: list, boot: ObjectBootstrap, kind: str) -> np.ndarray:
    """Test-sample (object-bootstrap) sigma of one run, RMS over the runs: ``[10]``."""
    var = [np.nanvar(summarize(boot, r["uniform"], kind)["boot"], axis=0)
           for r in runs]
    return np.sqrt(np.mean(var, axis=0))


def seed_sd(vals: np.ndarray) -> np.ndarray:
    """Sample std across seeds (rows), NaN-safe."""
    return np.nanstd(vals, axis=0, ddof=1)


def detectable_delta(sd: float, n: int, alpha: float = 0.05,
                     power: float = 0.8) -> float:
    """Smallest config difference a two-sided t-test resolves with n seeds per arm."""
    try:
        from scipy.stats import t

        q = t.ppf(1 - alpha / 2, 2 * n - 2) + t.ppf(power, 2 * n - 2)
    except ImportError:  # normal approximation
        q = 1.960 + 0.842
    return float(q * sd * np.sqrt(2.0 / n))


def band_separators(ax: plt.Axes) -> None:
    """Hairlines between band categories; a darker one before 'All'."""
    for gx in np.arange(len(GROUP_NAMES) - 2) + 0.5:
        ax.axvline(gx, color=GRID_SEP, lw=0.4, zorder=0)
    ax.axvline(N_BANDS - 0.5, color=AXIS, lw=0.6)


def seed_scatter(ax: plt.Axes, vals: np.ndarray, ref_vals: np.ndarray) -> None:
    """Per-group dodged marker per seed, a short ink tick at the seed mean,
    and each reference study as a hollow gray marker at the category's edge."""
    n = len(vals)
    x = np.arange(len(GROUP_NAMES))
    for i in range(n):
        st = seed_style(i, n)
        ax.plot(x + dodge(n, i, 0.66), vals[i], ls="none", marker=st["marker"],
                color=st["color"], markersize=3.5)
    ax.hlines(np.nanmean(vals, axis=0), x - 0.36, x + 0.36, color=INK, lw=1.1,
              zorder=3)
    for j, rv in enumerate(ref_vals):
        st = ref_style(j)
        ax.plot(x + 0.43, rv, ls="none", marker=st["marker"], mfc="white",
                mec=st["color"], mew=0.8, markersize=3.2)


# --------------------------------------------------------------------- figures


def fig_overview(seeds: list, refs: list, boot: ObjectBootstrap) -> plt.Figure:
    """Overall uniform RMSE and pooled bias per run, seed mean +- 1 sigma shaded."""
    runs = seeds + refs
    n, ns = len(runs), len(seeds)
    rmse = [summarize(boot, r["uniform"], "rmse", "all") for r in runs]
    bias = [summarize(boot, r["uniform"], "bias", "all") for r in runs]
    y = np.arange(n, dtype=float)[::-1]
    y[:ns] += 0.6 if refs else 0.0  # gap between the seeds and the references
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 0.34 * n + 1.3), sharey=True)
    panels = ((axes[0], rmse, "Uniform-truth RMSE, all bands (mag)", False),
              (axes[1], bias, "Uniform-truth bias, all bands (mag; + = too faint)",
               True))
    for (ax, stats, xlabel, signed), letter in zip(panels, "ab"):
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        vals = np.array([s["point"][0] for s in stats[:ns]])
        mean, sd = vals.mean(), vals.std(ddof=1)
        ax.axvspan(mean - sd, mean + sd, color=AXIS, alpha=0.35, lw=0)
        ax.axvline(mean, color=INK, lw=0.8)
        if signed:
            ax.axvline(0.0, color=AXIS, lw=0.6)
        ax.set_title(f"seed mean {mean:.3f} ± {sd:.3f} (1σ, {ns} seeds)",
                     loc="right", color=INK_2, fontsize=7)
        for i, r in enumerate(runs):
            st = seed_style(i, ns) if i < ns else ref_style(i - ns)
            kw = {"mfc": "white", "mew": 0.9} if i >= ns else {}
            errorbar(ax, [y[i]], stats[i], horizontal=True, fmt=st["marker"],
                     color=st["color"], markersize=5, **kw)
            v = stats[i]["point"][0]
            ax.annotate(f"{v:+.3f}" if signed else f"{v:.3f}", (stats[i]["hi"][0], y[i]),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7)
            if not signed and r["dense"] is not None:
                d = summarize(boot, r["dense"], "rmse", "all")["point"][0]
                ax.plot(d, y[i] - 0.25, ls="none", marker="D", markersize=3.5,
                        mfc="white", mec=INK, mew=0.8, zorder=3)
        lo = min(min(s["lo"][0] for s in stats), mean - sd)
        hi = max(max(s["hi"][0] for s in stats), mean + sd)
        if not signed:
            dense = [summarize(boot, r["dense"], "rmse", "all")["point"][0]
                     for r in runs if r["dense"] is not None]
            lo = min([lo, *dense])
        if signed:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + 2.5 * pad)
        ax.set_xlabel(xlabel)
        panel_letter(ax, letter)
    labels = [r["label"] for r in runs]
    axes[0].set_yticks(y, labels)
    handles = [plt.Line2D([], [], color=MUTED, marker="o", ls="-", lw=0.8,
                          markersize=4),
               plt.Line2D([], [], color=INK, lw=0.8),
               plt.Rectangle((0, 0), 1, 1, color=AXIS, alpha=0.35, lw=0)]
    names = ["Uniform truth (90% CI, test sample)", "Seed mean", "±1σ across seeds"]
    if any(r["dense"] is not None for r in runs):
        handles.insert(1, plt.Line2D([], [], ls="none", marker="D", markersize=3.5,
                                     mfc="white", mec=INK, mew=0.8))
        names.insert(1, "Dense truth RMSE")
    top_legend(fig, handles, names, ncol=len(names))
    return fig


def fig_band_spread(seeds: list, refs: list, boot: ObjectBootstrap) -> plt.Figure:
    """Per-band RMSE / bias per seed, and seed sigma vs test-sample sigma."""
    fig, axes = plt.subplots(4, 1, figsize=(FULL_W, 6.4), sharex=True,
                             height_ratios=(1.3, 1.1, 0.8, 0.8))
    x = np.arange(len(GROUP_NAMES))
    for ax, kind, ylabel in ((axes[0], "rmse", "Uniform RMSE (mag)"),
                             (axes[1], "bias", "Bias (mag)\n+ = too faint")):
        if kind == "bias":
            ax.axhline(0.0, color=AXIS, lw=0.6)
        seed_scatter(ax, points(seeds, boot, kind), points(refs, boot, kind))
        band_separators(ax)
        ax.set_ylabel(ylabel)
    for ax, kind, name in ((axes[2], "rmse", "RMSE"), (axes[3], "bias", "bias")):
        s_seed = seed_sd(points(seeds, boot, kind))
        s_test = test_sd(seeds, boot, kind)
        ax.bar(x - 0.19, s_seed, width=0.34, color=INK_2, lw=0)
        ax.bar(x + 0.19, s_test, width=0.34, color=AXIS, lw=0)
        band_separators(ax)
        ax.set_ylabel(f"σ of {name} (mag)")
    for ax, letter in zip(axes, "abcd"):
        panel_letter(ax, letter)
    band_axis(axes[-1])
    handles, labels = legend_handles(seeds, refs, lines=False)
    handles += [plt.Rectangle((0, 0), 1, 1, color=INK_2, lw=0),
                plt.Rectangle((0, 0), 1, 1, color=AXIS, lw=0)]
    labels += ["σ across seeds (c, d)", "σ test sample, one run (c, d)"]
    top_legend(fig, handles, labels, ncol=min(len(labels), 5))
    return fig


def fig_vs_lead(seeds: list, boot: ObjectBootstrap, edges: np.ndarray,
                kind: str) -> plt.Figure:
    """``kind`` (rmse / bias) vs lead time per band: thin line per seed + mean."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    grids = [lead_grid(boot, r["uniform"], kind, edges)["point"] for r in seeds]
    with warnings.catch_warnings():  # lead bins too thin in every seed stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack(grids), axis=0)
    n = len(seeds)
    fig, axes = lead_panels()
    for g, ax in enumerate(axes):
        if kind == "bias":
            ax.axhline(0.0, color=AXIS, lw=0.5)
        for i, grid in enumerate(grids):
            st = seed_style(i, n)
            ax.plot(centers, grid[g], color=st["color"], ls=st["ls"], lw=0.8)
        ax.plot(centers, mean[g], color=INK, lw=1.4)
    fig.supxlabel("Lead time (days)")
    fig.supylabel("Uniform RMSE (mag)" if kind == "rmse"
                  else "Uniform bias (mag; + = too faint)")
    handles, labels = legend_handles(seeds, [])
    for h in handles[:-1]:
        h.set_marker("none")
    top_legend(fig, handles, labels, ncol=min(len(labels), 6))
    return fig


def fig_coverage(seeds: list, refs: list, boot: ObjectBootstrap) -> plt.Figure | None:
    """Calibrated (else raw) central-band coverage per band, one marker per seed."""
    if not all(r["has_band"] for r in seeds):
        print("Not every seed has an interval; skipping coverage.")
        return None
    nominal = seeds[0]["interval"]

    def cov(runs: list) -> np.ndarray:
        return np.array([summarize(boot, r["uniform"],
                                   "hit_cal" if r["calibrated"] else "hit_raw")["point"]
                         for r in runs]).reshape(len(runs), len(GROUP_NAMES))

    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    ax.axhline(nominal, color=INK, lw=0.6, ls=":")
    seed_scatter(ax, cov(seeds), cov([r for r in refs if r["has_band"]]))
    band_separators(ax)
    band_axis(ax)
    raw = [r["label"] for r in seeds if not r["calibrated"]]
    note = f"  (raw band: {', '.join(raw)})" if raw else ""
    ax.set_ylabel(f"Coverage of calibrated\n{nominal:.0%} band (uniform truth)")
    ax.set_title(note, loc="right", color=INK_2, fontsize=7)
    handles, labels = legend_handles(seeds, [r for r in refs if r["has_band"]],
                                     lines=False)
    top_legend(fig, handles, labels, ncol=int(np.ceil(len(labels) / 2)))
    return fig


def fig_loss(seeds: list, refs: list) -> plt.Figure | None:
    """Validation loss (per-epoch median) per seed; evaluated checkpoint starred."""
    runs = seeds + refs
    if all(r["val_loss"] is None for r in runs):
        print("No validation records found; skipping loss.")
        return None
    n = len(seeds)
    last = max(r["val_loss"]["epoch"].max() for r in runs if r["val_loss"] is not None)
    zoom_from = int(last * 2 / 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.5))
    for i, r in enumerate(runs):
        v = r["val_loss"]
        if v is None:
            continue
        st = seed_style(i, n) if i < n else ref_style(i - n)
        late = v[v["epoch"] >= zoom_from]
        for ax, d in ((ax1, v), (ax2, late)):
            ax.plot(d["epoch"], d["median"], color=st["color"], ls=st["ls"],
                    lw=1.0 if i < n else 0.8)
        if r["epoch"] is not None:
            s = v.set_index("epoch")["median"]
            if r["epoch"] in s.index:
                for ax in (ax1, ax2):
                    ax.plot(r["epoch"], s[r["epoch"]], marker="*", markersize=8,
                            color=st["color"], mec="white", mew=0.6, ls="none",
                            zorder=4)
    ax1.set_yscale("log")
    ax1.axvspan(zoom_from, last, color=AXIS, alpha=0.25, lw=0)
    ax1.set_ylabel("Validation loss (per-epoch median)")
    ax2.set_title(f"epochs ≥ {zoom_from}", loc="right", color=INK_2, fontsize=7)
    for ax, letter in zip((ax1, ax2), "ab"):
        ax.set_xlabel("Epoch")
        panel_letter(ax, letter)
    handles, labels = legend_handles(seeds, refs, mean=False)
    for h in handles:
        h.set_marker("none")
    handles.append(plt.Line2D([], [], color=MUTED, marker="*", ls="none",
                              markersize=7))
    labels.append("Evaluated checkpoint")
    top_legend(fig, handles, labels, ncol=int(np.ceil(len(labels) / 2)))
    return fig


# ---------------------------------------------------------------------- tables


def write_tables(seeds: list, refs: list, boot: ObjectBootstrap,
                 outdir: str) -> None:
    """Per-run and per-band CSVs, the across-seed summary (CSV + LaTeX), printout."""
    runs = seeds + refs
    n = len(seeds)
    tables, rows = [], []
    for i, r in enumerate(runs):
        t = metrics_table(r, boot)
        t.insert(0, "study", r["study"])
        t.insert(1, "role", "seed" if i < n else "reference")
        t.insert(2, "seed", r["seed"])
        t.insert(3, "epoch", r["epoch"])
        tables.append(t)
        pooled = t.iloc[-1]
        rows.append({k: pooled[k] for k in
                     ("study", "role", "seed", "epoch", "uniform_rmse",
                      "uniform_rmse_lo", "uniform_rmse_hi", "uniform_bias")}
                    | {k: pooled.get(k, np.nan) for k in
                       ("beyond_bias", "dense_rmse", "cov_cal")})
    for name, df in (("seed_runs.csv", pd.DataFrame(rows)),
                     ("seed_bands.csv", pd.concat(tables))):
        path = os.path.join(outdir, name)
        df.to_csv(path, index=False, float_format="%.5f")
        print(f"Wrote {path}")

    metrics = [m for m in SUMMARY_METRICS if all(m in t for t in tables[:n])]
    stack = {m: np.stack([t[m].to_numpy(dtype=float) for t in tables[:n]])
             for m in metrics}
    tsd = {"uniform_rmse": test_sd(seeds, boot, "rmse"),
           "uniform_bias": test_sd(seeds, boot, "bias")}
    summary = []
    for g, group in enumerate(GROUP_NAMES):
        for m in metrics:
            v = stack[m][:, g]
            summary.append({"group": group, "metric": m, "n_seeds": n,
                            "mean": np.nanmean(v), "std": np.nanstd(v, ddof=1),
                            "sem": np.nanstd(v, ddof=1) / np.sqrt(n),
                            "min": np.nanmin(v), "max": np.nanmax(v),
                            "test_std": tsd[m][g] if m in tsd else np.nan})
    path = os.path.join(outdir, "seed_summary.csv")
    pd.DataFrame(summary).to_csv(path, index=False, float_format="%.5f")
    print(f"Wrote {path}")

    def pm(m: str, g: int, sign: bool = False) -> str:
        if m not in stack:
            return "--"
        v = stack[m][:, g]
        return f"${fmt(np.nanmean(v), sign=sign)} \\pm {fmt(np.nanstd(v, ddof=1))}$"

    cols = [("uniform_rmse", "Uniform RMSE", False), ("uniform_bias", "Bias", True),
            ("beyond_bias", "Beyond-depth bias", True),
            ("dense_rmse", "Dense RMSE", False), ("cov_cal", "Coverage", False)]
    cols = [c for c in cols if c[0] in stack]
    tex_rows = []
    for g, label in enumerate(GROUP_LABELS):
        if g == N_BANDS:
            tex_rows.append(None)
        tex_rows.append([label] + [pm(m, g, s) for m, _, s in cols])
    write_tex(os.path.join(outdir, "seed_summary.tex"),
              ["Band"] + [c[1] for c in cols], tex_rows,
              f"Late-time forecast error over {n} training seeds (mean $\\pm$ "
              "standard deviation across seeds; mag; residual = predicted "
              f"$-$ true, $+$ = too faint), on {boot.n_obj} common test objects.")

    s = stack["uniform_rmse"][:, -1]
    sd = float(np.std(s, ddof=1))
    t_all = float(tsd["uniform_rmse"][-1])
    print(f"\nAll bands, uniform truth, {boot.n_obj} common objects")
    print(f"  {'run':<30}{'epoch':>6}{'RMSE':>8}{'90% CI':>19}{'bias':>8}"
          f"{'dense':>8}{'cov':>7}")
    for r in rows:
        lab = next(x["label"] for x in runs if x["study"] == r["study"])
        print(f"  {lab[:29]:<30}{str(r['epoch']):>6}{r['uniform_rmse']:>8.4f}"
              f"   [{r['uniform_rmse_lo']:.4f}, {r['uniform_rmse_hi']:.4f}]"
              f"{r['uniform_bias']:>+8.3f}{r['dense_rmse']:>8.4f}{r['cov_cal']:>7.3f}")
    print(f"\n  seeds: mean {s.mean():.4f} ± {sd:.4f} (1σ), sem {sd / np.sqrt(n):.4f}, "
          f"range {s.min():.4f}–{s.max():.4f} ({s.max() - s.min():.4f})")
    print(f"  test-sample σ of one run {t_all:.4f} → seed σ / test σ = "
          f"{sd / t_all:.2f}")
    print("  (the test set is fixed, so between configs the SEED σ is the noise "
          "that matters)")
    for k in sorted({n, 5, 10}):
        print(f"  {k:>2} seeds per config resolve Δ ≥ {detectable_delta(sd, k):.4f} "
              "(80% power, two-sided α = 0.05)")
    print(f"  σ estimated from {n} seeds is itself uncertain by roughly "
          f"±{100 / np.sqrt(2 * (n - 1)):.0f}%")
    bsd = seed_sd(stack["uniform_bias"])[:N_BANDS]
    print("  per-band bias σ across seeds: " + ", ".join(
        f"{b} {v:.3f}" for b, v in zip(BAND_NAMES, bsd)))


# ------------------------------------------------------------------------ main


def load_available(studies: list, runs_root: str, labels: list) -> list:
    """``load_run`` each study, skipping (with a warning) any not yet evaluated."""
    out = []
    for s, lab in zip(studies, labels):
        try:
            out.append(load_run(s, runs_root, lab))
        except FileNotFoundError as err:
            print(f"WARNING: skipping study {s}: {err}")
    return out


def main() -> None:
    """Load the seed replicas' eval outputs and write the spread figures."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=int, default=None,
                   help="Config study; replicas are config*100 + seed.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--studies", type=int, nargs="+", default=None,
                   help="Replica study numbers (instead of --config/--seeds).")
    p.add_argument("--reference_studies", type=int, nargs="*", default=None,
                   help="Original runs drawn in gray (default: --config; pass "
                   "the flag with no studies for none).")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--outdir", default=None,
                   help="Figure dir (default runs/seeds_<config>).")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--boot_seed", type=int, default=0)
    args = p.parse_args()

    if args.studies is None:
        if args.config is None:
            p.error("give --config (with --seeds) or --studies")
        if any(not 0 <= s < 100 for s in args.seeds):
            p.error("--seeds must be in 0-99")
        studies = [args.config * 100 + s for s in args.seeds]
    else:
        studies = args.studies
    if len(set(studies)) != len(studies):
        p.error("duplicate replica studies")
    refs_in = (args.reference_studies if args.reference_studies is not None
               else [args.config] if args.config is not None else [])

    set_style()
    seeds = load_available(studies, args.runs_root,
                           [f"Seed {s % 100}" for s in studies])
    if len(seeds) < 2:
        raise SystemExit(f"Only {len(seeds)} replica(s) evaluated; need at least 2.")
    for r in seeds:
        r["seed"] = r["study"] % 100
        if r["epoch"] is not None:
            r["label"] += f" (ep {r['epoch']})"
    refs = load_available(refs_in, args.runs_root,
                          [f"Study {s} (original)" for s in refs_in])
    for r in refs:
        r["seed"] = np.nan

    runs = seeds + refs
    common = common_objects(runs)
    for r in runs:
        restrict_to(r, common)
    boot = ObjectBootstrap(list(common), args.n_boot, args.boot_seed)
    for r in runs:
        boot.attach(r)
    edges = lead_edges(seeds)
    configs = sorted({s // 100 for s in studies})
    outdir = args.outdir or os.path.join(
        args.runs_root, "seeds_" + "_".join(str(c) for c in configs))
    os.makedirs(outdir, exist_ok=True)
    fmts = tuple(args.formats)

    for name, fig in (
        ("overview", fig_overview(seeds, refs, boot)),
        ("band_spread", fig_band_spread(seeds, refs, boot)),
        ("rmse_vs_lead", fig_vs_lead(seeds, boot, edges, "rmse")),
        ("bias_vs_lead", fig_vs_lead(seeds, boot, edges, "bias")),
        ("coverage", fig_coverage(seeds, refs, boot)),
        ("loss", fig_loss(seeds, refs)),
    ):
        if fig is not None:
            save(fig, outdir, name, fmts)
    write_tables(seeds, refs, boot, outdir)


if __name__ == "__main__":
    main()
