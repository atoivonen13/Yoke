"""Publication figures comparing 2-8 KN LodeRunner runs (no model / GPU / torch).

Reads each study's eval outputs and loss records exactly as
``plot_run_report.py`` does (see its docstring for the files). Scores are
compared on the objects COMMON to every run, and every CI uses one shared set of
object-bootstrap resamples, so the run-vs-reference differences are PAIRED (the
same objects drawn in every resample) and much tighter than the per-run CIs.

Figures (PDF + PNG, 7.0 in two-column width) in ``--outdir``:

  1. ``overview``       (a) overall uniform RMSE per run (dense RMSE overlaid),
                        (b) paired difference vs the reference run, with the
                        seed-noise band shaded (``--seed_noise``);
  2. ``band_metrics``   uniform RMSE and bias per band, one marker per run;
  3. ``rmse_vs_lead``   uniform RMSE vs lead time per band, one line per run;
  4. ``bias_vs_lead``   the same for bias;
  5. ``coverage``       calibrated 90% coverage per band and vs lead;
  6. ``loss``           validation (and thin train) loss vs epoch per run;
  7. ``examples``       one object's forecast from every run vs the truth (the
                        reference run's median-RMSE object by default);

plus ``comparison.csv`` / ``comparison.tex`` (one row per run) and
``comparison_bands.csv`` (per band and run).

The CIs are test-sample noise only. Run-to-run seed noise (~0.05 overall
uniform RMSE for these models) is larger than most paired CIs, so read a
difference inside the shaded band as a possible seed effect.

    python plot_compare_runs.py --studies 209 208 211 205 --labels \
        "Fresh backbone" "Fine-tune decoder" "Fine-tune full" "Head only"
    # -> runs/compare_209_208_211_205/overview.pdf, ...
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_run_report import (  # noqa: E402
    AXIS,
    BAND_INDEX,
    BAND_LABELS,
    BAND_NAMES,
    FULL_W,
    GROUP_NAMES,
    HALF_W,
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
    fmt_ci,
    lead_edges,
    lead_grid,
    lead_line,
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

MAX_RUNS = len(SERIES)
SEED_NOISE = 0.05
# Hairline between band categories on the dodged per-band panels.
GRID_SEP = "#efeee9"


def common_objects(runs: list) -> set:
    """Objects scored (uniform truth) by every run; warns about any dropped."""
    sets = [set(r["uniform"]["stem"]) for r in runs]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("The runs share no scored objects (different splits?).")
    for r, s in zip(runs, sets):
        if len(s) != len(common):
            print(f"WARNING: {r['label']}: {len(s) - len(common)} of {len(s)} "
                  "objects are not scored by every run; comparing the "
                  f"{len(common)} common objects only.")
    return common


def paired_delta(stats: list, ref: int) -> list:
    """Per run: point, 90% CI of (run - reference) using the shared resamples."""
    out = []
    for s in stats:
        diff = s["boot"] - stats[ref]["boot"]
        out.append({"point": s["point"] - stats[ref]["point"],
                    "lo": np.nanpercentile(diff, 5, axis=0),
                    "hi": np.nanpercentile(diff, 95, axis=0)})
    return out


def legend_handles(runs: list, lines: bool = True) -> tuple:
    """Proxy artists so every figure's legend shows each run's full style."""
    handles = []
    for i in range(len(runs)):
        st = series_style(i)
        handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"],
                                  ls=st["ls"] if lines else "none", lw=1.2,
                                  markersize=4))
    return handles, [r["label"] for r in runs]


# --------------------------------------------------------------------- figures


def fig_overview(runs: list, boot: ObjectBootstrap, ref: int,
                 seed_noise: float) -> plt.Figure:
    """Overall uniform RMSE per run and the paired difference vs the reference."""
    rmse = [summarize(boot, r["uniform"], "rmse", "all") for r in runs]
    delta = paired_delta(rmse, ref)
    n = len(runs)
    y = np.arange(n)[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 0.42 * n + 1.0),
                                   sharey=True)
    for ax in (ax1, ax2):
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    for i, r in enumerate(runs):
        st = series_style(i)
        errorbar(ax1, [y[i]], rmse[i], horizontal=True, fmt=st["marker"],
                 color=st["color"], markersize=5)
        if r["dense"] is not None:
            d = summarize(boot, r["dense"], "rmse", "all")["point"][0]
            # offset below the row so it never hides the uniform marker
            ax1.plot(d, y[i] - 0.22, ls="none", marker="D", markersize=4, mfc="white",
                     mec=INK, mew=0.9, zorder=3,
                     label="Dense truth (within survey depth)" if i == 0 else None)
        ax1.annotate(f"{rmse[i]['point'][0]:.3f}", (rmse[i]["hi"][0], y[i]),
                     xytext=(4, 0), textcoords="offset points", va="center",
                     fontsize=7)
        if i != ref:
            errorbar(ax2, [y[i]], delta[i], horizontal=True, fmt=st["marker"],
                     color=st["color"], markersize=5)
            ax2.annotate(f"{delta[i]['point'][0]:+.3f}", (delta[i]["hi"][0], y[i]),
                         xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=7)
    ax1.set_yticks(y, [r["label"] for r in runs])
    ax1.set_xlabel("Uniform-truth RMSE, all bands (mag)")
    xmax = max(s["hi"][0] for s in rmse)
    ax1.set_xlim(0, xmax * 1.15)
    panel_letter(ax1, "a")
    if any(r["dense"] is not None for r in runs):
        handles = [plt.Line2D([], [], color=MUTED, marker="o", ls="-", lw=0.8,
                              markersize=4),
                   *ax1.get_legend_handles_labels()[0]]
        top_legend(fig, handles, ["Uniform truth (headline, 90% CI)",
                                  "Dense truth (within survey depth)"], ncol=2)

    if seed_noise > 0:
        ax2.axvspan(-seed_noise, seed_noise, color=AXIS, alpha=0.35, lw=0)
        ax2.annotate(f"seed noise ±{seed_noise:.2f}", (0, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(0, 1),
                     textcoords="offset points", ha="center", va="bottom",
                     color=INK_2, fontsize=7)
    ax2.axvline(0.0, color=INK, lw=0.6)
    ax2.plot([0], [y[ref]], ls="none", marker=series_style(ref)["marker"],
             color=series_style(ref)["color"], markersize=5)
    ax2.annotate("reference", (0, y[ref]), xytext=(5, 0), textcoords="offset points",
                 va="center", fontsize=7, color=INK_2)
    span = max([abs(d[k][0]) for d in delta for k in ("lo", "hi")] + [seed_noise])
    ax2.set_xlim(-1.3 * span, 1.3 * span)
    ax2.set_xlabel(f"Δ RMSE vs {runs[ref]['label']} (mag; − = better)")
    panel_letter(ax2, "b")
    return fig


def fig_band_metrics(runs: list, boot: ObjectBootstrap) -> plt.Figure:
    """Per-band uniform RMSE and bias, one dodged marker + CI per run."""
    n = len(runs)
    x = np.arange(len(GROUP_NAMES))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_W, 4.0), sharex=True,
                                   height_ratios=(1.3, 1))
    ax2.axhline(0.0, color=INK, lw=0.6)
    for i, r in enumerate(runs):
        st = series_style(i)
        kw = {"fmt": st["marker"], "color": st["color"], "markersize": 3.5}
        off = x + dodge(n, i, 0.72)
        errorbar(ax1, off, summarize(boot, r["uniform"], "rmse"), **kw)
        errorbar(ax2, off, summarize(boot, r["uniform"], "bias"), **kw)
    for gx in x[:-2] + 0.5:
        for ax in (ax1, ax2):
            ax.axvline(gx, color=GRID_SEP, lw=0.4, zorder=0)
    ax1.axvline(N_BANDS - 0.5, color=AXIS, lw=0.6)
    ax1.set_ylabel("Uniform RMSE (mag)")
    ax2.set_ylabel("Bias (mag)\n+ = too faint")
    band_axis(ax2)
    panel_letter(ax1, "a")
    panel_letter(ax2, "b")
    top_legend(fig, *legend_handles(runs, lines=False), ncol=min(n, 4))
    return fig


def fig_vs_lead(runs: list, boot: ObjectBootstrap, edges: np.ndarray, kind: str,
                ci: bool) -> plt.Figure:
    """``kind`` (rmse / bias) vs lead time per band, one line per run."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    grids = [lead_grid(boot, r["uniform"], kind, edges) for r in runs]
    fig, axes = lead_panels()
    for g, ax in enumerate(axes):
        if kind == "bias":
            ax.axhline(0.0, color=INK, lw=0.5)
        for i, grid in enumerate(grids):
            st = series_style(i)
            lead_line(ax, centers, grid, g, st["color"], ls=st["ls"],
                      marker=st["marker"], band=ci)
    fig.supxlabel("Lead time (days)")
    fig.supylabel("Uniform RMSE (mag)" if kind == "rmse"
                  else "Uniform bias (mag; + = too faint)")
    top_legend(fig, *legend_handles(runs), ncol=min(len(runs), 4))
    return fig


def fig_coverage(runs: list, boot: ObjectBootstrap,
                 edges: np.ndarray) -> plt.Figure | None:
    """Calibrated (else raw) central-band coverage per band and vs lead."""
    keep = [i for i, r in enumerate(runs) if r["has_band"]]
    if not keep:
        print("No run has an interval; skipping coverage.")
        return None
    nominal = runs[keep[0]]["interval"]
    centers = 0.5 * (edges[1:] + edges[:-1])
    x = np.arange(len(GROUP_NAMES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.7),
                                   width_ratios=(1.7, 1), sharey=True)
    for ax in (ax1, ax2):
        ax.axhline(nominal, color=INK, lw=0.6, ls=":")
    for j, i in enumerate(keep):
        r, st = runs[i], series_style(i)
        kind = "hit_cal" if r["calibrated"] else "hit_raw"
        errorbar(ax1, x + dodge(len(keep), j, 0.72), summarize(boot, r["uniform"],
                 kind), fmt=st["marker"], color=st["color"], markersize=3.5)
        lead_line(ax2, centers, lead_grid(boot, r["uniform"], kind, edges), N_BANDS,
                  st["color"], ls=st["ls"], marker=st["marker"], band=False)
    raw = [runs[i]["label"] for i in keep if not runs[i]["calibrated"]]
    note = f" (raw band: {', '.join(raw)})" if raw else ""
    band_axis(ax1)
    ax1.set_ylabel(f"Coverage of calibrated {nominal:.0%} band\n(uniform truth)")
    panel_letter(ax1, "a")
    ax2.set_xlabel("Lead time (days)")
    ax2.set_title("All bands" + note, loc="center")
    panel_letter(ax2, "b")
    top_legend(fig, *legend_handles([runs[i] for i in keep]), ncol=min(len(keep), 4))
    # keep each run's own style even when some runs were skipped
    for h, i in zip(fig.legends[0].legend_handles, keep):
        st = series_style(i)
        h.set_color(st["color"])
        h.set_marker(st["marker"])
        h.set_linestyle(st["ls"])
    return fig


def fig_loss(runs: list, train: bool) -> plt.Figure | None:
    """Validation (solid markers) and optional train (thin) loss vs epoch."""
    if all(r["val_loss"] is None and r["train_loss"] is None for r in runs):
        print("No loss records found; skipping loss.")
        return None
    fig, ax = plt.subplots(figsize=(HALF_W * 1.35, 2.5))
    for i, r in enumerate(runs):
        st = series_style(i)
        if train and r["train_loss"] is not None:
            t = r["train_loss"]
            ax.plot(t["epoch"], t["median"], color=st["color"], lw=0.6, alpha=0.45)
        if r["val_loss"] is not None:
            v = r["val_loss"]
            ax.plot(v["epoch"], v["median"], color=st["color"], ls=st["ls"],
                    marker=st["marker"], markevery=max(len(v) // 8, 1),
                    markersize=3, lw=1.2)
        if r["epoch"] is not None and r["val_loss"] is not None:
            v = r["val_loss"].set_index("epoch")["median"]
            if r["epoch"] in v.index:
                ax.plot(r["epoch"], v[r["epoch"]], marker="*", markersize=8,
                        color=st["color"], mec="white", mew=0.6, ls="none")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (per-epoch median)")
    handles, labels = legend_handles(runs)
    handles.append(plt.Line2D([], [], color=MUTED, marker="*", ls="none",
                              markersize=7))
    labels.append("Evaluated checkpoint")
    if train:
        handles.append(plt.Line2D([], [], color=MUTED, lw=0.6))
        labels.append("Train (thin)")
    ax.legend(handles, labels, loc="upper right")
    return fig


def fig_examples(runs: list, ref: int, bands: list, stem: str | None) -> plt.Figure:
    """One object's late-time forecast from every run vs the truth."""
    u_ref = runs[ref]["uniform"]
    if stem is None:
        per_obj = np.sqrt((u_ref["r"] ** 2).groupby(u_ref["stem"]).mean())
        stem = (per_obj - per_obj.median()).abs().idxmin()
    fig, axes = plt.subplots(1, len(bands), figsize=(FULL_W, 2.3), sharex=True,
                             squeeze=False)
    for j, name in enumerate(bands):
        ax = axes[0, j]
        b = BAND_INDEX[name]
        su = u_ref[(u_ref["stem"] == stem) & (u_ref["b"] == b)].sort_values("phase")
        ax.plot(su["phase"], su["true"], color=INK, lw=1.6, label="Uniform truth")
        d = runs[ref]["dense"]
        if d is not None:
            sd = d[(d["stem"] == stem) & (d["b"] == b)]
            ax.plot(sd["phase"], sd["true"], ls="none", marker="o", markersize=2.2,
                    color=MUTED, label="Dense truth")
        for i, r in enumerate(runs):
            st = series_style(i)
            u = r["uniform"]
            s = u[(u["stem"] == stem) & (u["b"] == b)].sort_values("phase")
            ax.plot(s["phase"], s["med"], color=st["color"], ls=st["ls"], lw=1.2,
                    label=r["label"])
        ax.invert_yaxis()
        ax.set_title(BAND_LABELS[b])
        if j == 0:
            ax.set_ylabel("Magnitude")
    fig.supxlabel(f"Days since first realistic detection   (object {stem})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=min(len(labels), 6))
    return fig


# ---------------------------------------------------------------------- tables


def write_tables(runs: list, boot: ObjectBootstrap, ref: int, outdir: str) -> None:
    """One-row-per-run comparison (CSV + LaTeX) and a per-band CSV."""
    rmse = [summarize(boot, r["uniform"], "rmse", "all") for r in runs]
    delta = paired_delta(rmse, ref)
    rows, tex_rows, band_rows = [], [], []
    has_cov = any(r["has_band"] for r in runs)
    for i, r in enumerate(runs):
        t = metrics_table(r, boot)
        t.insert(0, "label", r["label"])
        t.insert(1, "study", r["study"])
        band_rows.append(t)
        pooled = t.iloc[-1]
        row = {
            "label": r["label"], "study": r["study"], "epoch": r["epoch"],
            "uniform_rmse": rmse[i]["point"][0],
            "uniform_rmse_lo": rmse[i]["lo"][0], "uniform_rmse_hi": rmse[i]["hi"][0],
            "delta_vs_ref": delta[i]["point"][0],
            "delta_lo": delta[i]["lo"][0], "delta_hi": delta[i]["hi"][0],
            "uniform_bias": pooled["uniform_bias"],
            "beyond_bias": pooled["beyond_bias"],
            "dense_rmse": pooled.get("dense_rmse", np.nan),
            "cov_cal": pooled.get("cov_cal", np.nan),
            "worst_band": BAND_NAMES[int(np.nanargmax(t["uniform_rmse"][:N_BANDS]))],
        }
        rows.append(row)
        tex = [r["label"],
               fmt_ci(row["uniform_rmse"], row["uniform_rmse_lo"],
                      row["uniform_rmse_hi"]),
               "ref." if i == ref else fmt_ci(row["delta_vs_ref"], row["delta_lo"],
                                              row["delta_hi"], sign=True),
               fmt(row["dense_rmse"]), fmt(row["uniform_bias"], sign=True),
               fmt(row["beyond_bias"], sign=True)]
        if has_cov:
            tex.append(fmt(row["cov_cal"]))
        tex_rows.append(tex)

    table = pd.DataFrame(rows)
    path = os.path.join(outdir, "comparison.csv")
    table.to_csv(path, index=False, float_format="%.5f")
    print(f"Wrote {path}")
    path = os.path.join(outdir, "comparison_bands.csv")
    pd.concat(band_rows).to_csv(path, index=False, float_format="%.5f")
    print(f"Wrote {path}")
    header = ["Run", "Uniform RMSE", "$\\Delta$ vs ref.", "Dense RMSE", "Bias",
              "Beyond-depth bias"]
    if has_cov:
        header.append(f"Coverage ({runs[0]['interval']:.0%})")
    write_tex(os.path.join(outdir, "comparison.tex"), header, tex_rows,
              "Late-time forecast error, all bands pooled (mag; residual = "
              "predicted $-$ true, $+$ = too faint), on the "
              f"{boot.n_obj} test objects common to every run. Super/subscripts: "
              "object-bootstrap 90\\% CI; $\\Delta$ is paired (same resampled "
              f"objects). Seed-to-seed scatter is $\\sim${SEED_NOISE:.2f} mag.")

    print(f"\nAll bands, uniform truth, {boot.n_obj} common objects "
          f"(reference: {runs[ref]['label']})")
    print(f"  {'run':<28}{'epoch':>6}{'RMSE':>8}{'90% CI':>17}{'Δ ref':>9}"
          f"{'Δ 90% CI':>18}{'bias':>8}{'dense':>8}{'cov':>7}  worst")
    for r in rows:
        print(f"  {r['label'][:27]:<28}{str(r['epoch']):>6}{r['uniform_rmse']:>8.4f}"
              f"  [{r['uniform_rmse_lo']:.4f}, {r['uniform_rmse_hi']:.4f}]"
              f"{r['delta_vs_ref']:>+9.4f}  [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]"
              f"{r['uniform_bias']:>+8.3f}{r['dense_rmse']:>8.4f}{r['cov_cal']:>7.3f}"
              f"  {r['worst_band']}")
    print(f"  (CIs are test-sample noise only; seed noise ~{SEED_NOISE:.2f})")


def main() -> None:
    """Load several studies' eval outputs and write the comparison figures."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", type=int, nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None,
                   help="One name per study (default 'Study NNN').")
    p.add_argument("--reference", type=int, default=None,
                   help="Study the Δ panel and table are relative to "
                   "(default the first).")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--outdir", default=None,
                   help="Figure dir (default runs/compare_<studies>).")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--seed_noise", type=float, default=SEED_NOISE,
                   help="Half-width of the shaded seed-noise band on the Δ panel "
                   "(0 hides it).")
    p.add_argument("--lead_ci", action="store_true",
                   help="Shade CIs on the vs-lead figures (cluttered past 2 runs).")
    p.add_argument("--no_train_loss", action="store_true",
                   help="Validation loss only on the loss figure.")
    p.add_argument("--example_bands", nargs="+", default=["ztfg", "u", "g", "r"],
                   choices=BAND_NAMES)
    p.add_argument("--example_stem", default=None,
                   help="Object for the examples figure (default: the "
                   "reference run's median-RMSE object).")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not 2 <= len(args.studies) <= MAX_RUNS:
        p.error(f"--studies takes 2-{MAX_RUNS} study numbers")
    if len(set(args.studies)) != len(args.studies):
        p.error("--studies has duplicates")
    if args.labels is not None and len(args.labels) != len(args.studies):
        p.error("--labels needs one label per study")
    labels = args.labels or [None] * len(args.studies)
    ref_study = args.studies[0] if args.reference is None else args.reference
    if ref_study not in args.studies:
        p.error("--reference must be one of --studies")
    ref = args.studies.index(ref_study)

    set_style()
    runs = [load_run(s, args.runs_root, lab)
            for s, lab in zip(args.studies, labels)]
    common = common_objects(runs)
    for r in runs:
        restrict_to(r, common)
    boot = ObjectBootstrap(list(common), args.n_boot, args.seed)
    for r in runs:
        boot.attach(r)
    edges = lead_edges(runs)
    outdir = args.outdir or os.path.join(
        args.runs_root, "compare_" + "_".join(f"{s:03d}" for s in args.studies))
    os.makedirs(outdir, exist_ok=True)
    fmts = tuple(args.formats)

    for name, fig in (
        ("overview", fig_overview(runs, boot, ref, args.seed_noise)),
        ("band_metrics", fig_band_metrics(runs, boot)),
        ("rmse_vs_lead", fig_vs_lead(runs, boot, edges, "rmse", args.lead_ci)),
        ("bias_vs_lead", fig_vs_lead(runs, boot, edges, "bias", args.lead_ci)),
        ("coverage", fig_coverage(runs, boot, edges)),
        ("loss", fig_loss(runs, not args.no_train_loss)),
        ("examples", fig_examples(runs, ref, args.example_bands, args.example_stem)),
    ):
        if fig is not None:
            save(fig, outdir, name, fmts)
    write_tables(runs, boot, ref, outdir)


if __name__ == "__main__":
    main()
