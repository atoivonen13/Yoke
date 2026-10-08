"""Three-quantile reliability plot: empirical vs nominal rate at 0.05 / 0.5 / 0.95.

The quantile head predicts only tau = 0.05, 0.5 and 0.95, so these three rates
are the only calibration numbers measured directly from the model (the
continuous PP curve of ``plot_interval_coverage.py`` interpolates between them
with an assumed split-Gaussian shape). Per band, and for all bands pooled
(every point weighted equally), it plots

    F(tau) - tau,   F(tau) = fraction of points with truth <= predicted tau-quantile

with object-bootstrap CIs; 0 = calibrated. Magnitudes: larger = fainter, so
F(tau) > tau means the truth is brighter than predicted more often than nominal:

  * tau = 0.05: F is the miss rate off the bright end of the 90% interval,
  * tau = 0.95: 1 - F is the miss rate off the faint end (so the faint-side
    miss rate is 0.05 - (F - tau); < 0 on the plot = too many faint misses),
  * tau = 0.5:  F - 0.5 > 0 = the median forecast is too faint more often than
    too bright.

Reads the scored-points CSVs of ``eval_dense_latetime_9band.py`` and the
per-band calibration in ``<eval_dir>/interval_calibration.json`` (post hoc,
fit on the validation set; without it only the raw band is plotted).

    python plot_quantile_reliability.py --studies 138
    # -> runs/study_138/dense_latetime_eval_9band/interval_diagnostics/
    #    study138_reliability3_uniform.png / .pdf
    python plot_quantile_reliability.py --studies 138 --show_raw --no_title
    # overlay studies, one row per quantile; paired deltas vs the first
    python plot_quantile_reliability.py --studies 138 140 141
    # -> runs/interval_compare/study138_140_141_compare_reliability3_uniform.png

CIs are test-sample noise only (objects resampled); seed noise is not in them.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_interval_coverage import (  # noqa: E402
    BAND_NAMES,
    Q_HI,
    Q_LO,
    Q_MED,
    STUDY_COLORS,
    ObjectBootstrap,
    _band_masks,
    band_bounds,
    load_study,
)

TAUS = (Q_LO, Q_MED, Q_HI)
TAU_STYLE = {
    Q_LO: dict(marker="v", color="#2a78d6", label="0.05 (bright bound)"),
    Q_MED: dict(marker="o", color="k", label="0.5 (median)"),
    Q_HI: dict(marker="^", color="#d62828", label="0.95 (faint bound)"),
}
TAU_OFFSET = {Q_LO: -0.22, Q_MED: 0.0, Q_HI: 0.22}
GROUP_LABELS = {"all": "all\n(pooled)"}
YLABEL = "empirical - nominal\n(> 0: truth brighter than predicted)"


def rates(df, v, boot) -> dict:
    """``{(group, tau): (rate, per-resample rates, n)}`` for variant ``v``.

    The resample arrays keep NaNs (resamples without the group's objects) so
    they stay aligned across studies for paired deltas.
    """
    lo, hi = band_bounds(df, v)
    quant = {Q_LO: lo, Q_MED: df["med"].to_numpy(), Q_HI: hi}
    true = df["true"].to_numpy()
    out = {}
    for name, m in _band_masks(df):
        for tau in TAUS:
            r, bt = boot.boot_means(m, true <= quant[tau])
            out[(name, tau)] = (r, bt, int(m.sum()))
    return out


def _ci(bt, level) -> tuple:
    bt = bt[np.isfinite(bt)]
    tail = 50.0 - level / 2.0
    return float(np.percentile(bt, tail)), float(np.percentile(bt, 100.0 - tail))


def _groups(res) -> list:
    return [g for g in (*BAND_NAMES, "all") if (g, Q_MED) in res]


def _band_axis(ax, groups) -> None:
    """Band x-axis: zero line, ZTF | other separator, pooled group set apart."""
    ax.axhline(0.0, color="k", lw=0.8, zorder=1)
    ax.set_xticks(range(len(groups)), [GROUP_LABELS.get(g, g) for g in groups])
    n_ztf = sum(g.startswith("ztf") for g in groups)
    if 0 < n_ztf < len(groups) - 1:
        ax.axvline(n_ztf - 0.5, color="0.8", lw=0.8, ls=":", zorder=0)
    if groups[-1] == "all":
        ax.axvline(len(groups) - 1.5, color="0.5", lw=0.8, zorder=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)


def _point(ax, x, tau, r, bt, level, filled=True, **style) -> None:
    c_lo, c_hi = _ci(bt, level)
    ax.errorbar(x, r - tau, yerr=[[r - c_lo], [c_hi - r]], ms=5, capsize=2,
                lw=1.0, ls="none", mfc=style["color"] if filled else "white",
                zorder=3, **style)


def plot_single(results, label, truth, level, title, out_base, exts) -> None:
    """One axis; three markers per band (filled = calibrated, open = raw)."""
    final = list(results)[-1]
    groups = _groups(results[final])
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    if title:
        cal = "calibrated" if final == "cal" else "raw (no calibration)"
        fig.suptitle(f"{label}{truth} truth, {cal} quantiles; bars = "
                     f"object-bootstrap {level:g}% CI", fontsize=10)
    _band_axis(ax, groups)
    for x, g in enumerate(groups):
        for tau in TAUS:
            style = {k: s for k, s in TAU_STYLE[tau].items() if k != "label"}
            for v, res in results.items():
                # Raw (open) sits just right of the calibrated marker.
                shift = 0.08 if v != final else 0.0
                lab = TAU_STYLE[tau]["label"] if (x == 0 and v == final) else None
                _point(ax, x + TAU_OFFSET[tau] + shift, tau, *res[(g, tau)][:2],
                       level, filled=(v == final), label=lab, **style)
    if len(results) > 1:
        ax.plot([], [], "o", mfc="white", color="0.4", label="raw (open)")
    ax.set_ylabel(YLABEL)
    ax.set_xlabel("band")
    # Below the axes, so it never covers a point.
    fig.legend(fontsize=8, ncol=len(TAUS) + (len(results) > 1),
               loc="outside lower center", frameon=False)
    _save_all(fig, out_base, exts)


def plot_compare(entries, truth, level, title, out_base, exts) -> None:
    """One row per quantile; one marker per study per band."""
    groups = _groups(entries[0]["res"])
    fig, axes = plt.subplots(len(TAUS), 1, figsize=(7.2, 7.4), sharex=True,
                             layout="constrained")
    if title:
        fig.suptitle(f"{truth} truth, " + " / ".join(e["label"] for e in entries)
                     + f"; bars = object-bootstrap {level:g}% CI", fontsize=10)
    width = 0.6
    offs = (np.linspace(-width / 2, width / 2, len(entries)) if len(entries) > 1
            else np.zeros(1))
    for ax, tau in zip(axes, TAUS):
        _band_axis(ax, groups)
        for e, off in zip(entries, offs):
            for x, g in enumerate(groups):
                if (g, tau) not in e["res"]:
                    continue
                _point(ax, x + off, tau, *e["res"][(g, tau)][:2], level,
                       marker=TAU_STYLE[tau]["marker"], color=e["color"],
                       label=e["label"] if x == 0 else None)
        ax.set_ylabel(f"F({tau:g}) - {tau:g}")
        ax.set_title(f"tau = {TAU_STYLE[tau]['label']}", fontsize=9, loc="left")
    axes[0].set_title("> 0: truth brighter than predicted", fontsize=8,
                      loc="right", color="0.35")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8,
               ncol=len(entries), loc="outside lower center", frameon=False)
    axes[-1].set_xlabel("band")
    _save_all(fig, out_base, exts)


def _save_all(fig, out_base, exts) -> None:
    for ext in exts:
        fig.savefig(f"{out_base}.{ext}", dpi=200)
        print(f"Wrote {out_base}.{ext}")
    plt.close(fig)


def print_single(results, label, truth, level) -> None:
    """F(tau) with CI per band, and the two one-sided miss rates."""
    for v, res in results.items():
        print(f"\n[{label}{truth}, {v}] F(tau) = fraction truth <= predicted "
              f"quantile, object-bootstrap {level:g}% CI; misses: bright = "
              "F(.05), faint = 1 - F(.95) (each nominal 0.05)")
        print(f"  {'band':<6}{'n':>8}" + "".join(f"{f'F({t:g})':>22}" for t in TAUS)
              + f"{'bright':>8}{'faint':>8}{'cov':>7}")
        for g in _groups(res):
            line = f"  {g:<6}{res[(g, Q_MED)][2]:>8d}"
            for tau in TAUS:
                r, bt, _ = res[(g, tau)]
                c_lo, c_hi = _ci(bt, level)
                line += f"{f'{r:.3f} [{c_lo:.3f}, {c_hi:.3f}]':>22}"
            f_lo, f_hi = res[(g, Q_LO)][0], res[(g, Q_HI)][0]
            print(line + f"{f_lo:>8.3f}{1.0 - f_hi:>8.3f}{f_hi - f_lo:>7.3f}")


def print_compare(entries, truth, level) -> None:
    """Per quantile: F(tau) per study, paired-bootstrap delta vs the first."""
    base = entries[0]
    for tau in TAUS:
        print(f"\n[{truth}] F({tau:g}) (nominal {tau:g}); delta = study - "
              f"{base['study']}, paired object-bootstrap {level:g}% CI")
        print(f"  {'band':<6}{base['study']:>8}" + "".join(
            f"{e['study']:>8}{'delta':>9}{'CI':>18}" for e in entries[1:]))
        for g in _groups(base["res"]):
            r0, b0, _ = base["res"][(g, tau)]
            line = f"  {g:<6}{r0:>8.3f}"
            for e in entries[1:]:
                if (g, tau) not in e["res"]:
                    line += f"{'-':>8}{'':>27}"
                    continue
                r, bt, _ = e["res"][(g, tau)]
                c_lo, c_hi = _ci(bt - b0, level)
                line += f"{r:>8.3f}{r - r0:>+9.3f}" + \
                    f"{f'[{c_lo:+.3f}, {c_hi:+.3f}]':>18}"
            print(line)


def main() -> None:
    """Load the eval CSVs and write the three-quantile reliability plot(s)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", type=int, nargs="+", default=None,
                   help="One study -> single plot; 2-"
                   f"{len(STUDY_COLORS)} -> one row per quantile, studies "
                   "overlaid (first = delta baseline).")
    p.add_argument("--eval_dir", default=None,
                   help="A single eval dir instead of --studies.")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--truth", choices=("uniform", "dense", "both"),
                   default="uniform")
    p.add_argument("--show_raw", action="store_true",
                   help="Single plot: also the uncalibrated quantiles (open).")
    p.add_argument("--ci", type=float, default=90.0,
                   help="Bootstrap CI level (%%).")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ext", nargs="+", default=("png", "pdf"))
    p.add_argument("--no_title", action="store_true",
                   help="Drop the figure title (for a paper caption).")
    p.add_argument("--outdir", default=None,
                   help="Default <eval_dir>/interval_diagnostics; "
                   "<runs_dir>/interval_compare with several studies.")
    args = p.parse_args()
    truths = [t for t in ("uniform", "dense") if args.truth in (t, "both")]

    if args.eval_dir is not None:
        dirs, labels = [args.eval_dir], [""]
    elif args.studies:
        if len(args.studies) > len(STUDY_COLORS):
            p.error(f"--studies takes at most {len(STUDY_COLORS)} studies")
        dirs = [os.path.join(args.runs_dir, f"study_{s:03d}",
                             "dense_latetime_eval_9band") for s in args.studies]
        labels = [f"{s:03d}" for s in args.studies]
    else:
        p.error("give --studies or --eval_dir")
    data = [load_study(d) for d in dirs]

    if len(dirs) == 1:
        outdir = args.outdir or os.path.join(dirs[0], "interval_diagnostics")
        prefix = f"study{labels[0]}_" if labels[0] else ""
        tag = f"study {labels[0]}: " if labels[0] else ""
    else:
        outdir = args.outdir or os.path.join(args.runs_dir, "interval_compare")
        prefix = "study" + "_".join(labels) + "_compare_"
    os.makedirs(outdir, exist_ok=True)

    for truth in truths:
        base = os.path.join(outdir, f"{prefix}reliability3_{truth}")
        if len(dirs) == 1:
            df = data[0][truth]
            if df is None:
                print(f"No {truth} scored points; skipping.")
                continue
            boot = ObjectBootstrap(df["stem"].to_numpy(), args.n_boot, args.seed)
            variants = data[0]["variants"] if args.show_raw else \
                [data[0]["variant"]]
            results = {v: rates(df, v, boot) for v in variants}
            plot_single(results, tag, truth, args.ci, not args.no_title, base,
                        args.ext)
            print_single(results, tag, truth, args.ci)
            continue
        items = [(lab, d[truth], d["variant"]) for lab, d in zip(labels, data)
                 if d[truth] is not None]
        if len(items) < 2:
            print(f"[{truth}] fewer than 2 studies have this truth set; skipping.")
            continue
        # Shared object universe + seed -> identical bootstrap weights per
        # study, so the deltas are paired (the test objects are shared).
        universe = np.unique(np.concatenate([df["stem"].to_numpy()
                                             for _, df, _ in items]))
        entries = []
        for i, (lab, df, v) in enumerate(items):
            boot = ObjectBootstrap(df["stem"].to_numpy(), args.n_boot, args.seed,
                                   universe)
            entries.append({"study": lab, "label": f"{lab} ({v})",
                            "color": STUDY_COLORS[i], "res": rates(df, v, boot)})
        plot_compare(entries, truth, args.ci, not args.no_title, base, args.ext)
        print_compare(entries, truth, args.ci)


if __name__ == "__main__":
    main()
