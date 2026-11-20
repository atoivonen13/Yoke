"""Appendix figure: the forecast of one object in all nine bands.

Made for the worst object of study 218 (lc_01679, ~5 mag too bright in every
band; see ``find_outliers.py``), but takes any scored stem. One 3 x 3 figure,
per band:

  * the uniform (true) curve from 1 d before the first realistic detection;
  * the dense detections of the same object (where it would be detectable);
  * the realistic context the model saw (filled) and the other realistic
    detections it did not see (open: after the cutoff, or dropped by the
    trailing window / cap);
  * the forecast median and calibrated 90% band over the scored window.

Also prints each context detection next to the true curve at its time, i.e.
whether the input already disagrees with the truth.

Reads the eval outputs and npz sets like ``plot_run_report.py`` (no torch).

    python plot_outlier_event.py --study 218
    python plot_outlier_event.py --study 218 --stem lc_03544 --title "Study 218"
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from plot_run_report import (
    BAND_LABELS,
    BAND_NAMES,
    DEFAULT_REALISTIC_GLOB,
    DEFAULT_UNIFORM_GLOB,
    LOOK,
    N_BANDS,
    _find_npz,
    add_style_args,
    apply_style,
    color,
    load_run,
    note,
    object_context,
    per_object_rmse,
    read_npz_stream,
    save,
    sup_labels,
    top_legend,
)

DEFAULT_STEM = "lc_01679"


def other_detections(path: str, t0: float, ctx: dict, x_lim: tuple) -> tuple:
    """Realistic detections in ``x_lim`` that are not in the context."""
    t, v, b = read_npz_stream(path, True)
    phase = t - t0
    keep = (phase >= x_lim[0]) & (phase <= x_lim[1])
    in_ctx = {(round(float(p), 6), int(c)) for p, c in zip(ctx["phase"], ctx["b"])}
    keep &= np.array([(round(float(p), 6), int(c)) not in in_ctx
                      for p, c in zip(phase, b)], dtype=bool)
    return phase[keep], v[keep], b[keep]


def dense_detections(path: str | None, t0: float, x_lim: tuple) -> tuple | None:
    """Dense detections in ``x_lim`` on the realistic clock (None if no file)."""
    if path is None:
        return None
    t, v, b = read_npz_stream(path, True)
    phase = t - t0
    keep = (phase >= x_lim[0]) & (phase <= x_lim[1]) & np.isfinite(v)
    return phase[keep], v[keep], b[keep]


def print_context_vs_truth(ctx: dict, truth: tuple) -> None:
    """Each context detection minus the true curve at its time (+ = fainter)."""
    t_ph, t_mag, t_b = truth
    print("\n  context detection vs uniform truth at the same time:")
    print(f"  {'band':>5} {'phase (d)':>10} {'mag':>8} {'truth':>8} {'det - true':>11}")
    for p, m, b in zip(ctx["phase"], ctx["mag"], ctx["b"]):
        sel = t_b == b
        if sel.sum() >= 2:
            order = np.argsort(t_ph[sel])
            tru = float(np.interp(p, t_ph[sel][order], t_mag[sel][order]))
            print(f"  {BAND_NAMES[b]:>5} {p:>10.3f} {m:>8.2f} {tru:>8.2f} {m - tru:>+11.2f}")
        else:
            print(f"  {BAND_NAMES[b]:>5} {p:>10.3f} {m:>8.2f} {'--':>8} {'--':>11}")


def y_limits(b: int, *series: np.ndarray) -> tuple | None:
    """Faint-at-the-bottom limits over every drawn series of band ``b``."""
    vals = np.concatenate([s for s in series if s.size])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(0.06 * (hi - lo), 0.3)
    return hi + pad, lo - pad


def fig_event(run: dict, stem: str, obj: dict, other: tuple, dense: tuple | None,
              title: str | None) -> plt.Figure:
    """3 x 3 panels: truth, dense, context, other detections, forecast."""
    fig, axes = plt.subplots(3, 3, figsize=(LOOK["full_w"],
                                            LOOK["full_w"] * 0.78 * LOOK["h"]),
                             sharex=True)
    ctx, (t_ph, t_mag, t_b), fc = obj["ctx"], obj["truth"], obj["forecast"]
    band_label = "Calibrated" if run["calibrated"] else "Raw"
    for b, ax in zip(range(N_BANDS), axes.ravel()):
        ax.set_title(BAND_LABELS[b])
        ax.axvline(obj["cutoff"], color=LOOK["muted"], ls=":", lw=LOOK["lw_thin"],
                   label=f"Context cutoff ({obj['cutoff']:g} d)")
        f = fc[fc["b"] == b].sort_values("phase")
        if run["has_band"]:
            ax.fill_between(f["phase"], f["low_cal"], f["high_cal"], color=color(0),
                            alpha=0.2, lw=0,
                            label=f"{band_label} {run['interval']:.0%} band")
        m = t_b == b
        ax.plot(t_ph[m], t_mag[m], color=LOOK["ink2"], lw=LOOK["lw_thin"],
                label="Uniform truth")
        d_mag = np.empty(0)
        if dense is not None:
            dm = dense[2] == b
            d_mag = dense[1][dm]
            ax.plot(dense[0][dm], d_mag, ls="none", marker=".",
                    markersize=LOOK["ms_small"], color=LOOK["muted"], zorder=2,
                    label="Dense detections")
        ax.plot(f["phase"], f["med"], color=color(0), lw=LOOK["lw"],
                label="Forecast median")
        om = other[2] == b
        ax.plot(other[0][om], other[1][om], ls="none", marker="o",
                markersize=LOOK["ms"], mfc="none", mec=color(1),
                mew=LOOK["mew"], zorder=3, label="Realistic, not in context")
        cm = ctx["b"] == b
        ax.plot(ctx["phase"][cm], ctx["mag"][cm], ls="none", marker="o",
                markersize=LOOK["ms"], mfc=color(1), mec=LOOK["ink"],
                mew=0.5 * LOOK["mew"], zorder=4, label="Realistic context")
        text = f"{int(cm.sum())} in context"
        if len(f):
            text += f", RMSE {np.sqrt(np.mean(f['r'] ** 2)):.2f}"
        note(ax, text, (0.02, 0.03), xycoords="axes fraction", ha="left",
             va="bottom", fontsize=LOOK["tiny"], color=LOOK["muted"],
             bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 0.5})
        ax.set_xlim(obj["x_lim"])
        lim = y_limits(b, t_mag[m], d_mag, ctx["mag"][cm], other[1][om],
                       f["med"].to_numpy())
        if lim is not None:
            ax.set_ylim(lim)
    sup_labels(fig, "Days since first realistic detection", "Magnitude")
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    top_legend(fig, list(handles.values()), list(handles),
               ncol=-(-len(handles) // 2), title=title)
    return fig


def main() -> None:
    """Draw one object's nine-band forecast for the appendix."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=int, default=218)
    p.add_argument("--stem", default=DEFAULT_STEM)
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--eval_dir", default=None)
    p.add_argument("--outdir", default=None,
                   help="Figure dir (default runs/study_NNN/figures).")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--title", default=None,
                   help="Text above the legend (default none: the caption names it).")
    p.add_argument("--realistic_glob", default=None)
    p.add_argument("--dense_glob", default=None,
                   help="Dense npz set (default: the eval's; omitted if unknown).")
    p.add_argument("--uniform_glob", default=None)
    p.add_argument("--context_window_days", type=float, default=2.0)
    p.add_argument("--max_context_len", type=int, default=12)
    add_style_args(p)
    args = p.parse_args()

    apply_style(args.style, args.font_dir)
    run = load_run(args.study, args.runs_root, eval_dir=args.eval_dir)
    sig = run["signature"]
    sources = {
        "realistic_glob": (args.realistic_glob or sig.get("realistic_glob")
                           or DEFAULT_REALISTIC_GLOB),
        "uniform_glob": (args.uniform_glob or sig.get("uniform_glob")
                         or DEFAULT_UNIFORM_GLOB),
        "window_days": args.context_window_days,
        "max_context_len": args.max_context_len,
    }
    if args.stem not in set(run["uniform"]["stem"]):
        raise SystemExit(f"{args.stem} is not in study {args.study}'s scored set")
    obj = object_context(run, args.stem, sources, "cannot draw it")
    if obj is None:
        raise SystemExit(1)
    ctx = obj["ctx"]
    other = other_detections(_find_npz(sources["realistic_glob"], args.stem),
                             ctx["t0"], ctx, obj["x_lim"])
    dense_glob = args.dense_glob or sig.get("dense_glob")
    dense = dense_detections(dense_glob and _find_npz(dense_glob, args.stem),
                             ctx["t0"], obj["x_lim"])
    if dense is None:
        print("no dense npz found; drawing without dense detections")

    per_obj = per_object_rmse(run)
    rank = float((per_obj < per_obj[args.stem]).mean() * 100)
    fc = obj["forecast"]
    print(f"\n{args.stem}: uniform RMSE {per_obj[args.stem]:.3f} ({rank:.1f}th pct of "
          f"{len(per_obj)}), bias {fc['r'].mean():+.3f}; {len(ctx['phase'])} context "
          f"detections of {ctx['n_pre_cutoff']} before the {obj['cutoff']:g} d "
          f"cutoff, bands {sorted({BAND_NAMES[b] for b in ctx['b']})}")
    print_context_vs_truth(ctx, obj["truth"])

    outdir = args.outdir or os.path.join(run["run_dir"], "figures")
    os.makedirs(outdir, exist_ok=True)
    save(fig_event(run, args.stem, obj, other, dense, args.title), outdir,
         f"event_{args.stem}", tuple(args.formats))


if __name__ == "__main__":
    main()
