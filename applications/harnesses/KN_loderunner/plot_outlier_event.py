"""Appendix figure: the forecast of one object in all nine bands.

Made for the worst object of study 218 (lc_01679, ~5 mag too bright in every
band; see ``find_outliers.py``), but takes any scored stem. Draws a 3 x 3
figure (``event_<stem>``) and a one-panel overlay (``event_<stem>_overlay``) of
the bands with a realistic detection (color = filter; forecast median solid,
truth faint and dash-dot / squares = ZTF, dashed / circles = LSST; no interval
bands). Per band:

  * the uniform (true) curve from 1 d before the first realistic detection;
  * the realistic context the model saw (filled) and the other realistic
    detections it did not see (open: after the cutoff, or dropped by the
    trailing window / cap);
  * the forecast median, and in the 3 x 3 figure the calibrated 90% band, over
    the scored window.

Also prints each context detection next to the true curve at its time, i.e.
whether the input already disagrees with the truth, and the realistic upper
limits between the last context detection and the cutoff (known at forecast
time, but the eval and training discard them; not drawn).

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
    NPZ_BAND_KEYS,
    NPZ_ERROR_COL,
    NPZ_VALUE_COL,
    _find_npz,
    add_style_args,
    apply_style,
    color,
    figsize,
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
# Overlay: one color per filter (ZTF and LSST g, r, i share it), muted and
# ordered blue -> red with wavelength (u purple at the blue end); the survey is
# the truth's dash pattern and the marker.
FILTER_COLOR = {"u": "#8172b3", "g": "#4c72b0", "r": "#55a868", "i": "#ccb974",
                "z": "#dd8452", "y": "#c44e52"}
SURVEY_LOOK = {"ztf": ((0, (4.0, 1.4, 1.0, 1.4)), "s"), "lsst": ((0, (4.0, 1.6)), "o")}


def other_detections(path: str, t0: float, ctx: dict, x_lim: tuple) -> tuple:
    """Realistic detections in ``x_lim`` that are not in the context."""
    t, v, b = read_npz_stream(path, True)
    phase = t - t0
    keep = (phase >= x_lim[0]) & (phase <= x_lim[1])
    in_ctx = {(round(float(p), 6), int(c)) for p, c in zip(ctx["phase"], ctx["b"])}
    keep &= np.array([(round(float(p), 6), int(c)) not in in_ctx
                      for p, c in zip(phase, b)], dtype=bool)
    return phase[keep], v[keep], b[keep]


def upper_limits(path: str, t0: float, x_lim: tuple) -> tuple:
    """Realistic non-detections (limiting mag) in ``x_lim``; the eval drops them.

    Same file layout as ``read_npz_stream``: a non-finite error marks an upper
    limit, and the value column then holds the limiting magnitude.
    """
    t, v, b = [], [], []
    with np.load(path, allow_pickle=True) as data:
        names = set(data.files)
        keys = next((k for k in NPZ_BAND_KEYS if names & set(k[3:])), NPZ_BAND_KEYS[0])
        for band, key in enumerate(keys):
            if key not in names or np.asarray(data[key]).size == 0:
                continue
            arr = np.asarray(data[key])
            arr = arr[~np.isfinite(arr[:, NPZ_ERROR_COL])]
            t.append(arr[:, 0].astype(np.float64) - t0)
            v.append(arr[:, NPZ_VALUE_COL].astype(np.float64))
            b.append(np.full(len(arr), band, dtype=np.int64))
    if not t:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
    phase, v, b = np.concatenate(t), np.concatenate(v), np.concatenate(b)
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


def y_limits(*series: np.ndarray) -> tuple | None:
    """Faint-at-the-bottom limits over every drawn series."""
    vals = np.concatenate([s for s in series if s.size])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(0.06 * (hi - lo), 0.3)
    return hi + pad, lo - pad


def fig_event(run: dict, obj: dict, other: tuple, title: str | None) -> plt.Figure:
    """3 x 3 panels: truth, context, other detections, forecast and its band."""
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
        lim = y_limits(t_mag[m], ctx["mag"][cm], other[1][om], f["med"].to_numpy())
        if lim is not None:
            ax.set_ylim(lim)
    sup_labels(fig, "Days since first realistic detection", "Magnitude")
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    top_legend(fig, list(handles.values()), list(handles),
               ncol=min(3, len(handles)), title=title)
    return fig


def band_look(b: int) -> tuple:
    """(color, line style, marker) of band ``b`` in the overlay."""
    name = BAND_NAMES[b]
    ls, marker = SURVEY_LOOK["ztf" if name.startswith("ztf") else "lsst"]
    return FILTER_COLOR[name[-1]], ls, marker


def overlay_bands(ctx: dict, other: tuple) -> list:
    """Bands with at least one realistic detection in the plotted window."""
    return sorted(set(ctx["b"].tolist()) | set(other[2].tolist()))


def fig_event_overlay(run: dict, obj: dict, other: tuple,
                      title: str | None) -> plt.Figure:
    """The observed bands in one panel: color = filter; forecast median solid,
    truth faint and dashed (the dash pattern and marker = survey)."""
    fig, ax = plt.subplots(figsize=figsize("full_w", 3.8))
    ctx, (t_ph, t_mag, t_b), fc = obj["ctx"], obj["truth"], obj["forecast"]
    ax.axvline(obj["cutoff"], color=LOOK["muted"], ls=":", lw=LOOK["lw_thin"])
    faint = 0.45  # truth line alpha; the forecast is opaque
    drawn, band_handles, any_other = [], [], False
    for b in overlay_bands(ctx, other):
        c, ls, mk = band_look(b)
        f = fc[fc["b"] == b].sort_values("phase")
        ax.plot(f["phase"], f["med"], color=c, lw=1.3 * LOOK["lw"], zorder=3)
        m = t_b == b
        ax.plot(t_ph[m], t_mag[m], color=c, ls=ls, lw=LOOK["lw"], alpha=faint,
                zorder=2)
        om = other[2] == b
        any_other |= bool(om.any())
        ax.plot(other[0][om], other[1][om], ls="none", marker=mk,
                markersize=LOOK["ms"], mfc="none", mec=c, mew=LOOK["mew"], zorder=4)
        cm = ctx["b"] == b
        ax.plot(ctx["phase"][cm], ctx["mag"][cm], ls="none", marker=mk,
                markersize=LOOK["ms"], mfc=c, mec=LOOK["ink"],
                mew=0.5 * LOOK["mew"], zorder=5)
        drawn += [f["med"].to_numpy(), t_mag[m], other[1][om], ctx["mag"][cm]]
        band_handles.append(plt.Line2D([], [], color=c, ls=ls, lw=LOOK["lw"],
                                       marker=mk, markersize=LOOK["ms"], mfc=c,
                                       mec=LOOK["ink"], mew=0.5 * LOOK["mew"],
                                       label=BAND_LABELS[b]))
    ax.set_xlim(obj["x_lim"])
    lim = y_limits(*drawn)
    if lim is not None:
        ax.set_ylim(lim)
    ax.set_xlabel("Days since first realistic detection")
    ax.set_ylabel("Magnitude")
    note(ax, f"{len(ctx['phase'])} in context, RMSE {np.sqrt(np.mean(fc['r'] ** 2)):.2f} "
         "(all 9 bands)", (0.02, 0.03), xycoords="axes fraction", ha="left", va="bottom",
         fontsize=LOOK["tiny"], color=LOOK["muted"],
         bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 0.5})
    ax.legend(handles=band_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              handlelength=2.6)

    # What each mark means, drawn in gray (the color legend is beside the axes).
    gray, kinds = LOOK["ink2"], []
    kinds.append(plt.Line2D([], [], color=LOOK["muted"], ls=":", lw=LOOK["lw_thin"],
                            label=f"Context cutoff ({obj['cutoff']:g} d)"))
    kinds.append(plt.Line2D([], [], color=gray, lw=1.3 * LOOK["lw"],
                            label="Forecast median"))
    kinds.append(plt.Line2D([], [], color=gray, ls=SURVEY_LOOK["lsst"][0], lw=LOOK["lw"],
                            alpha=faint, label="Uniform truth"))
    kinds.append(plt.Line2D([], [], ls="none", marker="o", markersize=LOOK["ms"],
                            mfc=gray, mec=LOOK["ink"], mew=0.5 * LOOK["mew"],
                            label="Realistic context"))
    if any_other:
        kinds.append(plt.Line2D([], [], ls="none", marker="o", markersize=LOOK["ms"],
                                mfc="none", mec=gray, mew=LOOK["mew"],
                                label="Realistic, not in context"))
    top_legend(fig, kinds, [h.get_label() for h in kinds], ncol=min(3, len(kinds)),
               title=title)
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
    p.add_argument("--uniform_glob", default=None)
    p.add_argument("--layout", choices=("grid", "overlay", "both"), default="both",
                   help="3 x 3 panels (event_<stem>), all bands in one panel "
                        "(event_<stem>_overlay), or both.")
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
    real_path = _find_npz(sources["realistic_glob"], args.stem)
    other = other_detections(real_path, ctx["t0"], ctx, obj["x_lim"])
    uls = upper_limits(real_path, ctx["t0"], obj["x_lim"])

    per_obj = per_object_rmse(run)
    rank = float((per_obj < per_obj[args.stem]).mean() * 100)
    fc = obj["forecast"]
    print(f"\n{args.stem}: uniform RMSE {per_obj[args.stem]:.3f} ({rank:.1f}th pct of "
          f"{len(per_obj)}), bias {fc['r'].mean():+.3f}; {len(ctx['phase'])} context "
          f"detections of {ctx['n_pre_cutoff']} before the {obj['cutoff']:g} d "
          f"cutoff, bands {sorted({BAND_NAMES[b] for b in ctx['b']})}")
    print_context_vs_truth(ctx, obj["truth"])
    gap = (uls[0] > ctx["phase"][-1]) & (uls[0] <= obj["cutoff"])
    print(f"\n  realistic upper limits: {len(uls[0])} in the plotted window, "
          f"{int(gap.sum())} between the last context detection "
          f"({ctx['phase'][-1]:.2f} d) and the {obj['cutoff']:g} d cutoff "
          "(known at forecast time, not given to the model)")
    for p_, v_, b_ in zip(uls[0][gap], uls[1][gap], uls[2][gap]):
        print(f"  {BAND_NAMES[b_]:>5} {p_:>10.3f}  fainter than {v_:.2f}")

    outdir = args.outdir or os.path.join(run["run_dir"], "figures")
    os.makedirs(outdir, exist_ok=True)
    if args.layout in ("grid", "both"):
        save(fig_event(run, obj, other, args.title), outdir,
             f"event_{args.stem}", tuple(args.formats))
    if args.layout in ("overlay", "both"):
        shown = overlay_bands(ctx, other)
        print(f"\n  overlay bands (with a realistic detection): "
              f"{[BAND_NAMES[b] for b in shown]}; left out: "
              f"{[BAND_NAMES[b] for b in range(N_BANDS) if b not in shown]}")
        save(fig_event_overlay(run, obj, other, args.title),
             outdir, f"event_{args.stem}_overlay", tuple(args.formats))


if __name__ == "__main__":
    main()
