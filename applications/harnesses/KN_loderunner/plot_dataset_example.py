"""Paper figures: one kilonova in the realistic, dense and uniform datasets.

Two figures of the same event in the publication style (``publication.mplstyle``):

``<name>``: a 3 x 2 grid, rows the three datasets, columns the ZTF (g r i) and
LSST (u g r i z y) filters. One hue per filter letter, shared by the surveys.

  (a, b) realistic: survey-cadence detections (+-1 sigma) and upper limits;
  (c, d) dense:     the same event sampled densely in time;
  (e, f) uniform:   the light curve on a regular time grid (the scored truth).

``<name>_bands``: a 3 x 3 grid, one panel per band (ZTF g r i, then LSST u g r i
z y), the three datasets overlaid in each, one hue per dataset. Each panel fits
its own magnitude range unless ``--bands_mag_range`` is given.

Phase is from the first realistic detection (the eval's t0); the gray span is
the context window (phase <= ``--late_time_cutoff_days``) the model sees, the
forecast is scored on cutoff < phase <= ``--late_time_max_days``.

Without ``--stem`` the object is picked by ``--pick``:

  random          (default) a different random test object every run, among the
                  scorable ones with realistic detections in both surveys; the
                  stem and seed are printed (rerun with ``--stem`` or ``--seed``);
  characteristic  among the first ``--max_scan`` such objects, the one closest to
                  the median (in percentile rank) of the number of context
                  detections, of detections up to the horizon and of peak mag.

Writes ``<outdir>/<name>{,_bands}.{pdf,png}`` and a ``_caption.txt`` for each;
``<name>`` defaults to ``dataset_example_<stem>``, so runs do not overwrite.

    python plot_dataset_example.py                       # random event
    python plot_dataset_example.py --pick characteristic
    python plot_dataset_example.py --stem <stem printed by a run>
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_extrapolate_9band import (  # noqa: E402
    BAND_KEY_SETS,
    DATA_DIR,
    ERROR_COL,
    FILELIST_DIR,
    VALUE_COL,
    _read_stems,
    _stem_to_path,
    detect_band_set,
)
from pub_style import add_style_args, use_pub_style  # noqa: E402

SETS = ("realistic", "dense", "uniform")
SET_LABELS = {"realistic": "Realistic", "dense": "Dense", "uniform": "Uniform"}
# Columns: survey name and its bands (indices into the model's band order).
SURVEYS = (("ZTF", (0, 1, 2)), ("LSST", (3, 4, 5, 6, 7, 8)))
FILTERS = ("g", "r", "i", "u", "g", "r", "i", "z", "y")  # filter of each band
# Slot in the style's color cycle of each filter (one hue per filter, the same
# for ZTF and LSST); slot 5 (sky blue, too close to slot 0) is left out.
FILTER_SLOT = {"u": 0, "g": 2, "r": 1, "i": 3, "z": 4, "y": 6}
# Per-band figure: one hue per dataset (style cycle slots: 2 green, 4 orange,
# 6 black), drawn uniform line below dense dots below realistic points.
SET_SLOT = {"realistic": 2, "dense": 4, "uniform": 6}
SET_Z = {"uniform": 2, "dense": 3, "realistic": 4}
BANDS_DENSE_MS = 0.9  # dense dot size of the per-band figure (x markersize)
DET_MS = 0.9  # realistic detection marker of the per-band figure (x markersize)
CONTEXT_SHADE = "0.9"
X_LABEL = "Time since first detection (days)"
Y_LABEL = r"$M_{\mathrm{AB}}$"
# Selection features: (key, description) in the printout.
FEATURES = (("n_ctx", "context detections"), ("n_det", "detections to horizon"),
            ("peak", "peak mag"))


def read_bands(path: str, band_keys: tuple) -> list:
    """Per band ``(MJD, mag, err)`` of one npz, all rows (upper limits kept).

    A missing error column reads as zeros (= every row a measurement).
    """
    out = []
    with np.load(path, allow_pickle=True) as data:
        for key in band_keys:
            arr = np.asarray(data[key], dtype=np.float64) \
                if key in data.files else np.empty((0, 3))
            if arr.size == 0:
                arr = np.empty((0, 3))
            err = arr[:, ERROR_COL] if arr.shape[1] > ERROR_COL \
                else np.zeros(arr.shape[0])
            out.append((arr[:, 0], arr[:, VALUE_COL], err))
    return out


def load_object(stem: str, maps: dict) -> dict | None:
    """The three datasets of ``stem`` and its t0, or None (no detection)."""
    band_keys = BAND_KEY_SETS[detect_band_set(maps["realistic"][stem])]
    obj = {kind: read_bands(maps[kind][stem], band_keys) for kind in SETS}
    det_t = np.concatenate([t[np.isfinite(e) & np.isfinite(m)]
                            for t, m, e in obj["realistic"]])
    if det_t.size == 0:
        return None
    obj["stem"], obj["t0"] = stem, float(det_t.min())
    return obj


def features(obj: dict, cutoff: float, max_days: float, x_lo: float) -> dict:
    """Selection features of one object (see the module docstring)."""
    n_band, f = [0, 0], {}
    n_ctx = n_det = 0
    for b, (t, m, e) in enumerate(obj["realistic"]):
        ph = (t - obj["t0"])[np.isfinite(e) & np.isfinite(m)]
        n_ctx += int(np.sum(ph <= cutoff))
        k = int(np.sum(ph <= max_days))
        n_det += k
        n_band[0 if b < 3 else 1] += k
    f["n_ctx"], f["n_det"] = n_ctx, n_det
    f["both_surveys"] = min(n_band) > 0

    def in_window(kind, lo, hi):
        return np.concatenate([m[np.isfinite(m) & (t - obj["t0"] > lo)
                                 & (t - obj["t0"] <= hi)]
                               for t, m, _ in obj[kind]])

    # The eval scores an object only if it has dense truth after the cutoff.
    f["scorable"] = in_window("dense", cutoff, max_days).size > 0
    uni = in_window("uniform", x_lo, max_days)
    f["peak"] = float(uni.min()) if uni.size else np.nan
    return f


def mid_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1) of each entry (ties share the mid rank)."""
    s = np.sort(x)
    return (np.searchsorted(s, x, "left") + np.searchsorted(s, x, "right")) \
        / (2.0 * x.size)


def pick_characteristic(stems: list, maps: dict, cutoff: float, max_days: float,
                        x_lo: float) -> tuple:
    """``(object, features, percentile ranks, n candidates)`` of the most
    median-like scorable object in ``stems``."""
    objs, feats = [], []
    for stem in stems:
        obj = load_object(stem, maps)
        if obj is None:
            continue
        f = features(obj, cutoff, max_days, x_lo)
        if f["scorable"] and np.isfinite(f["peak"]):
            objs.append(obj)
            feats.append(f)
    if not objs:
        raise SystemExit("No scorable object among the scanned stems.")
    keep = [i for i, f in enumerate(feats) if f["both_surveys"]]
    if not keep:
        print("WARNING: no scanned object has realistic detections in both "
              "surveys; picking among all scorable ones.")
        keep = list(range(len(objs)))
    ranks = np.stack([mid_rank(np.array([feats[i][k] for i in keep], float))
                      for k, _ in FEATURES], axis=1)
    best = int(np.argmin(np.sum((ranks - 0.5) ** 2, axis=1)))
    i = keep[best]
    return objs[i], feats[i], dict(zip([k for k, _ in FEATURES], ranks[best])), \
        len(keep)


def pick_random(stems: list, maps: dict, cutoff: float, max_days: float,
                x_lo: float, rng: np.random.Generator) -> tuple:
    """``(object, features)`` of a random scorable object in ``stems`` with
    realistic detections in both surveys (else any scorable one)."""
    fallback = None
    for i in rng.permutation(len(stems)):
        obj = load_object(stems[i], maps)
        if obj is None:
            continue
        f = features(obj, cutoff, max_days, x_lo)
        if not (f["scorable"] and np.isfinite(f["peak"])):
            continue
        if f["both_surveys"]:
            return obj, f
        fallback = fallback or (obj, f)
    if fallback is None:
        raise SystemExit("No scorable object among the candidate stems.")
    print("WARNING: no candidate has realistic detections in both surveys; "
          "picked a scorable one without.")
    return fallback


def y_range(obj: dict, x_lim: tuple, bands: tuple = tuple(range(9))) -> tuple:
    """Inverted (faint, bright) magnitude limits of every point of ``bands`` in
    ``x_lim``, with headroom at the bright end for the panel labels."""
    mags = np.concatenate([m[np.isfinite(m) & (t - obj["t0"] >= x_lim[0])
                             & (t - obj["t0"] <= x_lim[1])]
                           for kind in SETS
                           for t, m, _ in (obj[kind][b] for b in bands)])
    if mags.size == 0:
        return 25.0, 15.0
    lo, hi = float(mags.min()), float(mags.max())
    span = max(hi - lo, 1.0)
    return hi + 0.05 * span, lo - 0.2 * span


def style_colors() -> list:
    """The style's categorical colors (fixed order)."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def draw_set(ax: plt.Axes, kind: str, band: tuple, t0: float, x_lim: tuple,
             color: str, z: int, dense_ms: float = 0.45, line_w: float = 1.0,
             line_ls: str = "-", open_detections: bool = False) -> None:
    """One band of one dataset: realistic detections (+-1 sigma) and upper
    limits, dense points, or the uniform curve (a line). ``dense_ms`` and
    ``line_w`` scale the style's marker size and line width.

    ``open_detections``: small white-filled detection markers (they mask the
    dense dots below) with heavier, wide-capped error bars drawn on top.
    """
    t, m, e = band
    ph = t - t0
    ms = plt.rcParams["lines.markersize"]
    mew = plt.rcParams["lines.markeredgewidth"]
    ok = np.isfinite(m) & (ph >= x_lim[0]) & (ph <= x_lim[1])
    if kind == "realistic":
        det, ul = ok & np.isfinite(e), ok & ~np.isfinite(e)
        if open_detections:
            ax.errorbar(ph[det], m[det], yerr=e[det], fmt="o", color=color,
                        ms=DET_MS * ms, mfc="white", mew=1.3 * mew,
                        elinewidth=1.4 * mew, capsize=3, capthick=1.4 * mew,
                        barsabove=True, zorder=z + 1)
        else:
            ax.errorbar(ph[det], m[det], yerr=e[det], fmt="o", color=color,
                        ms=ms, elinewidth=0.8, zorder=z + 1)
        ax.plot(ph[ul], m[ul], ls="none", marker="v", ms=ms, mfc="none",
                mec=color, alpha=0.7, zorder=z)
    elif kind == "dense":
        ax.plot(ph[ok], m[ok], ls="none", marker="o", ms=dense_ms * ms, mew=0,
                color=color, zorder=z, rasterized=True)
    else:
        # Keep NaNs inside the range so gaps break the line.
        sel = (ph >= x_lim[0]) & (ph <= x_lim[1])
        order = np.argsort(ph[sel], kind="stable")
        ax.plot(ph[sel][order], m[sel][order], color=color, zorder=z,
                lw=line_w * plt.rcParams["lines.linewidth"], ls=line_ls)


def fig_example(obj: dict, cutoff: float, x_lim: tuple, size: tuple,
                mag_range: tuple | None = None) -> plt.Figure:
    """The 3 (dataset) x 2 (survey) grid of one object."""
    colors = style_colors()

    def color(b):
        return colors[FILTER_SLOT[FILTERS[b]] % len(colors)]

    ms = plt.rcParams["lines.markersize"]
    fig, axes = plt.subplots(len(SETS), len(SURVEYS), figsize=size,
                             sharex=True, sharey=True)
    for col, (survey, bands) in enumerate(SURVEYS):
        for row, kind in enumerate(SETS):
            ax = axes[row, col]
            ax.axvspan(x_lim[0], cutoff, color=CONTEXT_SHADE, lw=0, zorder=0)
            for b in bands:
                draw_set(ax, kind, obj[kind][b], obj["t0"], x_lim, color(b), 2)
            letter = "abcdef"[row * len(SURVEYS) + col]
            ax.text(0.97, 0.94, f"({letter}) {SET_LABELS[kind]}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=plt.rcParams["legend.fontsize"])
        handles = [Line2D([], [], color=color(b), marker="o", ms=ms,
                          label=FILTERS[b]) for b in bands]
        if col == 0:
            handles.append(Line2D([], [], color="0.35", marker="v", ms=ms,
                                  mfc="none", ls="none", label="Upper limit"))
        axes[0, col].legend(handles=handles, title=survey, ncol=len(handles),
                            loc="lower center", bbox_to_anchor=(0.5, 1.0),
                            handlelength=1.2, handletextpad=0.4,
                            columnspacing=1.0, borderaxespad=0.2)
    axes[0, 0].set_xlim(*x_lim)
    axes[0, 0].set_ylim(*(y_range(obj, x_lim) if mag_range is None
                          else (max(mag_range), min(mag_range))))
    fig.supxlabel(X_LABEL, fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel(Y_LABEL, fontsize=plt.rcParams["axes.labelsize"])
    return fig


def fig_bands(obj: dict, cutoff: float, x_lim: tuple, size: tuple,
              mag_range: tuple | None = None) -> plt.Figure:
    """One panel per band (3 x 3), the three datasets overlaid in each."""
    colors = style_colors()
    ms = plt.rcParams["lines.markersize"]
    lw = plt.rcParams["lines.linewidth"]
    fig, axes = plt.subplots(3, 3, figsize=size, sharex=True)
    for b, ax in enumerate(axes.ravel()):
        ax.axvspan(x_lim[0], cutoff, color=CONTEXT_SHADE, lw=0, zorder=0)
        for kind in SETS:
            draw_set(ax, kind, obj[kind][b], obj["t0"], x_lim,
                     colors[SET_SLOT[kind] % len(colors)], SET_Z[kind],
                     dense_ms=BANDS_DENSE_MS, line_w=0.8, line_ls="--",
                     open_detections=True)
        ax.set_ylim(*(y_range(obj, x_lim, (b,)) if mag_range is None
                      else (max(mag_range), min(mag_range))))
        survey = "ZTF" if b < 3 else "LSST"
        ax.text(0.95, 0.92, f"({'abcdefghi'[b]}) {survey} {FILTERS[b]}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=plt.rcParams["legend.fontsize"])
    axes[0, 0].set_xlim(*x_lim)
    c = {kind: colors[SET_SLOT[kind] % len(colors)] for kind in SETS}
    mew = plt.rcParams["lines.markeredgewidth"]
    handles = [
        Line2D([], [], color=c["realistic"], marker="o", ms=DET_MS * ms,
               mfc="white", mew=1.3 * mew, ls="none", label="Realistic"),
        Line2D([], [], color=c["realistic"], marker="v", ms=ms, mfc="none",
               ls="none", label="Realistic upper limit"),
        Line2D([], [], color=c["dense"], marker="o", ms=BANDS_DENSE_MS * ms,
               mew=0, ls="none", label="Dense"),
        Line2D([], [], color=c["uniform"], lw=0.8 * lw, ls="--",
               label="Uniform"),
    ]
    kw = {"handles": handles, "ncol": len(handles), "handletextpad": 0.4,
          "columnspacing": 1.2}
    try:
        fig.legend(loc="outside upper center", **kw)
    except ValueError:  # matplotlib < 3.7 has no 'outside' locations
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), **kw)
    fig.supxlabel(X_LABEL, fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel(Y_LABEL, fontsize=plt.rcParams["axes.labelsize"])
    return fig


def caption_bands(obj: dict, how: str, cutoff: float, max_days: float,
                  max_context: int) -> str:
    """Caption of the per-band figure (plain text, edit for the paper)."""
    return (
        f"Example kilonova ({obj['stem']}; {how}) in each of the nine filters "
        "(a-c: ZTF g, r, i; d-i: LSST u, g, r, i, z, y), the three datasets "
        "overlaid. Open circles: realistic detections at the survey cadence "
        "(error bars: 1 sigma); open triangles: realistic upper limits; "
        "dots: the dense set; dashed line: the uniform set on a regular "
        f"{grid_step(obj):.2g} d grid, the truth the forecast is scored "
        "against. Time is measured from the first realistic detection. Gray: "
        f"the first {cutoff:g} days, the context window; the model sees up to "
        f"the last {max_context} realistic detections in it and forecasts "
        f"from {cutoff:g} to {max_days:g} days.")


def grid_step(obj: dict) -> float:
    """Median time step (d) of the uniform set."""
    steps = np.concatenate([np.diff(np.sort(t)) for t, _, _ in obj["uniform"]
                            if t.size > 1])
    return float(np.median(steps)) if steps.size else np.nan


def caption(obj: dict, how: str, cutoff: float, max_days: float,
            max_context: int) -> str:
    """Figure caption (plain text, edit for the paper)."""
    return (
        f"Example kilonova ({obj['stem']}; {how}) in the three datasets, ZTF "
        "(left) and LSST (right) filters. (a, b) Realistic: detections at the "
        "survey cadence (error bars: 1 sigma) and upper limits (open triangles). "
        "(c, d) Dense: the same event sampled densely in time. (e, f) Uniform: "
        f"the light curve on a regular {grid_step(obj):.2g} d grid, the truth "
        "the forecast is scored against. Time is measured from the first "
        f"realistic detection. Gray: the first {cutoff:g} days, the context "
        f"window; the model sees up to the last {max_context} realistic "
        f"detections in it and forecasts from {cutoff:g} to {max_days:g} days.")


def resolve_stem(stem: str, maps: dict, paired: set) -> str:
    """The paired stem ``stem`` names: the stem itself, an npz path, or the
    object number alone (``123`` matches ``lc_000123``, ``lc_123`` ...).

    Exits with the sets it is (not) in and example stems if none matches.
    """
    stem = os.path.splitext(os.path.basename(stem))[0]
    if stem in paired:
        return stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        hits = sorted(s for s in paired
                      if "".join(ch for ch in s if ch.isdigit()).lstrip("0")
                      == digits.lstrip("0"))
        if len(hits) == 1:
            print(f"--stem {stem}: using {hits[0]}")
            return hits[0]
        if len(hits) > 1:
            raise SystemExit(f"--stem {stem} is ambiguous: {', '.join(hits[:10])}")
    found = [kind for kind in SETS if stem in maps[kind]]
    examples = ", ".join(sorted(paired)[:5])
    raise SystemExit(
        f"--stem {stem} is not in all three sets (found in: "
        f"{', '.join(found) or 'none'}). {len(paired)} objects are; e.g. "
        f"{examples}. Omit --stem to pick a characteristic test object.")


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sets = os.path.join(DATA_DIR, "rubin_lsst_ztf{}_10000_dataset_same_seed",
                        "lc_*.npz")
    p.add_argument("--realistic_glob", default=sets.format(""))
    p.add_argument("--dense_glob", default=sets.format("_dense"))
    p.add_argument("--uniform_glob", default=sets.format("_uniform"))
    p.add_argument("--filelist", default=os.path.join(
        FILELIST_DIR, "kn_rubin_ztf_test.txt"),
        help="Objects to pick from ('' = every object in all three sets).")
    p.add_argument("--stem", default=None,
                   help="Plot this object instead of picking one.")
    p.add_argument("--pick", choices=("random", "characteristic"),
                   default="random",
                   help="How the object is picked without --stem.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed of the random pick (default: a fresh one, printed).")
    p.add_argument("--max_scan", type=int, default=500,
                   help="Objects scanned for the characteristic pick (0 = all).")
    p.add_argument("--late_time_cutoff_days", type=float, default=2.0)
    p.add_argument("--late_time_max_days", type=float, default=10.0)
    p.add_argument("--max_context_len", type=int, default=12,
                   help="Context cap (caption only).")
    p.add_argument("--x_min", type=float, default=-1.0,
                   help="Left phase limit (d); the right one is the horizon.")
    p.add_argument("--mag_range", type=float, nargs=2, default=None,
                   metavar=("BRIGHT", "FAINT"),
                   help="Magnitude axis limits (default: every plotted point).")
    p.add_argument("--size", type=float, nargs=2, default=(7.0, 5.2),
                   metavar=("W", "H"), help="Figure size (in).")
    p.add_argument("--bands_size", type=float, nargs=2, default=(7.0, 6.0),
                   metavar=("W", "H"), help="Size of the per-band figure (in).")
    p.add_argument("--bands_mag_range", type=float, nargs=2, default=None,
                   metavar=("BRIGHT", "FAINT"),
                   help="Magnitude limits shared by every per-band panel "
                   "(default: each panel fits its own band).")
    add_style_args(p)
    p.add_argument("--outdir", default="paper_figures")
    p.add_argument("--name", default=None,
                   help="Output file stem (default dataset_example_<stem>).")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return p.parse_args()


def main() -> None:
    """Pick (or load) the object, draw the figure, write the caption."""
    args = get_args()
    use_pub_style(args.style, args.font_dir)
    cutoff, max_days = args.late_time_cutoff_days, args.late_time_max_days
    maps = {kind: _stem_to_path(getattr(args, f"{kind}_glob")) for kind in SETS}
    paired = set.intersection(*(set(m) for m in maps.values()))
    if not paired:
        raise SystemExit("No object is in all three sets: "
                         + ", ".join(f"{k} {len(m)}" for k, m in maps.items()))

    if args.stem:
        args.stem = resolve_stem(args.stem, maps, paired)
        obj = load_object(args.stem, maps)
        if obj is None:
            raise SystemExit(f"{args.stem} has no realistic detection.")
        f = features(obj, cutoff, max_days, args.x_min)
        how = "chosen by hand"
        print(f"{args.stem}: " + ", ".join(f"{d} {f[k]:g}" for k, d in FEATURES))
    else:
        stems = sorted(paired & _read_stems(args.filelist)
                       if args.filelist else paired)
        if not stems:
            raise SystemExit(f"No object of {args.filelist} is in all three sets.")
        if args.pick == "random":
            seed = args.seed if args.seed is not None \
                else int(np.random.default_rng().integers(2 ** 31))
            obj, f = pick_random(stems, maps, cutoff, max_days, args.x_min,
                                 np.random.default_rng(seed))
            how = "a randomly drawn test event"
            print(f"Random pick (seed {seed}) of {len(stems)} candidates: "
                  f"{obj['stem']}, "
                  + ", ".join(f"{d} {f[k]:g}" for k, d in FEATURES))
            print(f"Plotting {obj['stem']}  (rerun it with --stem {obj['stem']} "
                  f"or --seed {seed})")
        else:
            if args.max_scan > 0:
                stems = stems[:args.max_scan]
            obj, f, ranks, n = pick_characteristic(stems, maps, cutoff, max_days,
                                                   args.x_min)
            how = f"the most median-like of {n} test events"
            print(f"Picked {obj['stem']} of {n} candidates ({len(stems)} "
                  "scanned): " + ", ".join(f"{d} {f[k]:g} "
                                           f"({100 * ranks[k]:.0f}th pct)"
                                           for k, d in FEATURES))
    if args.name is None:
        args.name = f"dataset_example_{obj['stem']}"

    os.makedirs(args.outdir, exist_ok=True)
    x_lim = (args.x_min, max_days)
    figures = (
        (args.name, fig_example(obj, cutoff, x_lim, tuple(args.size),
                                args.mag_range), caption),
        (f"{args.name}_bands", fig_bands(obj, cutoff, x_lim,
                                         tuple(args.bands_size),
                                         args.bands_mag_range), caption_bands),
    )
    for name, fig, cap in figures:
        for fmt in args.formats:
            path = os.path.join(args.outdir, f"{name}.{fmt}")
            fig.savefig(path)
            print(f"Wrote {path}")
        plt.close(fig)
        path = os.path.join(args.outdir, f"{name}_caption.txt")
        with open(path, "w") as fh:
            fh.write(textwrap.fill(cap(obj, how, cutoff, max_days,
                                       args.max_context_len), 88) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
