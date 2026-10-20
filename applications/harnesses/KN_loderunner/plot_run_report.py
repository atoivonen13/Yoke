"""Publication figures for ONE KN LodeRunner run (no model / GPU / torch).

Reads what ``eval_dense_latetime_9band.py`` and the training loop leave in
``runs/study_NNN/``:

  * ``dense_latetime_eval_9band/latetime_uniform_scored_points.csv`` -- the
    headline score: forecast vs the noise-free uniform truth (no depth limit);
  * ``dense_latetime_eval_9band/latetime_scored_points.csv`` -- forecast vs the
    survey-limited dense truth (also sets each band's dense depth, the 99th
    percentile of its dense truth, as in the eval);
  * ``dense_latetime_eval_9band/interval_calibration.json`` -- per-band k_lo /
    k_hi of the calibrated 90% band (raw band only if absent);
  * ``dense_latetime_eval_9band/eval_meta.json`` -- which checkpoint was scored;
  * ``training_study*_epoch*.csv`` / ``validation_study*_epoch*.csv`` -- loss
    records (``epoch, batch, loss``).

Figures (PDF + PNG, 7.0 in = two-column page width) in ``runs/study_NNN/figures``:

  1. ``loss``           train / validation loss (per-epoch median) vs epoch,
                        the evaluated checkpoint marked;
  2. ``band_metrics``   uniform RMSE (dense RMSE overlaid) and bias per band;
  3. ``error_vs_lead``  RMSE and bias vs lead time, per band + all bands;
  4. ``pred_vs_truth``  predicted median vs uniform truth (2-D histogram) per
                        band, with the median prediction at fixed truth, the
                        1:1 line and the dense depth;
  5. ``coverage``       raw vs calibrated 90% coverage per band and vs lead;
  6. ``examples``       forecast + calibrated band vs truth for the 10th / 50th /
                        90th percentile object by per-object uniform RMSE (the
                        CSVs hold only the scored late-time window, so the
                        realistic context is not drawn).
  7. ``median_example_truth`` / ``median_example_forecast``  a pair for the
                        median object by per-object uniform RMSE, all 9 bands:
                        the realistic context the model saw and the uniform
                        truth, then the same with the forecast median and
                        calibrated band added (same axes, so the two read as a
                        before / after). The context is rebuilt from the
                        realistic npz with the eval's rule (detections up to the
                        cutoff, trailing window, cap). Needs the realistic and
                        uniform npz sets (globs default to the ones the eval
                        recorded in ``eval_meta.json``); skipped if unreadable.

plus ``metrics.csv`` / ``metrics.tex`` (per band and pooled).

Error bars / shading are object-bootstrap 90% CIs: points of one object
correlate, so whole objects are resampled. They are test-sample noise only;
seed noise (~0.05 overall uniform RMSE) is NOT in them. Residual = pred - true,
so + = forecast too faint.

    python plot_run_report.py --study 209
    python plot_run_report.py --study 209 --label "Fresh backbone" --no_title

``plot_compare_runs.py`` overlays several runs using the loaders here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
BAND_LABELS = ("ZTF g", "ZTF r", "ZTF i", "LSST u", "LSST g", "LSST r",
               "LSST i", "LSST z", "LSST y")
BAND_INDEX = {name: b for b, name in enumerate(BAND_NAMES)}
N_BANDS = len(BAND_NAMES)
# Per-band groups plus the pooled "all bands" group, in plotting order.
GROUP_NAMES = (*BAND_NAMES, "all")
GROUP_LABELS = (*BAND_LABELS, "All")
EVAL_SUBDIR = "dense_latetime_eval_9band"
UNIFORM_CSV = "latetime_uniform_scored_points.csv"
DENSE_CSV = "latetime_scored_points.csv"
CALIBRATION_JSON = "interval_calibration.json"
META_JSON = "eval_meta.json"
# Each band's dense depth = this percentile of its dense truth (eval dense_depth).
DEPTH_PERCENTILE = 99
LEAD_BIN_DAYS = 0.5
# Lead bins with fewer points than this are not drawn (too noisy to read).
MIN_BIN_POINTS = 50
# Pred-vs-truth: truth bin width and minimum count for the median-given-truth line.
TRUTH_BIN_MAG = 0.25
MIN_TRUTH_BIN_POINTS = 20
EXAMPLE_PERCENTILES = (10, 50, 90)
# Median-example pair: days of uniform truth drawn before the first detection.
EXAMPLE_PRE_DAYS = 1.0

# Light-curve npz sets for the median-example pair (fallbacks when the eval's
# eval_meta.json does not name them). Band keys in band-index order, as
# kilonova_dataset.NINE_BAND_KEYS / LSST_NINE_BAND_KEYS (not imported: torch).
KN_DATA_DIR = "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves"
DEFAULT_REALISTIC_GLOB = (f"{KN_DATA_DIR}/rubin_lsst_ztf_10000_dataset_same_seed/"
                          "lc_*.npz")
DEFAULT_UNIFORM_GLOB = (f"{KN_DATA_DIR}/rubin_lsst_ztf_uniform_10000_dataset_"
                        "same_seed/lc_*.npz")
NPZ_BAND_KEYS = (
    ("arr_ztfg", "arr_ztfr", "arr_ztfi", "arr_sdssu", "arr_ps1__g",
     "arr_ps1__r", "arr_ps1__i", "arr_ps1__z", "arr_ps1__y"),
    ("arr_ztfg", "arr_ztfr", "arr_ztfi", "arr_lsstu", "arr_lsstg", "arr_lsstr",
     "arr_lssti", "arr_lsstz", "arr_lssty"),
)
NPZ_VALUE_COL = 1
NPZ_ERROR_COL = 2  # non-finite error = upper limit

# Colors: the dataviz reference palette, light mode. Categorical slots are used
# in this fixed order (run i -> slot i, never cycled); the first 4 pass the
# adjacent-pair CVD check. Every series also gets a marker and a dash pattern,
# so identity never rests on color alone (grayscale print, CVD).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
DASHES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 2)),
          (0, (6, 2)))
# Sequential (magnitude) = one hue light -> dark: the blue ramp 100..700.
SEQ_CMAP = LinearSegmentedColormap.from_list("kn_seq_blue", (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"))
FULL_W = 7.0   # two-column figure width (in)
HALF_W = 3.4   # one-column figure width (in)


def set_style() -> None:
    """Matplotlib defaults for print: 8 pt sans, hairline chrome, vector fonts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.default": "regular",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "errorbar.capsize": 0,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, outdir: str, name: str, formats: tuple) -> None:
    """Write ``fig`` as ``<outdir>/<name>.<fmt>`` for each format, then close it."""
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(path)
        print(f"Wrote {path}")
    plt.close(fig)


def series_style(i: int) -> dict:
    """Color / marker / dash of categorical series ``i`` (fixed order)."""
    return {"color": SERIES[i], "marker": MARKERS[i], "ls": DASHES[i]}


# --------------------------------------------------------------------- loading


def _read_points(path: str) -> pd.DataFrame:
    cols = ["stem", "band", "phase_days", "lead_time_days", "pred_mag",
            "pred_low", "pred_high", "true_mag"]
    df = pd.read_csv(path, usecols=cols)
    df = df.rename(columns={"phase_days": "phase", "lead_time_days": "lead",
                            "pred_mag": "med", "pred_low": "low",
                            "pred_high": "high", "true_mag": "true"})
    unknown = sorted(set(df["band"]) - set(BAND_NAMES))
    if unknown:
        raise ValueError(f"{path}: unknown bands {unknown}")
    df["b"] = df["band"].map(BAND_INDEX).astype(np.int64)
    df["r"] = df["med"] - df["true"]
    return df


def _load_calibration(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        cal = json.load(fh)
    k_lo, k_hi = np.ones(N_BANDS), np.ones(N_BANDS)
    for name, v in cal["bands"].items():
        if name in BAND_INDEX:
            k_lo[BAND_INDEX[name]] = v["k_lo"]
            k_hi[BAND_INDEX[name]] = v["k_hi"]
    return {"k_lo": k_lo, "k_hi": k_hi, "interval": float(cal["interval"]),
            "truth": cal.get("truth", "dense")}


def _add_intervals(df: pd.DataFrame, cal: dict | None) -> None:
    """Add the calibrated band (``calibrate_intervals.apply_scales``) and hits."""
    if cal is None:
        df["low_cal"], df["high_cal"] = df["low"], df["high"]
    else:
        b = df["b"].to_numpy()
        df["low_cal"] = df["med"] - cal["k_lo"][b] * (df["med"] - df["low"])
        df["high_cal"] = df["med"] + cal["k_hi"][b] * (df["high"] - df["med"])
    for suffix, lo, hi in (("raw", "low", "high"), ("cal", "low_cal", "high_cal")):
        df[f"hit_{suffix}"] = ((df["true"] >= np.minimum(df[lo], df[hi]))
                               & (df["true"] <= np.maximum(df[lo], df[hi])))
    df["width_cal"] = (df["high_cal"] - df["low_cal"]).abs()


def _read_records(pattern: str) -> pd.DataFrame | None:
    """Per-epoch median and 5-95th percentile of a set of loss record CSVs."""
    arrays = []
    for path in sorted(glob.glob(pattern)):
        if os.path.getsize(path) == 0:
            continue
        try:
            arr = np.loadtxt(path, delimiter=",", ndmin=2)
        except ValueError:
            continue  # half-written record from a crashed epoch
        if arr.size and arr.shape[1] >= 3:
            arrays.append(arr[:, [0, 2]])
    if not arrays:
        return None
    df = pd.DataFrame(np.vstack(arrays), columns=["epoch", "loss"])
    g = df.groupby("epoch")["loss"]
    return pd.DataFrame({"median": g.median(), "p05": g.quantile(0.05),
                         "p95": g.quantile(0.95)}).reset_index()


def _eval_signature(eval_dir: str) -> dict:
    """The eval's settings signature from eval_meta.json ({} if absent)."""
    path = os.path.join(eval_dir, META_JSON)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh).get("signature", {})


def _eval_epoch(signature: dict) -> int | None:
    """Epoch of the checkpoint the eval scored, if known."""
    hit = re.search(r"epoch(\d+)", os.path.basename(signature.get("ckpt", "")))
    return int(hit.group(1)) if hit else None


def load_run(study: int, runs_root: str = "runs", label: str | None = None,
             eval_dir: str | None = None) -> dict:
    """Scored points, calibration, dense depths and loss curves of one study."""
    run_dir = os.path.join(runs_root, f"study_{study:03d}")
    eval_dir = eval_dir or os.path.join(run_dir, EVAL_SUBDIR)
    uni_path = os.path.join(eval_dir, UNIFORM_CSV)
    if not os.path.exists(uni_path):
        raise FileNotFoundError(f"No {uni_path}; run eval_dense_latetime_9band.py "
                                f"--study {study} first.")
    uniform = _read_points(uni_path)
    dense_path = os.path.join(eval_dir, DENSE_CSV)
    dense = _read_points(dense_path) if os.path.exists(dense_path) else None
    cal = _load_calibration(os.path.join(eval_dir, CALIBRATION_JSON))
    for df in (uniform, dense):
        if df is not None:
            _add_intervals(df, cal)

    depth = np.full(N_BANDS, np.inf)
    if dense is not None:
        for b in range(N_BANDS):
            t = dense.loc[dense["b"] == b, "true"].to_numpy()
            if t.size:
                depth[b] = np.percentile(t, DEPTH_PERCENTILE)
    uniform["beyond"] = uniform["true"].to_numpy() > depth[uniform["b"].to_numpy()]

    tag = f"{study:03d}"
    signature = _eval_signature(eval_dir)
    run = {
        "study": study,
        "label": label or f"Study {study}",
        "run_dir": run_dir,
        "eval_dir": eval_dir,
        "uniform": uniform,
        "dense": dense,
        "depth": depth,
        "calibrated": cal is not None,
        "interval": cal["interval"] if cal else 0.90,
        "has_band": bool((uniform["high"] - uniform["low"]).abs().max() > 1e-6),
        "epoch": _eval_epoch(signature),
        "signature": signature,
        "train_loss": _read_records(
            os.path.join(run_dir, f"training_study{tag}_epoch*.csv")),
        "val_loss": _read_records(
            os.path.join(run_dir, f"validation_study{tag}_epoch*.csv")),
    }
    print(f"study {study} ({run['label']}): {len(uniform)} uniform points, "
          f"{uniform['stem'].nunique()} objects, checkpoint epoch {run['epoch']}, "
          f"calibration {'yes (' + cal['truth'] + ' truth)' if cal else 'none'}")
    return run


def restrict_to(run: dict, stems: set) -> None:
    """Drop every scored point whose object is not in ``stems`` (in place)."""
    for key in ("uniform", "dense"):
        if run[key] is not None:
            run[key] = run[key][run[key]["stem"].isin(stems)].reset_index(drop=True)


def lead_edges(runs: list) -> np.ndarray:
    """Common lead-time bin edges covering every run's uniform points."""
    top = max(float(r["uniform"]["lead"].max()) for r in runs)
    n = int(np.ceil(top / LEAD_BIN_DAYS - 1e-9))
    return np.arange(n + 1) * LEAD_BIN_DAYS


# ------------------------------------------------------------------ statistics


class ObjectBootstrap:
    """Object-resampled statistics (points of one object are correlated).

    One fixed set of multinomial object weights is shared by every call; with
    one ``universe`` of stems for several runs, each run sees the same
    resamples, so differences between runs are paired.
    """

    def __init__(self, universe: list, n_boot: int, seed: int) -> None:
        """Draw ``n_boot`` resamples of the objects in ``universe``."""
        self.index = pd.Index(sorted(universe))
        self.n_obj = len(self.index)
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(self.n_obj, np.full(self.n_obj, 1.0 / self.n_obj),
                                 size=n_boot).astype(np.float64)

    def attach(self, run: dict) -> None:
        """Add an integer object code column to a run's scored points."""
        for key in ("uniform", "dense"):
            df = run[key]
            if df is None:
                continue
            codes = self.index.get_indexer(df["stem"])
            if np.any(codes < 0):
                raise ValueError(f"{run['label']}: objects outside the bootstrap set")
            df["obj"] = codes


def _ci(draws: np.ndarray) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(draws, 5, axis=0),
                np.nanpercentile(draws, 95, axis=0))


def _values(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "rmse":
        return df["r"].to_numpy() ** 2
    if kind == "bias":
        return df["r"].to_numpy()
    if kind in ("hit_raw", "hit_cal", "beyond"):
        return df[kind].to_numpy(dtype=np.float64)
    if kind == "width_cal":
        return df["width_cal"].to_numpy()
    raise ValueError(f"unknown statistic {kind!r}")


def _groups(df: pd.DataFrame, grouping: str, edges: np.ndarray | None) -> tuple:
    if grouping == "band":
        return df["b"].to_numpy(), N_BANDS
    if grouping == "all":
        return np.zeros(len(df), dtype=np.int64), 1
    n_lead = len(edges) - 1
    lead = np.clip(np.digitize(df["lead"].to_numpy(), edges) - 1, 0, n_lead - 1)
    if grouping == "lead":
        return lead, n_lead
    if grouping == "band_lead":
        return df["b"].to_numpy() * n_lead + lead, N_BANDS * n_lead
    raise ValueError(f"unknown grouping {grouping!r}")


def summarize(boot: ObjectBootstrap, df: pd.DataFrame, kind: str,
              grouping: str = "band_all", edges: np.ndarray | None = None,
              min_points: int = 1) -> dict:
    """Point estimate, 90% CI and bootstrap draws of a per-group statistic.

    Args:
        boot (ObjectBootstrap): Shared object resamples (``df`` must be attached).
        df (pd.DataFrame): Scored points.
        kind (str): ``rmse``, ``bias``, ``hit_raw`` / ``hit_cal`` (coverage),
            ``beyond`` (fraction beyond dense depth) or ``width_cal``.
        grouping (str): ``band``, ``all``, ``band_all`` (9 bands then pooled),
            ``lead`` or ``band_lead`` (band-major; needs ``edges``).
        edges (np.ndarray | None): Lead-time bin edges.
        min_points (int): Groups with fewer points are NaN.

    Returns:
        dict: ``point``, ``lo``, ``hi``, ``n`` (per group) and ``boot``
        (``[n_boot, n_groups]``).
    """
    if grouping == "band_all":
        a = summarize(boot, df, kind, "band", edges, min_points)
        b = summarize(boot, df, kind, "all", edges, min_points)
        return {k: np.concatenate([a[k], b[k]], axis=-1) for k in a}
    groups, n_groups = _groups(df, grouping, edges)
    flat = df["obj"].to_numpy() * n_groups + groups
    size = boot.n_obj * n_groups
    s = np.bincount(flat, weights=_values(df, kind), minlength=size)
    n = np.bincount(flat, minlength=size).astype(np.float64)
    s = s.reshape(boot.n_obj, n_groups)
    n = n.reshape(boot.n_obj, n_groups)
    count = n.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = s.sum(axis=0) / count
        draws = (boot.w @ s) / (boot.w @ n)
    if kind == "rmse":
        point, draws = np.sqrt(point), np.sqrt(draws)
    thin = count < max(min_points, 1)
    point[thin] = np.nan
    draws[:, thin] = np.nan
    lo, hi = _ci(draws)
    return {"point": point, "lo": lo, "hi": hi, "n": count, "boot": draws}


def lead_grid(boot: ObjectBootstrap, df: pd.DataFrame, kind: str,
              edges: np.ndarray) -> dict:
    """``summarize`` per (band, lead bin) + pooled per lead bin: ``[10, n_lead]``."""
    n_lead = len(edges) - 1
    a = summarize(boot, df, kind, "band_lead", edges, MIN_BIN_POINTS)
    b = summarize(boot, df, kind, "lead", edges, MIN_BIN_POINTS)
    return {k: np.concatenate([a[k].reshape(N_BANDS, n_lead),
                               b[k].reshape(1, n_lead)])
            for k in ("point", "lo", "hi")}


def metrics_table(run: dict, boot: ObjectBootstrap) -> pd.DataFrame:
    """Per-band and pooled scalar metrics of one run (rows = GROUP_NAMES)."""
    u = run["uniform"]
    out = {"group": list(GROUP_NAMES)}

    def put(name: str, stat: dict, ci: bool = False) -> None:
        out[name] = stat["point"]
        if ci:
            out[f"{name}_lo"], out[f"{name}_hi"] = stat["lo"], stat["hi"]

    out["n_points"] = summarize(boot, u, "bias")["n"].astype(int)
    put("uniform_rmse", summarize(boot, u, "rmse"), ci=True)
    put("uniform_bias", summarize(boot, u, "bias"), ci=True)
    put("within_rmse", summarize(boot, u[~u["beyond"]], "rmse"))
    put("within_bias", summarize(boot, u[~u["beyond"]], "bias"))
    put("beyond_frac", summarize(boot, u, "beyond"))
    put("beyond_rmse", summarize(boot, u[u["beyond"]], "rmse"))
    put("beyond_bias", summarize(boot, u[u["beyond"]], "bias"))
    if run["dense"] is not None:
        put("dense_rmse", summarize(boot, run["dense"], "rmse"))
        put("dense_bias", summarize(boot, run["dense"], "bias"))
    if run["has_band"]:
        put("cov_raw", summarize(boot, u, "hit_raw"), ci=True)
        put("cov_cal", summarize(boot, u, "hit_cal"), ci=True)
        put("width_cal", summarize(boot, u, "width_cal"))
    out["dense_depth"] = np.append(run["depth"], np.nan)
    return pd.DataFrame(out)


def fmt_ci(point: float, lo: float, hi: float, digits: int = 3,
           sign: bool = False) -> str:
    """LaTeX ``value^{+up}_{-down}`` for an asymmetric CI."""
    if not np.isfinite(point):
        return "--"
    value = f"{point:+.{digits}f}" if sign else f"{point:.{digits}f}"
    return (f"${value}^{{+{hi - point:.{digits}f}}}"
            f"_{{-{point - lo:.{digits}f}}}$")


def fmt(value: float, digits: int = 3, sign: bool = False) -> str:
    """Fixed-point number for a table cell ('--' for NaN)."""
    if not np.isfinite(value):
        return "--"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def write_tex(path: str, header: list, rows: list, caption: str,
              align: str | None = None) -> None:
    """Write a booktabs LaTeX table (cells already formatted; None row = rule)."""
    align = align or "l" + "c" * (len(header) - 1)
    lines = ["% generated by " + os.path.basename(__file__),
             "\\begin{table}", "\\centering", f"\\caption{{{caption}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += ["\\midrule" if r is None else " & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {path}")


# -------------------------------------------------------------- figure helpers


def band_axis(ax: plt.Axes, with_all: bool = True) -> None:
    """Band category ticks; a hairline separates the pooled 'All' group."""
    labels = GROUP_LABELS if with_all else BAND_LABELS
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.tick_params(axis="x", length=0)
    if with_all:
        ax.axvline(N_BANDS - 0.5, color=AXIS, lw=0.6)


def dodge(n: int, i: int, span: float = 0.8) -> float:
    """Horizontal offset of series ``i`` of ``n`` within a category of width ``span``."""
    return (i - (n - 1) / 2) * span / n


def errorbar(ax: plt.Axes, x: np.ndarray, stat: dict, horizontal: bool = False,
             **kw: object) -> None:
    """Point + 90% CI whisker for each finite group (``x`` = category position).

    ``horizontal`` puts the value on the x axis and ``x`` on the y axis.
    """
    p = stat["point"]
    ok = np.isfinite(p)
    err = np.vstack([p - stat["lo"], stat["hi"] - p])[:, ok]
    pos = np.asarray(x)[ok]
    if horizontal:
        ax.errorbar(p[ok], pos, xerr=err, elinewidth=0.8, **kw)
    else:
        ax.errorbar(pos, p[ok], yerr=err, elinewidth=0.8, **kw)


def lead_line(ax: plt.Axes, centers: np.ndarray, grid: dict, row: int,
              color: str, ls: object = "-", marker: str | None = None,
              label: str | None = None, band: bool = True) -> None:
    """One lead-time curve (``grid`` row) with its CI as a light fill."""
    if band:
        ax.fill_between(centers, grid["lo"][row], grid["hi"][row], color=color,
                        alpha=0.15, lw=0)
    ax.plot(centers, grid["point"][row], color=color, ls=ls, marker=marker,
            markersize=2.5, markevery=2, lw=1.2, label=label)


def lead_panels(height: float = 3.3) -> tuple:
    """2 x 5 grid: nine bands then 'All', shared axes."""
    fig, axes = plt.subplots(2, 5, figsize=(FULL_W, height), sharex=True,
                             sharey=True)
    axes = axes.ravel()
    for g, ax in enumerate(axes):
        ax.set_title(GROUP_LABELS[g] if g < N_BANDS else "All bands")
    return fig, axes


def top_legend(fig: plt.Figure, handles: list, labels: list, ncol: int,
               title: str | None = None) -> None:
    """One legend above the panels; ``title`` sits on top of it (no suptitle clash)."""
    kw = {"ncol": ncol, "title": title, "title_fontsize": 8}
    try:
        fig.legend(handles, labels, loc="outside upper center", **kw)
    except ValueError:  # matplotlib < 3.7 has no 'outside' locations
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   **kw)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    """Bold (a), (b), ... at the top left of a panel."""
    ax.set_title(f"({letter})", loc="left", fontweight="bold")


# --------------------------------------------------------------------- figures


def fig_loss(run: dict, title: str | None) -> plt.Figure | None:
    """Train / validation loss per-epoch median vs epoch."""
    tr, va = run["train_loss"], run["val_loss"]
    if tr is None and va is None:
        print(f"{run['label']}: no loss records in {run['run_dir']}; skipping loss.")
        return None
    fig, ax = plt.subplots(figsize=(HALF_W, 2.4))
    if tr is not None:
        ax.fill_between(tr["epoch"], tr["p05"], tr["p95"], color=SERIES[0],
                        alpha=0.15, lw=0)
        ax.plot(tr["epoch"], tr["median"], color=SERIES[0], lw=1.0,
                label="Train (median, 5–95% of batches)")
    if va is not None:
        ax.plot(va["epoch"], va["median"], color=SERIES[1], ls="--", marker="s",
                markersize=2.5, lw=1.2, label="Validation (median)")
    if run["epoch"] is not None:
        ax.axvline(run["epoch"], color=MUTED, ls=":", lw=0.9)
        ax.annotate(f"evaluated\nepoch {run['epoch']}", (run["epoch"], 1.0),
                    xycoords=("data", "axes fraction"), xytext=(-3, -2),
                    textcoords="offset points", ha="right", va="top",
                    color=INK_2, fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="lower left")
    if title:
        ax.set_title(title, loc="left")
    return fig


def fig_band_metrics(run: dict, boot: ObjectBootstrap,
                     title: str | None) -> plt.Figure:
    """Uniform RMSE (+ dense RMSE) and bias per band, with object-bootstrap CIs."""
    u = run["uniform"]
    rmse = summarize(boot, u, "rmse")
    bias = summarize(boot, u, "bias")
    x = np.arange(len(GROUP_NAMES))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_W, 3.8), sharex=True,
                                   height_ratios=(1.4, 1))
    ax1.bar(x, rmse["point"], width=0.62, color=SERIES[0], edgecolor="white",
            lw=0.6, label="Uniform truth (headline)",
            yerr=np.vstack([rmse["point"] - rmse["lo"], rmse["hi"] - rmse["point"]]),
            error_kw={"elinewidth": 0.8, "ecolor": INK_2})
    if run["dense"] is not None:
        dense = summarize(boot, run["dense"], "rmse")
        ax1.plot(x, dense["point"], ls="none", marker="D", markersize=4,
                 mfc="white", mec=INK, mew=0.9,
                 label="Dense truth (within survey depth)")
    ax1.annotate(f"{rmse['point'][-1]:.3f}", (x[-1], rmse["hi"][-1]),
                 xytext=(0, 2), textcoords="offset points", ha="center",
                 va="bottom", fontsize=7)
    ax1.set_ylabel("RMSE (mag)")
    ax1.set_ylim(bottom=0, top=1.12 * np.nanmax(rmse["hi"]))
    ax1.axvline(N_BANDS - 0.5, color=AXIS, lw=0.6)
    panel_letter(ax1, "a")
    top_legend(fig, *ax1.get_legend_handles_labels(), ncol=2, title=title)

    ax2.axhline(0.0, color=INK, lw=0.6)
    errorbar(ax2, x, bias, fmt="o", color=SERIES[0], markersize=4)
    ax2.set_ylabel("Bias (mag)\n+ = too faint")
    panel_letter(ax2, "b")
    band_axis(ax2)
    return fig


def fig_error_vs_lead(run: dict, boot: ObjectBootstrap, edges: np.ndarray,
                      title: str | None) -> plt.Figure:
    """Uniform RMSE and bias vs lead time per band and pooled."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    rmse = lead_grid(boot, run["uniform"], "rmse", edges)
    bias = lead_grid(boot, run["uniform"], "bias", edges)
    fig, axes = lead_panels()
    for g, ax in enumerate(axes):
        ax.axhline(0.0, color=INK, lw=0.5)
        lead_line(ax, centers, rmse, g, SERIES[0], label="RMSE")
        lead_line(ax, centers, bias, g, SERIES[1], ls="--", label="Bias (+ = too faint)")
    fig.supxlabel("Lead time (days)")
    fig.supylabel("Uniform-truth error (mag)")
    handles, labels = axes[0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=2, title=title)
    return fig


def _median_given_truth(true: np.ndarray, pred: np.ndarray, lims: tuple) -> tuple:
    edges = np.arange(lims[0], lims[1] + TRUTH_BIN_MAG, TRUTH_BIN_MAG)
    idx = np.digitize(true, edges) - 1
    centers, meds = [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() >= MIN_TRUTH_BIN_POINTS:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            meds.append(np.median(pred[m]))
    return np.asarray(centers), np.asarray(meds)


def fig_pred_vs_truth(run: dict, title: str | None, bins: int = 70) -> plt.Figure:
    """2-D histogram of predicted median vs uniform truth, per band."""
    u = run["uniform"]
    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, FULL_W * 0.92))
    axes = axes.ravel()
    hists = []
    for b in range(N_BANDS):
        m = (u["b"] == b).to_numpy()
        t, p = u["true"].to_numpy()[m], u["med"].to_numpy()[m]
        if t.size == 0:
            hists.append(None)
            continue
        both = np.concatenate([t, p])
        lo, hi = np.percentile(both, 0.5), np.percentile(both, 99.5)
        pad = 0.04 * (hi - lo)
        lims = (lo - pad, hi + pad)
        h, xe, ye = np.histogram2d(t, p, bins=bins, range=(lims, lims))
        hists.append((t, p, lims, h, xe, ye))
    vmax = max(h[3].max() for h in hists if h is not None)
    norm = LogNorm(vmin=1, vmax=vmax)
    mesh = None
    for b, ax in enumerate(axes):
        ax.grid(False)
        ax.set_title(BAND_LABELS[b])
        if hists[b] is None:
            ax.set_axis_off()
            continue
        t, p, lims, h, xe, ye = hists[b]
        mesh = ax.pcolormesh(xe, ye, np.ma.masked_less(h.T, 1), cmap=SEQ_CMAP,
                             norm=norm, rasterized=True)
        ax.plot(lims, lims, color=MUTED, ls="--", lw=0.8, label="1:1")
        c, med = _median_given_truth(t, p, lims)
        ax.plot(c, med, color=INK, lw=1.2, label="Median prediction")
        depth = run["depth"][b]
        if lims[0] < depth < lims[1]:
            ax.axvline(depth, color=SERIES[1], ls=":", lw=1.1, label="Dense depth")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
    fig.supxlabel("Uniform truth (mag)")
    fig.supylabel("Predicted median (mag)")
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    top_legend(fig, list(handles.values()), list(handles), ncol=3, title=title)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.tolist(), shrink=0.5, aspect=30, pad=0.02,
                     label="Points per bin")
    return fig


def fig_coverage(run: dict, boot: ObjectBootstrap, edges: np.ndarray,
                 title: str | None) -> plt.Figure | None:
    """Raw vs calibrated central-band coverage on uniform truth."""
    if not run["has_band"]:
        print(f"{run['label']}: point head (no interval); skipping coverage.")
        return None
    u = run["uniform"]
    variants = [("hit_raw", "Raw quantiles", 1, "s", "--")]
    if run["calibrated"]:
        variants.append(("hit_cal", "Calibrated", 0, "o", "-"))
    nominal = run["interval"]
    centers = 0.5 * (edges[1:] + edges[:-1])
    x = np.arange(len(GROUP_NAMES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.6),
                                   width_ratios=(1.6, 1), sharey=True)
    for ax in (ax1, ax2):
        ax.axhline(nominal, color=INK, lw=0.6, ls=":")
    for i, (kind, label, slot, marker, ls) in enumerate(variants):
        errorbar(ax1, x + dodge(len(variants), i, 0.5), summarize(boot, u, kind),
                 fmt=marker, color=SERIES[slot], markersize=3.5, label=label)
        grid = lead_grid(boot, u, kind, edges)
        lead_line(ax2, centers, grid, N_BANDS, SERIES[slot], ls=ls, label=label)
    band_axis(ax1)
    ax1.set_ylabel(f"Coverage of {nominal:.0%} band\n(uniform truth)")
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, lw=0.6, ls=":"))
    labels.append(f"Nominal {nominal:.0%}")
    panel_letter(ax1, "a")
    ax2.set_xlabel("Lead time (days)")
    ax2.set_title("All bands", loc="center")
    panel_letter(ax2, "b")
    top_legend(fig, handles, labels, ncol=len(labels), title=title)
    return fig


def per_object_rmse(run: dict) -> pd.Series:
    """Uniform-truth RMSE of each object (index = stem)."""
    u = run["uniform"]
    return np.sqrt((u["r"] ** 2).groupby(u["stem"]).mean())


def pick_examples(run: dict, stems: list | None) -> list:
    """``(stem, caption)`` per example row: given stems, or RMSE percentiles."""
    per_obj = per_object_rmse(run)
    if stems:
        missing = [s for s in stems if s not in per_obj.index]
        if missing:
            raise ValueError(f"examples not in the scored set: {missing}")
        return [(s, f"RMSE {per_obj[s]:.2f}") for s in stems]
    out = []
    for q in EXAMPLE_PERCENTILES:
        target = np.percentile(per_obj.to_numpy(), q)
        stem = (per_obj - target).abs().idxmin()
        out.append((stem, f"{q}th pct, RMSE {per_obj[stem]:.2f}"))
    return out


def fig_examples(run: dict, bands: list, stems: list | None,
                 title: str | None) -> plt.Figure:
    """Late-time forecast + calibrated band vs truth for a few objects."""
    u, d = run["uniform"], run["dense"]
    picks = pick_examples(run, stems)
    fig, axes = plt.subplots(len(picks), len(bands),
                             figsize=(FULL_W, 1.75 * len(picks) + 0.4),
                             sharex=True, squeeze=False)
    band_label = "Calibrated" if run["calibrated"] else "Raw"
    for i, (stem, caption) in enumerate(picks):
        for j, name in enumerate(bands):
            ax = axes[i, j]
            b = BAND_INDEX[name]
            su = u[(u["stem"] == stem) & (u["b"] == b)].sort_values("phase")
            if run["has_band"]:
                ax.fill_between(su["phase"], su["low_cal"], su["high_cal"],
                                color=SERIES[0], alpha=0.2, lw=0,
                                label=f"{band_label} {run['interval']:.0%} band")
            ax.plot(su["phase"], su["true"], color=INK_2, lw=1.0,
                    label="Uniform truth")
            if d is not None:
                sd = d[(d["stem"] == stem) & (d["b"] == b)]
                ax.plot(sd["phase"], sd["true"], ls="none", marker="o",
                        markersize=2.2, color=MUTED, label="Dense truth")
            ax.plot(su["phase"], su["med"], color=SERIES[0], lw=1.3,
                    label="Forecast median")
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(BAND_LABELS[b])
            if j == 0:
                ax.set_ylabel(f"{caption}\nmag")
            if j == len(bands) - 1:
                ax.annotate(stem, (1.0, 0.0), xycoords="axes fraction",
                            xytext=(-2, 2), textcoords="offset points",
                            ha="right", va="bottom", fontsize=6, color=MUTED)
    fig.supxlabel("Days since first realistic detection")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=len(labels), title=title)
    return fig


def _find_npz(pattern: str, stem: str) -> str | None:
    """Path of object ``stem`` in the set matched by ``pattern`` (None if absent)."""
    direct = os.path.join(os.path.dirname(pattern),
                          stem + os.path.splitext(pattern)[1])
    if os.path.exists(direct):
        return direct
    for path in glob.glob(pattern):
        if os.path.splitext(os.path.basename(path))[0] == stem:
            return path
    return None


def read_npz_stream(path: str, drop_upper_limits: bool) -> tuple:
    """Merged, time-sorted ``(MJD, mag, band index)`` of one light-curve npz.

    Torch-free twin of ``eval_dense_latetime_9band.read_merged_stream``.
    """
    t, v, b = [], [], []
    with np.load(path, allow_pickle=True) as data:
        names = set(data.files)
        keys = next((k for k in NPZ_BAND_KEYS if names & set(k[3:])),
                    NPZ_BAND_KEYS[0])
        for band, key in enumerate(keys):
            if key not in names:
                continue
            arr = np.asarray(data[key])
            if arr.size == 0:
                continue
            if drop_upper_limits:
                arr = arr[np.isfinite(arr[:, NPZ_ERROR_COL])]
            t.append(arr[:, 0].astype(np.float64))
            v.append(arr[:, NPZ_VALUE_COL].astype(np.float64))
            b.append(np.full(len(arr), band, dtype=np.int64))
    if not t:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)
    t, v, b = np.concatenate(t), np.concatenate(v), np.concatenate(b)
    order = np.argsort(t, kind="stable")
    return t[order], v[order], b[order]


def eval_context(t: np.ndarray, v: np.ndarray, b: np.ndarray, cutoff_days: float,
                 window_days: float, max_context_len: int) -> dict | None:
    """The realistic context the eval fed the model (``eval_object``'s rule).

    Phase 0 = the first realistic detection. Keep detections with phase <=
    ``cutoff_days``, then the trailing ``window_days`` window ending at the last
    of them, then the ``max_context_len`` cap.
    """
    # numpy-only module (no torch).
    from yoke.utils.context_selection import window_select_positions

    if t.size == 0:
        return None
    t0 = float(t[0])
    pre = (t - t0) <= cutoff_days
    t, v, b = t[pre], v[pre], b[pre]
    # float32 times, as _select_window compares them.
    t32 = t.astype(np.float32)
    win = np.nonzero(t32 >= t32[-1] - window_days)[0]
    win = win[window_select_positions(win.shape[0], max_context_len)]
    return {"t0": t0, "phase": t[win] - t0, "mag": v[win], "b": b[win],
            "n_pre_cutoff": int(pre.sum())}


def median_example(run: dict, stem: str | None, realistic_glob: str,
                   uniform_glob: str, window_days: float,
                   max_context_len: int) -> dict | None:
    """Context, uniform truth and forecast of the median object (or ``stem``)."""
    per_obj = per_object_rmse(run)
    if stem is None:
        target = np.percentile(per_obj.to_numpy(), 50)
        stem = (per_obj - target).abs().idxmin()
        caption = f"median object, uniform RMSE {per_obj[stem]:.2f}"
    elif stem in per_obj.index:
        rank = float((per_obj < per_obj[stem]).mean() * 100)
        caption = f"uniform RMSE {per_obj[stem]:.2f} ({rank:.0f}th pct)"
    else:
        raise ValueError(f"median example {stem!r} is not in the scored set")
    sig = run["signature"]
    cutoff = float(sig.get("late_time_cutoff_days", 2.0))
    max_days = float(sig.get("late_time_max_days", 10.0))

    paths = {}
    for kind, pattern in (("realistic", realistic_glob), ("uniform", uniform_glob)):
        paths[kind] = _find_npz(pattern, stem)
        if paths[kind] is None:
            print(f"{run['label']}: {stem} not found in the {kind} set "
                  f"{pattern!r}; skipping the median-example pair.")
            return None
    ctx = eval_context(*read_npz_stream(paths["realistic"], True), cutoff,
                       window_days, max_context_len)
    if ctx is None:
        print(f"{run['label']}: {stem} has no realistic detections; skipping the "
              "median-example pair.")
        return None

    u = run["uniform"]
    forecast = u[u["stem"] == stem]
    # The eval's lead is measured from the last pre-cutoff detection, so
    # phase - lead must equal the rebuilt context's last phase.
    last = float(np.median(forecast["phase"] - forecast["lead"]))
    if abs(last - ctx["phase"][-1]) > 0.02:
        warnings.warn(f"{stem}: rebuilt context ends at phase {ctx['phase'][-1]:.3f} "
                      f"d but the eval's leads imply {last:.3f} d; the drawn "
                      "context may not be the one the model saw.", stacklevel=2)

    ut, uv, ub = read_npz_stream(paths["uniform"], False)
    x_lim = (-EXAMPLE_PRE_DAYS, max_days)
    phase = ut - ctx["t0"]
    keep = np.isfinite(uv) & (phase >= x_lim[0]) & (phase <= x_lim[1])
    truth = (phase[keep], uv[keep], ub[keep])

    # One y range per band, shared by the pair so they read as before / after.
    y_lim = []
    for b in range(N_BANDS):
        vals = np.concatenate([truth[1][truth[2] == b], ctx["mag"][ctx["b"] == b],
                               forecast.loc[forecast["b"] == b, "med"].to_numpy()])
        if vals.size == 0:
            y_lim.append(None)
            continue
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(0.06 * (hi - lo), 0.3)
        y_lim.append((hi + pad, lo - pad))  # faint at the bottom
    print(f"{run['label']}: median example {stem} (uniform RMSE "
          f"{per_obj[stem]:.3f}), {len(ctx['phase'])} context detections of "
          f"{ctx['n_pre_cutoff']} before the {cutoff:g} d cutoff")
    return {"stem": stem, "caption": caption, "cutoff": cutoff,
            "ctx": ctx, "truth": truth, "forecast": forecast, "x_lim": x_lim,
            "y_lim": y_lim}


def fig_median_example(run: dict, ex: dict, with_forecast: bool,
                       title: str | None) -> plt.Figure:
    """All 9 bands of one object: realistic context + uniform truth (+ forecast)."""
    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, FULL_W * 0.78), sharex=True)
    axes = axes.ravel()
    ctx, (t_ph, t_mag, t_b), fc = ex["ctx"], ex["truth"], ex["forecast"]
    band_label = "Calibrated" if run["calibrated"] else "Raw"
    for b, ax in enumerate(axes):
        ax.set_title(BAND_LABELS[b])
        ax.axvline(ex["cutoff"], color=MUTED, ls=":", lw=0.9,
                   label=f"Context cutoff ({ex['cutoff']:g} d)")
        m = t_b == b
        ax.plot(t_ph[m], t_mag[m], color=INK_2, lw=1.0, label="Uniform truth")
        note = f"{int(np.sum(ctx['b'] == b))} in context"
        if with_forecast:
            f = fc[fc["b"] == b].sort_values("phase")
            if run["has_band"]:
                ax.fill_between(f["phase"], f["low_cal"], f["high_cal"],
                                color=SERIES[0], alpha=0.2, lw=0,
                                label=f"{band_label} {run['interval']:.0%} band")
            ax.plot(f["phase"], f["med"], color=SERIES[0], lw=1.3,
                    label="Forecast median")
            if len(f):
                note += f", RMSE {np.sqrt(np.mean(f['r'] ** 2)):.2f}"
        m = ctx["b"] == b
        ax.plot(ctx["phase"][m], ctx["mag"][m], ls="none", marker="o",
                markersize=3.5, mfc=SERIES[1], mec=INK, mew=0.4, zorder=3,
                label="Realistic context")
        ax.annotate(note, (0.02, 0.03), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=6, color=MUTED)
        ax.set_xlim(ex["x_lim"])
        if ex["y_lim"][b] is not None:
            ax.set_ylim(ex["y_lim"][b])
    fig.supxlabel("Days since first realistic detection")
    fig.supylabel("Magnitude")
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    caption = f"{ex['stem']}: {ex['caption']}"
    top_legend(fig, list(handles.values()), list(handles), ncol=len(handles),
               title=f"{title}\n{caption}" if title else caption)
    return fig


def write_metrics(run: dict, table: pd.DataFrame, outdir: str) -> None:
    """Metrics CSV + a LaTeX table of the headline columns."""
    csv_path = os.path.join(outdir, "metrics.csv")
    table.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"Wrote {csv_path}")
    header = ["Band", "Uniform RMSE", "Bias", "Beyond-depth bias", "Dense RMSE"]
    if "cov_cal" in table:
        header.append(f"Coverage ({run['interval']:.0%})")
    rows = []
    for g, r in table.iterrows():
        row = [GROUP_LABELS[g],
               fmt_ci(r["uniform_rmse"], r["uniform_rmse_lo"], r["uniform_rmse_hi"]),
               fmt(r["uniform_bias"], sign=True), fmt(r["beyond_bias"], sign=True),
               fmt(r.get("dense_rmse", np.nan))]
        if "cov_cal" in table:
            row.append(fmt(r["cov_cal"]))
        rows.append(row)
    rows.insert(N_BANDS, None)  # rule above the pooled row
    write_tex(os.path.join(outdir, "metrics.tex"), header, rows,
              f"{run['label']}: late-time forecast error against the uniform "
              "truth (mag; residual = predicted $-$ true, $+$ = too faint). "
              "Superscripts/subscripts: object-bootstrap 90\\% CI. Beyond-depth: "
              "uniform points fainter than the band's dense depth.")
    print_summary(run, table)


def print_summary(run: dict, table: pd.DataFrame) -> None:
    """Console version of the metrics table."""
    print(f"\n{run['label']} (study {run['study']}, epoch {run['epoch']})")
    cols = ["uniform_rmse", "uniform_bias", "within_rmse", "beyond_frac",
            "beyond_bias", "dense_rmse", "cov_cal"]
    cols = [c for c in cols if c in table]
    print(f"  {'group':>6}" + "".join(f"{c:>14}" for c in cols))
    for _, r in table.iterrows():
        print(f"  {r['group']:>6}" + "".join(f"{r[c]:>14.4f}" for c in cols))


def main() -> None:
    """Load one study's eval outputs and write its figures and tables."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=int, required=True)
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--eval_dir", default=None,
                   help=f"Eval output dir (default runs/study_NNN/{EVAL_SUBDIR}).")
    p.add_argument("--label", default=None, help="Run name used in titles/tables.")
    p.add_argument("--outdir", default=None,
                   help="Figure dir (default runs/study_NNN/figures).")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--no_title", action="store_true",
                   help="Omit the run label from the figures (for a paper, "
                   "where the caption names the run).")
    p.add_argument("--example_bands", nargs="+", default=["ztfg", "u", "r"],
                   choices=BAND_NAMES)
    p.add_argument("--example_stems", nargs="+", default=None,
                   help="Objects for the examples figure (default: the 10th / "
                   "50th / 90th percentile object by uniform RMSE).")
    p.add_argument("--median_example_stem", default=None,
                   help="Object for the median-example pair (default: the "
                   "median object by uniform RMSE).")
    p.add_argument("--realistic_glob", default=None,
                   help="Realistic npz set for the median-example context "
                   "(default: the eval's, from eval_meta.json, else "
                   f"{DEFAULT_REALISTIC_GLOB}).")
    p.add_argument("--uniform_glob", default=None,
                   help="Uniform npz set for the median-example truth (default: "
                   "the eval's, else the LSST uniform set).")
    p.add_argument("--context_window_days", type=float, default=2.0,
                   help="The checkpoint's trailing context window (rebuilds the "
                   "eval context; not in eval_meta.json).")
    p.add_argument("--max_context_len", type=int, default=12,
                   help="The checkpoint's context cap.")
    p.add_argument("--n_boot", type=int, default=500,
                   help="Object-bootstrap resamples for the CIs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_style()
    run = load_run(args.study, args.runs_root, args.label, args.eval_dir)
    outdir = args.outdir or os.path.join(run["run_dir"], "figures")
    os.makedirs(outdir, exist_ok=True)
    stems = set(run["uniform"]["stem"])
    if run["dense"] is not None:
        stems |= set(run["dense"]["stem"])
    boot = ObjectBootstrap(list(stems), args.n_boot, args.seed)
    boot.attach(run)
    edges = lead_edges([run])
    title = None if args.no_title else (
        f"{run['label']} (study {run['study']}"
        + (f", epoch {run['epoch']})" if run["epoch"] is not None else ")"))
    fmts = tuple(args.formats)
    sig = run["signature"]
    example = median_example(
        run, args.median_example_stem,
        args.realistic_glob or sig.get("realistic_glob") or DEFAULT_REALISTIC_GLOB,
        args.uniform_glob or sig.get("uniform_glob") or DEFAULT_UNIFORM_GLOB,
        args.context_window_days, args.max_context_len)

    for name, fig in (
        ("loss", fig_loss(run, title)),
        ("band_metrics", fig_band_metrics(run, boot, title)),
        ("error_vs_lead", fig_error_vs_lead(run, boot, edges, title)),
        ("pred_vs_truth", fig_pred_vs_truth(run, title)),
        ("coverage", fig_coverage(run, boot, edges, title)),
        ("examples", fig_examples(run, args.example_bands, args.example_stems,
                                  title)),
        ("median_example_truth", example and fig_median_example(
            run, example, False, title)),
        ("median_example_forecast", example and fig_median_example(
            run, example, True, title)),
    ):
        if fig is not None:
            save(fig, outdir, name, fmts)
    write_metrics(run, metrics_table(run, boot), outdir)


if __name__ == "__main__":
    main()
