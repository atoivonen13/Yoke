"""Simple extrapolation baselines for the 9-band late-time forecast.

No model / GPU / torch. Same objects, context, scored points and output files as
``eval_dense_latetime_9band.py`` (DIRECT mode), so every script that reads an
eval dir works unchanged, and paired deltas against a study are valid (same test
objects, same points).

Per object, exactly the eval's protocol: t0 = first realistic detection; the
context is the realistic detections (upper limits dropped) with phase <= 2 d;
scored are the dense and uniform points with 2 < phase <= 10 d and lead > 0
from the last context detection. Each baseline forecasts every band from that
context:

  persist   the band's last context magnitude, held flat.
  linear    one decline rate per object (least squares over the context with
            per-band intercepts; slope clipped to [0, MAX_SLOPE] mag/d, since a
            kilonova fades), extrapolated from each band's fitted level at the
            last context detection.
  template  the population median light curve per band (train-split uniform
            truth vs phase since first realistic detection), shifted by one grey
            offset = median context residual from it.

A band with no context detection takes the template level at the last context
phase (persist / linear), so every scored point gets a forecast.

Intervals: per band and 1-d lead bin, split-conformal half-widths fit on the
VALIDATION split (``--calibration_truth``, uniform by default as in the eval):
k_lo / k_hi = the (1 - alpha/2) quantile of (med - true) / (true - med), so each
side misses ~alpha/2 and the median is untouched. They fill pred_low /
pred_high (and the identical pred_low_cal / pred_high_cal), and a k = 1
``interval_calibration.json`` makes the plot scripts read the band as
calibrated (it already is).

Outputs go to pseudo-study dirs, ``--study_base`` + 1 / 2 / 3 = persist / linear
/ template, so the ``--studies`` modes of the plot scripts take them as is:

    python baseline_extrapolate_9band.py
    # -> runs/study_901 (persist), study_902 (linear), study_903 (template)
    #    each with dense_latetime_eval_9band/latetime_scored_points.csv, ...
    python plot_metrics_by_day.py --studies 138 901 902 903 --truth uniform
    python plot_interval_coverage.py --studies 138 903
    python plot_quantile_reliability.py --studies 138 903
    python plot_pred_vs_truth.py --studies 903

The band set (PS1/SDSS or LSST ugrizy) is read from the files; band names, and so
the CSVs, are the same for both.
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_intervals import MIN_HALF_WIDTH, conformal_scale  # noqa: E402

BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
N_BANDS = len(BAND_NAMES)
# Both filter sets in the model's band order (ps1 = NINE_BAND_KEYS).
BAND_KEY_SETS = {
    "ps1": ("arr_ztfg", "arr_ztfr", "arr_ztfi", "arr_sdssu", "arr_ps1__g",
            "arr_ps1__r", "arr_ps1__i", "arr_ps1__z", "arr_ps1__y"),
    "lsst": ("arr_ztfg", "arr_ztfr", "arr_ztfi", "arr_lsstu", "arr_lsstg",
             "arr_lsstr", "arr_lssti", "arr_lsstz", "arr_lssty"),
}
VALUE_COL = 1
ERROR_COL = 2
DATA_DIR = "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves"
FILELIST_DIR = "/net/sescratch1/exempt/artimis/atoivonen/filelists"
METHODS = ("persist", "linear", "template")
METHOD_STUDY_OFFSET = {"persist": 1, "linear": 2, "template": 3}
# Phase grid step of the template (d), = the uniform set's time step.
TEMPLATE_STEP = 0.05
# Linear decline-rate clip (mag / d) and the within-band time spread (sum of
# squared deviations, d^2) needed to fit a rate at all; below it -> rate 0.
MAX_SLOPE = 2.0
MIN_SXX = 0.01
# Lead bins with fewer calibration points fall back to the band's all-lead scale.
MIN_BIN_POINTS = 100
IO_WORKERS = 8


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _stem_to_path(data_glob: str) -> dict:
    return {_stem(f): f for f in glob.glob(data_glob)} if data_glob else {}


def _read_stems(path: str) -> set:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def detect_band_set(path: str) -> str:
    """``"lsst"`` if the npz has LSST filter arrays, else ``"ps1"``."""
    with np.load(path, allow_pickle=True) as data:
        return "lsst" if any(k.startswith("arr_lsst") for k in data.files) \
            else "ps1"


def read_stream(path: str, band_keys: tuple, drop_upper_limits: bool) -> tuple:
    """Merged, time-sorted ``(times abs MJD, mags, band index)`` of one npz.

    As ``eval_dense_latetime_9band.read_merged_stream``, without the redshift
    and upper-limit flag the baselines do not use.
    """
    times, values, bands = [], [], []
    with np.load(path, allow_pickle=True) as data:
        for b, key in enumerate(band_keys):
            if key not in data.files or data[key].size == 0:
                continue
            arr = data[key]
            if drop_upper_limits:
                arr = arr[np.isfinite(arr[:, ERROR_COL])]
            if arr.shape[0] == 0:
                continue
            times.append(arr[:, 0].astype(np.float64))
            values.append(arr[:, VALUE_COL].astype(np.float32))
            bands.append(np.full(arr.shape[0], b, dtype=np.int64))
    if not times:
        return (np.empty(0), np.empty(0, np.float32), np.empty(0, np.int64))
    t, v, b = np.concatenate(times), np.concatenate(values), np.concatenate(bands)
    order = np.argsort(t, kind="stable")
    return t[order], v[order], b[order]


def split_object(real, dense, uniform, cutoff: float, max_days: float):
    """The eval's context and scored points for one object, or None.

    Mirrors ``eval_object`` (DIRECT mode), including when it returns None (no
    context, or no dense point to score), so the scored objects are the model's.

    Returns:
        dict | None: ``ctx`` (phase, mag, band of the context detections),
        ``last_phase``, and ``points`` = {truth: {band, phase, lead, true}}.
    """
    r_t, r_v, r_b = real
    d_t, d_v, d_b = dense
    if r_t.shape[0] < 1 or d_t.shape[0] < 1:
        return None
    t0 = float(r_t[0])
    ctx = (r_t - t0) <= cutoff
    if not np.any(ctx):
        return None
    last_t = float(r_t[ctx][-1])
    d_phase = d_t - t0
    late = np.nonzero((d_phase > cutoff) & (d_phase <= max_days))[0]
    lead = (d_t[late] - last_t).astype(np.float32)
    late, lead = late[lead > 0], lead[lead > 0]
    if late.shape[0] == 0:
        return None
    points = {"dense": {"band": d_b[late], "phase": d_phase[late],
                        "lead": lead, "true": d_v[late]}}
    if uniform is not None:
        u_t, u_v, u_b = uniform
        u_phase, u_lead = u_t - t0, u_t - last_t
        m = ((u_phase > cutoff) & (u_phase <= max_days) & (u_lead > 0)
             & np.isfinite(u_v))
        if np.any(m):
            points["uniform"] = {
                "band": u_b[m], "phase": u_phase[m].astype(np.float32),
                "lead": u_lead[m].astype(np.float32), "true": u_v[m]}
    return {"ctx": ((r_t[ctx] - t0), r_v[ctx], r_b[ctx]),
            "last_phase": last_t - t0, "points": points}


class Template:
    """Median light curve per band on a phase grid (phase from t0)."""

    def __init__(self, grid: np.ndarray, curves: np.ndarray, n_objects: int,
                 source: str) -> None:
        """Wrap a ``[N_BANDS, len(grid)]`` curve array on a TEMPLATE_STEP grid."""
        self.grid, self.curves = grid, curves  # curves [N_BANDS, len(grid)]
        self.n_objects, self.source = n_objects, source

    def at(self, band: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Template magnitude of ``band`` at ``phase`` (linear interpolation)."""
        pos = (np.asarray(phase, dtype=np.float64) - self.grid[0]) / TEMPLATE_STEP
        i = np.clip(np.floor(pos).astype(np.int64), 0, self.grid.shape[0] - 2)
        w = np.clip(pos - i, 0.0, 1.0)
        return (1.0 - w) * self.curves[band, i] + w * self.curves[band, i + 1]


def build_template(stems, real_map, src_map, band_keys, max_days,
                   source) -> Template:
    """Per-band nan-median over ``stems`` of the ``src_map`` truth vs phase.

    Phase is from each object's first realistic detection, as in the eval.
    Grid points no object covers take the nearest covered value.
    """
    grid = np.arange(0.0, max_days + 2.0 * TEMPLATE_STEP, TEMPLATE_STEP)

    def one(stem):
        r_t, _, _ = read_stream(real_map[stem], band_keys, True)
        if r_t.shape[0] == 0:
            return None
        t, v, b = read_stream(src_map[stem], band_keys, False)
        out = np.full((N_BANDS, grid.shape[0]), np.nan)
        for k in range(N_BANDS):
            m = (b == k) & np.isfinite(v)
            if m.sum() >= 2:
                out[k] = np.interp(grid, t[m] - r_t[0], v[m], left=np.nan,
                                   right=np.nan)
        return out

    with ThreadPoolExecutor(IO_WORKERS) as pool:
        curves = [c for c in pool.map(one, stems) if c is not None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        med = np.nanmedian(np.stack(curves), axis=0)
    for k in range(N_BANDS):
        ok = np.isfinite(med[k])
        if not ok.any():
            raise ValueError(f"template: no {BAND_NAMES[k]} data in {source}")
        med[k] = np.interp(grid, grid[ok], med[k][ok])
    return Template(grid, med, len(curves), source)


def fit_decline(c_p, c_v, c_b, last_phase) -> tuple:
    """Shared decline rate with per-band intercepts (least squares).

    Returns:
        tuple: ``(slope mag/d, {band: fitted level at last_phase})``; slope 0
        (persistence of the band means) when no band spans enough time.
    """
    sxx = sxy = 0.0
    centers = {}
    for b in np.unique(c_b):
        m = c_b == b
        tb, vb = c_p[m], c_v[m].astype(np.float64)
        centers[int(b)] = (tb.mean(), vb.mean())
        dt = tb - tb.mean()
        sxx += float(dt @ dt)
        sxy += float(dt @ (vb - vb.mean()))
    slope = float(np.clip(sxy / sxx, 0.0, MAX_SLOPE)) if sxx >= MIN_SXX else 0.0
    levels = {b: mv + slope * (last_phase - mt) for b, (mt, mv) in centers.items()}
    return slope, levels


def forecast(method, obj, template, band, phase) -> tuple:
    """Median forecast of ``method`` at (band, phase) points; ``(med, slope)``."""
    c_p, c_v, c_b = obj["ctx"]
    last = obj["last_phase"]
    offset = float(np.median(c_v - template.at(c_b, c_p)))
    if method == "template":
        return template.at(band, phase) + offset, None
    # Bands without a context detection: template level at the last context phase.
    level = template.at(np.arange(N_BANDS), np.full(N_BANDS, last)) + offset
    slope = 0.0
    if method == "persist":
        for b in np.unique(c_b):
            level[b] = c_v[c_b == b][-1]
    else:
        slope, fitted = fit_decline(c_p, c_v, c_b, last)
        for b, lv in fitted.items():
            level[b] = lv
    return level[band] + slope * (phase - last), slope


def run_split(stems, maps, band_keys, template, methods, args) -> dict:
    """Forecast every method over one split's objects.

    Returns:
        dict: ``{truth: {stem, band, phase, lead, true, med: {method: arr}}}``
        plus ``n_objects`` and the linear ``slopes``.
    """
    def read(stem):
        uni = (read_stream(maps["uniform"][stem], band_keys, False)
               if stem in maps["uniform"] else None)
        return stem, split_object(
            read_stream(maps["realistic"][stem], band_keys, True),
            read_stream(maps["dense"][stem], band_keys, False), uni,
            args.late_time_cutoff_days, args.late_time_max_days)

    acc = {}
    slopes = []
    n_obj = 0
    with ThreadPoolExecutor(IO_WORKERS) as pool:
        for stem, obj in pool.map(read, stems):
            if obj is None:
                continue
            n_obj += 1
            for truth, p in obj["points"].items():
                a = acc.setdefault(truth, {"stem": [], "band": [], "phase": [],
                                           "lead": [], "true": [],
                                           "med": {m: [] for m in methods}})
                a["stem"].append(np.full(p["band"].shape[0], stem, dtype=object))
                for k in ("band", "phase", "lead", "true"):
                    a[k].append(p[k])
                for m in methods:
                    med, slope = forecast(m, obj, template, p["band"], p["phase"])
                    a["med"][m].append(med)
                    if m == "linear" and truth == "dense":
                        slopes.append(slope)
    out = {truth: {k: (np.concatenate(v) if k != "med" else
                       {m: np.concatenate(x) for m, x in v.items()})
                   for k, v in a.items()} for truth, a in acc.items()}
    out["n_objects"] = n_obj
    out["slopes"] = np.asarray(slopes)
    return out


def lead_bin(lead, edges) -> np.ndarray:
    """Lead-bin index, clipped so leads past the last edge use the last bin."""
    return np.clip(np.digitize(lead, edges) - 1, 0, edges.shape[0] - 2)


def fit_bounds(band, lead, med, true, interval, edges) -> np.ndarray:
    """Split-conformal half-widths ``k[side (lo, hi), band, lead bin]`` (mag).

    Bins with fewer than MIN_BIN_POINTS points use the band's all-lead value,
    a band with fewer uses the all-band value.
    """
    level = 1.0 - 0.5 * (1.0 - interval)
    s_lo, s_hi = med - true, true - med
    j = lead_bin(lead, edges)

    def scales(m):
        return conformal_scale(s_lo[m], level), conformal_scale(s_hi[m], level)

    pooled = scales(np.ones(band.shape[0], dtype=bool))
    k = np.empty((2, N_BANDS, edges.shape[0] - 1))
    for b in range(N_BANDS):
        mb = band == b
        kb = scales(mb) if mb.sum() >= MIN_BIN_POINTS else pooled
        for i in range(k.shape[2]):
            m = mb & (j == i)
            k[:, b, i] = scales(m) if m.sum() >= MIN_BIN_POINTS else kb
    return np.maximum(k, MIN_HALF_WIDTH)


def apply_bounds(k, band, lead, med, edges) -> tuple:
    """``(low, high)`` = med -/+ the band x lead-bin half-widths."""
    j = lead_bin(lead, edges)
    return med - k[0, band, j], med + k[1, band, j]


def write_points(path, pts, med, low, high, with_cal) -> None:
    """Write a scored-points CSV in the eval's schema."""
    cols = ["stem", "band", "phase_days", "lead_time_days", "pred_mag",
            "pred_low", "pred_high", "true_mag", "residual_mag"]
    if with_cal:
        cols += ["pred_low_cal", "pred_high_cal"]
    resid = med - pts["true"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(resid.shape[0]):
            row = [pts["stem"][i], BAND_NAMES[pts["band"][i]],
                   f"{pts['phase'][i]:.4f}", f"{pts['lead'][i]:.4f}",
                   f"{med[i]:.4f}", f"{low[i]:.4f}", f"{high[i]:.4f}",
                   f"{pts['true'][i]:.4f}", f"{resid[i]:.4f}"]
            if with_cal:
                row += row[5:7]
            w.writerow(row)
    print(f"Wrote {path}")


def _stats(resid, cov) -> str:
    if resid.shape[0] == 0:
        return f"{'-':>7}{'-':>7}{'-':>8}{'-':>7}"
    return (f"{np.sqrt(np.mean(resid ** 2)):>7.3f}{np.mean(np.abs(resid)):>7.3f}"
            f"{np.mean(resid):>+8.3f}{np.mean(cov):>7.3f}")


def report(label, truth, pts, med, low, high, depths=None) -> dict:
    """Print RMSE / MAE / bias / coverage per band; returns per-band RMSE."""
    resid = med - pts["true"]
    cov = (pts["true"] >= low) & (pts["true"] <= high)
    print(f"\n[{label}] {truth} truth: {resid.shape[0]} points, "
          f"{np.unique(pts['stem']).shape[0]} objects "
          "(RMSE, MAE, bias = mean(pred - true), 90% coverage)")
    head = f"  {'band':<6}{'n':>8}{'RMSE':>7}{'MAE':>7}{'bias':>8}{'cov':>7}"
    if depths is not None:
        head += f"{'depth':>7}{'RMSE_in':>8}{'bias_in':>8}{'RMSE_out':>9}" \
                f"{'bias_out':>9}"
    print(head)
    rmse = {}
    for b in range(N_BANDS):
        m = pts["band"] == b
        if not m.any():
            continue
        line = f"  {BAND_NAMES[b]:<6}{m.sum():>8d}{_stats(resid[m], cov[m])}"
        rmse[BAND_NAMES[b]] = float(np.sqrt(np.mean(resid[m] ** 2)))
        if depths is not None and BAND_NAMES[b] in depths:
            d = depths[BAND_NAMES[b]]
            beyond = m & (pts["true"] > d)
            within = m & ~beyond
            line += f"{d:>7.2f}"
            for mm, wid in ((within, 8), (beyond, 9)):
                r = resid[mm]
                line += (f"{np.sqrt(np.mean(r ** 2)):>{wid}.3f}"
                         f"{np.mean(r):>+{wid}.3f}" if r.size else
                         f"{'-':>{wid}}{'-':>{wid}}")
        print(line)
    print(f"  {'ALL':<6}{resid.shape[0]:>8d}{_stats(resid, cov)}")
    rmse["ALL"] = float(np.sqrt(np.mean(resid ** 2)))
    return rmse


def dense_depths(dense) -> dict:
    """Per-band 99th percentile of the dense scored truth (as the eval)."""
    return {BAND_NAMES[b]: float(np.percentile(dense["true"][dense["band"] == b],
                                               99))
            for b in range(N_BANDS) if (dense["band"] == b).any()}


def _split_stems(path, paired, max_objects) -> list:
    stems = sorted(paired if not path else paired & _read_stems(path))
    return stems[:max_objects] if max_objects > 0 else stems


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--realistic_glob", default=os.path.join(
        DATA_DIR, "rubin_ztf_10000_dataset_same_seed/lc_*.npz"))
    p.add_argument("--dense_glob", default=os.path.join(
        DATA_DIR, "rubin_ztf_dense_10000_dataset_same_seed/lc_*.npz"))
    p.add_argument("--uniform_glob", default=os.path.join(
        DATA_DIR, "rubin_ztf_uniform_10000_dataset_same_seed/lc_*.npz"),
        help="Uniform truth: scored, and the template source (dense if absent).")
    p.add_argument("--test_filelist", default=os.path.join(
        FILELIST_DIR, "kn_rubin_ztf_test.txt"),
        help="'' = every paired object.")
    p.add_argument("--calibrate_filelist", default=os.path.join(
        FILELIST_DIR, "kn_rubin_ztf_val.txt"),
        help="Split the interval half-widths are fit on ('' = fit in-sample "
        "on the test split, not for reporting).")
    p.add_argument("--train_filelist", default=os.path.join(
        FILELIST_DIR, "kn_rubin_ztf_train.txt"),
        help="Template objects ('' = every paired object not in test / "
        "calibration).")
    p.add_argument("--n_template", type=int, default=2000,
                   help="Train objects in the template (0 = all).")
    p.add_argument("--calibration_truth", choices=("uniform", "dense"),
                   default="uniform")
    p.add_argument("--interval", type=float, default=0.9)
    p.add_argument("--lead_bin_days", type=float, default=1.0)
    p.add_argument("--late_time_cutoff_days", type=float, default=2.0)
    p.add_argument("--late_time_max_days", type=float, default=10.0)
    p.add_argument("--max_objects", type=int, default=0,
                   help="Cap objects per split (0 = all).")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--study_base", type=int, default=900,
                   help="Outputs -> <runs_dir>/study_<base + 1/2/3>/"
                   "dense_latetime_eval_9band for persist / linear / template.")
    return p.parse_args()


def main() -> None:
    """Build the template, fit the bounds on val, score test, write the CSVs."""
    args = get_args()
    maps = {k: _stem_to_path(getattr(args, f"{k}_glob"))
            for k in ("realistic", "dense", "uniform")}
    for k in ("realistic", "dense"):
        if not maps[k]:
            raise SystemExit(f"--{k}_glob matched no files")
    sets = {k: detect_band_set(next(iter(m.values()))) for k, m in maps.items() if m}
    if len(set(sets.values())) > 1:
        raise SystemExit(f"band sets differ across globs: {sets}")
    band_set = sets["realistic"]
    band_keys = BAND_KEY_SETS[band_set]
    n_uni = f"{len(maps['uniform'])} files" if maps["uniform"] else "none"
    print(f"Band set: {band_set}; uniform truth: {n_uni}")

    paired = set(maps["realistic"]) & set(maps["dense"])
    test = _split_stems(args.test_filelist, paired, args.max_objects)
    cal = (_split_stems(args.calibrate_filelist, paired, args.max_objects)
           if args.calibrate_filelist else [])
    if set(test) & set(cal):
        raise SystemExit("test and calibration splits overlap")
    src = "uniform" if maps["uniform"] else "dense"
    held = set(test) | set(cal)
    train = sorted((paired if not args.train_filelist
                    else paired & _read_stems(args.train_filelist)) - held)
    train = [s for s in train if s in maps[src]]
    if args.n_template > 0:
        train = train[:args.n_template]
    print(f"Objects: test {len(test)}, calibration {len(cal)}, template "
          f"{len(train)} ({src} truth)")

    template = build_template(train, maps["realistic"], maps[src], band_keys,
                              args.late_time_max_days, src)
    methods = list(args.methods)
    res_test = run_split(test, maps, band_keys, template, methods, args)
    res_cal = run_split(cal, maps, band_keys, template, methods, args) \
        if cal else res_test
    if not cal:
        print("WARNING: no calibration split -> bounds fit IN-SAMPLE on test.")
    cal_truth = args.calibration_truth if args.calibration_truth in res_cal \
        else "dense"
    edges = np.arange(0.0, args.late_time_max_days + args.lead_bin_days,
                      args.lead_bin_days)
    depths = dense_depths(res_test["dense"])
    sl = res_test["slopes"]
    if "linear" in methods and sl.size:
        print(f"linear: decline rate fit for {np.mean(sl > 0):.1%} of test "
              f"objects; median {np.median(sl):.3f} mag/d, clipped at MAX_SLOPE "
              f"{np.mean(sl >= MAX_SLOPE):.1%}")

    summary = {}
    for m in methods:
        study = args.study_base + METHOD_STUDY_OFFSET[m]
        outdir = os.path.join(args.runs_dir, f"study_{study:03d}",
                              "dense_latetime_eval_9band")
        os.makedirs(os.path.join(outdir, "calibration_split"), exist_ok=True)
        c = res_cal[cal_truth]
        k = fit_bounds(c["band"], c["lead"], c["med"][m], c["true"],
                       args.interval, edges)
        label = f"{m} (study {study})"
        summary[m] = {}
        for split, res, sub in (("test", res_test, outdir),
                                ("calibration", res_cal if cal else None,
                                 os.path.join(outdir, "calibration_split"))):
            if res is None:
                continue
            for truth in ("dense", "uniform"):
                if truth not in res:
                    continue
                pts = res[truth]
                lo, hi = apply_bounds(k, pts["band"], pts["lead"],
                                      pts["med"][m], edges)
                name = ("latetime_scored_points.csv" if truth == "dense"
                        else "latetime_uniform_scored_points.csv")
                write_points(os.path.join(sub, name), pts, pts["med"][m], lo, hi,
                             with_cal=truth == "dense")
                if split == "test":
                    summary[m][truth] = report(
                        label, truth, pts, pts["med"][m], lo, hi,
                        depths if truth == "uniform" else None)
        with open(os.path.join(outdir, "interval_calibration.json"), "w") as fh:
            json.dump({"interval": args.interval, "truth": cal_truth,
                       "bands": {b: {"k_lo": 1.0, "k_hi": 1.0} for b in BAND_NAMES},
                       "calibration_split": None,
                       "note": "baseline: pred_low/high are already the "
                       "val-fit conformal band (baseline_meta.json)"}, fh, indent=2)
        meta = {"method": m, "study": study, "band_set": band_set,
                "template": {"source": template.source,
                             "n_objects": template.n_objects},
                "n_objects": {"test": res_test["n_objects"],
                              "calibration": res_cal["n_objects"] if cal else 0},
                "calibration_truth": cal_truth, "lead_edges": edges.tolist(),
                "k_lo": {b: k[0, i].round(4).tolist()
                         for i, b in enumerate(BAND_NAMES)},
                "k_hi": {b: k[1, i].round(4).tolist()
                         for i, b in enumerate(BAND_NAMES)},
                "max_slope": MAX_SLOPE, "min_sxx": MIN_SXX,
                "args": vars(args)}
        with open(os.path.join(outdir, "baseline_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    for truth in ("dense", "uniform"):
        rows = [m for m in methods if truth in summary[m]]
        if not rows:
            continue
        print(f"\n[test] {truth}-truth RMSE by method")
        print(f"  {'band':<6}" + "".join(f"{m:>10}" for m in rows))
        for b in (*BAND_NAMES, "ALL"):
            if b in summary[rows[0]][truth]:
                print(f"  {b:<6}" + "".join(
                    f"{summary[m][truth][b]:>10.4f}" for m in rows))


if __name__ == "__main__":
    main()
