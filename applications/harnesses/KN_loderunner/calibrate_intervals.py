"""Per-band post-hoc calibration of the quantile-head forecast interval.

Study 131 showed the 0.05/0.95 band is calibrated in aggregate (0.899 vs 0.90)
but not per band (ztfg 0.82, g 0.86 too narrow; ztfr 0.93, i 0.93 too wide).
That is structural: the color-anchored head keeps the quantile spread in the
shared pivot, so one z-spread per object is mapped to every band and the head
cannot widen a single band. This fits one width scale per band and side on a
CALIBRATION split instead of retraining.

Split-conformal, per band b and side (asymmetric, so the median bias that pushes
truth out one side -- e.g. ztfg's under-fade -- is absorbed by that side only):

    s_hi = (true - med) / (high - med)      k_hi = Q_{1 - alpha/2}(s_hi)
    s_lo = (med - true) / (med - low)       k_lo = Q_{1 - alpha/2}(s_lo)

with alpha = 1 - interval and the finite-sample level ceil((n+1)(1-alpha/2))/n.
The calibrated band is ``[med - k_lo (med - low), med + k_hi (high - med)]``: the
median (and so RMSE) is untouched, and the learned per-point shape of the band
(wider at long lead, per-object spread) is kept -- only its per-band scale moves.
Points of one object are correlated, so the conformal guarantee is approximate;
the held-out ``--check_csv`` coverage is the honest read.

Calibrate on the VALIDATION split, never on test. Workflow (cluster):

    # 1. eval the checkpoint on the val objects (writes a scored CSV)
    python eval_dense_latetime_9band.py --study 131 --epoch 100 \\
        --test_filelist <FILELIST_DIR>/kn_rubin_ztf_val.txt \\
        --outdir runs/study_131/dense_latetime_eval_9band_val
    # 2. fit the scales on val, check them on test
    python calibrate_intervals.py \\
        --csv runs/study_131/dense_latetime_eval_9band_val/latetime_scored_points.csv \\
        --check_csv runs/study_131/dense_latetime_eval_9band/latetime_scored_points.csv \\
        --out runs/study_131/interval_calibration.json
    # 3. re-run the test eval with the scales (calibrated plots + CSV columns)
    python eval_dense_latetime_9band.py --study 131 --epoch 100 \\
        --test_filelist <FILELIST_DIR>/kn_rubin_ztf_test.txt \\
        --interval_calibration runs/study_131/interval_calibration.json

No model / GPU / torch.
"""

import argparse
import csv
import json
import math

import numpy as np

# Floor on a predicted half-width (mag) so a collapsed band cannot divide by ~0.
MIN_HALF_WIDTH = 1e-3


def load_scored(csv_path: str) -> dict:
    """Load band, median, outer quantiles, and truth from a scored-points CSV.

    Args:
        csv_path (str): ``latetime_scored_points.csv`` from the eval.

    Returns:
        dict: Aligned arrays keyed band (names), med, low, high, true.
    """
    bands, med, low, high, true = [], [], [], [], []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            bands.append(r["band"])
            med.append(float(r["pred_mag"]))
            low.append(float(r["pred_low"]))
            high.append(float(r["pred_high"]))
            true.append(float(r["true_mag"]))
    return {
        "band": np.array(bands, dtype=object),
        "med": np.array(med),
        "low": np.array(low),
        "high": np.array(high),
        "true": np.array(true),
    }


def conformal_scale(scores: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile of ``scores`` at ``level``."""
    n = scores.shape[0]
    q = min(1.0, math.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


def fit_band_scales(med, low, high, true, interval: float) -> tuple:
    """Fit (k_lo, k_hi) for one band so each side misses ``(1-interval)/2``.

    Args:
        med, low, high, true (np.ndarray): One band's median, outer quantiles,
            and truth (mag).
        interval (float): Target central coverage, e.g. 0.9.

    Returns:
        tuple: ``(k_lo, k_hi)``, each clipped at 0 so the band never excludes
        the median.
    """
    level = 1.0 - 0.5 * (1.0 - interval)
    s_lo = (med - true) / np.maximum(med - low, MIN_HALF_WIDTH)
    s_hi = (true - med) / np.maximum(high - med, MIN_HALF_WIDTH)
    return (max(0.0, conformal_scale(s_lo, level)),
            max(0.0, conformal_scale(s_hi, level)))


def apply_scales(med, low, high, k_lo, k_hi) -> tuple:
    """Rescale the band's half-widths about the median (arrays broadcast)."""
    return med - k_lo * (med - low), med + k_hi * (high - med)


def _coverage(true, low, high) -> float:
    return float(np.mean((true >= np.minimum(low, high))
                         & (true <= np.maximum(low, high))))


def report(d: dict, scales: dict, title: str) -> None:
    """Print raw vs calibrated coverage and mean width per band."""
    print(f"\n{title}")
    header = (f"  {'band':<6}{'n':>7}{'k_lo':>7}{'k_hi':>7}{'cov raw':>9}"
              f"{'cov cal':>9}{'width raw':>11}{'width cal':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_lo, all_hi = np.copy(d["low"]), np.copy(d["high"])
    for band in sorted(set(d["band"])):
        m = d["band"] == band
        s = scales.get(band, {"k_lo": 1.0, "k_hi": 1.0})
        lo, hi = apply_scales(d["med"][m], d["low"][m], d["high"][m],
                              s["k_lo"], s["k_hi"])
        all_lo[m], all_hi[m] = lo, hi
        print(f"  {band:<6}{int(m.sum()):>7}{s['k_lo']:>7.2f}{s['k_hi']:>7.2f}"
              f"{_coverage(d['true'][m], d['low'][m], d['high'][m]):>9.3f}"
              f"{_coverage(d['true'][m], lo, hi):>9.3f}"
              f"{np.mean(d['high'][m] - d['low'][m]):>11.3f}"
              f"{np.mean(hi - lo):>11.3f}")
    print(f"  {'ALL':<6}{d['true'].shape[0]:>7}{'':>14}"
          f"{_coverage(d['true'], d['low'], d['high']):>9.3f}"
          f"{_coverage(d['true'], all_lo, all_hi):>9.3f}"
          f"{np.mean(d['high'] - d['low']):>11.3f}"
          f"{np.mean(all_hi - all_lo):>11.3f}")


def main() -> None:
    """Fit per-band scales on a calibration CSV, report, and write the JSON."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="Calibration scored CSV (eval run on the VALIDATION split).")
    p.add_argument("--check_csv", default=None,
                   help="Optional held-out scored CSV (test split) to report the "
                   "calibrated coverage on.")
    p.add_argument("--interval", type=float, default=0.9,
                   help="Target central coverage of the calibrated band.")
    p.add_argument("--min_points", type=int, default=100,
                   help="Bands with fewer calibration points keep k = 1.")
    p.add_argument("--out", required=True, help="Output calibration JSON.")
    args = p.parse_args()

    d = load_scored(args.csv)
    print(f"Calibration set: {d['true'].shape[0]} points from {args.csv}")
    print(f"Target central coverage: {args.interval:.2f}")
    scales = {}
    for band in sorted(set(d["band"])):
        m = d["band"] == band
        n = int(m.sum())
        if n < args.min_points:
            print(f"  {band}: only {n} points (< {args.min_points}), keeping k = 1")
            k_lo, k_hi = 1.0, 1.0
        else:
            k_lo, k_hi = fit_band_scales(d["med"][m], d["low"][m], d["high"][m],
                                         d["true"][m], args.interval)
        scales[band] = {"k_lo": k_lo, "k_hi": k_hi, "n": n}

    report(d, scales, "Calibration split (in-sample -- cov cal ~target by construction)")
    if args.check_csv is not None:
        report(load_scored(args.check_csv), scales,
               f"Held-out check: {args.check_csv}")

    with open(args.out, "w") as fh:
        json.dump({"interval": args.interval, "calibration_csv": args.csv,
                   "bands": scales}, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
