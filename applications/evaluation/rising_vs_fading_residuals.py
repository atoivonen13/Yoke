"""Split dense late-time forecast residuals by whether the truth is still rising.

Read-only diagnostic against ``latetime_scored_points.csv`` (written by
``eval_dense_latetime_9band.py``). It answers one question raised by the per-object
plots: the forecast declines monotonically from the last context point, but some
objects are still *rising* to peak in the scored 2-10 d window -- how much of the
pooled RMSE comes from those still-rising points, and is it worth a targeted fix?

Magnitudes are inverted (smaller number == brighter). A light curve that is still
BRIGHTENING therefore has ``true_mag`` DECREASING with phase (negative slope); a
FADING curve has ``true_mag`` increasing (positive slope). For every scored point
we estimate the local truth slope d(true_mag)/d(phase) from its neighbours in the
same (object, band) track, label the point rising / fading / flat, and compare the
RMSE / MAE / bias and the point counts across labels -- overall and per band.

No model, GPU, or torch needed. Run on the cluster where the CSV lives:

    python rising_vs_fading_residuals.py \\
        --csv runs/study_075/dense_latetime_eval_9band/latetime_scored_points.csv

Optional ``--plot out.png`` writes a per-band rising-vs-fading RMSE bar chart.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


# A track's local slope must exceed this (mag/day, absolute) to count as rising or
# fading; smaller magnitudes are labelled "flat" (near peak / plateau) so genuine
# turnover points do not get force-sorted into one bin by measurement noise.
FLAT_SLOPE_EPS = 0.02


def _load_rows(csv_path: str) -> list[dict]:
    """Load the scored-points CSV into a list of typed row dicts.

    Args:
        csv_path (str): Path to ``latetime_scored_points.csv``.

    Returns:
        list[dict]: One dict per row with numeric fields cast to float.
    """
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                {
                    "stem": r["stem"],
                    "band": r["band"],
                    "phase": float(r["phase_days"]),
                    "lead_time": float(r["lead_time_days"]),
                    "pred_mag": float(r["pred_mag"]),
                    "true_mag": float(r["true_mag"]),
                    "residual": float(r["residual_mag"]),
                }
            )
    return rows


def _label_by_truth_slope(rows: list[dict]) -> np.ndarray:
    """Label each row rising / fading / flat by the local truth-magnitude slope.

    For each (stem, band) track sorted by phase, the slope at point i is the
    central difference of ``true_mag`` vs ``phase`` (one-sided at the ends). Because
    magnitude is inverted, slope < -eps => brightening (rising), slope > +eps =>
    fading, |slope| <= eps => flat (near peak/plateau). Tracks with a single point
    are labelled "flat" (no slope defined).

    Args:
        rows (list[dict]): Rows from :func:`_load_rows`.

    Returns:
        np.ndarray: Object array of labels ("rising" | "fading" | "flat"), aligned
        to ``rows``.
    """
    labels = np.empty(len(rows), dtype=object)
    labels[:] = "flat"

    tracks = defaultdict(list)
    for i, r in enumerate(rows):
        tracks[(r["stem"], r["band"])].append(i)

    for idxs in tracks.values():
        idxs.sort(key=lambda i: rows[i]["phase"])
        if len(idxs) < 2:
            continue
        phase = np.array([rows[i]["phase"] for i in idxs])
        mag = np.array([rows[i]["true_mag"] for i in idxs])
        # np.gradient gives central differences interior, one-sided at the ends,
        # and handles the non-uniform phase spacing of the dense grid.
        slope = np.gradient(mag, phase)
        for local_j, i in enumerate(idxs):
            s = slope[local_j]
            if s < -FLAT_SLOPE_EPS:
                labels[i] = "rising"
            elif s > FLAT_SLOPE_EPS:
                labels[i] = "fading"
            else:
                labels[i] = "flat"
    return labels


def _stats(residuals: np.ndarray) -> dict:
    """RMSE / MAE / bias / count for a residual array.

    Args:
        residuals (np.ndarray): pred - true (mag) values.

    Returns:
        dict: Keys n, rmse, mae, bias (bias/rmse/mae NaN when empty).
    """
    n = residuals.shape[0]
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan")}
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "bias": float(np.mean(residuals)),
    }


def _print_block(title: str, by_label: dict) -> None:
    """Print an aligned rising/fading/flat stats block.

    Args:
        title (str): Section header.
        by_label (dict): label -> stats dict from :func:`_stats`.
    """
    print(f"\n{title}")
    print(f"  {'label':<8}{'n':>7}{'RMSE':>10}{'MAE':>10}{'bias':>10}")
    total = sum(s["n"] for s in by_label.values())
    for label in ("rising", "flat", "fading"):
        s = by_label.get(label, {"n": 0, "rmse": float("nan"),
                                  "mae": float("nan"), "bias": float("nan")})
        frac = (100.0 * s["n"] / total) if total else 0.0
        print(f"  {label:<8}{s['n']:>7}{s['rmse']:>10.4f}{s['mae']:>10.4f}"
              f"{s['bias']:>+10.4f}   ({frac:4.1f}% of pts)")


def main() -> None:
    """Parse args, label points, and report rising-vs-fading residual stats."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="latetime_scored_points.csv path")
    p.add_argument("--plot", default=None, help="optional PNG path for a per-band "
                   "rising-vs-fading RMSE bar chart")
    args = p.parse_args()

    rows = _load_rows(args.csv)
    labels = _label_by_truth_slope(rows)
    resid = np.array([r["residual"] for r in rows])
    bands = np.array([r["band"] for r in rows], dtype=object)

    print(f"Loaded {len(rows)} scored points from {args.csv}")
    print(f"Slope threshold for rising/fading: |d(mag)/d(phase)| > "
          f"{FLAT_SLOPE_EPS} mag/day (else flat)")

    # Overall split.
    overall = {lab: _stats(resid[labels == lab])
               for lab in ("rising", "flat", "fading")}
    _print_block("OVERALL (all bands pooled)", overall)

    # How much of the total squared error lives in each label -- the number that
    # decides whether chasing the rising tail can move the pooled RMSE.
    sse = {lab: float(np.sum(resid[labels == lab] ** 2))
           for lab in ("rising", "flat", "fading")}
    tot_sse = sum(sse.values())
    print("\nShare of total squared error (drives the pooled RMSE):")
    for lab in ("rising", "flat", "fading"):
        share = (100.0 * sse[lab] / tot_sse) if tot_sse else 0.0
        print(f"  {lab:<8}{share:5.1f}%")

    # Per band.
    print("\n" + "=" * 60)
    print("PER BAND")
    for band in sorted(set(bands)):
        m = bands == band
        by_label = {lab: _stats(resid[m & (labels == lab)])
                    for lab in ("rising", "flat", "fading")}
        _print_block(f"band {band}", by_label)

    if args.plot is not None:
        _write_plot(args.plot, bands, labels, resid)


def _write_plot(path: str, bands: np.ndarray, labels: np.ndarray,
                resid: np.ndarray) -> None:
    """Write a grouped per-band rising/fading/flat RMSE bar chart.

    Args:
        path (str): Output PNG path.
        bands (np.ndarray): Per-point band names.
        labels (np.ndarray): Per-point rising/fading/flat labels.
        resid (np.ndarray): Per-point residuals (mag).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_list = sorted(set(bands))
    label_list = ("rising", "flat", "fading")
    colors = {"rising": "#1f77b4", "flat": "#999999", "fading": "#d62728"}
    width = 0.26
    x = np.arange(len(band_list))

    fig, ax = plt.subplots(figsize=(11, 5))
    for k, lab in enumerate(label_list):
        rmses = []
        for band in band_list:
            m = (bands == band) & (labels == lab)
            rmses.append(_stats(resid[m])["rmse"])
        ax.bar(x + (k - 1) * width, rmses, width, label=lab, color=colors[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(band_list, rotation=45, ha="right")
    ax.set_ylabel("late-time RMSE (mag)")
    ax.set_title("Rising vs fading residuals by band")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
