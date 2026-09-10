"""Calibration + coverage of the quantile-head 0.1/0.9 forecast bands.

Read-only diagnostic against ``latetime_scored_points.csv`` written by
``eval_dense_latetime_9band.py`` AFTER the quantile-emit change (columns
``pred_low`` / ``pred_high`` present). It answers the question the quantile run
was for: are the predicted 0.1-0.9 bands calibrated, and does their width match
the actual residual scatter (i.e. has the model learned the variance floor)?

Two numbers per band (and overall):

  * Coverage: fraction of true points with ``pred_low <= true <= pred_high``.
    A calibrated 0.1-0.9 band should cover ~0.80. Below that => overconfident
    (bands too narrow, true scatter escapes them); above => underconfident.
  * Band width vs residual RMSE: mean ``pred_high - pred_low`` compared to the
    residual RMSE. If a well-calibrated band's width ~ 2.56 * RMSE (the 0.1-0.9
    span of a Gaussian), the scatter is consistent with irreducible noise the
    model has correctly characterized -> the RMSE floor is real. A band far
    narrower than the residuals it must cover is the overconfidence signature.

No model / GPU / torch. Run where the CSV lives:

    python quantile_coverage.py \\
        --csv runs/study_080/dense_latetime_eval_9band/latetime_scored_points.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


# 0.1-0.9 span of a standard normal (z_0.9 - z_0.1); a calibrated Gaussian band
# has width ~ this * sigma, so width / GAUSS_1090_SPAN estimates the implied sigma
# to compare against the residual RMSE.
GAUSS_1090_SPAN = 2.5631


def _load(csv_path: str) -> dict:
    """Load residual, band, and quantile-band columns from the scored CSV.

    Args:
        csv_path (str): Path to ``latetime_scored_points.csv``.

    Returns:
        dict: Arrays keyed band, true, low, high, resid (all aligned).

    Raises:
        SystemExit: If the CSV lacks the ``pred_low`` / ``pred_high`` columns
            (i.e. it was written before the quantile-emit change, or by a point
            head where the columns collapse to the median).
    """
    bands, true, low, high, resid = [], [], [], [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "pred_low" not in reader.fieldnames or "pred_high" not in reader.fieldnames:
            raise SystemExit(
                "CSV has no pred_low/pred_high columns -- rerun eval with the "
                "quantile-emit change and a quantile-head checkpoint."
            )
        for r in reader:
            bands.append(r["band"])
            true.append(float(r["true_mag"]))
            low.append(float(r["pred_low"]))
            high.append(float(r["pred_high"]))
            resid.append(float(r["residual_mag"]))
    return {
        "band": np.array(bands, dtype=object),
        "true": np.array(true),
        "low": np.array(low),
        "high": np.array(high),
        "resid": np.array(resid),
    }


def _row(label: str, true: np.ndarray, low: np.ndarray, high: np.ndarray,
         resid: np.ndarray) -> str:
    """Format one coverage/width line for a band (or the pooled total).

    Args:
        label (str): Band name or "OVERALL".
        true (np.ndarray): True magnitudes.
        low (np.ndarray): Predicted 0.1 quantile.
        high (np.ndarray): Predicted 0.9 quantile.
        resid (np.ndarray): Residuals (median pred - true).

    Returns:
        str: Aligned summary row.
    """
    n = true.shape[0]
    if n == 0:
        return f"  {label:<8}{0:>7}  (no points)"
    # Bands are stored low <= high already (monotone head), but clip defensively.
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    covered = np.mean((true >= lo) & (true <= hi))
    width = np.mean(hi - lo)
    rmse = np.sqrt(np.mean(resid**2))
    implied_sigma = width / GAUSS_1090_SPAN
    # ratio > 1 => band wider than residuals need (underconfident); < 1 =>
    # narrower (overconfident); ~1 => width matches the scatter.
    ratio = implied_sigma / rmse if rmse > 0 else float("nan")
    return (f"  {label:<8}{n:>7}{covered:>10.3f}{width:>11.3f}{rmse:>11.3f}"
            f"{ratio:>11.2f}")


def main() -> None:
    """Parse args and print per-band + overall coverage/width calibration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="latetime_scored_points.csv path")
    args = p.parse_args()

    d = _load(args.csv)
    print(f"Loaded {d['true'].shape[0]} scored points from {args.csv}")
    print("Target coverage for a calibrated 0.1-0.9 band: 0.80")
    print("width/RMSE ratio ~1.0 => band width matches residual scatter "
          "(learned floor); <1 overconfident, >1 underconfident.\n")
    header = (f"  {'band':<8}{'n':>7}{'coverage':>10}{'width':>11}"
              f"{'RMSE':>11}{'sig/RMSE':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    print(_row("OVERALL", d["true"], d["low"], d["high"], d["resid"]))
    print()
    for band in sorted(set(d["band"])):
        m = d["band"] == band
        print(_row(band, d["true"][m], d["low"][m], d["high"][m], d["resid"][m]))


if __name__ == "__main__":
    main()
