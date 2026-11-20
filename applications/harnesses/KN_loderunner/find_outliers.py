"""List the objects behind the too-bright tracks in the pred_vs_truth figure.

The thin streaks below the 1:1 line (bottom right of every panel) are points
whose median forecast is several magnitudes brighter than the uniform truth.
This groups those points by object, so a streak that is one object shows up as
one row with many points in many bands.

Reads the same files as ``plot_run_report.py`` (no model / GPU / torch).
Residual = pred - true, so the streaks are residual < -threshold.

    python find_outliers.py --study 218
    python find_outliers.py --study 218 --threshold 2.5 --top 20

The printed stems can be drawn with
``plot_run_report.py --study 218 --example_stems <stems>``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from plot_run_report import BAND_NAMES, load_run


def outlier_objects(uniform: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per object with at least one point brighter than truth by ``threshold``."""
    df = uniform.copy()
    df["out"] = df["r"] < -threshold
    rows = []
    for stem, g in df.groupby("stem"):
        out = g[g["out"]]
        if out.empty:
            continue
        rows.append({
            "stem": stem,
            "n_out": len(out),
            "frac_out": len(out) / len(g),
            "bands_out": ",".join(b for b in BAND_NAMES if b in set(out["band"])),
            "n_bands_out": out["band"].nunique(),
            "median_r_out": out["r"].median(),
            "min_r": g["r"].min(),
            "obj_rmse": float(np.sqrt(np.mean(g["r"] ** 2))),
            "obj_bias": g["r"].mean(),
            "lead_out": f"{out['lead'].min():.1f}-{out['lead'].max():.1f}",
            "phase_out": f"{out['phase'].min():.1f}-{out['phase'].max():.1f}",
            "true_out": f"{out['true'].min():.1f}-{out['true'].max():.1f}",
            "outside_cal": float((~out["hit_cal"]).mean()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_out", ascending=False).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=int, required=True)
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--eval_dir", default=None)
    p.add_argument("--threshold", type=float, default=3.0,
                   help="Flag points with pred - true below -threshold (mag).")
    p.add_argument("--top", type=int, default=15, help="Rows to print.")
    p.add_argument("--out_csv", default=None,
                   help="Default: <eval_dir>/outliers_r<threshold>.csv")
    args = p.parse_args()

    run = load_run(args.study, args.runs_root, eval_dir=args.eval_dir)
    uniform = run["uniform"]
    table = outlier_objects(uniform, args.threshold)
    n_pts = int((uniform["r"] < -args.threshold).sum())
    print(f"\n{n_pts} of {len(uniform)} points ({n_pts / len(uniform):.2%}) have "
          f"pred - true < -{args.threshold:g} mag, from {len(table)} of "
          f"{uniform['stem'].nunique()} objects.")
    if table.empty:
        return
    share = table["n_out"].cumsum() / n_pts
    for k in (1, 3, 5, 10):
        if k <= len(table):
            print(f"  top {k:>2} objects hold {share.iloc[k - 1]:.0%} of them")

    # Effect on the headline: overall uniform RMSE with the top objects removed.
    rmse = float(np.sqrt(np.mean(uniform["r"] ** 2)))
    print(f"  uniform RMSE {rmse:.4f}", end="")
    for k in (1, 3, 5, 10):
        if k <= len(table):
            keep = ~uniform["stem"].isin(table["stem"].iloc[:k])
            print(f"; without top {k}: {np.sqrt(np.mean(uniform.loc[keep, 'r'] ** 2)):.4f}",
                  end="")
    print("\n")

    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(table.head(args.top).to_string())

    out_csv = args.out_csv or os.path.join(
        run["eval_dir"], f"outliers_r{args.threshold:g}.csv")
    table.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    stems = " ".join(table["stem"].head(min(args.top, 5)).astype(str))
    print(f"plot them: python plot_run_report.py --study {args.study} "
          f"--example_stems {stems}")


if __name__ == "__main__":
    main()
