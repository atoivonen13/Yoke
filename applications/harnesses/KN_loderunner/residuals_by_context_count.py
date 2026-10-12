"""Split late-time RMSE by how many detections sit in each object's eval context.

Read-only diagnostic for the MAX_CONTEXT_LEN A/B (study 132 vs 129). For every
scored object, count the realistic detections the eval context window holds
(phase <= ``--cutoff_days`` from the first realistic detection, trailing
``--window_days`` window from the last of them -- the same rule as
``eval_dense_latetime_9band.eval_object`` before the cap is applied), then bin each
study's ``latetime_scored_points.csv`` residuals by that count.

It separates two readings of a cap-change regression:

  * If the regression sits in objects whose count exceeds the OLD cap (the only
    objects whose input actually changed), the new slots are the problem --
    e.g. the flattened MLP barely trained the high slot positions, since
    training windows rarely fill them.
  * If objects at or below the old cap regress too (their input is byte-identical
    up to extra zero padding), the whole fit got worse -- a training problem,
    not the new slots.

No model / GPU. Run on the cluster, e.g.:

    python residuals_by_context_count.py \\
        --csv 129=runs/study_129/dense_latetime_eval_9band/latetime_scored_points.csv \\
        --csv 132=runs/study_132/dense_latetime_eval_9band/latetime_scored_points.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dense_latetime_9band import (  # noqa: E402
    _stem_to_path,
    read_merged_stream,
)


def eval_context_count(npz_path: str, cutoff_days: float, window_days: float) -> int:
    """Uncapped in-window detection count of one object's eval context."""
    r_t = read_merged_stream(npz_path, drop_upper_limits=True)[0]
    if r_t.shape[0] == 0:
        return 0
    ctx_t = r_t[(r_t - r_t[0]) <= cutoff_days]
    return int(np.sum(ctx_t >= ctx_t[-1] - window_days))


def load_residuals(csv_path: str) -> dict:
    """Map stem -> array of residuals (pred - true) from a scored-points CSV."""
    out = defaultdict(list)
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["stem"]].append(float(r["residual_mag"]))
    return {s: np.asarray(v) for s, v in out.items()}


def main() -> None:
    """Parse args, count each object's context, and print RMSE per count bin."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", action="append", required=True,
                   help="LABEL=path to latetime_scored_points.csv (repeatable).")
    p.add_argument(
        "--realistic_glob",
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_lsst_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
    )
    p.add_argument("--cutoff_days", type=float, default=2.0)
    p.add_argument("--window_days", type=float, default=2.0)
    p.add_argument("--bins", type=int, nargs="+", default=[0, 8, 12, 15, 18, 100],
                   help="Count-bin edges; bin i is edges[i] < n <= edges[i+1].")
    args = p.parse_args()

    studies = []
    for item in args.csv:
        label, path = item.split("=", 1)
        studies.append((label, load_residuals(path)))

    stems = sorted(set.intersection(*(set(res) for _, res in studies)))
    real_map = _stem_to_path(args.realistic_glob)
    counts = {s: eval_context_count(real_map[s], args.cutoff_days, args.window_days)
              for s in stems if s in real_map}
    print(f"Objects scored in every CSV and found in the realistic glob: "
          f"{len(counts)}")

    edges = args.bins
    header = f"  {'ctx count':>10}{'objects':>9}{'points':>8}"
    for label, _ in studies:
        header += f"{'RMSE ' + label:>13}{'bias ' + label:>13}"
    if len(studies) == 2:
        header += f"{'dRMSE':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = [s for s, n in counts.items() if lo < n <= hi]
        if not in_bin:
            continue
        row = f"  {f'{lo + 1}-{hi}':>10}{len(in_bin):>9}"
        rmses, cells = [], []
        for _, res in studies:
            r = np.concatenate([res[s] for s in in_bin])
            rmses.append(float(np.sqrt(np.mean(r**2))))
            cells.append(f"{rmses[-1]:>13.4f}{float(np.mean(r)):>+13.4f}")
        row += f"{r.shape[0]:>8}" + "".join(cells)
        if len(rmses) == 2:
            row += f"{rmses[1] - rmses[0]:>+9.4f}"
        print(row)


if __name__ == "__main__":
    main()
