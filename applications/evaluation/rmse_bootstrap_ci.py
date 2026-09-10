"""Bootstrap confidence interval for the late-time RMSE over objects.

Answers "is N objects enough?" for ``latetime_scored_points.csv``. The scored
points are NOT independent -- points within one object (and band track) share a
context and an underlying curve -- so the naive RMSE / sqrt(n_points) badly
understates the uncertainty. The correct unit of resampling is the OBJECT: draw
n_objects objects with replacement, pool all their points, recompute RMSE, repeat.
The spread of the bootstrap RMSEs is the sampling uncertainty on the reported
number, respecting both the within-object correlation and the heavy residual tail.

Compare the resulting CI half-width against the run-to-run seed noise (~0.05-0.07
mag here): if the object-sampling CI is comparable or larger, N objects is the
binding constraint on resolving lever differences and should be increased.

No model / GPU / torch. Run where the CSV lives:

    python rmse_bootstrap_ci.py \\
        --csv runs/study_080/dense_latetime_eval_9band/latetime_scored_points.csv

``--n_boot`` sets the resample count (default 2000); ``--seed`` fixes the draw.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def _load_by_object(csv_path: str) -> tuple[list, np.ndarray]:
    """Group squared residuals by object stem.

    Args:
        csv_path (str): Path to ``latetime_scored_points.csv``.

    Returns:
        tuple[list, np.ndarray]: (list of per-object squared-residual arrays,
        flat array of all squared residuals). One list entry per unique stem.
    """
    per_obj = defaultdict(list)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            per_obj[r["stem"]].append(float(r["residual_mag"]) ** 2)
    obj_sq = [np.array(v) for v in per_obj.values()]
    all_sq = np.concatenate(obj_sq) if obj_sq else np.array([])
    return obj_sq, all_sq


def _boot_rmse(obj_sq: list, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap RMSE distribution by resampling objects with replacement.

    Each replicate draws len(obj_sq) objects with replacement, pools their
    squared residuals, and takes sqrt(mean). Point counts vary per replicate
    (objects have different numbers of points), exactly as real resampling would.

    Args:
        obj_sq (list): Per-object arrays of squared residuals.
        n_boot (int): Number of bootstrap replicates.
        rng (np.random.Generator): Seeded RNG.

    Returns:
        np.ndarray: ``n_boot`` bootstrap RMSE values.
    """
    n_obj = len(obj_sq)
    idx_all = np.arange(n_obj)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx_all, size=n_obj, replace=True)
        pooled = np.concatenate([obj_sq[i] for i in pick])
        out[b] = np.sqrt(np.mean(pooled))
    return out


def main() -> None:
    """Parse args and report the point RMSE with a bootstrap 95% CI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="latetime_scored_points.csv path")
    p.add_argument("--n_boot", type=int, default=2000, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the draw")
    args = p.parse_args()

    obj_sq, all_sq = _load_by_object(args.csv)
    n_obj = len(obj_sq)
    n_pts = all_sq.shape[0]
    if n_obj == 0:
        raise SystemExit("No scored points found.")

    point_rmse = float(np.sqrt(np.mean(all_sq)))
    rng = np.random.default_rng(args.seed)
    boot = _boot_rmse(obj_sq, args.n_boot, rng)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    half = 0.5 * (hi - lo)
    print(f"Loaded {n_pts} points across {n_obj} objects from {args.csv}")
    print(f"Point-estimate late-time RMSE: {point_rmse:.4f}")
    print(f"Bootstrap over objects ({args.n_boot} replicates):")
    print(f"  mean {boot.mean():.4f}  std {boot.std():.4f}")
    print(f"  95% CI [{lo:.4f}, {hi:.4f}]  (half-width +/-{half:.4f})")
    print()
    print(f"Interpretation: differences between runs smaller than ~{half:.3f} mag "
          "are within the object-sampling noise of this eval set.")
    print("Compare that half-width to the ~0.05-0.07 mag seed-to-seed noise: the "
          "larger of the two is your real resolution limit.")


if __name__ == "__main__":
    main()
