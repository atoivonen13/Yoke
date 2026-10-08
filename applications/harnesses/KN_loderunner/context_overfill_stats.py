"""How often does the trailing context window hold more than MAX_CONTEXT_LEN events?

Read-only diagnostic for choosing ``MAX_CONTEXT_LEN``. The window rule (dataset
``_getitem_window`` and eval ``_select_window``) keeps every detection within
``context_window_days`` of the anchor, then only the MOST RECENT
``max_context_len`` if more qualify -- so an over-full window silently drops its
OLDEST events (the rise). Raising the cap only matters if this happens often.

Three populations, each counted exactly as the pipeline builds it:

  * EVAL: per object, the realistic detections with phase <= ``--cutoff_days``
    from the first realistic detection, windowed from the last of them. This is
    the context behind the scored late-time RMSE (DIRECT path).
  * TRAIN realistic anchors: every realistic detection as an anchor, window
    ``[t_anchor - W, t_anchor]`` (the realistic-context parts: cross-stream, and
    realistic-target when enabled). One row per anchor == one training sample.
  * TRAIN dense anchors: the same over the dense stream (the dense-dense part's
    context). Dense sampling is far denser, so over-fill is expected to be common.

Upper limits are dropped (matches training / eval ``DROP_UPPER_LIMITS``). No
model / GPU needed. Run on the cluster, e.g.:

    python context_overfill_stats.py \\
        --test_filelist <test split stems>   # optional; restricts EVAL objects
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dense_latetime_9band import (  # noqa: E402
    _stem_to_path,
    read_merged_stream,
)


def window_counts(times: np.ndarray, window_days: float) -> np.ndarray:
    """In-window event count for every event used as an anchor.

    Args:
        times (np.ndarray): Sorted event times of one stream.
        window_days (float): Trailing lookback W.

    Returns:
        np.ndarray: For each anchor i, the number of events with
        ``times[i] - W <= t <= times[i]`` among events 0..i (ties at the anchor
        time that sort after it are excluded, as in the dataset).
    """
    lo = np.searchsorted(times, times - window_days, side="left")
    return np.arange(1, times.shape[0] + 1) - lo


def summarize(label: str, counts: np.ndarray, caps: list) -> None:
    """Print over-fill fraction, dropped-event share, and percentiles."""
    counts = np.asarray(counts)
    n = counts.shape[0]
    if n == 0:
        print(f"\n{label}: no samples")
        return
    pct = np.percentile(counts, [50, 75, 90, 95, 99])
    print(f"\n{label}: {n} samples")
    print("  in-window count  p50 {:.0f}  p75 {:.0f}  p90 {:.0f}  p95 {:.0f}  "
          "p99 {:.0f}  max {:d}".format(*pct, int(counts.max())))
    print(f"  {'cap':>5}{'% samples over cap':>21}{'% in-window events dropped':>29}")
    total = counts.sum()
    for cap in caps:
        over = np.mean(counts > cap)
        dropped = np.maximum(counts - cap, 0).sum() / total
        print(f"  {cap:>5}{100 * over:>20.1f}%{100 * dropped:>28.1f}%")


def main() -> None:
    """Parse args, walk both streams, and print the over-fill tables."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--realistic_glob",
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_lsst_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
    )
    p.add_argument(
        "--dense_glob",
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_lsst_ztf_dense_10000_dataset_same_seed/lc_*.npz"
        ),
    )
    p.add_argument("--test_filelist", default=None,
                   help="Restrict EVAL objects to these stems (one per line).")
    p.add_argument("--window_days", type=float, default=2.0,
                   help="CONTEXT_WINDOW_DAYS.")
    p.add_argument("--cutoff_days", type=float, default=2.0,
                   help="Eval --late_time_cutoff_days.")
    p.add_argument("--caps", type=int, nargs="+", default=[12, 16, 24, 32],
                   help="Candidate MAX_CONTEXT_LEN values.")
    p.add_argument("--max_objects", type=int, default=0,
                   help="Cap objects read (0 = all), for a quick look.")
    p.add_argument("--skip_dense_train", action="store_true",
                   help="Skip the (slow, large) dense-anchor pass.")
    args = p.parse_args()

    real_map = _stem_to_path(args.realistic_glob)
    dense_map = _stem_to_path(args.dense_glob)
    stems = sorted(set(real_map) & set(dense_map))
    test_stems = None
    if args.test_filelist is not None:
        with open(args.test_filelist) as fh:
            test_stems = {line.strip() for line in fh if line.strip()}
    if args.max_objects > 0:
        stems = stems[: args.max_objects]
    print(f"Paired objects: {len(stems)}"
          + (f" (EVAL restricted to {len(test_stems)} test stems)"
             if test_stems is not None else ""))

    eval_counts, real_anchor_counts, dense_anchor_counts = [], [], []
    for stem in stems:
        r_t = read_merged_stream(real_map[stem], drop_upper_limits=True)[0]
        if r_t.shape[0] == 0:
            continue
        real_anchor_counts.append(window_counts(r_t, args.window_days))

        if test_stems is None or stem in test_stems:
            ctx_t = r_t[(r_t - r_t[0]) <= args.cutoff_days]
            eval_counts.append(
                int(np.sum(ctx_t >= ctx_t[-1] - args.window_days))
            )

        if not args.skip_dense_train:
            d_t = read_merged_stream(dense_map[stem], drop_upper_limits=True)[0]
            if d_t.shape[0] > 0:
                dense_anchor_counts.append(window_counts(d_t, args.window_days))

    caps = sorted(args.caps)
    summarize(
        f"EVAL context (realistic, phase <= {args.cutoff_days} d, "
        f"W = {args.window_days} d) -- one per object",
        np.asarray(eval_counts), caps,
    )
    summarize(
        f"TRAIN realistic anchors (W = {args.window_days} d)",
        np.concatenate(real_anchor_counts) if real_anchor_counts else [], caps,
    )
    if not args.skip_dense_train:
        summarize(
            f"TRAIN dense anchors (dense-dense part, W = {args.window_days} d)",
            np.concatenate(dense_anchor_counts) if dense_anchor_counts else [],
            caps,
        )


if __name__ == "__main__":
    main()
