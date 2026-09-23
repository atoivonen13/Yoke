"""Horizon-EXACT refinement of the ztf dense-support diagnostic.

The first pass (diagnose_ztf_dense_support.py) measured the NEXT-EVENT lead and
so ignored horizon-covering target sampling (TARGET_HORIZON_DAYS=8), which in
training REDRAWS the supervised target to a farther event from a FIXED anchor.
That made its "median lead 0.11 d" an underestimate. This script replicates the
trainer's `_draw_target_idx` EXACTLY and computes, for every faint-late ztfg
target, the TRUE probability-weighted distribution of the (anchor, lead, context)
it is actually trained under. It answers the one question the build hinges on:

  Are faint-late ztfg targets reached ONLY by short leads from recent (bright,
  ztfg-present) context -- so the hard "realistic-sparse-context -> faint-late
  ztfg" mapping is genuinely absent from training -- OR is there already
  meaningful training weight on LONG leads from early bright anchors (which would
  teach that mapping and undercut the proposed build)?

EXACT mechanics mirrored from kilonova_dataset._getitem_window / _draw_target_idx:
  * Samples are enumerated one per anchor: anchor_idx = target_idx-1 for
    target_idx in 1..N-1, i.e. anchor i in 0..N-2, each exactly ONCE.
  * For a sample with anchor i, the target is redrawn: lead ~ Uniform(0, H) with
    H = TARGET_HORIZON_DAYS, cand = events i+1..N-1, chosen = argmin|gap - lead|
    where gap_k = t_k - t_i. So the chosen target is the nearest-gap event to a
    uniform lead draw -> a 1-D Voronoi partition of [0, H] over candidate gaps.
  * Therefore P(target = j | anchor = i) = (width of j's Voronoi cell on [0,H]) / H,
    with the FIRST candidate's cell starting at 0 and the LAST candidate's cell
    extending to H (argmin sends any lead beyond the max gap to the last event).
  * Expected #times event j is trained as target = sum_{i<j} P(j | i). We weight
    each faint-late target j's (anchor i, lead t_j - t_i, context@i) by P(j | i).

Context is read from the DENSE stream (dense-set samples: context and target both
dense), the trailing CONTEXT_WINDOW_DAYS window ending at anchor i, exactly as the
concatenated dense training samples see it.

Run on the cluster:
  python diagnose_ztf_horizon_exact.py \
     --train_filelist /net/.../filelists/kn_rubin_ztf_train.txt [--limit 2000]
"""

import argparse
import os

import numpy as np

# Reuse the validated stream reader + constants from the first diagnostic.
from diagnose_ztf_dense_support import (
    DEFAULT_DENSE,
    DEFAULT_REAL,
    FAINT_ZTFG_MAG,
    LATE_CUTOFF_D,
    LATE_MAX_D,
    ZTFG,
    first_realistic_detection_time,
    read_stream,
)

CONTEXT_WINDOW_DAYS = 2.0
TARGET_HORIZON_DAYS = 8.0  # H; matches the champion config


def selection_probs_for_target(times, j, H):
    """P(target=j | anchor=i) for every anchor i in 0..j-1, as a length-j array.

    Uses the exact Voronoi-cell width of event j on the lead axis [0,H] for the
    candidate set {i+1,...,N-1}, per anchor i. Vectorized over i.
    """
    N = times.shape[0]
    i = np.arange(0, j)  # anchors that can reach j (i < j)
    ti = times[i]

    # Lower cell boundary (in lead units) for j given anchor i:
    #   if j is the FIRST candidate (i == j-1): 0
    #   else midpoint between t[j-1] and t[j], shifted by -t[i].
    lower = np.where(
        i == j - 1,
        0.0,
        0.5 * (times[j - 1] + times[j]) - ti,
    )
    # Upper cell boundary:
    #   if j is the LAST event (j == N-1): H (captures all larger leads)
    #   else midpoint between t[j] and t[j+1], shifted by -t[i].
    if j == N - 1:
        upper = np.full(i.shape, H, dtype=np.float64)
    else:
        upper = 0.5 * (times[j] + times[j + 1]) - ti

    lo = np.clip(lower, 0.0, H)
    hi = np.clip(upper, 0.0, H)
    width = np.clip(hi - lo, 0.0, None)
    return i, width / H  # probability mass per anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realistic_dir", default=DEFAULT_REAL)
    ap.add_argument("--dense_dir", default=DEFAULT_DENSE)
    ap.add_argument("--train_filelist", default="kn_rubin_ztf_train.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=TARGET_HORIZON_DAYS)
    ap.add_argument("--outfile", default="ztf_horizon_exact_report.csv")
    args = ap.parse_args()
    H = args.horizon

    with open(args.train_filelist) as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    if args.limit:
        stems = stems[: args.limit]

    # Probability-weighted accumulators over faint-late ztfg targets.
    w_total = 0.0
    lead_w, anchorphase_w = [], []          # (value, weight) via parallel arrays
    lead_vals, anchor_vals, wts = [], [], []
    w_ctx_has_ztfg = 0.0
    w_ctx_bright = 0.0                       # ctx has ztfg brighter than floor
    w_ctx_empty = 0.0
    # Long-lead mass: the training weight that ALREADY covers the hard mapping.
    w_lead_ge_2 = 0.0
    w_lead_ge_4 = 0.0
    n_targets = 0

    rows = []
    n_obj = 0

    for stem in stems:
        dense_path = os.path.join(args.dense_dir, f"{stem}.npz")
        if not os.path.exists(dense_path):
            continue
        dense = read_stream(dense_path, drop_ul=False)
        if dense is None:
            continue
        t0 = first_realistic_detection_time(args.realistic_dir, stem)
        if t0 is None:
            continue
        n_obj += 1
        d_t, d_m, d_b = dense
        N = d_t.shape[0]
        if N < 2:
            continue

        for j in range(1, N):
            if d_b[j] != ZTFG:
                continue
            phase = d_t[j] - t0
            if not (LATE_CUTOFF_D < phase <= LATE_MAX_D):
                continue
            if d_m[j] <= FAINT_ZTFG_MAG:  # faint only
                continue
            n_targets += 1

            anchors, probs = selection_probs_for_target(d_t, j, H)
            for i, p in zip(anchors, probs):
                if p <= 0.0:
                    continue
                lead = d_t[j] - d_t[i]
                anchor_phase = d_t[i] - t0
                w_total += p
                lead_vals.append(lead)
                anchor_vals.append(anchor_phase)
                wts.append(p)
                if lead >= 2.0:
                    w_lead_ge_2 += p
                if lead >= 4.0:
                    w_lead_ge_4 += p

                # Context window [t_i - CW, t_i] on the dense stream.
                lo = d_t[i] - CONTEXT_WINDOW_DAYS
                win = (d_t <= d_t[i]) & (d_t >= lo)
                wb = d_b[win]
                if not win.any():
                    w_ctx_empty += p
                elif np.any(wb == ZTFG):
                    w_ctx_has_ztfg += p
                    if np.min(d_m[win][wb == ZTFG]) < FAINT_ZTFG_MAG:
                        w_ctx_bright += p

            rows.append((stem, f"{phase:.3f}", f"{d_m[j]:.3f}",
                         f"{float(np.sum(probs)):.4f}"))

    lead_vals = np.array(lead_vals)
    anchor_vals = np.array(anchor_vals)
    wts = np.array(wts)

    def wpct(x):
        return 100.0 * x / w_total if w_total else 0.0

    def wquantile(v, w, q):
        if v.size == 0:
            return float("nan")
        order = np.argsort(v)
        v, w = v[order], w[order]
        cw = np.cumsum(w)
        cw /= cw[-1]
        return float(np.interp(q, cw, v))

    print("=" * 70)
    print("ZTF HORIZON-EXACT DIAGNOSTIC  (H = %.1f d)" % H)
    print("=" * 70)
    print(f"objects scanned:                 {n_obj}")
    print(f"faint-late ztfg targets:         {n_targets}")
    print(f"total training weight on them:   {w_total:.1f} "
          "(expected # supervised samples)")
    print()
    print("TRUE probability-weighted training distribution for faint-late ztfg:")
    print(f"  lead time (d):    p10 {wquantile(lead_vals,wts,0.10):.2f}  "
          f"median {wquantile(lead_vals,wts,0.50):.2f}  "
          f"p90 {wquantile(lead_vals,wts,0.90):.2f}")
    print(f"  anchor phase (d): p10 {wquantile(anchor_vals,wts,0.10):.2f}  "
          f"median {wquantile(anchor_vals,wts,0.50):.2f}  "
          f"p90 {wquantile(anchor_vals,wts,0.90):.2f}")
    print()
    print("Long-lead mass (does training ALREADY cover the hard mapping?):")
    print(f"  weight with lead >= 2 d:   {wpct(w_lead_ge_2):.1f}%")
    print(f"  weight with lead >= 4 d:   {wpct(w_lead_ge_4):.1f}%")
    print()
    print("Context at the anchor (dense-stream trailing window):")
    print(f"  ctx has >=1 ztfg:          {wpct(w_ctx_has_ztfg):.1f}% of weight")
    print(f"  ctx has BRIGHT ztfg(<21):  {wpct(w_ctx_bright):.1f}%")
    print(f"  ctx empty:                 {wpct(w_ctx_empty):.1f}%")
    print("=" * 70)
    print("READ: if long-lead mass is SMALL and context is mostly bright-ztfg,")
    print("the hard realistic-sparse-context->faint-late-ztfg mapping is genuinely")
    print("absent from training -> the realistic-context/dense-target build is")
    print("justified. If long-lead mass is LARGE, training already sees long leads")
    print("(from bright dense context) and the gap is context-content, not lead.")

    with open(args.outfile, "w") as fh:
        fh.write("stem,target_phase_d,target_mag,total_select_weight\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {len(rows)} target rows to {args.outfile}")


if __name__ == "__main__":
    main()
