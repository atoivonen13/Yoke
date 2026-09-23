"""Diagnose WHY the dense true-LC supervision does not fix the faint-ztf plateau.

The champion recipe (studies 123/125) already concatenates the DENSE companion
set (rubin_ztf_dense_..., no limiting-mag cut) onto the realistic training data
(train_LodeRunner_ddp.py ~1263), so faint late ztfg *targets* ARE in the loss.
Yet ztfg still plateaus in the dense late-time eval. This script tests the
hypothesis that the dense targets are supervised with the WRONG context:

  H: a dense faint-late-ztfg target is always paired (in training) with a
     contemporaneous, densely-sampled, still-bright trailing window, so the model
     learns "faint context near -> faint target". At EVAL the context is the
     REALISTIC stream, which is EMPTY/bright at those late leads (ZTF ~21 floor
     dropped the late detections), so the trained mapping never covers the hard
     bright-early-context -> faint-late-ztf forecast the metric demands.

It measures three things, all falsifiable, using ONLY the training-dataset window
logic (mirrored here so no repo import is needed) and the raw npz files:

 Q1. How many ztfg TARGETS in the dense TRAIN set fall in the scored late-time
     region (phase in (CUTOFF, MAX] d) and how faint are they? (support exists?)

 Q2. For each such dense faint-late-ztfg target, reconstruct the EXACT trailing
     context window the trainer would build (anchor = event before target, window
     = CONTEXT_WINDOW_DAYS before the anchor; horizon sampling only moves the
     target FARTHER from a FIXED anchor, so the anchor/context is well-defined by
     the enumerated (file, target_idx) pairs). Report:
       - anchor phase (days since first detection)
       - lead time = target_t - anchor_t
       - context: n events, n ztfg events, brightest/faintest ztfg mag in window
     If anchors cluster EARLY (small anchor phase) with SHORT leads and BRIGHT
     ztfg context, H is supported: the faint target is learned from easy context.

 Q3. Cross-check the eval mismatch: for the SAME objects, build the REALISTIC
     stream's trailing window at the anchor phase of each dense faint-late target
     and report how often it is EMPTY or has no ztfg (i.e. the context the model
     actually gets at deployment is nothing like the dense-training context).

Run on the cluster (data is not mounted locally):
  python diagnose_ztf_dense_support.py \
     --train_filelist kn_rubin_ztf_train.txt \
     [--limit 2000]
Prints a compact report; writes ztf_dense_support_report.csv for follow-up.
"""

import argparse
import glob
import json
import os

import numpy as np


# ---- constants mirrored from the training config / dataset -----------------
NINE_BAND_KEYS = (
    "arr_ztfg", "arr_ztfr", "arr_ztfi", "arr_sdssu",
    "arr_ps1__g", "arr_ps1__r", "arr_ps1__i", "arr_ps1__z", "arr_ps1__y",
)
ZTFG = 0  # band index of ztfg
ERROR_COL = 2  # non-finite error flags a non-detection (upper limit)
VALUE_COL = 1  # magnitude column
TIME_COL = 0

# Champion/eval knobs (train_LodeRunner_ddp.py + eval_dense_latetime_9band.py).
CONTEXT_WINDOW_DAYS = 2.0
LATE_CUTOFF_D = 2.0   # scored region lower bound (phase from first realistic det)
LATE_MAX_D = 10.0     # scored region upper bound
FAINT_ZTFG_MAG = 21.0  # ZTF detection floor; "faint" = truth fainter than this

DEFAULT_REAL = (
    "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
    "rubin_ztf_10000_dataset_same_seed"
)
DEFAULT_DENSE = (
    "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
    "rubin_ztf_dense_10000_dataset_same_seed"
)


def read_stream(npz_path, band_idx=None, drop_ul=False):
    """Return (times, mags, bands) sorted by time, file-relative.

    When band_idx is given, keep only that band. When drop_ul, drop rows with a
    non-finite error (upper limits), matching the realistic training stream.
    """
    data = np.load(npz_path, allow_pickle=True)
    times, mags, bands = [], [], []
    for bi, key in enumerate(NINE_BAND_KEYS):
        if band_idx is not None and bi != band_idx:
            continue
        if key not in data.files:
            continue
        arr = data[key]
        if arr.size == 0:
            continue
        if drop_ul:
            arr = arr[np.isfinite(arr[:, ERROR_COL])]
            if arr.shape[0] == 0:
                continue
        times.append(arr[:, TIME_COL].astype(np.float64))
        mags.append(arr[:, VALUE_COL].astype(np.float64))
        bands.append(np.full(arr.shape[0], bi, dtype=np.int64))
    data.close()
    if not times:
        return None
    t = np.concatenate(times)
    m = np.concatenate(mags)
    b = np.concatenate(bands)
    order = np.argsort(t, kind="stable")
    return t[order], m[order], b[order]


def first_realistic_detection_time(real_dir, stem):
    """Phase-zero clock = first realistic detection (the observed trigger).

    Returned in the DENSE file's relative frame is impossible (different min), so
    we work per-stream in ABSOLUTE time: read both raw (no min-subtraction) and
    align by absolute MJD, exactly as the eval does.
    """
    path = os.path.join(real_dir, f"{stem}.npz")
    if not os.path.exists(path):
        return None
    s = read_stream(path, drop_ul=True)
    if s is None:
        return None
    return float(s[0].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realistic_dir", default=DEFAULT_REAL)
    ap.add_argument("--dense_dir", default=DEFAULT_DENSE)
    ap.add_argument("--train_filelist", default="kn_rubin_ztf_train.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap objects for a quick pass (0 = all).")
    ap.add_argument("--outfile", default="ztf_dense_support_report.csv")
    args = ap.parse_args()

    with open(args.train_filelist) as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    if args.limit:
        stems = stems[: args.limit]

    # Accumulators.
    n_obj = 0
    n_obj_with_faint_late = 0
    q1_targets = 0            # dense ztfg targets in the scored late region
    q1_faint = 0             # ...of which truth fainter than the ZTF floor
    # For faint-late targets: training-context descriptors.
    anchor_phases, leads = [], []
    ctx_nztfg, ctx_bright_ztfg = [], []
    ctx_empty = 0
    # Eval-side realistic context at the same anchor phase.
    real_ctx_empty = 0
    real_ctx_no_ztfg = 0

    rows = []

    for stem in stems:
        dense_path = os.path.join(args.dense_dir, f"{stem}.npz")
        if not os.path.exists(dense_path):
            continue
        # ABSOLUTE-time streams (no min subtraction) so realistic & dense share a
        # clock, matching the eval's convention.
        dense = read_stream(dense_path, drop_ul=False)  # dense = all detections
        if dense is None:
            continue
        t0 = first_realistic_detection_time(args.realistic_dir, stem)
        if t0 is None:
            continue
        n_obj += 1
        d_t, d_m, d_b = dense

        # Realistic stream (context source at eval), absolute time, ULs dropped.
        real = read_stream(
            os.path.join(args.realistic_dir, f"{stem}.npz"), drop_ul=True
        )

        obj_has_faint_late = False

        # Enumerate dense ztfg events as candidate TARGETS (mirrors the trainer's
        # per-event target enumeration; band-filtered to ztfg here for the study).
        ztfg_mask = d_b == ZTFG
        ztfg_t = d_t[ztfg_mask]
        ztfg_m = d_m[ztfg_mask]
        for tt, mm in zip(ztfg_t, ztfg_m):
            phase = tt - t0
            if not (LATE_CUTOFF_D < phase <= LATE_MAX_D):
                continue
            q1_targets += 1
            is_faint = mm > FAINT_ZTFG_MAG
            if is_faint:
                q1_faint += 1
                obj_has_faint_late = True

                # ----- Q2: reconstruct the TRAINING context for this target -----
                # Anchor = the dense event immediately before this target in the
                # merged stream (all bands). Window = events within
                # CONTEXT_WINDOW_DAYS before the anchor.
                before = d_t < tt
                if not before.any():
                    ctx_empty += 1
                    continue
                anchor_i = np.nonzero(before)[0][-1]
                anchor_t = d_t[anchor_i]
                lo = anchor_t - CONTEXT_WINDOW_DAYS
                win = (d_t <= anchor_t) & (d_t >= lo)
                w_b = d_b[win]
                w_m = d_m[win]
                anchor_phases.append(anchor_t - t0)
                leads.append(tt - anchor_t)
                nzt = int(np.sum(w_b == ZTFG))
                ctx_nztfg.append(nzt)
                if nzt > 0:
                    ctx_bright_ztfg.append(float(np.min(w_m[w_b == ZTFG])))

                # ----- Q3: what does the EVAL realistic context look like here? --
                # Realistic trailing window at the SAME anchor phase.
                if real is None:
                    real_ctx_empty += 1
                else:
                    r_t, r_m, r_b = real
                    r_win = (r_t <= anchor_t) & (r_t >= lo)
                    if not r_win.any():
                        real_ctx_empty += 1
                    elif not np.any(r_b[r_win] == ZTFG):
                        real_ctx_no_ztfg += 1

                rows.append(
                    (stem, f"{phase:.3f}", f"{mm:.3f}",
                     f"{anchor_t - t0:.3f}", f"{tt - anchor_t:.3f}",
                     nzt,
                     f"{(np.min(w_m[w_b==ZTFG]) if nzt>0 else np.nan):.3f}")
                )

        if obj_has_faint_late:
            n_obj_with_faint_late += 1

    # ---------------------------- report ----------------------------------
    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    print("=" * 70)
    print("ZTF DENSE-SUPPORT DIAGNOSTIC")
    print("=" * 70)
    print(f"objects scanned:                 {n_obj}")
    print(f"objects w/ faint late ztfg:      {n_obj_with_faint_late} "
          f"({pct(n_obj_with_faint_late, n_obj):.1f}%)")
    print()
    print("Q1  Do faint late-ztfg TARGETS exist in dense training?")
    print(f"    ztfg targets in ({LATE_CUTOFF_D},{LATE_MAX_D}] d:   {q1_targets}")
    print(f"    ...fainter than {FAINT_ZTFG_MAG:.0f} mag (faint):   {q1_faint} "
          f"({pct(q1_faint, q1_targets):.1f}% of late ztfg targets)")
    print()
    print("Q2  What TRAINING context are the faint-late targets paired with?")
    if anchor_phases:
        ap_ = np.array(anchor_phases)
        ld_ = np.array(leads)
        nz_ = np.array(ctx_nztfg)
        print(f"    anchor phase (d):  median {np.median(ap_):.2f}  "
              f"p10 {np.percentile(ap_,10):.2f}  p90 {np.percentile(ap_,90):.2f}")
        print(f"    lead time  (d):    median {np.median(ld_):.2f}  "
              f"p10 {np.percentile(ld_,10):.2f}  p90 {np.percentile(ld_,90):.2f}")
        print(f"    ztfg events in ctx window: median {np.median(nz_):.1f}  "
              f"frac windows w/ >=1 ztfg {pct(np.sum(nz_>0), nz_.size):.1f}%")
        if ctx_bright_ztfg:
            bz = np.array(ctx_bright_ztfg)
            print(f"    brightest ztfg in ctx (mag): median {np.median(bz):.2f}  "
                  f"(faint floor {FAINT_ZTFG_MAG:.0f}); "
                  f"frac ctx BRIGHTER than floor {pct(np.sum(bz<FAINT_ZTFG_MAG), bz.size):.1f}%")
    else:
        print("    (no faint-late targets found)")
    print()
    print("Q3  At EVAL, what realistic context exists at those anchor phases?")
    if q1_faint:
        print(f"    realistic ctx EMPTY:        {real_ctx_empty} "
              f"({pct(real_ctx_empty, q1_faint):.1f}% of faint-late targets)")
        print(f"    realistic ctx has NO ztfg:  {real_ctx_no_ztfg} "
              f"({pct(real_ctx_no_ztfg, q1_faint):.1f}%)")
        print(f"    -> train/eval context mismatch on "
              f"{pct(real_ctx_empty + real_ctx_no_ztfg, q1_faint):.1f}% "
              "of faint-late ztfg targets")
    print("=" * 70)
    print("READ: if Q1 shows many faint targets, Q2 shows they are paired with")
    print("EARLY anchors / SHORT leads / BRIGHT dense ztfg context, and Q3 shows")
    print("the realistic context is mostly empty/no-ztfg there, the hypothesis")
    print("holds: dense TARGETS are present but supervised with easy contemporaneous")
    print("context, so they never teach the hard realistic-context late forecast.")

    with open(args.outfile, "w") as fh:
        fh.write("stem,target_phase_d,target_mag,anchor_phase_d,lead_d,"
                 "ctx_nztfg,ctx_brightest_ztfg_mag\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {len(rows)} faint-late-target rows to {args.outfile}")


if __name__ == "__main__":
    main()
