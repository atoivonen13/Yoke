"""Audit: would a censored one-sided (hinge) loss on upper limits actually bite?

Study 113 tried admitting upper limits (ULs) as a flagged CONTEXT feature and
REGRESSED (1.5199 vs 111's 1.3800), hurting the target band ztfg worst. The
lesson: a UL is a CONSTRAINT ("the true source is fainter than the limiting
magnitude L"), not a feature -- it belongs in the LOSS, not the input.

The proposed follow-up (Option 3) is a censored one-sided pinball term on the
median prediction:

    L_ul = relu(z_L - z_pred)          # normalized space, fainter = larger z

which is ZERO when the prediction already respects the bound (pred fainter than
L) and penalizes only violations (pred brighter than L). Crucially, such a term
has EXACTLY ZERO gradient wherever the model already plateaus fainter than the
limit. So before spending another full training run, this script measures --
per band -- what fraction of late-time ULs the champion model actually VIOLATES
(predicts brighter than the limit), and by how much. That fraction is an upper
bound on where the censored loss can do any work at all.

Method (mirrors eval_dense_latetime_9band.eval_object's context construction so
the prediction is apples-to-apples with the champion eval):
  1. Read the realistic stream KEEPING upper limits.
  2. Build the model context from realistic DETECTIONS with phase <= cutoff
     (identical to the deployed eval; detections-only, champion-parity).
  3. For every UL with phase > cutoff (a late-time bound -- exactly what the
     censored loss would supervise as a target), forecast the model's median
     prediction at that UL's band and lead time.
  4. Compare pred_mag vs limit_mag. VIOLATION = pred brighter than limit
     (pred_mag < limit_mag) -> the hinge would bite, pushing the forecast
     fainter. Report per-band violation rate, median violation depth, and how
     often the violation direction agrees with the band's known bias sign.

This is a READ-ONLY diagnostic: it loads the trained checkpoint, forecasts, and
prints/writes a per-band table. It never trains and writes no model state.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

from yoke.datasets.kilonova_dataset import (
    EPS,
    NINE_BAND_KEYS,
    load_or_compute_band_normalization,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pred_diagnostics_9band import (  # noqa: E402
    _select_window,
    build_context_input,
    load_9band_model,
)
from eval_dense_latetime_9band import (  # noqa: E402
    _batched_forward,
    _stem_to_path,
    read_merged_stream,
    study_tag,
)

BAND_KEYS = NINE_BAND_KEYS
BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
VALUE_COL = 1
ERROR_COL = 2
N_BANDS = len(BAND_KEYS)


def audit_object(
    real_stream,
    model,
    device,
    means,
    stds,
    context_window_days,
    max_context_len,
    late_time_cutoff_days,
    late_time_max_days,
):
    """Return per-UL audit rows for one object, or None if not auditable.

    Each row: (band_idx, phase, lead_time, limit_mag, pred_mag). The context is
    the pre-cutoff realistic DETECTIONS (champion-parity); the scored ULs are the
    late-time (phase > cutoff, <= max_days) non-detections.
    """
    r_t, r_v, r_b, r_ul = real_stream
    if r_t.shape[0] < 1:
        return None

    t0 = float(r_t[0])
    phase = r_t - t0

    is_det = r_ul < 0.5
    is_ul = r_ul > 0.5

    # Context: detections up to the cutoff (identical to the deployed eval).
    ctx_mask = is_det & (phase <= late_time_cutoff_days)
    if not np.any(ctx_mask):
        return None
    c_t, c_v, c_b = r_t[ctx_mask], r_v[ctx_mask], r_b[ctx_mask]
    last_ctx_t = float(c_t[-1])

    # Late-time ULs: the bounds the censored loss would supervise.
    ul_mask = is_ul & (phase > late_time_cutoff_days) & (phase <= late_time_max_days)
    if not np.any(ul_mask):
        return None
    ul_t, ul_lim, ul_b = r_t[ul_mask], r_v[ul_mask], r_b[ul_mask]
    lead_times = (ul_t - last_ctx_t).astype(np.float32)
    keep = lead_times > 0
    if not np.any(keep):
        return None
    ul_t, ul_lim, ul_b, lead_times = (
        ul_t[keep], ul_lim[keep], ul_b[keep], lead_times[keep],
    )

    # Build the champion-parity context input (detections only -> no UL channel).
    c_v_norm = (c_v - means[c_b]) / (stds[c_b] + EPS)
    win_v, win_t, win_b = _select_window(
        ctx_t=list(c_t.astype(np.float32)),
        ctx_v=list(c_v_norm.astype(np.float32)),
        ctx_b=list(c_b),
        context_window_days=context_window_days,
        max_context_len=max_context_len,
    )
    x = build_context_input(
        win_v=win_v,
        win_t=win_t,
        win_b=win_b,
        context_len=max_context_len,
        n_bands=N_BANDS,
        device=device,
        window_mode=True,
        phase_fourier_bands=getattr(model, "phase_fourier_bands", 0),
        phase0=t0,
    )

    # Forecast all bands at each UL's lead time; take the band the UL belongs to.
    preds_norm = _batched_forward(model, x, lead_times, device)  # [P, N_BANDS]
    pred_norm_band = preds_norm[np.arange(preds_norm.shape[0]), ul_b]
    pred_mag = pred_norm_band * (stds[ul_b] + EPS) + means[ul_b]

    rows = []
    for k in range(ul_b.shape[0]):
        rows.append(
            (
                int(ul_b[k]),
                float(ul_t[k] - t0),
                float(lead_times[k]),
                float(ul_lim[k]),
                float(pred_mag[k]),
            )
        )
    return rows


def get_args():
    """Parse command-line arguments (mirrors eval_dense_latetime_9band)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=int, default=111)
    p.add_argument("--epoch", type=int, default=500)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument(
        "--realistic_glob",
        type=str,
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
    )
    p.add_argument("--test_filelist", type=str, default=None)
    p.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_9band_norm_stats_trainonly.npz",
    )
    p.add_argument("--late_time_cutoff_days", type=float, default=2.0)
    p.add_argument("--late_time_max_days", type=float, default=10.0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--max_objects", type=int, default=0)
    return p.parse_args()


def main():
    """Run the upper-limit censoring audit."""
    args = get_args()
    tag = study_tag(args.study)
    if args.ckpt is None:
        args.ckpt = (
            f"runs/study_{tag}/study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )
    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/ul_censoring_audit"
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        model,
        context_len,
        n_bands,
        context_window_days,
        max_context_len,
    ) = load_9band_model(args.ckpt, device, use_ema=getattr(args, "use_ema", False))
    if context_window_days is None:
        raise ValueError("Audit requires a time-window checkpoint.")

    means, stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=True,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)

    real_map = _stem_to_path(args.realistic_glob)
    stems = sorted(real_map)
    if args.test_filelist is not None:
        with open(args.test_filelist) as fh:
            test_stems = {line.strip() for line in fh if line.strip()}
        stems = [s for s in stems if s in test_stems]
        print(f"Restricted to {len(stems)} test-split objects.")
    if args.max_objects > 0:
        stems = stems[: args.max_objects]
    print(f"Auditing {len(stems)} objects for late-time upper-limit censoring.")

    all_rows = []
    n_obj = 0
    for stem in stems:
        # KEEP ULs (drop_upper_limits=False) so the non-detections survive.
        real_stream = read_merged_stream(real_map[stem], drop_upper_limits=False)
        rows = audit_object(
            real_stream,
            model,
            device,
            means,
            stds,
            context_window_days,
            max_context_len,
            args.late_time_cutoff_days,
            args.late_time_max_days,
        )
        if rows is None:
            continue
        n_obj += 1
        for r in rows:
            all_rows.append((stem, *r))

    if not all_rows:
        print("No late-time upper limits found in any object; nothing to audit.")
        return

    arr_band = np.array([r[1] for r in all_rows])
    arr_limit = np.array([r[4] for r in all_rows])
    arr_pred = np.array([r[5] for r in all_rows])
    # Violation = model predicts BRIGHTER than the limit (pred_mag < limit_mag);
    # the hinge relu(z_L - z_pred) is > 0 exactly here (fainter = larger z/mag).
    violation = arr_pred < arr_limit
    depth = arr_limit - arr_pred  # > 0 iff violation; how far the hinge pushes

    csv_path = os.path.join(args.outdir, "ul_censoring_audit_points.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["stem", "band", "phase_days", "lead_days", "limit_mag", "pred_mag",
             "violation", "violation_depth_mag"]
        )
        for i, r in enumerate(all_rows):
            w.writerow(
                [r[0], BAND_NAMES[r[1]], f"{r[2]:.4f}", f"{r[3]:.4f}",
                 f"{r[4]:.4f}", f"{r[5]:.4f}", int(violation[i]),
                 f"{depth[i]:.4f}"]
            )

    print(f"\nAudited {n_obj} objects; {len(all_rows)} late-time upper limits.")
    print(
        f"Cutoff {args.late_time_cutoff_days} d < phase <= "
        f"{args.late_time_max_days} d.\n"
    )
    print(
        "VIOLATION = model predicts BRIGHTER than the limit -> the censored hinge "
        "would bite (push the forecast fainter). Non-violations already satisfy "
        "the bound -> exactly zero gradient.\n"
    )
    header = (
        f"{'band':>6} {'n_UL':>6} {'n_viol':>7} {'viol%':>7} "
        f"{'med_depth':>10} {'mean_depth':>11} {'max_depth':>10}"
    )
    print(header)
    print("-" * len(header))
    for b in range(N_BANDS):
        sel = arr_band == b
        n = int(sel.sum())
        if n == 0:
            print(f"{BAND_NAMES[b]:>6} {0:>6} {0:>7} {'--':>7} "
                  f"{'--':>10} {'--':>11} {'--':>10}")
            continue
        v = violation[sel]
        nv = int(v.sum())
        vdepth = depth[sel][v]
        med = float(np.median(vdepth)) if nv else 0.0
        mean = float(vdepth.mean()) if nv else 0.0
        mx = float(vdepth.max()) if nv else 0.0
        print(
            f"{BAND_NAMES[b]:>6} {n:>6} {nv:>7} {100.0 * nv / n:>6.1f}% "
            f"{med:>10.3f} {mean:>11.3f} {mx:>10.3f}"
        )
    # Overall
    nv_all = int(violation.sum())
    print("-" * len(header))
    print(
        f"{'ALL':>6} {len(all_rows):>6} {nv_all:>7} "
        f"{100.0 * nv_all / len(all_rows):>6.1f}% "
        f"{np.median(depth[violation]) if nv_all else 0.0:>10.3f} "
        f"{depth[violation].mean() if nv_all else 0.0:>11.3f} "
        f"{depth[violation].max() if nv_all else 0.0:>10.3f}"
    )
    print(f"\nWrote per-point audit to {csv_path}")
    print(
        "\nRead: a band with a HIGH violation% and a meaningful median depth is "
        "where the censored loss can help. A band near 0% (the model already "
        "plateaus fainter than its limits) will get ~zero gradient -- the loss "
        "cannot fix it. If ztfg (the study-113 target) is near 0%, its shallow "
        "ZTF non-detections sit brighter than the plateau and censoring won't "
        "touch it; deep PS1 u/g/z ULs are the likely beneficiaries."
    )


if __name__ == "__main__":
    main()
