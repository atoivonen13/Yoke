"""Pre-check for candidate (iii): a cross-band color-consistency AUX LOSS.

Why this exists
---------------
Studies 117 and 118 tried to fix ztfg's late-time plateau with an ARCHITECTURAL
color-coupled head (shared pivot + low-rank color code). Both REGRESSED. The
117+118 pair proved the blocker is the TRAINING SIGNAL, not the head shape: under
single-band supervision, no loss term ever coobserves ztfg and a deep band at the
same (object, phase), so "ztfg should track its neighbors" is never asked for.

Candidate (iii) puts that relationship into the loss: penalize the model's
predicted ztfg-vs-neighbor COLOR against a target color. But there is a trap the
original ceiling test (color_correlation_test.py) did NOT check:

    A color-consistency loss needs a color TARGET. At the phases where ztfg fails
    (faint, late, past ZTF's ~21 floor) there is NO ztfg truth to build that
    target from. The target must be learned where ztfg IS detectable (bright,
    early) and EXTRAPOLATED into the faint regime.

The original ceiling test fit its ridge on ALL rows -- including faint ztfg -- so
it had faint-ztfg truth to fit against (optimistic; unattainable in training).
This script re-runs the ceiling test the HONEST way and answers three questions
that together decide whether candidate (iii) can work AT ALL:

  Q1 COVERAGE: at faint-ztfg phases, is there >=1 deep band (z/i/y/...) with truth
     present AND above its own survey floor (a usable anchor)? No anchor -> dead.
  Q2 TRANSFER: fit the color/ztfg relation ONLY on rows where ztfg is BRIGHT
     (<= floor, i.e. detectable), then PREDICT ztfg on FAINT rows. Residual on the
     faint rows = the realistic recoverability of a target a real loss could build.
     Compare against (a) the optimistic all-rows ceiling and (b) predict-the-mean.
  Q3 PHASE EXTRAPOLATION: same, but the predictor is the color at an EARLIER phase
     carried forward (a smooth-color-evolution target), which is closer to what the
     forward model actually has. Reported as a stretch/robustness variant.

Decision rule
-------------
If the BRIGHT-fit -> FAINT-test residual (Q2) is well below the model's ztfg late
RMSE (~1.88) AND below predict-the-mean, AND coverage (Q1) is high -> candidate
(iii) has a constructible target -> BUILD IT. If it collapses toward the marginal
std (the color learned bright does NOT transfer faint), candidate (iii) is dead
too; the faint ztfg tail is genuinely information-limited and no in-model loss
recovers it -> pivot to the redshift lever / accept the floor.

Read-only. numpy-only (no repo imports; repo src needs py3.10 unions). Runs on the
cluster where the dense pool lives. Mirrors color_correlation_test.py conventions.

Usage
-----
    python aux_loss_precheck.py \
        --dense_glob '/path/to/dense/*.npz' \
        [--target ztfg] [--phase_min 0 --phase_max 10 --phase_step 0.5] \
        [--floor 20.8] [--n_folds 5] [--out aux_precheck]
"""

import argparse
import glob
import json
import os

import numpy as np

# Canonical band order + the npz array key for each (matches the eval + the
# original ceiling test).
BANDS = ["ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y"]
BAND_KEY = {
    "ztfg": "arr_ztfg",
    "ztfr": "arr_ztfr",
    "ztfi": "arr_ztfi",
    "u": "arr_sdssu",
    "g": "arr_ps1__g",
    "r": "arr_ps1__r",
    "i": "arr_ps1__i",
    "z": "arr_ps1__z",
    "y": "arr_ps1__y",
}
# Rough single-visit 5-sigma depths (mag). A band is a usable ANCHOR at a phase
# only if its truth is brighter than this. ztfg's floor defines "detectable".
SURVEY_FLOOR = {
    "ztfg": 20.8, "ztfr": 20.6, "ztfi": 19.9,
    "u": 23.5, "g": 24.5, "r": 24.0, "i": 23.5, "z": 22.5, "y": 21.5,
}


def load_object(path):
    """Return dict band -> (phase_array, mag_array) from one dense npz.

    Dense stream is noise-free truth; keep all finite-mag rows. Phase = MJD - t0.
    """
    d = np.load(path, allow_pickle=True)
    try:
        inj = json.loads(str(d["injection_parameters"][0]))
        t0 = float(inj["kilonova_trigger_time"])
        redshift = float(inj.get("redshift", np.nan))
    except Exception:
        return None
    out = {"redshift": redshift, "bands": {}}
    for b in BANDS:
        key = BAND_KEY[b]
        if key not in d.files:
            continue
        arr = np.asarray(d[key], dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        mjd, mag = arr[:, 0], arr[:, 1]
        good = np.isfinite(mjd) & np.isfinite(mag)
        if good.sum() < 2:
            continue
        ph = mjd[good] - t0
        order = np.argsort(ph)
        out["bands"][b] = (ph[order], mag[good][order])
    return out


def interp_on_grid(obj, grid):
    """Interpolate each band's dense truth onto the phase grid (no extrapolation).

    Returns [n_bands, n_grid] mags, NaN where a band does not cover that phase.
    """
    M = np.full((len(BANDS), grid.size), np.nan)
    for bi, b in enumerate(BANDS):
        if b not in obj["bands"]:
            continue
        ph, mag = obj["bands"][b]
        inside = (grid >= ph[0]) & (grid <= ph[-1])
        if inside.any():
            M[bi, inside] = np.interp(grid[inside], ph, mag)
    return M


def ridge_fit(Xtr, ytr, lam=1.0):
    """Standardize + ridge with unregularized intercept. Returns (w, mu, sd)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs = (Xtr - mu) / sd
    Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    w = np.linalg.solve(A, Xb.T @ ytr)
    return w, mu, sd


def ridge_pred(X, w, mu, sd):
    Xs = (X - mu) / sd
    Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    return Xb @ w


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense_glob", required=True)
    p.add_argument("--target", default="ztfg", choices=BANDS)
    p.add_argument("--phase_min", type=float, default=0.0)
    p.add_argument("--phase_max", type=float, default=10.0)
    p.add_argument("--phase_step", type=float, default=0.5)
    p.add_argument("--floor", type=float, default=None,
                   help="Target-band detectable floor (mag). Default = SURVEY_FLOOR.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--out", default="aux_precheck")
    args = p.parse_args()

    floor = args.floor if args.floor is not None else SURVEY_FLOOR[args.target]
    paths = sorted(glob.glob(args.dense_glob))
    if args.max_objects:
        paths = paths[: args.max_objects]
    if not paths:
        raise SystemExit(f"No files match {args.dense_glob}")
    print(f"Loading {len(paths)} dense objects; target={args.target}; "
          f"detectable floor={floor:.2f}")

    grid = np.arange(args.phase_min, args.phase_max + 1e-9, args.phase_step)
    tgt_i = BANDS.index(args.target)
    pred_bands = [b for b in BANDS if b != args.target]
    pred_idx = [BANDS.index(b) for b in pred_bands]

    rows_X, rows_y, rows_ph, rows_obj = [], [], [], []
    # anchor availability: does the row have >=1 deep band with truth ABOVE its
    # own floor (measurable) at this phase?
    rows_anchor = []
    n_ok = 0
    for oi, path in enumerate(paths):
        obj = load_object(path)
        if obj is None or args.target not in obj["bands"]:
            continue
        M = interp_on_grid(obj, grid)
        ytgt = M[tgt_i]
        for gi in range(grid.size):
            yv = ytgt[gi]
            if not np.isfinite(yv):
                continue
            xv = M[pred_idx, gi]
            if np.isfinite(xv).sum() == 0:
                continue
            # measurable anchor = a predictor band present AND brighter than floor
            measurable = False
            for j, b in enumerate(pred_bands):
                if np.isfinite(xv[j]) and xv[j] <= SURVEY_FLOOR[b]:
                    measurable = True
                    break
            rows_X.append(xv)
            rows_y.append(yv)
            rows_ph.append(grid[gi])
            rows_obj.append(oi)
            rows_anchor.append(measurable)
        n_ok += 1

    if not rows_y:
        raise SystemExit("No usable (object, phase) rows with a finite target.")
    X = np.array(rows_X)
    y = np.array(rows_y)
    ph = np.array(rows_ph)
    obj_id = np.array(rows_obj)
    anchor = np.array(rows_anchor)
    print(f"Usable objects: {n_ok}; matched (obj,phase) rows: {y.size}\n")

    # Mean-impute missing predictors (per column) so ridge has a full matrix.
    col_mean = np.nanmean(X, axis=0)
    Xf = X.copy()
    bad = np.where(~np.isfinite(Xf))
    Xf[bad] = np.take(col_mean, bad[1])

    bright = y <= floor      # ztfg detectable (target CAN be built from truth)
    faint = ~bright          # ztfg past floor (the FAILING regime; no truth in train)

    marg_std_faint = float(np.std(y[faint])) if faint.any() else float("nan")

    print("=" * 70)
    print("Q1  COVERAGE at faint-target phases (is there a usable deep anchor?)")
    print("=" * 70)
    n_faint = int(faint.sum())
    n_faint_anchored = int((faint & anchor).sum())
    frac = n_faint_anchored / n_faint if n_faint else float("nan")
    print(f"  faint rows (target > {floor:.2f})          : {n_faint}")
    print(f"  ...with >=1 MEASURABLE deep anchor band    : {n_faint_anchored} "
          f"({100*frac:.1f}%)")
    print(f"  (low % here => no anchor exists => aux loss cannot fire)\n")

    print("=" * 70)
    print("Q2  TRANSFER: fit color/target relation on BRIGHT rows, test on FAINT")
    print("    (the HONEST target a real loss could build -- never sees faint truth)")
    print("=" * 70)
    # Object-grouped CV so a test object's bright rows never train its faint rows.
    uniq = np.unique(obj_id)
    folds = np.array_split(uniq, args.n_folds)
    yhat_faint = np.full(y.shape, np.nan)      # bright-fit -> faint-test
    yhat_all = np.full(y.shape, np.nan)        # all-rows fit (optimistic ceiling)
    for f in folds:
        te = np.isin(obj_id, f)
        tr = ~te
        # Honest: TRAIN only on bright rows of the train objects.
        tr_bright = tr & bright
        if tr_bright.sum() >= Xf.shape[1] + 2 and (te & faint).sum() > 0:
            w, mu, sd = ridge_fit(Xf[tr_bright], y[tr_bright])
            yhat_faint[te & faint] = ridge_pred(Xf[te & faint], w, mu, sd)
        # Optimistic: TRAIN on ALL rows (incl. faint) -- the original ceiling.
        if tr.sum() >= Xf.shape[1] + 2:
            w2, mu2, sd2 = ridge_fit(Xf[tr], y[tr])
            yhat_all[te] = ridge_pred(Xf[te], w2, mu2, sd2)

    res_transfer = (y - yhat_faint)[faint]
    res_ceiling_faint = (y - yhat_all)[faint]
    print(f"  model ztfg late RMSE (reference)            :  ~1.880")
    print(f"  predict-the-mean on faint rows (marg std)   : {marg_std_faint:6.3f}")
    print(f"  OPTIMISTIC ceiling (fit-all) on faint rows  : {rms(res_ceiling_faint):6.3f}")
    print(f"  HONEST  bright-fit -> faint-test residual   : {rms(res_transfer):6.3f}  <-- decisive")
    print(f"  (also restricted to anchored faint rows     : "
          f"{rms((y - yhat_faint)[faint & anchor]):6.3f})\n")

    # Stratify the honest residual by phase to see where transfer breaks.
    print("  HONEST residual by phase bin (bright-fit -> faint-test):")
    edges = np.arange(args.phase_min, args.phase_max + 1e-9, 2.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = faint & (ph >= lo) & (ph < hi)
        if m.sum() == 0:
            continue
        print(f"    {lo:4.1f}-{hi:4.1f} d : {rms((y - yhat_faint)[m]):6.3f} "
              f"(n={int(m.sum())})")
    print()

    print("=" * 70)
    print("VERDICT GUIDE")
    print("=" * 70)
    print("  BUILD candidate (iii) if: HONEST residual << 1.88 AND << marg std,")
    print("    and Q1 coverage is high (a target is both constructible and useful).")
    print("  DO NOT build if: HONEST residual ~ marg std (color learned bright does")
    print("    NOT transfer to faint) OR coverage low. The faint tail is then")
    print("    information-limited in-model; pivot to redshift / accept the floor.")
    print("  Note the GAP optimistic-vs-honest: large gap => the original ceiling")
    print("    test was flattered by faint-truth it fit on but training never has.")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"{args.target}_aux_precheck_rows.csv")
    data = np.column_stack([ph, obj_id, y, yhat_faint, yhat_all,
                            faint.astype(float), anchor.astype(float)])
    hdr = "phase,obj,target_true,pred_honest_faint,pred_optimistic,is_faint,has_anchor"
    np.savetxt(csv_path, data, delimiter=",", header=hdr, comments="")
    print(f"\nWrote per-row CSV -> {csv_path}")


if __name__ == "__main__":
    main()
