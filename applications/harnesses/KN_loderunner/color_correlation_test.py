"""Color-correlation ceiling test: can the well-measured bands predict ztfg?

Motivation
----------
The dense late-time plots show a band-STRUCTURAL failure: ztfg (and to a lesser
degree ztfr/ztfi) plateau ~20-21 mag and flatline while the truth fades to 26+,
whereas u/g/r/i/z/y track the fade well. Physical reason: ZTF's shallow
detection floor (~21) means the model NEVER sees faint ztfg detections, so it
cannot learn ztfg's faint tail from ztfg data alone. The upper-limit hinge can
only push the plateau to the ZTF floor (~21), never to the true 26 -- the signal
is not in any ztfg observation.

A kilonova is ONE cooling ejecta blob with a single evolving SED; every band is
that SED through a filter. So faint ztfg might be recoverable from the bands that
DO stay measurable deep (i/z/y), via the (time-evolving) color relation. This
script measures whether that information is actually there -- the CEILING of a
"shared-SED, project-into-bands" model -- BEFORE we build one.

What it measures
----------------
On the DENSE (noise-free) truth (the intrinsic ceiling), at matched phase:
  1. STEREOTYPY: how much does each color track (X - ztfg) vs phase vary across
     objects? Tight spread => color evolution is predictable => idea has legs.
  2. PREDICTABILITY: cross-validated (object-split) ridge regression predicting
     ztfg from the other bands at matched phase. Report held-out residual RMS,
     stratified by phase and by ztfg faintness (the regime that actually fails).
  3. BASELINES to beat: (a) current model ztfg late-time RMSE ~= 1.9;
     (b) marginal std of ztfg (predict-the-mean); (c) restrict predictors to
     bands brighter than a per-survey floor (a deployability-flavored variant).

A held-out cross-band residual << 1.9 (and << marginal std) in the FAINT/late
regime => the color route has real headroom -> greenlight an SED-latent design.
Loose/scattered => color varies too much per object; the deep tail is genuinely
information-limited and no architecture recovers it.

Read-only. numpy-only (no repo imports; repo src needs py3.10 unions). Runs on
the cluster where the dense pool lives.

Usage
-----
    python color_correlation_test.py \
        --dense_glob '/path/to/dense/*.npz' \
        [--target ztfg] [--phase_min 0 --phase_max 10 --phase_step 0.5] \
        [--faint_cut 20.0] [--n_folds 5] [--out color_test]
"""

import argparse
import glob
import json
import os

import numpy as np

# Canonical band order + the npz array key for each.
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
# Rough single-visit 5-sigma depths (mag) for the deployability-flavored variant.
# A predictor band is "measurable" at a phase only if its truth is brighter.
SURVEY_FLOOR = {
    "ztfg": 20.8, "ztfr": 20.6, "ztfi": 19.9,
    "u": 23.5, "g": 24.5, "r": 24.0, "i": 23.5, "z": 22.5, "y": 21.5,
}


def load_object(path):
    """Return (phase->per-band interpolator inputs) for one dense npz.

    Returns dict band -> (phase_array, mag_array) using only finite-error
    (detection) rows on the dense stream. Phase = MJD - trigger_time.
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
        mjd, mag, err = arr[:, 0], arr[:, 1], arr[:, 2]
        # Dense truth: keep all finite-mag rows (dense stream is noise-free truth;
        # treat any finite mag as a valid truth sample regardless of err column).
        good = np.isfinite(mjd) & np.isfinite(mag)
        if good.sum() < 2:
            continue
        ph = mjd[good] - t0
        order = np.argsort(ph)
        out["bands"][b] = (ph[order], mag[good][order])
    return out


def interp_on_grid(obj, grid):
    """Interpolate each band's dense truth onto the phase grid (no extrapolation).

    Returns [n_bands, n_grid] array of mags with NaN where a band does not cover
    that phase.
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


def ridge_cv(X, y, groups, n_folds, lam=1.0):
    """Object-grouped k-fold ridge; return held-out predictions aligned to y.

    X: [N, F] predictors (already imputed/scaled), y: [N], groups: [N] object ids.
    Standardizes within each train fold. Returns yhat [N] (NaN if a row's group
    was never in any test fold -- shouldn't happen).
    """
    yhat = np.full(y.shape, np.nan)
    uniq = np.unique(groups)
    rng_order = uniq  # deterministic split by sorted object id (no RNG)
    folds = np.array_split(rng_order, n_folds)
    for f in folds:
        test_mask = np.isin(groups, f)
        train_mask = ~test_mask
        if train_mask.sum() < X.shape[1] + 2 or test_mask.sum() == 0:
            continue
        Xtr, ytr = X[train_mask], y[train_mask]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (X[test_mask] - mu) / sd
        # add intercept
        Xtr_b = np.hstack([Xtr_s, np.ones((Xtr_s.shape[0], 1))])
        Xte_b = np.hstack([Xte_s, np.ones((Xte_s.shape[0], 1))])
        A = Xtr_b.T @ Xtr_b + lam * np.eye(Xtr_b.shape[1])
        A[-1, -1] -= lam  # do not regularize the intercept
        w = np.linalg.solve(A, Xtr_b.T @ ytr)
        yhat[test_mask] = Xte_b @ w
    return yhat


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense_glob", required=True,
                   help="Glob for dense (noise-free) truth npz files.")
    p.add_argument("--target", default="ztfg", choices=BANDS)
    p.add_argument("--phase_min", type=float, default=0.0)
    p.add_argument("--phase_max", type=float, default=10.0)
    p.add_argument("--phase_step", type=float, default=0.5)
    p.add_argument("--faint_cut", type=float, default=20.0,
                   help="Target mag fainter than this = the failing regime.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_objects", type=int, default=0,
                   help="0 = all; else cap for a quick look.")
    p.add_argument("--out", default="color_test")
    args = p.parse_args()

    paths = sorted(glob.glob(args.dense_glob))
    if args.max_objects:
        paths = paths[: args.max_objects]
    if not paths:
        raise SystemExit(f"No files match {args.dense_glob}")
    print(f"Loading {len(paths)} dense objects; target band = {args.target}")

    grid = np.arange(args.phase_min, args.phase_max + 1e-9, args.phase_step)
    tgt_i = BANDS.index(args.target)
    pred_bands = [b for b in BANDS if b != args.target]
    pred_idx = [BANDS.index(b) for b in pred_bands]

    # Accumulate one row per (object, phase-grid point) with a finite target.
    rows_X, rows_y, rows_ph, rows_obj, rows_z = [], [], [], [], []
    # For the deployability variant: mask predictors fainter than their floor.
    rows_Xmeas = []
    n_ok = 0
    for oi, path in enumerate(paths):
        obj = load_object(path)
        if obj is None or args.target not in obj["bands"]:
            continue
        M = interp_on_grid(obj, grid)  # [n_bands, n_grid]
        ytgt = M[tgt_i]
        for gi in range(grid.size):
            yv = ytgt[gi]
            if not np.isfinite(yv):
                continue
            xv = M[pred_idx, gi]
            if np.isfinite(xv).sum() == 0:
                continue
            rows_X.append(xv)
            # deployability: NaN-out predictors fainter than their survey floor
            xmeas = xv.copy()
            for j, b in enumerate(pred_bands):
                if np.isfinite(xmeas[j]) and xmeas[j] > SURVEY_FLOOR[b]:
                    xmeas[j] = np.nan
            rows_Xmeas.append(xmeas)
            rows_y.append(yv)
            rows_ph.append(grid[gi])
            rows_obj.append(oi)
            rows_z.append(obj["redshift"])
        n_ok += 1

    if not rows_y:
        raise SystemExit("No usable (object, phase) rows with a finite target.")
    X = np.array(rows_X)
    Xmeas = np.array(rows_Xmeas)
    y = np.array(rows_y)
    ph = np.array(rows_ph)
    obj_id = np.array(rows_obj)
    zred = np.array(rows_z)
    print(f"Usable objects: {n_ok}; matched (obj,phase) rows: {y.size}\n")

    # Mean-impute missing predictors per column (fit on all rows -- fine for a
    # ceiling read; CV standardization is refit per fold).
    def impute(A):
        A = A.copy()
        col_mean = np.nanmean(A, axis=0)
        inds = np.where(~np.isfinite(A))
        A[inds] = np.take(col_mean, inds[1])
        # add a per-column present-flag so "missing" is distinguishable
        flags = np.isfinite(np.array(A)).astype(float)  # all 1 after fill; keep shape
        return A
    Xf = impute(X)
    Xmf = impute(Xmeas)

    # ---- Baselines -------------------------------------------------------
    marg_std = float(np.std(y))
    faint = y > args.faint_cut
    marg_std_faint = float(np.std(y[faint])) if faint.any() else float("nan")

    # ---- Predictability: full dense (ceiling) ----------------------------
    yhat = ridge_cv(Xf, y, obj_id, args.n_folds)
    res = y - yhat
    # ---- Predictability: deployability (measurable predictors only) ------
    yhat_m = ridge_cv(Xmf, y, obj_id, args.n_folds)
    res_m = y - yhat_m

    print("=" * 68)
    print(f"CROSS-BAND PREDICTABILITY of {args.target} (object-grouped {args.n_folds}-fold CV)")
    print("=" * 68)
    print(f"  Baseline  marginal std (predict mean)      : {marg_std:6.3f}")
    print(f"  Baseline  current model ztfg late RMSE ~    :  1.900  (reference)")
    print(f"  CEILING   held-out residual RMS (all bands) : {rms(res):6.3f}")
    print(f"  DEPLOY    held-out residual RMS (measurable): {rms(res_m):6.3f}")
    print()

    # ---- Stratify by phase ----------------------------------------------
    print("  Residual RMS by phase bin (ceiling / deploy / n):")
    edges = np.arange(args.phase_min, args.phase_max + 1e-9, 2.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ph >= lo) & (ph < hi)
        if m.sum() == 0:
            continue
        print(f"    {lo:4.1f}-{hi:4.1f} d : "
              f"{rms(res[m]):6.3f} / {rms(res_m[m]):6.3f}  (n={m.sum()})")
    print()

    # ---- Stratify by target faintness (the failing regime) --------------
    print(f"  Residual RMS by {args.target} brightness:")
    for label, m in [("bright (<=cut)", ~faint), (f"faint (>{args.faint_cut})", faint)]:
        if m.sum() == 0:
            continue
        print(f"    {label:16s}: ceiling {rms(res[m]):6.3f} / "
              f"deploy {rms(res_m[m]):6.3f} / marg {np.std(y[m]):6.3f} (n={m.sum()})")
    print()

    # ---- Stereotypy: color track spread across objects ------------------
    print("=" * 68)
    print(f"COLOR-TRACK STEREOTYPY: std across objects of (band - {args.target})")
    print("  (small = stereotyped color evolution = predictable)")
    print("=" * 68)
    print(f"  {'band':>5} | " + " ".join(f"{lo:.0f}-{hi:.0f}d"
          for lo, hi in zip(edges[:-1], edges[1:])))
    for b in pred_bands:
        bi = BANDS.index(b)
        line = f"  {b:>5} |"
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ph >= lo) & (ph < hi)
            color = X[m, pred_bands.index(b)] - y[m]
            color = color[np.isfinite(color)]
            line += f"  {np.std(color):5.2f}" if color.size > 3 else "    -- "
        print(line)
    print()

    # ---- Save per-row CSV for follow-up ---------------------------------
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"{args.target}_color_rows.csv")
    hdr = (["phase", "obj", "redshift", f"{args.target}_true",
            f"{args.target}_pred_ceiling", f"{args.target}_pred_deploy"]
           + [f"{b}_true" for b in pred_bands])
    data = np.column_stack([ph, obj_id, zred, y, yhat, yhat_m, X])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(hdr), comments="")
    print(f"Wrote per-row CSV -> {csv_path}")
    print("\nVERDICT GUIDE:")
    print("  If FAINT-regime ceiling residual << 1.9 and << marginal std ->")
    print("    color info IS there; build the shared-SED / project-into-bands model.")
    print("  If deploy residual also small -> recoverable from bands that stay")
    print("    measurable at late phase (the realistic case). Strong greenlight.")
    print("  If faint-regime residual ~ marginal std -> color varies too much per")
    print("    object; deep ztfg tail is information-limited. Do not build it.")


if __name__ == "__main__":
    main()
