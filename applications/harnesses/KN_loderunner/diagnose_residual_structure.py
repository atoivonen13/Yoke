r"""Locate the remaining late-time error (Study 203+ diagnostics).

Read-only: reads the ``latetime_uniform_scored_points.csv`` that
eval_dense_latetime_9band.py writes. No model / GPU. Four questions:

1. Jitter. Each lead is predicted independently, so the forecast curve can
   wiggle point to point. Prints the RMS second difference of pred and of truth
   on the same grid. Then prints the fraction of the high-pass forecast that is
   common to all bands at the same phase: ~1/9 if the wiggles are independent
   per band, near 1 if they come from the shared Dt / query pathway.
2. Smoothing. RMSE after a centered moving average of the forecast along phase.
   This is a free post-hoc fix if the jitter is noise; the truth is smooth.
3. Error decomposition. RMSE after an ORACLE removes, per object, one shift
   shared by all bands; then a shift per band; then a per-band linear trend in
   lead. The drops split the error into object level, band level, decay rate
   and remaining shape. They are upper bounds: the oracle uses the truth.
4. Long-lead tail. RMSE per lead bin, split into the objects that reach the
   last bin and the rest, both scored at every lead. If the tail objects are
   bad at short leads too, the tail is object selection (sparse early context),
   not lead.

Run on the cluster, e.g.:

    D=dense_latetime_eval_9band/latetime_uniform_scored_points.csv
    python diagnose_residual_structure.py \\
        --csv 203=runs/study_203/$D --csv 205=runs/study_205/$D
"""

import argparse

import numpy as np
import pandas as pd

BAND_NAMES = ["ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y"]
KEYS = ["stem", "band"]


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def load(path: str) -> pd.DataFrame:
    """Scored points sorted by (stem, band, phase), with ``resid`` = pred - true."""
    df = pd.read_csv(
        path,
        usecols=["stem", "band", "phase_days", "lead_time_days", "pred_mag",
                 "true_mag"],
    )
    df = df.sort_values(["stem", "band", "phase_days"], kind="mergesort")
    df = df.reset_index(drop=True)
    df["resid"] = df["pred_mag"] - df["true_mag"]
    return df


def grid_step(df: pd.DataFrame) -> float:
    """Median phase spacing between consecutive points of one curve."""
    d = df.groupby(KEYS, sort=False)["phase_days"].diff().to_numpy()
    return float(np.nanmedian(d))


def window_points(days: float, step: float) -> int:
    """Odd moving-average width (in grid points) covering ~``days``."""
    k = max(1, int(round(days / step)))
    return k if k % 2 == 1 else k + 1


def smooth(df: pd.DataFrame, col: str, k: int) -> np.ndarray:
    """Centered k-point moving average of ``col`` along phase, per curve."""
    if k <= 1:
        return df[col].to_numpy()
    out = (
        df.groupby(KEYS, sort=False)[col]
        .rolling(k, center=True, min_periods=1)
        .mean()
    )
    return out.reset_index(level=[0, 1], drop=True).sort_index().to_numpy()


def rms_second_diff(df: pd.DataFrame, col: str) -> float:
    """RMS second difference of ``col`` along phase, per curve (mag / step^2)."""
    d1 = df.groupby(KEYS, sort=False)[col].diff()
    d2 = d1.groupby([df["stem"], df["band"]], sort=False).diff().to_numpy()
    return float(np.sqrt(np.nanmean(np.square(d2))))


def report_jitter(df: pd.DataFrame, step: float, hp_days: float) -> None:
    """Section 1: how rough is the forecast, and is the roughness band-common?"""
    print("\n[1] Jitter (RMS second difference per grid step, mag)")
    print(f"    pred {rms_second_diff(df, 'pred_mag'):.4f}   "
          f"truth {rms_second_diff(df, 'true_mag'):.4f}")

    k = window_points(hp_days, step)
    hp = df["pred_mag"].to_numpy() - smooth(df, "pred_mag", k)
    g = pd.Series(hp).groupby(
        [df["stem"].to_numpy(), df["phase_days"].round(3).to_numpy()]
    )
    size = g.transform("size").to_numpy()
    n_full = int(size.max())
    if n_full < 2:
        print("    Band phases never coincide; band-common fraction skipped.")
        return
    cm = g.transform("mean").to_numpy()
    full = size == n_full
    frac = float(np.sum(cm[full] ** 2) / np.sum(hp[full] ** 2))
    print(f"    High-pass ({hp_days:g} d = {k} pts) RMS {_rmse(hp):.4f} mag; "
          f"band-common fraction {frac:.2f} "
          f"(independent bands -> {1.0 / n_full:.2f})")


def report_smoothing(df: pd.DataFrame, step: float, days_list: list) -> None:
    """Section 2: RMSE of the moving-averaged forecast, overall and per band."""
    print("\n[2] RMSE after a centered moving average of the forecast")
    band = df["band"].to_numpy()
    true = df["true_mag"].to_numpy()
    header = f"    {'window':>12}{'ALL':>8}" + "".join(f"{b:>8}" for b in BAND_NAMES)
    print(header)
    for days in [0.0, *days_list]:
        k = window_points(days, step) if days > 0 else 1
        r = smooth(df, "pred_mag", k) - true
        label = "none" if k == 1 else f"{days:g} d/{k} pt"
        row = f"    {label:>12}{_rmse(r):>8.4f}"
        row += "".join(f"{_rmse(r[band == b]):>8.4f}" for b in BAND_NAMES)
        print(row)


def report_decomposition(df: pd.DataFrame) -> None:
    """Section 3: oracle level / band / slope removal."""
    print("\n[3] RMSE after an oracle removes (uses truth -> upper bounds)")
    r = df["resid"]
    keys = [df["stem"], df["band"]]
    r_obj = r - r.groupby(df["stem"]).transform("mean")
    ym = r.groupby(keys).transform("mean")
    r_band = r - ym
    x = df["lead_time_days"]
    xc = x - x.groupby(keys).transform("mean")
    sxy = (xc * r_band).groupby(keys).transform("sum")
    sxx = (xc * xc).groupby(keys).transform("sum")
    slope = sxy / sxx.where(sxx > 0, np.inf)  # no lead spread -> slope 0
    r_slope = r_band - slope * xc

    cols = {
        "as is": r.to_numpy(),
        "obj shift": r_obj.to_numpy(),
        "+band shift": r_band.to_numpy(),
        "+band slope": r_slope.to_numpy(),
    }
    band = df["band"].to_numpy()
    print(f"    {'band':>6}" + "".join(f"{c:>13}" for c in cols))
    for name in ["ALL", *BAND_NAMES]:
        m = slice(None) if name == "ALL" else band == name
        print(f"    {name:>6}" + "".join(f"{_rmse(v[m]):>13.4f}" for v in cols.values()))


def report_lead_tail(df: pd.DataFrame, edges: np.ndarray) -> None:
    """Section 4: is the long-lead tail the lead, or the objects that reach it?"""
    lead = df["lead_time_days"].to_numpy()
    r = df["resid"].to_numpy()
    stem = df["stem"].to_numpy()
    b = np.digitize(lead, edges) - 1
    nonempty = [i for i in range(len(edges) - 1) if np.any(b == i)]
    last = nonempty[-1]
    is_tail = np.isin(stem, np.unique(stem[b == last]))
    n_tail = len(np.unique(stem[is_tail]))
    print(f"\n[4] RMSE by lead; 'tail' = the {n_tail} objects with points in "
          f"[{edges[last]:g}, {edges[last + 1]:g}) d")
    print(f"    {'lead (d)':>10}{'points':>9}{'objects':>9}{'RMSE':>8}"
          f"{'tail objs':>11}{'others':>8}")
    for i in nonempty:
        m = b == i
        mt, mo = m & is_tail, m & ~is_tail
        rt = f"{_rmse(r[mt]):>11.4f}" if mt.any() else f"{'-':>11}"
        ro = f"{_rmse(r[mo]):>8.4f}" if mo.any() else f"{'-':>8}"
        print(f"    {f'{edges[i]:g}-{edges[i + 1]:g}':>10}{m.sum():>9}"
              f"{len(np.unique(stem[m])):>9}{_rmse(r[m]):>8.4f}{rt}{ro}")


def main() -> None:
    """Parse args and print the four sections for each CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", action="append", required=True,
                   help="LABEL=path to latetime_uniform_scored_points.csv "
                        "(repeatable).")
    p.add_argument("--smooth_days", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0],
                   help="Moving-average widths (days) for section 2.")
    p.add_argument("--hp_days", type=float, default=0.5,
                   help="Smoothing width (days) whose residual defines the "
                        "high-pass jitter in section 1.")
    p.add_argument("--lead_edges", type=float, nargs="+",
                   default=list(range(0, 11)),
                   help="Lead-bin edges (days) for section 4.")
    args = p.parse_args()

    for item in args.csv:
        label, path = item.split("=", 1)
        df = load(path)
        step = grid_step(df)
        print(f"\n===== {label}: {len(df)} points, {df['stem'].nunique()} "
              f"objects, grid step {step:.4f} d; uniform RMSE "
              f"{_rmse(df['resid'].to_numpy()):.4f} =====")
        report_jitter(df, step, args.hp_days)
        report_smoothing(df, step, args.smooth_days)
        report_decomposition(df)
        report_lead_tail(df, np.asarray(args.lead_edges, dtype=float))


if __name__ == "__main__":
    main()
