r"""Where the u / g late-time error lives (Study 209+ diagnostics).

Read-only: reads the ``latetime_uniform_scored_points.csv`` that
eval_dense_latetime_9band.py writes. No model / GPU. The u and g bias is small
(bias^2 ~5% of u's MSE), so this looks for where the SPREAD sits:

1. Band summary. RMSE, bias and share of the total MSE for each target band and
   a reference band.
2. Truth magnitude. RMSE / bias / MSE share per truth-magnitude quintile of each
   band. If the faintest quintile carries most of the MSE, the error is the
   beyond-depth tail (the model never sees u/g that faint in context).
3. Context support. RMSE / bias per number of SAME-BAND detections in the eval
   context (same window + cap as the eval). If 0-detection objects dominate,
   u/g is a data-support problem; if the error is flat in the count, more u/g
   context would not help. Needs ``--realistic_glob`` (skipped otherwise).
4. Level vs shape. RMSE after an ORACLE removes each object's mean residual in
   the band, then a linear trend in lead (upper bounds: the oracle uses the
   truth). Then splits the per-object level into the part shared with the
   reference bands (the object's overall brightness, a common-level error) and
   the band-specific rest (a color error): ``corr`` and ``R^2`` of the band's
   per-object mean residual against the reference bands' mean, and the RMSE
   left after shifting by the reference level instead of the band's own.
5. Concentration. Share of the band's MSE from the worst 10% / 25% of objects,
   and what those objects look like (truth magnitude, context count).

Run on the cluster, e.g.:

    D=dense_latetime_eval_9band/latetime_uniform_scored_points.csv
    python diagnose_ug_error.py --csv 209=runs/study_209/$D \\
        --realistic_glob "$KN/rubin_lsst_ztf_10000_dataset_same_seed/lc_*.npz"
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BAND_NAMES = ["ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y"]


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def load(path: str) -> pd.DataFrame:
    """Scored points with ``resid`` = pred - true."""
    df = pd.read_csv(
        path,
        usecols=["stem", "band", "lead_time_days", "pred_mag", "true_mag"],
    )
    df["resid"] = df["pred_mag"] - df["true_mag"]
    return df


def context_band_counts(
    stems: list,
    realistic_glob: str,
    cutoff_days: float,
    window_days: float,
    max_context_len: int,
) -> pd.DataFrame:
    """Per-object detections per band in the eval context, after window + cap.

    Same rule as ``eval_dense_latetime_9band.eval_object``: detections with
    phase <= ``cutoff_days`` from the first realistic detection, the trailing
    ``window_days`` window ending at the last of them, then the cap.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_dense_latetime_9band import _stem_to_path, read_merged_stream

    from yoke.utils.context_selection import window_select_positions

    paths = _stem_to_path(realistic_glob)
    if not paths:
        raise SystemExit(f"--realistic_glob matched no files: {realistic_glob!r}")
    if not set(stems) & set(paths):
        raise SystemExit(
            f"--realistic_glob matched {len(paths)} files, but none share a stem "
            f"with the CSV (e.g. CSV {stems[0]!r} vs glob {next(iter(paths))!r})."
        )
    rows = []
    for s in stems:
        if s not in paths:
            continue
        t, _v, b, _ul, _z = read_merged_stream(paths[s], drop_upper_limits=True)
        counts = np.zeros(len(BAND_NAMES), dtype=np.int64)
        if t.shape[0]:
            keep = (t - t[0]) <= cutoff_days
            t, b = t[keep], b[keep]
            win = np.nonzero(t >= t[-1] - window_days)[0]
            win = win[window_select_positions(win.shape[0], max_context_len)]
            counts = np.bincount(b[win], minlength=len(BAND_NAMES))
        rows.append({"stem": s, **dict(zip(BAND_NAMES, counts))})
    return pd.DataFrame(rows).set_index("stem")


def report_summary(df: pd.DataFrame, bands: list) -> None:
    """Section 1: RMSE, bias and share of the total MSE per band."""
    total = float(np.sum(np.square(df["resid"])))
    print(f"\n[1] Band summary (overall RMSE {_rmse(df['resid'].to_numpy()):.4f})")
    print(f"    {'band':>6}{'RMSE':>8}{'bias':>8}{'bias^2/MSE':>12}{'MSE share':>11}")
    for b in bands:
        r = df.loc[df["band"] == b, "resid"].to_numpy()
        mse = float(np.mean(np.square(r)))
        print(f"    {b:>6}{_rmse(r):>8.4f}{np.mean(r):>+8.3f}"
              f"{np.mean(r) ** 2 / mse:>12.1%}"
              f"{np.sum(np.square(r)) / total:>11.1%}")


def report_truth_mag(df: pd.DataFrame, bands: list, n_bins: int) -> None:
    """Section 2: error per truth-magnitude quantile of each band."""
    print(f"\n[2] By truth magnitude ({n_bins} quantile bins per band)")
    print(f"    {'band':>6}{'true mag':>15}{'points':>9}{'RMSE':>8}{'bias':>8}"
          f"{'MSE share':>11}")
    for b in bands:
        d = df[df["band"] == b]
        sq_total = float(np.sum(np.square(d["resid"])))
        edges = np.quantile(d["true_mag"], np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, d["true_mag"], side="right") - 1,
                      0, n_bins - 1)
        for i in range(n_bins):
            r = d["resid"].to_numpy()[idx == i]
            print(f"    {b:>6}{f'{edges[i]:.1f}-{edges[i + 1]:.1f}':>15}"
                  f"{len(r):>9}{_rmse(r):>8.4f}{np.mean(r):>+8.3f}"
                  f"{np.sum(np.square(r)) / sq_total:>11.1%}")


def report_context(
    df: pd.DataFrame, counts: pd.DataFrame, bands: list, edges: list
) -> None:
    """Section 3: error per same-band context-detection count."""
    print("\n[3] By same-band detections in the eval context (after window + cap)")
    print(f"    {'band':>6}{'count':>7}{'objects':>9}{'RMSE':>8}{'bias':>8}"
          f"{'MSE share':>11}")
    for b in bands:
        d = df[(df["band"] == b) & df["stem"].isin(counts.index)]
        n = counts.loc[d["stem"], b].to_numpy()
        r = d["resid"].to_numpy()
        sq_total = float(np.sum(np.square(r)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (n >= lo) & (n < hi)
            if not m.any():
                continue
            if hi == lo + 1:
                label = str(lo)
            else:
                label = f"{lo}+" if hi > 99 else f"{lo}-{hi - 1}"
            print(f"    {b:>6}{label:>7}{d['stem'][m].nunique():>9}"
                  f"{_rmse(r[m]):>8.4f}{np.mean(r[m]):>+8.3f}"
                  f"{np.sum(np.square(r[m])) / sq_total:>11.1%}")


def report_level_shape(df: pd.DataFrame, bands: list, ref_bands: list) -> None:
    """Section 4: per-object level vs lead slope, and shared vs color level."""
    print("\n[4] Level vs shape (oracle, upper bounds)")
    ref = df[df["band"].isin(ref_bands)]
    ref_level = ref.groupby("stem")["resid"].mean()
    print(f"    reference level = per-object mean residual over "
          f"{'/'.join(ref_bands)}")
    print(f"    {'band':>6}{'as is':>8}{'-level':>8}{'+slope':>8}"
          f"{'-ref lvl':>10}{'corr':>7}{'R^2':>6}{'color bias':>12}")
    for b in bands:
        d = df[(df["band"] == b) & df["stem"].isin(ref_level.index)]
        r = d["resid"]
        own = r.groupby(d["stem"]).transform("mean")
        r_lvl = r - own
        x = d["lead_time_days"]
        xc = x - x.groupby(d["stem"]).transform("mean")
        sxy = (xc * r_lvl).groupby(d["stem"]).transform("sum")
        sxx = (xc * xc).groupby(d["stem"]).transform("sum")
        r_slope = r_lvl - sxy / sxx.where(sxx > 0, np.inf) * xc
        r_ref = r - ref_level.loc[d["stem"]].to_numpy()

        obj = r.groupby(d["stem"]).mean()
        refo = ref_level.loc[obj.index]
        corr = float(np.corrcoef(obj, refo)[0, 1])
        print(f"    {b:>6}{_rmse(r.to_numpy()):>8.4f}{_rmse(r_lvl.to_numpy()):>8.4f}"
              f"{_rmse(r_slope.to_numpy()):>8.4f}{_rmse(r_ref.to_numpy()):>10.4f}"
              f"{corr:>7.2f}{corr ** 2:>6.2f}{float(np.mean(obj - refo)):>+12.3f}")
    print("    -level: own per-object shift removed; +slope: also its lead trend;")
    print("    -ref lvl: shifted by the REFERENCE bands' per-object error instead")
    print("    (what a perfect shared-level fix would leave); color bias = mean of")
    print("    (band level - reference level) over objects.")


def report_concentration(
    df: pd.DataFrame, bands: list, counts: "pd.DataFrame | None"
) -> None:
    """Section 5: how concentrated the band's MSE is across objects."""
    print("\n[5] Concentration across objects")
    print(f"    {'band':>6}{'worst 10%':>11}{'worst 25%':>11}"
          f"{'med true (w10)':>16}{'med true (rest)':>17}"
          + (f"{'ctx n (w10)':>13}{'ctx n (rest)':>14}" if counts is not None else ""))
    for b in bands:
        d = df[df["band"] == b]
        g = d.groupby("stem")
        sq = g["resid"].apply(lambda v: float(np.sum(np.square(v))))
        sq = sq.sort_values(ascending=False)
        cum = np.cumsum(sq.to_numpy()) / sq.sum()
        n10 = max(1, int(round(0.10 * len(sq))))
        n25 = max(1, int(round(0.25 * len(sq))))
        worst = sq.index[:n10]
        true_med = g["true_mag"].median()
        row = (f"    {b:>6}{cum[n10 - 1]:>11.1%}{cum[n25 - 1]:>11.1%}"
               f"{true_med.loc[worst].median():>16.2f}"
               f"{true_med.drop(worst).median():>17.2f}")
        if counts is not None:
            c = counts[b]
            w = c.reindex(worst).dropna()
            rest = c.drop(worst, errors="ignore")
            row += f"{w.mean():>13.2f}{rest.mean():>14.2f}"
        print(row)
    print("    worst 10% / 25%: share of the band's MSE from that fraction of")
    print("    objects (uniform = 10% / 25%); med true = median truth mag; ctx n =")
    print("    mean same-band detections in the eval context.")


def main() -> None:
    """Parse args and print the five sections for each CSV."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", action="append", required=True,
                   help="LABEL=path to latetime_uniform_scored_points.csv "
                        "(repeatable).")
    p.add_argument("--bands", nargs="+", default=["u", "g"],
                   help="Bands to diagnose.")
    p.add_argument("--ref_bands", nargs="+", default=["r", "i", "z"],
                   help="Bands whose per-object error defines the shared level "
                        "in section 4. r is also shown in section 1 as a reference.")
    p.add_argument("--mag_bins", type=int, default=5,
                   help="Truth-magnitude quantile bins for section 2.")
    p.add_argument("--realistic_glob", default=None,
                   help="Realistic light-curve glob for section 3 (and the "
                        "context column of section 5). Skipped when omitted.")
    p.add_argument("--cutoff_days", type=float, default=2.0)
    p.add_argument("--window_days", type=float, default=2.0)
    p.add_argument("--max_context_len", type=int, default=12)
    p.add_argument("--count_edges", type=int, nargs="+", default=[0, 1, 2, 3, 5, 100],
                   help="Same-band count bins; bin i is edges[i] <= n < edges[i+1].")
    args = p.parse_args()

    counts = None
    for item in args.csv:
        label, path = item.split("=", 1)
        df = load(path)
        print(f"\n===== {label}: {len(df)} points, {df['stem'].nunique()} "
              f"objects; uniform RMSE {_rmse(df['resid'].to_numpy()):.4f} =====")
        if args.realistic_glob and counts is None:
            counts = context_band_counts(
                sorted(df["stem"].unique()), args.realistic_glob,
                args.cutoff_days, args.window_days, args.max_context_len,
            )
            print(f"Context counts for {len(counts)} objects "
                  f"(cutoff {args.cutoff_days:g} d, window {args.window_days:g} d, "
                  f"cap {args.max_context_len}).")
        summary_bands = list(dict.fromkeys([*args.bands, args.ref_bands[0]]))
        report_summary(df, summary_bands)
        report_truth_mag(df, args.bands, args.mag_bins)
        if counts is not None:
            report_context(df, counts, args.bands, args.count_edges)
        report_level_shape(df, args.bands, args.ref_bands)
        report_concentration(df, args.bands, counts)


if __name__ == "__main__":
    main()
