"""Dense late-time evaluation for the 9-band scalar temporal LodeRunner.

The model is trained on REALISTIC light curves (sparse, upper limits dropped),
which contain almost no late-time detections because kilonovae fade below the
detection limit. This eval measures how well the model, given a REALISTIC
observing context, forecasts the LATE-TIME behavior -- scored against a DENSE
companion set (the same objects, denser cadence, no limiting-mag cut, so all
late-time points are real detections). Realistic and dense views of an object
are paired by filename stem.

For each held-out (test-split) object:
  1. Build the model input from the realistic stream (trailing time-window
     context ending at the last realistic detection), exactly as in training.
  2. For every dense point in the late-time region (phase from the first
     realistic detection greater than ``--late_time_cutoff_days``), ask the model
     to predict all nine bands at that point's lead time and score the predicted
     magnitude of the dense point's band against the dense truth.
  3. Also sweep a smooth lead-time grid for a per-object forecast plot.

Only detections are used: realistic upper limits are dropped (matching
training); the dense set is all detections by construction.

IMPORTANT (time frames): the training dataset relativizes each stream to its own
first event, so the realistic and dense views of one object live in different
relative frames. Lead times (durations) are frame-independent and are what the
model's ``Dt`` consumes, so this script reads ABSOLUTE MJD (column 0) from the
raw npz files and works entirely in absolute-time differences.

Normalization uses the TRAIN-ONLY realistic stats the model was trained with
(``kilonova_9band_norm_stats_trainonly.npz`` by default) -- the exact encoding
the model saw. This is a read-only diagnostic: it writes plots and a CSV and
never trains.
"""

import argparse
import csv
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from yoke.datasets.kilonova_dataset import (
    EPS,
    NINE_BAND_KEYS,
    load_or_compute_band_normalization,
)
from yoke.utils.checkpointing import _epoch_median_val_losses

# Reuse the model loader and window/input helpers from the rollout diagnostics
# script that lives alongside this one. These harness scripts are run directly
# (not as an installed package), so make the script directory importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pred_diagnostics_9band import (  # noqa: E402
    _select_window,
    build_context_input,
    load_9band_model,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", family="serif")
plt.rcParams["figure.figsize"] = (7, 5)


BAND_KEYS = NINE_BAND_KEYS
BAND_NAMES = ("ztfg", "ztfr", "ztfi", "u", "g", "r", "i", "z", "y")
BAND_COLORS = (
    "#2A9D8F", "#E63946", "#F4A261", "#457B9D", "#1B9E77",
    "#D62828", "#E9C46A", "#8338EC", "#264653",
)
VALUE_COL = 1
ERROR_COL = 2
N_BANDS = len(BAND_KEYS)
DROP_UPPER_LIMITS = True  # matches training for the realistic (context) stream
FILELIST_DIR = "/net/sescratch1/exempt/artimis/atoivonen/filelists"
# npz reader threads and how many objects they may read ahead of the model.
IO_WORKERS = 8
IO_PREFETCH = 32


def study_tag(study: int) -> str:
    """Zero-padded study id used in default paths."""
    return f"{int(study):03d}"


def resolve_best_checkpoint(run_dir: str, tag: str, max_epoch: int) -> tuple:
    """Pick the lowest-median-val-loss checkpoint at or below ``max_epoch``.

    The training loop writes a per-epoch validation record CSV
    (``validation_study{tag}_epoch{NNNN}.csv``, columns ``epoch, batch, loss``)
    and a per-epoch checkpoint (``study{tag}_modelState_epoch{NNNN}.pth``) in the
    run directory. Rather than blindly loading a fixed epoch, this ranks every
    epoch that (a) has a readable val record, (b) is ``<= max_epoch``, and (c) has
    a checkpoint on disk, by MEDIAN validation loss -- the same statistic
    ``update_best_checkpoint`` uses -- and returns the best one.

    Validation loss is the trained objective and a proxy (not identical) for the
    late-time RMSE the studies are ranked on, so this picks the best-generalizing
    epoch within the requested budget instead of the last/arbitrary one. The
    ``<= max_epoch`` cap lets a caller reproduce an earlier read or bound the
    search to a finished-training horizon.

    Args:
        run_dir (str): Directory holding the checkpoints and val record CSVs.
        tag (str): Zero-padded study id (e.g. ``"125"``).
        max_epoch (int): Only consider epochs at or below this value.

    Returns:
        tuple: ``(best_epoch, best_loss, best_ckpt_path)``, or
        ``(None, None, None)`` when no val record + checkpoint pair qualifies (the
        caller then falls back to the fixed-epoch path).
    """
    val_glob = os.path.join(run_dir, f"validation_study{tag}_epoch*.csv")
    med = _epoch_median_val_losses(val_glob)
    if not med:
        return None, None, None

    best_epoch, best_loss, best_path = None, None, None
    for ep in sorted(med):
        if ep > max_epoch:
            continue
        ckpt = os.path.join(run_dir, f"study{tag}_modelState_epoch{ep:04d}.pth")
        if not os.path.exists(ckpt):
            continue
        if best_loss is None or med[ep] < best_loss:
            best_epoch, best_loss, best_path = ep, med[ep], ckpt

    return best_epoch, best_loss, best_path


def _stem(path: str) -> str:
    """Return the object identifier: filename without directory or extension."""
    return os.path.splitext(os.path.basename(path))[0]


def read_merged_stream(
    npz_path: str, drop_upper_limits: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Read one file's merged, time-sorted event stream in ABSOLUTE MJD.

    Unlike the training dataset, times are NOT relativized here, so streams from
    two directories (realistic and dense) remain on a common absolute clock.

    Args:
        npz_path (str): Path to the light-curve npz.
        drop_upper_limits (bool): Drop non-detections (non-finite error) so the
            realistic context stream matches training. The dense set is all
            detections, so this is a no-op there. When False, upper limits are
            KEPT and flagged in the returned ``is_ul`` array (for models trained
            with ``upper_limit_channel``).

    Returns:
        (times, values, bands, is_ul, redshift): absolute MJD, raw magnitude,
        band index, upper-limit flag (1.0 for a non-detection); each [N] and
        sorted by time. ``redshift`` is the object's physical redshift parsed
        from ``injection_parameters`` (a per-object constant; ``nan`` if absent).
        Empty arrays (and ``nan`` redshift) if the file has no usable events.
    """
    data = np.load(npz_path, allow_pickle=True)
    redshift = np.nan
    if "injection_parameters" in data.files:
        try:
            inj = json.loads(str(data["injection_parameters"][0]))
            redshift = float(inj.get("redshift", np.nan))
        except (ValueError, KeyError, TypeError):
            redshift = np.nan
    times, values, bands, is_ul = [], [], [], []
    for band_idx, key in enumerate(BAND_KEYS):
        if key not in data.files:
            continue
        arr = data[key]
        if arr.size == 0:
            continue
        detected = np.isfinite(arr[:, ERROR_COL])
        if drop_upper_limits:
            arr = arr[detected]
            detected = detected[detected]  # all True after the mask
            if arr.shape[0] == 0:
                continue
        times.append(arr[:, 0].astype(np.float64))
        values.append(arr[:, VALUE_COL].astype(np.float32))
        bands.append(np.full(arr.shape[0], band_idx, dtype=np.int64))
        is_ul.append((~detected).astype(np.float32))
    data.close()

    if not times:
        empty_f = np.empty(0, dtype=np.float64)
        return (
            empty_f,
            empty_f.astype(np.float32),
            np.empty(0, dtype=np.int64),
            empty_f.astype(np.float32),
            redshift,
        )

    times = np.concatenate(times)
    values = np.concatenate(values)
    bands = np.concatenate(bands)
    is_ul = np.concatenate(is_ul)
    order = np.argsort(times, kind="stable")
    return times[order], values[order], bands[order], is_ul[order], redshift


def _stem_to_path(data_glob: str) -> dict:
    """Map object stem -> file path for all files matched by a glob."""
    import glob

    return {_stem(f): f for f in glob.glob(data_glob)}


def _batched_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    lead_times: np.ndarray,
    device: torch.device,
    max_batch: int = 256,
    return_quantiles: bool = False,
) -> np.ndarray:
    """Predict all bands for many lead times in one (chunked) forward pass.

    The context ``x`` (shape [1, D]) is fixed; only the lead time varies. Tiling
    ``x`` to the batch dimension and passing a Dt vector runs every lead time
    together instead of one-at-a-time, which is dramatically faster on GPU and
    numerically identical to the per-point loop. Chunked at ``max_batch`` so a
    long lead-time sweep cannot exhaust GPU memory.

    Args:
        model: The 9-band scalar-temporal LodeRunner.
        x (torch.Tensor): Context input of shape [1, D].
        lead_times (np.ndarray): 1-D array of lead times (days).
        device (torch.device): Device to run on.
        max_batch (int): Maximum lead times evaluated per forward pass.
        return_quantiles (bool): When True and the model has a quantile head,
            return the FULL quantile axis [len(lead_times), n_quantiles, N_BANDS]
            instead of collapsing to the median. For a point head the axis is
            length 1. Used by the DIRECT scoring path to emit 0.1/0.9 bands; the
            curve sweep and rollout leave it False (median point forecast).

    Returns:
        np.ndarray: Normalized predictions, shape [len(lead_times), N_BANDS] when
        ``return_quantiles`` is False, else [len(lead_times), Q, N_BANDS].
    """
    lead_times = np.asarray(lead_times, dtype=np.float32)
    median_idx = getattr(model, "median_idx", 0)
    chunks = []
    with torch.inference_mode():
        for start in range(0, lead_times.shape[0], max_batch):
            chunk = lead_times[start : start + max_batch]
            x_batch = x.expand(chunk.shape[0], -1)
            Dt = torch.tensor(chunk, dtype=torch.float32, device=device)
            pred = model(x_batch, in_vars=None, out_vars=None, Dt=Dt)
            # Quantile head returns [B, n_quantiles, N_BANDS]; point head returns
            # [B, N_BANDS]. Normalize to a quantile axis of length >= 1 so the two
            # heads share one code path.
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)  # [B, 1, N_BANDS]
            if not return_quantiles:
                pred = pred[:, median_idx : median_idx + 1, :]  # keep median only
            chunks.append(pred.detach().cpu().numpy())
    out = np.concatenate(chunks, axis=0)  # [P, Q_or_1, N_BANDS]
    if not return_quantiles:
        return out[:, 0, :]  # [P, N_BANDS] -- unchanged contract for callers
    return out  # [P, Q, N_BANDS]


def load_interval_calibration(path):
    """Load per-band interval scales written by ``calibrate_intervals.py``.

    Args:
        path (str): Calibration JSON (``{"interval", "bands": {name: {k_lo,
            k_hi}}}``).

    Returns:
        dict: ``interval`` plus ``k_lo`` / ``k_hi`` arrays in BAND_NAMES order;
        bands absent from the JSON keep a scale of 1 (raw quantiles).
    """
    with open(path) as fh:
        cal = json.load(fh)
    k_lo = np.ones(N_BANDS, dtype=np.float32)
    k_hi = np.ones(N_BANDS, dtype=np.float32)
    for b, name in enumerate(BAND_NAMES):
        if name in cal["bands"]:
            k_lo[b] = cal["bands"][name]["k_lo"]
            k_hi[b] = cal["bands"][name]["k_hi"]
    return {"interval": float(cal["interval"]), "k_lo": k_lo, "k_hi": k_hi}


def interval_bounds(q, levels, median_idx, interval):
    """Lower/upper bounds of a central ``interval`` from the quantile axis.

    When the head emits the exact ``(1 - interval) / 2`` and ``(1 + interval) / 2``
    quantiles they are returned as-is. Otherwise the outer quantiles are widened
    (or narrowed) about the median by the Gaussian ratio
    ``z_{(1+interval)/2} / z_{level}``, separately on each side so any skew in the
    learned band is kept. E.g. a (0.1, 0.5, 0.9) head plotted at 90% scales each
    half-width by 1.645 / 1.282 = 1.28. This is an approximation -- the model only
    learned the quantiles it was trained on.

    Args:
        q (np.ndarray): Predictions with the quantile axis at dim 1, [P, Q, ...].
        levels (list | None): Quantile level of each index on the Q axis.
        median_idx (int): Index of the median on the Q axis.
        interval (float): Central coverage to plot, in (0, 1).

    Returns:
        tuple: ``(low, high, exact)`` -- arrays shaped like ``q[:, 0]`` and whether
        the bounds are learned quantiles (True) or Gaussian-scaled (False).
    """
    n_q = q.shape[1]
    if n_q == 1 or levels is None:
        return q[:, 0], q[:, n_q - 1], True
    lo_p, hi_p = 0.5 * (1.0 - interval), 0.5 * (1.0 + interval)
    levels = [float(v) for v in levels]
    for i, p_lo in enumerate(levels):
        for j, p_hi in enumerate(levels):
            if abs(p_lo - lo_p) < 1e-6 and abs(p_hi - hi_p) < 1e-6:
                return q[:, i], q[:, j], True
    med = q[:, median_idx]
    z_target = NormalDist().inv_cdf(hi_p)
    k_lo = z_target / -NormalDist().inv_cdf(levels[0])
    k_hi = z_target / NormalDist().inv_cdf(levels[-1])
    low = med - k_lo * (med - q[:, 0])
    high = med + k_hi * (q[:, n_q - 1] - med)
    return low, high, False


def _rollout_scored(
    model: torch.nn.Module,
    device: torch.device,
    means: np.ndarray,
    stds: np.ndarray,
    ctx_t0: list,
    ctx_v0: list,
    ctx_b0: list,
    target_t: np.ndarray,
    target_v: np.ndarray,
    target_b: np.ndarray,
    t0: float,
    context_window_days: float,
    max_context_len: int,
    ctx_ul0: list = None,
    obj_redshift: float = np.nan,
) -> list:
    """Autoregressive late-time forecast: feed each prediction back as context.

    Steps through the chronologically-ordered late-time dense targets. At each
    step the trailing-window context is rebuilt from the growing lists, the model
    predicts all bands at the lead time ``target_t[k] - ctx_t[-1]`` (from the last
    FED event, not the fixed last realistic detection), the target band's residual
    is recorded, then the model's own NORMALIZED prediction for that band is
    appended to the context -- matching the training rollout (``pred_obs.detach()``)
    and ``get_rollout_from_stream`` (``pred_norm``). Produces the same per-point
    ``scored`` dicts as the direct path, so plotting/CSV/aggregation are unchanged.

    Args:
        model: The 9-band scalar-temporal LodeRunner.
        device (torch.device): Device to run on.
        means (np.ndarray): Per-band normalization means.
        stds (np.ndarray): Per-band normalization standard deviations.
        ctx_t0 (list): Seed context absolute times (pre-cutoff realistic).
        ctx_v0 (list): Seed context NORMALIZED values (pre-cutoff realistic).
        ctx_b0 (list): Seed context band indices (pre-cutoff realistic).
        target_t (np.ndarray): Late-time dense target absolute times, chronological.
        target_v (np.ndarray): Late-time dense target magnitudes.
        target_b (np.ndarray): Late-time dense target band indices.
        t0 (float): Phase-zero time (first realistic detection).
        context_window_days (float): Trailing lookback W.
        max_context_len (int): Padded context width M.
        ctx_ul0 (list): Seed context upper-limit flags (1.0 for a non-detection),
            or None when the model has no is_upper_limit channel. Fed-back
            predictions are appended with flag 0.0 (a prediction is a detection,
            not a bound).

    Returns:
        list: One scored dict per target (phase/lead_time/band/pred_mag/true_mag/
        residual_mag).
    """
    ul_channel = getattr(model, "upper_limit_channel", False)
    ctx_t = list(ctx_t0)
    ctx_v = list(ctx_v0)
    ctx_b = list(ctx_b0)
    ctx_ul = list(ctx_ul0) if ctx_ul0 is not None else None

    scored = []
    with torch.no_grad():
        for k in range(target_t.shape[0]):
            win = _select_window(
                ctx_t=ctx_t,
                ctx_v=ctx_v,
                ctx_b=ctx_b,
                context_window_days=context_window_days,
                max_context_len=max_context_len,
                ctx_ul=ctx_ul if ul_channel else None,
            )
            if ul_channel:
                win_v, win_t, win_b, win_ul = win
            else:
                win_v, win_t, win_b = win
                win_ul = None
            x = build_context_input(
                win_v=win_v,
                win_t=win_t,
                win_b=win_b,
                context_len=max_context_len,
                n_bands=N_BANDS,
                device=device,
                window_mode=True,
                phase_fourier_bands=getattr(model, "phase_fourier_bands", 0),
                phase0=t0,  # win_t is absolute MJD; first detection at t0
                win_ul=win_ul,
                upper_limit_channel=ul_channel,
                redshift_fourier_bands=getattr(model, "redshift_fourier_bands", 0),
                redshift=obj_redshift,
                redshift_pivot_direct=getattr(
                    model, "redshift_pivot_direct", False
                ),
            )
            # Lead time from the last FED event (the running context tip).
            dt = float(target_t[k]) - float(ctx_t[-1])
            if dt <= 0:
                # Non-increasing time; skip feeding but still score at a tiny dt.
                dt = max(dt, 1e-3)
            Dt = torch.tensor([dt], dtype=torch.float32, device=device)
            pred_t = model(x, in_vars=None, out_vars=None, Dt=Dt)
            # Quantile head returns [B, n_quantiles, N_BANDS]; take the median as
            # the point forecast. Point head returns [B, N_BANDS] (no-op).
            if pred_t.dim() == 3:
                pred_t = pred_t[:, getattr(model, "median_idx", 0), :]
            pred_all = pred_t.reshape(N_BANDS).detach().cpu().numpy()
            band = int(target_b[k])
            pred_norm = float(pred_all[band])
            pred_mag = pred_norm * (stds[band] + EPS) + means[band]
            true_mag = float(target_v[k])
            scored.append(
                {
                    "phase": float(target_t[k]) - t0,
                    "lead_time": dt,
                    "band": band,
                    "pred_mag": float(pred_mag),
                    "true_mag": true_mag,
                    "residual_mag": float(pred_mag) - true_mag,
                }
            )
            # Feed the NORMALIZED prediction back as the next context event. A
            # fed-back prediction is a (pseudo-)detection, never a bound, so its
            # is_upper_limit flag is 0.
            ctx_t.append(float(target_t[k]))
            ctx_v.append(pred_norm)
            ctx_b.append(band)
            if ctx_ul is not None:
                ctx_ul.append(0.0)

    return scored


def eval_object(
    real_stream,
    dense_stream,
    model,
    device,
    means,
    stds,
    context_window_days,
    max_context_len,
    late_time_cutoff_days,
    late_time_max_days,
    uniform_stream: tuple | None = None,
    rollout: bool = False,
    probe_dense_context: bool = False,
    plot_interval: float = 0.9,
    interval_calibration: dict | None = None,
    need_curve: bool = True,
):
    """Score one object's late-time dense truth against a realistic-context forecast.

    Returns a dict with the scored late-time points and a smooth forecast curve,
    or None if the object cannot be evaluated (no realistic context, or no dense
    points in the late-time region).

    The scored/forecast region is the phase band
    ``late_time_cutoff_days < phase <= late_time_max_days`` (phase measured from
    the first realistic detection). Points beyond ``late_time_max_days`` are
    ignored so the forecast is only judged over a horizon we care about.

    Two forecast modes:

    * DIRECT (``rollout=False``, default): the pre-cutoff realistic context is
      fixed, and every late-time point is predicted in one batched pass at its
      true lead time from the last realistic detection. No feedback -- this
      measures the model's raw Dt-conditioned response.
    * AUTOREGRESSIVE (``rollout=True``): the context grows -- at each late-time
      point (chronological order) the model predicts, and its own (normalized)
      prediction is appended to the context before the next step, exactly as the
      training rollout and ``get_rollout_from_stream`` do. Each step's ``Dt`` is
      measured from the last FED event, not the fixed last realistic detection.
      This measures the true inference path (and exposes drift).

    ``interval_calibration`` (from ``load_interval_calibration``) rescales the raw
    outer quantiles per band about the median (``calibrate_intervals.py``). When
    given, scored points gain ``pred_low_cal`` / ``pred_high_cal`` and the plotted
    band is the calibrated one (``plot_interval`` is then ignored).

    ``need_curve=False`` skips the smooth plotting curve (its ``curve_*`` keys
    are None); the scored points are unchanged. Use it for objects that will
    not be plotted.
    """
    r_t, r_v, r_b, r_ul, r_z = real_stream
    d_t, d_v, d_b, d_ul, _d_z = dense_stream

    if r_t.shape[0] < 1 or d_t.shape[0] < 1:
        return None

    # Whether the model consumes an is_upper_limit channel (set at train time).
    # When True the realistic context stream retains its upper limits (flagged);
    # when False they were already dropped at read time.
    ul_channel = getattr(model, "upper_limit_channel", False)

    # Phase zero = the first REALISTIC detection (the observed trigger). Kept as
    # the realistic trigger even under the dense-context probe, so the scored
    # late-time region (below) is IDENTICAL to the normal eval and the RMSE is
    # directly comparable -- only the context SOURCE changes.
    t0 = float(r_t[0])

    # Context source. Normally the realistic stream (matches deployment). Under
    # the dense-context ceiling probe (study 089), the context is drawn from the
    # DENSE stream instead -- the smooth, densely-sampled curve the model was
    # trained on with PROBE_DENSE_CONTEXT=True -- still truncated at the same
    # cutoff below, so it is dense-WITHIN-window, not leakage toward the scored
    # points. drop_upper_limits was already applied when the streams were read.
    if probe_dense_context:
        c_t, c_v, c_b, c_ul = d_t, d_v, d_b, d_ul
    else:
        c_t, c_v, c_b, c_ul = r_t, r_v, r_b, r_ul

    # The cutoff splits context from forecast: the model may only see context
    # detections up to the cutoff phase, and must FORECAST everything after it
    # (scored against the dense truth). Truncating the context here -- rather
    # than feeding the whole stream and only scoring late points -- makes every
    # object forecast from the same phase boundary, instead of from wherever its
    # coverage happens to end. (Without this, a bright/well-covered object whose
    # detections run to ~14 d has an almost-zero forecast horizon and the curve
    # collapses to a stub.)
    ctx_mask = (c_t - t0) <= late_time_cutoff_days
    if not np.any(ctx_mask):
        return None
    r_t_ctx = c_t[ctx_mask]
    r_v_ctx = c_v[ctx_mask]
    r_b_ctx = c_b[ctx_mask]
    r_ul_ctx = c_ul[ctx_mask]
    last_real_t = float(r_t_ctx[-1])

    # Score the forecast only within the phase band cutoff < phase <= max_days.
    d_phase = d_t - t0
    late_mask = (d_phase > late_time_cutoff_days) & (d_phase <= late_time_max_days)
    if not np.any(late_mask):
        return None

    # Seed context from the truncated context stream (realistic, or dense under
    # the probe): trailing window ending at the last pre-cutoff detection,
    # normalized as in training. build_context_input subtracts win_t[0], so
    # absolute times are fine here.
    r_v_norm = (r_v_ctx - means[r_b_ctx]) / (stds[r_b_ctx] + EPS)
    win = _select_window(
        ctx_t=list(r_t_ctx.astype(np.float32)),
        ctx_v=list(r_v_norm.astype(np.float32)),
        ctx_b=list(r_b_ctx),
        context_window_days=context_window_days,
        max_context_len=max_context_len,
        ctx_ul=list(r_ul_ctx.astype(np.float32)) if ul_channel else None,
    )
    if ul_channel:
        win_v, win_t, win_b, win_ul = win
    else:
        win_v, win_t, win_b = win
        win_ul = None
    x = build_context_input(
        win_v=win_v,
        win_t=win_t,
        win_b=win_b,
        context_len=max_context_len,
        n_bands=N_BANDS,
        device=device,
        window_mode=True,
        phase_fourier_bands=getattr(model, "phase_fourier_bands", 0),
        phase0=t0,  # win_t is absolute MJD; first realistic detection at t0
        win_ul=win_ul,
        upper_limit_channel=ul_channel,
        redshift_fourier_bands=getattr(model, "redshift_fourier_bands", 0),
        redshift=r_z,
        redshift_pivot_direct=getattr(model, "redshift_pivot_direct", False),
    )

    # Score each late-time dense point at its true lead time from the last
    # realistic detection. The context ``x`` is fixed for this object, so all
    # lead times are evaluated in a SINGLE batched forward pass (tile x to the
    # batch dimension, pass a Dt vector) instead of one forward per point --
    # numerically identical, but far faster on GPU.
    late_idx = np.nonzero(late_mask)[0]
    lead_times = (d_t[late_idx] - last_real_t).astype(np.float32)
    # Only points strictly after the last realistic detection are forecasts.
    keep = lead_times > 0
    late_idx = late_idx[keep]
    lead_times = lead_times[keep]

    if late_idx.shape[0] == 0:
        return None

    # Uniform-truth points (DIRECT only), scored below. The uniform grid shares
    # epochs across bands: forecast each unique lead time once, then pick each
    # point's band.
    u_idx = None
    if uniform_stream is not None and not rollout:
        u_t, u_v, u_b, _u_ul, _u_z = uniform_stream
        u_phase = u_t - t0
        u_lead = u_t - last_real_t
        u_mask = (
            (u_phase > late_time_cutoff_days)
            & (u_phase <= late_time_max_days)
            & (u_lead > 0)
            & np.isfinite(u_v)
        )
        if np.any(u_mask):
            u_idx = np.nonzero(u_mask)[0]
            uniq, inv = np.unique(
                u_lead[u_idx].astype(np.float32), return_inverse=True
            )

    # Smooth forecast curve lead times for plotting: 0 to the farthest scored
    # late-time point.
    lead_grid = None
    if need_curve:
        max_dt = float(lead_times.max())
        lead_grid = np.linspace(0.0, max_dt, 60).astype(np.float32)

    # Every lead time above is forecast from the SAME fixed context x, so run
    # them all in ONE batched forward and split the rows afterwards (each row
    # depends only on x and its own Dt). The rollout path builds its own
    # contexts, so only the curve comes from here.
    segments = []
    if not rollout:
        segments.append(lead_times)
    if u_idx is not None:
        segments.append(uniq)
    if lead_grid is not None:
        segments.append(lead_grid)
    all_q = None
    if segments:
        all_q = _batched_forward(
            model, x, np.concatenate(segments), device, return_quantiles=True
        )  # [sum(P), Q, N_BANDS]
    offset = 0
    if not rollout:
        pred_q = all_q[offset : offset + lead_times.shape[0]]
        offset += lead_times.shape[0]
    if u_idx is not None:
        u_q = all_q[offset : offset + uniq.shape[0]][inv]
        offset += uniq.shape[0]
    if lead_grid is not None:
        curve_q = all_q[offset : offset + lead_grid.shape[0]]

    if rollout:
        # AUTOREGRESSIVE: grow the context, feeding each (normalized) prediction
        # back before the next step. Mirrors get_rollout_from_stream and the
        # training rollout. Each step's Dt is from the last FED event's time.
        scored = _rollout_scored(
            model=model,
            device=device,
            means=means,
            stds=stds,
            ctx_t0=list(r_t_ctx.astype(np.float32)),
            ctx_v0=list(r_v_norm.astype(np.float32)),
            ctx_b0=list(r_b_ctx),
            ctx_ul0=(
                list(r_ul_ctx.astype(np.float32)) if ul_channel else None
            ),
            target_t=d_t[late_idx].astype(np.float32),
            target_v=d_v[late_idx].astype(np.float32),
            target_b=d_b[late_idx].astype(np.int64),
            t0=t0,
            context_window_days=context_window_days,
            max_context_len=max_context_len,
            obj_redshift=r_z,
        )
    else:
        # Keep the full quantile axis so the 0.1/0.9 bands can be scored/plotted.
        # For a point head Q == 1 and low/high collapse to the median (no-op).
        # pred_q [P, Q, N_BANDS] came from the shared forward above.
        median_idx = getattr(model, "median_idx", 0)
        n_q = pred_q.shape[1]
        low_idx, high_idx = 0, n_q - 1  # outer quantiles (== median when Q == 1)
        scored = []
        for j, idx in enumerate(late_idx):
            band = int(d_b[idx])
            sb = stds[band] + EPS
            pred_mag = float(pred_q[j, median_idx, band] * sb + means[band])
            pred_low = float(pred_q[j, low_idx, band] * sb + means[band])
            pred_high = float(pred_q[j, high_idx, band] * sb + means[band])
            true_mag = float(d_v[idx])
            point = {
                "phase": float(d_t[idx]) - t0,
                "lead_time": float(lead_times[j]),
                "band": band,
                "pred_mag": pred_mag,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "true_mag": true_mag,
                "residual_mag": pred_mag - true_mag,
            }
            if interval_calibration is not None:
                k_lo = interval_calibration["k_lo"][band]
                k_hi = interval_calibration["k_hi"][band]
                point["pred_low_cal"] = pred_mag - k_lo * (pred_mag - pred_low)
                point["pred_high_cal"] = pred_mag + k_hi * (pred_high - pred_mag)
            scored.append(point)

    # Uniform-truth score (DIRECT only): the same forecast scored against the
    # noise-free, no-limiting-mag uniform grid over the same phase window and the
    # same lead-time rule. The dense truth is survey-depth limited, so it never
    # scores the forecast where an object has faded out of reach (e.g. faint-late
    # ZTF); this does. Diagnostic only -- the dense score stays the headline.
    uniform_scored = None
    if u_idx is not None:
        # u_q [P, Q, N_BANDS] came from the shared forward above.
        u_band = u_b[u_idx].astype(np.int64)
        rows = np.arange(u_idx.shape[0])
        sb = stds[u_band] + EPS
        mb = means[u_band]
        median_idx = getattr(model, "median_idx", 0)
        n_q = u_q.shape[1]
        uniform_scored = {
            "band": u_band,
            "phase": u_phase[u_idx].astype(np.float32),
            "lead_time": u_lead[u_idx].astype(np.float32),
            "pred_mag": u_q[rows, median_idx, u_band] * sb + mb,
            "pred_low": u_q[rows, 0, u_band] * sb + mb,
            "pred_high": u_q[rows, n_q - 1, u_band] * sb + mb,
            "true_mag": u_v[u_idx].astype(np.float32),
        }

    # Smooth forecast curve for plotting, predicting all bands at each lead
    # time of lead_grid (curve_q [60, Q, N_BANDS] from the shared forward).
    curve_phase = curve_mag = curve_low_mag = curve_high_mag = None
    interval_label = None
    if need_curve:
        median_idx = getattr(model, "median_idx", 0)
        n_q = curve_q.shape[1]
        curve = curve_q[:, median_idx, :]  # [60, N_BANDS] median point forecast
        curve_mag = curve * (stds[None, :] + EPS) + means[None, :]
        # Central-interval curves for the shaded uncertainty band (== median
        # when Q==1, so the band has zero width for a point head and nothing is
        # drawn). Only the plot uses plot_interval; the scored CSV keeps the raw
        # outer quantiles so quantile_coverage.py stays valid.
        curve_low, curve_high, interval_exact = interval_bounds(
            curve_q, getattr(model, "quantile_levels", None), median_idx,
            plot_interval,
        )
        curve_low_mag = curve_low * (stds[None, :] + EPS) + means[None, :]
        curve_high_mag = curve_high * (stds[None, :] + EPS) + means[None, :]
        interval_label = (
            f"forecast {plot_interval:.0%}"
            + ("" if interval_exact else " (Gaussian-scaled)")
        )
        if interval_calibration is not None:
            # Calibrated band: per-band scales applied to the RAW outer quantiles.
            raw_low = curve_q[:, 0, :] * (stds[None, :] + EPS) + means[None, :]
            raw_high = (curve_q[:, n_q - 1, :] * (stds[None, :] + EPS)
                        + means[None, :])
            k_lo = interval_calibration["k_lo"][None, :]
            k_hi = interval_calibration["k_hi"][None, :]
            curve_low_mag = curve_mag - k_lo * (curve_mag - raw_low)
            curve_high_mag = curve_mag + k_hi * (raw_high - curve_mag)
            interval_label = (
                f"forecast {interval_calibration['interval']:.0%} (calibrated)"
            )
        curve_phase = (last_real_t - t0) + lead_grid
        curve_mag = curve_mag.astype(np.float32)
        curve_low_mag = curve_low_mag.astype(np.float32)
        curve_high_mag = curve_high_mag.astype(np.float32)

    # Optional uniform-grid "true curve" for plotting, phase-aligned to the same
    # t0 (first realistic detection) so it overlays in the same frame. Never
    # scored or fed to the model -- it is the noise-free, no-limiting-mag target
    # curve, shown so the forecast is readable even where the survey-limited
    # dense truth goes dark.
    uniform = None
    if uniform_stream is not None:
        u_t, u_v, u_b, _u_ul, _u_z = uniform_stream
        if u_t.shape[0] > 0:
            uniform = (u_t - t0, u_v, u_b)

    return {
        "scored": scored,
        "uniform_scored": uniform_scored,
        "t0": t0,
        "last_real_t": last_real_t,
        "curve_phase": curve_phase,
        "curve_mag": curve_mag,
        "curve_low_mag": curve_low_mag,
        "curve_high_mag": curve_high_mag,
        "interval_label": interval_label,
        # Only the pre-cutoff realistic detections were shown to the model, so
        # plot those as the context (not the full realistic stream).
        "real": (r_t_ctx - t0, r_v_ctx, r_b_ctx),
        "dense": (d_t - t0, d_v, d_b),
        "uniform": uniform,
        # Right edge for plotting: the scored horizon. Beyond this the forecast
        # is unsupervised extrapolation, so it is not shown.
        "plot_max_phase": late_time_max_days,
    }


def plot_object(result, stem, outpath):
    """Plot realistic context, dense truth, and the late-time forecast per band."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    axes = axes.ravel()
    r_ph, r_v, r_b = result["real"]
    d_ph, d_v, d_b = result["dense"]
    uniform = result.get("uniform")
    # Show only the scored horizon; the forecast beyond it is unsupervised
    # extrapolation (where the late-time upturn artifact lives).
    plot_max = result.get("plot_max_phase")

    for b in range(N_BANDS):
        ax = axes[b]
        rm = r_b == b
        dm = d_b == b
        # Clip the dense-truth scatter to the plotted horizon as well.
        if plot_max is not None:
            dm = dm & (d_ph <= plot_max)
        if np.any(dm):
            ax.scatter(d_ph[dm], d_v[dm], s=14, c="0.6", label="dense truth")
        # Uniform-grid true curve: continuous line so forecast-vs-truth reads as
        # curve-vs-curve, including where the survey-limited dense truth has no
        # points. Clip to the plotted horizon and sort by phase for a clean line.
        if uniform is not None:
            u_ph, u_v, u_b = uniform
            um = u_b == b
            if plot_max is not None:
                um = um & (u_ph <= plot_max)
            if np.any(um):
                order = np.argsort(u_ph[um])
                ax.plot(
                    u_ph[um][order], u_v[um][order],
                    c="0.4", lw=1.0, ls="--", alpha=0.9, label="uniform truth",
                )
        if np.any(rm):
            ax.scatter(
                r_ph[rm], r_v[rm], s=26, c=BAND_COLORS[b],
                edgecolor="k", linewidth=0.4, label="realistic ctx",
            )
        # Shaded central interval (--plot_interval) when the model has a quantile
        # head. For a point head low == high == median, so skip a zero-width band.
        c_low = result.get("curve_low_mag")
        c_high = result.get("curve_high_mag")
        if (
            c_low is not None
            and c_high is not None
            and np.any(np.abs(c_high[:, b] - c_low[:, b]) > 1e-6)
        ):
            ax.fill_between(
                result["curve_phase"], c_low[:, b], c_high[:, b],
                color=BAND_COLORS[b], alpha=0.2, linewidth=0,
                label=result.get("interval_label", "forecast interval"),
            )
        ax.plot(
            result["curve_phase"], result["curve_mag"][:, b],
            c=BAND_COLORS[b], lw=1.6, label="forecast",
        )
        ax.invert_yaxis()  # magnitudes: brighter is smaller
        if plot_max is not None:
            ax.set_xlim(right=plot_max)
        ax.set_title(BAND_NAMES[b], fontsize=9)
        if b == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Dense late-time forecast: {stem}")
    fig.supxlabel("Phase from first realistic detection [days]")
    fig.supylabel("Magnitude")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=int, default=24)
    p.add_argument(
        "--epoch",
        type=int,
        default=500,
        help="Epoch BUDGET. When --ckpt is not given, the eval loads the "
        "lowest-median-val-loss checkpoint at or below this epoch (see "
        "resolve_best_checkpoint), NOT necessarily this exact epoch. Pass "
        "--exact_epoch to force the literal epoch instead.",
    )
    p.add_argument(
        "--exact_epoch",
        action="store_true",
        help="Load exactly --epoch's checkpoint (the legacy behavior) instead "
        "of the best val-loss checkpoint at or below it. Ignored when --ckpt is "
        "given explicitly.",
    )
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument(
        "--use_ema",
        action="store_true",
        help="Overlay the EMA (Polyak) shadow of the trainable params instead "
        "of the raw weights. Falls back to raw weights if the checkpoint has "
        "no EMA shadow.",
    )
    p.add_argument(
        "--realistic_glob",
        type=str,
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the realistic light-curve files (observing context).",
    )
    p.add_argument(
        "--dense_glob",
        type=str,
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_dense_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the dense light-curve files (late-time truth). Defaults to "
        "the same dense set the model was trained on "
        "(rubin_ztf_dense_10000_dataset_same_seed), whose Rubin bands reach "
        "~11-12 d median so the 2->10 d scored region is well covered.",
    )
    p.add_argument(
        "--uniform_glob",
        type=str,
        default=(
            "/net/sescratch1/exempt/artimis/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_uniform_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Optional glob for a UNIFORM-grid, noise-free, no-limiting-mag "
        "companion set (same objects/seed, sampled on a dense regular phase "
        "grid). When it matches files, each per-object plot overlays this as a "
        "continuous 'true curve' line so the forecast can be read curve-vs-curve "
        "even where the survey-limited dense truth has no detections (e.g. deep "
        "late-time u/g). Also scored as a SECONDARY diagnostic (test split, "
        "DIRECT mode): uniform-truth RMSE/bias per band, split within / beyond "
        "the dense depth, plus latetime_uniform_scored_points.csv. The dense "
        "score stays the headline; never fed to the model. Silently skipped if "
        "no files match.",
    )
    p.add_argument(
        "--test_filelist",
        type=str,
        default=os.path.join(FILELIST_DIR, "kn_rubin_ztf_test.txt"),
        help="Path to the test-split stem list (one object stem per line). Pass "
        "'' to evaluate all objects present in BOTH globs (no calibration).",
    )
    p.add_argument(
        "--norm_stats_path",
        type=str,
        default="kilonova_9band_norm_stats_trainonly.npz",
        help="Train-only normalization stats the model was trained with.",
    )
    p.add_argument(
        "--late_time_cutoff_days",
        type=float,
        default=2.0,
        help="Splits context from forecast. The model sees realistic detections "
        "with phase (from first realistic detection) up to this value, and "
        "forecasts all dense points after it -- the late-time region scored here. "
        "Defaults to 2.0 to match the model's trained context_window_days (a "
        "2-day trailing lookback), so the eval feeds the model the same context "
        "span it saw in training rather than a wider ->3-day slice.",
    )
    p.add_argument(
        "--late_time_max_days",
        type=float,
        default=10.0,
        help="Upper bound (phase from first realistic detection) on the scored "
        "forecast region. Dense points beyond this are ignored, so the forecast "
        "is judged only over cutoff < phase <= this horizon. Study 101: restored "
        "7 -> 10 d (TARGET_HORIZON_DAYS=8, i.e. lead 8 + 2 d context = phase 10) so "
        "the scored region is 2 < phase <= 10, matching the 080/093 champion and "
        "all studies <=094. For a 7 d-horizon model (studies 095-100) pass "
        "--late_time_max_days 7.0 to match its training horizon.",
    )
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help="Cap the number of objects evaluated (0 = all).",
    )
    p.add_argument(
        "--n_plots",
        type=int,
        default=12,
        help="Number of per-object forecast plots to write.",
    )
    p.add_argument(
        "--plot_interval",
        type=float,
        default=0.9,
        help="Central coverage of the shaded forecast band in the per-object "
        "plots. Uses the learned quantiles when the head emits them exactly; "
        "otherwise Gaussian-scales the outer quantiles about the median (e.g. a "
        "0.1/0.5/0.9 head at 0.9 -> half-widths x1.28). Plot only -- the scored "
        "CSV keeps the raw outer quantiles.",
    )
    p.add_argument(
        "--interval_calibration",
        type=str,
        default=None,
        help="Per-band interval scales JSON from calibrate_intervals.py (fit on "
        "the VALIDATION split). When given, the plots shade the calibrated band "
        "(overriding --plot_interval) and the CSV gains pred_low_cal / "
        "pred_high_cal. The median, and so RMSE, is unchanged. DIRECT mode only. "
        "Reuses a saved JSON instead of fitting (skips --calibrate_filelist).",
    )
    p.add_argument(
        "--calibrate_filelist",
        type=str,
        default=os.path.join(FILELIST_DIR, "kn_rubin_ztf_val.txt"),
        help="Calibration (VALIDATION) split stem list. By default the eval runs "
        "in one go: eval this split, fit per-band interval scales on it "
        "(calibrate_intervals.fit_scales), write <outdir>/interval_calibration."
        "json, then eval + plot the --test_filelist split with the calibrated "
        "band. Must be disjoint from --test_filelist. Skipped with "
        "--no_calibration, --interval_calibration, or --rollout.",
    )
    p.add_argument(
        "--calibration_interval",
        type=float,
        default=0.9,
        help="Target central coverage of the calibrated band.",
    )
    p.add_argument(
        "--no_calibration",
        action="store_true",
        help="Skip the calibration split: test eval with the raw quantile band.",
    )
    p.add_argument(
        "--rollout",
        action="store_true",
        help="Score the AUTOREGRESSIVE forecast: feed each prediction back as "
        "context before the next late-time point (the true inference path), "
        "instead of the default DIRECT single-pass forecast from a fixed "
        "pre-cutoff context. Comparing the two isolates rollout drift.",
    )
    p.add_argument(
        "--probe_dense_context",
        action="store_true",
        help="Study 089 dense-context CEILING probe. Build the model's CONTEXT "
        "from the DENSE set (same object, dense_glob) instead of the realistic "
        "set, still truncated at --late_time_cutoff_days (dense-WITHIN-window, "
        "not leakage toward the scored points). Scoring is unchanged (dense truth "
        "at cutoff < phase <= max_days). Use this to eval a model TRAINED with "
        "PROBE_DENSE_CONTEXT=True, so train and eval both feed dense context and "
        "the number is the true dense-context ceiling. NOT the deployable path.",
    )
    return p.parse_args()


def _read_stems(path):
    """Stems listed one per line in ``path``."""
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def run_split(
    args,
    model,
    device,
    means,
    stds,
    context_window_days,
    max_context_len,
    real_map,
    dense_map,
    uniform_map,
    split_stems,
    outdir,
    split_label,
    interval_calibration=None,
    n_plots=0,
    score_uniform=True,
):
    """Evaluate one split, print its summary, and write its CSV / lead plot.

    Args:
        args: Parsed CLI args (cutoffs, rollout, probe, max_objects, ...).
        model, device, means, stds, context_window_days, max_context_len: The
            loaded model and the encoding it was trained with.
        real_map, dense_map, uniform_map (dict): Stem -> npz path per set.
        split_stems (set | None): Objects to evaluate (None = every paired stem).
        outdir (str): Where the CSV, lead plot, and per-object plots go.
        split_label (str): Name printed in the summary (e.g. "test").
        interval_calibration (dict | None): Per-band scales for the band.
        n_plots (int): Number of per-object plots to write.
        score_uniform (bool): Also score against the uniform-grid truth (when
            ``uniform_map`` has the object) and write its summary and CSV.

    Returns:
        list | None: Scored points (dicts with ``stem``), or None if none scored.
    """
    os.makedirs(outdir, exist_ok=True)
    stems = sorted(set(real_map) & set(dense_map))
    if split_stems is not None:
        stems = [s for s in stems if s in split_stems]
    print(f"\n[{split_label}] paired & in-split objects: {len(stems)}")
    if args.max_objects > 0:
        stems = stems[: args.max_objects]

    all_scored = []
    uniform_scored = []  # per-object dicts of arrays (uniform-truth score)
    plotted = 0
    n_eval = 0
    # When the model was trained with the flagged-UL context channel, the
    # realistic (context) stream must KEEP upper limits so they reach the
    # flagged-context path; otherwise drop them as before. Auto-detected from
    # the checkpoint metadata so eval matches training.
    ul_channel = getattr(model, "upper_limit_channel", False)
    real_drop_ul = DROP_UPPER_LIMITS and not ul_channel

    # The uniform stream only feeds the uniform score and the plots, so it is
    # not read when the split needs neither (the calibration pass).
    read_uniform = score_uniform or n_plots > 0

    def read_streams(stem):
        real_stream = read_merged_stream(real_map[stem], real_drop_ul)
        dense_stream = read_merged_stream(dense_map[stem], drop_upper_limits=False)
        uniform_stream = None
        if read_uniform and stem in uniform_map:
            uniform_stream = read_merged_stream(
                uniform_map[stem], drop_upper_limits=False
            )
        return real_stream, dense_stream, uniform_stream

    # Read the npz files on a thread pool, a bounded window ahead of the model,
    # so file IO overlaps the forward passes. Objects are still consumed in
    # stem order, so the output is unchanged.
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        pending = deque()
        next_i = 0
        for stem in stems:
            while next_i < len(stems) and len(pending) < IO_PREFETCH:
                pending.append(pool.submit(read_streams, stems[next_i]))
                next_i += 1
            real_stream, dense_stream, uniform_stream = pending.popleft().result()
            result = eval_object(
                real_stream,
                dense_stream,
                model,
                device,
                means,
                stds,
                context_window_days,
                max_context_len,
                args.late_time_cutoff_days,
                args.late_time_max_days,
                uniform_stream=uniform_stream,
                rollout=args.rollout,
                probe_dense_context=args.probe_dense_context,
                plot_interval=args.plot_interval,
                interval_calibration=interval_calibration,
                # Only plotted objects need the smooth forecast curve.
                need_curve=plotted < n_plots,
            )
            if result is None:
                continue
            n_eval += 1
            for s in result["scored"]:
                s["stem"] = stem
                all_scored.append(s)
            if score_uniform and result["uniform_scored"] is not None:
                result["uniform_scored"]["stem"] = stem
                uniform_scored.append(result["uniform_scored"])
            if plotted < n_plots:
                plot_object(
                    result, stem,
                    os.path.join(outdir, f"latetime_{stem}.png"),
                )
                plotted += 1

    if not all_scored:
        print(f"[{split_label}] No late-time points scored (check the cutoff and "
              "globs).")
        return None

    # Per-band late-time error summary.
    resid = np.asarray([s["residual_mag"] for s in all_scored])
    bands = np.asarray([s["band"] for s in all_scored])
    mode = "AUTOREGRESSIVE rollout" if args.rollout else "DIRECT single-pass"
    print(f"\n[{split_label}] Forecast mode: {mode}")
    print(f"Evaluated {n_eval} objects; {len(all_scored)} late-time points "
          f"(cutoff {args.late_time_cutoff_days} d).")
    print(f"Overall late-time RMSE (mag): {np.sqrt(np.mean(resid**2)):.4f}  "
          f"MAE: {np.mean(np.abs(resid)):.4f}")
    print("Per-band late-time error (mag):")
    for b in range(N_BANDS):
        m = bands == b
        if np.any(m):
            print(f"  {BAND_NAMES[b]:>5}: n={m.sum():5d}  "
                  f"RMSE={np.sqrt(np.mean(resid[m]**2)):.4f}  "
                  f"MAE={np.mean(np.abs(resid[m])):.4f}  "
                  f"bias={np.mean(resid[m]):+.4f}")

    # Error vs lead time (binned) plot.
    lead = np.asarray([s["lead_time"] for s in all_scored])
    fig, ax = plt.subplots()
    edges = np.linspace(0, lead.max(), 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rmse_bin = np.full(centers.shape[0], np.nan)
    for i in range(centers.shape[0]):
        m = (lead >= edges[i]) & (lead < edges[i + 1])
        if np.any(m):
            rmse_bin[i] = np.sqrt(np.mean(resid[m] ** 2))
    ax.plot(centers, rmse_bin, "o-", label="dense truth (scored)")
    uni = _concat_uniform(uniform_scored) if uniform_scored else None
    if uni is not None:
        u_resid = uni["pred_mag"] - uni["true_mag"]
        u_rmse_bin = np.full(centers.shape[0], np.nan)
        for i in range(centers.shape[0]):
            m = (uni["lead_time"] >= edges[i]) & (uni["lead_time"] < edges[i + 1])
            if np.any(m):
                u_rmse_bin[i] = np.sqrt(np.mean(u_resid[m] ** 2))
        ax.plot(centers, u_rmse_bin, "s--", label="uniform truth")
        ax.legend()
    ax.set_xlabel("Lead time from last realistic detection [days]")
    ax.set_ylabel("Late-time forecast RMSE [mag]")
    ax.set_title("Dense late-time forecast error vs lead time")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latetime_rmse_vs_lead.png"), dpi=130)
    plt.close(fig)

    # Full per-point CSV.
    csv_path = os.path.join(outdir, "latetime_scored_points.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["stem", "band", "phase_days", "lead_time_days",
                "pred_mag", "pred_low", "pred_high", "true_mag", "residual_mag"]
        if interval_calibration is not None:
            cols += ["pred_low_cal", "pred_high_cal"]
        w.writerow(cols)
        for s in all_scored:
            # pred_low/high present only on the DIRECT quantile path; fall back to
            # the point forecast (rollout path, or a point head) so the columns are
            # always populated.
            row = [
                s["stem"], BAND_NAMES[s["band"]], f"{s['phase']:.4f}",
                f"{s['lead_time']:.4f}", f"{s['pred_mag']:.4f}",
                f"{s.get('pred_low', s['pred_mag']):.4f}",
                f"{s.get('pred_high', s['pred_mag']):.4f}",
                f"{s['true_mag']:.4f}", f"{s['residual_mag']:.4f}",
            ]
            if interval_calibration is not None:
                row += [f"{s['pred_low_cal']:.4f}", f"{s['pred_high_cal']:.4f}"]
            w.writerow(row)

    if interval_calibration is not None:
        # Calibrated coverage on THIS split's points (held out when the scales
        # were fit on the validation split).
        true = np.asarray([s["true_mag"] for s in all_scored])
        lo = np.asarray([s["pred_low_cal"] for s in all_scored])
        hi = np.asarray([s["pred_high_cal"] for s in all_scored])
        lo_raw = np.asarray([s["pred_low"] for s in all_scored])
        hi_raw = np.asarray([s["pred_high"] for s in all_scored])
        cov = (true >= lo) & (true <= hi)
        cov_raw = (true >= lo_raw) & (true <= hi_raw)
        print(f"\n[{split_label}] Interval coverage (target "
              f"{interval_calibration['interval']:.2f}): raw / calibrated")
        print(f"  {'ALL':>5}: {cov_raw.mean():.3f} / {cov.mean():.3f}")
        for b in range(N_BANDS):
            m = bands == b
            if np.any(m):
                print(f"  {BAND_NAMES[b]:>5}: {cov_raw[m].mean():.3f} / "
                      f"{cov[m].mean():.3f}")

    if uni is not None:
        dense_true = np.asarray([s["true_mag"] for s in all_scored])
        report_uniform(uni, bands, dense_true, split_label)
        write_uniform_csv(
            uni, os.path.join(outdir, "latetime_uniform_scored_points.csv")
        )

    print(f"\n[{split_label}] Wrote {plotted} per-object plots, the RMSE-vs-lead "
          f"plot, and {csv_path} in {outdir}")
    return all_scored


def _concat_uniform(per_object):
    """Concatenate per-object uniform-truth score dicts into flat arrays."""
    keys = ("band", "phase", "lead_time", "pred_mag", "pred_low", "pred_high",
            "true_mag")
    out = {k: np.concatenate([o[k] for o in per_object]) for k in keys}
    out["stem"] = np.concatenate(
        [np.full(o["band"].shape[0], o["stem"], dtype=object) for o in per_object]
    )
    return out


def report_uniform(uni, dense_bands, dense_true, split_label):
    """Print the uniform-truth score, split at each band's dense-truth depth.

    The dense truth only scores points brighter than the survey depth, so the
    headline RMSE is blind to a forecast that stops fading once an object drops
    out of reach. Scoring against the noise-free uniform grid over the same
    window shows that error. Each band's "dense depth" is the 99th percentile of
    the scored dense truth magnitudes; uniform points fainter than it fall where
    the dense metric has (almost) no points.

    Args:
        uni (dict): Flat uniform-truth arrays from ``_concat_uniform``.
        dense_bands, dense_true (np.ndarray): Band / truth of the dense scored
            points (sets the per-band depth).
        split_label (str): Split name for the printout.
    """
    resid = uni["pred_mag"] - uni["true_mag"]
    print(f"\n[{split_label}] UNIFORM-truth late-time error (same window and lead "
          f"rule; noise-free, no depth limit): {resid.shape[0]} points, "
          f"{np.unique(uni['stem']).shape[0]} objects")
    print(f"Overall uniform RMSE (mag): {np.sqrt(np.mean(resid**2)):.4f}  "
          f"MAE: {np.mean(np.abs(resid)):.4f}  bias: {np.mean(resid):+.4f}")

    def _stats(r):
        if r.shape[0] == 0:
            return f"{'-':>7}{'-':>8}"
        return f"{np.sqrt(np.mean(r**2)):>7.3f}{np.mean(r):>+8.3f}"

    print("Per band: all | within dense depth | beyond dense depth "
          "(RMSE, bias; + = forecast too faint)")
    print(f"  {'band':>5}{'depth':>7}{'n':>8}{'RMSE':>7}{'bias':>8}"
          f"{'n_in':>8}{'RMSE':>7}{'bias':>8}{'n_out':>8}{'RMSE':>7}{'bias':>8}")
    for b in range(N_BANDS):
        m = uni["band"] == b
        dm = dense_bands == b
        if not np.any(m) or not np.any(dm):
            continue
        depth = float(np.percentile(dense_true[dm], 99))
        beyond = m & (uni["true_mag"] > depth)
        within = m & ~beyond
        print(f"  {BAND_NAMES[b]:>5}{depth:>7.2f}{m.sum():>8d}{_stats(resid[m])}"
              f"{within.sum():>8d}{_stats(resid[within])}"
              f"{beyond.sum():>8d}{_stats(resid[beyond])}")


def write_uniform_csv(uni, csv_path):
    """Write the per-point uniform-truth score."""
    resid = uni["pred_mag"] - uni["true_mag"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "band", "phase_days", "lead_time_days", "pred_mag",
                    "pred_low", "pred_high", "true_mag", "residual_mag"])
        for i in range(resid.shape[0]):
            w.writerow([
                uni["stem"][i], BAND_NAMES[uni["band"][i]],
                f"{uni['phase'][i]:.4f}", f"{uni['lead_time'][i]:.4f}",
                f"{uni['pred_mag'][i]:.4f}", f"{uni['pred_low'][i]:.4f}",
                f"{uni['pred_high'][i]:.4f}", f"{uni['true_mag'][i]:.4f}",
                f"{resid[i]:.4f}",
            ])
    print(f"Wrote {csv_path}")


def fit_calibration_from_scored(scored, interval, out_path):
    """Fit per-band interval scales on scored points and write the JSON.

    Args:
        scored (list): Scored points from ``run_split`` on the CALIBRATION split.
        interval (float): Target central coverage.
        out_path (str): Calibration JSON to write.

    Returns:
        dict | None: The calibration in ``load_interval_calibration`` form, or
        None when the checkpoint has no quantile head (nothing to calibrate).
    """
    from calibrate_intervals import fit_scales, report

    if "pred_low" not in scored[0]:
        print("Checkpoint has no quantile head; skipping interval calibration.")
        return None
    d = {
        "band": np.array([BAND_NAMES[s["band"]] for s in scored], dtype=object),
        "med": np.array([s["pred_mag"] for s in scored]),
        "low": np.array([s["pred_low"] for s in scored]),
        "high": np.array([s["pred_high"] for s in scored]),
        "true": np.array([s["true_mag"] for s in scored]),
    }
    scales = fit_scales(d, interval)
    report(d, scales,
           "Calibration split (in-sample -- cov cal ~target by construction)")
    with open(out_path, "w") as fh:
        json.dump({"interval": interval, "bands": scales}, fh, indent=2)
    print(f"Wrote {out_path}")
    return load_interval_calibration(out_path)


def main():
    """Run the dense late-time evaluation.

    By default the whole post-training pass runs in one go on ONE checkpoint:
    pick it (best val loss unless --exact_epoch / --ckpt), eval the calibration
    (validation) split, fit per-band interval scales on it, then eval + plot the
    test split with the calibrated band.
    """
    args = get_args()
    tag = study_tag(args.study)
    if args.ckpt is None:
        run_dir = f"runs/study_{tag}"
        fixed_ckpt = os.path.join(
            run_dir, f"study{tag}_modelState_epoch{args.epoch:04d}.pth"
        )
        if args.exact_epoch:
            args.ckpt = fixed_ckpt
            print(f"Using exact epoch {args.epoch}: {args.ckpt}")
        else:
            best_epoch, best_loss, best_path = resolve_best_checkpoint(
                run_dir, tag, args.epoch
            )
            if best_path is not None:
                args.ckpt = best_path
                print(
                    f"Best val-loss checkpoint at or below epoch {args.epoch}: "
                    f"epoch {best_epoch} (median val loss {best_loss:.6f}) -> "
                    f"{args.ckpt}"
                )
            else:
                # No usable val records (e.g. records not co-located, or an old
                # run): fall back to the literal epoch so the eval still runs.
                args.ckpt = fixed_ckpt
                print(
                    f"No val records found under {run_dir}; falling back to the "
                    f"exact epoch {args.epoch}: {args.ckpt}"
                )
    if args.outdir is None:
        args.outdir = f"runs/study_{tag}/dense_latetime_eval_9band"
    os.makedirs(args.outdir, exist_ok=True)

    if not args.test_filelist:
        args.test_filelist = None
    if args.rollout and args.interval_calibration is not None:
        raise ValueError("--interval_calibration applies to DIRECT mode only.")
    # One-shot calibration is the default; a reused JSON, --no_calibration, the
    # rollout path (no quantile band), or an unrestricted test set turn it off.
    if (args.no_calibration or args.interval_calibration is not None
            or args.rollout or args.test_filelist is None
            or not args.calibrate_filelist):
        args.calibrate_filelist = None
    elif not os.path.exists(args.calibrate_filelist):
        raise FileNotFoundError(
            f"--calibrate_filelist {args.calibrate_filelist} not found (pass "
            "--no_calibration to skip calibration)."
        )
    print("Calibration split: "
          + (args.calibrate_filelist or "none (raw quantile band)"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        model,
        context_len,
        n_bands,
        context_window_days,
        max_context_len,
    ) = load_9band_model(args.ckpt, device, use_ema=getattr(args, "use_ema", False))

    if context_window_days is None:
        raise ValueError(
            "This eval requires a time-window checkpoint (context_window_days "
            "set); the loaded checkpoint is fixed-count."
        )

    # Load the TRAIN-ONLY stats the model was trained with (loaded if present;
    # no eval-set recomputation).
    means, stds = load_or_compute_band_normalization(
        stats_path=args.norm_stats_path,
        band_keys=BAND_KEYS,
        value_col=VALUE_COL,
        error_col=ERROR_COL,
        drop_upper_limits=DROP_UPPER_LIMITS,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)

    # Pair realistic and dense objects by stem; each split filters these.
    real_map = _stem_to_path(args.realistic_glob)
    dense_map = _stem_to_path(args.dense_glob)

    # Optional uniform-grid plotting companion (same stems). Absent files are
    # fine: the overlay just doesn't appear.
    uniform_map = _stem_to_path(args.uniform_glob) if args.uniform_glob else {}
    if uniform_map:
        print(f"Uniform-grid overlay files: {len(uniform_map)}")
    print(f"Realistic files: {len(real_map)}; dense files: {len(dense_map)}")
    if args.probe_dense_context:
        print(
            "PROBE_DENSE_CONTEXT: model CONTEXT drawn from the DENSE set "
            f"(truncated at {args.late_time_cutoff_days} d), scored against dense "
            "truth. Dense-context CEILING probe -- NOT the deployable path."
        )

    common = dict(
        args=args, model=model, device=device, means=means, stds=stds,
        context_window_days=context_window_days, max_context_len=max_context_len,
        real_map=real_map, dense_map=dense_map, uniform_map=uniform_map,
    )
    test_stems = (_read_stems(args.test_filelist)
                  if args.test_filelist is not None else None)

    interval_calibration = None
    if args.calibrate_filelist is not None:
        cal_stems = _read_stems(args.calibrate_filelist)
        overlap = cal_stems & test_stems
        if overlap:
            raise ValueError(
                f"{len(overlap)} objects are in both --calibrate_filelist and "
                "--test_filelist; the calibration split must be disjoint."
            )
        cal_scored = run_split(
            **common, split_stems=cal_stems,
            outdir=os.path.join(args.outdir, "calibration_split"),
            split_label="calibration", n_plots=0, score_uniform=False,
        )
        if cal_scored is None:
            raise ValueError("Calibration split scored no points.")
        interval_calibration = fit_calibration_from_scored(
            cal_scored, args.calibration_interval,
            os.path.join(args.outdir, "interval_calibration.json"),
        )
    elif args.interval_calibration is not None:
        interval_calibration = load_interval_calibration(args.interval_calibration)

    if interval_calibration is not None:
        print(
            f"\nInterval calibration ({interval_calibration['interval']:.0%}): "
            + ", ".join(
                f"{BAND_NAMES[b]} {interval_calibration['k_lo'][b]:.2f}/"
                f"{interval_calibration['k_hi'][b]:.2f}"
                for b in range(N_BANDS)
            )
        )

    run_split(
        **common, split_stems=test_stems, outdir=args.outdir, split_label="test",
        interval_calibration=interval_calibration, n_plots=args.n_plots,
    )
    print(f"\nCheckpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
