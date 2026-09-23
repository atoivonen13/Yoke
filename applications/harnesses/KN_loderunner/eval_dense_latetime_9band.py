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
    with torch.no_grad():
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
        pred_q = _batched_forward(
            model, x, lead_times, device, return_quantiles=True
        )  # [P, Q, N_BANDS]
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
            scored.append(
                {
                    "phase": float(d_t[idx]) - t0,
                    "lead_time": float(lead_times[j]),
                    "band": band,
                    "pred_mag": pred_mag,
                    "pred_low": pred_low,
                    "pred_high": pred_high,
                    "true_mag": true_mag,
                    "residual_mag": pred_mag - true_mag,
                }
            )

    # Smooth forecast curve for plotting: sweep lead time from 0 to the farthest
    # scored late-time point, predicting all bands at each lead time -- also a
    # single batched forward pass.
    max_dt = float(lead_times.max())
    lead_grid = np.linspace(0.0, max_dt, 60).astype(np.float32)
    curve_q = _batched_forward(
        model, x, lead_grid, device, return_quantiles=True
    )  # [60, Q, N_BANDS]
    median_idx = getattr(model, "median_idx", 0)
    n_q = curve_q.shape[1]
    curve = curve_q[:, median_idx, :]  # [60, N_BANDS] median point forecast
    curve_mag = curve * (stds[None, :] + EPS) + means[None, :]
    # Outer-quantile curves for the shaded uncertainty band (== median when Q==1,
    # so the band has zero width for a point head and nothing is drawn).
    curve_low_mag = curve_q[:, 0, :] * (stds[None, :] + EPS) + means[None, :]
    curve_high_mag = curve_q[:, n_q - 1, :] * (stds[None, :] + EPS) + means[None, :]

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
        "t0": t0,
        "last_real_t": last_real_t,
        "curve_phase": (last_real_t - t0) + lead_grid,
        "curve_mag": curve_mag.astype(np.float32),
        "curve_low_mag": curve_low_mag.astype(np.float32),
        "curve_high_mag": curve_high_mag.astype(np.float32),
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
        # Shaded quantile band (0.1-0.9) when the model has a quantile head. For a
        # point head low == high == median, so skip drawing a zero-width band.
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
                label="forecast 0.1-0.9",
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
        "late-time u/g). Plotting only -- never scored, never fed to the model. "
        "Silently skipped if no files match, so it auto-activates once the set "
        "is generated.",
    )
    p.add_argument(
        "--test_filelist",
        type=str,
        default=None,
        help="Path to the test-split stem list (one object stem per line). If "
        "omitted, all objects present in BOTH globs are evaluated.",
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


def main():
    """Run the dense late-time evaluation."""
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

    # Pair realistic and dense objects by stem, restricted to the test split.
    real_map = _stem_to_path(args.realistic_glob)
    dense_map = _stem_to_path(args.dense_glob)
    stems = sorted(set(real_map) & set(dense_map))

    # Optional uniform-grid plotting companion (same stems). Absent files are
    # fine: the overlay just doesn't appear.
    uniform_map = _stem_to_path(args.uniform_glob) if args.uniform_glob else {}
    if uniform_map:
        print(f"Uniform-grid overlay files: {len(uniform_map)}")

    if args.test_filelist is not None:
        with open(args.test_filelist) as fh:
            test_stems = {line.strip() for line in fh if line.strip()}
        stems = [s for s in stems if s in test_stems]
        print(f"Restricted to {len(stems)} test-split objects.")

    print(
        f"Realistic files: {len(real_map)}; dense files: {len(dense_map)}; "
        f"paired & in-split: {len(stems)}"
    )
    if args.probe_dense_context:
        print(
            "PROBE_DENSE_CONTEXT: model CONTEXT drawn from the DENSE set "
            f"(truncated at {args.late_time_cutoff_days} d), scored against dense "
            "truth. Dense-context CEILING probe -- NOT the deployable path."
        )
    if args.max_objects > 0:
        stems = stems[: args.max_objects]

    all_scored = []
    plotted = 0
    n_eval = 0
    # When the model was trained with the flagged-UL context channel, the
    # realistic (context) stream must KEEP upper limits so they reach the
    # flagged-context path; otherwise drop them as before. Auto-detected from
    # the checkpoint metadata so eval matches training.
    ul_channel = getattr(model, "upper_limit_channel", False)
    real_drop_ul = DROP_UPPER_LIMITS and not ul_channel
    for stem in stems:
        real_stream = read_merged_stream(real_map[stem], real_drop_ul)
        dense_stream = read_merged_stream(dense_map[stem], drop_upper_limits=False)
        uniform_stream = None
        if stem in uniform_map:
            uniform_stream = read_merged_stream(
                uniform_map[stem], drop_upper_limits=False
            )
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
        )
        if result is None:
            continue
        n_eval += 1
        for s in result["scored"]:
            s["stem"] = stem
            all_scored.append(s)
        if plotted < args.n_plots:
            plot_object(
                result, stem,
                os.path.join(args.outdir, f"latetime_{stem}.png"),
            )
            plotted += 1

    if not all_scored:
        print("No late-time points scored (check the cutoff and globs).")
        return

    # Per-band late-time error summary.
    resid = np.asarray([s["residual_mag"] for s in all_scored])
    bands = np.asarray([s["band"] for s in all_scored])
    mode = "AUTOREGRESSIVE rollout" if args.rollout else "DIRECT single-pass"
    print(f"\nForecast mode: {mode}")
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
    ax.plot(centers, rmse_bin, "o-")
    ax.set_xlabel("Lead time from last realistic detection [days]")
    ax.set_ylabel("Late-time forecast RMSE [mag]")
    ax.set_title("Dense late-time forecast error vs lead time")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "latetime_rmse_vs_lead.png"), dpi=130)
    plt.close(fig)

    # Full per-point CSV.
    csv_path = os.path.join(args.outdir, "latetime_scored_points.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["stem", "band", "phase_days", "lead_time_days",
             "pred_mag", "pred_low", "pred_high", "true_mag", "residual_mag"]
        )
        for s in all_scored:
            # pred_low/high present only on the DIRECT quantile path; fall back to
            # the point forecast (rollout path, or a point head) so the columns are
            # always populated.
            w.writerow([
                s["stem"], BAND_NAMES[s["band"]], f"{s['phase']:.4f}",
                f"{s['lead_time']:.4f}", f"{s['pred_mag']:.4f}",
                f"{s.get('pred_low', s['pred_mag']):.4f}",
                f"{s.get('pred_high', s['pred_mag']):.4f}",
                f"{s['true_mag']:.4f}", f"{s['residual_mag']:.4f}",
            ])

    print(f"\nWrote {plotted} per-object plots, the RMSE-vs-lead plot, and "
          f"{csv_path} in {args.outdir}")


if __name__ == "__main__":
    main()
