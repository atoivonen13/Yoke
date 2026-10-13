r"""Show what the ViT sees on the event-token path (Study 203+).

On this path the transformer blocks run on a small token SET per sample:

  * one token per valid context event, embedded from
    ``[value, tau, one_hot_band]``, and
  * one query token per band, a learned band vector plus the Fourier
    Dt / phase / redshift features.

Each token sits at a 2-D RoPE position (band column, anchor-relative time row).
The 9 query outputs are read out and appended to the bypass head input. This
script runs a checkpoint's real forward on eval objects (the same realistic
context, cutoff and window as eval_dense_latetime_9band.py) and records every
block's attention weights with hooks. They are recomputed from the block's own
q/k projections and checked against its output once per run. It writes:

  tokens_<stem>.png     The token set at its RoPE positions; query-to-token
                        attention per block at ``--ref_dt``; and how the focus
                        band's query attention shifts with Dt (last block).
  forecast_<stem>.png   Per-band forecast with the token path, and with its
                        readout zeroed (the bypass part of the same weights),
                        against the uniform truth.
  attention_summary.png Over ``--n_summary`` objects: attention to an event vs
                        its age, same band vs other band, per block, as a ratio
                        to uniform attention (1 = no preference).

No training. Needs the checkpoint and a GPU or CPU. Run on the cluster, e.g.:

    python visualize_event_tokens.py --study 207 --epoch 150
    python visualize_event_tokens.py --study 203 --epoch 100 --n_objects 6
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from yoke.datasets.kilonova_dataset import (
    EPS,
    default_kn_glob,
    load_or_compute_band_normalization,
)
from yoke.models.vit.rope import apply_rotary_emb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dense_latetime_9band import (  # noqa: E402
    BAND_COLORS,
    BAND_NAMES,
    DROP_UPPER_LIMITS,
    ERROR_COL,
    FILELIST_DIR,
    N_BANDS,
    VALUE_COL,
    _read_stems,
    _stem_to_path,
    read_merged_stream,
    resolve_best_checkpoint,
    study_tag,
)
from plot_pred_diagnostics_9band import (  # noqa: E402
    _select_window,
    build_context_input,
    load_9band_model,
)

# Lead times the summary averages over (days).
SUMMARY_DTS = (0.5, 2.0, 4.0, 8.0)


class AttentionRecorder:
    """Record every block's attention weights during a forward pass.

    A pre-hook on each block's ``RopeAttention`` recomputes the softmax weights
    from the module's own projections, QK RMSNorm and RoPE, exactly as
    ``RopeAttention.forward`` feeds them to SDPA. On the first call a forward
    hook rebuilds the attention output from those weights and compares it with
    the module's real output, so a drift between this script and the model
    code fails loudly instead of plotting wrong weights.
    """

    def __init__(self, blocks):
        """Hook ``blocks`` (the backbone's transformer blocks)."""
        self.weights = []
        self.max_check_err = None
        self._pending = None
        self.handles = []
        for block in blocks:
            self.handles.append(
                block.attn.register_forward_pre_hook(self._pre, with_kwargs=True)
            )
            self.handles.append(block.attn.register_forward_hook(self._post))

    def _pre(self, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        rot = kwargs.get("image_rotary_emb")
        mask = kwargs.get("attention_mask")
        q = module.norm_q(module.to_q(h).unflatten(-1, (module.heads, module.dim_head)))
        k = module.norm_k(module.to_k(h).unflatten(-1, (module.heads, module.dim_head)))
        if rot is not None:
            q = apply_rotary_emb(q, rot, sequence_dim=1)
            k = apply_rotary_emb(k, rot, sequence_dim=1)
        q = q.transpose(1, 2).float()  # [B, H, N, Dh]
        k = k.transpose(1, 2).float()
        scores = q @ k.transpose(-1, -2) / math.sqrt(module.dim_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = scores.softmax(dim=-1)  # [B, H, N, N]
        self.weights.append(w.cpu())
        if self.max_check_err is None:
            v = module.to_v(h).unflatten(-1, (module.heads, module.dim_head))
            self._pending = (w, v.transpose(1, 2).float())

    def _post(self, module, args, output):
        if self._pending is None:
            return
        w, v = self._pending
        self._pending = None
        merged = (w @ v).transpose(1, 2).flatten(2).to(output.dtype)
        expected = module.to_out[0](merged)
        err = (expected - output).abs().max().item()
        scale = output.abs().max().item() + 1e-12
        self.max_check_err = err / scale
        if self.max_check_err > 1e-3:
            raise RuntimeError(
                "Recomputed attention does not reproduce RopeAttention's output "
                f"(max rel. err {self.max_check_err:.2e}); the script is out of "
                "sync with plain_vit.py."
            )

    def pop(self) -> np.ndarray:
        """Weights of the last forward, [B, n_blocks, heads, N, N]."""
        out = torch.stack(self.weights, dim=1).numpy()
        self.weights = []
        return out

    def remove(self) -> None:
        """Detach the hooks."""
        for h in self.handles:
            h.remove()


def get_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=int, required=True)
    p.add_argument("--epoch", type=int, default=500,
                   help="Epoch budget: loads the best-val checkpoint at or below "
                        "it (as the eval does), unless --exact_epoch.")
    p.add_argument("--exact_epoch", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--test_filelist",
                   default=os.path.join(FILELIST_DIR, "kn_rubin_ztf_test.txt"))
    p.add_argument("--stems", nargs="+", default=None,
                   help="Objects to plot (default: --n_objects spread by "
                        "context count).")
    p.add_argument("--n_objects", type=int, default=4)
    p.add_argument("--n_summary", type=int, default=200,
                   help="Objects averaged in attention_summary.png.")
    p.add_argument("--ref_dt", type=float, default=3.0,
                   help="Lead time (days) of the per-block attention heatmaps.")
    p.add_argument("--horizon_days", type=float, default=10.0)
    p.add_argument("--cutoff_days", type=float, default=2.0)
    p.add_argument("--focus_band", type=str, default=None,
                   help="Query band for the attention lines and attention-vs-Dt "
                        "panel (default: the band with most context events).")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def resolve_ckpt(args: argparse.Namespace) -> str:
    """Checkpoint path from --ckpt, or the eval's best-val / exact-epoch rule."""
    if args.ckpt is not None:
        return args.ckpt
    tag = study_tag(args.study)
    run_dir = f"runs/study_{tag}"
    fixed = os.path.join(run_dir, f"study{tag}_modelState_epoch{args.epoch:04d}.pth")
    if args.exact_epoch:
        return fixed
    best_epoch, _, best_path = resolve_best_checkpoint(run_dir, tag, args.epoch)
    if best_path is None:
        return fixed
    print(f"Best val-loss checkpoint at or below epoch {args.epoch}: {best_epoch}")
    return best_path


def build_object(real_stream, model, means, stds, cw, max_len, cutoff, device):
    """Eval-identical context input for one object, plus its plotting context.

    Returns None when the object has no pre-cutoff detection. Otherwise a dict
    with ``x`` [1, D], ``t_anchor`` (absolute time of the last context
    detection), ``n_window`` (in-window detections before the cap) and the raw
    context (``ctx_tau`` anchor-relative days, ``ctx_mag``, ``ctx_band``).
    """
    r_t, r_v, r_b, _r_ul, r_z = real_stream
    if r_t.shape[0] == 0:
        return None
    t0 = float(r_t[0])
    m = (r_t - t0) <= cutoff
    c_t, c_v, c_b = r_t[m], r_v[m], r_b[m]
    t_anchor = float(c_t[-1])
    v_norm = (c_v - means[c_b]) / (stds[c_b] + EPS)
    win_v, win_t, win_b = _select_window(
        ctx_t=list(c_t.astype(np.float32)),
        ctx_v=list(v_norm.astype(np.float32)),
        ctx_b=list(c_b),
        context_window_days=cw,
        max_context_len=max_len,
    )
    x = build_context_input(
        win_v=win_v, win_t=win_t, win_b=win_b, context_len=max_len,
        n_bands=N_BANDS, device=device, window_mode=True,
        phase_fourier_bands=getattr(model, "phase_fourier_bands", 0),
        phase0=t0,
        redshift_fourier_bands=getattr(model, "redshift_fourier_bands", 0),
        redshift=r_z,
        redshift_pivot_direct=getattr(model, "redshift_pivot_direct", False),
    )
    in_win = c_t >= t_anchor - cw
    return dict(
        x=x, t_anchor=t_anchor, n_window=int(in_win.sum()),
        ctx_tau=(c_t[in_win] - t_anchor), ctx_mag=c_v[in_win], ctx_band=c_b[in_win],
    )


def token_meta(model, x):
    """Per-slot event metadata as the model sees it (``_split_events``)."""
    with torch.inference_mode():
        value, tau, valid, band = model._split_events(x[:, : model.input_dim])
    return dict(value=value[0].cpu().numpy(), tau=tau[0].cpu().numpy(),
                valid=valid[0].cpu().numpy(), band=band[0].cpu().numpy())


def forward(model, x, dts, device, ablate=False):
    """Median prediction (normalized) for each lead time, [len(dts), N_BANDS].

    ``ablate`` zeroes the token readout, leaving the bypass part of the same
    weights (not a separately trained bypass model).
    """
    dt = torch.tensor(np.asarray(dts, dtype=np.float32), device=device)
    if ablate:
        width = model.n_bands * model.event_token_readout
        model._event_token_readout = (
            lambda x_ev, _dt, _g: x_ev.new_zeros(x_ev.shape[0], width)
        )
    try:
        with torch.inference_mode():
            pred = model(x.expand(dt.shape[0], -1), in_vars=None, out_vars=None, Dt=dt)
    finally:
        if ablate:
            del model._event_token_readout
    if pred.dim() == 3:
        pred = pred[:, getattr(model, "median_idx", 0), :]
    return pred.cpu().numpy()


def token_labels(meta, valid_idx):
    """Short labels for the shown keys: valid events, then the band queries."""
    ev = [f"{BAND_NAMES[meta['band'][i]]} {meta['tau'][i]:+.2f}" for i in valid_idx]
    return ev + [f"Q {b}" for b in BAND_NAMES]


def plot_tokens(obj, meta, w_h, dts, ref_i, focus, ref_dt, path, title):
    """tokens_<stem>.png: token layout, per-block attention, attention vs Dt.

    ``w_h`` is the head-mean attention, [n_dt, n_blocks, N, N].
    """
    L = meta["tau"].shape[0]
    valid_idx = np.nonzero(meta["valid"])[0]
    keys = np.concatenate([valid_idx, np.arange(L, L + N_BANDS)])
    labels = token_labels(meta, valid_idx)
    n_blocks = w_h.shape[1]
    q_rows = np.arange(L, L + N_BANDS)

    fig = plt.figure(figsize=(4 * n_blocks, 10))
    gs = GridSpec(2, n_blocks, figure=fig, height_ratios=[1.15, 1])

    # (a) Token set at its RoPE positions (band column, time row).
    ax = fig.add_subplot(gs[0, : n_blocks // 2])
    vals = meta["value"][valid_idx]
    vmax = max(1e-6, float(np.abs(vals).max()))
    sc = ax.scatter(meta["band"][valid_idx], meta["tau"][valid_idx], c=vals,
                    cmap="coolwarm_r", vmin=-vmax, vmax=vmax, s=140,
                    edgecolors="k", zorder=3)
    for j, i in enumerate(valid_idx):
        ax.annotate(str(j), (meta["band"][i], meta["tau"][i]), xytext=(7, 4),
                    textcoords="offset points", fontsize=8)
    ax.scatter(np.arange(N_BANDS), np.full(N_BANDS, ref_dt), marker="*", s=260,
               c=BAND_COLORS, edgecolors="k", zorder=3,
               label=f"band queries at Dt = {ref_dt:g} d")
    # Last-block attention from the focus query, line width ~ weight.
    last = w_h[ref_i, -1, L + focus]
    for i in valid_idx:
        a = last[i]
        if a > 0.02:
            ax.plot([focus, meta["band"][i]], [ref_dt, meta["tau"][i]],
                    color=BAND_COLORS[focus], lw=12 * a, alpha=0.6, zorder=2)
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8)
    ax.text(N_BANDS - 0.6, 0.05, "anchor (last detection)", fontsize=8,
            ha="right", va="bottom", color="0.3")
    ax.set_xticks(range(N_BANDS), BAND_NAMES)
    ax.set_xlim(-0.6, N_BANDS - 0.4)
    ax.set_ylabel("anchor-relative time tau (d) = RoPE row")
    ax.set_xlabel("band = RoPE column")
    ax.set_title(f"Token set: {len(valid_idx)} events + {N_BANDS} queries; lines = "
                 f"last-block attention from Q {BAND_NAMES[focus]}")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, label="event value (z-scored mag; + = fainter)",
                 pad=0.01)

    # (b) Focus query attention vs Dt (last block).
    ax = fig.add_subplot(gs[0, n_blocks // 2 :])
    a_dt = w_h[:, -1, L + focus][:, keys]  # [n_dt, n_keys]
    im = ax.pcolormesh(np.arange(len(keys)), dts, a_dt, cmap="magma",
                       shading="nearest")
    ax.set_xticks(range(len(keys)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("lead time Dt (d)")
    ax.set_title(f"Q {BAND_NAMES[focus]} attention vs Dt (last block, head mean)")
    fig.colorbar(im, ax=ax, pad=0.01)

    # (c) Query -> key attention per block at ref_dt.
    vmax = float(w_h[ref_i][:, q_rows][:, :, keys].max())
    for b in range(n_blocks):
        ax = fig.add_subplot(gs[1, b])
        im = ax.imshow(w_h[ref_i, b][q_rows][:, keys], cmap="magma", vmin=0,
                       vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(keys)), [str(j) for j in range(len(valid_idx))]
                      + [f"Q{n}" for n in BAND_NAMES], rotation=90, fontsize=7)
        ax.set_yticks(range(N_BANDS), [f"Q {n}" for n in BAND_NAMES], fontsize=7)
        ax.axvline(len(valid_idx) - 0.5, color="w", lw=1)
        ev_share = w_h[ref_i, b][q_rows][:, valid_idx].sum(axis=1).mean()
        ax.set_title(f"block {b + 1}: {100 * ev_share:.0f}% on events", fontsize=9)
    fig.colorbar(im, ax=fig.axes[-n_blocks:], pad=0.01, label="attention")
    fig.suptitle(f"{title}  (event numbers match panel a; Dt = {ref_dt:g} d)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_forecast(obj, dts, mag_tok, mag_abl, truth, path, title):
    """forecast_<stem>.png: per-band forecast with / without the token path."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for b, ax in enumerate(axes.flat):
        c = BAND_COLORS[b]
        if truth is not None:
            m = truth[2] == b
            ax.plot(truth[0][m], truth[1][m], color="0.55", lw=1.0,
                    label="uniform truth")
        m = obj["ctx_band"] == b
        ax.scatter(obj["ctx_tau"][m], obj["ctx_mag"][m], color=c, edgecolors="k",
                   zorder=3, label="context")
        ax.plot(dts, mag_tok[:, b], color=c, lw=2, label="with token path")
        ax.plot(dts, mag_abl[:, b], color=c, lw=1.3, ls="--",
                label="token readout zeroed")
        ax.axvline(0.0, color="0.4", ls=":", lw=0.8)
        ax.invert_yaxis()
        ax.set_title(BAND_NAMES[b])
        if b == 0:
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time from last context detection (d)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mag")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize(records, n_blocks, cw, path, title):
    """attention_summary.png: event attention vs age, same / other band."""
    edges = np.linspace(0.0, cw, 9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    s = np.zeros((n_blocks, 2, len(centers)))
    n = np.zeros_like(s)
    ev_share = np.zeros(n_blocks)
    n_q = 0
    for meta, w in records:  # w: [n_dt, n_blocks, N, N], head mean
        L = meta["tau"].shape[0]
        valid_idx = np.nonzero(meta["valid"])[0]
        n_keys = len(valid_idx) + N_BANDS
        age = -meta["tau"][valid_idx]
        a_bin = np.clip(np.digitize(age, edges) - 1, 0, len(centers) - 1)
        for q in range(N_BANDS):
            same = (meta["band"][valid_idx] == q).astype(int)
            a = w[:, :, L + q][:, :, valid_idx] * n_keys  # [n_dt, n_blocks, n_ev]
            for d in range(a.shape[0]):
                for b in range(n_blocks):
                    np.add.at(s[b], (1 - same, a_bin), a[d, b])
                    np.add.at(n[b], (1 - same, a_bin), 1)
            ev_share += w[:, :, L + q][:, :, valid_idx].sum(axis=2).sum(axis=0)
            n_q += w.shape[0]
    ratio = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ev_share /= max(n_q, 1)

    fig, axes = plt.subplots(1, n_blocks, figsize=(3.4 * n_blocks, 3.6), sharey=True)
    print("\nMean query attention per event, as a ratio to uniform (1 = none):")
    print(f"  {'block':>5} {'events %':>9}  same-band by age bin | other-band")
    for b, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(centers, ratio[b, 0], "o-", label="event in query's band")
        ax.plot(centers, ratio[b, 1], "s--", label="other band")
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("event age before anchor (d)")
        ax.set_title(f"block {b + 1}: {100 * ev_share[b]:.0f}% on events", fontsize=9)
        print(f"  {b + 1:>5} {100 * ev_share[b]:>8.1f}%  "
              + " ".join(f"{v:4.2f}" for v in ratio[b, 0]) + " | "
              + " ".join(f"{v:4.2f}" for v in ratio[b, 1]))
    np.atleast_1d(axes)[0].set_ylabel("attention x n_keys (1 = uniform)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    """Load the checkpoint, hook the blocks, and write the figures."""
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_ckpt(args)
    model, _cl, _nb, cw, max_len = load_9band_model(ckpt, device, use_ema=args.use_ema)
    if not getattr(model, "event_tokens", False):
        raise ValueError(f"{ckpt} is not an event-token checkpoint.")
    if getattr(model, "upper_limit_channel", False):
        raise NotImplementedError("Upper-limit-channel models are not supported.")

    tag = study_tag(args.study)
    outdir = args.outdir or f"runs/study_{tag}/event_token_viz"
    os.makedirs(outdir, exist_ok=True)
    label = f"Study {tag} ({os.path.basename(ckpt)})"

    means, stds = load_or_compute_band_normalization(
        stats_path=model.norm_stats_path, band_keys=model.band_keys,
        value_col=VALUE_COL, error_col=ERROR_COL, drop_upper_limits=DROP_UPPER_LIMITS,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    real_map = _stem_to_path(default_kn_glob(model.band_keys, "realistic"))
    uni_map = _stem_to_path(default_kn_glob(model.band_keys, "uniform"))

    stems = sorted(set(real_map) & _read_stems(args.test_filelist))
    if args.stems:
        stems = [s for s in args.stems if s in real_map] + [
            s for s in stems if s not in args.stems
        ]
    stems = stems[: max(args.n_summary, len(args.stems or []))]

    dts = np.unique(np.concatenate([
        np.linspace(0.05, args.horizon_days, 80), [args.ref_dt], SUMMARY_DTS,
    ])).astype(np.float32)
    ref_i = int(np.argmin(np.abs(dts - args.ref_dt)))
    sum_i = [int(np.argmin(np.abs(dts - d))) for d in SUMMARY_DTS]

    blocks = model.backbone.backbone.transformer_blocks
    rec = AttentionRecorder(blocks)
    objs, records = {}, []
    try:
        for stem in stems:
            obj = build_object(read_merged_stream(real_map[stem], DROP_UPPER_LIMITS),
                               model, means, stds, cw, max_len, args.cutoff_days,
                               device)
            if obj is None:
                continue
            meta = token_meta(model, obj["x"])
            pred = forward(model, obj["x"], dts, device)
            w_h = rec.pop().mean(axis=2)  # head mean: [n_dt, n_blocks, N, N]
            records.append((meta, w_h[sum_i]))
            objs[stem] = (obj, meta, pred, w_h)
    finally:
        rec.remove()
    if not objs:
        raise RuntimeError("No test object had a pre-cutoff detection.")
    print(f"Attention recompute check: max rel. err {rec.max_check_err:.2e}")
    print(f"Objects: {len(objs)} (summary over {len(records)})")

    summarize(records, len(blocks), cw, os.path.join(outdir, "attention_summary.png"),
              f"{label}: query attention over {len(records)} test objects, "
              f"Dt in {SUMMARY_DTS}")

    if args.stems:
        plot_stems = [s for s in args.stems if s in objs]
    else:
        by_count = sorted(objs, key=lambda s: objs[s][0]["n_window"])
        pick = np.linspace(0, len(by_count) - 1, args.n_objects).round().astype(int)
        plot_stems = [by_count[i] for i in np.unique(pick)]

    for stem in plot_stems:
        obj, meta, pred, w_h = objs[stem]
        if args.focus_band is not None:
            focus = BAND_NAMES.index(args.focus_band)
        else:
            focus = int(np.bincount(obj["ctx_band"], minlength=N_BANDS).argmax())
        title = f"{label}: {stem}, {obj['n_window']} in-window detections"
        plot_tokens(obj, meta, w_h, dts, ref_i, focus, args.ref_dt,
                    os.path.join(outdir, f"tokens_{stem}.png"), title)

        abl = forward(model, obj["x"], dts, device, ablate=True)
        mag_tok = pred * (stds + EPS) + means
        mag_abl = abl * (stds + EPS) + means
        truth = None
        if stem in uni_map:
            u_t, u_v, u_b, _ul, _z = read_merged_stream(uni_map[stem], False)
            lead = u_t - obj["t_anchor"]
            m = (lead > 0) & (lead <= args.horizon_days) & np.isfinite(u_v)
            truth = (lead[m], u_v[m], u_b[m])
        plot_forecast(obj, dts, mag_tok, mag_abl, truth,
                      os.path.join(outdir, f"forecast_{stem}.png"), title)
        print(f"  wrote tokens_{stem}.png, forecast_{stem}.png "
              f"(n_window {obj['n_window']}, focus Q {BAND_NAMES[focus]})")
    print(f"Figures in {outdir}")


if __name__ == "__main__":
    main()
