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
  tokens_paper_<stem>   Paper version of the first two panels (pdf + png).
  lightcurve_image_<stem>
                        Paper figure: the context light curve, and the same
                        detections as cells of a time x band image (which the
                        code never builds; the tokens carry only its
                        positions), with the band queries at ``--ref_dt``.
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
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from yoke.datasets.kilonova_dataset import (
    EPS,
    default_kn_glob,
    load_or_compute_band_normalization,
)
from yoke.models.vit.rope import apply_rotary_emb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dense_latetime_9band import (  # noqa: E402
    BAND_COLORS as HOUSE_BAND_COLORS,
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
from pub_style import (  # noqa: E402
    add_style_args,
    pub_band_colors,
    pub_dpi,
    pub_fs,
    pub_layout,
    pub_size,
    pub_text,
    pub_title,
    rc_lw,
    rc_ms,
    styled,
    use_pub_style,
)

# Lead times the summary averages over (days).
SUMMARY_DTS = (0.5, 2.0, 4.0, 8.0)
# Restyled in place by ``main`` (one style hue per filter).
BAND_COLORS = list(HOUSE_BAND_COLORS)


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
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="Formats of the paper figures tokens_paper_<stem> and "
                        "lightcurve_image_<stem>.")
    p.add_argument("--cell_days", type=float, default=0.25,
                   help="Time cell (days) of the conceptual image in "
                        "lightcurve_image_<stem>.")
    add_style_args(p)
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

    # Styled: two columns wide, taller than the house aspect (two panel rows of
    # labeled heatmaps).
    fig = plt.figure(figsize=pub_size((16, 13)) if styled() else (4 * n_blocks, 10))
    gs = GridSpec(2, n_blocks, figure=fig, height_ratios=[1.15, 1])
    # Per-token tick labels: smaller than the style's ticks so they fit.
    tok_fs = 0.7 * pub_fs(10, "xtick.labelsize")

    # (a) Token set at its RoPE positions (band column, time row).
    ax = fig.add_subplot(gs[0, : n_blocks // 2])
    vals = meta["value"][valid_idx]
    vmax = max(1e-6, float(np.abs(vals).max()))
    # s is an area: scale with the marker size squared.
    sc = ax.scatter(meta["band"][valid_idx], meta["tau"][valid_idx], c=vals,
                    cmap="coolwarm_r", vmin=-vmax, vmax=vmax, s=140 * rc_ms(1) ** 2,
                    edgecolors="k", linewidths=rc_lw(1), zorder=3)
    for j, i in enumerate(valid_idx):
        ax.annotate(str(j), (meta["band"][i], meta["tau"][i]), xytext=(7, 4),
                    textcoords="offset points", fontsize=pub_fs(8, "xtick.labelsize"))
    ax.scatter(np.arange(N_BANDS), np.full(N_BANDS, ref_dt), marker="*",
               s=260 * rc_ms(1) ** 2, c=BAND_COLORS, edgecolors="k",
               linewidths=rc_lw(1), zorder=3, label=f"band queries at Dt = {ref_dt:g} d")
    # Last-block attention from the focus query, line width ~ weight.
    last = w_h[ref_i, -1, L + focus]
    for i in valid_idx:
        a = last[i]
        if a > 0.02:
            ax.plot([focus, meta["band"][i]], [ref_dt, meta["tau"][i]],
                    color=BAND_COLORS[focus], lw=rc_lw(12 * a), alpha=0.6, zorder=2)
    ax.axhline(0.0, color="0.4", ls="--", lw=rc_lw(0.8))
    ax.text(N_BANDS - 0.6, 0.05, "anchor (last detection)", fontsize=pub_fs(8),
            ha="right", va="bottom", color="0.3")
    ax.set_xticks(range(N_BANDS), BAND_NAMES, rotation=90 if styled() else 0)
    if styled():
        # Categorical axis: no minor ticks between the bands.
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlim(-0.6, N_BANDS - 0.4)
    ax.set_ylabel(pub_text("anchor-relative time tau (d) = RoPE row",
                           r"$\tau$ (d) = RoPE row"))
    ax.set_xlabel("band = RoPE column")
    ax.set_title(pub_text(f"Token set: {len(valid_idx)} events + {N_BANDS} queries; lines = "
                          f"last-block attention from Q {BAND_NAMES[focus]}",
                          f"(a) {len(valid_idx)} events + {N_BANDS} queries"))
    ax.legend(loc="upper left", fontsize=pub_fs(8))
    fig.colorbar(sc, ax=ax, label=pub_text("event value (z-scored mag; + = fainter)",
                                           "value (z-scored mag)"), pad=0.01)

    # (b) Focus query attention vs Dt (last block).
    ax = fig.add_subplot(gs[0, n_blocks // 2 :])
    a_dt = w_h[:, -1, L + focus][:, keys]  # [n_dt, n_keys]
    im = ax.pcolormesh(np.arange(len(keys)), dts, a_dt, cmap="magma",
                       shading="nearest")
    ax.set_xticks(range(len(keys)), labels, rotation=90, fontsize=tok_fs)
    if styled():
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel(pub_text("lead time Dt (d)", r"lead time $\Delta t$ (d)"))
    ax.set_title(pub_text(f"Q {BAND_NAMES[focus]} attention vs Dt (last block, head mean)",
                          f"(b) Q {BAND_NAMES[focus]} attention, last block"))
    fig.colorbar(im, ax=ax, pad=0.01)

    # (c) Query -> key attention per block at ref_dt.
    vmax = float(w_h[ref_i][:, q_rows][:, :, keys].max())
    for b in range(n_blocks):
        ax = fig.add_subplot(gs[1, b])
        im = ax.imshow(w_h[ref_i, b][q_rows][:, keys], cmap="magma", vmin=0,
                       vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(keys)), [str(j) for j in range(len(valid_idx))]
                      + [f"Q{n}" for n in BAND_NAMES], rotation=90, fontsize=tok_fs)
        ax.set_yticks(range(N_BANDS), [f"Q {n}" for n in BAND_NAMES], fontsize=tok_fs)
        if styled():
            # Heatmap cells: no minor ticks.
            ax.minorticks_off()
            if b:
                ax.tick_params(axis="y", labelleft=False)
        ax.axvline(len(valid_idx) - 0.5, color="w", lw=rc_lw(1))
        ev_share = w_h[ref_i, b][q_rows][:, valid_idx].sum(axis=1).mean()
        ax.set_title(pub_text(f"block {b + 1}: {100 * ev_share:.0f}% on events",
                              f"block {b + 1}: {100 * ev_share:.0f}%"), fontsize=pub_fs(9))
    fig.colorbar(im, ax=fig.axes[-n_blocks:], pad=0.01, label="attention")
    fig.suptitle(pub_title(f"{title}  (event numbers match panel a; Dt = {ref_dt:g} d)"),
                 fontsize=pub_fs(None, "axes.titlesize"))
    fig.savefig(path, dpi=pub_dpi(110), bbox_inches="tight")
    plt.close(fig)


def plot_tokens_paper(meta, w_h, dts, ref_i, focus, ref_dt, canvas, path_stem, formats):
    """tokens_paper_<stem>.<fmt>: panels (a) and (b) of tokens_<stem>.png, for print.

    (a) The light curve mapped to tokens: each event at (band, tau), the band
    queries at (band, Dt = ref_dt), and the last-block attention from the focus
    query. The right axis gives tau as a row of the RoPE canvas.
    (b) The focus query's last-block attention to every key vs Dt.

    ``canvas`` is ``(H, render_context_days, render_t_span)`` of the model, as
    in ``_row_of_tau``. Events are numbered 1..n in key order in both panels.
    """
    H, c0, span = canvas
    L = meta["tau"].shape[0]
    valid_idx = np.nonzero(meta["valid"])[0]
    n_ev = len(valid_idx)
    keys = np.concatenate([valid_idx, np.arange(L, L + N_BANDS)])
    ev_band = meta["band"][valid_idx]
    ev_tau = meta["tau"][valid_idx]
    last = w_h[ref_i, -1, L + focus]
    q_name = BAND_NAMES[focus]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=pub_size((12, 5.2)) if styled()
                                  else (12, 5.2), width_ratios=[1, 1.15])

    # (a) Token positions.
    for i in valid_idx:
        if last[i] > 0.02:
            ax.plot([focus, meta["band"][i]], [ref_dt, meta["tau"][i]],
                    color=BAND_COLORS[focus], lw=rc_lw(12 * last[i]), alpha=0.6,
                    zorder=2, solid_capstyle="round")
    ax.scatter(ev_band, ev_tau, c=[BAND_COLORS[b] for b in ev_band],
               s=90 * rc_ms(1) ** 2, edgecolors="k", linewidths=rc_lw(1), zorder=3)
    for j in range(n_ev):
        ax.annotate(str(j + 1), (ev_band[j], ev_tau[j]), xytext=(6, 0), va="center",
                    textcoords="offset points", fontsize=pub_fs(8, "xtick.labelsize"))
    sizes = np.where(np.arange(N_BANDS) == focus, 300, 170) * rc_ms(1) ** 2
    ax.scatter(np.arange(N_BANDS), np.full(N_BANDS, ref_dt), marker="*", s=sizes,
               c=BAND_COLORS, edgecolors="k", linewidths=rc_lw(1), zorder=3)
    ax.axhline(0.0, color="0.4", ls="--", lw=rc_lw(0.8), zorder=1)
    ax.set_xticks(range(N_BANDS), BAND_NAMES, rotation=90 if styled() else 0)
    if styled():
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.tick_params(axis="y", which="both", right=False)
    # Extra room on the right for the event numbers of the y band.
    ax.set_xlim(-0.6, N_BANDS - 0.15)
    ax.set_ylim(-c0 - 0.15, ref_dt + 0.45)
    ax.set_xlabel("band (RoPE column)")
    ax.set_ylabel(pub_text("tau (d)", r"$\tau$ (d)"))
    rows = ax.secondary_yaxis(
        "right", functions=(lambda t: (t + c0) / span * (H - 1),
                            lambda r: r / (H - 1) * span - c0))
    rows.set_ylabel("canvas row")
    ev_share = 100 * last[valid_idx].sum()
    ax.set_title("(a) Event and query tokens")

    # (b) Focus query attention vs Dt.
    a_dt = w_h[:, -1, L + focus][:, keys]
    im = axb.pcolormesh(np.arange(len(keys)), dts, a_dt, cmap="magma", shading="nearest")
    axb.axvline(n_ev - 0.5, color="w", lw=rc_lw(1))
    axb.axhline(ref_dt, color="w", ls=":", lw=rc_lw(1))
    key_band = np.concatenate([ev_band, np.arange(N_BANDS)])
    axb.set_xticks(range(len(keys)),
                   [str(j + 1) for j in range(n_ev)] + [f"Q {n}" for n in BAND_NAMES],
                   rotation=90, fontsize=0.85 * pub_fs(10, "xtick.labelsize"))
    for tick, b in zip(axb.get_xticklabels(), key_band):
        tick.set_color(BAND_COLORS[b])
    if styled():
        axb.minorticks_off()
    axb.set_xlabel("key token (events | band queries)")
    axb.set_ylabel(pub_text("lead time Dt (d)", r"lead time $\Delta t$ (d)"))
    axb.set_title(f"(b) Attention from Q {q_name}")
    fig.colorbar(im, ax=axb, pad=0.01, label="attention")

    handles = [
        Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=rc_ms(7),
               label="detection (event token)"),
        Line2D([], [], ls="", marker="*", mfc="w", mec="k", ms=rc_ms(11),
               label=pub_text(f"band query at Dt = {ref_dt:g} d",
                              rf"band query at $\Delta t$ = {ref_dt:g} d")),
        Line2D([], [], color="0.4", ls="--", lw=rc_lw(0.8),
               label=pub_text("last detection (tau = 0)", r"last detection ($\tau = 0$)")),
        Line2D([], [], color=BAND_COLORS[focus], lw=rc_lw(2.5), alpha=0.6,
               label=f"attention from Q {q_name} ({ev_share:.0f}% on events)"),
    ]
    if styled():
        fig.legend(handles=handles, ncol=2, loc="outside lower center")
    else:
        ax.legend(handles=handles, loc="center left", fontsize=7)
        fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{path_stem}.{fmt}", dpi=pub_dpi(150), bbox_inches="tight")
    plt.close(fig)


def plot_lightcurve_image(meta, means, stds, truth, ref_dt, canvas, horizon,
                          cell_days, path_stem, formats):
    """lightcurve_image_<stem>.{pdf,png}: a light curve and the time x band
    image it corresponds to.

    (a) The context detections the model gets (magnitude vs anchor-relative
    time tau), the uniform truth faintly, and the forecast time. (b) The same
    detections as pixels of a conceptual image: one row per band, one column
    per ``cell_days`` of time, each detection's cell colored by its
    standardized magnitude, all other cells empty; stars mark the band queries
    at the forecast time. The code never builds this image: the tokens only
    carry these (band, time) positions into attention via RoPE. The top axis
    gives the canvas row of each time (time is the row axis of the RoPE
    canvas, drawn horizontally here). Events in (b) are numbered as in
    tokens_paper.
    """
    H, c0, span = canvas
    valid_idx = np.nonzero(meta["valid"])[0]
    ev_band = meta["band"][valid_idx]
    ev_tau = meta["tau"][valid_idx]
    ev_val = meta["value"][valid_idx]
    ev_mag = ev_val * (stds[ev_band] + EPS) + means[ev_band]
    t_hi = min(horizon, span - c0)

    fig, (ax, axb) = plt.subplots(
        2, 1, sharex=True, height_ratios=[1.4, 1], constrained_layout=True,
        figsize=pub_size((7, 6.2)) if styled() else (7, 6.2))

    # (a) The light curve.
    ax.axvspan(0.0, t_hi, color="0.93", lw=0, zorder=0)
    ax.axvline(0.0, color="0.4", ls="--", lw=rc_lw(0.8), zorder=1)
    ax.axvline(ref_dt, color="0.4", ls=":", lw=rc_lw(0.8), zorder=1)
    if truth is not None:
        for b in range(N_BANDS):
            m = truth[2] == b
            if m.any():
                o = np.argsort(truth[0][m])
                ax.plot(truth[0][m][o], truth[1][m][o], color=BAND_COLORS[b],
                        lw=rc_lw(0.8), alpha=0.45, zorder=1)
    ax.scatter(ev_tau, ev_mag, c=[BAND_COLORS[b] for b in ev_band],
               s=60 * rc_ms(1) ** 2, edgecolors="k", linewidths=rc_lw(1), zorder=3)
    ax.invert_yaxis()
    ax.set_ylabel("magnitude")
    ax.set_title("(a) Light curve", loc="left")
    seen = sorted(set(ev_band.tolist()))
    handles = [Line2D([], [], ls="", marker="o", mfc=BAND_COLORS[b], mec="k",
                      ms=rc_ms(6), label=BAND_NAMES[b]) for b in seen]
    if truth is not None:
        handles.append(Line2D([], [], color="0.5", lw=rc_lw(0.8), alpha=0.6,
                              label="true light curve"))
    handles.append(Line2D([], [], color="0.4", ls=":", lw=rc_lw(0.8),
                          label=pub_text(f"forecast time Dt = {ref_dt:g} d",
                                         rf"forecast time $\Delta t$ = {ref_dt:g} d")))
    ax.legend(handles=handles, loc="lower left", ncol=2,
              fontsize=pub_fs(7, "legend.fontsize"))
    ax.text(0.5 * t_hi, 0.96, "forecast window", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color="0.4", fontsize=pub_fs(8, "xtick.labelsize"))

    # (b) The conceptual image: rows = band, columns = time cells.
    n_cells = int(round(span / cell_days))
    edges = -c0 + np.arange(n_cells + 1) * span / n_cells
    img = np.full((N_BANDS, n_cells), np.nan)
    cols = np.clip(((ev_tau + c0) / span * n_cells).astype(int), 0, n_cells - 1)
    img[ev_band, cols] = ev_val
    axb.set_facecolor("0.96")
    lim = max(np.nanmax(np.abs(img)), 1e-3) if np.isfinite(img).any() else 1.0
    im = axb.pcolormesh(edges, np.arange(N_BANDS + 1) - 0.5, img, cmap="RdBu_r",
                        vmin=-lim, vmax=lim, shading="flat")
    for gy in np.arange(N_BANDS + 1) - 0.5:
        axb.axhline(gy, color="0.8", lw=rc_lw(0.5), zorder=1)
    axb.axvline(0.0, color="0.4", ls="--", lw=rc_lw(0.8), zorder=2)
    axb.scatter(np.full(N_BANDS, ref_dt), np.arange(N_BANDS), marker="*",
                s=110 * rc_ms(1) ** 2, c=BAND_COLORS, edgecolors="k",
                linewidths=rc_lw(0.8), zorder=3)
    # White halo: the numbers stay legible on dark cells.
    halo = [patheffects.withStroke(linewidth=rc_lw(2), foreground="w")]
    for j in range(len(valid_idx)):
        axb.annotate(str(j + 1), (ev_tau[j], ev_band[j]), xytext=(4, 0), va="center",
                     textcoords="offset points", fontsize=pub_fs(7, "xtick.labelsize"),
                     path_effects=halo, zorder=4)
    axb.set_ylim(N_BANDS - 0.5, -0.5)
    axb.set_yticks(range(N_BANDS), BAND_NAMES)
    for tick, b in zip(axb.get_yticklabels(), range(N_BANDS)):
        tick.set_color(BAND_COLORS[b])
    if styled():
        axb.tick_params(axis="y", which="minor", left=False, right=False)
    axb.set_xlim(-c0, t_hi)
    axb.set_xlabel(pub_text("time from last detection tau (d)",
                            r"time from last detection $\tau$ (d)"))
    axb.set_ylabel("band")
    axb.set_title(pub_text(f"(b) As an image (cells of {cell_days:g} d; "
                           f"stars = band queries at Dt)",
                           rf"(b) As an image (cells of {cell_days:g} d; "
                           r"stars = band queries at $\Delta t$)"), loc="left")
    top = axb.secondary_xaxis("top", functions=(lambda t: (t + c0) / span * (H - 1),
                                                lambda r: r / (H - 1) * span - c0))
    top.set_xlabel("canvas row", fontsize=pub_fs(8, "xtick.labelsize"))
    top.tick_params(labelsize=pub_fs(7, "xtick.labelsize"))
    fig.colorbar(im, ax=axb, location="bottom", shrink=0.5, aspect=30,
                 label="standardized magnitude (+ = fainter)")
    for fmt in formats:
        fig.savefig(f"{path_stem}.{fmt}", dpi=pub_dpi(150), bbox_inches="tight")
    plt.close(fig)


def plot_forecast(obj, dts, mag_tok, mag_abl, truth, path, title):
    """forecast_<stem>.png: per-band forecast with / without the token path."""
    fig, axes = plt.subplots(3, 3, figsize=pub_size((13, 10)), sharex=True)
    for b, ax in enumerate(axes.flat):
        c = BAND_COLORS[b]
        if truth is not None:
            m = truth[2] == b
            ax.plot(truth[0][m], truth[1][m], color="0.55", lw=rc_lw(1.0),
                    label="uniform truth")
        m = obj["ctx_band"] == b
        ax.scatter(obj["ctx_tau"][m], obj["ctx_mag"][m], s=36 * rc_ms(1) ** 2, color=c,
                   edgecolors="k", linewidths=rc_lw(1), zorder=3, label="context")
        ax.plot(dts, mag_tok[:, b], color=c, lw=rc_lw(2), label="with token path")
        ax.plot(dts, mag_abl[:, b], color=c, lw=rc_lw(1.3), ls="--",
                label="token readout zeroed")
        ax.axvline(0.0, color="0.4", ls=":", lw=rc_lw(0.8))
        ax.invert_yaxis()
        ax.set_title(BAND_NAMES[b])
        if b == 0 and not styled():
            ax.legend(fontsize=7)
    if styled():
        # One legend below the grid, so it covers no band's curves.
        fig.legend(*axes.flat[0].get_legend_handles_labels(), ncol=4,
                   loc="outside lower center")
        fig.supxlabel("time from last context detection (d)",
                      fontsize=pub_fs(None, "axes.labelsize"))
        fig.supylabel("mag", fontsize=pub_fs(None, "axes.labelsize"))
    else:
        for ax in axes[-1]:
            ax.set_xlabel("time from last context detection (d)")
        for ax in axes[:, 0]:
            ax.set_ylabel("mag")
    fig.suptitle(pub_title(title), fontsize=pub_fs(None, "axes.titlesize"))
    pub_layout(fig)
    fig.savefig(path, dpi=pub_dpi(110))
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

    # Styled: one row two columns wide, shared x label and legend below.
    fig, axes = plt.subplots(1, n_blocks, sharey=True,
                             figsize=pub_size((10, 4.4)) if styled()
                             else (3.4 * n_blocks, 3.6))
    print("\nMean query attention per event, as a ratio to uniform (1 = none):")
    print(f"  {'block':>5} {'events %':>9}  same-band by age bin | other-band")
    for b, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(centers, ratio[b, 0], "o-", ms=rc_ms(6), label="event in query's band")
        ax.plot(centers, ratio[b, 1], "s--", ms=rc_ms(6), label="other band")
        ax.axhline(1.0, color="0.5", lw=rc_lw(0.8), ls=":")
        if not styled():
            ax.set_xlabel("event age before anchor (d)")
        ax.set_title(pub_text(f"block {b + 1}: {100 * ev_share[b]:.0f}% on events",
                              f"block {b + 1}: {100 * ev_share[b]:.0f}%"), fontsize=pub_fs(9))
        print(f"  {b + 1:>5} {100 * ev_share[b]:>8.1f}%  "
              + " ".join(f"{v:4.2f}" for v in ratio[b, 0]) + " | "
              + " ".join(f"{v:4.2f}" for v in ratio[b, 1]))
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel(pub_text("attention x n_keys (1 = uniform)",
                            r"attention $\times\,n_{\rm keys}$"))
    if styled():
        fig.supxlabel("event age before anchor (d)", fontsize=pub_fs(None, "axes.labelsize"))
        fig.legend(*ax0.get_legend_handles_labels(), ncol=2, loc="outside lower center")
    else:
        ax0.legend(fontsize=7)
    fig.suptitle(pub_title(title), fontsize=pub_fs(None, "axes.titlesize"))
    pub_layout(fig)
    fig.savefig(path, dpi=pub_dpi(110))
    plt.close(fig)


def main() -> None:
    """Load the checkpoint, hook the blocks, and write the figures."""
    args = get_args()
    use_pub_style(args.style, args.font_dir)
    BAND_COLORS[:] = pub_band_colors(BAND_COLORS)
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

    canvas = (model.image_size[0], model.render_context_days, model._render_t_span)
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
        plot_tokens_paper(meta, w_h, dts, ref_i, focus, args.ref_dt, canvas,
                          os.path.join(outdir, f"tokens_paper_{stem}"), args.formats)

        truth = lc_truth = None
        if stem in uni_map:
            u_t, u_v, u_b, _ul, _z = read_merged_stream(uni_map[stem], False)
            lead = u_t - obj["t_anchor"]
            ok = (lead <= args.horizon_days) & np.isfinite(u_v)
            m = ok & (lead > 0)
            truth = (lead[m], u_v[m], u_b[m])
            m = ok & (lead >= -canvas[1])
            lc_truth = (lead[m], u_v[m], u_b[m])
        plot_lightcurve_image(meta, means, stds, lc_truth, args.ref_dt, canvas,
                              args.horizon_days, args.cell_days,
                              os.path.join(outdir, f"lightcurve_image_{stem}"),
                              args.formats)

        abl = forward(model, obj["x"], dts, device, ablate=True)
        mag_tok = pred * (stds + EPS) + means
        mag_abl = abl * (stds + EPS) + means
        plot_forecast(obj, dts, mag_tok, mag_abl, truth,
                      os.path.join(outdir, f"forecast_{stem}.png"), title)
        print(f"  wrote tokens_{stem}.png, forecast_{stem}.png "
              f"(n_window {obj['n_window']}, focus Q {BAND_NAMES[focus]})")
    print(f"Figures in {outdir}")


if __name__ == "__main__":
    main()
