"""1D U-Net baseline for the 9-band late-time forecast.

A conventional CNN to compare the ViT LodeRunner against: no attention, no
pretraining, no event tokens. Standalone (torch + numpy, no yoke import); the
context / scored-point protocol and the I/O are ``baseline_extrapolate_9band.py``'s,
which mirrors ``eval_dense_latetime_9band.py`` (DIRECT mode), so the test objects
and points are the model's and paired deltas against a study are valid.

Per object, as the eval: t0 = first realistic detection; the context is the
realistic detections (upper limits dropped) with phase <= 2 d, capped at the most
recent 12; scored are the dense and uniform points with 2 < phase <= 10 d and
lead > 0.

Input: the context binned onto a phase grid of ``--n_bins`` bins of ``--bin_days``
(default 128 x 0.08 d = 10.24 d), one channel per band (per-band z-scored mag,
mean of the detections in the bin, 0 if none), one observed mask per band, and
the phase -> [19, T]. Output: 3 quantiles (0.05 / 0.5 / 0.95, monotone) per band
on the same grid, read at each scored point's phase by linear interpolation.

Network: encoder ConvBlock + max-pool per level, a bottleneck with a global mix
(mean over time -> linear -> added back, so every late bin sees the whole
context; plain 3-tap convs at 16 bins do not reach from day 1 to day 10), and a
decoder of transposed convs with skip connections.

Training: train-split objects, eval-matched (same context rule, uniform-truth
late targets), pinball loss in per-band z units with the recipe's manual band
weights, AdamW + one cosine anneal over the whole run, optional context-
detection dropout. The kept checkpoint is the epoch with the best validation
uniform RMSE. Intervals: per-band k_lo / k_hi fit on the validation uniform
truth (``calibrate_intervals.fit_scales``, as the eval's default).

Outputs mimic a trained + evaluated study, so the plot scripts read it as one:
``runs/study_<--study>/`` gets per-epoch ``training_`` / ``validation_study<N>_
epoch<E>.csv`` loss records (pinball, z units -- not on the LodeRunner's scale),
``dense_latetime_eval_9band/`` the scored-point CSVs, ``interval_calibration.json``
and an ``eval_meta.json`` naming the kept epoch, and ``unet/`` the checkpoint,
per-epoch history and settings:

    python unet_baseline_9band.py --study 910 --seed 0
    python plot_metrics_by_day.py --studies 213 910 903 --truth uniform
    python plot_compare_runs.py --studies 213 910

``--seed`` makes seed replicas (e.g. --study 91000 .. 91004) for the
``plot_seed_replicas.py`` comparison.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_extrapolate_9band import (  # noqa: E402
    BAND_KEY_SETS,
    BAND_NAMES,
    DATA_DIR,
    FILELIST_DIR,
    IO_WORKERS,
    N_BANDS,
    _read_stems,
    _stem_to_path,
    dense_depths,
    detect_band_set,
    read_stream,
    report,
    split_object,
)
from calibrate_intervals import apply_scales, fit_scales  # noqa: E402

QUANTILE_LEVELS = (0.05, 0.5, 0.95)
# The LodeRunner recipe's manual BAND_WEIGHTS (ztf 2, u 3, g 2, rest 1).
BAND_WEIGHTS = (2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0)
IN_CHANNELS = 2 * N_BANDS + 1  # mags, observed masks, phase
EVAL_SUBDIR = "dense_latetime_eval_9band"


# --------------------------------------------------------------------- model
class ConvBlock(nn.Sequential):
    """Two (conv3 -> GroupNorm -> GELU) layers."""

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0) -> None:
        """Map ``c_in`` -> ``c_out`` channels at the same length."""
        super().__init__(
            nn.Conv1d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.GELU(),
        )


class UNet1d(nn.Module):
    """1D U-Net: [B, in_ch, T] -> quantiles [B, n_bands, 3, T] (z units).

    ``widths`` = channels per level; the last is the bottleneck, so T must be
    divisible by 2 ** (len(widths) - 1).
    """

    def __init__(self, in_ch: int = IN_CHANNELS, n_bands: int = N_BANDS,
                 widths: tuple = (32, 64, 128, 256), dropout: float = 0.1) -> None:
        """Build the encoder, bottleneck (+ global mix), decoder and head."""
        super().__init__()
        self.n_bands = n_bands
        self.down = nn.ModuleList()
        c = in_ch
        for w in widths[:-1]:
            self.down.append(ConvBlock(c, w, dropout))
            c = w
        self.bottleneck = ConvBlock(c, widths[-1], dropout)
        self.mix = nn.Linear(widths[-1], widths[-1])
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        c = widths[-1]
        for w in reversed(widths[:-1]):
            self.up.append(nn.ConvTranspose1d(c, w, 2, stride=2))
            self.dec.append(ConvBlock(2 * w, w, dropout))
            c = w
        self.head = nn.Conv1d(c, n_bands * len(QUANTILE_LEVELS), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Monotone quantiles: low = med - softplus, high = med + softplus."""
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool1d(x, 2)
        x = self.bottleneck(x)
        x = x + self.mix(x.mean(-1)).unsqueeze(-1)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            x = block(torch.cat([up(x), skip], dim=1))
        o = self.head(x).view(x.shape[0], self.n_bands, len(QUANTILE_LEVELS), -1)
        med = o[:, :, 1]
        return torch.stack([med - F.softplus(o[:, :, 0]), med,
                            med + F.softplus(o[:, :, 2])], dim=2)


def at_points(q: torch.Tensor, band: torch.Tensor, phase: torch.Tensor,
              bin_days: float) -> torch.Tensor:
    """Quantiles [B, P, 3] of grid output ``q`` [B, n_bands, 3, T] at points.

    Linear interpolation between bin centers ((i + 0.5) * bin_days).
    """
    n_b, n_q, n_t = q.shape[1], q.shape[2], q.shape[3]
    pos = (phase / bin_days - 0.5).clamp(0.0, n_t - 1.0)
    i0 = pos.floor().long().clamp(max=n_t - 2)
    w = (pos - i0).unsqueeze(-1)
    flat = q.permute(0, 1, 3, 2).reshape(q.shape[0], n_b * n_t, n_q)

    def take(i):
        return flat.gather(1, (band * n_t + i).unsqueeze(-1).expand(-1, -1, n_q))

    return (1.0 - w) * take(i0) + w * take(i0 + 1)


def pinball(pred: torch.Tensor, true: torch.Tensor, weight: torch.Tensor
            ) -> torch.Tensor:
    """Weighted mean pinball loss; pred [B, P, 3], true / weight [B, P]."""
    levels = pred.new_tensor(QUANTILE_LEVELS)
    e = true.unsqueeze(-1) - pred
    loss = torch.maximum(levels * e, (levels - 1.0) * e).mean(-1)
    return (loss * weight).sum() / weight.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------- data
def load_objects(stems, maps, band_keys, args) -> tuple:
    """The eval's context and scored points per object (None-objects dropped)."""
    def read(stem):
        uni = (read_stream(maps["uniform"][stem], band_keys, False)
               if stem in maps["uniform"] else None)
        return stem, split_object(
            read_stream(maps["realistic"][stem], band_keys, True),
            read_stream(maps["dense"][stem], band_keys, False), uni,
            args.late_time_cutoff_days, args.late_time_max_days)

    with ThreadPoolExecutor(IO_WORKERS) as pool:
        pairs = [(s, o) for s, o in pool.map(read, stems) if o is not None]
    # The eval's cap: the most recent max_context_len detections (its trailing
    # 2 d window already holds every detection at phase <= 2 d).
    for _, obj in pairs:
        obj["ctx"] = tuple(a[-args.max_context_len:] for a in obj["ctx"])
    return [s for s, _ in pairs], [o for _, o in pairs]


def grid_inputs(objs, mu, sd, args) -> torch.Tensor:
    """Binned context -> [N, IN_CHANNELS, n_bins] (z-scored mags, masks, phase)."""
    n_t = args.n_bins
    x = np.zeros((len(objs), IN_CHANNELS, n_t), dtype=np.float32)
    x[:, -1] = (np.arange(n_t) + 0.5) / n_t
    for n, obj in enumerate(objs):
        c_p, c_v, c_b = obj["ctx"]
        i = np.clip((c_p / args.bin_days).astype(np.int64), 0, n_t - 1)
        total, count = np.zeros((N_BANDS, n_t)), np.zeros((N_BANDS, n_t))
        np.add.at(total, (c_b, i), (c_v - mu[c_b]) / sd[c_b])
        np.add.at(count, (c_b, i), 1.0)
        seen = count > 0
        x[n, :N_BANDS][seen] = total[seen] / count[seen]
        x[n, N_BANDS:2 * N_BANDS] = seen
    return torch.from_numpy(x)


def pad_points(objs, truth: str) -> dict:
    """Points of one truth padded to [N, P] (objects without it are all-masked)."""
    lens = [obj["points"][truth]["band"].shape[0] if truth in obj["points"] else 0
            for obj in objs]
    p = max(max(lens), 1)
    band = np.zeros((len(objs), p), dtype=np.int64)
    phase = np.zeros((len(objs), p), dtype=np.float32)
    true = np.zeros((len(objs), p), dtype=np.float32)
    mask = np.zeros((len(objs), p), dtype=bool)
    for n, (obj, k) in enumerate(zip(objs, lens)):
        if k:
            pts = obj["points"][truth]
            band[n, :k], phase[n, :k] = pts["band"], pts["phase"]
            true[n, :k], mask[n, :k] = pts["true"], True
    return {k: torch.from_numpy(v) for k, v in
            (("band", band), ("phase", phase), ("true", true), ("mask", mask))}


def flat_points(stems, objs, truth: str) -> dict:
    """Points of one truth concatenated in object order (``pad_points`` order)."""
    keep = [(s, o["points"][truth]) for s, o in zip(stems, objs)
            if truth in o["points"]]
    out = {k: np.concatenate([p[k] for _, p in keep])
           for k in ("band", "phase", "lead", "true")}
    out["stem"] = np.concatenate([np.full(p["band"].shape[0], s, dtype=object)
                                  for s, p in keep])
    return out


def norm_stats(objs) -> tuple:
    """Per-band mean / std of the train uniform late truth (finite values)."""
    pts = [o["points"]["uniform"] for o in objs if "uniform" in o["points"]]
    band = np.concatenate([p["band"] for p in pts])
    true = np.concatenate([p["true"] for p in pts]).astype(np.float64)
    mu, sd = np.zeros(N_BANDS), np.ones(N_BANDS)
    for b in range(N_BANDS):
        v = true[(band == b) & np.isfinite(true)]
        if v.size:
            mu[b], sd[b] = v.mean(), max(v.std(), 1e-3)
    return mu, sd


# ------------------------------------------------------------ train / predict
def drop_context(x: torch.Tensor, p: float) -> torch.Tensor:
    """Drop each observed (band, bin) with prob ``p``, keeping >= 1 per object."""
    if p <= 0.0:
        return x
    seen = x[:, N_BANDS:2 * N_BANDS] > 0
    drop = (torch.rand_like(x[:, :N_BANDS]) < p) & seen
    drop[(seen & ~drop).flatten(1).sum(1) == 0] = False
    keep = (~drop).float()
    x = x.clone()
    x[:, :N_BANDS] *= keep
    x[:, N_BANDS:2 * N_BANDS] *= keep
    return x


@torch.no_grad()
def predict(model, x, pts, mu_t, sd_t, args, device) -> np.ndarray:
    """Quantiles in mag at the valid points, flat [n_points, 3] (object order)."""
    model.eval()
    out = []
    for s in range(0, x.shape[0], args.batch_size):
        sl = slice(s, s + args.batch_size)
        band = pts["band"][sl].to(device)
        q = at_points(model(x[sl].to(device)), band, pts["phase"][sl].to(device),
                      args.bin_days)
        q = q * sd_t[band].unsqueeze(-1) + mu_t[band].unsqueeze(-1)
        out.append(q[pts["mask"][sl].to(device)].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def batch_loss(model, x, pts, mu_t, sd_t, w_band, args, device) -> torch.Tensor:
    """Weighted pinball (z units) of one batch's valid uniform points."""
    band = pts["band"].to(device)
    true = pts["true"].to(device)
    mask = pts["mask"].to(device) & torch.isfinite(true)
    z = ((true - mu_t[band]) / sd_t[band]).nan_to_num()
    q = at_points(model(x), band, pts["phase"].to(device), args.bin_days)
    return pinball(q, z, w_band[band] * mask)


def write_record(path, epoch, losses) -> None:
    """One loss row per batch, ``epoch, batch, loss`` (the trainer's format)."""
    np.savetxt(path, np.column_stack([np.full(len(losses), epoch),
                                      np.arange(1, len(losses) + 1), losses]),
               fmt="%d, %d, %.8f")


def train(model, x_tr, p_tr, x_va, p_va, mu, sd, args, device, outdir,
          run_dir) -> dict:
    """Train; keep the best-validation-uniform-RMSE weights. Returns history.

    Per-epoch train / validation batch losses go to ``run_dir`` as
    ``training_study<N>_epoch<E>.csv`` / ``validation_study<N>_epoch<E>.csv``,
    which the plot scripts' loss panels read.
    """
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)
    w_band = torch.tensor(BAND_WEIGHTS, dtype=torch.float32, device=device)
    record = os.path.join(run_dir, "{}_study" + f"{args.study:03d}" + "_epoch{:04d}.csv")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = args.epochs * -(-x_tr.shape[0] // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=args.warmup_frac,
        anneal_strategy="cos")
    va_true = p_va["true"][p_va["mask"]].numpy().astype(np.float64)
    va_ok = np.isfinite(va_true)
    best, history, ckpt = np.inf, [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start, losses = time.time(), []
        for idx in torch.randperm(x_tr.shape[0]).split(args.batch_size):
            x = drop_context(x_tr[idx].to(device), args.context_dropout)
            loss = batch_loss(model, x, {k: v[idx] for k, v in p_tr.items()},
                              mu_t, sd_t, w_band, args, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_losses = [
                batch_loss(model, x_va[s:s + args.batch_size].to(device),
                           {k: v[s:s + args.batch_size] for k, v in p_va.items()},
                           mu_t, sd_t, w_band, args, device).item()
                for s in range(0, x_va.shape[0], args.batch_size)]
        write_record(record.format("training", epoch), epoch, losses)
        write_record(record.format("validation", epoch), epoch, val_losses)
        med = predict(model, x_va, p_va, mu_t, sd_t, args, device)[:, 1]
        rmse = float(np.sqrt(np.mean((med - va_true)[va_ok] ** 2)))
        if rmse < best:
            best = rmse
            if ckpt is not None:
                os.remove(ckpt)
            # "epoch<E>" in the name is how the plot scripts find the kept epoch.
            ckpt = os.path.join(outdir, f"unet_best_epoch{epoch:04d}.pth")
            torch.save(model.state_dict(), ckpt)
        history.append({"epoch": epoch, "train_pinball_z": float(np.mean(losses)),
                        "val_pinball_z": float(np.mean(val_losses)),
                        "val_uniform_rmse": rmse, "lr": sched.get_last_lr()[0]})
        print(f"epoch {epoch:4d}  train pinball {np.mean(losses):.4f}  "
              f"val pinball {np.mean(val_losses):.4f}  "
              f"val uniform RMSE {rmse:.4f}{'  *' if rmse == best else ''}  "
              f"({time.time() - t_start:.1f} s)", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    best_epoch = min(history, key=lambda h: h["val_uniform_rmse"])["epoch"]
    print(f"Kept epoch {best_epoch} (val uniform RMSE {best:.4f})")
    return {"history": history, "best_epoch": best_epoch, "best_val_rmse": best,
            "ckpt": ckpt}


# --------------------------------------------------------------------- output
def write_points(path, pts, q, cal=None) -> None:
    """Scored-points CSV in the eval's schema (+ calibrated band when ``cal``)."""
    cols = ["stem", "band", "phase_days", "lead_time_days", "pred_mag",
            "pred_low", "pred_high", "true_mag", "residual_mag"]
    if cal is not None:
        cols += ["pred_low_cal", "pred_high_cal"]
        lo_cal, hi_cal = cal
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(q.shape[0]):
            row = [pts["stem"][i], BAND_NAMES[pts["band"][i]],
                   f"{pts['phase'][i]:.4f}", f"{pts['lead'][i]:.4f}",
                   f"{q[i, 1]:.4f}", f"{q[i, 0]:.4f}", f"{q[i, 2]:.4f}",
                   f"{pts['true'][i]:.4f}", f"{q[i, 1] - pts['true'][i]:.4f}"]
            if cal is not None:
                row += [f"{lo_cal[i]:.4f}", f"{hi_cal[i]:.4f}"]
            w.writerow(row)
    print(f"Wrote {path}")


def calibrated(pts, q, scales) -> tuple:
    """``calibrate_intervals.apply_scales`` with the per-band k_lo / k_hi."""
    names = [BAND_NAMES[b] for b in pts["band"]]
    k_lo = np.array([scales.get(n, {"k_lo": 1.0})["k_lo"] for n in names])
    k_hi = np.array([scales.get(n, {"k_hi": 1.0})["k_hi"] for n in names])
    return apply_scales(q[:, 1], q[:, 0], q[:, 2], k_lo, k_hi)


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sets = os.path.join(DATA_DIR, "rubin_lsst_ztf{}_10000_dataset_same_seed",
                        "lc_*.npz")
    p.add_argument("--realistic_glob", default=sets.format(""))
    p.add_argument("--dense_glob", default=sets.format("_dense"))
    p.add_argument("--uniform_glob", default=sets.format("_uniform"))
    for split in ("train", "val", "test"):
        p.add_argument(f"--{split}_filelist", default=os.path.join(
            FILELIST_DIR, f"kn_rubin_ztf_{split}.txt"))
    p.add_argument("--max_objects", type=int, default=0,
                   help="Cap objects per split (0 = all; >0 for a smoke test).")
    p.add_argument("--late_time_cutoff_days", type=float, default=2.0)
    p.add_argument("--late_time_max_days", type=float, default=10.0)
    p.add_argument("--max_context_len", type=int, default=12,
                   help="Keep the most recent N context detections (the eval's "
                   "cap; 0 = all).")
    p.add_argument("--n_bins", type=int, default=128)
    p.add_argument("--bin_days", type=float, default=0.08)
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--context_dropout", type=float, default=0.2,
                   help="Train-time prob of dropping each context (band, bin).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--interval", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--study", type=int, default=910,
                   help="Outputs -> <runs_dir>/study_<study>/.")
    return p.parse_args()


def main() -> None:
    """Load, train, select on val, calibrate on val, score test, write."""
    args = get_args()
    if args.n_bins % 2 ** (len(args.widths) - 1):
        raise SystemExit("--n_bins must be divisible by 2 ** (len(widths) - 1)")
    if args.n_bins * args.bin_days < args.late_time_max_days:
        raise SystemExit("the grid (n_bins * bin_days) must cover "
                         "--late_time_max_days")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maps = {k: _stem_to_path(getattr(args, f"{k}_glob"))
            for k in ("realistic", "dense", "uniform")}
    for k, m in maps.items():
        if not m:
            raise SystemExit(f"--{k}_glob matched no files")
    band_set = detect_band_set(next(iter(maps["realistic"].values())))
    band_keys = BAND_KEY_SETS[band_set]
    paired = set(maps["realistic"]) & set(maps["dense"])
    split = {}
    for name in ("train", "val", "test"):
        stems = sorted(paired & _read_stems(getattr(args, f"{name}_filelist")))
        if args.max_objects > 0:
            stems = stems[:args.max_objects]
        split[name] = load_objects(stems, maps, band_keys, args)
    if set(split["test"][0]) & (set(split["val"][0]) | set(split["train"][0])):
        raise SystemExit("test split overlaps train / val")
    # Training needs late uniform targets.
    split["train"] = tuple(map(list, zip(*[
        (s, o) for s, o in zip(*split["train"]) if "uniform" in o["points"]])))
    print(f"Band set {band_set}; objects: train {len(split['train'][0])}, val "
          f"{len(split['val'][0])}, test {len(split['test'][0])}; device {device}")

    mu, sd = norm_stats(split["train"][1])
    x = {k: grid_inputs(objs, mu, sd, args) for k, (_, objs) in split.items()}
    padded = {k: {t: pad_points(objs, t) for t in ("dense", "uniform")}
              for k, (_, objs) in split.items()}

    run_dir = os.path.join(args.runs_dir, f"study_{args.study:03d}")
    unet_dir = os.path.join(run_dir, "unet")
    eval_dir = os.path.join(run_dir, EVAL_SUBDIR)
    for d in (unet_dir, os.path.join(eval_dir, "calibration_split")):
        os.makedirs(d, exist_ok=True)
    # A re-run replaces the old records / checkpoint (a shorter run would
    # otherwise leave the old later epochs in the loss plots).
    stale = (glob.glob(os.path.join(run_dir, f"*_study{args.study:03d}_epoch*.csv"))
             + glob.glob(os.path.join(unet_dir, "unet_best_epoch*.pth")))
    for path in stale:
        os.remove(path)
    if stale:
        print(f"Removed {len(stale)} records / checkpoints of a previous run")

    model = UNet1d(widths=tuple(args.widths), dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"UNet1d widths {args.widths}: {n_par / 1e6:.2f} M parameters")
    fit = train(model, x["train"], padded["train"]["uniform"], x["val"],
                padded["val"]["uniform"], mu, sd, args, device, unet_dir, run_dir)
    with open(os.path.join(unet_dir, "history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fit["history"][0]))
        w.writeheader()
        w.writerows(fit["history"])

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)
    pred = {k: {t: (flat_points(*split[k], t),
                    predict(model, x[k], padded[k][t], mu_t, sd_t, args, device))
                for t in ("dense", "uniform")} for k in ("val", "test")}

    # Interval scales on the validation uniform truth (the eval's default).
    pts, q = pred["val"]["uniform"]
    ok = np.isfinite(pts["true"])
    scales = fit_scales({"band": np.array(BAND_NAMES, dtype=object)[pts["band"][ok]],
                         "med": q[ok, 1], "low": q[ok, 0], "high": q[ok, 2],
                         "true": pts["true"][ok].astype(np.float64)}, args.interval)
    with open(os.path.join(eval_dir, "interval_calibration.json"), "w") as fh:
        json.dump({"interval": args.interval, "truth": "uniform", "bands": scales,
                   "calibration_split": None,
                   "note": f"unet_baseline_9band.py study {args.study}"}, fh,
                  indent=2)

    for name, sub in (("test", eval_dir),
                      ("val", os.path.join(eval_dir, "calibration_split"))):
        for truth, fname in (("dense", "latetime_scored_points.csv"),
                             ("uniform", "latetime_uniform_scored_points.csv")):
            pts, q = pred[name][truth]
            cal = calibrated(pts, q, scales)
            write_points(os.path.join(sub, fname), pts, q,
                         cal if truth == "dense" else None)
            if name == "test":
                report(f"unet (study {args.study})", truth, pts, q[:, 1], *cal,
                       dense_depths(pred["test"]["dense"][0])
                       if truth == "uniform" else None)

    with open(os.path.join(unet_dir, "unet_meta.json"), "w") as fh:
        json.dump({"study": args.study, "band_set": band_set,
                   "n_parameters": n_par, "best_epoch": fit["best_epoch"],
                   "best_val_uniform_rmse": fit["best_val_rmse"],
                   "norm_mu": mu.tolist(), "norm_sd": sd.tolist(),
                   "quantile_levels": QUANTILE_LEVELS, "band_weights": BAND_WEIGHTS,
                   "n_objects": {k: len(v[0]) for k, v in split.items()},
                   "args": vars(args)}, fh, indent=2)
    # The subset of the eval's signature the plot scripts read: the kept epoch
    # (from the checkpoint name), the scored phase range, and the data globs for
    # plot_run_report's median-example panel.
    with open(os.path.join(eval_dir, "eval_meta.json"), "w") as fh:
        json.dump({"signature": {
            "ckpt": os.path.realpath(fit["ckpt"]),
            "late_time_cutoff_days": args.late_time_cutoff_days,
            "late_time_max_days": args.late_time_max_days,
            "realistic_glob": args.realistic_glob,
            "dense_glob": args.dense_glob,
            "uniform_glob": args.uniform_glob,
            "n_stems": len(split["test"][0]),
            "score_uniform": True,
            "model": "unet_baseline_9band"},
            "has_uniform": True}, fh, indent=2)


if __name__ == "__main__":
    main()
