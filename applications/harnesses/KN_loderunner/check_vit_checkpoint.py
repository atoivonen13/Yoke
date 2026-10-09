"""Check that a LodeRunnerViT checkpoint strict-loads into the Study 145 config.

Builds the backbone with the ViT model_args from ``train_LodeRunner_ddp.py``,
loads the checkpoint, prints missing/unexpected keys and any shape mismatches,
then wraps it in the 9-band KN wrapper and runs one tiny CPU forward.

Usage:
    python check_vit_checkpoint.py /path/to/study002_modelState_epoch0100_ema_weights.pth
"""

import sys

import torch

from yoke.models.vit.swin.bomberman import (
    LodeRunnerViT,
    ScalarTemporalConditionedLodeRunner_9band,
)

VIT_MODEL_ARGS = {
    "default_vars": [
        "density_case",
        "density_cushion",
        "density_maincharge",
        "density_outside_air",
        "density_striker",
        "density_throw",
        "Uvelocity",
        "Wvelocity",
    ],
    "image_size": (1120, 400),
    "patch_size": (10, 5),
    "embed_dim": 2304,
    "num_heads": 8,
    "num_attention_heads": 12,
    "attention_head_dim": 192,
    "num_layers": 6,
    "rope_theta": 10000,
    "rope_scale": (80, 224),
    "mlp_ratio": 1.0,
    "concat_mlp": True,
    "verbose": False,
    "num_input_frames": 2,
    "eps": 1e-7,
    "bias": True,
}


def main(path: str) -> None:
    """Load ``path`` into the Study 145 ViT and report."""
    model = LodeRunnerViT(**VIT_MODEL_ARGS)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    if all(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    print(f"checkpoint tensors: {len(sd)}, params: {sum(v.numel() for v in sd.values()):,}")

    own = model.state_dict()
    bad_shapes = [
        (k, tuple(v.shape), tuple(own[k].shape))
        for k, v in sd.items()
        if k in own and v.shape != own[k].shape
    ]
    print("shape mismatches:", bad_shapes or "none")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing keys:", missing or "none")
    print("unexpected keys:", unexpected or "none")
    if missing or unexpected or bad_shapes:
        sys.exit("FAIL: checkpoint does not match the Study 145 ViT config.")

    wrapper = ScalarTemporalConditionedLodeRunner_9band(
        backbone=model,
        context_len=12,
        n_bands=9,
        image_size=VIT_MODEL_ARGS["image_size"],
        backbone_channels=8,
        context_window_days=2.0,
        pool_mode="meanstdmax",
        backbone_dt_in=0.25,
    ).eval()
    x = torch.randn(1, 12 * (3 + 9))  # time-window layout: [value, rel_t, valid, one-hot]
    with torch.no_grad():
        out = wrapper(x, torch.arange(8), torch.arange(8), torch.tensor([3.0]))
    print("wrapper forward:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
    print("OK: strict load + forward succeeded.")


if __name__ == "__main__":
    main(sys.argv[1])
