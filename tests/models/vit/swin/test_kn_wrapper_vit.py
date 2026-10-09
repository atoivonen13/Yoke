"""Tests for the 9-band KN wrapper around a LodeRunnerViT backbone.

Covers the 2-frame backbone call (duplicated pseudo-image, ``(dt_in, Dt)`` lead
times), the single-frame ViT path, and the ViT fine-tune scopes built by
:func:`yoke.utils.checkpointing.build_finetune_optimizer`.
"""

import pytest
import torch

from yoke.models.vit.swin.bomberman import (
    LodeRunnerViT,
    ScalarTemporalConditionedLodeRunner_9band,
)
from yoke.utils.checkpointing import (
    VIT_DECODER_BLOCKS,
    build_finetune_optimizer,
)

N_BANDS = 9
CONTEXT_LEN = 5
# Time-window context, as in training (CONTEXT_WINDOW_DAYS=2.0): each event is
# [value, rel_t, valid, one_hot_band].
CONTEXT_WINDOW_DAYS = 2.0
EVENT_WIDTH = 3 + N_BANDS
IMAGE_SIZE = (20, 10)  # (10, 5) patches -> 2 x 2 token grid


def _make_wrapper(num_input_frames: int) -> ScalarTemporalConditionedLodeRunner_9band:
    """Tiny ViT backbone inside the 9-band wrapper."""
    backbone = LodeRunnerViT(
        default_vars=[f"v{i}" for i in range(8)],
        image_size=IMAGE_SIZE,
        patch_size=(10, 5),
        embed_dim=32,
        num_heads=4,
        num_attention_heads=4,
        attention_head_dim=8,
        num_layers=3,
        mlp_ratio=1.0,
        num_input_frames=num_input_frames,
        bias=True,
    )
    return ScalarTemporalConditionedLodeRunner_9band(
        backbone=backbone,
        context_len=CONTEXT_LEN,
        n_bands=N_BANDS,
        image_size=IMAGE_SIZE,
        backbone_channels=8,
        hidden=16,
        context_window_days=CONTEXT_WINDOW_DAYS,
        pool_mode="meanstdmax",
        backbone_dt_in=0.25,
    )


def _inputs(batch: int = 3) -> tuple:
    x = torch.randn(batch, CONTEXT_LEN * EVENT_WIDTH)
    in_vars = torch.arange(8)
    Dt = torch.rand(batch) * 5.0
    return x, in_vars, Dt


@pytest.mark.parametrize("num_input_frames", [1, 2])
def test_wrapper_forward_shape(num_input_frames: int) -> None:
    """The wrapper runs end-to-end on either ViT input mode."""
    model = _make_wrapper(num_input_frames)
    x, in_vars, Dt = _inputs()
    out = model(x, in_vars, in_vars, Dt)
    assert out.shape == (3, N_BANDS)
    assert torch.isfinite(out).all()


def test_two_frame_backbone_inputs() -> None:
    """A 2-frame backbone gets [img, img] and lead_times (dt_in, Dt)."""
    model = _make_wrapper(2)
    x, in_vars, Dt = _inputs()

    seen = {}

    def hook(_mod: torch.nn.Module, args: tuple) -> None:
        seen["x"], _, _, seen["lt"] = args

    model.backbone.register_forward_pre_hook(hook)
    model(x, in_vars, in_vars, Dt)

    assert seen["x"].shape == (3, 2, 8, *IMAGE_SIZE)
    assert torch.equal(seen["x"][:, 0], seen["x"][:, 1])
    assert seen["lt"].shape == (3, 2)
    assert torch.allclose(seen["lt"][:, 0], torch.full((3,), 0.25))
    assert torch.allclose(seen["lt"][:, 1], Dt)


@pytest.mark.parametrize("scope", ["tail", "decoder", "full"])
def test_vit_finetune_scopes(scope: str) -> None:
    """Each ViT scope unfreezes exactly its modules, and they get gradients."""
    model = _make_wrapper(2)
    vit = model.backbone
    blocks = vit.backbone.transformer_blocks

    _, lr_mults = build_finetune_optimizer(
        model,
        {"lr": 1e-3},
        backbone_tail_lr_mult=0.1,
        backbone_finetune_scope=scope,
    )
    assert lr_mults == [1.0, 0.1]

    n_trainable = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    if scope == "tail":
        expected = sum(p.numel() for p in vit.linear4unpatch.parameters())
    elif scope == "decoder":
        expected = sum(
            p.numel()
            for mod in [*list(blocks)[-VIT_DECODER_BLOCKS:], vit.linear4unpatch]
            for p in mod.parameters()
        )
    else:
        expected = sum(p.numel() for p in vit.parameters())
    assert n_trainable == expected

    # Every unfrozen backbone param must be used by the forward (DDP runs with
    # find_unused_parameters=False).
    x, in_vars, Dt = _inputs()
    model(x, in_vars, in_vars, Dt).sum().backward()
    for name, p in vit.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name


def test_vit_bad_scope_raises() -> None:
    """An unknown scope is rejected for the ViT backbone too."""
    model = _make_wrapper(2)
    with pytest.raises(ValueError):
        build_finetune_optimizer(
            model,
            {"lr": 1e-3},
            backbone_tail_lr_mult=0.1,
            backbone_finetune_scope="encoder",
        )
