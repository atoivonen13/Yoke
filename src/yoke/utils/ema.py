"""Exponential-moving-average (EMA) utilities for LodeRunner training.

This module provides a self-contained, ``diffusers``-free implementation of a
Diffusers-style *warmup* EMA built on top of
:class:`torch.optim.swa_utils.AveragedModel`. It maintains a non-gradient shadow
copy of the trainable parameters that is updated after each optimizer step, with
a decay schedule that warms up according to the number of averaging steps.

Target behavior (matching the ArtIMich recipe):

- EMA is a non-gradient shadow copy of the trainable parameters, updated
  *after each optimizer step*.
- Diffusers warmup decay schedule::

      s = num_averaged
      decay = clamp(1 - (1 + s / inv_gamma) ** (-power), min_decay, max_decay)
      ema_p = ema_p * decay + model_p * (1 - decay)

- The update is only applied once the global optimizer-step counter exceeds
  ``ema_update_after_step`` (default 1000). The first call to
  :meth:`AveragedModel.update_parameters` simply copies the model
  (``num_averaged == 0`` before the ``multi_avg_fn`` runs), and subsequent calls
  begin averaging with ``s = 1, 2, ...``.
- The EMA weights are saved as the production checkpoint.
- Under DDP each rank maintains an identical EMA copy after the synchronized
  optimizer step; only rank 0 need write the checkpoint.

Typical usage::

    from yoke.utils.ema import build_ema_model, save_ema_checkpoint

    ema_model = build_ema_model(model, device=device)
    global_step = 0
    ...
    # inside the training loop, after optimizer.step() and LRsched.step():
    global_step += 1
    if global_step > ema_update_after_step:
        ema_model.update_parameters(model)
    ...
    # at end of training (rank 0):
    save_ema_checkpoint(ema_model, "ema_weights.pth")

:class:`ParamEMA` is a lighter alternative used by the KN LodeRunner harness: a
small, framework-light EMA shadow for a *subset* of a model's parameters --
here the trainable conditioner + output-head of the scalar-temporal LodeRunner
wrapper, with the frozen backbone deliberately excluded. Keeping the shadow keyed
by parameter NAME (rather than by position) makes it robust to reload: the
state_dict round-trips through the training checkpoint so the average survives the
per-epoch process restart used by the DDP harness (``cycle_epochs=1``), and the
eval scripts can overlay it onto a freshly-constructed model without needing an
optimizer.

Typical use in training::

    ema = ParamEMA(
        ((n, p) for n, p in model.named_parameters() if p.requires_grad),
        decay=0.999,
    )
    ...
    optimizer.step()
    ema.update((n, p) for n, p in model.named_parameters() if p.requires_grad)
    ...
    torch.save({..., "ema_state_dict": ema.state_dict()}, path)

and in eval to compare against the raw weights::

    ema = ParamEMA(named_trainable_params, decay=ckpt["ema_decay"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    ema.copy_to(model.named_parameters())  # partial overlay of the shadowed subset

"""

import os
from collections.abc import Callable, Iterable

import torch
from torch.optim.swa_utils import AveragedModel


def make_warmup_ema_fn(
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
    min_decay: float = 0.0,
) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
    r"""Build a Diffusers-style warmup ``multi_avg_fn`` for ``AveragedModel``.

    The returned callable has the signature expected by
    :class:`torch.optim.swa_utils.AveragedModel`'s ``multi_avg_fn`` argument:
    ``fn(ema_params, model_params, num_averaged)`` where ``ema_params`` and
    ``model_params`` are lists of parameter tensors and ``num_averaged`` is the
    number of models already averaged (a scalar tensor). The EMA parameters are
    updated *in place*.

    The decay at averaging step ``s = num_averaged`` is:

    .. math::

        \beta(s) = \mathrm{clamp}\left(
            1 - (1 + s / \gamma)^{-p},\; \beta_{\min},\; \beta_{\max}
        \right)

    and the update is ``ema_p = ema_p * beta + model_p * (1 - beta)``.

    Args:
        max_decay (float): Maximum (asymptotic) decay :math:`\beta_{\max}`.
        inv_gamma (float): Inverse-gamma factor :math:`\gamma` controlling the
            warmup rate.
        power (float): Warmup power :math:`p`.
        min_decay (float): Minimum decay :math:`\beta_{\min}` (floor at early
            steps).

    Returns:
        Callable: A ``multi_avg_fn`` for :class:`AveragedModel`.

    """

    def warmup_ema_fn(
        ema_params: list[torch.Tensor],
        model_params: list[torch.Tensor],
        num_averaged: int | torch.Tensor,
    ) -> None:
        """In-place warmup-EMA update of ``ema_params`` toward ``model_params``."""
        decay = compute_warmup_decay(
            num_averaged=num_averaged,
            max_decay=max_decay,
            inv_gamma=inv_gamma,
            power=power,
            min_decay=min_decay,
        )
        one_minus_decay = 1.0 - decay

        # torch._foreach_* gives an efficient fused update across the parameter
        # lists, matching AveragedModel's internal convention.
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, model_params, alpha=one_minus_decay)

    return warmup_ema_fn


def compute_warmup_decay(
    num_averaged: int | torch.Tensor,
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
    min_decay: float = 0.0,
) -> float:
    r"""Compute the Diffusers warmup decay :math:`\beta(s)`.

    Args:
        num_averaged (int | torch.Tensor): Number of models already averaged,
            ``s`` in the schedule.
        max_decay (float): Maximum (asymptotic) decay.
        inv_gamma (float): Inverse-gamma factor controlling warmup rate.
        power (float): Warmup power.
        min_decay (float): Minimum decay floor.

    Returns:
        float: The decay :math:`\beta(s)` clamped to
        ``[min_decay, max_decay]``.

    """
    if isinstance(num_averaged, torch.Tensor):
        step = float(num_averaged.item())
    else:
        step = float(num_averaged)

    value = 1.0 - (1.0 + step / inv_gamma) ** (-power)
    return float(min(max(value, min_decay), max_decay))


def build_ema_model(
    model: torch.nn.Module,
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
    min_decay: float = 0.0,
    device: torch.device | str | None = None,
    use_buffers: bool = False,
) -> AveragedModel:
    """Build an :class:`AveragedModel` with a warmup-EMA averaging function.

    ``use_buffers=False`` is correct for the LodeRunner backbone because it uses
    LayerNorm/RMSNorm rather than BatchNorm running statistics; the buffers are
    non-stateful.

    Args:
        model (torch.nn.Module): The trainable model to shadow. For DDP wrap the
            *underlying* module (``ddp_model.module``), not the DDP wrapper.
        max_decay (float): Maximum (asymptotic) decay.
        inv_gamma (float): Inverse-gamma factor controlling warmup rate.
        power (float): Warmup power.
        min_decay (float): Minimum decay floor.
        device (torch.device | str | None): Device for the EMA copy.
        use_buffers (bool): Whether to average module buffers as well.

    Returns:
        AveragedModel: The EMA model wrapping a copy of ``model``.

    """
    avg_fn = make_warmup_ema_fn(
        max_decay=max_decay,
        inv_gamma=inv_gamma,
        power=power,
        min_decay=min_decay,
    )
    ema_model = AveragedModel(
        model,
        device=device,
        multi_avg_fn=avg_fn,
        use_buffers=use_buffers,
    )
    return ema_model


def save_ema_checkpoint(ema_model: AveragedModel, path: str) -> None:
    """Save the EMA weights as a plain model ``state_dict``.

    The saved state dict loads cleanly into a fresh model instance of the same
    class (e.g. :class:`~yoke.models.vit.swin.bomberman.LodeRunnerViT`) via
    :func:`load_ema_into_model`.

    Args:
        ema_model (AveragedModel): The EMA model whose ``module`` weights to save.
        path (str): Destination checkpoint path.

    """
    torch.save(ema_model.module.state_dict(), path)


def load_ema_into_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load EMA weights saved by :func:`save_ema_checkpoint` into ``model``.

    Args:
        model (torch.nn.Module): A freshly-constructed model to receive the
            EMA weights.
        path (str): Path to the EMA checkpoint written by
            :func:`save_ema_checkpoint`.

    Returns:
        torch.nn.Module: The same ``model`` with EMA weights loaded.

    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def make_ema_hooks(
    model_class: type,
    max_decay: float = 0.9999,
    inv_gamma: float = 1.0,
    power: float = 2.0 / 3.0,
) -> tuple[
    Callable[[object], None],
    Callable[[object, int], dict | None],
]:
    """Build ``HarnessTrainer`` hooks that maintain a warmup-EMA shadow.

    This factory encapsulates the EMA boilerplate that was previously copied
    into the ``ch_ldrViT`` training scripts: building the EMA shadow from the
    DDP-wrapped module, restoring it (plus the persisted ``global_step``) from a
    companion checkpoint on continuation, and writing the EMA companion and
    production checkpoints at save time. The two returned callables plug
    directly into :class:`yoke.harnesses.trainer.TrainerHooks` as
    ``on_after_ddp_wrap`` and ``on_before_save``.

    The EMA companion checkpoint is written next to the main checkpoint with an
    ``_ema.pth`` suffix (via :func:`save_model_and_optimizer`, so it reconstructs
    class-aware), and a production ``_ema_weights.pth`` plain ``state_dict`` is
    written on rank 0 (via :func:`save_ema_checkpoint`). The main checkpoint
    receives ``{"global_step": ...}`` as ``extra_state`` so the warmup schedule
    stays continuous across restarts.

    The trainer threads the built ``ema_model`` and the live ``global_step`` into
    each ``epoch_fn`` call (see
    :meth:`yoke.harnesses.trainer.HarnessTrainer._epoch_call_kwargs`), so the
    epoch function performs the actual per-step EMA update and returns the
    advanced ``global_step``.

    Args:
        model_class (type): The model class of the shadowed module (e.g.
            :class:`~yoke.models.vit.swin.bomberman.LodeRunnerViT`). Used to save
            the EMA companion checkpoint class-aware.
        max_decay (float): Maximum (asymptotic) EMA decay.
        inv_gamma (float): Inverse-gamma factor controlling the warmup rate.
        power (float): Warmup power for the EMA decay schedule.

    Returns:
        tuple: ``(on_after_ddp_wrap, on_before_save)`` callables for
        :class:`TrainerHooks`.
    """
    # Imported lazily to avoid a hard import cycle at module import time
    # (checkpointing imports are cheap but kept local for symmetry).
    from yoke.utils.checkpointing import (
        load_model_and_optimizer,
        save_model_and_optimizer,
    )

    def on_after_ddp_wrap(trainer: object) -> None:
        """Build the EMA shadow and restore it from a companion checkpoint.

        Args:
            trainer (object): The live :class:`HarnessTrainer` instance.
        """
        # Build the EMA shadow from the underlying (unwrapped) module.
        ema_model = build_ema_model(
            trainer.model.module,
            max_decay=max_decay,
            inv_gamma=inv_gamma,
            power=power,
            device=trainer.device,
        )

        # Restore EMA weights + global step on continuation, if present.
        args = trainer.args
        checkpoint = getattr(args, "checkpoint", None)
        if getattr(args, "continuation", False) and checkpoint:
            ema_state_path = checkpoint.replace(".pth", "_ema.pth")
            if os.path.exists(ema_state_path):
                ema_loaded_model, _, ema_epoch, ema_ckpt = load_model_and_optimizer(
                    ema_state_path,
                    optimizer_class=torch.optim.AdamW,
                    optimizer_kwargs={
                        "lr": getattr(args, "anchor_lr", 1e-4),
                        "betas": (0.9, 0.999),
                        "eps": 1e-08,
                        "weight_decay": 0.01,
                    },
                    available_models={model_class.__name__: model_class},
                    device=trainer.device,
                    return_checkpoint=True,
                )
                # Copy reconstructed EMA weights into the shadow's inner module.
                ema_model.module.load_state_dict(ema_loaded_model.state_dict())
                trainer.global_step = int(ema_ckpt.get("global_step", 0))

                # The EMA companion must come from the same restart point as the
                # main checkpoint (``starting_epoch`` set by the model_builder).
                if ema_epoch != trainer.starting_epoch:
                    raise ValueError(
                        f"EMA checkpoint epoch ({ema_epoch}) does not match the "
                        f"main checkpoint epoch ({trainer.starting_epoch}). The "
                        f"main and EMA checkpoints appear to be out of sync; "
                        f"refusing to continue with a corrupt EMA warmup schedule."
                    )

                if trainer.global_step == 0 and trainer.starting_epoch > 0:
                    print(
                        "WARNING: EMA global_step is 0 while resuming at epoch "
                        f"{trainer.starting_epoch}; the EMA warmup schedule will "
                        "restart from scratch."
                    )

                print(
                    f"EMA state restored from {ema_state_path} "
                    f"(epoch={ema_epoch}, global_step={trainer.global_step})."
                )
            else:
                print(
                    f"No EMA companion checkpoint at {ema_state_path}; "
                    "starting EMA fresh."
                )

        # Register the shadow so the trainer threads it into each epoch call.
        trainer.ema_model = ema_model

    def on_before_save(trainer: object, epochIDX: int) -> dict | None:
        """Save the EMA companion + production checkpoints; return extra_state.

        Args:
            trainer (object): The live :class:`HarnessTrainer` instance.
            epochIDX (int): The epoch index being checkpointed.

        Returns:
            dict | None: ``{"global_step": ...}`` merged into the main
            checkpoint, or ``None`` if no EMA shadow exists.
        """
        ema_model = getattr(trainer, "ema_model", None)
        if ema_model is None:
            return None

        main_path = trainer.new_chkpt_path
        ema_state_path = main_path.replace(".pth", "_ema.pth")

        # Persist the EMA shadow class-aware. The optimizer is reused purely to
        # satisfy the signature; it is ignored on EMA restore.
        save_model_and_optimizer(
            ema_model.module,
            trainer.optimizer,
            epochIDX,
            ema_state_path,
            model_class=model_class,
            model_args=trainer.model_args,
            extra_state={"global_step": trainer.global_step},
        )

        # Production EMA weights (loads cleanly into a fresh model).
        if trainer.rank == 0:
            ema_prod_path = main_path.replace(".pth", "_ema_weights.pth")
            save_ema_checkpoint(ema_model, ema_prod_path)
            print(f"[Rank {trainer.rank}] Saved EMA checkpoints -> {ema_state_path}")

        # The main checkpoint records global_step so warmup stays continuous.
        return {"global_step": trainer.global_step}

    return on_after_ddp_wrap, on_before_save


if __name__ == "__main__":
    # Minimal smoke test / schedule illustration.
    for s in [0, 1, 2, 5, 10, 100, 10000]:
        beta = compute_warmup_decay(s)
        print(f"s={s:>6d}  beta={beta:.8f}")

    toy = torch.nn.Linear(4, 4)
    ema = build_ema_model(toy)
    opt = torch.optim.SGD(toy.parameters(), lr=0.1)
    for _ in range(5):
        loss = toy(torch.randn(8, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(toy)
    print("EMA num_averaged:", int(ema.n_averaged.item()))


class ParamEMA:
    """Name-keyed exponential moving average over a subset of parameters.

    The shadow tracks ``shadow <- decay * shadow + (1 - decay) * param`` for each
    supplied (name, parameter) pair. Only the provided parameters are shadowed, so
    passing just the trainable params keeps the frozen backbone out of the average
    (and out of the checkpoint).
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
        decay: float = 0.999,
    ) -> None:
        """Initialize the shadow from the current parameter values.

        Args:
            named_params (Iterable[tuple[str, torch.Tensor]]): (name, parameter)
                pairs to shadow, e.g. the ``requires_grad`` subset of
                ``model.named_parameters()``. Consumed once (may be a generator).
            decay (float): EMA decay in [0, 1). Higher = slower/smoother. The shadow
                is initialized to a detached clone of each parameter.
        """
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in named_params
        }

    @torch.no_grad()
    def update(self, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Update the shadow toward the current parameter values.

        Args:
            named_params (Iterable[tuple[str, torch.Tensor]]): The same (name,
                parameter) pairs supplied at construction (order-independent; matched
                by name). Names not present in the shadow are ignored; missing shadow
                entries are skipped.
        """
        d = self.decay
        for name, p in named_params:
            s = self.shadow.get(name)
            if s is None:
                continue
            # shadow <- decay * shadow + (1 - decay) * param
            s.lerp_(p.detach().to(s.device, s.dtype), 1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow as a CPU-resident name->tensor dict for checkpointing."""
        return {name: s.detach().cpu().clone() for name, s in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a previously saved shadow, copying into the existing tensors in place.

        Args:
            state_dict (dict[str, torch.Tensor]): Mapping produced by
                :meth:`state_dict`. Only keys present in the current shadow are
                loaded; the copy preserves each shadow tensor's device/dtype.
        """
        for name, s in self.shadow.items():
            saved = state_dict.get(name)
            if saved is not None:
                s.copy_(saved.to(s.device, s.dtype))

    @torch.no_grad()
    def copy_to(self, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Overlay the shadow onto live parameters (a partial, in-place assignment).

        Only parameters whose names are in the shadow are overwritten, so this is a
        PARTIAL overlay -- the frozen/backbone params of the target model are left
        untouched. Use this at eval time to swap in the averaged weights.

        Args:
            named_params (Iterable[tuple[str, torch.Tensor]]): (name, parameter) pairs
                of the target model, e.g. ``model.named_parameters()``.
        """
        for name, p in named_params:
            s = self.shadow.get(name)
            if s is not None:
                p.data.copy_(s.to(p.device, p.dtype))
