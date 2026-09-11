"""Custom schedulers for Yoke.

A module of custom schedulers to use to train Yoke models.

"""

import math
import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# matplotlib.use('QtAgg')
# Get rid of type 3 fonts in figures
import matplotlib.pyplot as plt


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)


def power_decay_lr_calc(
    step: int = 0,
    anchor_lr: float = 1.0e-3,
    decay_power: float = 0.5,
    warmup_steps: int = 1000,
    dim_embed: int = 1,
) -> float:
    """Warm-up with power decay.

    Used in `WarmUpPowerDecayScheduler`

    """
    embed_factor = 1.0 / np.power(float(dim_embed), decay_power)
    anchor = np.power(float(warmup_steps), decay_power)

    if step < warmup_steps:
        lr = float(step) / np.power(float(warmup_steps), 1.0 + decay_power)
        lr = anchor * embed_factor * lr
    else:
        lr = 1.0 / np.power(float(step), decay_power)
        lr = anchor * embed_factor * lr

    return anchor_lr * lr


def cos_decay_lr_calc(
    step: int = 0,
    anchor_lr: float = 1.0e-3,
    warmup_steps: int = 100,
    terminal_steps: int = 1000,
    min_fraction: float = 0.5,
    num_cycles: float = 0.5,
) -> float:
    """Cosine decay, linear warmup.

    Used in `WarmUpCosineDecayScheduler`

    """
    if step < warmup_steps:
        lr = float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps)
        progress = progress / float(max(1, terminal_steps - warmup_steps))

        tmp_lr = 1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        tmp_lr = 0.5 * (1.0 - min_fraction) * tmp_lr
        lr = max(0.0, tmp_lr)
        lr = min_fraction + lr

    return anchor_lr * lr


def warmup_to_constant_lr(
    step: int = 0, warmup_steps: int = 100, lr_constant: float = 1e-3
) -> float:
    """Linear warm up to constant LR.

    Used in `ConstantWithWarmupScheduler`

    """
    if step < warmup_steps:
        lr = lr_constant * float(step) / float(max(1.0, warmup_steps))
    else:
        lr = lr_constant

    return lr


class WarmUpPowerDecayScheduler(_LRScheduler):
    """Scheduler with warm-up and power decay.

    Scheduler mentioned in *https://kikaben.com/transformers-training-details*

    This goes through a linear warm-up phase for the first `warmup_steps` in
    which the learning rate goes from 0 to `anchor_lr`. Then the learning rate
    decays as the inverse `decay_power` of the number of steps. The scheduler's
    learning rate is scaled by `dim_embed` to the negative `decay_power`.

    Args:
        optimizer (torch.nn.Optimizer): Optimizer which scheduler modifies
        anchor_lr (float): LR that scheduler warms up to
        decay_power (float): Power that LR decays with after warm up
        warmup_steps (int): Number of steps for linear warm up
        dim_embed (int): Dimension of transformer embedding used to scale LR
        last_epoch (int): Last step if restarting

    """

    def __init__(
        self,
        optimizer: Optimizer,
        anchor_lr: float = 1.0e-3,
        decay_power: float = 0.5,
        warmup_steps: int = 1000,
        dim_embed: int = 1,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler."""
        self.anchor_lr = anchor_lr
        self.decay_power = decay_power
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        """Return learning rate."""
        lr = power_decay_lr_calc(
            step=self._step_count,
            anchor_lr=self.anchor_lr,
            decay_power=self.decay_power,
            dim_embed=self.dim_embed,
            warmup_steps=self.warmup_steps,
        )
        return [lr] * self.num_param_groups


class CosineWithWarmupScheduler(_LRScheduler):
    """Cosine decay after warmup.

    From:
    https://github.com/krasserm/perceiver-io/blob/main/perceiver/scripts/lrs.py

    Linear warm-up to `anchor_lr` then goes through `num_cycles` of a scaled
    cosine varying between [min_fraction*anchor_lr, anchor_lr] between
    `warmup_steps` and `terminal_steps`.

    Args:
        optimizer (torch.nn.Optimizer): Optimizer which scheduler modifies
        anchor_lr (float): LR that scheduler warms up to
        terminal_steps (int): Number of total steps scheduler uses to determine
                              cosine cycles
        num_cycles (float): Number of periods of cosine between warm up and termination
        min_fraction (float): Fraction of anchor LR at cosine trough
        warmup_steps (int): Number of steps for linear warm up
        last_epoch (int): Last step if restarting
        lr_mults (list[float] | None): Per-parameter-group multipliers applied to
            the scheduled LR. Length must equal the number of optimizer param
            groups. ``None`` (default) applies the same scheduled LR to every
            group -- byte-identical to the legacy single-group behavior. Use this
            for discriminative fine-tuning (e.g. a smaller LR on an unfrozen
            backbone tail than on the freshly-trained head).

    """

    def __init__(
        self,
        optimizer: Optimizer,
        anchor_lr: float = 1.0e-3,
        terminal_steps: int = 1000,
        warmup_steps: int = 100,
        num_cycles: float = 0.5,
        min_fraction: float = 0.5,
        last_epoch: int = -1,
        lr_mults: list = None,
    ) -> None:
        """Initialize scheduler."""
        self.anchor_lr = anchor_lr
        self.warmup_steps = warmup_steps
        self.num_cycles = num_cycles
        self.min_fraction = min_fraction
        self.terminal_steps = terminal_steps

        self.num_param_groups = len(optimizer.param_groups)

        if lr_mults is None:
            self.lr_mults = [1.0] * self.num_param_groups
        else:
            if len(lr_mults) != self.num_param_groups:
                raise ValueError(
                    f"lr_mults has {len(lr_mults)} entries but the optimizer has "
                    f"{self.num_param_groups} param groups."
                )
            self.lr_mults = list(lr_mults)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        """Return learning rate."""
        lr = cos_decay_lr_calc(
            step=self._step_count,
            anchor_lr=self.anchor_lr,
            warmup_steps=self.warmup_steps,
            terminal_steps=self.terminal_steps,
            min_fraction=self.min_fraction,
            num_cycles=self.num_cycles,
        )
        return [lr * m for m in self.lr_mults]


class ConstantWithWarmupScheduler(_LRScheduler):
    """Constant LR after warmup.

    From:
    https://github.com/krasserm/perceiver-io/blob/main/perceiver/scripts/lrs.py

    Args:
        optimizer (torch.nn.Optimizer): Optimizer which scheduler modifies
        lr_constant (float): LR that scheduler warms up to
        warmup_steps (int): Number of steps for linear warm up
        last_epoch (int): Last step if restarting

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 100,
        lr_constant: float = 1e-3,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler."""
        self.warmup_steps = warmup_steps
        self.lr_constant = lr_constant
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        """Return learning rate."""
        lr = warmup_to_constant_lr(
            step=self._step_count,
            warmup_steps=self.warmup_steps,
            lr_constant=self.lr_constant,
        )

        return [lr] * self.num_param_groups


if __name__ == "__main__":
    """Plotting for schedulers."""

    # Total steps
    steps = np.arange(0, 2000 + 1)
    warmup_steps = 500
    anchor_lr = 1.0e-3

    # Power decay
    decay_power = 0.6
    dim_embed = 1
    power_decay = [
        power_decay_lr_calc(
            step=step,
            anchor_lr=anchor_lr,
            decay_power=decay_power,
            dim_embed=dim_embed,
            warmup_steps=warmup_steps,
        )
        for step in steps
    ]

    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(steps, power_decay, label="Power-decay LR")
    plt.xlabel("Current Step")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid()

    # Cosine decay
    terminal_steps = 2000
    min_fraction = 0.7
    num_cycles = 0.5

    cos_decay = [
        cos_decay_lr_calc(
            step=step,
            anchor_lr=anchor_lr,
            warmup_steps=warmup_steps,
            terminal_steps=terminal_steps,
            min_fraction=min_fraction,
            num_cycles=num_cycles,
        )
        for step in steps
    ]

    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(steps, cos_decay, label="Cosine-decay LR")
    plt.xlabel("Current Step")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid()

    # Constant with warm-up
    warmup_constant = [
        warmup_to_constant_lr(
            step=step, warmup_steps=warmup_steps, lr_constant=anchor_lr
        )
        for step in steps
    ]

    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(steps, warmup_constant, label="Warm-up to Constant LR")
    plt.xlabel("Current Step")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid()

    if False:
        fig1.savefig(
            "./power_decay_scheduler.png",
            bbox_inches="tight",
        )
        fig2.savefig(
            "./cosine_decay_scheduler.png",
            bbox_inches="tight",
        )
        fig3.savefig(
            "./constant_warmup_scheduler.png",
            bbox_inches="tight",
        )
    else:
        plt.show()
