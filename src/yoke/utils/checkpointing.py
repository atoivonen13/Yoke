"""Functions for checkpointing models and optimizers in PyTorch.

The functions provided here offer checkpointing capabilities that have behavior specific
to the Yoke framework. They are designed to work seamlessly with the Yoke training and
evaluation processes, ensuring that model states can be saved and restored effectively.
"""

import os
import glob
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import h5py

from yoke.models.vit.swin.bomberman import (
    LodeRunner,
    ScalarTemporalConditionedLodeRunner_gri,
    ScalarTemporalConditionedLodeRunner_9band,
)


def backbone_tail_modules(model: torch.nn.Module) -> list:
    """Output-proximal decoder tail of the wrapped Swin U-Net backbone.

    These are the last parameter-bearing modules the backbone runs before its
    output feeds the wrapper's meanstdmax pool: the final PatchExpand, the final
    up_connect (SwinConnectDecoder), and ``backbone.linear4unpatch``. With the
    default ``block_structure=(1,1,3,1)`` the decoder's up_stage2 / up_stage3
    SwinEncoder loops are empty, so these three modules hold the trainable
    capacity closest to the loss. Single source of truth so the fresh-study
    optimizer build and the continuation loader unfreeze exactly the same params.

    Args:
        model (torch.nn.Module): A ScalarTemporalConditionedLodeRunner_9band whose
            ``.backbone`` is a LodeRunner (has ``.unet`` and ``.linear4unpatch``).

    Returns:
        list: The submodules constituting the tail, output-proximal last.
    """
    unet = model.backbone.unet
    return [
        unet.PatchExpand[-1],
        unet.up_connect[-1],
        model.backbone.linear4unpatch,
    ]


def backbone_decoder_modules(model: torch.nn.Module) -> list:
    """Full decoder + bottleneck of the wrapped Swin U-Net backbone.

    A superset of :func:`backbone_tail_modules`: every parameter-bearing module on
    the backbone's UP path plus the bottleneck -- ``bottleneck_stage4``, the three
    UP SwinEncoder stages (``up_stage1/2/3``), every skip-connection receptor
    (``up_connect``), every ``PatchExpand``, and ``backbone.linear4unpatch``. Under
    the default ``block_structure=(1,1,3,1)`` the ``up_stage2`` / ``up_stage3``
    loops are empty, so this reduces to bottleneck(1) + up_stage1(2) + up_connect(3)
    + PatchExpand(3) + linear4unpatch. The lists are iterated (not indexed ``[-1]``)
    so the selection is correct for any ``block_structure``.

    The ENCODER stays frozen -- ``parallel_embed``, ``var_embed_layer``,
    ``agg_vars``, ``pos_embed``, ``temporal_encoding``, ``dwn_stage1/2/3``,
    ``down_connect``, and ``PatchMerge`` are the shared feature extractor reused,
    unchanged, across downstream multiphysics applications. This is the "decoder"
    fine-tune scope: freeze the shared encoder, adapt the per-task decoder.

    Args:
        model (torch.nn.Module): A ScalarTemporalConditionedLodeRunner_9band whose
            ``.backbone`` is a LodeRunner (has ``.unet`` and ``.linear4unpatch``).

    Returns:
        list: The submodules constituting the bottleneck + decoder.
    """
    unet = model.backbone.unet
    return [
        *unet.bottleneck_stage4,
        *unet.up_stage1,
        *unet.up_stage2,
        *unet.up_stage3,
        *unet.up_connect,
        *unet.PatchExpand,
        model.backbone.linear4unpatch,
    ]


def build_finetune_optimizer(
    model: torch.nn.Module,
    optimizer_kwargs: dict,
    backbone_tail_lr_mult: float = 0.0,
    backbone_finetune_scope: str = "tail",
    verbose: bool = False,
) -> tuple:
    """Freeze the backbone, set trainable params, and build the AdamW optimizer.

    Always trains ``conditioner`` + ``output_head``. When
    ``backbone_tail_lr_mult > 0`` it additionally unfreezes a subset of the
    pretrained backbone -- selected by ``backbone_finetune_scope`` -- and puts it in
    a SECOND optimizer param group at ``base_lr * backbone_tail_lr_mult``
    (discriminative fine-tuning). When ``0`` (default) the whole backbone stays
    frozen and a single-group optimizer is built, byte-identical to the legacy
    frozen regime.

    ``backbone_finetune_scope`` selects which modules the second group unfreezes:
      - ``"tail"`` (default): the thin output-proximal tail
        (:func:`backbone_tail_modules`) -- study 082's set, kept as the default so
        that study stays reproducible.
      - ``"decoder"``: the whole bottleneck + decoder
        (:func:`backbone_decoder_modules`), leaving the encoder frozen as a shared
        feature extractor -- study 084.

    The per-group LR ratio is returned separately as ``lr_mults`` because
    ``CosineWithWarmupScheduler`` overwrites each group's ``lr`` every step with
    one scheduled value; the scheduler must be given these mults to preserve the
    ratio (per-group ``lr`` set on the optimizer alone would be ignored).

    Args:
        model (torch.nn.Module): The wrapper model (pre-DDP).
        optimizer_kwargs (dict): AdamW kwargs; ``optimizer_kwargs["lr"]`` is the
            head (base) learning rate.
        backbone_tail_lr_mult (float): Fraction of the head LR for the unfrozen
            backbone modules. 0 keeps the backbone frozen (no second group).
        backbone_finetune_scope (str): Which backbone modules the second group
            unfreezes when ``backbone_tail_lr_mult > 0`` -- ``"tail"`` (default,
            study 082) or ``"decoder"`` (study 084). Ignored when the mult is 0.
        verbose (bool): If True, print trainable-parameter counts (rank-0 only).

    Returns:
        tuple: ``(optimizer, lr_mults)`` where ``lr_mults`` is a list with one
        entry per optimizer param group (``[1.0]`` frozen, ``[1.0, mult]`` when
        the tail is unfrozen), to be passed to the LR scheduler.
    """
    for p in model.backbone.parameters():
        p.requires_grad = False

    # The trainable "head" set depends on the wrapper's forward path. The spatial-
    # render model (Study 086) drops the conditioner/output_head for a single
    # read_head; the legacy model trains conditioner + output_head. Selecting the
    # modules that actually run keeps DDP's find_unused_parameters=False valid.
    if getattr(model, "spatial_render", False):
        trainable_mods = [model.read_head]
    else:
        trainable_mods = [model.conditioner, model.output_head]

    head_params = []
    for mod in trainable_mods:
        for p in mod.parameters():
            p.requires_grad = True
            head_params.append(p)

    if backbone_tail_lr_mult and backbone_tail_lr_mult > 0.0:
        # The backbone must actually run for its tail to receive gradients. Under
        # bypass the forward skips the backbone entirely, so unfreezing the tail
        # would leave its params unused -- silently wasted, and a hard DDP error
        # (find_unused_parameters=False). Fail loudly instead.
        if getattr(model, "bypass_backbone", False):
            raise ValueError(
                "backbone_tail_lr_mult > 0 requires bypass_backbone=False; the "
                "bypass path never runs the backbone, so unfreezing its tail has "
                "no effect and breaks DDP (unused parameters)."
            )
        if backbone_finetune_scope == "tail":
            finetune_mods = backbone_tail_modules(model)
        elif backbone_finetune_scope == "decoder":
            finetune_mods = backbone_decoder_modules(model)
        else:
            raise ValueError(
                "backbone_finetune_scope must be 'tail' or 'decoder', got "
                f"{backbone_finetune_scope!r}."
            )
        tail_params = []
        for mod in finetune_mods:
            for p in mod.parameters():
                p.requires_grad = True
                tail_params.append(p)

        base_lr = optimizer_kwargs["lr"]
        param_groups = [
            {"params": head_params},
            {"params": tail_params, "lr": base_lr * backbone_tail_lr_mult},
        ]
        lr_mults = [1.0, backbone_tail_lr_mult]
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)

        if verbose:
            n_head = sum(p.numel() for p in head_params)
            n_tail = sum(p.numel() for p in tail_params)
            print(
                f"[finetune] head params: {n_head:,}; unfrozen backbone "
                f"({backbone_finetune_scope}) params: {n_tail:,} "
                f"(LR mult {backbone_tail_lr_mult})"
            )
    else:
        optimizer = torch.optim.AdamW(head_params, **optimizer_kwargs)
        lr_mults = [1.0]

        if verbose:
            n_head = sum(p.numel() for p in head_params)
            print(f"[finetune] head params: {n_head:,}; backbone frozen")

    return optimizer, lr_mults


def _epoch_median_val_losses(
    val_rcrd_glob: str,
) -> dict:
    """Median validation loss per epoch, read from the per-epoch record CSVs.

    Each epoch's validation record is a CSV of ``epoch, batch, loss`` rows (see
    the ``train_DDP_scalar_temporal_loderunner_epoch_9band*`` writers). This globs
    every matching file, reads the loss column, and reduces to the MEDIAN per
    epoch -- the same statistic the loss-curve plot shows, and robust to the heavy
    per-batch tail that would make the mean noisy. Files that are empty or
    unreadable are skipped (a half-written CSV from a crashed epoch does not
    poison the ranking).

    Args:
        val_rcrd_glob (str): Glob matching the validation record CSVs, e.g.
            ``./validation_study083_epoch*.csv``.

    Returns:
        dict: ``{epoch_int: median_loss_float}`` for every epoch with a readable,
        non-empty record. Empty dict if nothing matches.
    """
    out = {}
    for path in glob.glob(val_rcrd_glob):
        # Skip empty files up front: np.loadtxt emits a UserWarning on a
        # zero-byte record (e.g. an epoch skipped by train_per_val, or a crashed
        # half-written file), which would otherwise spam the training log.
        try:
            if os.path.getsize(path) == 0:
                continue
        except OSError:
            continue
        try:
            arr = np.loadtxt(path, delimiter=",", ndmin=2)
        except (ValueError, OSError):
            continue
        if arr.size == 0:
            continue
        # Columns: epoch, batch, loss. A single epoch per file, but derive the
        # epoch from the data (not the filename) so it is authoritative.
        epochs = arr[:, 0].astype(int)
        losses = arr[:, 2]
        for ep in np.unique(epochs):
            out[int(ep)] = float(np.median(losses[epochs == ep]))
    return out


def update_best_checkpoint(
    studyIDX: int,
    val_rcrd_glob: str,
    ckpt_dir: str = "./",
) -> tuple:
    """Copy the lowest-median-val-loss epoch's checkpoint to a stable ``_best`` file.

    Designed for the ``cycle_epochs=1`` restart pattern where every epoch runs as a
    fresh process: this is STATELESS. It rebuilds the full per-epoch val-loss
    history from the record CSVs on disk each call, so no "best-so-far" value needs
    to survive the process restart. Call it on rank 0 after each epoch's checkpoint
    has been written.

    The per-epoch ``study{IDX}_modelState_epoch{NNNN}.pth`` files are left
    untouched (the continuation chain still resumes from the latest one); this only
    maintains a COPY at ``study{IDX}_modelState_best.pth`` pointing at the best
    epoch seen so far. Selection is by validation loss (the trained objective),
    which is a proxy for the late-time RMSE the studies are ranked on -- use it to
    pick a small candidate set for the dense eval, not as the final word.

    Args:
        studyIDX (int): Study index, for the checkpoint / record filename pattern.
        val_rcrd_glob (str): Glob for the validation record CSVs (e.g.
            ``./validation_study083_epoch*.csv``).
        ckpt_dir (str): Directory holding the per-epoch checkpoints.

    Returns:
        tuple: ``(best_epoch, best_loss, best_path)`` on success, or
        ``(None, None, None)`` if no val records or no matching checkpoint exist
        yet (e.g. before the first validation epoch).
    """
    med = _epoch_median_val_losses(val_rcrd_glob)
    if not med:
        return None, None, None

    # Only consider epochs whose checkpoint actually exists on disk.
    best_epoch, best_loss, best_src = None, None, None
    for ep in sorted(med):
        src = os.path.join(
            ckpt_dir, f"study{studyIDX:03d}_modelState_epoch{ep:04d}.pth"
        )
        if not os.path.exists(src):
            continue
        if best_loss is None or med[ep] < best_loss:
            best_epoch, best_loss, best_src = ep, med[ep], src

    if best_src is None:
        return None, None, None

    best_dst = os.path.join(ckpt_dir, f"study{studyIDX:03d}_modelState_best.pth")
    # Copy (not symlink/rename) so the per-epoch file stays intact for the
    # continuation chain and _best is self-contained. Atomic-ish: write to a temp
    # name then replace, so a crash mid-copy never leaves a truncated _best.
    tmp_dst = best_dst + ".tmp"
    shutil.copyfile(best_src, tmp_dst)
    os.replace(tmp_dst, best_dst)
    return best_epoch, best_loss, best_dst


def save_model_and_optimizer_hdf5(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    compiled: bool = False,
) -> None:
    """Saves the state of a model and optimizer in portable hdf5 format.

    Model and optimizer should be moved to the CPU prior to using this function.

    Args:
        model (torch.nn.Module): Pytorch model to save
        optimizer (torch.optim.Optimizer): Pytorch optimizer to save
        epoch (int): Epoch associated with training
        filepath (str): Where to save
        compiled (bool): Flag to extract original model if model being saved
                         was compiled.

    """
    # If model is wrapped in DataParallel, access the underlying module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # If the model is a `torch.compiled` version the original model must be
    # extracted first.
    if compiled:
        model = model._orig_mod

    with h5py.File(filepath, "w") as h5f:
        # Save epoch number
        h5f.attrs["epoch"] = epoch

        # Save model parameters and buffers
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy()
            if data.ndim == 0:  # It's a scalar!
                h5f.attrs["model/parameters/" + name] = data
            else:
                h5f.create_dataset("model/parameters/" + name, data=data)

        for name, buffer in model.named_buffers():
            data = buffer.cpu().numpy()
            if data.ndim == 0:  # It's a scalar!
                h5f.attrs["model/buffers/" + name] = data
            else:
                h5f.create_dataset("model/buffers/" + name, data=data)

        # Save optimizer state
        optimizer_state = optimizer.state_dict()
        for idx, group in enumerate(optimizer_state["param_groups"]):
            group_name = f"optimizer/group{idx}"
            for k, v in group.items():
                # print('group_name:', group_name, k)
                if isinstance(v, (int, float)):
                    h5f.attrs[group_name + "/" + k] = v
                elif isinstance(v, list):
                    h5f.create_dataset(group_name + "/" + k, data=v)

        # Save state values, like momentums
        for idx, state in enumerate(optimizer_state["state"].items()):
            state_name = f"optimizer/state{idx}"
            for k, v in state[1].items():
                # print('state_name:', state_name, k)
                if isinstance(v, torch.Tensor):
                    h5f.create_dataset(
                        state_name + "/" + k, data=v.detach().cpu().numpy()
                    )


def load_model_and_optimizer_hdf5(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> int:
    """Loads state of model and optimizer stored in an hdf5 format.

    Args:
        model (torch.nn.Module): Pytorch model to load state into.
        optimizer (torch.optim.Optimizer): Pytorch optimizer to load state into.
        filepath (str): Path to the hdf5 checkpoint file.

    Returns:
        epoch (int): Epoch associated with training

    """
    # If model is wrapped in DataParallel, access the underlying module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    with h5py.File(filepath, "r") as h5f:
        # Get epoch number
        epoch = h5f.attrs["epoch"]

        # Load model parameters and buffers
        for name in h5f.get("model/parameters", []):  # Get the group
            if isinstance(h5f["model/parameters/" + name], h5py.Dataset):
                data = torch.from_numpy(h5f["model/parameters/" + name][:])
            else:
                data = torch.tensor(h5f.attrs["model/parameters/" + name])

            name_list = name.split(".")
            param_name = name_list.pop()
            submod_name = ".".join(name_list)

            model.get_submodule(submod_name)._parameters[param_name].data.copy_(data)

        for name in h5f.get("model/buffers", []):
            if isinstance(h5f["model/buffers/" + name], h5py.Dataset):
                buffer = torch.from_numpy(h5f["model/buffers/" + name][:])
            else:
                buffer = torch.tensor(h5f.attrs["model/buffers/" + name])

            name_list = name.split(".")
            param_name = name_list.pop()
            submod_name = ".".join(name_list)
            model.get_submodule(submod_name)._buffers[param_name].data.copy_(buffer)

        # Rebuild optimizer state (need to call this before loading state)
        optimizer_state = optimizer.state_dict()

        # Load optimizer parameter groups
        for k in h5f.attrs:
            if "optimizer/group" in k:
                # print('k-string:', k)
                idx, param = k.split("/")[1:]
                optimizer_state["param_groups"][int(idx.lstrip("group"))][param] = (
                    h5f.attrs[k]
                )

        # Load state values, like momentums
        for name, group in h5f.items():
            if "optimizer/state" in name:
                state_idx = int(name.split("state")[1])
                param_idx, param_state = list(optimizer_state["state"].items())[
                    state_idx
                ]
                for k in group:
                    optimizer_state["state"][param_idx][k] = torch.from_numpy(
                        group[k][:]
                    )

        # Load optimizer state
        optimizer.load_state_dict(optimizer_state)

    return epoch


def save_model_and_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    model_class: type,
    model_args: dict,
) -> None:
    """Class-aware torch checkpointing.

    Saves model & optimizer state along with model-class information using torch.save.
    Works for both DDP and non-DDP training. Model's saved in this way should not be
    considered *deployable*. For deployment the model should be converted to ONNX format.

    - Stores the model's class name and initialization args.
    - Works for both DDP and non-DDP training.
    - If model is wrapped in DDP (`model.module` exists), saves
      `model.module.state_dict()`.
    - If model is NOT using DDP, saves `model.state_dict()`.
    - Moves model and optimizer to CPU to avoid CUDA-specific issues.
    - Saves only on rank 0 when using DDP to prevent redundant writes.
    - If using DDP, synchronizes all processes after saving to ensure consistency.

    Args:
        model (torch.nn.Module): Torch nn.Module instance or DDP version thereof.
        optimizer (torch.optim): Torch optimizer instance
        epoch (int): Epoch index being checkpointed.
        filepath (str): Checkpoint filename.
        model_class (torch.nn.Module class): Class of model being checkpointed.
        model_args (dict): Dictionary of model parameters.
    """
    is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)

    # Get rank if in DDP, else assume single process
    if dist.is_initialized():
        save_rank = dist.get_rank()
    else:
        save_rank = 0

    # Save only on rank 0 in DDP or always in single-GPU mode
    if save_rank == 0:
        if is_ddp:
            model_cpu = model.module.to("cpu")
        else:
            model_cpu = model.to("cpu")

        optimizer_cpu = optimizer.state_dict()
        for state in optimizer_cpu["state"].values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu")

        checkpoint = {
            "epoch": epoch,
            "model_class": model_class.__name__,  # Store model class as a string
            "model_args": model_args,  # Store model init arguments
            "model_state_dict": model_cpu.state_dict(),
            "optimizer_state_dict": optimizer_cpu,
        }

        torch.save(checkpoint, filepath)
        print(f"[Rank {save_rank}] Saved checkpoint at epoch {epoch} -> {filepath}")

    # Ensure all processes synchronize before moving on (only if using DDP)
    if dist.is_initialized():
        dist.barrier()


def load_model_and_optimizer(
    filepath: str,
    optimizer_class: type,
    optimizer_kwargs: dict,
    available_models: dict,
    device: str = "cuda",
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Dynamically load model & optimizer state from checkpoint.

    NOTE: This function only works while loading checkpoints created by
    `save_model_and_optimizer`

    - Working for both DDP and non-DDP training.
    - Loads the checkpoint only on rank 0 when in DDP.
    - If using DDP, broadcasts the checkpoint to all other ranks.
    - Handles models both inside and outside of `DistributedDataParallel`.

    Args:
        filepath (str): Checkpoint filename.
        optimizer_class (type): Torch optimizer class
        optimizer_kwargs (dict): Dictionary of optimizer parameters.
        available_models (dict): Dictionary mapping class names to class references.
        device (torch.device): String or device specifier.

    """
    # Get rank if in DDP, else assume single process
    if dist.is_initialized():
        load_rank = dist.get_rank()
    else:
        load_rank = 0

    checkpoint = None

    if load_rank == 0:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        epochIDX = checkpoint["epoch"]
        print(f"[Rank {load_rank}] Loaded checkpoint from epoch {epochIDX}")

    # If in DDP, broadcast checkpoint to all ranks
    if dist.is_initialized():
        checkpoint_list = [checkpoint]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]  # Unpack checkpoint on all ranks

    # Retrieve model class and arguments
    model_class_name = checkpoint["model_class"]
    model_args = checkpoint["model_args"]

    # Ensure model class exists
    if model_class_name not in available_models:
        raise ValueError(
            f"Unknown model class: {model_class_name}. Add it to `available_models`."
        )

    # Dynamically create the model
    model = available_models[model_class_name](**model_args)

    # Load state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move model to GPU if necessary
    model.to(device)

    # Initialize optimizer and move to device
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move optimizer state to GPU if necessary
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    # Synchronize all processes in DDP
    if dist.is_initialized():
        dist.barrier()

    return model, optimizer, checkpoint["epoch"]


def load_direct_loderunner_checkpoint(
    checkpoint_path: str,
    model_args: dict,
    optimizer_kwargs: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Load a ScalarTemporalConditionedLodeRunner_gri model from a checkpoint.

    Handles two checkpoint types:
      - An old plain LodeRunner checkpoint, whose weights are loaded into the
        wrapper's backbone (conditioner/output-head are freshly initialized and
        this is not treated as a true continuation).
      - A ScalarTemporalConditionedLodeRunner_gri wrapper checkpoint, which is
        loaded in full and treated as a continuation.

    The backbone is frozen and only the conditioner and output-head parameters
    are trainable.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model_args (dict): Fallback LodeRunner init args if the checkpoint has
            none stored.
        optimizer_kwargs (dict): Kwargs for the AdamW optimizer.
        device (torch.device): Device to load the model/optimizer onto.

    Returns:
        model (torch.nn.Module): The wrapper model.
        optimizer (torch.optim.Optimizer): Optimizer over trainable parameters.
        starting_epoch (int): Epoch to continue training from.
    """
    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    saved_model_args = checkpoint_data.get("model_args", model_args)
    context_len = checkpoint_data.get("context_len", 5)

    backbone = LodeRunner(**saved_model_args).to(device)

    model = ScalarTemporalConditionedLodeRunner_gri(
        backbone=backbone,
        context_len=context_len,
        n_input_channels=checkpoint_data.get("n_input_channels", 3),
        n_output_channels=checkpoint_data.get("n_output_channels", 3),
        image_size=saved_model_args["image_size"],
        backbone_channels=checkpoint_data.get("backbone_channels", 8),
        hidden=checkpoint_data.get("hidden", 64),
    ).to(device)

    state_dict = checkpoint_data["model_state_dict"]

    # Remove DDP prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    # Detect checkpoint type
    is_wrapper_checkpoint = any(
        k.startswith("backbone.") for k in state_dict.keys()
    )

    # -------------------------------------------------
    # OLD plain LodeRunner checkpoint
    # -------------------------------------------------
    if not is_wrapper_checkpoint:
        missing_keys, unexpected_keys = model.backbone.load_state_dict(
            state_dict,
            strict=False,
        )

        print("Loaded old LodeRunner checkpoint into model.backbone")
        print("Missing backbone keys:", missing_keys)
        print("Unexpected backbone keys:", unexpected_keys)

        # This is NOT a true continuation.
        # Conditioner is newly initialized.
        starting_epoch = 0

    # -------------------------------------------------
    # NEW ScalarTemporalConditionedLodeRunner checkpoint
    # -------------------------------------------------
    else:
        model.load_state_dict(state_dict, strict=True)

        print("Loaded ScalarTemporalConditionedLodeRunner checkpoint")

        starting_epoch = checkpoint_data.get("epoch", 0)

    noise_scale = checkpoint_data.get("noise_scale", 0.0)
    model.backbone.noise_scale = noise_scale

    # Freeze pretrained backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Train conditioner
    for p in model.conditioner.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        list(model.conditioner.parameters())
        + list(model.output_head.parameters()),
        **optimizer_kwargs,
    )

    # Only restore optimizer for TRUE continuation checkpoints
    if (
        is_wrapper_checkpoint
        and "optimizer_state_dict" in checkpoint_data
    ):
        optimizer.load_state_dict(
            checkpoint_data["optimizer_state_dict"]
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    return model, optimizer, starting_epoch


def load_direct_loderunner_checkpoint_9band(
    checkpoint_path: str,
    model_args: dict,
    optimizer_kwargs: dict,
    device: torch.device,
    backbone_tail_lr_mult: float = 0.0,
    backbone_finetune_scope: str = "tail",
) -> tuple:
    """Load a ScalarTemporalConditionedLodeRunner_9band model from a checkpoint.

    The 9-band analogue of ``load_direct_loderunner_checkpoint``. Handles two
    checkpoint types:
      - An old plain LodeRunner checkpoint, whose weights are loaded into the
        wrapper's backbone (conditioner/output-head are freshly initialized and
        this is not treated as a true continuation).
      - A ScalarTemporalConditionedLodeRunner_9band wrapper checkpoint, which is
        loaded in full and treated as a continuation.

    The backbone is frozen and only the conditioner and output-head parameters
    are trainable, UNLESS ``backbone_tail_lr_mult > 0``, in which case the decoder
    tail is also unfrozen at a reduced LR (see :func:`build_finetune_optimizer`).
    This MUST match the value used to create the checkpoint, or the rebuilt
    optimizer's param-group structure will not match the saved optimizer state.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model_args (dict): Fallback LodeRunner init args if the checkpoint has
            none stored.
        optimizer_kwargs (dict): Kwargs for the AdamW optimizer.
        device (torch.device): Device to load the model/optimizer onto.
        backbone_tail_lr_mult (float): Fraction of the head LR for the unfrozen
            backbone modules. 0 (default) keeps the backbone fully frozen -- the
            legacy behavior.
        backbone_finetune_scope (str): Which backbone modules to unfreeze when
            ``backbone_tail_lr_mult > 0`` -- ``"tail"`` (default) or ``"decoder"``.
            MUST match the value used to create the checkpoint, or the rebuilt
            optimizer's param-group structure will not match the saved state.

    Returns:
        model (torch.nn.Module): The wrapper model.
        optimizer (torch.optim.Optimizer): Optimizer over trainable parameters.
        starting_epoch (int): Epoch to continue training from.
        lr_mults (list): Per-param-group LR multipliers for the LR scheduler.
    """
    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    saved_model_args = checkpoint_data.get("model_args", model_args)
    context_len = checkpoint_data.get("context_len", 5)

    # Time-window context mode: the model's first layer is sized by the padded
    # width (max_context_len) and each event carries an extra validity flag.
    # Falls through to None for legacy fixed-count checkpoints, preserving the
    # original sizing.
    context_window_days = checkpoint_data.get("context_window_days", None)
    if context_window_days is not None:
        # In window mode the padded context width drives input_dim.
        context_len = checkpoint_data.get("max_context_len", context_len)

    backbone = LodeRunner(**saved_model_args).to(device)

    model = ScalarTemporalConditionedLodeRunner_9band(
        backbone=backbone,
        context_len=context_len,
        n_bands=checkpoint_data.get("n_bands", 9),
        image_size=saved_model_args["image_size"],
        backbone_channels=checkpoint_data.get("backbone_channels", 8),
        hidden=checkpoint_data.get("hidden", 64),
        context_window_days=context_window_days,
        # 0 for legacy checkpoints (no key) -> Fourier Dt disabled -> the
        # architecture matches the saved weights and strict load succeeds.
        dt_fourier_bands=checkpoint_data.get("dt_fourier_bands", 0),
        # False for legacy checkpoints (no key) -> absolute head. Adds no params,
        # so this only changes forward() behavior, never the state_dict.
        predict_delta=checkpoint_data.get("predict_delta", False),
        # False/3 for legacy checkpoints (no key) -> flat-hold anchor. Derived
        # from x/Dt, so this only changes forward() behavior, never the
        # state_dict, and strict load stays valid.
        trend_decay_anchor=checkpoint_data.get("trend_decay_anchor", False),
        trend_slope_k=checkpoint_data.get("trend_slope_k", 3),
        trend_max_offset=checkpoint_data.get("trend_max_offset", None),
        # "mean" for legacy checkpoints (no key) -> global average pool, matching
        # the saved output_head first-layer shape so strict load succeeds.
        pool_mode=checkpoint_data.get("pool_mode", "mean"),
        # 1 for legacy checkpoints (no key) -> point head, matching the saved
        # output_head last-layer width (n_bands) so strict load succeeds. > 1
        # reconstructs the wider quantile head.
        n_quantiles=checkpoint_data.get("n_quantiles", 1),
        bypass_backbone=checkpoint_data.get("bypass_backbone", False),
        # None for legacy checkpoints (no key) -> waist == backbone_channels.
        # When set (bypass-only), it widens the trainable waist, changing the
        # conditioner last-layer + output_head first-layer shapes, so it MUST
        # match the training config for the strict load to succeed.
        bypass_channels=checkpoint_data.get("bypass_channels", None),
        # 0 for legacy checkpoints (no key) -> no phase input. > 0 widens the
        # conditioner first-layer (input_dim + dt_extra + phase_extra), so it
        # MUST match the training config for the strict load to succeed.
        phase_fourier_bands=checkpoint_data.get("phase_fourier_bands", 0),
        # False for legacy checkpoints (no key) -> the tile+global-pool path with
        # a conditioner/output_head. True (Study 086) drops those for a render +
        # gather + read_head, changing the trainable-module set, so it MUST match
        # the training config for the strict load to succeed.
        spatial_render=checkpoint_data.get("spatial_render", False),
        render_context_days=checkpoint_data.get("render_context_days", None),
        render_horizon_days=checkpoint_data.get("render_horizon_days", 8.0),
        render_splat=checkpoint_data.get("render_splat", 5),
        gather_rows_k=checkpoint_data.get("gather_rows_k", 5),
    ).to(device)

    state_dict = checkpoint_data["model_state_dict"]

    # Remove DDP prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    # Detect checkpoint type
    is_wrapper_checkpoint = any(
        k.startswith("backbone.") for k in state_dict.keys()
    )

    # -------------------------------------------------
    # OLD plain LodeRunner checkpoint
    # -------------------------------------------------
    if not is_wrapper_checkpoint:
        missing_keys, unexpected_keys = model.backbone.load_state_dict(
            state_dict,
            strict=False,
        )

        print("Loaded old LodeRunner checkpoint into model.backbone")
        print("Missing backbone keys:", missing_keys)
        print("Unexpected backbone keys:", unexpected_keys)

        # This is NOT a true continuation.
        # Conditioner is newly initialized.
        starting_epoch = 0

    # -------------------------------------------------
    # NEW ScalarTemporalConditionedLodeRunner_9band checkpoint
    # -------------------------------------------------
    else:
        model.load_state_dict(state_dict, strict=True)

        print("Loaded ScalarTemporalConditionedLodeRunner_9band checkpoint")

        starting_epoch = checkpoint_data.get("epoch", 0)

    noise_scale = checkpoint_data.get("noise_scale", 0.0)
    model.backbone.noise_scale = noise_scale

    # Freeze the backbone (optionally unfreezing its decoder tail) and build the
    # optimizer with the SAME param-group structure the fresh-study branch used,
    # so a saved 2-group optimizer state restores cleanly on the cycle_epochs=1
    # restart. backbone_tail_lr_mult MUST match the training config.
    optimizer, lr_mults = build_finetune_optimizer(
        model,
        optimizer_kwargs,
        backbone_tail_lr_mult=backbone_tail_lr_mult,
        backbone_finetune_scope=backbone_finetune_scope,
    )

    # Only restore optimizer for TRUE continuation checkpoints
    if (
        is_wrapper_checkpoint
        and "optimizer_state_dict" in checkpoint_data
    ):
        optimizer.load_state_dict(
            checkpoint_data["optimizer_state_dict"]
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    return model, optimizer, starting_epoch, lr_mults
