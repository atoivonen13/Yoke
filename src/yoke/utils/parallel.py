"""Yoke module to assist GPU-parallel training.

Some models within Yoke require specific modifications to PyTorch multi-GPU
training utilities.

"""

import os
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Setup distributed training environment using SLURM environment variables.

    Required Environment Variables:
        SLURM_PROCID: Global rank of this process across all nodes
        SLURM_NTASKS: Total number of processes (world size)
        SLURM_LOCALID: Local rank on this node
        MASTER_ADDR: Address of the master node for communication
        MASTER_PORT: Port for distributed communication

    Returns:
        tuple: A 4-tuple containing:
            - rank (int): Global rank of this process (0 to world_size-1)
            - world_size (int): Total number of processes across all nodes
            - gpu_index (int): The CUDA device index this process was assigned.
              This is the actual device ordinal to use for DDP ``device_ids``/
              ``output_device`` -- 0 under per-task GPU binding (one visible
              GPU), or the local rank when all node GPUs are visible. NOT
              necessarily equal to SLURM_LOCALID.
            - device (torch.device): CUDA device object for this process
    """
    # ----- 1) Basic setup & environment variables -----
    rank = int(os.environ["SLURM_PROCID"])  # global rank
    world_size = int(os.environ["SLURM_NTASKS"])  # total number of processes
    local_rank = int(os.environ["SLURM_LOCALID"])  # local rank on this node

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # ----- 2) Set the current GPU device for this process -----
    # Two Slurm GPU-visibility models must both work, so pick the device index
    # from what THIS task actually sees rather than assuming local_rank is it:
    #   * Per-task binding (default cgroup isolation): each task sees exactly one
    #     GPU, renumbered to cuda:0. device_count() == 1, so the device index is
    #     0 for every rank (local_rank 1,2,3 would be invalid ordinals here).
    #   * All-visible (e.g. srun --gpu-bind=none): each task sees all node GPUs,
    #     device_count() == NGPUS, and local_rank is the correct distinct index.
    # Choosing 0 when only one GPU is visible, else local_rank, gives each rank a
    # DISTINCT physical GPU under both models and avoids "invalid device ordinal"
    # and "Duplicate GPU detected".
    n_visible = torch.cuda.device_count()
    gpu_index = 0 if n_visible <= 1 else local_rank
    print(
        f"[setup_distributed] rank={rank} local_rank={local_rank} "
        f"world_size={world_size} visible_gpus={n_visible} gpu_index={gpu_index} "
        f"master={master_addr}:{master_port} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if gpu_index >= n_visible:
        raise RuntimeError(
            f"Chosen GPU index {gpu_index} >= visible GPUs {n_visible} for "
            f"local_rank {local_rank}. Each task sees only a partial subset of the "
            "node's GPUs (neither clean per-task binding nor full visibility). "
            "Check the Slurm GPU request/binding for this job."
        )
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")

    # ----- 3) Initialize the process group -----
    # Pass ``device_id`` so the process group is eagerly bound to this rank's
    # device, matching the modern PyTorch DDP recommendation.
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        device_id=device,
    )

    # Return the ACTUAL device index (gpu_index), not SLURM_LOCALID, so callers
    # place DDP on the device this process really owns under both binding models.
    return rank, world_size, gpu_index, device


def cleanup_distributed() -> None:
    """Clean up distributed training environment.

    This function destroys the PyTorch distributed process group that was
    initialized by setup_distributed().
    """
    dist.destroy_process_group()


# Custom nn.DataParallel class to handle input to LodeRunner that should not be
# split by batch.
class LodeRunner_DataParallel(nn.DataParallel):
    """Handle unique GPU splitting of LodeRunner inputs.

    Since LodeRunner's *forward* method has multiple inputs consisting of
    several different shapes, some of which include a batch dimension and some
    of which do not, we must handle the splitting of data across multiple GPUs
    explicitly.

    .. deprecated::
        Vanilla :class:`torch.nn.DataParallel`-based training is deprecated and
        this class is targeted for **hard removal in December 2026**. DDP is the
        single supported parallelism model for Yoke harnesses; use
        :func:`setup_distributed` with
        :class:`torch.nn.parallel.DistributedDataParallel` instead.

    """

    def __init__(self, model: nn.Module) -> None:
        """Get it initialized using parent."""
        warnings.warn(
            "LodeRunner_DataParallel is deprecated and will be removed in "
            "December 2026. Use DistributedDataParallel (via "
            "yoke.utils.parallel.setup_distributed) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(model)

    def forward(self, *inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Handle explicit GPU splitting."""
        # Input is (start_img, in_vars, out_vars, Dt)
        image_input = inputs[0]
        in_vars = inputs[1]
        out_vars = inputs[2]
        Dt_input = inputs[3]

        # Split batchsize-dependent inputs and replicate fixed inputs
        if self.device_ids:
            # Copy model to device
            replicas = self.replicate(self.module, self.device_ids)

            # Split batchsize-dependent inputs
            inputs_split = nn.parallel.scatter((image_input, Dt_input), self.device_ids)

            # Replicate non-batchsize-dependent inputs
            in_vars_replicas = [in_vars.to(device) for device in self.device_ids]

            out_vars_replicas = [out_vars.to(device) for device in self.device_ids]

            # Combine splits and replicas
            inputs_combined = [
                (split_inputs[0], in_vars, out_vars, split_inputs[1])
                for split_inputs, in_vars, out_vars in zip(
                    inputs_split, in_vars_replicas, out_vars_replicas
                )
            ]

            # Forward pass with replicas and custom splits
            outputs = nn.parallel.parallel_apply(replicas, inputs_combined)

            return nn.parallel.gather(outputs, self.output_device)
        else:
            return self.module(*inputs, **kwargs)
