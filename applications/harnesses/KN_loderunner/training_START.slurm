#!/bin/bash

# This is a setup for GPU training on Venado. Find out how much
# memory per node, number of CPUs/node.

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. 

# The following are one set of SBATCH options for the Venado GPU
# partition. There are optional other constraints.

#SBATCH --job-name=ddp_s<studyIDX>_e0001
#SBATCH --account=y26_artimis_ddc_g
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --nodes=<KNODES>
#SBATCH --ntasks-per-node=<NGPUS>
#SBATCH --gpus-per-node=<NGPUS>
# nodes have 8 H100s and 192-208 CPUs, i.e. ~24 cores/GPU. One task per GPU
# (ntasks-per-node=NGPUS), so cpus-per-task=24 gives each rank enough cores for
# its dataloader workers + OMP threads (24*NGPUS stays within budget for NGPUS
# up to 8). Must match the continuation template so epoch 1 behaves like later
# epochs; without it Slurm may hand each task a single core and starve the
# dataloaders at multi-GPU scale.
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=50G
#SBATCH --exclude=se024
#SBATCH --output=study<studyIDX>_epoch0001.out
#SBATCH --error=study<studyIDX>_epoch0001.err

# Set the master node's address and port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 1024-65535 -n 1)  # Choose a random port
export MASTER_ADDR
export MASTER_PORT

# Enable shell debugging
set -xv

# Check available GPUs
sinfo  -o "%P %.24G %N"
srun -vv --cpu-bind=verbose /usr/bin/echo $CUDA_AVAILABLE_DEVICES
nvidia-smi

# Specify NCCL communication
export NCCL_SOCKET_IFNAME=ib0  # Check possible interfaces with `ip link show`

# for multi-node training
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3  # sometimes 0 or 3 depending on subnet manager config

# Debugging distributed data parallel
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

# Debugging selene slurm
export SLURM_CPU_BIND=verbose

# Load correct conda environment. sbatch exports the submitting shell's
# environment, so a conda env active there gets stacked under base here and
# miniforge's bin (no numpy) lands ahead of the env's bin on PATH. Clear any
# inherited activation, activate cleanly, and run the env's python by path.
module load miniforge/26.9.4
eval "$(conda shell.bash hook)"
for _ in 1 2 3 4 5 6 7 8; do
    [ "${CONDA_SHLVL:-0}" -gt 0 ] || break
    conda deactivate
done
conda activate <YOKE_TORCH_ENV>
PYTHON="$CONDA_PREFIX/bin/python"
"$PYTHON" -c "import numpy, torch" || {
    echo "ERROR: $PYTHON cannot import numpy/torch (CONDA_PREFIX=$CONDA_PREFIX)"
    exit 1
}

# Set number of threads per GPU. Must be <= cpus-per-task (24). Leaves cores for
# the dataloader workers (NUM_WORKERS per rank) alongside the OMP compute threads.
export OMP_NUM_THREADS=8

# Get start time
export date00=`date`

# Start the Code
# Explicitly set TCP environment for the following...
srun -vv --cpu-bind=verbose "$PYTHON" -u <train_script> @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
