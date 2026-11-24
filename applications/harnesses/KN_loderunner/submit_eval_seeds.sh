#!/bin/bash
# Submit the late-time eval of every seed replica: study <config><SS> = seed SS
# (make_seed_sweep.py), one single-GPU job each. eval_dense_latetime_9band.py
# sweeps the --sweep_epochs checkpoints on the validation split, keeps the best,
# calibrates on uniform truth and scores the test set, writing
# runs/study_<N>/dense_latetime_eval_9band/ for plot_seed_configs.py.
#
# Skips a replica whose training has not reached the last sweep epoch (no val
# record) or whose eval already exists (FORCE=1 re-runs it).
#
# Run from the KN_loderunner dir (where runs/ is):
#   bash submit_eval_seeds.sh                  # configs 218 219 220, seeds 0-4
#   bash submit_eval_seeds.sh 211 213          # other configs
#   bash submit_eval_seeds.sh 21803 22001      # single replicas (5-digit = study)
#   DRY_RUN=1 bash submit_eval_seeds.sh        # print what would be submitted
# Env overrides: SEEDS ("0 1 2 3 4"), SWEEP (200:300:10), CONDA_ENV (yoke311),
# TIME (4:00:00), ACCOUNT, PARTITION, FORCE, DRY_RUN.

set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4}"
SWEEP="${SWEEP:-200:300:10}"
CONDA_ENV="${CONDA_ENV:-yoke311}"
TIME="${TIME:-4:00:00}"
ACCOUNT="${ACCOUNT:-y26_artimis_ddc_g}"
PARTITION="${PARTITION:-standard}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(218 219 220)

if [ ! -f eval_dense_latetime_9band.py ]; then
    echo "ERROR: eval_dense_latetime_9band.py not in $(pwd); run from the KN_loderunner dir." >&2
    exit 1
fi

# Studies to evaluate: a config expands to its seed replicas.
STUDIES=()
for a in "${ARGS[@]}"; do
    if [ "$a" -ge 10000 ]; then
        STUDIES+=("$a")
    else
        for s in $SEEDS; do STUDIES+=($((a * 100 + s))); done
    fi
done

# Last epoch of the start:stop:step sweep; training must have reached it.
last_epoch=$(printf "%04d" "$(echo "$SWEEP" | cut -d: -f2)")
n_sub=0
for study in "${STUDIES[@]}"; do
    run_dir="runs/study_${study}"
    if [ ! -f "${run_dir}/validation_study${study}_epoch${last_epoch}.csv" ]; then
        echo "skip ${study}: training has not reached epoch ${last_epoch}"
        continue
    fi
    if [ "$FORCE" != 1 ] && \
       [ -f "${run_dir}/dense_latetime_eval_9band/latetime_uniform_scored_points.csv" ]; then
        echo "skip ${study}: eval exists (FORCE=1 to re-run)"
        continue
    fi
    cmd="eval_dense_latetime_9band.py --study ${study} --sweep_epochs ${SWEEP}"
    if [ "$DRY_RUN" = 1 ]; then
        echo "would submit ${study}: python -u ${cmd}"
        n_sub=$((n_sub + 1))
        continue
    fi
    sbatch --job-name="eval_${study}" --account="$ACCOUNT" --partition="$PARTITION" \
        --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 \
        --mem-per-gpu=50G --time="$TIME" --exclude=se024 \
        --output="eval_${study}_%j.out" <<EOF
#!/bin/bash
# Same clean conda activation as training_slurm.tmpl.
module load miniforge/26.9.4
eval "\$(conda shell.bash hook)"
for _ in 1 2 3 4 5 6 7 8; do
    [ "\${CONDA_SHLVL:-0}" -gt 0 ] || break
    conda deactivate
done
conda activate ${CONDA_ENV}
export OMP_NUM_THREADS=8
"\$CONDA_PREFIX/bin/python" -u ${cmd}
EOF
    n_sub=$((n_sub + 1))
done
echo "${n_sub} of ${#STUDIES[@]} studies submitted$([ "$DRY_RUN" = 1 ] && echo ' (dry run)')"
