"""Set up and launch seed replicas of existing studies (paper seed-noise runs).

Each replica is a NEW study that copies a config study's ddp_production.csv row
and recipe (``--config_study``) under its own ``--seed``. The replica study
number is ``config * 100 + seed`` (216 seed 3 -> study 21603), so the run
directories, checkpoints and eval outputs never collide with the originals.

Replicas are split into ``--lanes`` chains. The head of every lane is submitted
now; when a run finishes its last epoch, train_LodeRunner_ddp.py reads the
``NEXT_STUDY`` file in its study directory and submits the next run in that
lane. So:

    --lanes 1                 one run at a time, all replicas back to back
    --lanes 4                 at most 4 runs in the queue at once
    (default) lanes = runs    everything submitted at once

Run from this directory, like START_study.py:

    python make_seed_sweep.py --configs 213 216 --n_seeds 10 --lanes 4

Use ``--dry_run`` to write the CSV and print the plan without touching runs/.
"""

import argparse
import os
import subprocess

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--configs", type=int, nargs="+", required=True,
                    help="Study numbers to replicate.")
parser.add_argument("--n_seeds", type=int, default=10)
parser.add_argument("--first_seed", type=int, default=0)
parser.add_argument("--lanes", type=int, default=None,
                    help="Number of concurrent chains. Default: all at once.")
parser.add_argument("--csv", default="./ddp_production.csv",
                    help="CSV holding the config studies' rows.")
parser.add_argument("--sweep_csv", default=None,
                    help="CSV written for the replica rows. Default: "
                    "./seed_sweep_<configs>.csv.")
parser.add_argument("--rundir", default="./runs")
parser.add_argument("--cpFile", default="./cp_files.txt")
parser.add_argument("--start_script", default=None,
                    help="Path to START_study.py. Default: ./ then ../, then "
                    "applications/harnesses/ of the installed yoke repo.")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()
if args.sweep_csv is None:
    args.sweep_csv = "./seed_sweep_" + "_".join(map(str, args.configs)) + ".csv"


def find_start_script() -> str:
    """Locate START_study.py (standalone harness copies have no ../)."""
    if args.start_script is not None:
        candidates = [args.start_script]
    else:
        candidates = ["./START_study.py", "../START_study.py"]
        try:
            import yoke

            repo = os.path.dirname(os.path.dirname(os.path.dirname(yoke.__file__)))
            candidates.append(
                os.path.join(repo, "applications", "harnesses", "START_study.py")
            )
        except ImportError:
            pass
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"START_study.py not found (tried {candidates}); pass --start_script."
    )


start_script = find_start_script()

if args.first_seed + args.n_seeds > 100:
    raise ValueError("Replica study = config*100 + seed, so seeds must be < 100.")

base = pd.read_csv(args.csv, sep=",", header=0, index_col=0, comment="#",
                   engine="python")
missing = [c for c in args.configs if c not in base.index]
if missing:
    raise ValueError(f"Config studies not in {args.csv}: {missing}")

rows = []
for config in args.configs:
    for seed in range(args.first_seed, args.first_seed + args.n_seeds):
        row = base.loc[config].copy()
        row["CONFIG_STUDY"] = config
        row["SEED"] = seed
        row.name = config * 100 + seed
        rows.append(row)
sweep = pd.DataFrame(rows)
sweep.index.name = base.index.name

existing = [s for s in sweep.index
            if os.path.exists(os.path.join(args.rundir, f"study_{s:03d}"))]
if existing and not args.dry_run:
    raise FileExistsError(
        f"Study dirs already exist for {existing}; pick other seeds "
        "(--first_seed) or remove them."
    )

# Round-robin over configs so every lane interleaves them: with --lanes 1 an
# early stop still leaves seeds of every config, not 10 of the first one.
order = sorted(sweep.index, key=lambda s: (s % 100, s // 100))
n_lanes = min(args.lanes or len(order), len(order))
lanes = [order[i::n_lanes] for i in range(n_lanes)]

print(f"{len(order)} runs in {n_lanes} lane(s):")
for i, lane in enumerate(lanes):
    print(f"  lane {i}: " + " -> ".join(str(s) for s in lane))

sweep.to_csv(args.sweep_csv)
print(f"Wrote {args.sweep_csv}")
print(f"Using {start_script}")
if args.dry_run:
    raise SystemExit(0)

# Build every study dir (templates, inputs, copied scripts) without submitting.
# An older START_study.py without --no_submit exits on the unknown flag here, so
# it can never submit behind the chaining.
subprocess.run(
    ["python", start_script, "--csv", args.sweep_csv,
     "--rundir", args.rundir, "--cpFile", args.cpFile, "--no_submit"],
    check=True,
)

for lane in lanes:
    for cur, nxt in zip(lane, lane[1:]):
        next_dir = os.path.abspath(os.path.join(args.rundir, f"study_{nxt:03d}"))
        with open(os.path.join(args.rundir, f"study_{cur:03d}", "NEXT_STUDY"),
                  "w") as f:
            f.write(f"{next_dir}\nstudy{nxt:03d}_START.slurm\n")

for lane in lanes:
    head = lane[0]
    subprocess.run(["sbatch", f"study{head:03d}_START.slurm"],
                   cwd=os.path.join(args.rundir, f"study_{head:03d}"), check=True)
