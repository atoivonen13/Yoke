"""Set up (and by default submit) KN LodeRunner studies from a CSV.

KN replacement for the retired ``applications/harnesses/START_study.py``. It is
upstream's :class:`~yoke.harnesses.base.HarnessStudy` (``yoke-start-study``)
except for the first launch: this harness keeps separate epoch-1 templates,
``training_START.input`` and ``training_START.slurm``, which differ from the
continuation templates (no ``--continuation``/``--checkpoint``, a different
``--ema_decay`` and ``--cpus-per-task``), so they are rendered as-is instead of
from ``training_input.tmpl``. The continuation templates are written exactly as
upstream writes them and are consumed by ``HarnessStudy.continuation_setup`` in
``train_LodeRunner_ddp.py``.

Run from this directory:

    python START_study.py --csv ddp_production.csv --cpFile cp_files.txt

``--dryrun`` (alias ``--no_submit``, used by make_seed_sweep.py) builds the study
directories without submitting.
"""

from pathlib import Path

from yoke.harnesses.base import HarnessStudy
from yoke.helpers import cli, strings


class KNHarnessStudy(HarnessStudy):
    """HarnessStudy that renders the KN epoch-1 templates for the first launch."""

    def generate_initial_inputs(self, study_dir: Path, study: dict) -> Path:
        """Render ``training_START.{input,slurm}`` into the study directory.

        Args:
            study_dir (Path): Directory in which to write the first-launch files.
            study (dict): Study substitution dictionary (one CSV row).

        Returns:
            Path: Path to the generated first-launch submission script.
        """
        if self.submission_type != "slurm":
            raise ValueError("The KN harness only ships slurm START templates.")
        sid = study["studyIDX"]
        input_path = study_dir / f"study{sid:03d}_START.input"
        submit_path = study_dir / f"study{sid:03d}_START.slurm"
        for src, dst in (
            ("training_START.input", input_path),
            ("training_START.slurm", submit_path),
        ):
            text = (self.template_dir / src).read_text()
            dst.write_text(strings.replace_keys(study, text))
        return submit_path


def main() -> None:
    """Build every CSV row's study directory and submit unless ``--dryrun``."""
    parser = cli.add_default_args()
    parser.add_argument(
        "--no_submit",
        dest="dryrun",
        action="store_true",
        help="Alias for --dryrun (the flag make_seed_sweep.py passes).",
    )
    args = parser.parse_args()

    harness = KNHarnessStudy(
        rundir=args.rundir,
        template_dir=".",
        cp_file=args.cpFile,
        submission_type=args.submissionType,
        dryrun=args.dryrun,
    )
    for study in harness.load_hyperparameters(args.csv):
        harness.run_study(study)


if __name__ == "__main__":
    main()
