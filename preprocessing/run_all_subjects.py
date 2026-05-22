#!/usr/bin/env python3
"""
Run preprocessing and/or dipole extraction for every subject under
data_epoched/raw_eeglab_and_block_idents/{Aalto,Tuebingen}/.

Example:
  python preprocessing/run_all_subjects.py --step both
  python preprocessing/run_all_subjects.py --step preprocess --site Tuebingen
  python preprocessing/run_all_subjects.py --step dipole --subject sub-018
  python preprocessing/run_all_subjects.py --dry-run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_PREP_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PREP_ROOT.parent
RAW_DATA_DIR = _PREP_ROOT / "data_epoched" / "raw_eeglab_and_block_idents"
PREPROCESS_SCRIPT = _PREP_ROOT / "prep_data" / "preprocessing_single_subject.py"
DIPOLE_SCRIPT = _PREP_ROOT / "extract_teps" / "single_trial_dipole_amplitude_slurm.py"
PROCESSED_DIR = _PREP_ROOT / "data_processed_pre_ica_False_v4"
DIPOLES_DIR = _PREP_ROOT / "dipoles_pre_ica_False_v4"


def discover_subjects(data_dir: Path) -> List[Tuple[str, str]]:
    """Return sorted (site, subject) pairs from raw data layout."""
    subjects: List[Tuple[str, str]] = []
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")
    for site_dir in sorted(data_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for subject_dir in sorted(site_dir.iterdir()):
            if subject_dir.is_dir():
                subjects.append((site_dir.name, subject_dir.name))
    return subjects


def _run_command(command: List[str], env: dict, label: str, continue_on_error: bool) -> bool:
    print(f"\n--- {label} ---")
    print(f"  {' '.join(command)}")
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            env=env,
            stderr=subprocess.STDOUT,
        )
        print(f"  OK: {label}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {label} (exit {e.returncode})")
        if not continue_on_error:
            raise
        return False


def preprocess_done(subject: str) -> bool:
    return (PROCESSED_DIR / subject / f"{subject}_post.fif").is_file()


def dipole_done(subject: str) -> bool:
    return (DIPOLES_DIR / subject / f"{subject}_response_extraction_info.npz").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing and dipole extraction for all Aalto/Tuebingen subjects."
    )
    parser.add_argument(
        "--step",
        choices=("preprocess", "dipole", "both"),
        default="both",
        help="Which pipeline to run (default: both).",
    )
    parser.add_argument("--site", type=str, help="Only run subjects from this site (e.g. Aalto, Tuebingen).")
    parser.add_argument("--subject", type=str, help="Only run this subject id (e.g. sub-018, sub-038_rep).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects whose output files already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue with remaining subjects after a failure (default: on).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failing subject.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    args = parser.parse_args()
    continue_on_error = args.continue_on_error and not args.stop_on_error

    subjects = discover_subjects(RAW_DATA_DIR)
    if args.site:
        subjects = [(s, sub) for s, sub in subjects if s == args.site]
    if args.subject:
        subjects = [(s, sub) for s, sub in subjects if sub == args.subject]
    if not subjects:
        print("No subjects matched filters. Exiting.")
        sys.exit(1)

    print(f"Found {len(subjects)} subject(s) under {RAW_DATA_DIR}")

    prep_env = os.environ.copy()
    prep_pythonpath = str(PREPROCESS_SCRIPT.parent)
    if prep_env.get("PYTHONPATH"):
        prep_pythonpath = f"{prep_pythonpath}{os.pathsep}{prep_env['PYTHONPATH']}"
    prep_env["PYTHONPATH"] = prep_pythonpath

    python = sys.executable
    n_ok = 0
    n_fail = 0
    n_skip = 0

    for i, (site, subject) in enumerate(subjects, start=1):
        print(f"\n========== [{i}/{len(subjects)}] {site}/{subject} ==========")

        if args.step in ("preprocess", "both"):
            if args.skip_existing and preprocess_done(subject):
                print(f"  SKIP preprocess: output exists for {subject}")
                n_skip += 1
            elif args.dry_run:
                print(
                    f"  DRY-RUN: {python} {PREPROCESS_SCRIPT} --site {site} --subject {subject}"
                )
            else:
                ok = _run_command(
                    [python, "-u", str(PREPROCESS_SCRIPT), "--site", site, "--subject", subject],
                    prep_env,
                    f"preprocess {site}/{subject}",
                    continue_on_error,
                )
                if not ok:
                    n_fail += 1
                    if args.step == "both":
                        print(f"  Skipping dipole for {subject} after preprocess failure")
                        continue
                else:
                    n_ok += 1

        if args.step in ("dipole", "both"):
            if args.skip_existing and dipole_done(subject):
                print(f"  SKIP dipole: output exists for {subject}")
                n_skip += 1
            elif args.dry_run:
                print(f"  DRY-RUN: {python} {DIPOLE_SCRIPT} --subject {subject}")
            else:
                ok = _run_command(
                    [python, "-u", str(DIPOLE_SCRIPT), "--subject", subject],
                    os.environ.copy(),
                    f"dipole {subject}",
                    continue_on_error,
                )
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1

    print(f"\nDone. successes={n_ok}, failures={n_fail}, skipped={n_skip}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
