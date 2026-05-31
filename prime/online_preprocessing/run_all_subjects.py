#!/usr/bin/env python3
"""
Run preprocessing and/or dipole extraction for every subject under
data/raw/{subject}/.

Example:
  python online_preprocessing/run_all_subjects.py --step both
  python online_preprocessing/run_all_subjects.py --step preprocess
  python online_preprocessing/run_all_subjects.py --step dipole --subject sub-018
  python online_preprocessing/run_all_subjects.py --max-subjects 10 --jobs 4
  python online_preprocessing/run_all_subjects.py --dry-run
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent.parent / "data"

RAW_DATA_DIR = DATA_ROOT / "raw"
DIPOLE_FIT_SCRIPT = SCRIPT_DIR / "fit_dipole.py"
PROCESSED_DIR = DATA_ROOT / "processed"


def discover_subjects(data_dir: Path) -> List[str]:
    """Return sorted subject ids from raw data layout."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")
    return sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def _run_command(command: List[str], env: dict, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"  {' '.join(command)}")
    subprocess.run(
        command,
        check=True,
        text=True,
        env=env,
        stderr=subprocess.STDOUT,
    )
    print(f"  OK: {label}")


def preprocess_done(subject: str) -> bool:
    return (PROCESSED_DIR / subject / f"{subject}_intervention_post.fif").is_file()


def dipole_done(subject: str) -> bool:
    return (PROCESSED_DIR / subject / f"{subject}_intervention_amplitudes.npy").is_file()


def _build_prep_env() -> dict:
    prep_env = os.environ.copy()
    prep_pythonpath = str(SCRIPT_DIR.parent)
    if prep_env.get("PYTHONPATH"):
        prep_pythonpath = f"{prep_pythonpath}{os.pathsep}{prep_env['PYTHONPATH']}"
    prep_env["PYTHONPATH"] = prep_pythonpath
    return prep_env


def _process_subject(
    subject: str,
    step: str,
    skip_existing: bool,
    dry_run: bool,
    python: str,
    prep_env: dict,
) -> Tuple[int, int]:
    """Run preprocess and/or dipole for one subject. Returns (ok, skip)."""
    n_ok = n_skip = 0
    print(f"\n========== {subject} ==========")

    if step in ("preprocess", "both"):
        if skip_existing and preprocess_done(subject):
            print(f"  SKIP preprocess: output exists for {subject}")
            n_skip += 1
        elif dry_run:
            print(f"  DRY-RUN: {python} -m online_preprocessing.preprocess --subject {subject}")
        else:
            _run_command(
                [python, "-u", "-m", "online_preprocessing.preprocess", "--subject", subject],
                prep_env,
                f"preprocess {subject}",
            )
            n_ok += 1

    if step in ("dipole", "both"):
        if skip_existing and dipole_done(subject):
            print(f"  SKIP dipole: output exists for {subject}")
            n_skip += 1
        elif dry_run:
            print(f"  DRY-RUN: {python} -m online_preprocessing.fit_dipole --subject {subject}")
        else:
            _run_command(
                [python, "-u", "-m", "online_preprocessing.fit_dipole", "--subject", subject],
                os.environ.copy(),
                f"dipole fit {subject}",
            )
            n_ok += 1

    return n_ok, n_skip


def _process_subject_worker(args: Tuple) -> Tuple[str, int, int]:
    subject, step, skip_existing, dry_run, python, prep_env = args
    n_ok, n_skip = _process_subject(subject, step, skip_existing, dry_run, python, prep_env)
    return subject, n_ok, n_skip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing and dipole extraction for all subjects."
    )
    parser.add_argument(
        "--step",
        choices=("preprocess", "dipole", "both"),
        default="both",
        help="Which pipeline to run (default: both).",
    )
    parser.add_argument("--subject", type=str, help="Only run this subject id (e.g. sub-018, sub-038_rep).")
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N subjects (after other filters).",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=4,
        metavar="N",
        help="Number of subjects to process in parallel (default: 4).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects whose output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.max_subjects is not None and args.max_subjects < 1:
        parser.error("--max-subjects must be at least 1")

    subjects = discover_subjects(RAW_DATA_DIR)
    if args.subject:
        subjects = [s for s in subjects if s == args.subject]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        print("No subjects matched filters. Exiting.")
        sys.exit(1)

    print(f"Found {len(subjects)} subject(s) under {RAW_DATA_DIR} (jobs={args.jobs})")

    prep_env = _build_prep_env()
    python = sys.executable

    worker_args = [
        (subject, args.step, args.skip_existing, args.dry_run, python, prep_env)
        for subject in subjects
    ]

    n_ok = n_skip = 0

    try:
        if args.dry_run or args.jobs == 1:
            for wa in worker_args:
                _, ok, skip = _process_subject_worker(wa)
                n_ok += ok
                n_skip += skip
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(_process_subject_worker, wa): wa[0] for wa in worker_args}
                for future in as_completed(futures):
                    subject = futures[future]
                    try:
                        _, ok, skip = future.result()
                        n_ok += ok
                        n_skip += skip
                    except Exception:
                        print(f"\nAborting: failure in {subject}")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
    except subprocess.CalledProcessError as e:
        print(f"\nAborting: command failed (exit {e.returncode})")
        sys.exit(e.returncode if e.returncode else 1)
    except Exception:
        sys.exit(1)

    print(f"\nDone. successes={n_ok}, skipped={n_skip}")


if __name__ == "__main__":
    main()
