#!/usr/bin/env python3
"""
Utility for inspecting or editing a single trial in an Optuna study (.db).

Two mutually-exclusive actions:
    --check       : print the state of the trial
    --mark-fail   : set the trial state to FAILED

Usage examples
--------------
# Check a trial
python optuna_trial_tool.py --db study.db --trial 5 --check

# Force-fail a trial
python optuna_trial_tool.py --db study.db --trial 5 --mark-fail
"""

from __future__ import annotations

import argparse
import sys
import optuna
from optuna.trial import TrialState
from typing import Optional


# --------------------------------------------------------------------- #
# ------------------------- helper functions -------------------------- #
# --------------------------------------------------------------------- #
def _load_study(db_path: str, study_name: Optional[str]) -> optuna.study.Study:
    """Open the first (or named) study inside an SQLite file."""
    storage_url = f"sqlite:///{db_path}"
    if study_name is not None:
        return optuna.load_study(study_name=study_name, storage=storage_url)

    # no name provided → pick the first one
    studies = optuna.get_all_study_summaries(storage=storage_url)
    if not studies:
        sys.exit(f"[ERROR] No studies found in {db_path}")
    return optuna.load_study(study_name=studies[0].study_name, storage=storage_url)


def check_trial(study: optuna.study.Study, number: int) -> None:
    trial = study.trials[number]  # raises if out of range
    print(f"Trial {number}: state = {trial.state}")


def mark_fail(study: optuna.study.Study, number: int) -> None:
    trial = study.trials[number]
    storage = study._storage  # private attr but stable across versions

    storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
    print(f"Trial {number} marked as FAILED.")


# --------------------------------------------------------------------- #
# -------------------------------  CLI  ------------------------------- #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:  # noqa: D401
    ap = argparse.ArgumentParser(description="Inspect or edit a single Optuna trial.")
    ap.add_argument("--db", required=True, help="Path to the SQLite .db file.")
    ap.add_argument("--study-name", help="Study name (optional if only one).")
    ap.add_argument("--trial", type=int, required=True, help="Trial number (0-based).")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Print the trial state.")
    group.add_argument("--mark-fail", action="store_true", help="Set state to FAILED.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    study = _load_study(args.db, args.study_name)

    if args.check:
        check_trial(study, args.trial)
    elif args.mark_fail:
        mark_fail(study, args.trial)


if __name__ == "__main__":
    main()
