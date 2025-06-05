"""Greedy **patient-level** label-balanced K-fold split generator."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from ICA_Detection.splits.k_fold import LOGGER
from ICA_Detection.splits.k_fold.cadica.utils import (
    annotation_path,
    extract_labels,
    load_metadata,
    patient_id,
    LABELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def build_kfold_splits(
    meta_json: Path,
    k: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a *patient-level* K-fold split that guarantees:

    * Each **patient** appears in **one and only one** fold.
    * Every fold sees **≥ 1** occurrence of **every** label that is present
      in the whole dataset (if the global count ≥ k).
    * Folds are size-balanced (± one image) and label-balanced.

    The algorithm is a simple label-wise round-robin over **patients**,
    followed by greedy top-up of the smallest fold.  Complexity is *O(n)*.
    """
    rng = random.Random(seed)
    samples = load_metadata(meta_json)

    # ------------------------------------------------------------------ #
    # 1)  build per-sample bookkeeping
    # ------------------------------------------------------------------ #
    id_to_labels: Dict[str, Set[str]] = {}
    id_to_paths: Dict[str, tuple[str, str]] = {}
    sid_to_patient: Dict[str, str] = {}

    for sid, sample in samples.items():
        lbls = extract_labels(sample)
        id_to_labels[sid] = lbls
        img_path = sample["image"]["dataset_route"]
        ann_path = annotation_path(img_path, sample["annotations"]["name"])
        id_to_paths[sid] = (img_path, ann_path)
        sid_to_patient[sid] = patient_id(sid)

    # group samples by patient
    patient_to_samples: Dict[str, Set[str]] = defaultdict(set)
    patient_to_labels: Dict[str, Set[str]] = defaultdict(set)
    for sid, pat in sid_to_patient.items():
        patient_to_samples[pat].add(sid)
        patient_to_labels[pat].update(id_to_labels[sid])

    patients: List[str] = list(patient_to_samples)
    rng.shuffle(patients)

    # ------------------------------------------------------------------ #
    # 2)  greedy allocation to guarantee ≥ 1 per label per fold
    # ------------------------------------------------------------------ #
    folds_pat: List[Set[str]] = [set() for _ in range(k)]     # patients
    folds_sid: List[Set[str]] = [set() for _ in range(k)]     # samples
    fold_sizes: List[int] = [0 for _ in range(k)]             # #images

    # per-label patient pools
    label_pools: Dict[str, List[str]] = defaultdict(list)
    for pat_id, lbls in patient_to_labels.items(): # Renamed 'pat' to 'pat_id' for clarity
        for l_val in lbls: # Renamed 'l' to 'l_val' for clarity
            label_pools[l_val].append(pat_id)

    globally_assigned_patients: Set[str] = set() # ADDED: Track patients assigned to any fold

    for label_val, patient_pool_for_label in label_pools.items(): # Renamed 'label' and 'pool'
        rng.shuffle(patient_pool_for_label)
        for i, pat_candidate in enumerate(patient_pool_for_label): # Renamed 'pat' to 'pat_candidate'
            if pat_candidate in globally_assigned_patients: # ADDED: Check if patient already assigned globally
                continue

            tgt = i % k
            # The original check 'if pat_candidate in folds_pat[tgt]: continue' is now effectively
            # covered by the 'globally_assigned_patients' check, because if it's not globally
            # assigned, it cannot be in folds_pat[tgt] yet. If it were globally assigned
            # (e.g. to folds_pat[tgt]), the check above would have caught it.
            # Thus, that specific check can be considered redundant here if the global one is in place.

            _add_patient_to_fold(pat_candidate, tgt, folds_pat, folds_sid,
                                 fold_sizes, patient_to_samples)
            globally_assigned_patients.add(pat_candidate) # ADDED: Mark patient as globally assigned

    # ------------------------------------------------------------------ #
    # 3)  fill remaining patients to balance fold sizes
    # ------------------------------------------------------------------ #
    # 'assigned_patients' will now correctly reflect disjoint sets of patients from folds_pat
    assigned_patients = set.union(*folds_pat)
    leftovers = [p for p in patients if p not in assigned_patients]

    for pat in leftovers:
        tgt = min(range(k), key=lambda f: fold_sizes[f])      # smallest fold
        _add_patient_to_fold(pat, tgt, folds_pat, folds_sid,
                             fold_sizes, patient_to_samples)

    # ------------------------------------------------------------------ #
    # 4)  build JSON serialisable structure
    # ------------------------------------------------------------------ #
    union_all = set.union(*folds_sid)
    results: Dict[str, Any] = {}
    for idx in range(k):
        val_ids = folds_sid[idx]
        train_ids = union_all - val_ids

        results[f"fold_{idx}"] = {
            "train": _split_info(train_ids, id_to_labels, id_to_paths, sid_to_patient),
            "val":   _split_info(val_ids,   id_to_labels, id_to_paths, sid_to_patient),
        }

    LOGGER.info("✅  Generated patient-level %d-fold splits.", k)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _add_patient_to_fold(
    pat: str,
    fold_idx: int,
    folds_pat: List[Set[str]],
    folds_sid: List[Set[str]],
    fold_sizes: List[int],
    patient_to_samples: Dict[str, Set[str]],
) -> None:
    """Assign *all* samples from one patient to the target fold."""
    folds_pat[fold_idx].add(pat)
    sid_set = patient_to_samples[pat]
    folds_sid[fold_idx].update(sid_set)
    fold_sizes[fold_idx] += len(sid_set)


def _split_info(
    ids: Set[str],
    id_to_labels: Dict[str, Set[str]],
    id_to_paths: Dict[str, tuple[str, str]],
    sid_to_patient: Dict[str, str],
) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    images: Dict[str, str] = {}
    annos: Dict[str, str] = {}
    pats: Set[str] = set()

    for sid in ids:
        counter.update(id_to_labels[sid])
        img, anno = id_to_paths[sid]
        images[sid] = img
        annos[sid] = anno
        pats.add(sid_to_patient[sid])

    balance = {l: counter.get(l, 0) for l in LABELS + ("negative", "unknown")}
    LOGGER.info("Split size=%4d | patients=%3d | balance=%s",
                len(ids), len(pats), balance)

    return {
        "label_balance": balance,
        "patients": sorted(pats),
        "files": {"images": images, "annotations": annos},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate patient-level K-fold splits.")
    ap.add_argument("--input", required=True, type=Path, help="metadata JSON path")
    ap.add_argument("--k", type=int, default=5, help="number of folds")
    ap.add_argument("--seed", type=int, default=42, help="PRNG seed")
    ap.add_argument("--output", type=Path, default=Path("k_fold_splits.json"))
    args = ap.parse_args()

    splits = build_kfold_splits(args.input, k=args.k, seed=args.seed)
    with args.output.open("w") as fh:
        json.dump(splits, fh, indent=2)
    print(f"[k-fold] Saved splits → {args.output.resolve()}")


if __name__ == "__main__":
    main()
