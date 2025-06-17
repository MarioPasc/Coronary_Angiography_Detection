"""Stratified **sample-level** K-fold generator for the ARCADE dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from ICA_Detection.splits.k_fold import LOGGER
from ICA_Detection.splits.k_fold.arcade.utils import (
    LABELS,
    annotation_path,
    extract_labels,
    load_metadata,
    patient_id,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def build_kfold_splits(meta_json: Path, k: int = 5, seed: int = 42) -> Dict[str, Any]:
    """
    Create a label-balanced **sample-level** K-fold split.

    Guarantees
    ----------
    * Every sample appears in exactly one validation fold.
    * Every fold contains ≥ 1 occurrence of each global label
      (``"stenosis"``, ``"negative"``) – if that label is present at all.
    * Fold sizes differ by ≤ 1 sample.
    """
    rng = random.Random(seed)
    samples = load_metadata(meta_json)

    # ------------------------------------------------------------------ #
    # Book-keeping: per-sample mappings
    # ------------------------------------------------------------------ #
    id_to_labels: Dict[str, Set[str]] = {}
    id_to_paths: Dict[str, tuple[str, str]] = {}
    sid_to_pat: Dict[str, str] = {}

    for sid, record in samples.items():
        id_to_labels[sid] = extract_labels(record)
        img_path = record["image"]["dataset_route"]
        ann_path = annotation_path(img_path, record["annotations"]["name"])
        id_to_paths[sid] = (img_path, ann_path)
        sid_to_pat[sid] = patient_id(sid)

    # ------------- collect label pools -------------------------------- #
    label_pools: Dict[str, List[str]] = defaultdict(list)
    for sid, lbls in id_to_labels.items():
        for lbl in lbls:
            label_pools[lbl].append(sid)

    # ------------------------------------------------------------------ #
    # Round-robin allocation
    # ------------------------------------------------------------------ #
    folds: List[Set[str]] = [set() for _ in range(k)]
    fold_sizes = [0] * k

    for lbl, pool in label_pools.items():
        rng.shuffle(pool)
        for i, sid in enumerate(pool):
            tgt = i % k
            folds[tgt].add(sid)
            fold_sizes[tgt] += 1

    # Fill the remaining samples (if any) to balance fold sizes
    assigned = set.union(*folds)
    leftovers = [sid for sid in samples if sid not in assigned]
    rng.shuffle(leftovers)
    for sid in leftovers:
        tgt = min(range(k), key=fold_sizes.__getitem__)
        folds[tgt].add(sid)
        fold_sizes[tgt] += 1

    # ------------------------------------------------------------------ #
    # Build serialisable structure
    # ------------------------------------------------------------------ #
    union_all = set.union(*folds)
    results: Dict[str, Any] = {}
    for idx in range(k):
        val_ids = folds[idx]
        train_ids = union_all - val_ids
        results[f"fold_{idx}"] = {
            "train": _split_info(train_ids, id_to_labels, id_to_paths, sid_to_pat),
            "val":   _split_info(val_ids,   id_to_labels, id_to_paths, sid_to_pat),
        }

    LOGGER.info("✅  Generated sample-level %d-fold splits.", k)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _split_info(
    ids: Set[str],
    id_to_labels: Dict[str, Set[str]],
    id_to_paths: Dict[str, tuple[str, str]],
    sid_to_pat: Dict[str, str],
) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    images: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    patients: Set[str] = set()

    for sid in ids:
        counter.update(id_to_labels[sid])
        img, anno = id_to_paths[sid]
        images[sid] = img
        labels[sid] = anno
        patients.add(sid_to_pat[sid])

    balance = {lbl: counter.get(lbl, 0) for lbl in LABELS + ("negative",)}
    LOGGER.info("Split size=%4d | patients=%4d | balance=%s",
                len(ids), len(patients), balance)

    return {
        "label_balance": balance,
        "patients": sorted(patients),
        "files": {"images": images, "annotations": labels},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI convenience
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate K-fold splits for ARCADE.")
    ap.add_argument("--input", required=True, type=Path, help="processed.json path")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("k_fold_splits.json"))
    args = ap.parse_args()

    splits = build_kfold_splits(args.input, k=args.k, seed=args.seed)
    args.output.write_text(json.dumps(splits, indent=2))
    print(f"[k-fold] Saved splits → {args.output.resolve()}")


if __name__ == "__main__":
    main()
