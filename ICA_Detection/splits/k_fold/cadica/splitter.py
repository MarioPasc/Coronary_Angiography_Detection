"""Greedy label–balanced K-fold split generator."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from ICA_Detection.splits.k_fold.cadica.utils import annotation_path, extract_labels, load_metadata, LABELS
from ICA_Detection.splits.k_fold import LOGGER

def build_kfold_splits(
    meta_json: Path,
    k: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Produce `k_fold_splits.json` structure.

    Strategy
    --------
    *   Build a pool per label.
    *   Distribute samples round-robin label-wise so every fold sees **≥1**
        occurrence of each label (if available globally).
    *   When a sample carries multiple labels, it is assigned the first
        *unallocated* fold encountered in the round-robin cycle.
    *   Remaining (still unassigned) samples are filled in to keep folds
        size-balanced (±1).

    This simple greedy method gives excellent balance for < 10 labels and
    thousands of images without external dependencies.
    """
    rng = random.Random(seed)
    samples = load_metadata(meta_json)

    # ------------------------------------------------------------------ #
    # Step 1: prepare per-sample info
    # ------------------------------------------------------------------ #
    sample_ids: List[str] = list(samples)
    rng.shuffle(sample_ids)

    id_to_labels: Dict[str, Set[str]] = {}
    id_to_paths: Dict[str, tuple[str, str]] = {}

    for sid in sample_ids:
        s = samples[sid]
        lbls = extract_labels(s)
        id_to_labels[sid] = lbls
        img_path = s["image"]["dataset_route"]
        ann_path = annotation_path(img_path, s["annotations"]["name"])
        id_to_paths[sid] = (img_path, ann_path)

    # ------------------------------------------------------------------ #
    # Step 2: greedy allocation to guarantee ≥1 per label per fold
    # ------------------------------------------------------------------ #
    folds: List[Set[str]] = [set() for _ in range(k)]
    label_pools: Dict[str, List[str]] = defaultdict(list)
    for sid, lbls in id_to_labels.items():
        for l in lbls:
            label_pools[l].append(sid)

    # round-robin
    for label, pool in label_pools.items():
        rng.shuffle(pool)
        for i, sid in enumerate(pool):
            target_fold = i % k
            if sid not in folds[target_fold]:
                folds[target_fold].add(sid)

    # ------------------------------------------------------------------ #
    # Step 3: fill gaps to balance fold sizes
    # ------------------------------------------------------------------ #
    all_assigned = set.union(*folds)
    left_over = [sid for sid in sample_ids if sid not in all_assigned]

    # simple size-based top-up
    for sid in left_over:
        smallest = min(range(k), key=lambda f: len(folds[f]))
        folds[smallest].add(sid)

    # ------------------------------------------------------------------ #
    # Step 4: build JSON serialisable result
    # ------------------------------------------------------------------ #
    result: Dict[str, Any] = {}
    for idx in range(k):
        test_ids = folds[idx]
        train_ids = set.union(*folds) - test_ids

        result[f"fold_{idx}"] = {
            "train": _split_info(train_ids, id_to_labels, id_to_paths),
            "test": _split_info(test_ids, id_to_labels, id_to_paths),
        }
    return result


def _split_info(
    ids: Set[str],
    id_to_labels: Dict[str, Set[str]],
    id_to_paths: Dict[str, tuple[str, str]],
) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    images: Dict[str, str] = {}
    annos: Dict[str, str] = {}
    for sid in ids:
        counter.update(id_to_labels[sid])
        img, anno = id_to_paths[sid]
        images[sid] = img
        annos[sid] = anno

    # ensure all labels appear in dict (even count 0)
    balance = {l: counter.get(l, 0) for l in LABELS + ("negative", "unknown")}
    LOGGER.info(f"[CADICA] Split size: {len(ids)} samples, balance: {balance}")
    return {
        "label_balance": balance,
        "files": {
            "images": images,
            "annotations": annos,
        },
    }


# ---------------------------------------------------------------------- #
# ----------------------------   CLI   --------------------------------- #
# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label-balanced K-fold splits.")
    parser.add_argument("--input", required=True, type=Path, help="metadata JSON path")
    parser.add_argument("--k", type=int, default=5, help="number of folds")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--output", type=Path, default=Path("k_fold_splits.json"))
    args = parser.parse_args()

    splits = build_kfold_splits(args.input, k=args.k, seed=args.seed)
    with args.output.open("w") as fh:
        json.dump(splits, fh, indent=2)
    print(f"[k-fold] Saved splits → {args.output.resolve()}")


if __name__ == "__main__":
    main()
