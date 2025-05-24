"""Utility helpers (pure functions, no side-effects)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


LABELS: Tuple[str, ...] = (
    "p0_20",
    "p20_50",
    "p50_70",
    "p70_90",
    "p90_98",
    "p99",
    "p100",
)


def load_metadata(path: Path) -> Dict[str, dict]:
    """Load the JSON produced by the annotation pipeline."""
    with path.open("r") as fh:
        root = json.load(fh)
    # top-level task → dict[id → sample]
    # we only care about the first key ("Stenosis_Detection")
    return next(iter(root.values()))


def extract_labels(sample: dict) -> Set[str]:
    """
    Return *all* stenosis labels contained in one image.

    Non-lesion images yield {"negative"} to allow stratification.
    """
    stenosis: dict = sample["annotations"]["stenosis"]
    if not stenosis:  # empty dict
        return {"negative"}

    lbls: Set[str] = set()
    for box in stenosis.values():
        raw = box["label"]
        lbls.add(raw if raw in LABELS else "unknown")
    return lbls


def annotation_path(dataset_route: str, anno_name: str) -> str:
    """
    Given an image path `/.../images/foo.png` → `/.../labels/yolo/foo.txt`.
    """
    images_dir = Path(dataset_route).parent           # .../images
    return str(images_dir.parent / "labels" / "yolo" / anno_name)
