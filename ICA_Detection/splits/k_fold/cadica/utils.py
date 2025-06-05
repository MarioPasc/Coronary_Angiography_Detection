"""Utility helpers (pure functions, no side-effects)."""

from __future__ import annotations

import json
import re
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


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata(path: Path) -> Dict[str, dict]:
    """Load the JSON produced by the annotation pipeline."""
    with path.open("r") as fh:
        root = json.load(fh)
    # we only care about the first (and only) task key
    return next(iter(root.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_labels(sample: dict) -> Set[str]:
    """
    Return **all** stenosis labels contained in one image.

    Empty / non-lesion images yield ``{"negative"}`` so we can stratify on it.
    """
    stenosis: dict = sample["annotations"]["stenosis"]
    if not stenosis:
        return {"negative"}

    lbls: Set[str] = set()
    for box in stenosis.values():
        raw = box["label"]
        lbls.add(raw if raw in LABELS else "unknown")
    return lbls


def annotation_path(dataset_route: str, anno_name: str) -> str:
    """
    Given an image path ``/…/images/foo.png`` return
    ``/…/labels/yolo/foo.txt``.
    """
    images_dir = Path(dataset_route).parent               # …/images
    return str(images_dir.parent / "labels" / "yolo" / anno_name)


# ─────────────────────────────────────────────────────────────────────────────
# Patient helpers
# ─────────────────────────────────────────────────────────────────────────────
_PAT_REGEX = re.compile(r"_p(\d+)_", re.IGNORECASE)


def patient_id(sample_id: str) -> str:
    """
    Extract *patient identifier* from an image id.

    Example
    -------
    >>> patient_id("cadica_p26_v5_00014")
    'p26'
    """
    m = _PAT_REGEX.search(sample_id)
    if not m:
        raise ValueError(f"Cannot parse patient id from {sample_id!r}")
    return f"p{m.group(1)}"
