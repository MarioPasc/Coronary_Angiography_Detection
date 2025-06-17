"""
Light-weight helpers for the ARCADE dataset.

Differences to the CADICA version
---------------------------------
* No stenosis–severity buckets – we only care about the presence/absence of
  lesions, so the label space collapses to ``{"stenosis"}``.
* One PNG ≙ one patient, one video, one timeframe → every sample can be
  treated independently, but we still expose a ``patient_id`` helper so that
  downstream code keeps working unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple

# Only a single positive class; negatives are added on-the-fly
LABELS: Tuple[str, ...] = ("stenosis",)

# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata(path: Path) -> Dict[str, dict]:
    """Return the *sample-id → record* dict stored in the processed ARCADE JSON."""
    with path.open("r") as fh:
        root = json.load(fh)
    return next(iter(root.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_labels(sample: dict) -> Set[str]:
    """
    Map one sample to the set of **present labels**.

    If the annotation dict is empty we yield ``{"negative"}``; otherwise the
    function always returns ``{"stenosis"}``.
    """
    return {"stenosis"} if sample["annotations"] else {"negative"}


def annotation_path(dataset_route: str, anno_name: str) -> str:
    """
    Translate an *image* path like

        …/images/foo.png

    to the corresponding YOLO label path

        …/labels/yolo/foo.txt
    """
    img_dir = Path(dataset_route).parent
    return str(img_dir.parent / "labels" / "yolo" / anno_name)


# ─────────────────────────────────────────────────────────────────────────────
# “Patient” helpers  (kept for API compatibility)
# ─────────────────────────────────────────────────────────────────────────────
_PAT_REGEX = re.compile(r"_p(\d+)_", re.IGNORECASE)


def patient_id(sample_id: str) -> str:
    """
    Extract a pseudo-patient identifier.

    Even though ARCADE has exactly one frame per patient, we still derive a
    *p###* token so that the splitter logic (which operates on patients) can
    remain unchanged.
    """
    m = _PAT_REGEX.search(sample_id)
    if not m:
        raise ValueError(f"Cannot parse patient id from {sample_id!r}")
    return f"p{m.group(1)}"
