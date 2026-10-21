"""Load the dataset and derive pass/fail labels.

The dataset is a Roboflow export in COCO format: one folder per split
(``train``, ``valid``, ``test``), each holding the JPEG photos and an
``_annotations.coco.json`` file listing the defect boxes in every photo.

Key decisions made in this module (each one is explained in the report):

* **Pass/fail rule.** A photo with at least one defect box is labelled
  ``FAIL`` (1); a photo with no boxes is labelled ``PASS`` (0).
* **Augmented copies.** Roboflow stores 3 exposure-shifted copies of every
  training photo (``<name>_jpg.rf.<hash>.jpg``, same ``<name>``). Statistics
  about the *content* of the dataset count each original photo once
  (``one_per_source``); model training uses every copy.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")
"""Folder names of the official Roboflow splits."""

PASS, FAIL = 0, 1
"""Integer class labels used everywhere in the project."""

_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+")


@dataclass(frozen=True)
class ImageRecord:
    """One photo and its annotations.

    Attributes:
        path: Location of the JPEG file.
        split: Official Roboflow split (train/valid/test).
        defects: Defect class name of every box in the photo (empty = no defects).
        boxes: Bounding boxes as ``(x, y, width, height)`` in pixels.
    """

    path: Path
    split: str
    defects: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...]

    @property
    def label(self) -> int:
        """``FAIL`` (1) if the photo has any defect box, otherwise ``PASS`` (0)."""
        return FAIL if self.boxes else PASS

    @property
    def source(self) -> str:
        """Original photo name, shared by all augmented copies of that photo."""
        return source_name(self.path.name)


def source_name(file_name: str) -> str:
    """Remove Roboflow's ``.rf.<hash>`` suffix to get the original photo name.

    Example:
        ``"frame_with_red_8214_jpg.rf.5acd.jpg"`` -> ``"frame_with_red_8214_jpg.jpg"``
    """
    return _ROBOFLOW_SUFFIX.sub("", file_name)


def load_split(dataset_dir: str | Path, split: str) -> list[ImageRecord]:
    """Read one official split's COCO annotation file.

    Args:
        dataset_dir: Folder containing the ``train``/``valid``/``test`` folders.
        split: One of ``SPLITS``.

    Returns:
        One ``ImageRecord`` per photo listed in the annotation file.

    Raises:
        ValueError: If ``split`` is not a known split name.
        FileNotFoundError: If the annotation file is missing.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    split_dir = Path(dataset_dir) / split
    with open(split_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    # Roboflow lists 'packaging-error' twice (id 0 is an unused parent
    # category), so classes are identified by name, not by id.
    class_name = {c["id"]: c["name"] for c in coco["categories"]}
    boxes_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        boxes_by_image[ann["image_id"]].append(ann)

    records = []
    for img in coco["images"]:
        anns = boxes_by_image[img["id"]]
        records.append(
            ImageRecord(
                path=split_dir / img["file_name"],
                split=split,
                defects=tuple(class_name[a["category_id"]] for a in anns),
                boxes=tuple(tuple(a["bbox"]) for a in anns),
            )
        )
    return records


def load_dataset(dataset_dir: str | Path) -> dict[str, list[ImageRecord]]:
    """Load all three official splits, e.g. ``data["test"]`` is a list of records."""
    return {split: load_split(dataset_dir, split) for split in SPLITS}


def one_per_source(records: list[ImageRecord]) -> list[ImageRecord]:
    """Keep only the first copy of each original photo (drops augmented duplicates)."""
    seen = {}
    for r in records:
        seen.setdefault(r.source, r)
    return list(seen.values())


def label_counts(records: list[ImageRecord]) -> dict[str, int]:
    """Count PASS and FAIL photos, e.g. ``{"pass": 12, "fail": 68}``."""
    n_fail = sum(r.label for r in records)
    return {"pass": len(records) - n_fail, "fail": n_fail}


def nearest_neighbour_distance(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Distance from each query vector to its closest reference vector.

    Distance = mean absolute difference between the two vectors. Used with
    ``features.thumbnail`` vectors to look for near-duplicate photos.

    Args:
        queries: Array of shape (n_queries, n_features).
        references: Array of shape (n_references, n_features).

    Returns:
        Array of shape (n_queries,).
    """
    return np.array([np.abs(references - q).mean(axis=1).min() for q in queries])
