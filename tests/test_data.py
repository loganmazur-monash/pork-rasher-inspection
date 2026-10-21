"""Tests for rasher_inspection.data and rasher_inspection.features (tiny fake data, no real photos needed)."""

import json

import numpy as np
import pytest
from PIL import Image

from rasher_inspection.data import (
    FAIL,
    PASS,
    label_counts,
    load_dataset,
    load_split,
    nearest_neighbour_distance,
    one_per_source,
    source_name,
)
from rasher_inspection.features import colour_histogram, meat_fraction, thumbnail


def write_fake_split(root, split, boxes_per_photo, copies=1):
    """Write a minimal COCO annotation file; ``boxes_per_photo[i]`` = number of boxes in photo i."""
    split_dir = root / split
    split_dir.mkdir(parents=True)
    images, annotations = [], []
    for photo, n_boxes in enumerate(boxes_per_photo):
        for c in range(copies):
            image_id = len(images)
            images.append({"id": image_id, "file_name": f"frame_with_red_{photo}_jpg.rf.{c:04x}.jpg"})
            for _ in range(n_boxes):
                annotations.append(
                    {"id": len(annotations), "image_id": image_id, "category_id": 4, "bbox": [10, 10, 20, 20]}
                )
    coco = {
        "categories": [{"id": 0, "name": "packaging-error"}, {"id": 4, "name": "unsealed"}],
        "images": images,
        "annotations": annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))


@pytest.fixture
def fake_dataset(tmp_path):
    write_fake_split(tmp_path, "train", [0, 1, 2, 1], copies=3)
    write_fake_split(tmp_path, "valid", [0, 1])
    write_fake_split(tmp_path, "test", [1, 0, 3])
    return tmp_path


def test_source_name_removes_roboflow_suffix():
    assert source_name("frame_with_red_8214_jpg.rf.5acda821.jpg") == "frame_with_red_8214_jpg.jpg"


def test_label_is_fail_only_when_boxes_exist(fake_dataset):
    records = load_split(fake_dataset, "test")
    assert [r.label for r in records] == [FAIL, PASS, FAIL]
    assert [len(r.defects) for r in records] == [1, 0, 3]


def test_load_split_rejects_unknown_split(fake_dataset):
    with pytest.raises(ValueError):
        load_split(fake_dataset, "holdout")


def test_load_dataset_reads_every_split_and_copy(fake_dataset):
    data = load_dataset(fake_dataset)
    assert {k: len(v) for k, v in data.items()} == {"train": 12, "valid": 2, "test": 3}


def test_one_per_source_drops_augmented_copies(fake_dataset):
    train = load_dataset(fake_dataset)["train"]
    unique = one_per_source(train)
    assert len(unique) == 4
    assert len({r.source for r in unique}) == 4


def test_label_counts(fake_dataset):
    assert label_counts(load_split(fake_dataset, "test")) == {"pass": 1, "fail": 2}


def test_nearest_neighbour_distance():
    refs = np.array([[0.0, 0.0], [10.0, 10.0]])
    queries = np.array([[1.0, 1.0], [10.0, 12.0]])
    np.testing.assert_allclose(nearest_neighbour_distance(queries, refs), [1.0, 1.0])


def test_thumbnail_ignores_brightness_changes(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(30, 200, size=(54, 72), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "a.png")
    Image.fromarray(np.clip(pixels * 1.1, 0, 255).astype(np.uint8)).save(tmp_path / "brighter.png")
    a, b = thumbnail(tmp_path / "a.png"), thumbnail(tmp_path / "brighter.png")
    assert np.abs(a - b).mean() < 0.05


def test_meat_fraction_counts_red_pixels_only(tmp_path):
    img = np.zeros((40, 40, 3), dtype=np.uint8)  # black belt
    img[:20] = (200, 40, 50)                       # top half red "meat"
    Image.fromarray(img).save(tmp_path / "half_red.png")
    assert meat_fraction(tmp_path / "half_red.png") == pytest.approx(0.5, abs=0.05)


def test_colour_histogram_shape_and_sums(tmp_path):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:20] = (200, 40, 50)
    Image.fromarray(img).save(tmp_path / "x.png")
    h = colour_histogram(tmp_path / "x.png", bins=8)
    assert h.shape == (24,)
    np.testing.assert_allclose([h[:8].sum(), h[8:16].sum(), h[16:].sum()], 1.0)
