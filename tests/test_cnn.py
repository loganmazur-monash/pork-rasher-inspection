"""Tests for rasher_inspection.cnn (small random inputs; no pre-trained download needed)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from rasher_inspection.cnn import (  # noqa: E402
    TrayDataset,
    build_model,
    class_weights,
    count_flops,
    count_parameters,
    input_size,
    make_loader,
    make_transform,
    predict,
)
from rasher_inspection.data import FAIL, PASS, ImageRecord  # noqa: E402


def fake_records(tmp_path, labels):
    """Small random photos saved to disk, wrapped as ImageRecords."""
    rng = np.random.default_rng(0)
    records = []
    for i, label in enumerate(labels):
        path = tmp_path / f"frame_with_red_{i}_jpg.rf.{i:04x}.jpg"
        Image.fromarray(rng.integers(0, 255, (54, 72, 3), dtype=np.uint8)).save(path)
        boxes = ((1.0, 1.0, 5.0, 5.0),) if label == FAIL else ()
        records.append(ImageRecord(path=Path(path), split="test", defects=("unsealed",) * len(boxes), boxes=boxes))
    return records


def test_input_size_keeps_4_by_3_shape():
    assert input_size(224) == (168, 224)
    assert input_size(448) == (336, 448)


@pytest.mark.parametrize("train", [True, False])
def test_transform_output_shape(train):
    image = Image.new("RGB", (720, 540))
    assert make_transform(64, train)(image).shape == (3, 48, 64)


def test_dataset_returns_tensor_and_label(tmp_path):
    records = fake_records(tmp_path, [PASS, FAIL])
    ds = TrayDataset(records, make_transform(64, train=False))
    image, label = ds[1]
    assert image.shape == (3, 48, 64) and label == FAIL
    assert len(ds) == 2


def test_model_outputs_two_scores_per_photo():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(3, 3, 48, 64))
    assert out.shape == (3, 2)
    assert 2_000_000 < count_parameters(model) < 3_000_000


def test_class_weights_balance_the_classes():
    w = class_weights([PASS, FAIL, FAIL, FAIL])
    assert w.tolist() == pytest.approx([2.0, 2 / 3])
    # weight × count is equal for both classes
    assert w[PASS] * 1 == pytest.approx(w[FAIL] * 3)


def test_flops_grow_with_input_size():
    model = build_model(pretrained=False)
    small, large = count_flops(model, 64), count_flops(model, 128)
    assert small > 0
    assert 3.5 < large / small < 4.5   # 2× width and height → about 4× the work


def test_predict_returns_one_prediction_per_photo(tmp_path):
    records = fake_records(tmp_path, [PASS, FAIL, FAIL])
    loader = make_loader(records, 64, train=False, batch_size=2)
    y_true, y_pred, p_fail = predict(build_model(pretrained=False), loader, torch.device("cpu"))
    assert list(y_true) == [PASS, FAIL, FAIL]
    assert y_pred.shape == (3,) and set(y_pred) <= {PASS, FAIL}
    assert np.all((p_fail >= 0) & (p_fail <= 1))
