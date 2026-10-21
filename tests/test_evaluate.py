"""Tests for rasher_inspection.evaluate."""

import numpy as np
import pytest

from rasher_inspection.data import FAIL, PASS
from rasher_inspection.evaluate import (
    balanced_accuracy,
    bootstrap_interval,
    confusion,
    summarise,
    wilson_interval,
)

Y_TRUE = [FAIL, FAIL, FAIL, FAIL, PASS, PASS]
Y_PRED = [FAIL, FAIL, FAIL, PASS, PASS, FAIL]


def test_confusion_counts_each_outcome():
    assert confusion(Y_TRUE, Y_PRED) == {"caught": 3, "missed": 1, "passed": 1, "rejected": 1}


def test_balanced_accuracy_averages_both_classes():
    assert balanced_accuracy(Y_TRUE, Y_PRED) == pytest.approx((3 / 4 + 1 / 2) / 2)


def test_always_fail_has_balanced_accuracy_one_half():
    assert balanced_accuracy(Y_TRUE, [FAIL] * len(Y_TRUE)) == pytest.approx(0.5)


def test_wilson_interval_contains_estimate_and_stays_in_range():
    lo, hi = wilson_interval(9, 12)
    assert 0 <= lo < 9 / 12 < hi <= 1
    lo, hi = wilson_interval(12, 12)       # perfect score: upper bound is 1, lower bound below 1
    assert hi == pytest.approx(1.0) and lo < 1


def test_wilson_interval_narrows_with_more_data():
    small = np.diff(wilson_interval(9, 12))[0]
    large = np.diff(wilson_interval(90, 120))[0]
    assert large < small


def test_wilson_interval_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_interval(13, 12)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, 1 - y_true)
    lo, hi = bootstrap_interval(y_true, y_pred)
    assert lo <= balanced_accuracy(y_true, y_pred) <= hi


def test_summarise_reports_all_metrics():
    s = summarise(Y_TRUE, Y_PRED)
    assert s["defects_caught"]["value"] == pytest.approx(0.75)
    assert s["good_passed"]["value"] == pytest.approx(0.5)
    assert s["accuracy"]["value"] == pytest.approx(4 / 6)
