"""Metrics for comparing pass/fail inspection models.

Because about 80% of photos are FAIL, plain accuracy is misleading (always
answering FAIL already scores ~85%). Every model is therefore described by:

* **defects caught** – share of FAIL photos predicted FAIL (recall of FAIL).
  A missed defect means a faulty tray reaches a customer.
* **good trays passed** – share of PASS photos predicted PASS (recall of PASS).
  A false reject means a good tray is wasted or re-inspected.
* **balanced accuracy** – the average of the two, so both classes count equally.
  Always answering FAIL scores exactly 50%.

Each value comes with a 95% confidence interval, because the test set is small
(80 photos, only 12 PASS).
"""

from __future__ import annotations

import numpy as np

from .data import FAIL, PASS


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score method).

    Better behaved than "p ± 1.96·√(p(1−p)/n)" when n is small or p is
    close to 0 or 1, both of which happen here.

    Args:
        successes: Number of "successes" (e.g. correctly classified photos).
        n: Number of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        ``(lower, upper)`` as fractions between 0 and 1.

    Raises:
        ValueError: If ``n`` is not positive or ``successes`` is outside 0..n.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n > 0, got {successes}/{n}")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def confusion(y_true, y_pred) -> dict[str, int]:
    """Count the four outcomes, treating FAIL (a defect) as the "positive" class.

    Returns:
        ``{"caught": ..., "missed": ..., "passed": ..., "rejected": ...}`` where
        caught = FAIL→FAIL, missed = FAIL→PASS, passed = PASS→PASS,
        rejected = PASS→FAIL.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "caught": int(np.sum((y_true == FAIL) & (y_pred == FAIL))),
        "missed": int(np.sum((y_true == FAIL) & (y_pred == PASS))),
        "passed": int(np.sum((y_true == PASS) & (y_pred == PASS))),
        "rejected": int(np.sum((y_true == PASS) & (y_pred == FAIL))),
    }


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean of defects caught and good trays passed (0.5 = no better than guessing)."""
    c = confusion(y_true, y_pred)
    return 0.5 * (c["caught"] / (c["caught"] + c["missed"]) + c["passed"] / (c["passed"] + c["rejected"]))


def bootstrap_interval(y_true, y_pred, metric=balanced_accuracy, n_resamples: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for any metric by bootstrap resampling.

    The test photos are resampled with replacement many times, the metric is
    recomputed each time, and the middle 95% of the results is returned.
    Resamples that happen to contain only one class are skipped.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            values.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(y_true, y_pred) -> dict:
    """All headline metrics for one model on one set of photos, with 95% intervals.

    Returns:
        A dictionary with the confusion counts and, for ``accuracy``,
        ``defects_caught``, ``good_passed`` and ``balanced_accuracy``, a
        ``{"value": ..., "ci": (low, high)}`` entry.
    """
    c = confusion(y_true, y_pred)
    n_fail, n_pass = c["caught"] + c["missed"], c["passed"] + c["rejected"]
    n_correct = c["caught"] + c["passed"]

    def proportion(k, n):
        return {"value": k / n, "ci": wilson_interval(k, n)}

    return {
        **c,
        "accuracy": proportion(n_correct, n_fail + n_pass),
        "defects_caught": proportion(c["caught"], n_fail),
        "good_passed": proportion(c["passed"], n_pass),
        "balanced_accuracy": {"value": balanced_accuracy(y_true, y_pred),
                              "ci": bootstrap_interval(y_true, y_pred)},
    }


def format_summary(name: str, s: dict) -> str:
    """One readable line per metric, e.g. for printing in a notebook."""
    lines = [f"{name}   (caught {s['caught']}, missed {s['missed']}, "
             f"passed {s['passed']}, rejected {s['rejected']})"]
    for key, label in [("balanced_accuracy", "Balanced accuracy"), ("defects_caught", "Defects caught"),
                       ("good_passed", "Good trays passed"), ("accuracy", "Accuracy")]:
        v, (lo, hi) = s[key]["value"], s[key]["ci"]
        lines.append(f"  {label:<18} {v:6.1%}   (95% CI {lo:5.1%} – {hi:5.1%})")
    return "\n".join(lines)
