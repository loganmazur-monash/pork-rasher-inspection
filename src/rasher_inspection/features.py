"""Hand-crafted image features.

These are simple, explainable numbers computed from a photo's pixels.
``thumbnail`` is used to look for near-duplicate photos and
``meat_fraction`` to check for a shortcut (both in notebook 01);
``colour_histogram`` gives the input features of the baseline model (notebook 02).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_hsv(path: str | Path, downscale: int = 4) -> np.ndarray:
    """Load a photo as an HSV array scaled to [0, 1].

    HSV (hue, saturation, value) separates *colour* (hue) from *brightness*
    (value), which makes it easier to pick out red meat on a dark belt.

    Args:
        path: JPEG file.
        downscale: Integer shrink factor (4 turns 720x540 into 180x135),
            which is plenty for colour statistics and much faster.

    Returns:
        Float array of shape (height, width, 3) with values in [0, 1].
    """
    with Image.open(path) as img:
        hsv = img.convert("RGB").convert("HSV").reduce(downscale)  # RGB first: handles greyscale/RGBA too
    return np.asarray(hsv, dtype=float) / 255.0


def thumbnail(path: str | Path, size: tuple[int, int] = (48, 36)) -> np.ndarray:
    """A tiny, brightness-normalised greyscale version of a photo, as a flat vector.

    Shrinking removes fine detail, and normalising (subtract the mean, divide
    by the standard deviation) removes overall brightness differences such as
    Roboflow's exposure augmentation. Two copies of the same photo therefore
    give almost identical vectors, while different photos do not.

    Args:
        path: JPEG file.
        size: Thumbnail (width, height) in pixels.

    Returns:
        1-D float array of length width * height.
    """
    with Image.open(path) as img:
        grey = np.asarray(img.convert("L").resize(size), dtype=float)
    return ((grey - grey.mean()) / (grey.std() + 1e-6)).ravel()


def colour_histogram(path: str | Path, bins: int = 16) -> np.ndarray:
    """Colour-histogram feature vector used by the baseline model.

    For each HSV channel (hue, saturation, value) the pixel values are sorted
    into ``bins`` equal-width bins, and each bin holds the *fraction* of
    pixels that fall into it. The three histograms are joined into one vector.
    The vector describes which colours appear and how much of each, but not
    *where* they are in the photo.

    Args:
        path: JPEG file.
        bins: Number of bins per channel.

    Returns:
        1-D array of length ``3 * bins``; each third sums to 1.
    """
    hsv = load_hsv(path)
    return np.concatenate([
        np.histogram(hsv[..., c], bins=bins, range=(0.0, 1.0))[0] / hsv[..., c].size
        for c in range(3)
    ])


def meat_fraction(path: str | Path) -> float:
    """Fraction of the photo covered by red/pink, reasonably bright pixels.

    A rough measure of how much meat is visible. The thresholds were chosen
    by eye on sample photos: the belt is dark (low value) and the plastic tray
    is colourless (low saturation), so neither is counted.

    Returns:
        A number between 0 and 1.
    """
    hsv = load_hsv(path)
    hue_deg = hsv[..., 0] * 360
    is_red = (hue_deg < 25) | (hue_deg > 330)
    is_meat = is_red & (hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.3)
    return float(is_meat.mean())
