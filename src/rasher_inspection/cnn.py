"""Transfer-learning CNN (MobileNetV2) for pass/fail tray inspection.

**Transfer learning** means starting from a network that has already learned
general visual features (edges, textures, shapes) on ImageNet – 1.2 million
everyday photos – and re-training it briefly on our tray photos. Only the
final layer is replaced: instead of 1,000 ImageNet classes it outputs 2
scores, one for PASS and one for FAIL.

**Why MobileNetV2?** It was designed for phones and embedded devices: about
3.5 million parameters, so it is small, fast and trains quickly – a good fit
for a camera on a production line and for training on a laptop.

This module holds the reusable building blocks; the experiment itself is in
``notebooks/03_cnn.ipynb``.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import FAIL, PASS, ImageRecord
from .evaluate import balanced_accuracy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""The pre-trained network expects inputs normalised with ImageNet statistics."""


def best_device() -> torch.device:
    """Pick the fastest available hardware: NVIDIA GPU, Apple-silicon GPU, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def input_size(width: int) -> tuple[int, int]:
    """(height, width) of the network input; keeps the photos' 4:3 shape (720×540)."""
    return round(width * 3 / 4), width


def make_transform(width: int, train: bool) -> transforms.Compose:
    """Turn a PIL photo into a normalised tensor of shape (3, height, width).

    For training, random flips and small colour changes are added
    (*data augmentation*): a flipped or slightly brighter tray has the same
    label, so the model learns that these differences do not matter and
    overfits less.

    Args:
        width: Input width in pixels (height follows from the 4:3 ratio).
        train: Whether to add random augmentation.
    """
    steps = [transforms.Resize(input_size(width))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class TrayDataset(Dataset):
    """PyTorch dataset: returns ``(image_tensor, label)`` for each ImageRecord."""

    def __init__(self, records: list[ImageRecord], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        record = self.records[i]
        with Image.open(record.path) as img:
            image = self.transform(img.convert("RGB"))
        return image, record.label


def build_model(pretrained: bool = True) -> nn.Module:
    """MobileNetV2 with its last layer replaced by a 2-output layer (PASS, FAIL).

    Args:
        pretrained: Start from ImageNet weights (downloaded once, ~14 MB).
            ``False`` gives random weights – only used in tests.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model


def class_weights(labels) -> torch.Tensor:
    """Loss weight per class, inversely proportional to how common the class is.

    With ~20% PASS photos, a PASS mistake gets ~4× the weight of a FAIL
    mistake, so both classes contribute equally to the loss (the same idea as
    ``class_weight="balanced"`` in scikit-learn).

    Returns:
        Float tensor ``[weight_PASS, weight_FAIL]``.
    """
    labels = np.asarray(labels)
    counts = np.array([np.sum(labels == PASS), np.sum(labels == FAIL)])
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """One pass over the training data. Returns the mean training loss."""
    model.train()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    """Run the model on every photo in ``loader``.

    Returns:
        ``(y_true, y_pred, p_fail)`` as NumPy arrays, where ``p_fail`` is the
        predicted probability of FAIL.
    """
    model.eval()
    y_true, p_fail = [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)[:, FAIL]
        p_fail.append(probs.cpu().numpy())
        y_true.append(labels.numpy())
    y_true, p_fail = np.concatenate(y_true), np.concatenate(p_fail)
    return y_true, (p_fail >= 0.5).astype(int), p_fail


def fit(model, train_loader, valid_loader, epochs: int, lr: float, weights: torch.Tensor,
        device, log=print) -> tuple[nn.Module, list[dict]]:
    """Train with Adam and keep the epoch with the lowest validation loss.

    Validation *loss* is used to pick the epoch rather than validation balanced
    accuracy, because with only 14 PASS photos in the validation set the
    accuracy jumps around a lot between epochs, while the loss changes smoothly.

    Returns:
        ``(model with the best weights, history)`` where history has one dict per
        epoch: ``epoch, train_loss, valid_loss, valid_balanced_accuracy, seconds``.
    """
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, history = float("inf"), None, []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss = evaluate_loss(model, valid_loader, loss_fn, device)
        y_true, y_pred, _ = predict(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
               "valid_balanced_accuracy": balanced_accuracy(y_true, y_pred),
               "seconds": time.perf_counter() - start}
        history.append(row)
        log(f"epoch {epoch:>2}: train loss {train_loss:.3f}  valid loss {valid_loss:.3f}  "
            f"valid bal. acc {row['valid_balanced_accuracy']:.1%}  ({row['seconds']:.0f} s)")
        if valid_loss < best_loss:
            best_loss, best_state = valid_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    """Mean loss over a data loader, without updating the model."""
    model.eval()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += loss_fn(model(images), labels).item() * len(labels)
        n += len(labels)
    return total / n


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers (weights and biases) in the model."""
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, width: int) -> int:
    """Floating-point operations for ONE forward pass (one photo) at the given width.

    Uses PyTorch's built-in FLOP counter, which counts a multiply-add as 2 FLOPs.
    """
    from torch.utils.flop_counter import FlopCounterMode

    height, width = input_size(width)
    model = copy.deepcopy(model).cpu().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 3, height, width))
    return int(counter.get_total_flops())


def _sync(device) -> None:
    # GPUs run asynchronously; wait for them so the timing is real.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_inference(model: nn.Module, width: int, device, repeats: int = 30) -> float:
    """Average milliseconds to classify ONE photo (batch size 1) on ``device``.

    This excludes loading the JPEG, so it measures the network itself.
    A few warm-up runs come first because the first calls are always slower.
    """
    height, width = input_size(width)
    model = model.to(device).eval()
    x = torch.zeros(1, 3, height, width, device=device)
    for _ in range(5):
        model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    _sync(device)
    return 1000 * (time.perf_counter() - start) / repeats


def make_loader(records, width: int, train: bool, batch_size: int = 32) -> DataLoader:
    """DataLoader that shuffles only when training."""
    return DataLoader(TrayDataset(records, make_transform(width, train)),
                      batch_size=batch_size, shuffle=train, num_workers=0)
