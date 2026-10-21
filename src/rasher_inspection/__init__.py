"""Pass/fail inspection of packaged pork rasher trays from RGB images.

MMA3001 individual project (Dataset 3). Given a photo of a tray on the
conveyor belt, predict PASS (no defect visible) or FAIL (at least one defect
visible), and compare a simple baseline with a transfer-learning CNN.

Modules:

* `rasher_inspection.data` – load the Roboflow COCO annotations and derive pass/fail labels.
* `rasher_inspection.features` – hand-crafted image features (colour histogram, thumbnails, meat fraction).
* `rasher_inspection.evaluate` – metrics for imbalanced data (balanced accuracy, defects caught,
  good trays passed) with 95% confidence intervals.
* `rasher_inspection.cnn` – MobileNetV2 transfer learning: data loading, training with early
  stopping, prediction, FLOP counting and timing.

The experiments themselves are in the `notebooks/` folder (01–04).
"""
