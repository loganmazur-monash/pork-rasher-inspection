# Pork Rasher Packaging Inspection

Automated pass/fail inspection of packaged pork rasher trays from RGB images.

MMA3001 – Numerical Methods and Machine Learning, Monash University, 2026 (individual project, Dataset 3).

## Problem

Food-packaging lines rely on manual visual inspection to catch defective trays
(loose or twisted meat, unsealed packaging, wrinkles). This project builds and
compares computer-vision classifiers that predict whether a tray image should
**pass** or **fail** inspection.

*(Problem statement, inputs/outputs and results will be expanded as the project develops.)*

## Repository structure

```
pork-rasher-inspection/
├── data/                 # dataset goes here (not committed – see below)
├── docs/                 # generated HTML documentation (pdoc)
├── notebooks/            # exploration and training notebooks
├── results/              # figures used in the report
├── src/rasher_inspection/  # project source code
├── tests/                # pytest tests
├── AI_USE_LOG.md         # running log of AI assistance
├── requirements.txt
└── LICENSE
```

## Setup

```bash
git clone https://github.com/<your-username>/pork-rasher-inspection.git
cd pork-rasher-inspection
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The dataset (MMA3001 Project Dataset 3, Roboflow export, Version 4) is **not**
stored in this repository.

1. Download it from the link on the MMA3001 "Project Datasets" slide (Moodle).
2. Choose **COCO** format and download the zip.
3. Unzip it into `data/raw/` so you have `data/raw/train/`, `data/raw/valid/` and `data/raw/test/`.

Source: [Pork Rasher Error(Packaging), Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging), v4 (2025-02-09), licensed **CC BY 4.0**.
3,549 photos at 720×540 (train 3,349 = about 1,100 originals × 3 exposure copies; valid 120; test 80), with bounding boxes for five defect classes:
`packaging-error`, `unsealed`, `twisted-meat`, `loose-meat`, `wrinkle`.

## Usage

Run the notebooks in order from the `notebooks/` folder:

| Notebook | What it does |
|---|---|
| `01_explore_data.ipynb` | Class balance, defect types, near-duplicate check of the official split, defect sizes, shortcut check, test-set uncertainty |
| `02_baselines.ipynb` | "Always FAIL" and colour-histogram + logistic-regression baselines, with grouped cross-validation and 95% confidence intervals |

Run the tests from the repository root:

```bash
pytest
```

## AI use

AI assistance is disclosed in [`AI_USE_LOG.md`](AI_USE_LOG.md) and summarised in the project report.

## License

MIT – see [`LICENSE`](LICENSE).
