# AI Use Log

Running record of AI assistance, used to write the AI reflection section of the report.
Add an entry whenever AI helps with something.

| Date | Tool | What I asked / what it did | How I checked it | What I changed or decided myself |
|------|------|----------------------------|------------------|----------------------------------|
| 2026-10-07 | Claude | Helped choose a project direction (pass/fail classification), explained the brief, and generated this repo skeleton (README, .gitignore, requirements, LICENSE). | Read through each file; checked structure against the brief's GitHub requirements (README, HTML docs, license, source, tests). | |
| 2026-10-21 | Claude | Read the annotation files and sample photos; found the class imbalance, rare defect types and small test set. Wrote `data.py`, `features.py`, `tests/test_data.py` and notebook 01. Claude first claimed the official test set leaked (based on frame numbers in file names), then disproved this itself with a pixel-similarity check – the numbers don't indicate time order. | | |
| 2026-10-21 | Claude | Wrote `evaluate.py` (balanced accuracy, Wilson and bootstrap confidence intervals), `colour_histogram` in `features.py`, `tests/test_evaluate.py` and notebook 02 (always-FAIL and colour-histogram + logistic-regression baselines, C chosen by cross-validation grouped by original photo). | | |
