# Cross-validation

*Date: 2026-07-17*

*Tags: type:validation | claim:switching-vs-integration | status:supporting | presentation:backup*

Block-fold cross-validation: held-out NLL results, all-12 and interim CSVs, verdict figure, and switch-family CV code. The canonical per-subject CV fit JSONs live under `results/fits/comparison_cv/<model>/subject<N>_cv.json` (written by `observers/comparison/cross_validate.py`).

## Files

- `cv_holdout_nll_all12.csv`
- `cv_holdout_nll_interim.csv`
- `cv_results.md`
- `cv_switch_family.py`
- `cv_verdict.png`
- `results/` — 36 files
