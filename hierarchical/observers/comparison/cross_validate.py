"""
cross_validate.py — block-fold held-out per-trial NLL
=====================================================

Sequence-preserving cross-validation for every registered model. Folds are
CONTIGUOUS segments aligned to block (prior-width) boundaries where possible, so
a whole block is held out at a time and the learning dynamics are tested
out-of-sample rather than interpolated. For each fold we fit on the training
trials and score the held-out per-trial NLL; the model's belief filter always
runs over the full ordered sequence so trial order / learning is preserved.

Cross-validated NLL is the overfitting-proof metric and the one to LEAD with:
it needs no complexity penalty because held-out data penalises over-flexible
models directly. AIC/BIC (in the batch stage) corroborate.

Caveat (documented, same as the house fitters): feedback (true direction) is
available on every trial regardless of the mask, so the belief still 'sees'
held-out feedback. This tests predictive fit of the RESPONSES with order
intact, not a strictly causal forecast.

Writes one JSON per model x subject under results/fits/comparison_cv/.
Resumable (skip existing unless --force).

Usage:
  python -m observers.comparison.cross_validate --subjects 1 2 --folds 5
  python -m observers.comparison.cross_validate --models hb_adaptive switch
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from observers.comparison.registry import (
    build_registry, load_subject, ALL_SUBJECTS, DEFAULT_MODELS)
from observers.helpers.paths import FITS_DIR

OUT_DIR = FITS_DIR / "comparison_cv"


def _result_path(model: str, sid: int) -> Path:
    return OUT_DIR / f"{model}_subject{sid}_cv.json"


def _block_folds(prior_std: np.ndarray, K: int) -> list:
    """Contiguous folds aligned to block boundaries. Blocks are maximal runs of
    a constant prior_std; we group whole blocks into K contiguous folds so a
    held-out fold is one or more complete blocks (never a block split)."""
    ps = np.asarray(prior_std)
    # block id increments whenever prior_std changes
    change = np.concatenate([[True], ps[1:] != ps[:-1]])
    block_id = np.cumsum(change) - 1
    n_blocks = int(block_id.max()) + 1
    if n_blocks < K:              # fall back to plain contiguous trial folds
        idx = np.arange(len(ps))
        return [f for f in np.array_split(idx, K)]
    block_groups = np.array_split(np.arange(n_blocks), K)
    folds = []
    for grp in block_groups:
        mask = np.isin(block_id, grp)
        folds.append(np.flatnonzero(mask))
    return folds


def cv_one(spec, data, sid: int, K: int, maxiter: int) -> dict:
    n = int(data["estimates"].size)
    folds = _block_folds(data["prior_std"], K)
    t0 = time.time()
    fold_nlls, tot = [], 0.0
    for f, test_idx in enumerate(folds):
        test = np.zeros(n, dtype=bool); test[test_idx] = True
        train = ~test
        fr = spec.fit(data, maxiter=maxiter, mask=train)
        # score held-out per-trial NLL using the model's own trial-loglik fn
        ll = spec.trial_logliks(fr.obs, data)
        test_nll = float(-ll[test].sum())
        fold_nlls.append({"fold": f + 1, "n_test": int(test.sum()),
                          "held_out_nll": test_nll,
                          "per_trial": test_nll / max(int(test.sum()), 1)})
        tot += test_nll
    return {
        "model": spec.name, "label": spec.label, "subject": sid, "n_trials": n,
        "k": int(spec.n_params), "folds": K,
        "cv_nll": float(tot), "cv_per_trial": float(tot / n),
        "fold_detail": fold_nlls,
        "maxiter": int(maxiter), "seconds": round(time.time() - t0, 1),
    }


def run(models=None, subjects=None, folds=5, maxiter=400, force=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    reg = build_registry(models)
    done, skipped = [], []
    for sid in subjects:
        data = load_subject(int(sid))
        for name in models:
            path = _result_path(name, int(sid))
            if path.exists() and path.stat().st_size > 0 and not force:
                skipped.append(path.name)
                print(f"skip  {name:12s} subject {sid} (exists)", flush=True)
                continue
            row = cv_one(reg[name], data, int(sid), folds, maxiter)
            json.dump(row, open(path, "w"), indent=2)
            done.append(path.name)
            print(f"done  {name:12s} subject {sid}  CV-NLL={row['cv_nll']:.1f} "
                  f"(per-trial {row['cv_per_trial']:.3f}) ({row['seconds']}s)", flush=True)
    print(f"\nCV complete: {len(done)} run, {len(skipped)} skipped -> {OUT_DIR}", flush=True)
    return done, skipped


def load_all(models=None, subjects=None) -> dict:
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    out = {m: {} for m in models}
    for m in models:
        for sid in subjects:
            p = _result_path(m, int(sid))
            if p.exists() and p.stat().st_size > 0:
                out[m][int(sid)] = json.load(open(p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects, folds=a.folds, maxiter=a.maxiter, force=a.force)
