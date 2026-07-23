"""
split_half_bimodality.py — split-half parameter-stability test of the
bimodality signature (makes the qualitative axis genuinely OUT-OF-SAMPLE).

Motivation
----------
Reproducing bimodality in the FITTED estimate distribution is an in-sample,
architectural check: it shows Basic-Bayes cannot express two clusters at any
parameterization, while the Switch mixture can. It is NOT a generalization test,
because the fit objective already sees the full distribution shape.

This script makes it out-of-sample the way Laquitaine & Gardner did: for each
subject, FIT parameters on one half of the trials and EVALUATE the far-band
bimodality signature on the HELD-OUT half. If the fitted parameters (not just
the architecture) predict where and how much bimodality appears in unseen
trials, the signature generalizes.

Protocol
--------
* Two complementary interleaved halves (even/odd trial index) so each half
  spans the whole session (all prior-width blocks, matched exposure). Fit on
  half A, score signature on half B; then swap. Report both directions.
* Single Nelder-Mead start per half (the CV convention in registry._starts_for:
  held-out evaluation uses one start; multi-start is for the reported point fit).
* Signature metrics reuse the CANONICAL definitions from shape_analysis.py:
    - far-band prior-cluster mass: fraction of held-out response mass within
      +/-18 deg of the 225 deg prior when the stimulus is >=90 deg from 225,
      low coherence (<=0.12), widest prior (80).
    - valley-depth: two-separated-peaks metric per far stimulus direction.
  Predicted signature is computed from the model fitted on the OTHER half,
  pooled over the held-out trials of the target cells.

Models: switch, basic_bayes (the two paper models). Extendable via --models.

Usage:
  PYTHONPATH=. python experiments/rachel/23_split_half_bimodality/split_half_bimodality.py --subjects 1 2
  PYTHONPATH=. python experiments/rachel/23_split_half_bimodality/split_half_bimodality.py           # all 12
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np

from observers.comparison.registry import build_registry, load_subject, ALL_SUBJECTS
# reuse the canonical metrics verbatim
from observers.comparison.shape_analysis import (
    _circ_dist, _hist_360, _prior_cluster_mass, _valley_depth, PRIOR_MEAN)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
MODELS = ["switch", "basic_bayes"]


def _subset(data, idx):
    """Slice the trial-parallel data dict to trial indices `idx` (order kept)."""
    return {k: (np.asarray(v)[idx] if hasattr(v, "__len__") and len(np.atleast_1d(v)) == len(data["estimates"]) else v)
            for k, v in data.items()}


def _far_sel(data):
    """Boolean over trials: far-from-prior, low-coh, widest-prior regime."""
    d, c, ps = data["motion_direction"], data["motion_coherence"], data["prior_std"]
    return (_circ_dist(d, PRIOR_MEAN) >= 90) & (c <= 0.12) & (ps == 80)


def _signature_on(spec, obs, data, sel):
    """Predicted far-band mass + valley-depth on the trials `sel`, from a fitted obs."""
    if sel.sum() < 15:
        return None
    pred = spec.predict_distributions(obs, _subset(data, np.where(sel)[0]))  # (n_sel, 360)
    pooled = pred.mean(axis=0)
    mass = _prior_cluster_mass(pooled)
    # valley per single far stim direction, averaged
    d = data["motion_direction"][sel]
    vals = []
    for stim in sorted(set(int(x) for x in d)):
        m2 = d == stim
        if m2.sum() < 8:
            continue
        vals.append(_valley_depth(pred[m2].mean(axis=0), stim))
    valley = float(np.mean(vals)) if vals else None
    return {"far_mass_pred": mass, "valley_pred": valley, "n_far": int(sel.sum())}


def _observed_signature(data, sel):
    if sel.sum() < 15:
        return None
    e = data["estimates"][sel]
    mass = _prior_cluster_mass(_hist_360(e))
    d = data["motion_direction"][sel]
    vals = []
    for stim in sorted(set(int(x) for x in d)):
        m2 = d == stim
        if m2.sum() < 8:
            continue
        vals.append(_valley_depth(_hist_360(e[m2]), stim))
    valley = float(np.mean(vals)) if vals else None
    return {"far_mass_obs": mass, "valley_obs": valley, "n_far": int(sel.sum())}


def _one_subject(sid, models, maxiter):
    t0 = time.time()
    specs = build_registry(models)
    data = load_subject(int(sid))
    n = data["estimates"].size
    idx = np.arange(n)
    halves = {"A": idx[0::2], "B": idx[1::2]}   # interleaved even/odd
    out = {"subject": int(sid), "n_trials": int(n), "models": models, "directions": {}}

    for train_key, test_key in [("A", "B"), ("B", "A")]:
        train_idx, test_idx = halves[train_key], halves[test_key]
        train_mask = np.zeros(n, bool); train_mask[train_idx] = True
        test_data = data                      # full arrays; select via sel below
        test_far = _far_sel(data) & _mask_bool(test_idx, n)
        rec = {"train": train_key, "test": test_key,
               "observed": _observed_signature(data, test_far)}
        for name in models:
            spec = specs[name]
            res = spec.fit(data, maxiter=maxiter, mask=train_mask)  # fit on TRAIN half
            obs = res.obs
            sig = _signature_on(spec, obs, data, test_far)          # score on TEST half
            rec[name] = {**(sig or {}),
                         "train_nll": float(res.nll),
                         "params": _params_of(obs, name)}
        out["directions"][f"{train_key}->{test_key}"] = rec

    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"split_half_subject{sid}.json"), "w"), indent=2)
    return out


def _mask_bool(idx, n):
    b = np.zeros(n, bool); b[idx] = True; return b


def _params_of(obs, name):
    """Extract the fitted param dict for reporting (best-effort, JSON-safe)."""
    p = {}
    for attr in ("k_like", "k_prior", "k_motor", "p_random", "prior_mean", "k_cardinal"):
        if hasattr(obs, attr):
            v = getattr(obs, attr)
            if isinstance(v, dict):
                p[attr] = {str(k): float(x) for k, x in v.items()}
            elif v is not None:
                try: p[attr] = float(v)
                except (TypeError, ValueError): pass
    return p


def run(models=None, subjects=None, maxiter=400, workers=None):
    """Serial by default. The sandbox blocks ProcessPoolExecutor (semaphore
    syscall), and the fits are CPU-bound (GIL-limited) so threads would not
    parallelize the compute anyway; serial is the honest execution model here.
    `workers` is accepted for CLI compatibility but ignored."""
    models = models or MODELS
    subjects = subjects or ALL_SUBJECTS
    os.makedirs(OUT, exist_ok=True)
    results = {}
    print(f"split-half bimodality: {len(subjects)} subjects x {len(models)} models "
          f"x 2 directions (serial)", flush=True)
    for s in subjects:
        try:
            r = _one_subject(s, models, maxiter); results[s] = r
            print(f"  subject {s} done ({r['seconds']}s)", flush=True)
        except Exception as e:
            print(f"  subject {s} FAILED: {type(e).__name__}: {e}", flush=True)
    json.dump(results, open(os.path.join(OUT, "split_half_summary.json"), "w"), indent=2)
    print(f"done -> {OUT}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects, maxiter=a.maxiter, workers=a.workers)
