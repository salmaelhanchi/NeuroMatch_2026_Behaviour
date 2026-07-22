"""
fit_batch.py — resumable batch fit of all registered models x subjects
=======================================================================

Fits every model in the registry to every requested subject, writing **one JSON
per model x subject** under ``results/fits/comparison/<model>/subject<N>.json``.
Resumable: an existing
(non-empty) result file is skipped unless ``--force``, so an interrupted run
picks up where it left off and a single slow cell never loses finished work.

All models are fit with the SAME protocol (same optimiser budget, multi-start
where the fitter provides it) and each row records ``start_spread`` (NLL range
across starts) as a convergence diagnostic, so a win can never be quietly
attributed to one model being under-converged.

Because every model in the registry is on the 360-deg grid, the reported NLLs
are directly comparable (no grid-resolution artifact to correct).

Usage:
  python -m observers.comparison.fit_batch                     # default models, all 12 subjects
  python -m observers.comparison.fit_batch --models hb_adaptive switch --subjects 1 2
  python -m observers.comparison.fit_batch --force             # refit even if a JSON exists
  python -m observers.comparison.fit_batch --maxiter 600
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from observers.comparison.registry import (
    build_registry, load_subject, ALL_SUBJECTS, DEFAULT_MODELS)
from observers.helpers.paths import FITS_DIR

OUT_DIR = FITS_DIR / "comparison"


def _result_path(model: str, sid: int) -> Path:
    return OUT_DIR / model / f"subject{sid}.json"


def _observer_params(obs) -> dict:
    """Best-effort extraction of a model's fitted parameters for the record.
    Covers every registered model's constructor attributes, including HB-Salma's
    (rho / sensory_kappas / motor_kappa / lapse), so each model's rebuild() can
    reconstruct the observer from the stored dict."""
    p = {}
    scalar_attrs = ("k_motor", "p_random", "lam", "alpha",
                    "rho", "motor_kappa", "lapse")           # + Salma scalars
    dict_attrs = ("k_like", "k_prior")
    for attr in scalar_attrs:
        if hasattr(obs, attr):
            try: p[attr] = float(getattr(obs, attr))
            except (TypeError, ValueError): pass
    for attr in dict_attrs:
        if hasattr(obs, attr) and isinstance(getattr(obs, attr), dict):
            p[attr] = {str(k): float(x) for k, x in getattr(obs, attr).items()}
    # Salma's per-coherence sensory concentrations (tuple, ascending coherence)
    if hasattr(obs, "sensory_kappas"):
        try: p["sensory_kappas"] = [float(x) for x in obs.sensory_kappas]
        except (TypeError, ValueError): pass
    return p


def _theta_list(x):
    """Serialize a fitter's raw parameter vector, tolerating None (some fitters
    return x=None when they don't expose a flat vector)."""
    if x is None:
        return None
    try:
        return [float(v) for v in np.asarray(x, dtype=float).ravel()]
    except (TypeError, ValueError):
        return None


def fit_one(spec, data, sid: int, maxiter: int) -> dict:
    n = int(data["estimates"].size)
    t0 = time.time()
    fr = spec.fit(data, maxiter=maxiter)
    aic = 2 * fr.n_params + 2 * fr.nll
    bic = fr.n_params * float(np.log(n)) + 2 * fr.nll
    return {
        "model": spec.name, "label": spec.label, "subject": sid, "n_trials": n,
        "nll": float(fr.nll), "aic": float(aic), "bic": float(bic),
        "k": int(fr.n_params), "learns": bool(spec.learns), "grid_deg": int(spec.grid_deg),
        "start_spread": float(fr.start_spread),
        "params": _observer_params(fr.obs),
        "theta": _theta_list(fr.x),  # raw fitted vector (rebuild fallback)
        "convergence": getattr(fr.obs, "_fit_info", None),  # scipy status
        "maxiter": int(maxiter), "seconds": round(time.time() - t0, 1),
    }


def run(models=None, subjects=None, maxiter=400, force=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    reg = build_registry(models)
    done, skipped = [], []
    for sid in subjects:
        data = load_subject(int(sid))
        for name in models:
            path = _result_path(name, int(sid))
            # Resume-skip only a file fit at >= the requested budget. A stale
            # lower-maxiter file (e.g. a maxiter=15 smoke result) must NOT count
            # as done, or the resume silently keeps junk fits.
            if path.exists() and path.stat().st_size > 0 and not force:
                try:
                    prev = json.load(open(path)).get("maxiter", 0)
                except Exception:
                    prev = -1
                if prev >= maxiter:
                    skipped.append(path.name)
                    print(f"skip  {name:12s} subject {sid} "
                          f"(exists, maxiter={prev})", flush=True)
                    continue
                print(f"refit {name:12s} subject {sid} "
                      f"(stale maxiter={prev} < {maxiter})", flush=True)
            row = fit_one(reg[name], data, int(sid), maxiter)
            path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(row, open(path, "w"), indent=2)
            done.append(path.name)
            print(f"done  {name:12s} subject {sid}  NLL={row['nll']:.1f} "
                  f"AIC={row['aic']:.1f} BIC={row['bic']:.1f} k={row['k']} "
                  f"spread={row['start_spread']:.2f} ({row['seconds']}s)", flush=True)
    print(f"\nbatch complete: {len(done)} fit, {len(skipped)} skipped -> {OUT_DIR}", flush=True)
    return done, skipped


def load_all(models=None, subjects=None) -> dict:
    """Collect written results into {model: {subject: row}} for downstream stages."""
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
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects, maxiter=a.maxiter, force=a.force)
