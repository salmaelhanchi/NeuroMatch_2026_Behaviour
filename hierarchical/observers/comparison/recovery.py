"""
recovery.py — parameter recovery + model-recovery confusion matrix
==================================================================

Two identifiability checks that license the whole comparison:

1. PARAMETER RECOVERY (per model): simulate data from known parameters, refit,
   and check the fitted parameters return. For HB-Adaptive this also maps the
   kappa-alpha ridge (both are learned latents that trade off from a single
   feedback draw), so any reported alpha or kappa is interpretable.

2. MODEL RECOVERY (confusion matrix): simulate from each model, fit ALL models,
   and record which model AIC/BIC selects. The diagonal must dominate — if the
   metrics cannot recover the true generator, the model comparison on real data
   is meaningless. This is the check that makes Table / Panel C trustworthy.

Both iterate the registry, so adding a model extends the confusion matrix
automatically. Outputs saved under results/fits/comparison_recovery/ for the
supplementary figure.

Usage:
  python -m observers.comparison.recovery --n-sim 2 --maxiter 300
  python -m observers.comparison.recovery --models hb_adaptive switch
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from observers.comparison.registry import build_registry, DEFAULT_MODELS
from observers.helpers.dataset import make_synthetic_design
from observers.helpers.paths import FITS_DIR

OUT_DIR = FITS_DIR / "comparison_recovery"


# ground-truth params to simulate from, per model
TRUTHS = {
    "hb_adaptive": dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                        k_motor=40.0, p_random=0.02, lam=0.10),
    "switch":      dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                        k_prior={"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7},
                        k_motor=40.0, p_random=0.02),
    "hb_rachel":   dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, alpha=0.6,
                        k_motor=30.0, p_random=0.03, lam=0.05),
    "basic_bayes": dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                        k_prior={"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7},
                        k_motor=40.0, p_random=0.02),
    "hb_salma":    dict(rho=0.85, sensory_kappas=(1.5, 3.0, 8.0),
                        motor_kappa=40.0, lapse=0.02),
}


def _make_generator(name):
    if name == "hb_adaptive":
        from observers.models.hb_adaptive_confidence import HBAdaptiveConfidenceObserver
        return HBAdaptiveConfidenceObserver(**TRUTHS[name])
    if name == "switch":
        from observers.models.switching_observer import SwitchingObserver
        return SwitchingObserver(**TRUTHS[name])
    if name == "hb_rachel":
        from observers.models.hb_integration import HBIntegrationObserver
        return HBIntegrationObserver(**TRUTHS[name])
    if name == "basic_bayes":
        from observers.models.basic_bayesian import BasicBayesianObserver
        return BasicBayesianObserver(**TRUTHS[name])
    if name == "hb_salma":
        from observers.models.hb_salma import HBSalmaObserver
        return HBSalmaObserver(**TRUTHS[name])
    raise KeyError(name)


def _aic(nll, k):  return 2 * k + 2 * nll
def _bic(nll, k, n): return k * float(np.log(n)) + 2 * nll


def parameter_recovery(specs, n_sim=3, maxiter=400, trials_per_block=200):
    """Simulate from each model's truth, refit the SAME model, report recovery."""
    out = {}
    design = make_synthetic_design(trials_per_block=trials_per_block, seed=1)
    # ensure prior_std present for switch simulate
    if "prior_std" not in design:
        pass
    for name, spec in specs.items():
        gen = _make_generator(name)
        recs = []
        for seed in range(1, n_sim + 1):
            data = spec.simulate(gen, design, seed)
            fr = spec.fit(data, maxiter=maxiter)
            recs.append(_extract_params(fr.obs))
        out[name] = {"truth": _flatten_truth(name), "recovered": recs}
        print(f"[param recovery] {name}: {n_sim} sims fit", flush=True)
    return out


def _extract_params(obs):
    # reuse the batch extractor so it covers every model (incl. Salma's rho /
    # sensory_kappas / motor_kappa / lapse) — one source of truth for params.
    from observers.comparison.fit_batch import _observer_params
    return _observer_params(obs)


def _flatten_truth(name):
    t = dict(TRUTHS[name])
    if "k_like" in t:
        t["k_like"] = {str(k): v for k, v in t["k_like"].items()}
    if "sensory_kappas" in t:
        t["sensory_kappas"] = [float(x) for x in t["sensory_kappas"]]
    return t


def model_recovery(specs, n_sim=2, maxiter=400, trials_per_block=180):
    """Confusion matrix: simulate from each model, fit ALL, tally AIC/BIC winner."""
    names = list(specs.keys())
    design = make_synthetic_design(trials_per_block=trials_per_block, seed=7)
    conf_aic = {g: {f: 0 for f in names} for g in names}
    conf_bic = {g: {f: 0 for f in names} for g in names}
    detail = []
    for gen_name in names:
        gen = _make_generator(gen_name)
        for seed in range(1, n_sim + 1):
            data = specs[gen_name].simulate(gen, design, seed)
            n = len(np.asarray(data["estimates"]))
            scores = {}
            for fit_name, spec in specs.items():
                fr = spec.fit(data, maxiter=maxiter)
                scores[fit_name] = {"nll": fr.nll, "k": fr.n_params,
                                    "aic": _aic(fr.nll, fr.n_params),
                                    "bic": _bic(fr.nll, fr.n_params, n)}
            win_aic = min(scores, key=lambda m: scores[m]["aic"])
            win_bic = min(scores, key=lambda m: scores[m]["bic"])
            conf_aic[gen_name][win_aic] += 1
            conf_bic[gen_name][win_bic] += 1
            detail.append({"generator": gen_name, "seed": seed,
                           "scores": scores, "win_aic": win_aic, "win_bic": win_bic})
            print(f"[model recovery] gen={gen_name} seed={seed} "
                  f"-> AIC picks {win_aic}, BIC picks {win_bic}", flush=True)
    return {"names": names, "confusion_aic": conf_aic,
            "confusion_bic": conf_bic, "detail": detail}


def run(models=None, n_sim=2, maxiter=400):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or DEFAULT_MODELS
    specs = build_registry(models)
    pr = parameter_recovery(specs, n_sim=n_sim, maxiter=maxiter)
    json.dump(pr, open(OUT_DIR / "parameter_recovery.json", "w"), indent=2)
    mr = model_recovery(specs, n_sim=n_sim, maxiter=maxiter)
    json.dump(mr, open(OUT_DIR / "model_recovery.json", "w"), indent=2)
    # print the confusion matrix
    print("\n=== model-recovery confusion (AIC) — rows=generator, cols=selected ===")
    names = mr["names"]
    print("            " + "  ".join(f"{n[:10]:>10s}" for n in names))
    for g in names:
        print(f"{g[:10]:>10s}  " + "  ".join(f"{mr['confusion_aic'][g][f]:10d}" for f in names))
    print(f"\nrecovery complete -> {OUT_DIR}", flush=True)
    return pr, mr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-sim", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=400)
    a = ap.parse_args()
    run(models=a.models, n_sim=a.n_sim, maxiter=a.maxiter)
