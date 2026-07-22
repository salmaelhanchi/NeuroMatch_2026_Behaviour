"""
reliability_mixture_fit.py
==========================

Maximum-likelihood fitter for :class:`ReliabilityMixtureObserver` (Romi's
reliability-mixture model). Mirrors the house fitter contract so the registry
adapter is uniform with every other model:

    N_PARAMS, pack/unpack, _trial_logliks(obs, data), nll_masked(obs, data, mask),
    fit(data, x0=, maxiter=, mask=), _simulate(obs, design, seed)

The 10 parameters (transformed to an unconstrained vector for Nelder-Mead):
    [log k_like(.06), log k_like(.12), log k_like(.24),
     log k_prior(80), log k_prior(40), log k_prior(20), log k_prior(10),
     logit alpha, log k_motor, logit(lapse / LAPSE_MAX)]

The lapse cap (LAPSE_MAX = 0.3) and the sigmoid/log transforms follow Romi's
original ``hb_mixture_nll`` encoding, so a fit here explores the same parameter
space her notebooks did.

Cross-validation note (same fork as the other learning models): a CV ``mask``
selects which trials are SCORED, but the reliance filter always runs over the
full ordered sequence with all feedback visible, so trial order / learning is
preserved and only the response prediction is held out.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from observers.models.reliability_mixture import ReliabilityMixtureObserver
from observers.fitting.online_recovery import conv_info as _conv_info

N_PARAMS = 10
COHS = [0.06, 0.12, 0.24]
PRIORS = [80, 40, 20, 10]
LAPSE_MAX = 0.3


# --------------------------- param transforms ------------------------------
def _sig(x):   return 1.0 / (1.0 + np.exp(-x))
def _logit(p): return np.log(p / (1.0 - p))


def pack(k_like, k_prior, alpha, k_motor, lapse):
    return np.array(
        [np.log(k_like[c]) for c in COHS]
        + [np.log(k_prior[p]) for p in PRIORS]
        + [_logit(min(max(alpha, 1e-4), 1 - 1e-4)),
           np.log(k_motor),
           _logit(min(max(lapse / LAPSE_MAX, 1e-4), 1 - 1e-4))])


def unpack(theta) -> ReliabilityMixtureObserver:
    i = 0
    k_like = {c: float(np.exp(theta[i + j])) for j, c in enumerate(COHS)}; i += len(COHS)
    k_prior = {p: float(np.exp(theta[i + j])) for j, p in enumerate(PRIORS)}; i += len(PRIORS)
    alpha = float(_sig(theta[i])); i += 1
    k_motor = float(np.exp(theta[i])); i += 1
    lapse = float(LAPSE_MAX * _sig(theta[i]))
    return ReliabilityMixtureObserver(k_like=k_like, k_prior=k_prior,
                                      alpha=alpha, k_motor=k_motor, lapse=lapse)


# --------------------------- likelihood helpers ----------------------------
def _trial_logliks(obs: ReliabilityMixtureObserver, data):
    """Per-trial log p(estimate_t | ...) with the reliance filter over the FULL
    ordered sequence (so learning/order is preserved even under a CV mask)."""
    out = obs.filter(data["motion_direction"], data["motion_coherence"],
                     data["prior_std"], session_id=data.get("session_id"),
                     feedback=data["motion_direction"], sample=False)
    e = np.asarray(data["estimates"], int)
    return np.array([np.log(max(out["dists"][t][(e[t] - 1) % 360], 1e-320))
                     for t in range(len(e))])


def nll_masked(obs, data, mask=None):
    ll = _trial_logliks(obs, data)
    return -float(ll.sum() if mask is None else ll[mask].sum())


def fit(data, x0=None, maxiter=400, mask=None):
    def obj(theta):
        try:
            v = nll_masked(unpack(theta), data, mask)
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12
    if x0 is None:
        x0 = pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                  {80: 0.75, 40: 2.8, 20: 8.7, 10: 33.0},
                  0.05, 30.0, 0.02)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    obs = unpack(res.x)
    nll = float(res.fun)
    obs._fit_info = _conv_info(res, maxiter)
    return obs, nll, res.x


# ------------------------------ recovery -----------------------------------
def _simulate(obs, design, seed):
    """Roll the observer through a design, sampling one response per trial.
    Accepts either a DataFrame (helpers.dataset design) or a dict of arrays."""
    rng = np.random.RandomState(seed)
    def col(name):
        v = design[name]
        return v.values if hasattr(v, "values") else np.asarray(v)
    d = col("motion_direction").astype(int)
    c = col("motion_coherence").astype(float)
    ps = col("prior_std").astype(int)
    sid = None
    try:
        sid = col("session_id")
    except Exception:
        sid = None
    out = obs.filter(d, c, ps, session_id=sid, feedback=d, sample=True, rng=rng)
    return {"motion_direction": d, "motion_coherence": c, "prior_std": ps,
            "estimates": out["responses"],
            **({"session_id": sid} if sid is not None else {})}
