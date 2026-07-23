"""
basic_bayesian_fit.py
=====================

Fit the Basic Bayesian observer to one subject, with the same CLI as the other
per-model fitters:

    python -m observers.fitting.basic_bayesian_fit human 1

Nine free parameters (3 k_like + 4 k_prior + k_motor + p_random), optimised by
Nelder-Mead on the trial-level negative log-likelihood. Returns / prints
NLL, AIC, BIC. Does not write to disk.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import minimize

from observers.models.basic_bayesian import BasicBayesianObserver, N_PARAMS
from observers.fitting.online_recovery import conv_info as _conv_info
from observers.helpers.dataset import load_subject_design
from observers.helpers.paths import DATA_CSV

SD_LABELS = {80: "80", 40: "40", 20: "20", 10: "10"}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def pack(k_like, k_prior, k_motor, p_random):
    """Pack the 9 parameters into an unconstrained vector (log / logit space)."""
    return np.array([
        np.log(k_like[0.06]), np.log(k_like[0.12]), np.log(k_like[0.24]),
        np.log(k_prior["80"]), np.log(k_prior["40"]),
        np.log(k_prior["20"]), np.log(k_prior["10"]),
        np.log(k_motor), _logit(p_random)])


def unpack(theta) -> BasicBayesianObserver:
    return BasicBayesianObserver(
        k_like={0.06: np.exp(theta[0]), 0.12: np.exp(theta[1]),
                0.24: np.exp(theta[2])},
        k_prior={"80": np.exp(theta[3]), "40": np.exp(theta[4]),
                 "20": np.exp(theta[5]), "10": np.exp(theta[6])},
        k_motor=np.exp(theta[7]), p_random=_sig(theta[8]))


def fit(data, x0=None, maxiter=500, tol=1e-2):
    """Fit the Basic Bayesian observer to one subject. Returns (obs, nll, aic, bic).

    ``tol`` sets both the Nelder-Mead function and parameter tolerances
    (xatol/fatol). Default 1e-2 is the house value (unchanged for any existing
    caller); the comparison registry passes the paper-standard 1e-4 so this
    model is fit to the same tolerance Laquitaine & Gardner held ALL observers
    to ("very strict function and parameter tolerances of 1e-4").
    """
    from observers.fitting import fit_heartbeat as _hb
    e = np.asarray(data["estimates"], dtype=int)
    d = np.asarray(data["motion_direction"], dtype=int)
    c = np.asarray(data["motion_coherence"], dtype=float)
    pl = np.array([SD_LABELS[int(s)] for s in data["prior_std"]])

    def obj(theta):
        try:
            v = unpack(theta).negative_log_likelihood(e, d, c, pl)
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12

    obj = _hb.wrap(obj, maxiter)   # intra-fit progress beat (no-op unless configured)
    if x0 is None:
        x0 = pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                  {"80": 0.7, "40": 2.8, "20": 8.7, "10": 33.0}, 30.0, 0.05)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": tol, "fatol": tol})
    obs = unpack(res.x)
    nll = float(res.fun)
    obs._fit_info = _conv_info(res, maxiter)   # convergence diagnostics
    n = int(e.size)
    return obs, nll, BasicBayesianObserver.aic(nll), BasicBayesianObserver.bic(nll, n)


def _load_subject(sid):
    d = load_subject_design(str(DATA_CSV), int(sid))
    return dict(
        motion_direction=d.motion_direction.values.astype(int),
        motion_coherence=d.motion_coherence.values.astype(float),
        prior_std=d.prior_std.values.astype(int),
        estimates=d.estimate_dir.values.astype(int),
    )


def human(subject_ids, maxiter=500):
    out = {}
    for sid in subject_ids:
        data = _load_subject(int(sid))
        obs, nll, aic, bic = fit(data, maxiter=maxiter)
        n = len(np.asarray(data["estimates"]))
        out[int(sid)] = {"nll": nll, "aic": aic, "bic": bic, "n": n,
                         "k_like": obs.k_like, "k_prior": obs.k_prior,
                         "k_motor": obs.k_motor, "p_random": obs.p_random}
        print(f"subject {sid}: n={n}  NLL={nll:.1f}  AIC={aic:.1f}  BIC={bic:.1f}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "human":
        ids = [int(a) for a in sys.argv[2:]] or [1]
        human(ids)
    else:
        print(__doc__)
        print("usage: python -m observers.fitting.basic_bayesian_fit human <subject_id> [...]")
