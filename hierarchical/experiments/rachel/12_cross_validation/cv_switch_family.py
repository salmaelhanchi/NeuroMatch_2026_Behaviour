"""
cv_switch_family.py — sequence-preserving K-fold cross-validation for the
switch-family models, to test whether AT's in-sample win is overfitting.

Metric: held-out per-trial NLL (lower = better predictive fit). No complexity
penalty is needed — out-of-sample scoring penalizes overfitting automatically.
If AT is overfit, its held-out NLL will FAIL to beat the simpler models (static,
online, adaptive) despite winning on in-sample AIC/BIC.

Folds are CONTIGUOUS trial segments (K=5); the belief filter always runs over
the FULL ordered sequence, so sequential learning is preserved. We fit on the
train mask and score the test mask. (Caveat, same as hb_integration_fit.cv:
feedback = the exogenous stimulus direction is present on every trial, so the
belief 'sees' held-out feedback; this scores predictive fit of RESPONSES with
order intact, not a strictly causal forecast of learning. Documented.)

Warm-started from each model's full-data MLE (all12_fit_results.json) so folds
converge fast.

Models: static(9), online(6), AT(11), adaptive_volatility(6).

Usage: python cv_switch_family.py <subject ...>   (default all 12)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HIER = "/Users/vestige/code/heartwood/nma-prep/group-project/hierarchical"
WORK = os.path.dirname(os.path.abspath(__file__))
for p in (HIER, WORK):
    if p not in sys.path:
        sys.path.insert(0, p)

from online_simulate import load_subject_design
from online_recovery import pack_online, unpack_online, pack_static, unpack_static
from online_learner import OnlineHierarchicalObserver
import asymptote_transient_fit as atf
from adaptive_volatility_switching import AdaptiveVolatilitySwitchingObserver

CSV = os.path.join(HIER, "data", "data01_direction4priors.csv")
COHS = [0.06, 0.12, 0.24]
K = 5
OUTDIR = os.path.join(WORK, "cv_results")
os.makedirs(OUTDIR, exist_ok=True)
FULL = json.load(open(os.path.join(WORK, "all12_fit_results.json")))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def _clamp(v, lo=1e-3, hi=1e3):
    return float(min(max(v, lo), hi))


def load_data(sid):
    d = load_subject_design(CSV, sid)
    return dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                prior_std=d.prior_std.values.astype(int),
                estimates=d.estimate_dir.values.astype(int))


# ----- per-trial log-likelihood over the FULL ordered sequence -------------
def _ll_from_dists(dists, e):
    return np.array([np.log(max(dists[t][(e[t] - 1) % 360], 1e-320))
                     for t in range(len(e))])


def ll_belief(obs, data):
    """online / adaptive: filter over full sequence."""
    out = obs.filter(data["motion_direction"], data["motion_coherence"],
                     feedback=data["motion_direction"], sample=False)
    return _ll_from_dists(out["dists"], data["estimates"])


def ll_at(obs, data):
    dists, _ = obs.estimate_distributions(data["motion_direction"],
                                          data["motion_coherence"], data["prior_std"])
    return _ll_from_dists(dists, data["estimates"])


def ll_static(obs, kp, data):
    d, c, s, e = (data["motion_direction"], data["motion_coherence"],
                  data["prior_std"], data["estimates"])
    obs._prepare(d, c)          # prime _evi_conv / _prior_conv caches
    kpp = np.array([kp[si] for si in s])
    ll = np.empty(len(e))
    for t in range(len(e)):
        dist = obs.estimate_distribution_fixedk(c[t], int(d[t]), kpp[t])
        ll[t] = np.log(max(dist[(e[t] - 1) % 360], 1e-320))
    return ll


# ----- packers / unpackers per model ---------------------------------------
def _pack_adv(kl, km, pr, h):
    return np.array([np.log(kl[0.06]), np.log(kl[0.12]), np.log(kl[0.24]),
                     np.log(km), _logit(pr), _logit(h)])


def _unpack_adv(th):
    return AdaptiveVolatilitySwitchingObserver(
        k_like={0.06: np.exp(th[0]), 0.12: np.exp(th[1]), 0.24: np.exp(th[2])},
        k_motor=np.exp(th[3]), p_random=_sig(th[4]), hazard=_sig(th[5]))


def warm_x0(sid, model):
    """Warm-start vector from the full-data MLE."""
    r = FULL[str(sid)][model]
    kl = {0.06: _clamp(r["k_like"][0]), 0.12: _clamp(r["k_like"][1]),
          0.24: _clamp(r["k_like"][2])}
    pr = min(max(r["p_random"], 1e-3), 0.3)
    km = _clamp(r["k_motor"])
    if model == "static":
        kp = r["k_prior"]
        return pack_static(kl, {80: _clamp(kp["80"]), 40: _clamp(kp["40"]),
                                20: _clamp(kp["20"]), 10: _clamp(kp["10"])}, km, pr)
    if model == "online":
        return pack_online(kl, km, pr, min(max(r["lam"], 1e-3), 0.9))
    if model == "at":
        seed = atf.AsymptoteTransientObserver(
            k_like=kl, k_asym={80: _clamp(r["k_asym"]["80"]), 40: _clamp(r["k_asym"]["40"]),
                               20: _clamp(r["k_asym"]["20"]), 10: _clamp(r["k_asym"]["10"])},
            k_motor=km, p_random=pr,
            tau_tighten=_clamp(r["tau_tighten"], 1e-3, 500),
            tau_loosen=_clamp(r["tau_loosen"], 1e-3, 500))
        return atf.pack(seed)
    if model == "adaptive_volatility":
        return _pack_adv(kl, km, pr, min(max(r["hazard"], 1e-3), 0.5))


# ----- fit on train mask, score test mask ----------------------------------
def fit_score(model, data, train, test, x0, maxiter=300):
    e = data["estimates"]

    def masked_nll(theta):
        try:
            if model == "static":
                obs, kp = unpack_static(theta); ll = ll_static(obs, kp, data)
            elif model == "online":
                ll = ll_belief(unpack_online(theta), data)
            elif model == "at":
                ll = ll_at(atf.unpack(theta), data)
            else:
                ll = ll_belief(_unpack_adv(theta), data)
            v = -float(ll[train].sum())
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12

    res = minimize(masked_nll, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    # score held-out with the fitted params
    th = res.x
    if model == "static":
        obs, kp = unpack_static(th); ll = ll_static(obs, kp, data)
    elif model == "online":
        ll = ll_belief(unpack_online(th), data)
    elif model == "at":
        ll = ll_at(atf.unpack(th), data)
    else:
        ll = ll_belief(_unpack_adv(th), data)
    return -float(ll[test].sum()), int(test.sum())


MODELS = ["static", "online", "at", "adaptive_volatility"]


def run_subject(sid):
    path = os.path.join(OUTDIR, f"subj_{sid}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    data = load_data(sid)
    n = data["estimates"].size
    res["n_trials"] = n
    folds = np.array_split(np.arange(n), K)
    for model in MODELS:
        if model in res and "cv_nll" in res[model]:
            continue
        x0 = warm_x0(sid, model)
        tot, ntest = 0.0, 0
        for test_idx in folds:
            test = np.zeros(n, dtype=bool); test[test_idx] = True
            train = ~test
            nll, nt = fit_score(model, data, train, test, x0)
            tot += nll; ntest += nt
        res[model] = {"cv_nll": tot, "cv_nll_per_trial": tot / ntest, "n_test": ntest}
        tmp = path + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, path)
        print(f"[s{sid}] {model}: held-out NLL={tot:.1f} (per-trial {tot/ntest:.4f})", flush=True)
    best = min(MODELS, key=lambda m: res[m]["cv_nll"])
    print(f"[s{sid}] DONE  best held-out = {best}", flush=True)


if __name__ == "__main__":
    subs = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
    print(f"CV: subjects {subs}", flush=True)
    for sid in subs:
        try:
            run_subject(sid)
        except Exception as ex:
            print(f"[s{sid}] ERROR: {ex}", flush=True)
    print("CV ALL DONE", flush=True)
