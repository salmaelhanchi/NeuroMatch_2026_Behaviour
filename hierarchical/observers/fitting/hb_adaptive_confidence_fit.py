"""
hb_adaptive_confidence_fit.py
=============================

House-style fitter for the HB-Adaptive-Confidence observer
(``hb_adaptive_confidence.py``) — the model that learns BOTH the prior
confidence alpha and the prior width kappa online, and so has only SIX fitted
parameters (alpha and kappa are learned latents, not fitted).

Mirrors ``hb_integration_fit.py`` exactly so the two plug into the comparison
registry through one interface. The only structural difference is the parameter
vector: there is NO alpha to fit here (it is learned), so ``pack``/``unpack``
carry six numbers instead of seven.

Fitted parameters (6):
    k_like[0.06], k_like[0.12], k_like[0.24]   sensory reliabilities per coherence
    k_motor                                     motor concentration
    p_random                                    lapse
    lam                                         volatility / forgetting

Modes:
  recover   parameter recovery on simulated data (the six fitted params).
  human N.. fit subject(s); report NLL / AIC / BIC.
  cv    N.. sequence-preserving K-fold cross-validation for a subject.

Usage:
  python -m observers.fitting.hb_adaptive_confidence_fit recover
  python -m observers.fitting.hb_adaptive_confidence_fit human 1 3
  python -m observers.fitting.hb_adaptive_confidence_fit cv 1

Public API (used by the comparison registry):
  N_PARAMS, pack, unpack, fit(data, x0=None, maxiter=400, mask=None),
  nll_masked, _load_subject, _simulate
"""

from __future__ import annotations

import json, sys
import numpy as np
from scipy.optimize import minimize

from observers.models.hb_adaptive_confidence import HBAdaptiveConfidenceObserver
from observers.helpers.dataset import load_subject_design, make_synthetic_design
from observers.helpers.paths import DATA_CSV
from observers.fitting.online_recovery import conv_info as _conv_info

CSV = DATA_CSV
COHS = [0.06, 0.12, 0.24]

N_PARAMS = 6  # 3 k_like + k_motor + p_random + lam  (alpha AND kappa are learned)


# --------------------------- param transforms ------------------------------
def _sig(x):   return 1.0 / (1.0 + np.exp(-x))
def _logit(p): return np.log(p / (1.0 - p))


def pack(k_like, k_motor, p_random, lam):
    """Six fitted params -> unconstrained theta (no alpha: it is learned)."""
    return np.array([np.log(k_like[0.06]), np.log(k_like[0.12]), np.log(k_like[0.24]),
                     np.log(k_motor), _logit(p_random), _logit(lam)])


def unpack(theta) -> HBAdaptiveConfidenceObserver:
    k_like = {0.06: np.exp(theta[0]), 0.12: np.exp(theta[1]), 0.24: np.exp(theta[2])}
    return HBAdaptiveConfidenceObserver(k_like=k_like, k_motor=np.exp(theta[3]),
                                        p_random=_sig(theta[4]), lam=_sig(theta[5]))


# --------------------------- likelihood helpers ----------------------------
def _trial_logliks(obs: HBAdaptiveConfidenceObserver, data):
    """Per-trial log p(estimate_t | ...) with the joint (kappa, alpha) belief
    filter over the FULL ordered sequence (learning/order preserved under a
    CV mask)."""
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
    out = obs.filter(d, c, feedback=d, sample=False)
    ll = np.array([np.log(max(out["dists"][t][(e[t] - 1) % 360], 1e-320))
                   for t in range(len(e))])
    return ll


def nll_masked(obs, data, mask=None):
    ll = _trial_logliks(obs, data)
    return -float(ll.sum() if mask is None else ll[mask].sum())


def fit(data, x0=None, maxiter=400, mask=None):
    """Fit the six params to one subject (Nelder-Mead). Returns (obs, nll, x)."""
    def obj(theta):
        try:
            v = nll_masked(unpack(theta), data, mask)
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12
    if x0 is None:
        x0 = pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    obs = unpack(res.x)
    nll = float(res.fun)
    obs._fit_info = _conv_info(res, maxiter)   # convergence diagnostics
    return obs, nll, res.x


def _starts():
    """Multi-start x0 list guarding the 6-D simplex against a local basin that
    leaves the volatility (lam) badly off. Cold default + two variants."""
    return [
        ("cold", pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05)),
        ("var1", pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 25.0, 0.03, 0.10)),
        ("var2", pack({0.06: 1.5, 0.12: 4.0, 0.24: 12.0}, 40.0, 0.02, 0.03)),
    ]


def fit_multistart(data, maxiter=400, mask=None):
    """Fit from several starts; keep the best. Returns (obs, nll, x, spread)
    where spread is the NLL range across starts (a convergence diagnostic)."""
    results = []
    for _, x0 in _starts():
        obs, nll, x = fit(data, x0=x0, maxiter=maxiter, mask=mask)
        results.append((nll, obs, x))
    results.sort(key=lambda r: r[0])
    nlls = [r[0] for r in results]
    spread = float(max(nlls) - min(nlls))
    best_nll, best_obs, best_x = results[0]
    return best_obs, best_nll, best_x, spread


# ------------------------------ recover ------------------------------------
def _simulate(obs, design, seed):
    rng = np.random.RandomState(seed)
    d = design["motion_direction"].values.astype(int)
    c = design["motion_coherence"].values.astype(float)
    out = obs.filter(d, c, feedback=d, sample=True, rng=rng)
    return {"motion_direction": d, "motion_coherence": c, "estimates": out["responses"]}


def recover():
    print("=== parameter recovery (adaptive-confidence model) ===")
    truth = dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                 k_motor=40.0, p_random=0.02, lam=0.10)
    gen = HBAdaptiveConfidenceObserver(**truth)
    design = make_synthetic_design(trials_per_block=200, seed=1)
    rec = {k: [] for k in ["k06", "k12", "k24", "k_motor", "p_random", "lam"]}
    for seed in (1, 2, 3):
        data = _simulate(gen, design, seed)
        obs, nll, x = fit(data, maxiter=500)
        rec["k06"].append(obs.k_like[0.06]); rec["k12"].append(obs.k_like[0.12])
        rec["k24"].append(obs.k_like[0.24]); rec["k_motor"].append(obs.k_motor)
        rec["p_random"].append(obs.p_random); rec["lam"].append(obs.lam)
        print(f"  seed {seed}: p_rand={obs.p_random:.3f} lam={obs.lam:.3f} "
              f"k=({obs.k_like[0.06]:.2f},{obs.k_like[0.12]:.2f},{obs.k_like[0.24]:.2f}) "
              f"k_motor={obs.k_motor:.1f}")
    tvals = dict(k06=1.0, k12=3.0, k24=8.0, k_motor=40.0, p_random=0.02, lam=0.10)
    print("  --- recovered mean vs truth ---")
    ok = True
    for k in rec:
        m = float(np.mean(rec[k])); rel = abs(m - tvals[k]) / abs(tvals[k])
        tag = "ok" if rel < 0.35 else "WEAK"
        ok = ok and (k in ("lam", "p_random") or rel < 0.35)
        print(f"    {k:9s}: recovered {m:8.3f}  truth {tvals[k]:7.3f}  rel {rel:4.2f}  [{tag}]")
    return ok


# ------------------------------ human fits ---------------------------------
def _load_subject(sid):
    d = load_subject_design(str(DATA_CSV), int(sid))
    return dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                prior_std=d.prior_std.values.astype(int),
                estimates=d.estimate_dir.values.astype(int))


def human(subject_ids, maxiter=400):
    out = {}
    for sid in subject_ids:
        sid = int(sid)
        data = _load_subject(sid)
        n = data["estimates"].size
        obs, nll, x, spread = fit_multistart(data, maxiter=maxiter)
        aic = HBAdaptiveConfidenceObserver.aic(nll)
        bic = HBAdaptiveConfidenceObserver.bic(nll, n)
        out[sid] = {"nll": nll, "aic": aic, "bic": bic, "n": n, "start_spread": spread,
                    "k_like": [obs.k_like[0.06], obs.k_like[0.12], obs.k_like[0.24]],
                    "k_motor": obs.k_motor, "p_random": obs.p_random, "lam": obs.lam}
        print(f"subject {sid}: n={n}  NLL={nll:.1f}  AIC={aic:.1f}  BIC={bic:.1f}  "
              f"(start-spread {spread:.2f})")
    return out


def cv(subject_ids, K=5):
    """K-fold CV with CONTIGUOUS folds; the joint belief always runs over the
    full ordered sequence, so sequential learning is preserved. Fit on train
    mask, score held-out predictive NLL per trial on the test mask.

    Caveat (identical to hb_integration_fit.cv): feedback is available on every
    trial regardless of mask, so the belief still 'sees' held-out feedback; this
    tests predictive fit of the RESPONSES with order intact, not a strictly
    causal forecast."""
    for sid in subject_ids:
        sid = int(sid)
        data = _load_subject(sid)
        n = data["estimates"].size
        folds = np.array_split(np.arange(n), K)
        tot_test = 0.0
        print(f"subject {sid}: {K}-fold sequence-preserving CV (n={n})")
        for f, test_idx in enumerate(folds):
            test = np.zeros(n, dtype=bool); test[test_idx] = True
            train = ~test
            obs, _, x = fit(data, maxiter=400, mask=train)
            test_nll = nll_masked(obs, data, mask=test)
            tot_test += test_nll
            print(f"  fold {f+1}/{K}: test trials={test.sum():4d}  "
                  f"held-out NLL={test_nll:8.1f}  (per-trial {test_nll/test.sum():.3f})")
        print(f"  total held-out NLL = {tot_test:.1f}  (per-trial {tot_test/n:.3f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "recover"
    if mode == "recover":
        recover()
    elif mode == "human":
        human(sys.argv[2:])
    elif mode == "cv":
        cv(sys.argv[2:])
    else:
        print(__doc__)
