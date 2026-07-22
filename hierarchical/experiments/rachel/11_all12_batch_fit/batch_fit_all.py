"""
batch_fit_all.py — fit all 6 observer models to all 12 subjects, fairly.

Models (params): static(9), online(6), asymptote_transient/AT(11),
integration_free_alpha(7), integration_derived_alpha(6), adaptive_volatility(6).

Reuses each model's EXISTING fit function so conventions match prior results:
  static  : fair_refit._fit_static_x0 + online_recovery.unpack_static
  online  : online_recovery.fit_online
  AT      : asymptote_transient_fit.fit
  integ   : hb_integration_fit.fit / pack
  derived : hb_integration_derived (workspace) + local packer
  adaptive: adaptive_volatility_switching (workspace) + local packer

Fair multi-start: every model gets a cold default start plus warm/variant
starts (warm-from-static for the learners, warm-from-online for integration),
mirroring fair_refit.py. Best NLL kept.

Resumable: writes batch_results/subj_<sid>.json after EACH model, and skips any
(subject, model) already present. Safe to re-run after an interruption.

Usage:
  python batch_fit_all.py 1 2 3 ... 12      # subjects (default: all 12)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HIER = "/Users/vestige/code/heartwood/nma-prep/group-project/hierarchical"
WORK = os.path.dirname(os.path.abspath(__file__))
for p in (HIER, WORK):
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy.optimize import minimize

from online_simulate import load_subject_design
from online_recovery import pack_online, unpack_online, pack_static, unpack_static, fit_online
from fair_refit import _fit_static_x0
import asymptote_transient_fit as atf
import hb_integration_fit as hbf
from hb_integration_derived import HBIntegrationDerivedObserver
from adaptive_volatility_switching import AdaptiveVolatilitySwitchingObserver

CSV = os.path.join(HIER, "data", "data01_direction4priors.csv")
COHS = [0.06, 0.12, 0.24]
OUTDIR = os.path.join(WORK, "batch_results")
os.makedirs(OUTDIR, exist_ok=True)


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


def aic_bic(nll, k, n):
    return 2 * k + 2 * nll, k * float(np.log(n)) + 2 * nll


# ----------------------- derived-alpha packer (6p) -------------------------
def _pack_der(kl, km, pr, lam):
    return np.array([np.log(kl[0.06]), np.log(kl[0.12]), np.log(kl[0.24]),
                     np.log(km), _logit(pr), _logit(lam)])


def _unpack_der(th):
    return HBIntegrationDerivedObserver(
        k_like={0.06: np.exp(th[0]), 0.12: np.exp(th[1]), 0.24: np.exp(th[2])},
        k_motor=np.exp(th[3]), p_random=_sig(th[4]), lam=_sig(th[5]))


def _fit_der(data, x0, maxiter=400):
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]

    def obj(th):
        try:
            out = _unpack_der(th).filter(d, c, feedback=d, sample=False)
            ll = sum(np.log(max(out["dists"][t][(e[t] - 1) % 360], 1e-320))
                     for t in range(len(e)))
            return -float(ll) if np.isfinite(ll) else 1e12
        except Exception:
            return 1e12
    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    return _unpack_der(r.x), float(r.fun)


# ----------------------- adaptive-volatility packer (6p) -------------------
def _pack_adv(kl, km, pr, h):
    return np.array([np.log(kl[0.06]), np.log(kl[0.12]), np.log(kl[0.24]),
                     np.log(km), _logit(pr), _logit(h)])


def _unpack_adv(th):
    return AdaptiveVolatilitySwitchingObserver(
        k_like={0.06: np.exp(th[0]), 0.12: np.exp(th[1]), 0.24: np.exp(th[2])},
        k_motor=np.exp(th[3]), p_random=_sig(th[4]), hazard=_sig(th[5]))


def _fit_adv(data, x0, maxiter=400):
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]

    def obj(th):
        try:
            v = _unpack_adv(th).negative_log_likelihood(e, d, c, feedback=d)
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12
    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    return _unpack_adv(r.x), float(r.fun)


# --------------------------------- I/O -------------------------------------
def _path(sid):
    return os.path.join(OUTDIR, f"subj_{sid}.json")


def _load(sid):
    return json.load(open(_path(sid))) if os.path.exists(_path(sid)) else {}


def _save(sid, res):
    tmp = _path(sid) + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, _path(sid))


# ------------------------------- per model ---------------------------------
def fit_static(data, n):
    best_x, best = None, np.inf
    starts = [pack_static({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                          {80: 0.7, 40: 2.8, 20: 8.7, 10: 33.0}, 30.0, 0.05),
              pack_static({0.06: 1.5, 0.12: 6.0, 0.24: 20.0},
                          {80: 1.0, 40: 4.0, 20: 12.0, 10: 50.0}, 25.0, 0.03)]
    for x0 in starts:
        x, nll = _fit_static_x0(data, x0, maxiter=400)
        if nll < best:
            best_x, best = x, nll
    obs, kp = unpack_static(best_x)
    a, b = aic_bic(best, 9, n)
    return dict(nll=best, aic=a, bic=b, k_motor=obs.k_motor, p_random=obs.p_random,
                k_like=[obs.k_like[c] for c in COHS],
                k_prior={str(s): kp[s] for s in kp}), best_x


def fit_online_m(data, n, static_x):
    best, bnll = None, np.inf
    starts = [pack_online({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05)]
    if static_x is not None:
        os_, _ = unpack_static(static_x)
        starts.append(pack_online({k: _clamp(v) for k, v in os_.k_like.items()},
                                   _clamp(os_.k_motor),
                                   min(max(os_.p_random, 1e-3), 0.2), 0.05))
    for x0 in starts:
        obs, nll, _ = fit_online(data, x0=x0, maxiter=400)
        if nll < bnll:
            best, bnll = obs, nll
    a, b = aic_bic(bnll, 6, n)
    return dict(nll=bnll, aic=a, bic=b, k_motor=best.k_motor, p_random=best.p_random,
                lam=best.lam, k_like=[best.k_like[c] for c in COHS])


def fit_at(data, n, static_x):
    best, bnll = None, np.inf
    starts = [atf.pack(atf.AsymptoteTransientObserver())]
    if static_x is not None:
        os_, kp = unpack_static(static_x)
        seed = atf.AsymptoteTransientObserver(
            k_like={c: _clamp(os_.k_like[c]) for c in COHS},
            k_asym={s: _clamp(kp[s]) for s in kp},
            k_motor=_clamp(os_.k_motor),
            p_random=min(max(os_.p_random, 1e-3), 0.2),
            tau_tighten=3.0, tau_loosen=3.0)
        starts.append(atf.pack(seed))
    for x0 in starts:
        obs, nll, _ = atf.fit(data, x0=x0, maxiter=800)
        if nll < bnll:
            best, bnll = obs, nll
    a, b = aic_bic(bnll, 11, n)
    return dict(nll=bnll, aic=a, bic=b, k_motor=best.k_motor, p_random=best.p_random,
                tau_tighten=best.tau_tighten, tau_loosen=best.tau_loosen,
                k_like=[best.k_like[c] for c in COHS],
                k_asym={str(s): best.k_asym[s] for s in best.k_asym})


def fit_integ(data, n, online_res):
    best, bnll = None, np.inf
    starts = [hbf.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.6, 30.0, 0.05, 0.05),
              hbf.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.75, 25.0, 0.03, 0.10)]
    if online_res is not None:
        kl = {0.06: online_res["k_like"][0], 0.12: online_res["k_like"][1],
              0.24: min(online_res["k_like"][2], 300)}
        starts.append(hbf.pack(kl, 0.6, online_res["k_motor"],
                               min(max(online_res["p_random"], 1e-3), 0.2), 0.05))
    for x0 in starts:
        obs, nll, _ = hbf.fit(data, x0=x0, maxiter=400)
        if nll < bnll:
            best, bnll = obs, nll
    a, b = aic_bic(bnll, 7, n)
    return dict(nll=bnll, aic=a, bic=b, alpha=best.alpha, k_motor=best.k_motor,
                p_random=best.p_random, lam=best.lam,
                k_like=[best.k_like[c] for c in COHS])


def fit_derived(data, n, online_res):
    best, bnll = None, np.inf
    starts = [_pack_der({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05)]
    if online_res is not None:
        kl = {0.06: online_res["k_like"][0], 0.12: online_res["k_like"][1],
              0.24: min(online_res["k_like"][2], 300)}
        starts.append(_pack_der(kl, online_res["k_motor"],
                                min(max(online_res["p_random"], 1e-3), 0.2), 0.05))
    for x0 in starts:
        obs, nll = _fit_der(data, x0, maxiter=400)
        if nll < bnll:
            best, bnll = obs, nll
    a, b = aic_bic(bnll, 6, n)
    return dict(nll=bnll, aic=a, bic=b, k_motor=best.k_motor, p_random=best.p_random,
                lam=best.lam, k_like=[best.k_like[c] for c in COHS])


def fit_adaptive(data, n, online_res):
    best, bnll = None, np.inf
    starts = [_pack_adv({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05),
              _pack_adv({0.06: 1.5, 0.12: 6.0, 0.24: 20.0}, 25.0, 0.02, 0.15)]
    if online_res is not None:
        kl = {0.06: online_res["k_like"][0], 0.12: online_res["k_like"][1],
              0.24: min(online_res["k_like"][2], 300)}
        starts.append(_pack_adv(kl, online_res["k_motor"],
                                min(max(online_res["p_random"], 1e-3), 0.2), 0.08))
    for x0 in starts:
        obs, nll = _fit_adv(data, x0, maxiter=400)
        if nll < bnll:
            best, bnll = obs, nll
    a, b = aic_bic(bnll, 6, n)
    return dict(nll=bnll, aic=a, bic=b, k_motor=best.k_motor, p_random=best.p_random,
                hazard=best.hazard, k_like=[best.k_like[c] for c in COHS])


# --------------------------------- driver ----------------------------------
MODEL_ORDER = ["static", "online", "at", "integration_free_alpha",
               "integration_derived_alpha", "adaptive_volatility"]


def run_subject(sid):
    res = _load(sid)
    data = load_data(sid)
    n = int(data["estimates"].size)
    res["n_trials"] = n
    static_x = res.get("_static_x")
    static_x = np.array(static_x) if static_x is not None else None

    def done(m):
        return m in res and isinstance(res[m], dict) and "aic" in res[m]

    if not done("static"):
        t = time.time()
        r, static_x = fit_static(data, n)
        res["static"] = r
        res["_static_x"] = static_x.tolist()
        _save(sid, res)
        print(f"[s{sid}] static AIC={r['aic']:.1f} ({time.time()-t:.0f}s)", flush=True)
    elif static_x is None and "_static_x" in res:
        static_x = np.array(res["_static_x"])

    if not done("online"):
        t = time.time()
        res["online"] = fit_online_m(data, n, static_x)
        _save(sid, res)
        print(f"[s{sid}] online AIC={res['online']['aic']:.1f} ({time.time()-t:.0f}s)", flush=True)

    if not done("at"):
        t = time.time()
        res["at"] = fit_at(data, n, static_x)
        _save(sid, res)
        print(f"[s{sid}] AT AIC={res['at']['aic']:.1f} ({time.time()-t:.0f}s)", flush=True)

    if not done("integration_free_alpha"):
        t = time.time()
        res["integration_free_alpha"] = fit_integ(data, n, res.get("online"))
        _save(sid, res)
        print(f"[s{sid}] integ AIC={res['integration_free_alpha']['aic']:.1f} ({time.time()-t:.0f}s)", flush=True)

    if not done("integration_derived_alpha"):
        t = time.time()
        res["integration_derived_alpha"] = fit_derived(data, n, res.get("online"))
        _save(sid, res)
        print(f"[s{sid}] derived AIC={res['integration_derived_alpha']['aic']:.1f} ({time.time()-t:.0f}s)", flush=True)

    if not done("adaptive_volatility"):
        t = time.time()
        res["adaptive_volatility"] = fit_adaptive(data, n, res.get("online"))
        _save(sid, res)
        print(f"[s{sid}] adaptive AIC={res['adaptive_volatility']['aic']:.1f} ({time.time()-t:.0f}s)", flush=True)

    best = min(MODEL_ORDER, key=lambda m: res[m]["aic"])
    print(f"[s{sid}] DONE  best={best}  AIC={res[best]['aic']:.1f}", flush=True)
    return sid


if __name__ == "__main__":
    subs = [int(a) for a in sys.argv[1:]] or list(range(1, 13))
    print(f"batch fit: subjects {subs}", flush=True)
    for sid in subs:
        try:
            run_subject(sid)
        except Exception as e:
            print(f"[s{sid}] ERROR: {e}", flush=True)
    print("ALL DONE", flush=True)
