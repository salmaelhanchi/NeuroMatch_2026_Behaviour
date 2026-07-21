"""
fair_refit.py
=============

Give the switch models (online, static) the same multi-start treatment as the
integration and AT models, so the four-model comparison is like-for-like. A
single cold-start fit can settle in a local optimum and understate a model's
fit; running several starts and keeping the best NLL puts every model on equal
footing.

Starts per model:
  static  : cold default; warm from the AT fit (AT nests static, so its per-block
            asymptotes k_asym are natural k_prior seeds).
  online  : cold default; warm from this subject's best static fit; warm from AT.

Writes results/fits/fair_fit_results.json (best NLL/AIC/BIC per model), then
`table` prints the four-model comparison, pulling the integration fit from
hb_integration_results.json and the AT fit from at_fit_results.json.

Usage:
  python fair_refit.py refit 1 3
  python fair_refit.py table
"""

from __future__ import annotations

import json, os, sys
import numpy as np
from scipy.optimize import minimize

from observers.fitting.online_recovery import (fit_online, pack_online, pack_static, unpack_static)
from observers.helpers.dataset import load_subject_design
from observers.helpers.paths import DATA_CSV, AT_FITS, HB_FITS, FAIR_FITS

CSV = DATA_CSV
FAIR = FAIR_FITS


def _data(sid):
    d = load_subject_design(CSV, sid)
    return dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                prior_std=d.prior_std.values.astype(int),
                estimates=d.estimate_dir.values.astype(int))


def _clamp(v, lo=1e-3, hi=1e3):
    """Clamp a (possibly Infinity) fitted concentration to a finite range so it
    is a usable warm-start seed (AT's k_e[0.24] can be Infinity)."""
    return float(min(max(v, lo), hi))


# ------------------------------ static -------------------------------------
def _fit_static_x0(data, x0, maxiter=400):
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
    sd = np.asarray(data["prior_std"], dtype=int)
    def kpp(kp): return np.array([kp[s] for s in sd])
    def obj(theta):
        try:
            obs, kp = unpack_static(theta)
            v = obs.negative_log_likelihood_fixedk(e, d, c, kpp(kp))
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    return res.x, float(res.fun)


def _static_starts(sid, at):
    starts = [("cold", pack_static({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                                    {80: 0.7, 40: 2.8, 20: 8.7, 10: 33.0}, 30.0, 0.05))]
    if str(sid) in at:
        a = at[str(sid)]
        kl = {0.06: _clamp(a["k_like"]["0.06"]), 0.12: _clamp(a["k_like"]["0.12"]),
              0.24: _clamp(a["k_like"]["0.24"])}
        kp = {80: _clamp(a["k_asym"]["80"]), 40: _clamp(a["k_asym"]["40"]),
              20: _clamp(a["k_asym"]["20"]), 10: _clamp(a["k_asym"]["10"])}
        starts.append(("warm_AT", pack_static(kl, kp, _clamp(a["k_motor"]),
                                              min(max(a["p_random"], 1e-3), 0.2))))
    return starts


# ------------------------------ online -------------------------------------
def _online_starts(sid, at, static_best_x):
    starts = [("cold", pack_online({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05))]
    if static_best_x is not None:
        obs_s, _ = unpack_static(static_best_x)
        starts.append(("warm_static", pack_online(
            {k: _clamp(v) for k, v in obs_s.k_like.items()},
            _clamp(obs_s.k_motor), min(max(obs_s.p_random, 1e-3), 0.2), 0.05)))
    if str(sid) in at:
        a = at[str(sid)]
        kl = {0.06: _clamp(a["k_like"]["0.06"]), 0.12: _clamp(a["k_like"]["0.12"]),
              0.24: _clamp(a["k_like"]["0.24"])}
        starts.append(("warm_AT", pack_online(kl, _clamp(a["k_motor"]),
                                              min(max(a["p_random"], 1e-3), 0.2), 0.05)))
    return starts


def _save(sid, upd):
    disk = json.load(open(FAIR)) if os.path.exists(FAIR) else {}
    disk.setdefault(str(sid), {}).update(upd)
    json.dump(disk, open(FAIR, "w"), indent=2)


def refit(subject_ids):
    at = json.load(open(AT_FITS)) if AT_FITS.exists() else {}
    for sid in subject_ids:
        sid = int(sid)
        data = _data(sid); n = data["estimates"].size
        print(f"subject {sid} (n={n}) — fair multi-start refit of switch models", flush=True)

        # STATIC
        best_x, best_nll = None, np.inf
        for name, x0 in _static_starts(sid, at):
            x, nll = _fit_static_x0(data, x0, maxiter=400)
            print(f"    static[{name}]: NLL={nll:.1f}{'  <-best' if nll < best_nll else ''}", flush=True)
            if nll < best_nll:
                best_x, best_nll = x, nll
                _save(sid, dict(n_trials=n, nll_static=best_nll,
                                aic_static=2*9+2*best_nll, bic_static=9*float(np.log(n))+2*best_nll))
        static_best_x = best_x

        # ONLINE
        best_nll = np.inf
        for name, x0 in _online_starts(sid, at, static_best_x):
            _obs, nll, _aic = fit_online(data, x0=x0, maxiter=400)
            print(f"    online[{name}]: NLL={nll:.1f}{'  <-best' if nll < best_nll else ''}", flush=True)
            if nll < best_nll:
                best_nll = nll
                _save(sid, dict(nll_online=best_nll,
                                aic_online=2*6+2*best_nll, bic_online=6*float(np.log(n))+2*best_nll))
        r = json.load(open(FAIR))[str(sid)]
        print(f"  subject {sid} done: static AIC={r['aic_static']:.1f}  "
              f"online AIC={r['aic_online']:.1f}", flush=True)


def table():
    fair = json.load(open(FAIR))
    integ = json.load(open(HB_FITS))
    at = json.load(open(AT_FITS))
    print(f"{'subj':>4} | {'integration':>11} {'online':>9} {'static':>9} {'AT':>9} | best (fair, all multi-started)")
    for sid in sorted(fair, key=int):
        c = {"integration": integ[sid]["aic_integration"],
             "online": fair[sid]["aic_online"], "static": fair[sid]["aic_static"],
             "AT": at[sid]["aic_at"]}
        best = min(c, key=c.get)
        print(f"{sid:>4} | {c['integration']:>11.1f} {c['online']:>9.1f} "
              f"{c['static']:>9.1f} {c['AT']:>9.1f} | {best}  (Δ2nd={sorted(c.values())[1]-sorted(c.values())[0]:.1f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "table"
    if mode == "refit":
        refit(sys.argv[2:])
    elif mode == "table":
        table()
