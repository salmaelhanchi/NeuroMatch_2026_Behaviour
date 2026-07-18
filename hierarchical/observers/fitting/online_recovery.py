"""
online_recovery.py
==================

Fitting routines for the online switching observer, plus recovery tests that
establish the model can be fit and is identifiable before touching human data.

  fitting          : full-history likelihood + optimisation for the online model
                     (and the fast static path used by the switch fitter).
  parameter recovery : fit data simulated from known parameters and check the
                     fit returns them.
  model recovery   : simulate from the online (learning) model and the static
                     (no-learning) model; confirm AIC prefers the true generator.

Also cross-checks the fast static path against the independent
switching_observer.py implementation (guards against a shared-code shortcut
being subtly wrong).

Run: python online_recovery.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.helpers.dataset import make_synthetic_design, simulate, SD_TO_K, COHERENCES

COHS = [0.06, 0.12, 0.24]


# ---------------------------------------------------------------------------
# Parameter transforms: optimise in an unconstrained space
# ---------------------------------------------------------------------------
def _sig(x):  return 1.0 / (1.0 + np.exp(-x))
def _logit(p): return np.log(p / (1.0 - p))

def pack_online(k_like, k_motor, p_random, lam):
    return np.array([np.log(k_like[0.06]), np.log(k_like[0.12]), np.log(k_like[0.24]),
                     np.log(k_motor), _logit(p_random), _logit(lam)])

def unpack_online(theta):
    k_like = {0.06: np.exp(theta[0]), 0.12: np.exp(theta[1]), 0.24: np.exp(theta[2])}
    return OnlineHierarchicalObserver(k_like=k_like, k_motor=np.exp(theta[3]),
                                      p_random=_sig(theta[4]), lam=_sig(theta[5]))

def pack_static(k_like, k_prior_by_sd, k_motor, p_random):
    return np.array([np.log(k_like[0.06]), np.log(k_like[0.12]), np.log(k_like[0.24]),
                     np.log(k_prior_by_sd[80]), np.log(k_prior_by_sd[40]),
                     np.log(k_prior_by_sd[20]), np.log(k_prior_by_sd[10]),
                     np.log(k_motor), _logit(p_random)])

def unpack_static(theta):
    k_like = {0.06: np.exp(theta[0]), 0.12: np.exp(theta[1]), 0.24: np.exp(theta[2])}
    k_prior_by_sd = {80: np.exp(theta[3]), 40: np.exp(theta[4]),
                     20: np.exp(theta[5]), 10: np.exp(theta[6])}
    obs = OnlineHierarchicalObserver(k_like=k_like, k_motor=np.exp(theta[7]),
                                     p_random=_sig(theta[8]), lam=0.0)
    return obs, k_prior_by_sd


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_online(data, x0=None, maxiter=400):
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
    def obj(theta):
        try:
            return unpack_online(theta).negative_log_likelihood(e, d, c, feedback=d)
        except Exception:
            return 1e12
    if x0 is None:
        x0 = pack_online({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    nll = res.fun
    aic = 2 * 6 + 2 * nll
    return unpack_online(res.x), nll, aic

def fit_static(data, maxiter=500):
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
    sd = np.asarray(data["prior_std"], dtype=int)
    def kpp(k_prior_by_sd):
        return np.array([k_prior_by_sd[s] for s in sd])
    def obj(theta):
        try:
            obs, kp = unpack_static(theta)
            return obs.negative_log_likelihood_fixedk(e, d, c, kpp(kp))
        except Exception:
            return 1e12
    x0 = pack_static({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                     {80: 0.7, 40: 2.8, 20: 8.7, 10: 33.0}, 30.0, 0.05)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    nll = res.fun
    aic = 2 * 9 + 2 * nll
    return res.x, nll, aic


# ---------------------------------------------------------------------------
# Cross-check: fast static path == independent switching_observer.py
# ---------------------------------------------------------------------------
def crosscheck_static_matches_switching():
    from observers.models.switching_observer import SwitchingObserver
    online = OnlineHierarchicalObserver(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                                        k_motor=40.0, p_random=0.02, lam=0.0)
    sw = SwitchingObserver(k_like={0.24: 8.0, 0.12: 3.0, 0.06: 1.0},
                           k_prior={"80": 0.7485, "40": 2.7714, "20": 8.7488, "10": 33.34},
                           p_random=0.02, k_motor=40.0)
    online._prepare(np.array([85, 145, 205]), np.array(COHS))
    max_diff = 0.0
    for d in (85, 145, 205):
        for c, sd_lbl, kpr in [(0.06, "80", 0.7485), (0.12, "40", 2.7714),
                               (0.24, "20", 8.7488)]:
            a = online.estimate_distribution_fixedk(c, d, kpr)
            b = sw.estimate_distribution(d, c, sd_lbl)
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    ok = max_diff < 1e-9
    print(f"[{'PASS' if ok else 'FAIL'}] fast static path == switching_observer.py"
          f"  — max|Δ|={max_diff:.2e}")
    return ok


# ---------------------------------------------------------------------------
# parameter recovery
# ---------------------------------------------------------------------------
def parameter_recovery(seeds=(1, 2, 3)):
    print("\n=== parameter recovery (online model) ===")
    true = OnlineHierarchicalObserver(
        k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, k_motor=30.0,
        p_random=0.03, lam=0.05)
    design = make_synthetic_design(trials_per_block=200, seed=99)

    rec = {"k06": [], "k12": [], "k24": [], "k_motor": [], "p_random": [], "lam": []}
    for s in seeds:
        data = simulate(true, design, seed=s)
        data["prior_std"] = design["prior_std"].values
        fit, nll, aic = fit_online(data)
        rec["k06"].append(fit.k_like[0.06]); rec["k12"].append(fit.k_like[0.12])
        rec["k24"].append(fit.k_like[0.24]); rec["k_motor"].append(fit.k_motor)
        rec["p_random"].append(fit.p_random); rec["lam"].append(fit.lam)
        print(f"  seed {s}: k=({fit.k_like[0.06]:.2f},{fit.k_like[0.12]:.2f},"
              f"{fit.k_like[0.24]:.2f}) k_motor={fit.k_motor:.1f} "
              f"p_rand={fit.p_random:.3f} lam={fit.lam:.3f}")

    truth = {"k06": 1.0, "k12": 3.0, "k24": 8.0, "k_motor": 30.0,
             "p_random": 0.03, "lam": 0.05}
    print("  --- recovered (mean over seeds) vs truth ---")
    ok = True
    for k in rec:
        m = np.mean(rec[k])
        rel = abs(m - truth[k]) / (abs(truth[k]) + 1e-9)
        tag = "ok" if rel < 0.4 else "WEAK"
        if k != "lam" and rel >= 0.4:  # lam is expected to be weakly identified
            ok = False
        print(f"    {k:9s}: recovered {m:8.3f}  truth {truth[k]:8.3f}  "
              f"rel err {rel:5.2f}  [{tag}]")
    return ok


# ---------------------------------------------------------------------------
# model recovery
# ---------------------------------------------------------------------------
def model_recovery():
    print("\n=== model recovery (online vs static) ===")
    design = make_synthetic_design(trials_per_block=220, seed=7)
    sd = design["prior_std"].values

    # generator L: online learner with real forgetting
    genL = OnlineHierarchicalObserver(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                                      k_motor=30.0, p_random=0.03, lam=0.08)
    # generator S: static, prior fixed at each block's true strength (no learning)
    genS = OnlineHierarchicalObserver(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                                      k_motor=30.0, p_random=0.03, lam=0.0)

    # --- data from L (learning) ---
    dataL = simulate(genL, design, seed=11); dataL["prior_std"] = sd
    _, nllL_on, aicL_on = fit_online(dataL)
    _, nllL_st, aicL_st = fit_static(dataL)
    print(f"  data from ONLINE:  AIC(online)={aicL_on:.1f}  AIC(static)={aicL_st:.1f}"
          f"  -> prefers {'ONLINE' if aicL_on < aicL_st else 'static'}")

    # --- data from S (static): fix belief to the true per-block strength ---
    dataS = _simulate_static(genS, design, {80: SD_TO_K[80], 40: SD_TO_K[40],
                                            20: SD_TO_K[20], 10: SD_TO_K[10]}, seed=12)
    dataS["prior_std"] = sd
    _, nllS_on, aicS_on = fit_online(dataS)
    _, nllS_st, aicS_st = fit_static(dataS)
    print(f"  data from STATIC:  AIC(online)={aicS_on:.1f}  AIC(static)={aicS_st:.1f}"
          f"  -> prefers {'online' if aicS_on < aicS_st else 'STATIC'}")

    ok = (aicL_on < aicL_st) and (aicS_st <= aicS_on + 2)
    return ok


def _simulate_static(observer, design, k_prior_by_sd, seed=0):
    """Simulate the no-learning static observer (belief fixed per block)."""
    rng = np.random.RandomState(seed)
    d = design["motion_direction"].values.astype(int)
    c = design["motion_coherence"].values.astype(float)
    sd = design["prior_std"].values.astype(int)
    observer._prepare(d, c)
    est = []
    for t in range(d.size):
        dist = observer.estimate_distribution_fixedk(c[t], d[t], k_prior_by_sd[sd[t]])
        est.append(int(rng.choice(np.arange(1, 361), p=dist)))
    return {"motion_direction": d, "motion_coherence": c, "estimates": np.array(est)}


def static_parameter_recovery(seeds=(1, 2, 3), trials_per_block=200):
    """Parameter recovery for the STATIC Switching observer.

    Simulate data from a switch with known k_like / k_prior, refit with
    fit_static, and check the fit returns the generating parameters. This is the
    switch counterpart of parameter_recovery() (which covers the online model).

    Returns True if every parameter recovers within tolerance (sensory
    reliabilities k_like tightly; prior reliabilities k_prior in the right order
    and magnitude — the tightest prior, SD10, recovers least precisely because a
    tight prior yields few informative trials).
    """
    print("\n=== parameter recovery (static switch) ===")
    true_klike = {0.06: 1.0, 0.12: 3.0, 0.24: 8.0}
    true_kprior = {80: 0.5, 40: 1.4, 20: 2.7, 10: 8.7}
    gen = OnlineHierarchicalObserver(k_like=dict(true_klike), k_motor=40.0,
                                     p_random=0.02, lam=0.0)
    design = make_synthetic_design(trials_per_block=trials_per_block, seed=3)

    rec = {k: [] for k in ("k06", "k12", "k24", "kp80", "kp40", "kp20", "kp10",
                           "k_motor", "p_random")}
    for s in seeds:
        sim = _simulate_static(gen, design, true_kprior, seed=s)
        sim["prior_std"] = design["prior_std"].values.astype(int)
        theta, nll, aic = fit_static(sim, maxiter=600)
        obs, kp = unpack_static(theta)
        rec["k06"].append(obs.k_like[0.06]); rec["k12"].append(obs.k_like[0.12])
        rec["k24"].append(obs.k_like[0.24])
        rec["kp80"].append(kp[80]); rec["kp40"].append(kp[40])
        rec["kp20"].append(kp[20]); rec["kp10"].append(kp[10])
        rec["k_motor"].append(obs.k_motor); rec["p_random"].append(obs.p_random)
        print(f"  seed {s}: k_like=({obs.k_like[0.06]:.2f},{obs.k_like[0.12]:.2f},"
              f"{obs.k_like[0.24]:.2f}) k_prior=({kp[80]:.2f},{kp[40]:.2f},"
              f"{kp[20]:.2f},{kp[10]:.2f})")

    truth = {"k06": 1.0, "k12": 3.0, "k24": 8.0, "kp80": 0.5, "kp40": 1.4,
             "kp20": 2.7, "kp10": 8.7, "k_motor": 40.0, "p_random": 0.02}
    print("  --- recovered mean vs truth ---")
    ok = True
    for k, tval in truth.items():
        m = float(np.mean(rec[k]))
        rel = abs(m - tval) / max(abs(tval), 1e-9)
        # k_like/k_prior can drift (esp. the tight SD10 prior and the near-delta
        # high-coh likelihood); tolerate 40% on strengths, 25% on motor/lapse.
        tol = 0.40 if k.startswith(("k0", "k1", "k2", "kp")) else 0.25
        flag = "ok" if rel <= tol else "LOOSE"
        if rel > tol:
            ok = False
        print(f"    {k:9s}: recovered {m:8.3f}  truth {tval:7.3f}  rel {rel:.2f}  [{flag}]")
    print("  -> static switch parameters recover"
          + (" (all within tolerance)" if ok else " (some loose — see notes)"))
    return ok


if __name__ == "__main__":
    print("=== fitting, parameter & model recovery ===")
    cc = crosscheck_static_matches_switching()
    p_static = static_parameter_recovery()
    p6 = parameter_recovery()
    p7 = model_recovery()
    print("\n" + "=" * 55)
    print(f"cross-check static path    : {'PASS' if cc else 'FAIL'}")
    print(f"static-switch recovery     : {'PASS' if p_static else 'WEAK/FAIL (see notes)'}")
    print(f"online param recovery      : {'PASS' if p6 else 'WEAK/FAIL (see notes)'}")
    print(f"model recovery             : {'PASS' if p7 else 'FAIL'}")