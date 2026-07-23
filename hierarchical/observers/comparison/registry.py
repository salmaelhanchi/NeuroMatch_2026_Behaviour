"""
registry.py — the model registry (extensibility spine)
=======================================================

One place that knows how to fit, score, and plot every model in the comparison.
Every pipeline stage iterates ``MODEL_REGISTRY``, so **adding a model is a
single entry here** — no stage code changes.

Why a registry at all
----------------------
The models come from two families with DIFFERENT fitter signatures:

* the **Switch** family is non-learning; its ``fit`` returns a 4-tuple
  ``(obs, nll, aic, bic)`` and its NLL call takes ``(estimates, theta_true,
  coherence, prior_label)``.
* the **HB** family learns a belief; its ``fit`` returns ``(obs, nll, x)`` and
  its NLL call takes ``(estimates, directions, coherences, feedback=...)``.

Each ``ModelSpec`` wraps a model behind ONE uniform interface the stages use:

    spec.fit(data, maxiter=, mask=)        -> FitResult(obs, nll, x, n_params, spread)
    spec.trial_logliks(obs, data)          -> np.ndarray of per-trial log-lik
    spec.n_params, spec.color, spec.grid_deg, spec.learns

To add a model:
    1. write its observer (observers/models/) and, if it fits parameters,
       a house-style fitter (observers/fitting/) exposing fit/nll/pack/unpack.
    2. add ONE ModelSpec entry to MODEL_REGISTRY below.
    3. rerun the pipeline. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from observers.helpers.dataset import load_subject_design
from observers.helpers.paths import DATA_CSV


# --------------------------------------------------------------------------- #
#  Uniform result container
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    obs: object
    nll: float
    x: Optional[np.ndarray]
    n_params: int
    start_spread: float = 0.0     # NLL range across multi-start (0 if single)
    extra: dict = field(default_factory=dict)


# Multi-start default: how many starts for the single-start (non-learning /
# geometric) models. HB-Adaptive has its own bespoke 3-start scheme. Set to 1
# to disable (e.g. quick smoke runs).
N_STARTS = 10   # matches Laquitaine & Gardner's ~10 initial starting points

# Nelder-Mead function/parameter tolerance for the paper-standard fit. Laquitaine
# & Gardner fit ALL observers to "very strict function and parameter tolerances
# of 1e-4" (Methods, "Model Fit Convergence"); "more relaxed tolerance worsened
# the fit." The three headline models (switch, basic_bayes, hb_adaptive) are all
# fit at this tolerance so their AIC/BIC/CV comparison is on equal footing and
# matches the paper. (The earlier house default was a looser 1e-2, which fit the
# competitors below the paper's bar and biased in-sample selection toward switch.)
PAPER_TOL = 1e-4


def _starts_for(mask):
    """Multi-start the REPORTED point fit (mask is None); use a single start
    inside CV folds (mask given). CV measures held-out generalization and
    averages over 5 folds, so per-fold global-optimum search is unnecessary and
    would multiply runtime by N_STARTS*n_folds. This mirrors the paper, which
    multi-started its reported fits; CV is our addition."""
    return 1 if mask is not None else N_STARTS


# Structured starts matching Laquitaine & Gardner: the base start plus width
# grid at 8x stronger / weaker. For the cheap models we approximate their
# 10-start recipe with the base + jittered perturbations (scale-free in the
# transformed space); STARTS controls how many.
def multistart(fit_one, base_x0, n_starts=N_STARTS, jitter=0.4, seed=0):
    """Run ``fit_one(x0) -> (obs, nll, x, res)`` from ``n_starts`` starts: the
    base start plus Gaussian-jittered perturbations of it (in the fitter's
    transformed/unbounded parameter space, so jitter is scale-free). Keep the
    lowest-NLL result. Returns (obs, nll, x, res, spread) where ``spread`` is the
    NLL range across starts — a convergence diagnostic (large spread => the
    objective is multi-modal / the single-start fit was start-dependent).
    A start that raises is skipped; if all but one fail, spread is 0."""
    rng = np.random.RandomState(seed)
    base = np.asarray(base_x0, dtype=float)
    starts = [base] + [base + jitter * rng.randn(base.size) for _ in range(max(0, n_starts - 1))]
    results = []
    for x0 in starts:
        try:
            obs, nll, x, res = fit_one(x0)
            if np.isfinite(nll):
                results.append((float(nll), obs, x, res))
        except Exception:
            continue
    if not results:
        raise RuntimeError("all starts failed")
    results.sort(key=lambda r: r[0])
    nlls = [r[0] for r in results]
    spread = float(max(nlls) - min(nlls))
    best_nll, best_obs, best_x, best_res = results[0]
    return best_obs, best_nll, best_x, best_res, spread


# --------------------------------------------------------------------------- #
#  Model specification — the one interface every stage sees
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str                       # short key, e.g. "hb_adaptive"
    label: str                      # display label, e.g. "HB-Adaptive"
    n_params: int
    color: str                      # fixed plot colour, threaded across panels
    grid_deg: int                   # direction-grid resolution (for comparability note)
    learns: bool                    # does it learn a latent across trials?
    _fit: Callable                  # (data, maxiter, mask) -> FitResult
    _trial_logliks: Callable        # (obs, data) -> per-trial log-lik array
    _simulate: Optional[Callable] = None   # (obs, design, seed) -> data dict (for recovery)
    _rebuild: Optional[Callable] = None     # (params dict) -> observer (from saved fit)
    _predict: Optional[Callable] = None     # (obs, data) -> (N,360) per-trial predicted dists

    def fit(self, data, maxiter: int = 400, mask=None) -> FitResult:
        return self._fit(data, maxiter, mask)

    def trial_logliks(self, obs, data) -> np.ndarray:
        return self._trial_logliks(obs, data)

    def simulate(self, obs, design, seed):
        if self._simulate is None:
            raise NotImplementedError(f"{self.name} has no simulate() (needed for recovery)")
        data = self._simulate(obs, design, seed)
        # every simulated dataset must carry prior_std so ANY model (incl. the
        # Switch, whose fitter keys on it) can be fit to it during model recovery
        if "prior_std" not in data and hasattr(design, "__getitem__"):
            try:
                data["prior_std"] = np.asarray(design["prior_std"], int)
            except Exception:
                pass
        return data

    def rebuild(self, params: dict):
        """Reconstruct a fitted observer from a saved params dict (batch JSON)."""
        if self._rebuild is None:
            raise NotImplementedError(f"{self.name} has no rebuild()")
        return self._rebuild(params)

    def predict_distributions(self, obs, data) -> np.ndarray:
        """Per-trial predicted response distribution, shape (N, 360)."""
        if self._predict is None:
            raise NotImplementedError(f"{self.name} has no predict_distributions()")
        return self._predict(obs, data)


# --------------------------------------------------------------------------- #
#  Shared data loader (all models consume the same subject dict)
# --------------------------------------------------------------------------- #
def load_subject(sid: int) -> dict:
    d = load_subject_design(str(DATA_CSV), int(sid))
    return dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                prior_std=d.prior_std.values.astype(int),
                estimates=d.estimate_dir.values.astype(int),
                # Additive: session_id lets learning models (hierarchical_online)
                # reset their online belief at each session boundary. The other
                # six models never read this key, so their inputs are unchanged.
                session_id=d.session_id.values.astype(int))


ALL_SUBJECTS: List[int] = list(range(1, 13))


# --------------------------------------------------------------------------- #
#  Adapters: wrap each fitter family into the uniform FitResult interface
# --------------------------------------------------------------------------- #
def _hb_adaptive_spec() -> ModelSpec:
    from observers.fitting import hb_adaptive_confidence_fit as F

    from observers.models.hb_adaptive_confidence import HBAdaptiveConfidenceObserver

    def _fit(data, maxiter, mask):
        # Paper parity: fit through the SAME registry multistart() as switch and
        # basic_bayes — N_STARTS (10) jittered starts for the reported point fit,
        # 1 inside a CV fold — at the paper tolerance PAPER_TOL (1e-4). Previously
        # this used F.fit_multistart's bespoke 3-start scheme at the looser 1e-2,
        # which left HB-Adaptive under-searched relative to its competitors.
        def fit_one(x0):
            obs, nll, x = F.fit(data, x0=x0, maxiter=maxiter, mask=mask, tol=PAPER_TOL)
            return obs, float(nll), x, getattr(obs, "_fit_info", None)

        base_x0 = F.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.05)
        obs, nll, x, info, spread = multistart(fit_one, base_x0, n_starts=_starts_for(mask))
        return FitResult(obs, nll, x, F.N_PARAMS, spread)

    def _rebuild(params):
        kl = params["k_like"]
        k_like = {0.06: kl["0.06"], 0.12: kl["0.12"], 0.24: kl["0.24"]}
        return HBAdaptiveConfidenceObserver(
            k_like=k_like, k_motor=params["k_motor"],
            p_random=params["p_random"], lam=params["lam"])

    def _predict(obs, data):
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         feedback=data["motion_direction"], sample=False)
        return np.array(out["dists"])

    return ModelSpec(
        name="hb_adaptive", label="HB-Adaptive", n_params=F.N_PARAMS,
        color="#d1495b", grid_deg=360, learns=True,
        _fit=_fit, _trial_logliks=F._trial_logliks, _simulate=F._simulate,
        _rebuild=_rebuild, _predict=_predict)


def _switch_spec() -> ModelSpec:
    from observers.fitting import switching_observer_fit as F

    def _trial_logliks(obs, data):
        # Switch is non-learning; score each trial's response under its
        # per-condition predicted distribution. prior_std -> label key.
        d = np.asarray(data["motion_direction"], int)
        c = np.asarray(data["motion_coherence"], float)
        e = np.asarray(data["estimates"], int)
        ps = np.asarray(data["prior_std"], int)
        ll = np.empty(len(e))
        cache = {}
        for t in range(len(e)):
            key = (int(d[t]), float(c[t]), str(int(ps[t])))
            dist = cache.get(key)
            if dist is None:
                dist = obs.estimate_distribution(int(d[t]), float(c[t]), str(int(ps[t])))
                # Guard: an extreme fitted concentration can leave a NaN or a
                # true 0 in the predicted pmf for a held-out response bin.
                # np.log(max(nan, floor)) is nan (max propagates nan), which
                # poisons the whole CV fold. Sanitize NaN->0 so the floor below
                # bounds every bin (1e-12 caps one bad bin at ~27.6 nats).
                dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
                cache[key] = dist
            ll[t] = np.log(max(float(dist[(e[t] - 1) % 360]), 1e-12))
        return ll

    def _fit(data, maxiter, mask):
        import os
        from observers.fitting.online_recovery import (
            fit_static, fit_static_cmaes, pack_static, conv_info)
        # Two paper-standard knobs for the ill-conditioned k=9 switch:
        #  SWITCH_FIT_TOL   — Nelder-Mead tolerance. Default PAPER_TOL (1e-4, the
        #                     paper's "very strict" value, now shared by all three
        #                     headline models); override to 1e-2 only to reproduce
        #                     the old looser fits.
        #  SWITCH_OPTIMIZER — "nm" (default, Nelder-Mead) or "cmaes". The paper
        #                     fit the Switch model with BOTH NM and CMA-ES
        #                     (Hansen & Kern 2004) and reported they agree; CMA-ES
        #                     is the robustness optimizer for the "noisy and
        #                     ill-conditioned" surface with "many local maxima".
        # Both default to the unchanged house behaviour, so the normal pipeline
        # is bit-identical and no other fit_static caller is affected.
        tol = float(os.environ.get("SWITCH_FIT_TOL", str(PAPER_TOL)))
        optimizer = os.environ.get("SWITCH_OPTIMIZER", "nm").lower()
        sub = (data if mask is None else
               {k: (np.asarray(v)[mask] if hasattr(v, "__len__") else v)
                for k, v in data.items()})

        base_x0 = pack_static({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                              {80: 0.7, 40: 2.8, 20: 8.7, 10: 33.0}, 30.0, 0.05)

        if optimizer in ("cmaes", "cma", "cma-es"):
            # CMA-ES is population-based: ONE run already explores the multimodal
            # surface (its whole point), so it does not need NM's 10 restarts —
            # doing so would run 10 full CMA-ES fits per subject for no gain. Use
            # a few distinct seeds as a genuine multimodality diagnostic (spread
            # across seeds), or 1 inside a CV fold. Each fit is ~150s.
            n_cma = 1 if mask is not None else 3
            def fit_one(x0, seed):
                theta, nll, aic, info = fit_static_cmaes(
                    sub, x0=x0, seed=seed, return_result=True)
                return F.observer_from_theta(theta), float(nll), theta, info
            results = [fit_one(base_x0, s) for s in range(n_cma)]
            results.sort(key=lambda r: r[1])
            obs, nll, x, res = results[0]
            spread = float(results[-1][1] - results[0][1]) if n_cma > 1 else 0.0
        else:
            def fit_one(x0):
                theta, nll, aic, res = fit_static(sub, maxiter=maxiter,
                                                  return_result=True, x0=x0, tol=tol)
                return F.observer_from_theta(theta), float(nll), theta, res
            obs, nll, x, res, spread = multistart(fit_one, base_x0, n_starts=_starts_for(mask))
        # conv_info expects a scipy OptimizeResult; CMA-ES returns its own info
        # dict already in the right shape — pass it through, wrap NM's.
        obs._fit_info = res if isinstance(res, dict) else conv_info(res, maxiter)
        return FitResult(obs, nll, None, F.N_PARAMS, spread)

    def _simulate(obs, design, seed):
        rng = np.random.RandomState(seed)
        d = np.asarray(design["motion_direction"], int)
        c = np.asarray(design["motion_coherence"], float)
        ps = np.asarray(design["prior_std"], int)
        est = np.empty(len(d), int)
        for t in range(len(d)):
            dist = obs.estimate_distribution(int(d[t]), float(c[t]), str(int(ps[t])))
            est[t] = int(rng.choice(np.arange(1, 361), p=dist / dist.sum()))
        return {"motion_direction": d, "motion_coherence": c,
                "prior_std": ps, "estimates": est}

    def _rebuild(params):
        from observers.models.switching_observer import SwitchingObserver
        kl = params["k_like"]; kp = params["k_prior"]
        return SwitchingObserver(
            k_like={float(k): v for k, v in kl.items()},
            k_prior={str(k): v for k, v in kp.items()},
            k_motor=params["k_motor"], p_random=params["p_random"])

    def _predict(obs, data):
        d = np.asarray(data["motion_direction"], int)
        c = np.asarray(data["motion_coherence"], float)
        ps = np.asarray(data["prior_std"], int)
        dists = np.empty((len(d), 360))
        cache = {}
        for t in range(len(d)):
            key = (int(d[t]), float(c[t]), str(int(ps[t])))
            dist = cache.get(key)
            if dist is None:
                dist = obs.estimate_distribution(int(d[t]), float(c[t]), str(int(ps[t])))
                cache[key] = dist
            dists[t] = dist
        return dists

    return ModelSpec(
        name="switch", label="Switch", n_params=F.N_PARAMS,
        color="#30638e", grid_deg=360, learns=False,
        _fit=_fit, _trial_logliks=_trial_logliks, _simulate=_simulate,
        _rebuild=_rebuild, _predict=_predict)


def _hb_rachel_spec() -> ModelSpec:
    from observers.fitting import hb_rachel_fit as F

    def _fit(data, maxiter, mask):
        # Multi-start on the reported point fit (N_STARTS), single start inside
        # a CV fold, identical to every other learning model. F.fit() is
        # single-start from one fixed x0, so calling it directly left the
        # reported fits under-converged (start_spread was hardcoded 0.0); route
        # it through the shared multistart() helper instead.
        base_x0 = F.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.6, 30.0, 0.05, 0.05)

        def fit_one(x0):
            obs, nll, x = F.fit(data, x0=x0, maxiter=maxiter, mask=mask)
            return obs, float(nll), x, getattr(obs, "_fit_info", None)

        obs, nll, x, res, spread = multistart(
            fit_one, base_x0, n_starts=_starts_for(mask))
        return FitResult(obs, nll, x, 7, spread)

    def _rebuild(params):
        from observers.models.hb_rachel import HBRachelObserver
        kl = params["k_like"]
        k_like = {0.06: kl["0.06"], 0.12: kl["0.12"], 0.24: kl["0.24"]}
        return HBRachelObserver(
            k_like=k_like, alpha=params["alpha"], k_motor=params["k_motor"],
            p_random=params["p_random"], lam=params["lam"])

    def _predict(obs, data):
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         feedback=data["motion_direction"], sample=False)
        return np.array(out["dists"])

    return ModelSpec(
        name="hb_rachel", label="HB-Rachel", n_params=7,
        color="#edae49", grid_deg=360, learns=True,
        _fit=_fit, _trial_logliks=F._trial_logliks, _simulate=F._simulate,
        _rebuild=_rebuild, _predict=_predict)


def _recombined_spec() -> ModelSpec:
    """HB-Rachel's engine + Salma's integrate-BEFORE combination rule. Same 7
    params and pack/unpack transforms as HB-Rachel (it subclasses it); the only
    difference is the observer class built from theta."""
    from observers.fitting import hb_rachel_fit as F
    from observers.models.hb_integrate_before import HBIntegrateBeforeObserver

    def _unpack(theta):
        k_like = {0.06: np.exp(theta[0]), 0.12: np.exp(theta[1]), 0.24: np.exp(theta[2])}
        return HBIntegrateBeforeObserver(
            k_like=k_like, alpha=F._sig(theta[3]), k_motor=np.exp(theta[4]),
            p_random=F._sig(theta[5]), lam=F._sig(theta[6]))

    def _trial_logliks(obs, data):
        d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
        out = obs.filter(d, c, feedback=d, sample=False)
        return np.array([np.log(max(out["dists"][t][(int(e[t]) - 1) % 360], 1e-320))
                         for t in range(len(e))])

    def _nll_masked(obs, data, mask):
        ll = _trial_logliks(obs, data)
        return -float(ll.sum() if mask is None else ll[mask].sum())

    def _fit(data, maxiter, mask):
        from scipy.optimize import minimize
        def obj(theta):
            try:
                v = _nll_masked(_unpack(theta), data, mask)
                return v if np.isfinite(v) else 1e12
            except Exception:
                return 1e12
        x0 = F.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.6, 30.0, 0.05, 0.05)
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
        from observers.fitting.online_recovery import conv_info
        obs = _unpack(res.x); obs._fit_info = conv_info(res, maxiter)
        return FitResult(obs, float(res.fun), res.x, 7, 0.0)

    def _rebuild(params):
        kl = params["k_like"]
        return HBIntegrateBeforeObserver(
            k_like={0.06: kl["0.06"], 0.12: kl["0.12"], 0.24: kl["0.24"]},
            alpha=params["alpha"], k_motor=params["k_motor"],
            p_random=params["p_random"], lam=params["lam"])

    def _predict(obs, data):
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         feedback=data["motion_direction"], sample=False)
        return np.array(out["dists"])

    def _simulate(obs, design, seed):
        rng = np.random.RandomState(seed)
        d = np.asarray(design["motion_direction"], int)
        c = np.asarray(design["motion_coherence"], float)
        out = obs.filter(d, c, feedback=d, sample=True, rng=rng)
        return {"motion_direction": d, "motion_coherence": c, "estimates": out["responses"]}

    return ModelSpec(
        name="recombined", label="Recombined", n_params=7,
        color="#66a182", grid_deg=360, learns=True,
        _fit=_fit, _trial_logliks=_trial_logliks, _simulate=_simulate,
        _rebuild=_rebuild, _predict=_predict)


def _hb_salma_spec() -> ModelSpec:
    """HB-Salma (geometric forget, integrate-before, native 72-bin grid). Scored
    on the up-sampled 360-grid (grid='deg360') so its NLL is comparable to the
    other models. 6 params: rho + 3 sensory kappas + motor kappa + lapse.
    No house fitter exists, so a compact Nelder-Mead fitter lives here."""
    from observers.models.hb_salma import HBSalmaObserver

    def _sig(x):   return 1.0 / (1.0 + np.exp(-x))
    def _logit(p): return np.log(p / (1.0 - p))
    COHS = (0.06, 0.12, 0.24)

    def _unpack(theta):
        return HBSalmaObserver(
            rho=_sig(theta[0]),
            sensory_kappas=(np.exp(theta[1]), np.exp(theta[2]), np.exp(theta[3])),
            motor_kappa=np.exp(theta[4]), lapse=_sig(theta[5]))

    def _pack(rho, sk, motor, lapse):
        return np.array([_logit(min(max(rho, 1e-3), 1 - 1e-3)),
                         np.log(sk[0]), np.log(sk[1]), np.log(sk[2]),
                         np.log(motor), _logit(min(max(lapse, 1e-4), 0.5))])

    def _trial_logliks(obs, data):
        d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
        out = obs.filter(d, c, feedback=d, grid="deg360")
        return np.array([np.log(max(out["dists"][t][(int(e[t]) - 1) % 360], 1e-320))
                         for t in range(len(e))])

    def _nll_masked(obs, data, mask):
        ll = _trial_logliks(obs, data)
        return -float(ll.sum() if mask is None else ll[mask].sum())

    def _fit(data, maxiter, mask):
        from scipy.optimize import minimize
        from observers.fitting.online_recovery import conv_info
        def obj(theta):
            try:
                v = _nll_masked(_unpack(theta), data, mask)
                return v if np.isfinite(v) else 1e12
            except Exception:
                return 1e12

        def fit_one(x0):
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
            return _unpack(res.x), float(res.fun), res.x, res

        base_x0 = _pack(0.85, (1.5, 3.0, 8.0), 40.0, 0.02)
        obs, nll, x, res, spread = multistart(fit_one, base_x0, n_starts=_starts_for(mask))
        obs._fit_info = conv_info(res, maxiter)
        return FitResult(obs, nll, x, 6, spread)

    def _rebuild(params):
        sk = params.get("sensory_kappas") or params.get("k_like")
        if isinstance(sk, dict):
            sk = (sk["0.06"], sk["0.12"], sk["0.24"])
        return HBSalmaObserver(rho=params["rho"], sensory_kappas=tuple(sk),
                               motor_kappa=params["motor_kappa"], lapse=params["lapse"])

    def _predict(obs, data):
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         feedback=data["motion_direction"], grid="deg360")
        return np.array(out["dists"])

    def _simulate(obs, design, seed):
        rng = np.random.RandomState(seed)
        d = np.asarray(design["motion_direction"], int)
        c = np.asarray(design["motion_coherence"], float)
        out = obs.filter(d, c, feedback=d, sample=True, rng=rng, grid="deg360")
        return {"motion_direction": d, "motion_coherence": c, "estimates": out["responses"]}

    return ModelSpec(
        name="hb_salma", label="HB-Salma", n_params=6,
        color="#8e6c8a", grid_deg=360, learns=True,
        _fit=_fit, _trial_logliks=_trial_logliks, _simulate=_simulate,
        _rebuild=_rebuild, _predict=_predict)


def _basic_bayes_spec() -> ModelSpec:
    """The paper's always-integrate baseline (BasicBayesianObserver). Structurally
    identical to the Switch — same 9 params (3 k_like + 4 k_prior + k_motor +
    p_random), non-learning, per-condition read-out — but it always integrates
    (no switching / mixture), so it is unimodal by construction. It is the
    reference floor: how much does any variant improve on naive optimal
    integration? Expect valley-depth ~0 (cannot produce two clusters)."""
    from observers.fitting import basic_bayesian_fit as F

    def _trial_logliks(obs, data):
        d = np.asarray(data["motion_direction"], int)
        c = np.asarray(data["motion_coherence"], float)
        e = np.asarray(data["estimates"], int)
        ps = np.asarray(data["prior_std"], int)
        ll = np.empty(len(e)); cache = {}
        for t in range(len(e)):
            key = (int(d[t]), float(c[t]), str(int(ps[t])))
            dist = cache.get(key)
            if dist is None:
                dist = obs.estimate_distribution(int(d[t]), float(c[t]), str(int(ps[t])))
                # Guard against NaN/0 in the predicted pmf (see switch note).
                dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
                cache[key] = dist
            ll[t] = np.log(max(float(dist[(e[t] - 1) % 360]), 1e-12))
        return ll

    def _fit(data, maxiter, mask):
        sub = (data if mask is None else
               {k: (np.asarray(v)[mask] if hasattr(v, "__len__") else v)
                for k, v in data.items()})

        def fit_one(x0):
            obs, nll, aic, bic = F.fit(sub, x0=x0, maxiter=maxiter, tol=PAPER_TOL)
            # basic fitter attaches obs._fit_info; return theta via obs? use x0-space
            return obs, float(nll), None, getattr(obs, "_fit_info", None)

        base_x0 = F.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                         {"80": 0.7, "40": 2.8, "20": 8.7, "10": 33.0}, 30.0, 0.05)
        obs, nll, x, info, spread = multistart(fit_one, base_x0, n_starts=_starts_for(mask))
        # obs already carries its own _fit_info from the winning start
        return FitResult(obs, nll, None, F.N_PARAMS, spread)

    def _simulate(obs, design, seed):
        rng = np.random.RandomState(seed)
        d = np.asarray(design["motion_direction"], int)
        c = np.asarray(design["motion_coherence"], float)
        ps = np.asarray(design["prior_std"], int)
        est = np.empty(len(d), int)
        for t in range(len(d)):
            dist = obs.estimate_distribution(int(d[t]), float(c[t]), str(int(ps[t])))
            est[t] = int(rng.choice(np.arange(1, 361), p=dist / dist.sum()))
        return {"motion_direction": d, "motion_coherence": c,
                "prior_std": ps, "estimates": est}

    def _rebuild(params):
        from observers.models.basic_bayesian import BasicBayesianObserver
        kl = params["k_like"]; kp = params["k_prior"]
        return BasicBayesianObserver(
            k_like={float(k): v for k, v in kl.items()},
            k_prior={str(k): v for k, v in kp.items()},
            k_motor=params["k_motor"], p_random=params["p_random"])

    def _predict(obs, data):
        d = np.asarray(data["motion_direction"], int)
        c = np.asarray(data["motion_coherence"], float)
        ps = np.asarray(data["prior_std"], int)
        dists = np.empty((len(d), 360)); cache = {}
        for t in range(len(d)):
            key = (int(d[t]), float(c[t]), str(int(ps[t])))
            dist = cache.get(key)
            if dist is None:
                dist = obs.estimate_distribution(int(d[t]), float(c[t]), str(int(ps[t])))
                cache[key] = dist
            dists[t] = dist
        return dists

    return ModelSpec(
        name="basic_bayes", label="Basic-Bayes", n_params=F.N_PARAMS,
        color="#a0a0a0", grid_deg=360, learns=False,
        _fit=_fit, _trial_logliks=_trial_logliks, _simulate=_simulate,
        _rebuild=_rebuild, _predict=_predict)


def _hierarchical_online_spec() -> ModelSpec:
    """Hierarchical Online observer: mixture prior + resultant-vector online
    learner (learns prior mean AND width). 8 params; readout='sample'.

    Numerics are the coworker's build spec verbatim (via
    observers.models.hierarchical_online); only the interface is adapted.
    Default fitter packing is 'penalty' (his exact penalty-box search) so fits
    reproduce his run bit-for-bit; 'house' (unconstrained) is available for a
    registry-consistent search.
    """
    from observers.fitting import hierarchical_online_fit as F
    from observers.models.hierarchical_online import HierarchicalOnlineObserver

    READOUT = "sample"
    PACKING = "penalty"

    def _fit(data, maxiter, mask):
        obs, nll, x, spread = F.fit_multistart(
            data, maxiter=maxiter, mask=mask, readout=READOUT, packing=PACKING)
        return FitResult(obs, nll, x, F.N_PARAMS, spread)

    def _rebuild(params):
        # Robust to an incomplete named-params dict: the shared driver's
        # _observer_params extractor uses a fixed attribute list keyed to the
        # other models' names (p_random/k_like) and does not know this model's
        # pi/R0 or its p_rand/k_llh naming, so the saved "params" block can be
        # partial. The full 8-vector is always preserved in "theta"; when the
        # named keys are missing, reconstruct from theta via the fitter's
        # unpack(). This keeps rebuild() total and identical either way.
        needed = ("k_llh", "pi", "p_rand", "k_motor", "alpha", "R0")
        if not all(k in params for k in needed):
            theta = params.get("theta")
            if theta is None:
                raise KeyError(
                    "hierarchical_online rebuild: params incomplete and no "
                    f"'theta' fallback present (have keys {sorted(params)})")
            params = F.unpack(np.asarray(theta, float))
        k_llh = {float(k): float(v) for k, v in params["k_llh"].items()}
        return HierarchicalOnlineObserver(
            k_llh=k_llh, pi=params["pi"], p_rand=params["p_rand"],
            k_motor=params["k_motor"], alpha=params["alpha"], R0=params["R0"],
            mode_init=params.get("mode_init", 225.0),
            readout=params.get("readout", READOUT))

    def _predict(obs, data):
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         feedback=data["motion_direction"],
                         session_id=data.get("session_id"))
        return np.array(out["dists"])

    return ModelSpec(
        name="hierarchical_online", label="Hier-Online", n_params=F.N_PARAMS,
        color="#3a7d44", grid_deg=360, learns=True,
        _fit=_fit, _trial_logliks=F._trial_logliks, _simulate=F._simulate,
        _rebuild=_rebuild, _predict=_predict)


def _reliability_mixture_spec() -> ModelSpec:
    """Reliability-Mixture (Romi's ReliabilityMixtureObserver): the percept is a
    genuine discrete either/or mixture between a prior-centered and a
    likelihood-centered von Mises, with the mixture weight (prior_reliance)
    learned trial-by-trial via a delta rule against a 5-trial feedback window
    that resets at session boundaries. 10 params: 3 sensory kappa + 4 prior
    kappa + reliance learning-rate + motor kappa + lapse. Learns; scored on the
    native 360-grid so its NLL is directly comparable to the other models."""
    from observers.fitting import reliability_mixture_fit as F
    from observers.models.reliability_mixture import ReliabilityMixtureObserver

    def _fit(data, maxiter, mask):
        # Multi-start the reported point fit (mask is None), single start in CV
        # folds — same policy as the other single-start models via multistart().
        def fit_one(x0):
            obs, nll, x = F.fit(data, x0=x0, maxiter=maxiter, mask=mask)
            return obs, nll, x, getattr(obs, "_fit_info", None)

        base_x0 = F.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                         {80: 0.75, 40: 2.8, 20: 8.7, 10: 33.0}, 0.05, 30.0, 0.02)
        obs, nll, x, info, spread = multistart(fit_one, base_x0, n_starts=_starts_for(mask))
        return FitResult(obs, nll, x, F.N_PARAMS, spread)

    def _rebuild(params):
        kl = params["k_like"]; kp = params["k_prior"]
        return ReliabilityMixtureObserver(
            k_like={float(k): v for k, v in kl.items()},
            k_prior={int(k): v for k, v in kp.items()},
            alpha=params["alpha"], k_motor=params["k_motor"], lapse=params["lapse"])

    def _predict(obs, data):
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         data["prior_std"], session_id=data.get("session_id"),
                         feedback=data["motion_direction"], sample=False)
        return np.array(out["dists"])

    return ModelSpec(
        name="reliability_mixture", label="Reliability-Mixture", n_params=F.N_PARAMS,
        color="#c17817", grid_deg=360, learns=True,
        _fit=_fit, _trial_logliks=F._trial_logliks, _simulate=F._simulate,
        _rebuild=_rebuild, _predict=_predict)


# --------------------------------------------------------------------------- #
#  THE REGISTRY — add a model by adding one line here
# --------------------------------------------------------------------------- #
# The registered-model builders — the SINGLE source of truth for "which models
# exist". Adding a model is one entry here (and its _*_spec above); every stage
# that iterates the registry (fits, CV, figure, table, observers.api) then picks
# it up automatically. Insertion order is the canonical display order.
_BUILDERS = {
    "switch":      _switch_spec,        # paper's Switching observer (non-learning)
    "basic_bayes": _basic_bayes_spec,   # paper baseline integrator (always-integrate, unimodal)
    "hb_adaptive": _hb_adaptive_spec,   # the abstract's model (learns alpha AND kappa)
    "hb_rachel":   _hb_rachel_spec,     # fixed-alpha integrator, integrate-after
    "hb_salma":    _hb_salma_spec,      # geometric-forget, integrate-before, 72-bin (scored on 360)
    "recombined":  _recombined_spec,    # Rachel engine + Salma's integrate-BEFORE rule
    "hierarchical_online": _hierarchical_online_spec,  # mixture prior + online-learned mean & width
    "reliability_mixture": _reliability_mixture_spec,  # Romi's discrete-mixture, learned reliance weight
}

# Canonical list of every registered model key, derived from the builders so it
# can never drift. Iterate this to support all models generically.
ALL_MODELS: List[str] = list(_BUILDERS.keys())


def build_registry(names: Optional[List[str]] = None) -> Dict[str, ModelSpec]:
    """Return the active registry. Pass ``names`` to select a subset;
    default is the two headline models (HB-Adaptive vs Switch)."""
    if names is None:
        names = ["hb_adaptive", "switch"]
    reg = {}
    for n in names:
        if n not in _BUILDERS:
            raise KeyError(f"unknown model {n!r}; known: {sorted(_BUILDERS)}")
        reg[n] = _BUILDERS[n]()
    return reg


# convenience default used by stages when the caller doesn't specify
DEFAULT_MODELS = ["hb_adaptive", "switch"]
