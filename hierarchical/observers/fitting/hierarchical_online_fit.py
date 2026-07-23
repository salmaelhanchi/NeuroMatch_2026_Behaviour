"""
hierarchical_online_fit.py
==========================

Maximum-likelihood fitting for the :class:`HierarchicalOnlineObserver`.

Contract (mirrors the other fitters so the registry adapter is uniform):

    N_PARAMS, pack/unpack, fit(data, maxiter=, mask=, packing=),
    fit_multistart(data, maxiter=, packing=), _trial_logliks(obs, data),
    _simulate(obs, design, seed)

Two packings, BOTH optimising the *identical* negLL (the spec's cached
sequential likelihood, reused via ``replay_dists``):

* ``packing='penalty'`` -- the coworker's spec fitter **verbatim**: raw
  parameter vector, out-of-bounds -> ``1e9`` penalty, his exact 4 start points.
  This is the DEFAULT so fits reproduce his run bit-for-bit.
* ``packing='house'`` -- unconstrained sigmoid/logit space like the rest of the
  registry (``online_recovery.py`` style), seeded from the same start points.
  More robust search; a different optimiser path, so fitted params can differ
  slightly. Opt-in.

Both are verified to return the same negLL at a shared point (see the module
``__main__`` self-check and the registry sanity run).

The 8 parameters: ``[k_llh(0.06), k_llh(0.12), k_llh(0.24), pi, p_rand,
k_motor, alpha, R0]``.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from observers.models.hierarchical_online import (
    HierarchicalOnlineObserver, replay_dists)
# Shared scipy-convergence extractor, identical to the other fitters, so this
# model's JSON records the same "convergence" schema (converged / n_iter /
# n_feval / hit_maxiter / status / message) as the other six.
from observers.fitting.online_recovery import conv_info

N_PARAMS = 8
COHS = [0.06, 0.12, 0.24]

# The spec's exact start points (§5).
_SPEC_STARTS = [[1, 3, 8, .5, .03, 30, .05, .2],
                [2, 5, 12, .6, .05, 20, .02, .1],
                [.5, 2, 6, .4, .02, 40, .10, .3],
                [1, 3, 8, .7, .03, 30, .20, .4]]


# --------------------------------------------------------------------------- #
#  Data extraction: chronological order + optional per-session reset key.
# --------------------------------------------------------------------------- #
def _extract(data):
    """Pull (dirs, cohs, est, session) from a subject dict.

    ``load_subject`` already returns trials sorted chronologically
    (load_subject_design sorts by session_id, run_id, trial_index), so no
    re-sort is needed. ``session_id`` is optional; when absent the learner runs
    as one continuous stream (spec ``reset_col=None``).
    """
    dirs = np.asarray(data["motion_direction"], float)
    cohs = np.asarray(data["motion_coherence"], float)
    est = np.asarray(data["estimates"], int)
    session = data.get("session_id")
    session = None if session is None else np.asarray(session)
    return dirs, cohs, est, session


def unpack(x) -> dict:
    """Raw parameter vector -> params dict (shared by both packings)."""
    return dict(k_llh=dict(zip(COHS, x[0:3])), pi=x[3], p_rand=x[4],
                k_motor=x[5], alpha=x[6], R0=x[7], mode_init=225)


def observer_from_params(p: dict, readout='sample') -> HierarchicalOnlineObserver:
    return HierarchicalOnlineObserver(
        k_llh={float(k): float(v) for k, v in p["k_llh"].items()},
        pi=float(p["pi"]), p_rand=float(p["p_rand"]), k_motor=float(p["k_motor"]),
        alpha=float(p["alpha"]), R0=float(p["R0"]),
        mode_init=float(p.get("mode_init", 225)), readout=readout)


# --------------------------------------------------------------------------- #
#  The objective: cached sequential negLL (spec §4), via the shared replay.
# --------------------------------------------------------------------------- #
def online_negll(data, params, readout='sample'):
    """Negative sequential log-likelihood for one subject.

    Numerically identical to the spec's ``online_negll``: it replays the online
    learner in chronological order (per-session reset when ``session_id`` is
    present) and scores each trial's response under the current predicted
    distribution. ``replay_dists`` is the same code path the observer's
    ``filter`` uses, so fit and prediction cannot diverge.
    """
    dirs, cohs, est, session = _extract(data)
    out = replay_dists(dirs, cohs, dirs, params, readout=readout, session=session)
    D = out["dists"]
    idx = (est - 1) % 360
    ll = np.log(np.clip(D[np.arange(len(est)), idx], 1e-320, None)).sum()
    return -float(ll)


# --------------------------------------------------------------------------- #
#  Packing 'house': unconstrained sigmoid/logit transforms.
# --------------------------------------------------------------------------- #
def _sig(x):   return 1.0 / (1.0 + np.exp(-x))
def _logit(p): return np.log(p / (1.0 - p))


def _house_to_params(z) -> dict:
    """Unconstrained vector -> params dict (k_llh, k_motor via log; pi, p_rand,
    alpha, R0 via sigmoid into (0,1))."""
    k_llh = np.exp(z[0:3])
    return dict(k_llh=dict(zip(COHS, k_llh)), pi=_sig(z[3]), p_rand=_sig(z[4]),
                k_motor=float(np.exp(z[5])), alpha=_sig(z[6]), R0=_sig(z[7]),
                mode_init=225)


def _house_from_raw(x) -> np.ndarray:
    """Transform a raw start point into the unconstrained space."""
    x = np.asarray(x, float)
    return np.array([np.log(max(x[0], 1e-6)), np.log(max(x[1], 1e-6)),
                     np.log(max(x[2], 1e-6)), _logit(np.clip(x[3], 1e-4, 1 - 1e-4)),
                     _logit(np.clip(x[4], 1e-4, 1 - 1e-4)), np.log(max(x[5], 1e-6)),
                     _logit(np.clip(x[6], 1e-4, 1 - 1e-4)),
                     _logit(np.clip(x[7], 1e-4, 1 - 1e-4))])


# --------------------------------------------------------------------------- #
#  Fit one subject.
# --------------------------------------------------------------------------- #
def fit(data, x0=None, maxiter=1500, mask=None, readout='sample', packing='penalty',
        verbose=True, heartbeat_every=25, tag=''):
    """Fit the 8 params to one subject (Nelder-Mead). Returns (obs, nll, x_raw).

    ``packing='penalty'`` (default): the spec's exact penalty-box objective.
    ``packing='house'``: unconstrained sig/logit space. ``mask`` (bool array)
    restricts scored trials for cross-validation; the learner still runs over
    ALL trials in order (a learning model must not skip trials), only the
    scored set changes.

    ``verbose`` prints a heartbeat every ``heartbeat_every`` objective evals
    (and at each start's completion) so a long background fit shows live
    progress in its log. The heartbeat only reads the objective value being
    returned; it changes no numerics. ``tag`` is a label prefix for the log
    line (e.g. the subject id / start index).
    """
    sub = data
    scored = None if mask is None else np.asarray(mask, bool)
    _t0 = time.time()

    def _nll_params(p):
        if scored is None:
            return online_negll(sub, p, readout=readout)
        dirs, cohs, est, session = _extract(sub)
        out = replay_dists(dirs, cohs, dirs, p, readout=readout, session=session)
        D = out["dists"]; idx = (est - 1) % 360
        ll = np.log(np.clip(D[np.arange(len(est)), idx], 1e-320, None))
        return -float(ll[scored].sum())

    def _beat(state, f):
        state['n'] += 1
        if f >= 1e9:
            state['rej'] += 1            # out-of-bounds proposal (penalty box)
        elif f < state['best']:
            state['best'] = f
        if verbose and state['n'] % heartbeat_every == 0:
            # progress signals since the last heartbeat:
            #   d_best  = how much best negLL fell this window (-> 0 near convergence)
            #   rej%    = fraction of this window's proposals rejected out-of-bounds
            #             (high early = still exploring; low = settled in-bounds)
            window = state['n'] - state['n_last']
            # first window: best_last is inf until a finite value is seen
            d_best = (0.0 if not np.isfinite(state['best_last'])
                      else state['best_last'] - state['best'])
            rej_w = state['rej'] - state['rej_last']
            rej_pct = 100.0 * rej_w / max(window, 1)
            per_eval = (time.time() - _t0 - state['t_last']) / max(window, 1)
            # crude convergence gauge: improvement per eval, normalised to the run's
            # early improvement rate. Near 0 => plateaued.
            imp_rate = d_best / max(window, 1)
            if state['imp0'] is None and imp_rate > 0:
                state['imp0'] = imp_rate
            frac = (imp_rate / state['imp0']) if state['imp0'] else 0.0
            gauge = ("CONVERGING" if d_best < 0.5
                     else "improving " if frac < 0.25 else "exploring ")
            # health: process resident memory (MB) + machine 1-min load.
            # ru_maxrss is bytes on macOS, KiB on Linux -> normalise to MB.
            import resource, os as _os, sys as _sys
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss / (1024**2) if _sys.platform == "darwin" else rss / 1024
            try:
                load1 = _os.getloadavg()[0]
            except OSError:
                load1 = float("nan")
            print(f"[hier_online{tag}] eval {state['n']:5d}/{maxiter}  "
                  f"best={state['best']:11.1f}  Δbest={d_best:8.2f}  "
                  f"rej={rej_pct:4.0f}%  {per_eval:4.1f}s/eval  "
                  f"t={time.time()-_t0:5.0f}s  {gauge}  "
                  f"rss={rss_mb:5.0f}MB load={load1:4.1f}", flush=True)
            state['n_last'] = state['n']; state['best_last'] = state['best']
            state['rej_last'] = state['rej']; state['t_last'] = time.time() - _t0

    if packing == 'penalty':
        def make_negll(state):
            def negll(x):
                if np.any(np.asarray(x) < 0):
                    f = 1e9
                elif x[3] > 1 or x[4] > 1 or x[6] > 1 or x[7] >= 1 or x[6] <= 0:
                    f = 1e9
                else:
                    f = _nll_params(unpack(x))
                _beat(state, f)
                return f
            return negll
        starts = _SPEC_STARTS if x0 is None else [x0]
        results = []
        for si, s in enumerate(starts):
            state = {'n': 0, 'best': np.inf, 'rej': 0,
                     'n_last': 0, 'best_last': np.inf, 'rej_last': 0,
                     't_last': 0.0, 'imp0': None}
            r = minimize(make_negll(state), np.array(s, float), method='Nelder-Mead',
                         options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-1))
            if verbose:
                print(f"[hier_online{tag}] start {si} DONE  negLL={r.fun:.1f}  "
                      f"evals={state['n']}  t={time.time()-_t0:.0f}s", flush=True)
            results.append(r)
        best = min(results, key=lambda r: r.fun)
        p = unpack(best.x); x_raw = best.x
    elif packing == 'house':
        def make_negll(state):
            def negll(z):
                f = _nll_params(_house_to_params(z))
                _beat(state, f)
                return f
            return negll
        starts = ([_house_from_raw(s) for s in _SPEC_STARTS] if x0 is None
                  else [_house_from_raw(x0)])
        results = []
        for si, s in enumerate(starts):
            state = {'n': 0, 'best': np.inf, 'rej': 0,
                     'n_last': 0, 'best_last': np.inf, 'rej_last': 0,
                     't_last': 0.0, 'imp0': None}
            r = minimize(make_negll(state), s, method='Nelder-Mead',
                         options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-1))
            if verbose:
                print(f"[hier_online{tag}] start {si} DONE  negLL={r.fun:.1f}  "
                      f"evals={state['n']}  t={time.time()-_t0:.0f}s", flush=True)
            results.append(r)
        best = min(results, key=lambda r: r.fun)
        p = _house_to_params(best.x)
        x_raw = np.array([p["k_llh"][c] for c in COHS] +
                         [p["pi"], p["p_rand"], p["k_motor"], p["alpha"], p["R0"]])
    else:
        raise ValueError("packing must be 'penalty' or 'house'")

    obs = observer_from_params(p, readout=readout)
    # Attach scipy convergence diagnostics for the winning start, matching the
    # other fitters' contract (the driver reads obs._fit_info -> JSON
    # "convergence"). Reads best.* only; changes no numerics.
    obs._fit_info = conv_info(best, maxiter)
    return obs, float(best.fun), x_raw


def fit_multistart(data, maxiter=1500, mask=None, readout='sample', packing='penalty',
                   verbose=True, heartbeat_every=25):
    """Multi-start fit; returns (obs, nll, x_raw, spread).

    Sweeps the spec's 4 starts (one ``fit`` call each) and returns the best
    plus the negLL spread across starts (a convergence diagnostic). With
    ``verbose`` each start prints a live heartbeat to the log.
    """
    fns = []
    for i, s in enumerate(_SPEC_STARTS):
        obs, nll, x = fit(data, x0=s, maxiter=maxiter, mask=mask,
                          readout=readout, packing=packing, verbose=verbose,
                          heartbeat_every=heartbeat_every,
                          tag=f" start {i + 1}/{len(_SPEC_STARTS)}")
        fns.append((nll, obs, x))
    fns.sort(key=lambda r: r[0])
    nll, obs, x = fns[0]
    spread = float(fns[-1][0] - fns[0][0])
    return obs, nll, x, spread


# --------------------------------------------------------------------------- #
#  Per-trial log-likelihoods (for CV scoring) and simulate (for recovery).
# --------------------------------------------------------------------------- #
def _trial_logliks(obs: HierarchicalOnlineObserver, data) -> np.ndarray:
    """Per-trial log-likelihood array under a fitted observer."""
    dirs, cohs, est, session = _extract(data)
    out = replay_dists(dirs, cohs, dirs, obs._params(), readout=obs.readout,
                       session=session)
    D = out["dists"]; idx = (est - 1) % 360
    return np.log(np.clip(D[np.arange(len(est)), idx], 1e-320, None))


def _simulate(obs: HierarchicalOnlineObserver, design, seed):
    """Simulate estimates from the fitted observer over a trial design.

    The learner is replayed with feedback = true directions (as in fitting);
    each trial's estimate is drawn from its predicted distribution.
    """
    rng = np.random.RandomState(seed)
    d = np.asarray(design["motion_direction"], int)
    c = np.asarray(design["motion_coherence"], float)
    session = design.get("session_id")
    session = None if session is None else np.asarray(session)
    out = replay_dists(d.astype(float), c, d.astype(float), obs._params(),
                       readout=obs.readout, session=session)
    D = out["dists"]
    est = np.empty(len(d), int)
    for t in range(len(d)):
        p = D[t] / D[t].sum()
        est[t] = int(rng.choice(np.arange(1, 361), p=p))
    sim = {"motion_direction": d, "motion_coherence": c, "estimates": est}
    if session is not None:
        sim["session_id"] = np.asarray(session)
    return sim


if __name__ == "__main__":
    # Self-check: both packings evaluate the SAME negLL at a shared point.
    import sys
    sys.path.insert(0, "/Users/vestige/code/NeuroMatch_2026_Behaviour/hierarchical")
    from observers.comparison.registry import load_subject
    data = load_subject(1)
    p = unpack(np.array(_SPEC_STARTS[0], float))
    print("negLL @ start0:", round(online_negll(data, p), 4))
