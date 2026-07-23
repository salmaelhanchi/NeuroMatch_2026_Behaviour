"""
hierarchical_online.py
======================

The **Hierarchical Online observer** (Laquitaine & Gardner 2018 dataset).

Model in one paragraph
----------------------
Each trial the observer sees noisy motion under a learned directional prior. It
treats the true direction as coming from a **mixture prior** -- a peaked von
Mises *or* a flat uniform -- and reads out the responsibility-weighted
posterior. This reproduces the paper's bimodal "switching" behaviour as
principled Bayesian inference. The prior's **mean and concentration are learned
online, trial by trial, from the feedback** (the true direction shown after each
response) via a leaky resultant-vector delta rule. Because narrow-prior blocks
show clustered directions and wide blocks show spread ones, the learned
concentration automatically comes out strong or weak per block -- the prior
width is *learned, not fitted*.

Provenance
----------
The readout math (``hierarchical_lookup``), the reference lookups
(``girshick_lookup``, ``sampling_lookup``) and the leaky resultant-vector
learner are ported **verbatim** from the coworker's build spec
(IMPLEMENT_hierarchical_online_FULL.md). The only adaptation is that
``vm_pdf`` / ``circ_convolve`` delegate to the repo's
``observers.helpers.circular`` (``von_mises_pdfs`` / ``circular_convolution``),
which have been verified numerically identical to the spec's own copies
(max abs diff ~1.7e-15 on von Mises; convolution bit-identical). No numerics
are changed; only the interface is adapted to the registry.

Parameters (8): ``k_llh x3, pi, p_rand, k_motor, alpha, R0``.
Readouts: ``'sample'`` (default; closest to the paper's Switching observer),
``'select'`` (winner-take-all), ``'average'`` (posterior mean, Basic-Bayesian
limit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from observers.helpers.circular import von_mises_pdfs, circular_convolution

DEG = np.arange(1, 361)


# --------------------------------------------------------------------------- #
#  Base functions -- vm_pdf / circ_convolve delegate to the repo helpers
#  (verified identical to the spec's own copies).
# --------------------------------------------------------------------------- #
def vm_pdf(support_deg, mu_deg, k, norm=True):
    """Von Mises pmf on ``support_deg`` for each mean in ``mu_deg`` (column j).

    Thin wrapper over ``observers.helpers.circular.von_mises_pdfs`` so this
    module and the rest of the registry share one von Mises implementation.
    Returns a ``(len(support), n_means)`` array, columns summing to 1 when
    ``norm``.
    """
    return von_mises_pdfs(np.asarray(support_deg, float),
                          np.asarray(mu_deg, float), float(k), normalize=norm)


def circ_convolve(P, kernel_col):
    """Column-wise circular convolution of ``P`` with ``kernel_col`` (FFT)."""
    return circular_convolution(np.asarray(P, float),
                                np.asarray(kernel_col, float)[:, None])


# --------------------------------------------------------------------------- #
#  Reference readouts (verbatim from the spec) -- used by the sanity checks.
# --------------------------------------------------------------------------- #
def girshick_lookup(motdir, k_llh, mode_prior, k_prior, weight_tail=0.0, readout='MAP'):
    """Static Girshick Bayesian lookup (MAP or BLS). Reference for the §8 checks."""
    m = DEG; mPdfs = vm_pdf(m, motdir, k_llh); like = vm_pdf(DEG, m, k_llh)
    prior = np.tile(vm_pdf(DEG, mode_prior, k_prior), (1, len(m)))
    if weight_tail > 0:
        prior = (1 - weight_tail) * prior + weight_tail * (1 / 360)
    post = like * prior; post = post / post.sum(0, keepdims=True)
    post = np.round(post * 1e6) / 1e6
    if readout == 'MAP':
        perc = [np.where(post[:, i] == post[:, i].max())[0] + 1 for i in range(len(m))]
    else:                                                # BLS = circular posterior mean
        a = np.deg2rad(DEG); perc = []
        for i in range(len(m)):
            p = post[:, i]
            mn = np.rad2deg(np.arctan2((p * np.sin(a)).sum(), (p * np.cos(a)).sum())) % 360
            perc.append(np.array([int(round(mn)) or 360]))
    L = np.zeros((360, np.atleast_1d(motdir).size))
    for i in range(len(m)):
        ps = perc[i]; w = 1.0 / len(ps)
        for p in ps:
            L[p - 1, :] += w * mPdfs[i, :]
    return L / L.sum(0, keepdims=True)


def sampling_lookup(motdir, k_llh, mode_prior, k_prior, weight_tail=0.0):
    """Static sampling readout (posterior-averaged). Reference for the §8 checks."""
    m = DEG; mPdfs = vm_pdf(m, motdir, k_llh); like = vm_pdf(DEG, m, k_llh)
    prior = np.tile(vm_pdf(DEG, mode_prior, k_prior), (1, len(m)))
    if weight_tail > 0:
        prior = (1 - weight_tail) * prior + weight_tail * (1 / 360)
    post = like * prior; post = post / post.sum(0, keepdims=True)
    L = post @ mPdfs
    return L / L.sum(0, keepdims=True)


# --------------------------------------------------------------------------- #
#  The hierarchical readout (mixture prior + responsibility) -- verbatim.
# --------------------------------------------------------------------------- #
def hierarchical_lookup(motdir, k_llh, k_prior, pi, mode_prior=225, readout='select'):
    """Mixture-prior posterior readout.

    ``post(theta|m) ∝ VM(theta; m, k_llh)·[pi·VM(theta; mode_prior, k_prior)
    + (1-pi)/360]``. Returns ``(360, n_dir)`` P(estimate | motdir) before
    lapse/motor noise. ``readout`` in {'sample','average','select'}.
    """
    motdir = np.atleast_1d(motdir).astype(float); m = DEG
    mPdfs = vm_pdf(m, motdir, k_llh); like = vm_pdf(DEG, m, k_llh)
    A = np.tile(vm_pdf(DEG, mode_prior, k_prior), (1, len(m)))
    post = like * (pi * A + (1.0 - pi) / 360.0); post = post / post.sum(0, keepdims=True)
    L = np.zeros((360, len(motdir)))
    if readout == 'sample':
        L = post @ mPdfs
    elif readout == 'average':
        a = np.deg2rad(DEG)
        for i in range(len(m)):
            p = post[:, i]
            mn = int(round(np.rad2deg(np.arctan2((p * np.sin(a)).sum(),
                                                 (p * np.cos(a)).sum())) % 360)) or 360
            L[mn - 1, :] += mPdfs[i, :]
    elif readout == 'select':
        pr = np.round(post * 1e6) / 1e6
        for i in range(len(m)):
            pk = np.where(pr[:, i] == pr[:, i].max())[0] + 1; w = 1.0 / len(pk)
            for p in pk:
                L[p - 1, :] += w * mPdfs[i, :]
    else:
        raise ValueError("readout must be 'average', 'select', or 'sample'")
    return L / L.sum(0, keepdims=True)


# --------------------------------------------------------------------------- #
#  Online learner (leaky resultant vector) + shared per-trial replay.
# --------------------------------------------------------------------------- #
def _init_resultant(R0, mode0):
    """Initial resultant-vector components (Cx, Cy) at angle mode0, length R0."""
    return R0 * np.cos(np.deg2rad(mode0)), R0 * np.sin(np.deg2rad(mode0))


def _learner_state(Cx, Cy):
    """(mu_deg, kappa) from the current resultant vector (spec formulas)."""
    R = min(np.hypot(Cx, Cy), 0.9999)
    k = R * (2 - R * R) / (1 - R * R)
    mu = np.rad2deg(np.arctan2(Cy, Cx)) % 360
    return mu, k


def replay_dists(dirs, cohs, feedback, params, readout='sample', session=None):
    """Replay the online learner over one subject's chronological trials.

    Yields, per trial, the predicted ``(360,)`` estimate distribution
    (after lapse + motor noise), plus the learner trajectories. This is the
    single source of truth the fitter's negLL and the observer's ``filter``
    both build on, so they cannot drift.

    Parameters mirror the spec's ``online_negll`` learner exactly:
    ``dirs``/``feedback`` = true directions (feedback), ``cohs`` = coherence,
    ``session`` = per-trial reset key (belief re-initialised when it changes;
    ``None`` = one continuous learner). Assumes trials already in chronological
    order.
    """
    dirs = np.asarray(dirs, float); cohs = np.asarray(cohs)
    feedback = np.asarray(feedback, float)
    udirs = np.array(sorted(np.unique(dirs))); didx = {int(d): i for i, d in enumerate(udirs)}
    ker = vm_pdf(DEG, 360, params['k_motor'])[:, 0]
    a = params['alpha']; mode0 = params.get('mode_init', 225); R0 = params['R0']
    Cx, Cy = _init_resultant(R0, mode0)
    cache: Dict = {}
    n = len(dirs)
    dists = np.empty((n, 360)); k_traj = np.empty(n); mu_traj = np.empty(n)
    prev = None
    for t in range(n):
        if session is not None and prev is not None and session[t] != prev:
            Cx, Cy = _init_resultant(R0, mode0)
        prev = session[t] if session is not None else None
        mu, k = _learner_state(Cx, Cy)
        k_traj[t] = k; mu_traj[t] = mu
        key = (cohs[t], round(float(np.log(max(k, 1e-3))), 1), round(mu / 4) * 4)
        M = cache.get(key)
        if M is None:
            L = hierarchical_lookup(udirs, params['k_llh'][cohs[t]], k, params['pi'], key[2], readout)
            L = (1 - params['p_rand']) * L + params['p_rand'] / 360
            L = circ_convolve(L, ker)
            M = np.clip(L / L.sum(0, keepdims=True), 1e-320, None); cache[key] = M
        dists[t] = M[:, didx[int(dirs[t])]]
        ang = np.deg2rad(feedback[t])
        Cx = (1 - a) * Cx + a * np.cos(ang); Cy = (1 - a) * Cy + a * np.sin(ang)
    return dict(dists=dists, k_traj=k_traj, mu_traj=mu_traj)


# --------------------------------------------------------------------------- #
#  Observer object (rebuilt from a fitted params dict).
# --------------------------------------------------------------------------- #
@dataclass
class HierarchicalOnlineObserver:
    """Mixture-prior hierarchical observer that learns prior mean & width online.

    Free parameters (8): ``k_llh`` (per coherence, 3), ``pi`` (prior mixing
    weight), ``p_rand`` (lapse), ``k_motor`` (motor precision), ``alpha``
    (learning rate), ``R0`` (initial prior strength). ``mode_init`` fixed 225.
    """
    k_llh: Dict[float, float] = field(
        default_factory=lambda: {0.06: 1.0, 0.12: 3.0, 0.24: 8.0})
    pi: float = 0.6
    p_rand: float = 0.03
    k_motor: float = 30.0
    alpha: float = 0.05
    R0: float = 0.2
    mode_init: float = 225.0
    readout: str = 'sample'

    N_PARAMS = 8

    def _params(self) -> dict:
        return dict(k_llh=self.k_llh, pi=self.pi, p_rand=self.p_rand,
                    k_motor=self.k_motor, alpha=self.alpha, R0=self.R0,
                    mode_init=self.mode_init)

    def filter(self, directions, coherences, feedback=None, session_id=None,
               record_belief: bool = False):
        """Replay the learner; return per-trial predicted distributions.

        API mirrors the other HB observers. ``feedback`` defaults to the true
        directions. ``session_id`` (optional) re-initialises the learner at each
        session boundary (the spec's ``reset_col='session_id'`` behaviour).
        With ``record_belief`` the learned ``k_traj``/``mu_traj`` are returned
        (and ``believed_sd`` derived from ``k_traj`` for parity with the other
        HB models' ``filter`` output).
        """
        directions = np.asarray(directions, int)
        coherences = np.asarray(coherences, float)
        feedback = directions if feedback is None else np.asarray(feedback, int)
        session = None if session_id is None else np.asarray(session_id)
        out = replay_dists(directions, coherences, feedback, self._params(),
                           readout=self.readout, session=session)
        res = {"dists": [out["dists"][t] for t in range(len(directions))]}
        if record_belief:
            from observers.helpers.circular import von_mises_std
            res["k_traj"] = out["k_traj"]
            res["mu_traj"] = out["mu_traj"]
            res["believed_sd"] = np.array([von_mises_std(max(k, 1e-6), self.mode_init)
                                           for k in out["k_traj"]])
        return res

    @staticmethod
    def aic(nll: float, n_params: int = N_PARAMS) -> float:
        return 2.0 * n_params + 2.0 * nll

    @staticmethod
    def bic(nll: float, n_trials: int, n_params: int = N_PARAMS) -> float:
        return n_params * np.log(n_trials) + 2.0 * nll
