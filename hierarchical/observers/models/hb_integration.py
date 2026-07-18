"""
hb_integration.py
=================

**Hierarchical Bayesian *integration* observer** — the model described in the
project abstract, and the missing counterpart to the switching family.

Where ``switching_observer.py`` / ``online_switching_observer.py`` *impose* a
selection between prior and evidence (paper Eq. 6), this observer imposes nothing. It
places a **mixed hyper-prior** on the true direction

    p(theta | kappa, alpha) = alpha * V(theta; 225, kappa)  +  (1 - alpha)/360   (prior)

— a peaked von Mises at the prior mean *unioned with* a uniform component — and
simply reads out the MAP of the resulting posterior. The bimodality that
motivated the Switching observer then **emerges** from Bayesian inference:

  * when the internal measurement lands near 225, the von Mises component
    dominates the posterior and the estimate is pulled toward the prior;
  * when it lands far from 225, the uniform floor dominates and the estimate
    sits at the sensory evidence.

Sweeping the measurement therefore yields a response distribution with a peak
near the stimulus *and* a peak near the prior — with no switch. The "switch"
is a *derived responsibility* (which mixture component explains the
measurement), not a hand-coded competition.

Online learning (the abstract's "trial-by-trial learning of prior precision").
The block manipulation in the experiment *is* a width manipulation (prior SD
10/20/40/80 deg), so the quantity that changes across blocks is the prior
concentration ``kappa``. The observer therefore carries a belief ``b_t(kappa)``
and updates it from feedback, exactly as the switch model does — while the
mixture weight ``alpha`` ("does the prior structure ever apply") is held fixed
across blocks. Learning kappa (not alpha) also keeps this model point-for-point
comparable to ``online_switching_observer.py``: same learned latent, differing
*only* in switch-vs-integration read-out.

Equations are labelled M1-M5 to parallel the switch model's H1-H5.

Modelling choices (fixed so the switch-vs-integration comparison is
like-for-like):

  F1  Learned latent = prior precision kappa; alpha fixed (justified above).
  F2  Belief marginalisation for the estimate = **read-out-then-average**:
      p(est) = sum_kappa b_t(kappa) * readout(kappa), i.e. the same H4
      convention the online switching observer uses. (The alternative "average
      the prior into one posterior, read out once" cannot be precomputed because
      the belief changes every trial, and would change the integration rule
      relative to the switch model — confounding the intended comparison.)
  F3  Read-out = MAP (mode) of the posterior, reusing the paper's Girshick
      push-forward machinery. This is what makes the response distribution, not
      just a point estimate.

Tractability note. The switch model's speed comes from the prior read-out being
a kappa-independent delta at 225, collapsing the belief average to a scalar.
That shortcut does not apply here: the read-out *location* depends on kappa, so
one Girshick push-forward is needed per kappa-grid point. To keep it tractable
this model (a) uses a modest read-out grid (~15 kappa), (b) vectorises the
mode extraction (no per-measurement Python loop), and (c) precomputes the
motor-convolved read-outs once per parameter set. Always time one
``negative_log_likelihood`` call on a real subject before launching a batch
(see hb_integration_verify.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# --- reused, already-verified primitives from the switching layer ------------
from observers.helpers.circular import (
    DIRECTION_SPACE,
    von_mises_pdfs,
    circular_convolution,
    von_mises_std,
)
# the degenerate-posterior closed form, reused so that alpha=1 reduces EXACTLY
# to girshick_map_lookup (the reduction cross-check in the verify script).
from observers.helpers.bayes_lookup import _fix_degenerate_posteriors
from observers.helpers.belief_grid import make_k_grid, forget, bayes_correct

PRIOR_MEAN = 225.0


# ===========================================================================
#  The mixed hyper-prior and its MAP read-out
# ===========================================================================
def mixture_prior(kappa: float, alpha: float,
                  prior_mode: float = PRIOR_MEAN) -> np.ndarray:
    """The mixed hyper-prior p(theta) = alpha*V(theta;mode,kappa) + (1-alpha)/360.

    Returns a proper pmf on the 1..360 grid (sums to 1)."""
    vm = von_mises_pdfs(DIRECTION_SPACE, prior_mode, kappa, normalize=True).ravel()
    return alpha * vm + (1.0 - alpha) / 360.0


def _map_readout(like: np.ndarray, m_pdfs: np.ndarray, prior_vec: np.ndarray,
                 motion_dirs: np.ndarray, di: np.ndarray,
                 rescue: tuple = None) -> np.ndarray:
    """MAP push-forward given a PRECOMPUTED likelihood and measurement pdfs.

    ``like`` (θ×m) and ``m_pdfs`` (m×d) depend only on ``k_like`` — not on
    kappa or alpha — so a caller sweeping kappa at fixed coherence builds them
    ONCE and reuses them across every kappa here. That reuse is the speed fix:
    it removes the 360×360 von-Mises rebuild that otherwise dominates each of
    the ~15 kappa read-outs per coherence.

    The per-measurement loop of the original Girshick code is a boolean mode
    mask plus a single matrix product ``L = P(percept|m) @ P(m|d)``.

    ``rescue = (prior_mode, k_like, kappa)`` enables the alpha=1 underflow
    closed form (needed only for the exact reduction to girshick_map_lookup);
    pass None on the mixture path, where the uniform floor prevents underflow.
    """
    posterior = like * prior_vec[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        posterior = posterior / posterior.sum(axis=0, keepdims=True)
    posterior = np.round(posterior * 1e6) / 1e6      # robust tie detection
    if rescue is not None:
        _fix_degenerate_posteriors(posterior, di, di, *rescue)
    # a measurement column can go all-NaN if the likelihood underflows for the
    # extreme k_like the optimiser probes; treat it as uninformative (uniform)
    # so q>0 and no NaN propagates into the fit.
    bad_col = ~np.isfinite(posterior).any(axis=0)
    if bad_col.any():
        posterior[:, bad_col] = 1.0 / posterior.shape[0]
    mx = posterior.max(axis=0, keepdims=True)
    is_mode = (posterior == mx).astype(float)         # (percept, m); ties kept
    q = is_mode.sum(axis=0, keepdims=True)
    p_percept_given_m = is_mode / np.where(q > 0, q, 1.0)  # each of q modes->1/q
    P = p_percept_given_m @ m_pdfs                    # L[percept,d]=Σ_m P(p|m)P(m|d)
    L = np.full((360, 360), np.nan)
    L[:, motion_dirs - 1] = P
    with np.errstate(invalid="ignore", divide="ignore"):
        L = L / np.nansum(L, axis=0, keepdims=True)
    return L


def mixture_map_lookup(k_like: float, prior_mode: float, kappa: float,
                       alpha: float, motion_dirs: np.ndarray) -> np.ndarray:
    """P(MAP percept | displayed direction) for the *mixture-prior* posterior.

    Standalone convenience wrapper (used by the verify/reduction checks). The
    hot fitting path calls ``_map_readout`` directly with cached ``like`` /
    ``m_pdfs``. With ``alpha = 1`` this returns exactly
    ``girshick_map_lookup(k_like, prior_mode, kappa, motion_dirs)`` (~1e-15).
    """
    di = DIRECTION_SPACE.astype(float)
    motion_dirs = np.atleast_1d(np.asarray(motion_dirs)).astype(int)
    m_pdfs = von_mises_pdfs(di, motion_dirs, k_like, normalize=True)   # (360,D)
    like = von_mises_pdfs(di, di, k_like, normalize=True)             # (360,360)
    prior = mixture_prior(kappa, alpha, prior_mode)
    rescue = (prior_mode, k_like, kappa) if alpha >= 1.0 else None
    return _map_readout(like, m_pdfs, prior, motion_dirs, di, rescue=rescue)


# ===========================================================================
#  The observer
# ===========================================================================
@dataclass
class HBIntegrationObserver:
    """Hierarchical Bayesian integration observer (the abstract's model).

    Free parameters (7):
        k_like   : dict coherence -> sensory concentration k_e            (3)
        alpha    : prior confidence = weight of the von Mises vs uniform  (1)
        k_motor  : motor-noise concentration                             (1)
        p_random : response lapse rate                                   (1)
        lam      : volatility / forgetting on the belief over kappa      (1)

    Fixed structure:
        prior_mean : 225 deg.
        k_grid     : discrete support for BOTH the belief over kappa and the
                     read-out marginalisation (one grid so no interpolation;
                     ~15 points keeps the per-eval cost tractable).
        belief0    : initial belief over kappa; uniform on the grid.
    """

    k_like: Dict[float, float] = field(
        default_factory=lambda: {0.24: 8.0, 0.12: 3.0, 0.06: 1.0})
    alpha: float = 0.6
    k_motor: float = 40.0
    p_random: float = 0.01
    lam: float = 0.0
    prior_mean: float = PRIOR_MEAN
    k_grid: np.ndarray = field(default_factory=lambda: make_k_grid(n=15))

    # precomputed caches (filled by _prepare)
    _prepared_for: tuple = field(default=None, repr=False)
    _readout: dict = field(default_factory=dict, repr=False)   # coh -> (nk,360,D)
    _dir_col: dict = field(default_factory=dict, repr=False)   # direction -> col
    _uniform: np.ndarray = field(default=None, repr=False)
    _obs_table: np.ndarray = field(default=None, repr=False)   # (nk, 360) mixture
    _belief0: np.ndarray = field(default=None, repr=False)

    # ----------------------------------------------------------------------
    def initial_belief(self) -> np.ndarray:
        n = self.k_grid.size
        return np.ones(n) / n

    # ----------------------------------------------------------------------
    #  Precompute everything that does not depend on the (changing) belief.
    #  Rebuilt whenever k_like / alpha / k_motor change (i.e. every optimiser
    #  step) — this is the expensive part; see the tractability note.
    # ----------------------------------------------------------------------
    def _prepare(self, directions: np.ndarray, coherences: np.ndarray):
        sig = (tuple(sorted(self.k_like.items())), float(self.alpha),
               float(self.k_motor), self.k_grid.size)
        if self._prepared_for == sig and self._readout:
            return
        dirs = np.unique(np.asarray(directions, dtype=int))
        cohs = np.unique(np.asarray(coherences, dtype=float))
        self._dir_col = {int(d): j for j, d in enumerate(dirs)}

        motor = von_mises_pdfs(np.arange(0, 360), 0, self.k_motor,
                               normalize=True)                       # (360,1)
        self._uniform = np.ones(360) / 360.0

        # (M2/M3) per-coherence, per-kappa MAP read-out, motor-convolved.
        # like_c and m_pdfs_c depend only on k_like[c], so build them ONCE per
        # coherence and reuse across all kappa (the profiled speed fix).
        di = DIRECTION_SPACE.astype(float)
        self._readout = {}
        for c in cohs:
            like_c = von_mises_pdfs(di, di, self.k_like[c], normalize=True)
            m_pdfs_c = von_mises_pdfs(di, dirs, self.k_like[c], normalize=True)
            stack = np.empty((self.k_grid.size, 360, dirs.size))
            for i, kap in enumerate(self.k_grid):
                prior = mixture_prior(kap, self.alpha, self.prior_mean)
                L = _map_readout(like_c, m_pdfs_c, prior, dirs, di)  # (360,360)
                L = np.nan_to_num(L[:, dirs - 1], nan=0.0)          # (360,D)
                stack[i] = circular_convolution(L, motor)           # (360,D)
            self._readout[c] = stack

        # (M1) belief-update likelihood of feedback under the MIXTURE prior:
        #   T[i, f-1] = alpha*V(f;225,kappa_i) + (1-alpha)/360
        vm = np.vstack([
            von_mises_pdfs(DIRECTION_SPACE, self.prior_mean, kap,
                           normalize=True).ravel()
            for kap in self.k_grid])                                 # (nk,360)
        self._obs_table = self.alpha * vm + (1.0 - self.alpha) / 360.0

        self._belief0 = self.initial_belief()
        self._prepared_for = sig

    # ----------------------------------------------------------------------
    #  (M1) Belief update over kappa — predict/correct, mixture likelihood.
    # ----------------------------------------------------------------------
    def update_belief(self, belief: np.ndarray, feedback_dir: int) -> np.ndarray:
        belief = forget(belief, self._belief0, self.lam)
        return bayes_correct(belief, self._obs_table[:, feedback_dir - 1])

    # ----------------------------------------------------------------------
    #  (M4) Estimate distribution: belief-average of the read-outs + lapse.
    # ----------------------------------------------------------------------
    def estimate_distribution(self, coherence: float, direction: int,
                              belief: np.ndarray) -> np.ndarray:
        j = self._dir_col[int(direction)]
        readouts = self._readout[coherence][:, :, j]        # (nk, 360)
        percept = belief @ readouts                         # (360,) integrated
        # lapse renormalisation (paper convention, as in the switch model)
        denom = 1.0 + self.p_random
        dist = (percept + self.p_random * self._uniform) / denom
        s = dist.sum()
        return dist / s if s > 0 else self._uniform.copy()

    # ----------------------------------------------------------------------
    #  Roll forward over a trial sequence (chronological).
    # ----------------------------------------------------------------------
    def filter(self, directions, coherences, feedback=None,
               sample: bool = False, rng: Optional[np.random.RandomState] = None,
               record_belief: bool = False):
        directions = np.asarray(directions, dtype=int)
        coherences = np.asarray(coherences, dtype=float)
        feedback = directions if feedback is None else np.asarray(feedback, dtype=int)
        self._prepare(directions, coherences)

        belief = self._belief0.copy()
        dists, responses, believed_sd = [], [], []
        for t in range(directions.size):
            dist = self.estimate_distribution(coherences[t], directions[t], belief)
            dists.append(dist)
            if sample:
                responses.append(int(rng.choice(np.arange(1, 361), p=dist)))
            if record_belief:
                k_mean = float(np.sum(belief * self.k_grid))
                believed_sd.append(von_mises_std(k_mean, self.prior_mean))
            belief = self.update_belief(belief, feedback[t])

        out = {"dists": dists}
        if sample:
            out["responses"] = np.array(responses)
        if record_belief:
            out["believed_sd"] = np.array(believed_sd)
        return out

    # ----------------------------------------------------------------------
    #  (M5) Negative log-likelihood + information criteria.
    # ----------------------------------------------------------------------
    N_PARAMS = 7  # 3 k_e + alpha + k_motor + p_random + lam

    def negative_log_likelihood(self, estimates, directions, coherences,
                                feedback=None) -> float:
        estimates = np.asarray(estimates, dtype=int)
        out = self.filter(directions, coherences, feedback, sample=False)
        ll = 0.0
        for t, dist in enumerate(out["dists"]):
            ll += np.log(max(dist[(estimates[t] - 1) % 360], 1e-320))
        return -float(ll)

    @staticmethod
    def aic(nll: float, n_params: int = N_PARAMS) -> float:
        return 2.0 * n_params + 2.0 * nll

    @staticmethod
    def bic(nll: float, n_trials: int, n_params: int = N_PARAMS) -> float:
        return n_params * np.log(n_trials) + 2.0 * nll
