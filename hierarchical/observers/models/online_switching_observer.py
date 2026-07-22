"""
online_switching_observer.py
============================

**Online switching observer** — the equation-level model file.

This extends the static Switching observer (``switching_observer.py``) by making
the prior's strength a *latent quantity the observer learns online* from trial
feedback, instead of a fixed fitted parameter. The trial equations are labelled
H1–H5; the numerical machinery for the belief over prior strength lives in
``belief_grid.py``, and the circular/Girshick primitives are reused from the
switching layer (``circular.py`` / ``bayes_lookup.py``).

The pipeline for one trial:

    (estimate)   use the CURRENT belief b_t about prior strength to form the
                 switch weight (H2/H3) and the estimate distribution (H4);
    (respond)    sample or score the observed estimate;
    (learn)      after feedback f_t is revealed, update the belief (H1).

Because the belief path is deterministic given the feedback sequence and
parameters, the likelihood (H5) needs no particle filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# --- reused from the switching layer (frozen, already verified) -------------
from observers.helpers.circular import (
    DIRECTION_SPACE,
    von_mises_pdfs,
    circular_convolution,
    circular_weighted_mean_std,
    von_mises_std,
)
from observers.helpers.bayes_lookup import girshick_map_lookup

# --- hierarchical-online helpers -------------------------------------------
from observers.helpers.belief_grid import (
    make_k_grid,
    observation_likelihood_table,
    forget,
    bayes_correct,
    expected_prior_weight,
)

PRIOR_MEAN = 225.0


@dataclass
class OnlineHierarchicalObserver:
    """Switching observer that learns the prior strength online.

    Free parameters:
        k_like   : dict coherence -> sensory concentration k_e.
        k_motor  : motor-noise concentration.
        p_random : lapse rate.
        lam      : volatility / forgetting (the learning knob), in [0, 1].

    Fixed structure:
        prior_mean : 225 deg.
        k_grid     : discrete support for the belief over prior strength.
        belief0    : initial (hyper) belief over k; default uniform on the grid.
    """

    k_like: Dict[float, float] = field(
        default_factory=lambda: {0.24: 8.0, 0.12: 3.0, 0.06: 1.0})
    k_motor: float = 40.0
    p_random: float = 0.01
    lam: float = 0.0
    prior_mean: float = PRIOR_MEAN
    k_grid: np.ndarray = field(default_factory=make_k_grid)

    # internal precomputed caches (filled by _prepare)
    _prepared_for: tuple = field(default=None, repr=False)
    _evi_conv: dict = field(default_factory=dict, repr=False)
    _prior_conv: np.ndarray = field(default=None, repr=False)
    _uniform: np.ndarray = field(default=None, repr=False)
    _obs_table: np.ndarray = field(default=None, repr=False)
    _belief0: np.ndarray = field(default=None, repr=False)

    # ----------------------------------------------------------------------
    #  Belief initialisation (Level-3 hyper-belief)
    # ----------------------------------------------------------------------
    def initial_belief(self) -> np.ndarray:
        """Uniform belief over the k-grid: at trial 1 the observer does not yet
        know how strong the prior is."""
        n = self.k_grid.size
        return np.ones(n) / n

    # ----------------------------------------------------------------------
    #  One-off precompute for a given design (the tractability shortcuts, §5)
    # ----------------------------------------------------------------------
    def _prepare(self, directions: np.ndarray, coherences: np.ndarray):
        """Precompute the pieces of the estimate distribution that do not
        depend on the (changing) belief: the motor-convolved sensory read-out
        per (coherence, direction), the convolved prior spike, the uniform
        lapse, and the belief-update likelihood table.

        Recomputed only when k_like or k_motor change (they set these pieces);
        p_random and lam enter later as cheap scalars.
        """
        sig = (tuple(sorted(self.k_like.items())), self.k_motor)
        if self._prepared_for == sig and self._evi_conv:
            return
        dirs = np.unique(np.asarray(directions, dtype=int))
        cohs = np.unique(np.asarray(coherences, dtype=float))

        # motor noise kernel V(.;0,k_motor) on the 0..359 grid (peak at 0)
        motor = von_mises_pdfs(np.arange(0, 360), 0, self.k_motor,
                               normalize=True)  # (360,1)

        # prior read-out: delta at the prior mean, then convolved with motor
        delta = np.zeros((360, 1))
        delta[int(self.prior_mean) - 1, 0] = 1.0
        self._prior_conv = circular_convolution(delta, motor).ravel()

        # uniform lapse (unchanged by convolution with a normalised kernel)
        self._uniform = np.ones(360) / 360.0

        # sensory read-out per coherence: Girshick MAP with the prior OFF
        # (k_prior=0), i.e. p(theta_e | theta_true); then convolve all columns
        # with motor noise in one shot.
        self._evi_conv = {}
        for c in cohs:
            L = girshick_map_lookup(k_like=self.k_like[c], prior_mode=self.prior_mean,
                                    k_prior=0.0, motion_dirs=dirs, k_cardinal=0.0)
            L = np.nan_to_num(L, nan=0.0)
            self._evi_conv[c] = circular_convolution(L, motor)  # (360,360)

        # belief-update likelihood table V(f;225,k) over the k-grid
        self._obs_table = observation_likelihood_table(self.k_grid, self.prior_mean)
        self._belief0 = self.initial_belief()
        self._prepared_for = sig

    # ----------------------------------------------------------------------
    #  Belief update  (H1a predict + H1b correct)
    # ----------------------------------------------------------------------
    def update_belief(self, belief: np.ndarray, feedback_dir: int) -> np.ndarray:
        """Advance the belief over prior strength after one feedback direction.

        (H1a) leak toward the initial belief by ``lam`` (volatility), then
        (H1b) multiply by the likelihood of the revealed direction under each
        candidate strength and renormalise. Returns the new belief.
        """
        belief = forget(belief, self._belief0, self.lam)
        return bayes_correct(belief, self._obs_table[:, feedback_dir - 1])

    # ----------------------------------------------------------------------
    #  Estimate distribution for one trial  (H2 -> H3 -> H4)
    # ----------------------------------------------------------------------
    def estimate_distribution(self, coherence: float, direction: int,
                              belief: np.ndarray) -> np.ndarray:
        """p(estimate | direction, current belief) for one trial.

        H2: expected switch weight under the belief, w_prior = E_b[k/(k+k_e)].
        H3: fold in the lapse (linear, so the belief-average is exact).
        H4: mix the sensory read-out, the prior spike, and the uniform lapse,
            already convolved with motor noise, into the estimate distribution.
        """
        k_e = self.k_like[coherence]

        # H2: reliability-ratio switch weight averaged over the belief
        w_prior = expected_prior_weight(belief, self.k_grid, k_e)
        w_e = 1.0 - w_prior

        # H3: lapse renormalisation (matches the static switching code)
        denom = 1.0 + self.p_random
        P_prior = w_prior / denom
        P_e = w_e / denom
        P_rand = self.p_random / denom

        # H4: mixture of (motor-convolved) components
        evidence = self._evi_conv[coherence][:, direction - 1]
        dist = P_e * evidence + P_prior * self._prior_conv + P_rand * self._uniform
        s = dist.sum()
        return dist / s if s > 0 else self._uniform.copy()

    # ----------------------------------------------------------------------
    #  Roll the model forward over a trial sequence
    # ----------------------------------------------------------------------
    def filter(self, directions, coherences, feedback=None,
               sample: bool = False, rng: Optional[np.random.RandomState] = None,
               record_belief: bool = False):
        """Run the observer through a trial sequence (chronological order).

        On each trial the estimate distribution uses the belief formed from
        *previous* feedback (the current trial's feedback arrives only after the
        response), then the belief is updated with this trial's feedback for the
        next trial.

        Parameters
        ----------
        directions, coherences : per-trial displayed direction and coherence.
        feedback : revealed true direction per trial (defaults to ``directions``
            — in this task feedback is the displayed direction).
        sample : if True, draw a response from each trial's distribution
            (generative mode); requires ``rng``.
        record_belief : if True, also return the believed-prior-SD trajectory.

        Returns dict with 'dists' (per-trial distributions), optionally
        'responses' and 'believed_sd' (circular SD of the belief-mean prior).
        """
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
            # learn from this trial's feedback (for the next trial)
            belief = self.update_belief(belief, feedback[t])

        out = {"dists": dists}
        if sample:
            out["responses"] = np.array(responses)
        if record_belief:
            out["believed_sd"] = np.array(believed_sd)
        return out

    # ----------------------------------------------------------------------
    #  No-learning special case (= the static Switching observer)
    # ----------------------------------------------------------------------
    #  Setting the belief to a fixed point mass at a per-block prior strength
    #  (never updated) recovers the winning static Switching observer of
    #  Laquitaine & Gardner. Implemented on the same fast precompute so it can
    #  serve as the competitor in Phase-7 model recovery. Verified against the
    #  independent switching_observer.py implementation in online_recovery.py.
    def estimate_distribution_fixedk(self, coherence: float, direction: int,
                                     k_prior: float) -> np.ndarray:
        """Estimate distribution with a FIXED prior strength (no learning):
        the switch weight is the plain reliability ratio k_prior/(k_prior+k_e)
        rather than a belief average."""
        k_e = self.k_like[coherence]
        w_prior = 1.0 if k_e == 0 else k_prior / (k_prior + k_e)
        w_e = 1.0 - w_prior
        denom = 1.0 + self.p_random
        P_prior, P_e, P_rand = w_prior / denom, w_e / denom, self.p_random / denom
        evidence = self._evi_conv[coherence][:, direction - 1]
        dist = P_e * evidence + P_prior * self._prior_conv + P_rand * self._uniform
        s = dist.sum()
        return dist / s if s > 0 else self._uniform.copy()

    def negative_log_likelihood_fixedk(self, estimates, directions, coherences,
                                       k_prior_per_trial) -> float:
        """NLL of the no-learning static observer given a fixed per-trial prior
        strength (i.e. a fitted k_prior for each block)."""
        estimates = np.asarray(estimates, dtype=int)
        directions = np.asarray(directions, dtype=int)
        coherences = np.asarray(coherences, dtype=float)
        k_prior_per_trial = np.asarray(k_prior_per_trial, dtype=float)
        self._prepare(directions, coherences)
        ll = 0.0
        for t in range(directions.size):
            dist = self.estimate_distribution_fixedk(coherences[t], directions[t],
                                                     k_prior_per_trial[t])
            ll += np.log(max(dist[(estimates[t] - 1) % 360], 1e-320))
        return -float(ll)

    # ----------------------------------------------------------------------
    #  Negative log-likelihood of observed estimates  (H5)
    # ----------------------------------------------------------------------
    def negative_log_likelihood(self, estimates, directions, coherences,
                                feedback=None) -> float:
        """-sum_t log p(estimate_t | direction_t, belief_t(params))  (H5).

        The belief trajectory is deterministic given the feedback sequence and
        parameters, so this is a plain sum over trials with no sampling.
        """
        estimates = np.asarray(estimates, dtype=int)
        out = self.filter(directions, coherences, feedback, sample=False)
        ll = 0.0
        for t, dist in enumerate(out["dists"]):
            idx = (estimates[t] - 1) % 360
            ll += np.log(max(dist[idx], 1e-320))
        return -float(ll)