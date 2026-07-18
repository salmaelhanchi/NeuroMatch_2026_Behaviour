"""
switching_observer.py
======================

The **Switching observer** of Laquitaine & Gardner (2018), "A Switching
Observer for Human Perceptual Estimation", *Neuron* 97(2), 462-474 -- the
model that won their model comparison (Figure 5, Figure 6A/B).

This file is the readable heart of the port.  It follows the paper's STAR
Methods equations step for step; the fiddly numerical machinery (von Mises
densities, the Girshick MAP look-up, circular convolution) lives in
``circular.py`` and ``bayes_lookup.py`` so it does not obscure the logic here.

Ported from the MATLAB reference implementation:
    SLcodes/Behavior/analyses/modeling/models/Switching/SLgetLoglCompCmae.m
    SLcodes/.../Switching/SLsimSwitchingEstimatesDensity.m
(https://github.com/steevelaquitaine/projInference, master branch).


The idea in one paragraph
-------------------------
The Basic Bayesian observer *multiplies* the sensory likelihood and the
learnt prior into a single posterior and reports its mode.  That predicts a
*unimodal* estimate distribution.  Subjects instead produced *bimodal*
distributions -- one peak at the stimulus direction, one at the prior mean.
The Switching observer reproduces this by **not** combining the two
distributions: on each trial it commits to *either* the sensory evidence
*or* the prior mean, choosing between them in proportion to their reliability.
Because the choice is stochastic across trials, the estimate distribution has
two peaks.


The experimental variables (what the parameters mean)
-----------------------------------------------------
The task: subjects viewed a random-dot motion stimulus and reported its
direction. The generative structure the model represents:

* ``theta_true``  -- the displayed motion direction (1..360 deg).
* coherence       -- fraction of coherent dots (0.06, 0.12, 0.24). Higher
                     coherence = more reliable evidence, represented by a
                     larger sensory concentration ``k_like`` (one per
                     coherence: ``k_like[0.24] > k_like[0.12] > k_like[0.06]``).
* prior width     -- across blocks the true directions were drawn from a
                     von Mises prior of SD 80/40/20/10 deg centred on 225 deg.
                     A tighter (more informative) prior = larger prior
                     concentration ``k_prior`` (one per prior condition).
* ``p_random``    -- lapse rate: fraction of trials estimated at random.
* ``k_motor``     -- motor precision: concentration of von Mises motor noise
                     added to every estimate.

The nine free parameters of the winning model are exactly these:
three ``k_like`` (coherence), four ``k_prior`` (prior width), ``p_random`` and
``k_motor`` -- *the same set as the Basic Bayesian observer*, so switching
buys the bimodality "for free" (paper, Results & Figure 5A caption).


The equations implemented here
------------------------------
Von Mises density (Eq. 1)          -> circular.von_mises_pdfs
Switching probabilities (Eq. 6):

        p_prior = k_prior / (k_prior + k_e)      p_e = 1 - p_prior

Estimate distribution (Eq. 7):

  p(est | theta_true) =
      V(est; 0, k_motor)  ⊛  [ (1 - p_r) ( p_e * p(theta_e | theta_true)
                                          + p_prior * delta(theta - mu_prior) )
                              + p_r / 360 ]

  where ⊛ is circular convolution and delta() a spike at the prior mean.

In code the two inner terms are produced by the *same* Girshick MAP look-up:
  * evidence term  p(theta_e | theta_true): look-up with the prior switched
    off (k_prior = 0) -> read-out follows the measurement.
  * prior term  delta(theta - mu_prior): look-up with the likelihood switched
    off (k_like = 0) -> read-out collapses onto the prior mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from observers.helpers.circular import (
    DIRECTION_SPACE,
    von_mises_pdfs,
    circular_convolution,
    circular_weighted_mean_std,
    von_mises_std,
)
from observers.helpers.bayes_lookup import girshick_map_lookup

PRIOR_MEAN = 225.0  # experimental prior mean (degrees), fixed in the task


@dataclass
class SwitchingObserver:
    """The Switching observer with its nine free parameters.

    Parameters map coherence -> sensory strength and prior-SD -> prior
    strength, matching how the model is fit in the paper.

    Attributes
    ----------
    k_like : dict coherence -> concentration k_e  (sensory reliability).
    k_prior : dict prior-SD-label -> concentration k_prior (prior reliability).
    p_random : lapse rate p_r in [0, 1].
    k_motor : motor-noise concentration k_motor.
    prior_mean : mean of the learnt prior (deg); 225 in the experiment.
    k_cardinal : cardinal-prior strength; 0 for the winning "withoutCardinal"
                 model.
    """

    k_like: Dict[float, float] = field(
        default_factory=lambda: {0.24: 8.0, 0.12: 3.0, 0.06: 1.0})
    k_prior: Dict[str, float] = field(
        default_factory=lambda: {"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7})
    p_random: float = 0.01
    k_motor: float = 40.0
    prior_mean: float = PRIOR_MEAN
    k_cardinal: float = 0.0

    # -- reporting helper ---------------------------------------------------
    def prior_sd_degrees(self) -> Dict[str, float]:
        """Return each fitted prior concentration as a circular SD in degrees
        (paper Eq. 2) -- e.g. verifies an '80' prior really is ~80 deg wide."""
        return {label: von_mises_std(k, self.prior_mean)
                for label, k in self.k_prior.items()}

    # ----------------------------------------------------------------------
    #  Core: predicted estimate distribution for one condition
    # ----------------------------------------------------------------------
    def estimate_distribution(self, theta_true: int, coherence: float,
                              prior_label: str) -> np.ndarray:
        """Predicted distribution p(estimate | theta_true) for one condition.

        This is Eq. 7 assembled from Eq. 6, for a single displayed direction,
        coherence and prior. Returns a length-360 probability vector over
        estimate directions 1..360 deg.

        Steps (each maps to a block of the paper's methods):

        1. Evidence read-out  p(theta_e | theta_true):
           Girshick MAP look-up with the prior flat (k_prior = 0). With no
           prior pull, the MAP of each measurement is the measurement itself,
           so this is the sensory percept distribution -- a bump at the
           stimulus direction whose width shrinks as coherence (k_like) grows.

        2. Prior read-out  delta(theta - mu_prior):
           Girshick MAP look-up with the likelihood flat (k_like = 0). Every
           measurement now maps to the prior mode, giving a spike at 225 deg.

        3. Switching weights (Eq. 6): p_prior = k_prior/(k_prior+k_e),
           p_e = k_e/(k_prior+k_e). Renormalise together with the lapse rate
           so p_e + p_prior + p_random_effective = 1.

        4. Mixture over percepts:
              P(percept) = p_e * evidence + p_prior * prior + p_rand * uniform

        5. Motor noise (Eq. 7): circularly convolve with V(.;0,k_motor).
        """
        k_e = self.k_like[coherence]
        k_p = self.k_prior[prior_label]
        motion_dirs = np.array([theta_true])

        # 1. evidence read-out: prior switched OFF (k_prior = 0)
        L_evidence = girshick_map_lookup(
            k_like=k_e, prior_mode=self.prior_mean, k_prior=0.0,
            motion_dirs=motion_dirs, k_cardinal=self.k_cardinal)
        p_evidence = _column(L_evidence, theta_true)

        # 2. prior read-out: likelihood switched OFF (k_like = 0)
        L_prior = girshick_map_lookup(
            k_like=0.0, prior_mode=self.prior_mean, k_prior=k_p,
            motion_dirs=motion_dirs, k_cardinal=self.k_cardinal)
        p_prior_mean = _column(L_prior, theta_true)

        # 3. switching weights (Eq. 6) + lapse, renormalised to sum to 1
        w_prior, w_evidence, w_random = self._switching_weights(k_e, k_p)

        # 4. percept mixture (pre motor noise). If both strengths are zero the
        #    observer is at chance -> uniform (matches the MATLAB guard).
        if k_e == 0 and k_p == 0:
            percept = np.ones(360) / 360.0
        else:
            uniform = np.ones(360) / 360.0
            percept = (w_evidence * p_evidence
                       + w_prior * p_prior_mean
                       + w_random * uniform)

        # 5. convolve with motor noise V(est; 0, k_motor)  (Eq. 7)
        estimate = self._apply_motor_noise(percept)
        return estimate

    # ----------------------------------------------------------------------
    #  Eq. 6  (+ lapse renormalisation), lifted out for clarity
    # ----------------------------------------------------------------------
    def _switching_weights(self, k_e: float, k_p: float):
        """Return (P_prior, P_evidence, P_random) that sum to 1.

        Paper Eq. 6 gives the *relative* competition weights

            w_prior = k_prior / (k_prior + k_e)
            w_e     = k_e      / (k_prior + k_e)

        which express the winner-take-all competition: the more reliable
        representation is chosen more often. High coherence (large k_e) ->
        evidence usually wins -> veridical, low-variance estimates. Low
        coherence -> prior wins more -> estimates biased toward 225 deg.

        These two plus the fixed lapse ``p_random`` are then renormalised so
        the three mixture probabilities sum to exactly 1 (MATLAB:
        ``sumP = w_prior + w_e + Prandom`` then divide).
        """
        if (k_p + k_e) == 0:
            return 0.0, 0.0, 1.0
        w_prior = k_p / (k_p + k_e)
        w_evidence = k_e / (k_p + k_e)

        total = w_prior + w_evidence + self.p_random
        P_prior = 1.0 - (self.p_random + w_evidence) / total
        P_evidence = 1.0 - (self.p_random + w_prior) / total
        P_random = 1.0 - (w_evidence + w_prior) / total
        return P_prior, P_evidence, P_random

    # ----------------------------------------------------------------------
    #  Motor noise convolution (tail of Eq. 7)
    # ----------------------------------------------------------------------
    def _apply_motor_noise(self, percept: np.ndarray) -> np.ndarray:
        """Circularly convolve a percept distribution with motor noise.

        Motor noise is a von Mises centred on 0 with concentration k_motor.
        It is built on the shifted grid 0..359 (so its peak sits at 0) and
        convolved with the percept distribution; the result is renormalised
        to a probability vector. This is the ``⊛ V(est; 0, k_motor)`` in Eq. 7.
        """
        shifted_grid = np.arange(0, 360)  # 0..359 so the VM peaks at 0
        motor = von_mises_pdfs(shifted_grid, 0, self.k_motor,
                               normalize=True).ravel()
        est = circular_convolution(percept, motor)
        est = np.clip(est, 1e-320, None)   # kill tiny negative FFT residue
        est = est / est.sum()
        return est

    # ----------------------------------------------------------------------
    #  Predicted mean & SD of estimates for a condition (Figures 5B, 5C)
    # ----------------------------------------------------------------------
    def predicted_mean_std(self, theta_true: int, coherence: float,
                           prior_label: str):
        """Circular mean and SD (deg) of the predicted estimate distribution."""
        dist = self.estimate_distribution(theta_true, coherence, prior_label)
        return circular_weighted_mean_std(DIRECTION_SPACE, dist)

    # ----------------------------------------------------------------------
    #  Trial-by-trial negative log-likelihood of data (the fitting objective)
    # ----------------------------------------------------------------------
    def negative_log_likelihood(self, estimates, theta_true, coherence,
                                prior_label) -> float:
        """Negative summed log-likelihood of observed estimates (Eq. 7 as a
        likelihood), the objective minimised when fitting the model.

        Parameters (all equal length, one entry per trial)
        ----------
        estimates : subject's reported directions (1..360 deg).
        theta_true : displayed directions.
        coherence : coherence of each trial (0.06 / 0.12 / 0.24).
        prior_label : prior condition of each trial ('80'/'40'/'20'/'10').

        Returns
        -------
        -sum_t log p(estimate_t | theta_true_t, model).

        The predicted distribution depends only on the *condition*
        (direction, coherence, prior), so distributions are cached per unique
        condition and reused -- exactly the speed-up structure of the MATLAB
        code, which builds one look-up per condition and indexes trials into
        it.
        """
        estimates = np.asarray(estimates, dtype=int)
        theta_true = np.asarray(theta_true, dtype=int)
        coherence = np.asarray(coherence, dtype=float)
        prior_label = np.asarray(prior_label)

        cache: Dict[tuple, np.ndarray] = {}
        logl = 0.0
        for est, d, c, pl in zip(estimates, theta_true, coherence, prior_label):
            key = (int(d), float(c), str(pl))
            dist = cache.get(key)
            if dist is None:
                dist = self.estimate_distribution(int(d), float(c), str(pl))
                cache[key] = dist
            idx = (est - 1) % 360           # estimate 360 -> index 359
            logl += np.log(max(dist[idx], 1e-320))
        return -float(logl)


# ---------------------------------------------------------------------------
# small internal helper
# ---------------------------------------------------------------------------
def _column(L: np.ndarray, direction: int) -> np.ndarray:
    """Extract the percept distribution for one displayed direction and
    replace the look-up's NaNs (unqueried directions) with 0."""
    col = L[:, direction - 1]
    return np.nan_to_num(col, nan=0.0)


# ---------------------------------------------------------------------------
# Demonstration: reproduce the hallmark bimodality (run `python switching_observer.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    obs = SwitchingObserver(
        k_like={0.24: 8.0, 0.12: 3.0, 0.06: 1.0},
        k_prior={"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7},
        p_random=0.01,
        k_motor=40.0,
    )

    print("Fitted prior concentrations expressed as circular SD (deg):")
    for label, sd in obs.prior_sd_degrees().items():
        print(f"    k_prior['{label}'] -> {sd:5.1f} deg wide")

    # A direction far from the prior mean at low coherence: the hallmark
    # regime where the estimate distribution is clearly bimodal (one peak at
    # the stimulus, one at the 225 deg prior mean).
    theta, coh, prior = 85, 0.06, "80"
    dist = obs.estimate_distribution(theta, coh, prior)
    mean_deg, std_deg = obs.predicted_mean_std(theta, coh, prior)

    w_prior, w_evid, w_rand = obs._switching_weights(
        obs.k_like[coh], obs.k_prior[prior])

    print(f"\nCondition: direction={theta} deg, coherence={coh}, prior={prior} deg")
    print(f"  switching weights  P(evidence)={w_evid:.3f}  "
          f"P(prior)={w_prior:.3f}  P(random)={w_rand:.3f}")
    print(f"  predicted estimate mean={mean_deg:.1f} deg  SD={std_deg:.1f} deg")
    print(f"  distribution sums to {dist.sum():.6f} (should be 1)")

    # locate the two peaks
    peak_stim = int(np.argmax(dist[max(0, theta - 30):theta + 30])) + max(0, theta - 30) + 1
    peak_prior = int(np.argmax(dist[195:255])) + 195 + 1
    print(f"  local peak near stimulus  ~{peak_stim} deg")
    print(f"  local peak near prior mean ~{peak_prior} deg")

    # coherence sweep: bias toward the prior should grow as coherence falls
    print("\nBias toward prior grows as coherence drops (direction 85, prior 80):")
    for c in (0.24, 0.12, 0.06):
        m, s = obs.predicted_mean_std(theta, c, prior)
        print(f"    coh={c}: mean={m:6.1f} deg (225=prior, 85=stimulus)  SD={s:4.1f}")