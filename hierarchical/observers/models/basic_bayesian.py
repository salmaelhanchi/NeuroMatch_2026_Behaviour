"""
basic_bayesian.py
=================

The **Basic Bayesian observer** of Laquitaine & Gardner (2018) -- the baseline
the Switching observer was compared against. It is the textbook Bayesian ideal
observer: it *multiplies* the sensory likelihood and the learnt prior into a
single posterior and reports its mode (MAP). Because the two sources are
combined rather than selected between, its estimate distribution is **unimodal**
-- one bump, pulled between the stimulus and the prior mean by their relative
reliability. (The Switching observer keeps the two read-outs separate and so
produces the *bimodal* distributions subjects actually showed; see
``switching_observer.py``.)

This is "the Switching observer minus the switch": it shares the exact same nine
free parameters -- three ``k_like`` (coherence), four ``k_prior`` (prior width),
``p_random`` and ``k_motor`` -- and reuses the same Girshick MAP look-up. The
only difference is that the look-up is called ONCE with *both* the likelihood and
the prior switched on (the joint posterior), instead of twice with each switched
off and then mixed.

Ported from the project's earlier exploratory implementation
(``behavior_and_theory/laquitaine_switching_project.py`` §4, the 7 generative
steps), reworked here into the analytic, likelihood-marginalised form the other
package models use so it can be fit and compared directly.

Equations
---------
Von Mises density (Eq. 1)                     -> circular.von_mises_pdfs
Integrated posterior read-out (Eq. 3-5):

    p(percept | theta_true) = MAP of [ V(.; theta_e, k_e) * V(.; 225, k_prior) ]
                              pushed forward over the measurement theta_e

Estimate distribution (Eq. 7, single-component form of the switch's Eq. 6/7):

    p(est | theta_true) =
        V(est; 0, k_motor)  ⊛  [ (percept + p_random/360) / (1 + p_random) ]

    where ⊛ is circular convolution. The lapse renormalisation
    ``(percept + p_random*uniform)/(1+p_random)`` is the same convention the
    Switching and HB-integration models use, so the three are directly
    comparable.
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
N_PARAMS = 9        # 3 k_like + 4 k_prior + k_motor + p_random (same as the switch)


@dataclass
class BasicBayesianObserver:
    """The Basic Bayesian observer with its nine free parameters.

    Attributes mirror ``SwitchingObserver`` exactly (same parameter set), so the
    two models can be fit and compared on equal footing.

    k_like : dict coherence -> concentration k_e  (sensory reliability).
    k_prior : dict prior-SD-label -> concentration k_prior (prior reliability).
    p_random : lapse rate p_r in [0, 1].
    k_motor : motor-noise concentration k_motor.
    prior_mean : mean of the learnt prior (deg); 225 in the experiment.
    k_cardinal : cardinal-prior strength; 0 for the winning "withoutCardinal" model.
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
        """Each fitted prior concentration as a circular SD in degrees (Eq. 2)."""
        return {label: von_mises_std(k, self.prior_mean)
                for label, k in self.k_prior.items()}

    # -- AIC / BIC (static; N_PARAMS is fixed for this model) ---------------
    @staticmethod
    def aic(nll: float) -> float:
        return 2 * N_PARAMS + 2 * nll

    @staticmethod
    def bic(nll: float, n: int) -> float:
        return N_PARAMS * np.log(n) + 2 * nll

    # ----------------------------------------------------------------------
    #  Core: predicted estimate distribution for one condition
    # ----------------------------------------------------------------------
    def estimate_distribution(self, theta_true: int, coherence: float,
                              prior_label: str) -> np.ndarray:
        """Predicted p(estimate | theta_true) for one condition (length-360).

        1. Integrated read-out: Girshick MAP look-up with BOTH the likelihood
           (k_like) and the prior (k_prior) switched ON -> the mode of the joint
           posterior for each measurement, pushed forward over measurements.
           This single bump is the Bayesian combined percept.
        2. Lapse: mix with a uniform guess, renormalised (switch/HB convention).
        3. Motor noise (Eq. 7): circularly convolve with V(.; 0, k_motor).
        """
        k_e = self.k_like[coherence]
        k_p = self.k_prior[prior_label]

        L = girshick_map_lookup(
            k_like=k_e, prior_mode=self.prior_mean, k_prior=k_p,
            motion_dirs=np.array([theta_true]), k_cardinal=self.k_cardinal)
        percept = _column(L, theta_true)

        percept = self._apply_lapse(percept)
        return self._apply_motor_noise(percept)

    # ----------------------------------------------------------------------
    #  Lapse + motor noise (tail of Eq. 7)
    # ----------------------------------------------------------------------
    def _apply_lapse(self, percept: np.ndarray) -> np.ndarray:
        """Mix the percept with a uniform lapse, renormalised so the result is a
        probability vector: ``(percept + p_random*uniform)/(1 + p_random)``. This
        is the single-component form of the switch's Eq. 6 renormalisation and
        matches the HB-integration model's lapse exactly."""
        uniform = np.ones(360) / 360.0
        return (percept + self.p_random * uniform) / (1.0 + self.p_random)

    def _apply_motor_noise(self, percept: np.ndarray) -> np.ndarray:
        """Circularly convolve with von Mises motor noise V(.; 0, k_motor)."""
        shifted_grid = np.arange(0, 360)  # 0..359 so the VM peaks at 0
        motor = von_mises_pdfs(shifted_grid, 0, self.k_motor,
                               normalize=True).ravel()
        est = circular_convolution(percept, motor)
        est = np.clip(est, 1e-320, None)
        return est / est.sum()

    # ----------------------------------------------------------------------
    #  Predicted mean & SD of estimates for a condition
    # ----------------------------------------------------------------------
    def predicted_mean_std(self, theta_true: int, coherence: float,
                           prior_label: str):
        dist = self.estimate_distribution(theta_true, coherence, prior_label)
        return circular_weighted_mean_std(DIRECTION_SPACE, dist)

    # ----------------------------------------------------------------------
    #  Trial-by-trial negative log-likelihood (the fitting objective)
    # ----------------------------------------------------------------------
    def negative_log_likelihood(self, estimates, theta_true, coherence,
                                prior_label) -> float:
        """-sum_t log p(estimate_t | condition_t).

        Efficient: the read-out depends only on (coherence, prior_label), so the
        Girshick look-up is called ONCE per (coherence, prior) pair (12 at most,
        each returning all directions), then each unique condition's distribution
        is built once and reused across its trials.
        """
        estimates = np.asarray(estimates, dtype=int)
        theta_true = np.asarray(theta_true, dtype=int)
        coherence = np.asarray(coherence, dtype=float)
        prior_label = np.asarray(prior_label)

        # one Girshick look-up per (coherence, prior) pair, over its directions
        lookups: Dict[tuple, np.ndarray] = {}
        for c, pl in {(float(c), str(pl))
                      for c, pl in zip(coherence, prior_label)}:
            mask = (coherence == c) & (prior_label == pl)
            dirs = np.unique(theta_true[mask])
            lookups[(c, pl)] = girshick_map_lookup(
                k_like=self.k_like[c], prior_mode=self.prior_mean,
                k_prior=self.k_prior[pl], motion_dirs=dirs,
                k_cardinal=self.k_cardinal)

        dist_cache: Dict[tuple, np.ndarray] = {}
        logl = 0.0
        for est, d, c, pl in zip(estimates, theta_true, coherence, prior_label):
            key = (int(d), float(c), str(pl))
            dist = dist_cache.get(key)
            if dist is None:
                percept = _column(lookups[(float(c), str(pl))], int(d))
                percept = self._apply_lapse(percept)
                dist = self._apply_motor_noise(percept)
                dist_cache[key] = dist
            logl += np.log(max(dist[(est - 1) % 360], 1e-320))
        return -float(logl)


# ---------------------------------------------------------------------------
# small internal helper (shared shape with switching_observer._column)
# ---------------------------------------------------------------------------
def _column(L: np.ndarray, direction: int) -> np.ndarray:
    """Percept distribution for one displayed direction; NaNs (unqueried
    directions) -> 0."""
    col = L[:, direction - 1]
    return np.nan_to_num(col, nan=0.0)


# ---------------------------------------------------------------------------
# Demonstration: the Bayesian percept is UNIMODAL (contrast switching_observer)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    obs = BasicBayesianObserver(
        k_like={0.24: 8.0, 0.12: 3.0, 0.06: 1.0},
        k_prior={"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7},
        p_random=0.01, k_motor=40.0)

    theta, coh, prior = 85, 0.06, "80"
    dist = obs.estimate_distribution(theta, coh, prior)
    mean_deg, std_deg = obs.predicted_mean_std(theta, coh, prior)
    print(f"Condition: direction={theta} deg, coherence={coh}, prior={prior} deg")
    print(f"  predicted estimate mean={mean_deg:.1f} deg  SD={std_deg:.1f} deg")
    print(f"  distribution sums to {dist.sum():.6f} (should be 1)")
    print(f"  single MAP peak at ~{int(np.argmax(dist)) + 1} deg "
          f"(one bump, between stimulus 85 and prior 225)")
    print("\nBias toward prior grows as coherence drops (direction 85, prior 80):")
    for c in (0.24, 0.12, 0.06):
        m, s = obs.predicted_mean_std(theta, c, prior)
        print(f"    coh={c}: mean={m:6.1f} deg (225=prior, 85=stimulus)  SD={s:4.1f}")
