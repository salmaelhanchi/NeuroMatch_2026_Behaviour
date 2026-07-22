"""
reliability_mixture.py
======================

Display label: **Reliability-Mixture** (Romi's ``ReliabilityMixtureObserver``).

A faithful port of Rommana Ruamkonthong's reliability-mixture hierarchical
Bayesian observer, folded into the shared observer registry. The original
lives untouched at
``hierarchical/reliability_mixture_hb_model/reliability_mixture_hb_model/src/reliability_mixture_model.py``;
this module reproduces its numerics exactly (cross-checked to <1e-9 in
``observers/verification/verify_reliability_mixture.py``) behind the same
class/fitter/verify scaffolding every other registered model uses.

What the model is
-----------------
On each trial the percept is a **genuine discrete mixture** (an either/or, never
a multiplicative blend) between

    * a prior-driven component: von Mises at the fixed prior mean (225 deg),
      concentration ``k_prior`` for that block's prior width, and
    * a likelihood-driven component: von Mises at that trial's true motion
      direction, concentration ``k_like`` for that trial's coherence.

The mixture weight ``prior_reliance`` in [0,1] is the hyper-belief. It is NOT a
free parameter: it starts at 0.5 and updates trial-by-trial via a delta rule
toward how well recent (5-trial smoothed) feedback agrees with the prior mean,
at learning rate ``alpha``. That discrete-mixture construction is what lets the
model produce genuinely **bimodal** response distributions (a peak at the prior
AND a peak at the stimulus), which a single integrated posterior cannot.

This is deliberately a different construction from ``hb_rachel.py`` /
``hb_integration.py`` (which learns prior *concentration* kappa and holds its
mixture weight fixed): here the learned latent is the *reliance weight* and the
per-condition concentrations are fixed parameters. See Romi's folder README for
the full comparison and the identifiability note about ``k_prior`` being reused
in both the percept mixture and the reliance-update agreement term.

Free parameters per subject (10 total):
    k_like[coherence]   -- 3, sensory (likelihood) concentration   (her k_llh)
    k_prior[prior_std]  -- 4, prior concentration per block width
    alpha               -- reliance learning rate
    k_motor             -- motor/response-noise concentration
    lapse               -- lapse probability                       (her lapse_rate)

Attribute names (``k_like``, ``k_prior``, ``alpha``, ``k_motor``, ``lapse``) are
chosen to match ``fit_batch._observer_params`` so the batch stage records and
``rebuild()`` reconstructs the observer with no pipeline edits.

Window reset in the pipeline
----------------------------
Romi's original resets the 5-trial feedback window only at SESSION boundaries
(so belief carries over across blocks within a session, and is not assumed away
by construction). ``load_subject`` feeds chronological (session, run, trial)
order and a per-trial ``session_id``; this observer resets the window exactly
when ``session_id`` changes, which is equivalent to her ``same_session_prev``
rule. With no ``session_id`` supplied (e.g. synthetic recovery designs) the
sequence is treated as one continuous stream.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.special import i0e

PRIOR_MEAN = 225.0
WINDOW_SIZE = 5
DEG = np.arange(1, 361)


# ===========================================================================
#  Circular / mixture primitives (ported verbatim from Romi's module)
# ===========================================================================
def vm_pdf(support_deg, mu_deg, k, norm=True):
    """Von Mises pmf over a discrete circular support (higher k = tighter).
    Exact port of the original so cached values are bit-identical to a
    per-trial call."""
    mu = np.atleast_1d(np.asarray(mu_deg, float))
    x = np.deg2rad(np.asarray(support_deg, float))[:, None]
    u = np.deg2rad(mu)[None, :]
    k = float(k)
    if np.isinf(k) or k > 1e300:
        out = np.zeros((len(support_deg), len(mu)))
        for j, mm in enumerate(mu):
            out[np.argmin(np.abs(np.asarray(support_deg) - mm)), j] = 1.0
    else:
        out = np.exp(k * np.cos(x - u) - k) / (2 * np.pi * i0e(k))
    if norm:
        out = out / out.sum(0, keepdims=True)
    return out


def wrap_signed_deg(diff_deg):
    return ((diff_deg + 180) % 360) - 180


def circular_mean_deg(angles_deg):
    angles_rad = np.deg2rad(np.asarray(angles_deg))
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    return float(np.degrees(np.arctan2(mean_y, mean_x)) % 360)


def prior_agreement(measurement_deg, prior_mean_deg, k_prior):
    """How well recent (smoothed) feedback agrees with the fixed prior mean.
    Reuses k_prior — the same value that shapes the percept mixture for that
    condition (see Romi's README for the identifiability consequence)."""
    delta_rad = np.deg2rad(wrap_signed_deg(measurement_deg - prior_mean_deg))
    return float(np.exp(k_prior * (np.cos(delta_rad) - 1.0)))


def trial_percept_distribution(prior_mean_deg, measurement_deg, k_prior, k_llh,
                               prior_reliance):
    """The discrete either/or mixture: prior_reliance on the prior-centered
    component, (1 - prior_reliance) on the likelihood-centered component."""
    prior_component = vm_pdf(DEG, prior_mean_deg, k_prior)[:, 0]
    llh_component = vm_pdf(DEG, measurement_deg, k_llh)[:, 0]
    mixture = prior_reliance * prior_component + (1 - prior_reliance) * llh_component
    return mixture / mixture.sum()


def circ_convolve_vec(p, kernel):
    return np.real(np.fft.ifft(np.fft.fft(p) * np.fft.fft(kernel)))


# ===========================================================================
#  The observer
# ===========================================================================
@dataclass
class ReliabilityMixtureObserver:
    """Reliability-mixture hierarchical Bayesian observer (Romi's model).

    Free parameters (10):
        k_like   : dict coherence -> sensory concentration        (3)
        k_prior  : dict prior_std -> prior concentration          (4)
        alpha    : reliance learning rate                         (1)
        k_motor  : motor-noise concentration                      (1)
        lapse    : lapse probability                              (1)

    Fixed structure:
        prior_mean  : 225 deg.
        window_size : 5-trial smoothing window on feedback (reset per session).
        reliance0   : initial prior_reliance = 0.5.
    """

    k_like: Dict[float, float] = field(
        default_factory=lambda: {0.06: 1.0, 0.12: 3.0, 0.24: 8.0})
    k_prior: Dict[int, float] = field(
        default_factory=lambda: {80: 0.75, 40: 2.8, 20: 8.7, 10: 33.0})
    alpha: float = 0.05
    k_motor: float = 30.0
    lapse: float = 0.02
    prior_mean: float = PRIOR_MEAN
    window_size: int = WINDOW_SIZE

    # precomputed caches (filled by _prepare) — motor-convolved components
    _prepared_for: tuple = field(default=None, repr=False)
    _prior_conv: dict = field(default_factory=dict, repr=False)   # width -> (360,)
    _llh_conv: dict = field(default_factory=dict, repr=False)     # coh -> (361,360)
    _uniform: np.ndarray = field(default=None, repr=False)

    N_PARAMS = 10  # 3 k_like + 4 k_prior + alpha + k_motor + lapse

    # ----------------------------------------------------------------------
    #  Precompute motor-convolved prior & likelihood components.
    #  Circular convolution is linear and both components are normalized
    #  von Mises (sum 1), so the mixture sums to 1 and
    #    conv(r*prior + (1-r)*llh) = r*conv(prior) + (1-r)*conv(llh).
    #  Precomputing the convolved components and combining per trial is thus
    #  numerically identical (to FFT roundoff) to the original per-trial path,
    #  while removing the per-trial von-Mises rebuild + FFT. Rebuilt whenever
    #  k_like / k_prior / k_motor change (i.e. every optimiser step).
    # ----------------------------------------------------------------------
    def _prepare(self, directions, coherences, prior_stds):
        sig = (tuple(sorted(self.k_like.items())),
               tuple(sorted(self.k_prior.items())),
               float(self.k_motor))
        if self._prepared_for == sig and self._llh_conv:
            return
        self._uniform = np.ones(360) / 360.0
        motor_kernel = vm_pdf(DEG, 360.0, self.k_motor)[:, 0]

        widths = np.unique(np.asarray(prior_stds, dtype=int))
        self._prior_conv = {
            int(w): circ_convolve_vec(vm_pdf(DEG, self.prior_mean, self.k_prior[int(w)])[:, 0],
                                      motor_kernel)
            for w in widths}

        cohs = np.unique(np.asarray(coherences, dtype=float))
        self._llh_conv = {}
        for c in cohs:
            # column d (1..360) = motor-convolved likelihood centered at d
            like = vm_pdf(DEG, DEG.astype(float), self.k_like[float(c)])   # (360,360)
            conv = np.empty((361, 360))
            conv[1:] = np.array([circ_convolve_vec(like[:, j], motor_kernel)
                                 for j in range(360)])
            self._llh_conv[float(c)] = conv
        self._prepared_for = sig

    # ----------------------------------------------------------------------
    #  Roll forward over a trial sequence (chronological). Returns per-trial
    #  predicted response distributions and the reliance trajectory.
    # ----------------------------------------------------------------------
    def filter(self, directions, coherences, prior_std, session_id=None,
               feedback=None, sample: bool = False,
               rng: Optional[np.random.RandomState] = None):
        directions = np.asarray(directions, dtype=int)
        coherences = np.asarray(coherences, dtype=float)
        prior_std = np.asarray(prior_std, dtype=int)
        feedback = directions if feedback is None else np.asarray(feedback, dtype=int)
        session_id = None if session_id is None else np.asarray(session_id)
        self._prepare(directions, coherences, prior_std)

        reliance = 0.5
        window_buf = deque(maxlen=self.window_size)
        prev_session = None

        n = directions.size
        dists, responses = [], []
        reliance_trace = np.empty(n)

        for t in range(n):
            # reset the feedback window at each session boundary (== her rule)
            if session_id is not None:
                s = session_id[t]
                if prev_session is None or s != prev_session:
                    window_buf.clear()
                prev_session = s

            reliance_trace[t] = reliance
            c = float(coherences[t]); w = int(prior_std[t]); d = int(directions[t])

            with_motor = (reliance * self._prior_conv[w]
                          + (1.0 - reliance) * self._llh_conv[c][d])
            with_motor = np.clip(with_motor, 0, None)
            with_motor = with_motor / with_motor.sum()
            final = (1 - self.lapse) * with_motor + self.lapse * (1.0 / 360)
            dists.append(final)

            if sample:
                responses.append(int(rng.choice(np.arange(1, 361), p=final / final.sum())))

            # delta-rule update of reliance toward feedback-vs-prior agreement
            window_buf.append(feedback[t])
            smoothed = circular_mean_deg(list(window_buf))
            agreement = prior_agreement(smoothed, self.prior_mean, self.k_prior[w])
            reliance = float(np.clip(reliance + self.alpha * (agreement - reliance),
                                     1e-4, 1 - 1e-4))

        out = {"dists": dists, "reliance_trace": reliance_trace}
        if sample:
            out["responses"] = np.array(responses)
        return out

    # ----------------------------------------------------------------------
    #  Negative log-likelihood + information criteria.
    # ----------------------------------------------------------------------
    def negative_log_likelihood(self, estimates, directions, coherences,
                                prior_std, session_id=None, feedback=None) -> float:
        estimates = np.asarray(estimates, dtype=int)
        out = self.filter(directions, coherences, prior_std,
                          session_id=session_id, feedback=feedback, sample=False)
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
