"""
hb_integration_derived.py
==========================

**Hierarchical Bayesian integration observer with a DERIVED mixture weight** —
the variant that implements the *original authors'* proposal from the
Laquitaine & Gardner (2018) Discussion, rather than the free-alpha model in
``hb_integration.py``.

Why this file exists
--------------------
``hb_integration.py`` puts a mixed hyper-prior on direction

    p(theta | kappa, alpha) = alpha * V(theta; 225, kappa) + (1 - alpha)/360

and treats ``alpha`` (the peaked-vs-uniform weight) as a FREE, trial-constant
parameter. When fit, subject 1 drove alpha -> 1.0, deleting the uniform
component and collapsing to plain single-von-Mises integration — i.e. the model
shed the very mixture the abstract proposed.

The paper's Discussion (p11-12) proposed something more specific:

    "...motion directions were drawn from either a uniform or a peaked
     distribution at each trial, WITH PROBABILITIES DETERMINED BY THE RATIO OF
     LIKELIHOOD AND PRIOR STRENGTHS. This would effectively amount to a
     reformulation of our Switching observer in Bayesian terminology."

So the mixing weight is NOT free — it is COMPUTED from the reliability ratio,
exactly the switch model's Eq. 6:

    alpha(kappa, k_e) = kappa / (kappa + k_e)                          (D1)

This observer implements (D1): per kappa-grid point and coherence, the weight on
the peaked prior lobe rises with the (learned) prior strength kappa and falls
with the sensory strength k_e. There is NO free alpha. kappa is still learned
online (belief b_t(kappa), forgetting lam), so alpha EVOLVES trial-by-trial as
the belief sharpens — the dynamic the paper never modelled (it fit static
priors).

What changes vs. hb_integration.py
----------------------------------
1. ``alpha`` is removed from the parameter set (6 params, not 7):
   3 k_e + k_motor + p_random + lam.
2. In ``_prepare`` the per-(coherence, kappa) read-out uses alpha_i = kappa_i /
   (kappa_i + k_e) instead of a shared fixed alpha.
3. The belief-update feedback likelihood is the PURE von Mises V(f; 225, kappa)
   (kappa parameterises the peaked prior's WIDTH; the uniform lobe is the
   read-out competitor, not part of the generative prior over the feedback
   direction). This matches ``online_learner``'s belief update, keeping the two
   learning models point-for-point comparable.

Everything else (the MAP push-forward ``_map_readout``, motor convolution,
lapse renormalisation, the deterministic-belief likelihood, AIC/BIC) is
inherited unchanged from ``HBIntegrationObserver``.

Predicted behaviour (verified)
------------------------------
- Bimodality still EMERGES with no free alpha (far-from-prior + low coherence
  gives a stimulus peak and a prior peak).
- Prior reliance now DECLINES with |stimulus - 225| (0.59 -> 0.44 over 30-120
  deg at coh 0.06, uniform belief) — the falsifiable signature that
  distinguishes an integration read-out from the switch model's direction-flat
  weight (hb_integration_verify.py documents the switch model's flat 0.27->0.24).
- As the paper predicted, this weight IS the switch reliability ratio, so the
  model is expected to behave much like the Switching observer; the value of
  the comparison is quantifying HOW close, and whether online kappa-learning
  adds anything the static switch lacks.

This is the model to fit alongside the free-alpha version: their difference
answers "does the uniform mixture ever earn its place, or is the
reliability-determined weight the only version that matches subjects?".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circular import DIRECTION_SPACE, von_mises_pdfs, circular_convolution
from hb_integration import HBIntegrationObserver, mixture_prior, _map_readout


@dataclass
class HBIntegrationDerivedObserver(HBIntegrationObserver):
    """Integration observer whose mixture weight is derived from the
    reliability ratio kappa/(kappa+k_e) (paper Discussion), not fitted.

    Free parameters (6): 3 k_e + k_motor + p_random + lam.  (No alpha.)
    The inherited ``alpha`` field is ignored by this subclass — the weight is
    computed per kappa-grid point in ``_prepare``.
    """

    N_PARAMS = 6  # alpha removed

    def _prepare(self, directions, coherences):
        # cache signature EXCLUDES alpha (there is none) and tags the variant
        sig = (tuple(sorted(self.k_like.items())), float(self.k_motor),
               self.k_grid.size, "derived")
        if self._prepared_for == sig and self._readout:
            return
        dirs = np.unique(np.asarray(directions, dtype=int))
        cohs = np.unique(np.asarray(coherences, dtype=float))
        self._dir_col = {int(d): j for j, d in enumerate(dirs)}

        motor = von_mises_pdfs(np.arange(0, 360), 0, self.k_motor, normalize=True)
        self._uniform = np.ones(360) / 360.0
        di = DIRECTION_SPACE.astype(float)

        # (D1) per-coherence, per-kappa MAP read-out with the DERIVED weight.
        # like_c / m_pdfs_c depend only on k_like[c]; built once per coherence.
        self._readout = {}
        for c in cohs:
            k_e = self.k_like[c]
            like_c = von_mises_pdfs(di, di, k_e, normalize=True)
            m_pdfs_c = von_mises_pdfs(di, dirs, k_e, normalize=True)
            stack = np.empty((self.k_grid.size, 360, dirs.size))
            for i, kap in enumerate(self.k_grid):
                alpha_i = 1.0 if k_e == 0 else kap / (kap + k_e)   # <-- (D1)
                prior = mixture_prior(kap, alpha_i, self.prior_mean)
                L = _map_readout(like_c, m_pdfs_c, prior, dirs, di)
                L = np.nan_to_num(L[:, dirs - 1], nan=0.0)
                stack[i] = circular_convolution(L, motor)
            self._readout[c] = stack

        # belief update: PURE von Mises feedback likelihood V(f;225,kappa),
        # i.e. learn the peaked prior's width (same as online_learner).
        self._obs_table = np.vstack([
            von_mises_pdfs(DIRECTION_SPACE, self.prior_mean, kap,
                           normalize=True).ravel()
            for kap in self.k_grid])

        self._belief0 = self.initial_belief()
        self._prepared_for = sig


if __name__ == "__main__":
    # smoke test: bimodality emerges and prior reliance declines with distance
    obs = HBIntegrationDerivedObserver(
        k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
        k_motor=30.0, p_random=0.02, lam=0.05)
    stims = [int((225 - d) % 360) or 360 for d in (30, 60, 90, 120)]
    obs.filter(np.array(stims), np.full(len(stims), 0.06),
               feedback=np.array(stims), sample=False)
    b0 = obs._belief0.copy()
    print("prior-window mass vs |stimulus - 225| (coh 0.06, uniform belief):")
    for d, s in zip((30, 60, 90, 120), stims):
        dd = obs.estimate_distribution(0.06, s, b0)
        print(f"  {d:3d} deg: {dd[195:255].sum():.3f}")
