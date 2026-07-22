"""
hb_integrate_before.py
======================

**Recombined hierarchical Bayesian observer** — the "best-of" cross of
HB - Rachel (`hb_integration.py`) and HB - Salma (branch
`hierarchical_confidence/model.py`), built to serve the abstract's three
targets (always integrate; learn prior confidence online; emergent, not
imposed, bimodality).

It differs from HB - Rachel in exactly ONE structural respect — the
prior/evidence combination rule (the "defining axis"):

    HB - Rachel  (integrate-then-average):  percept = Σ_κ b(κ) · readout_κ
                 one MAP read-out PER kappa, averaged after the fact.

    THIS model   (integrate-then-read-out): percept = readout( Σ_κ b(κ) · prior_κ )
                 collapse the belief into ONE effective prior, ONE read-out.

Everything else is deliberately taken from HB - Rachel and left byte-identical
by inheritance:
  * the explicit mixture prior  α·V(θ;225,κ) + (1-α)/360   (Salma has no α;
    we keep Rachel's floor because it is what makes far-from-prior bimodality
    emerge robustly and it cleanly separates "how much to trust the prior"
    (α) from "how tight it is" (κ));
  * the learned belief b_t(κ) over prior concentration;
  * the LINEAR leak-toward-anchor forgetting  (1-λ)b + λ·b₀  (Salma forgets
    geometrically; Rachel's linear form re-opens ruled-out widths in one trial,
    which suits the task's ABRUPT block boundaries, and its anchor b₀ can be
    made informative);
  * the 360-bin (1°) grid and plain MAP read-out (so no tie-aware step needed);
  * motor convolution + lapse.

Parameter count is unchanged at 7 (3 k_like + α + k_motor + p_random + λ):
the swap of combination rule adds no parameters, so any AIC comparison against
the switching observer stays clean.

This module ADDS a class; it imports and subclasses the existing
HBRachelObserver and touches none of its code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observers.helpers.circular import (
    DIRECTION_SPACE,
    von_mises_pdfs,
    circular_convolution,
)
from observers.models.hb_rachel import (
    HBRachelObserver,
    mixture_prior,
    _map_readout,
    _map_readout_col,
)


@dataclass
class HBIntegrateBeforeObserver(HBRachelObserver):
    """Integrate-then-read-out recombination (see module docstring).

    Same 7 free parameters and same fields as HBRachelObserver; only the
    combination rule changes. Overrides `_prepare` (cache the per-kappa prior
    BASES + the coherence-specific likelihood/measurement pdfs, instead of the
    per-kappa read-out STACK) and `estimate_distribution` (one effective prior,
    one MAP read-out).
    """

    # extra caches (filled by _prepare); parent's _readout stays empty here
    _prior_bases: np.ndarray = None          # (nk, 360) mixture prior per kappa
    _like_c: dict = None                     # coherence -> (360,360)
    _m_pdfs_c: dict = None                   # coherence -> (360, D)
    _dirs: np.ndarray = None                 # unique displayed directions
    _motor: np.ndarray = None                # (360,1) motor kernel

    def _prepare(self, directions: np.ndarray, coherences: np.ndarray):
        sig = (tuple(sorted(self.k_like.items())), float(self.alpha),
               float(self.k_motor), self.k_grid.size, "integrate_before")
        if self._prepared_for == sig and self._prior_bases is not None:
            return
        di = DIRECTION_SPACE.astype(float)
        dirs = np.unique(np.asarray(directions, dtype=int))
        self._dirs = dirs
        self._dir_col = {int(d): j for j, d in enumerate(dirs)}
        self._uniform = np.ones(360) / 360.0
        self._motor = von_mises_pdfs(np.arange(0, 360), 0, self.k_motor,
                                     normalize=True)                  # (360,1)

        # per-coherence sensory pieces (depend only on k_like[c])
        self._like_c, self._m_pdfs_c = {}, {}
        for c in np.unique(np.asarray(coherences, dtype=float)):
            self._like_c[c] = von_mises_pdfs(di, di, self.k_like[c],
                                             normalize=True)          # (360,360)
            self._m_pdfs_c[c] = von_mises_pdfs(di, dirs, self.k_like[c],
                                               normalize=True)        # (360,D)

        # per-kappa PRIOR BASES (the belief will average THESE, before readout)
        self._prior_bases = np.vstack([
            mixture_prior(kap, self.alpha, self.prior_mean)           # (360,)
            for kap in self.k_grid])                                  # (nk,360)

        # belief-update likelihood table (identical to parent M1): mixture prior
        vm = np.vstack([
            von_mises_pdfs(DIRECTION_SPACE, self.prior_mean, kap,
                           normalize=True).ravel()
            for kap in self.k_grid])                                  # (nk,360)
        self._obs_table = self.alpha * vm + (1.0 - self.alpha) / 360.0

        self._belief0 = self.initial_belief()
        self._prepared_for = sig

    def estimate_distribution(self, coherence: float, direction: int,
                              belief: np.ndarray) -> np.ndarray:
        # (1) collapse the belief over kappa into ONE effective prior
        eff_prior = belief @ self._prior_bases                        # (360,)
        eff_prior = eff_prior / eff_prior.sum()
        # (2) ONE MAP read-out on that single prior, for THIS trial's direction
        # only. Building the full (360,360) L and keeping one column wastes
        # ~360x; _map_readout_col computes just the needed column (verified
        # identical to _map_readout(...)[:, direction-1] to 1e-16).
        col = _map_readout_col(self._like_c[coherence], self._m_pdfs_c[coherence],
                               eff_prior, self._dirs, int(direction))   # (360,)
        # (3) motor convolution (one percept, so convolve once)
        percept = circular_convolution(col[:, None], self._motor)[:, 0]
        # (4) lapse (paper convention, same as parent)
        dist = (percept + self.p_random * self._uniform) / (1.0 + self.p_random)
        s = dist.sum()
        return dist / s if s > 0 else self._uniform.copy()
