"""
hb_adaptive_confidence.py
=========================

**The abstract's model, taken literally.** The project abstract specifies a
"mixed distribution hyperprior" — a naive (uniform) prior and an informed
(spiked von Mises) prior held simultaneously — plus "a hyperparameter that holds
the confidence of the observer regarding ... a naive (uniform) prior or an
informed (spiked von Mises) prior, **which is updated over trials**."

The existing `hb_integration` observer has the mixed prior but its confidence
weight α is a *fixed* fitted parameter — only the prior's width κ is learned. So
it satisfies the mixed-prior clause but NOT the "confidence updated over trials"
clause. This model closes that gap: **both** the confidence α (uniform vs.
informed) **and** the width κ are latent quantities the observer learns online
from feedback. Nothing about the prior is fixed a priori.

How
---
The observer carries a joint belief ``b_t(κ, α)`` over a grid of (width,
confidence) pairs. Each candidate pair defines a mixed prior over direction

    p(θ | κ, α) = α · V(θ; 225, κ)  +  (1 − α) · (1/360)          (mixed prior)

At the end of each trial the revealed direction ``f`` is treated as one draw
from the prior, so its likelihood under a pair is exactly that pair's prior
density at ``f``. The belief is updated by the same predict/correct filter the
other HB models use — forget slightly toward the flat initial belief (volatility
λ), then multiply by the feedback likelihood and renormalise:

    b_t(κ,α) ∝ [(1−λ) b_{t−1}(κ,α) + λ b_0(κ,α)] · p(f | κ, α)

The estimate marginalises the read-out over the belief (integrate-after, the
`hb_integration` convention): one MAP read-out per grid pair, averaged by the
belief, then motor noise and a lapse. Because the belief puts mass on both
low-α (near-uniform) and high-α (informed) pairs, the observer's *effective
confidence in the prior* — E[α] — rises when feedback clusters near 225° and
falls when feedback is spread, trial by trial, exactly as the abstract requires.

Free parameters (6): three sensory concentrations ``k_like`` (per coherence),
``k_motor``, ``p_random`` (lapse), and ``lam`` (volatility). Both α and κ are
**learned, not fitted**, so this model has one fewer parameter than
`hb_integration` (which fits α). The prior confidence and width are emergent.

Caveat: from a single feedback direction, a wide von Mises (low κ) and a mostly
uniform prior (low α) look similar, so κ and α trade off along a ridge in the
belief. This is expected — the joint belief represents that ambiguity honestly —
but it means fitted/recovered κ and α should be read jointly, and a
parameter-recovery check is worth running before interpreting either alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from observers.helpers.circular import (
    DIRECTION_SPACE,
    von_mises_pdfs,
    circular_convolution,
    von_mises_std,
)
from observers.helpers.belief_grid import make_k_grid, forget, bayes_correct
from observers.models.hb_integration import (
    mixture_prior, _map_readout, _map_readout_cols, PRIOR_MEAN)


def make_alpha_grid(n: int = 9) -> np.ndarray:
    """Support for the belief over the prior-confidence α, spanning naive
    (α=0, uniform) to fully informed (α=1, von Mises)."""
    return np.linspace(0.0, 1.0, n)


@dataclass
class HBAdaptiveConfidenceObserver:
    """Hierarchical Bayesian observer that learns BOTH the prior's confidence α
    and its width κ online (the abstract's model, taken literally).

    Free parameters (6): k_like (3) + k_motor + p_random + lam. α and κ are
    latent and learned, not fitted.
    """

    k_like: Dict[float, float] = field(
        default_factory=lambda: {0.24: 8.0, 0.12: 3.0, 0.06: 1.0})
    k_motor: float = 40.0
    p_random: float = 0.01
    lam: float = 0.05
    prior_mean: float = PRIOR_MEAN
    k_grid: np.ndarray = field(default_factory=lambda: make_k_grid(n=15))
    a_grid: np.ndarray = field(default_factory=lambda: make_alpha_grid(9))

    # precomputed caches (filled by _prepare)
    _prepared_for: tuple = field(default=None, repr=False)
    _readout: dict = field(default_factory=dict, repr=False)   # coh -> (G,360,D)
    _dir_col: dict = field(default_factory=dict, repr=False)
    _uniform: np.ndarray = field(default=None, repr=False)
    _table: np.ndarray = field(default=None, repr=False)       # (G,360) mixed prior per grid pair
    _kap_flat: np.ndarray = field(default=None, repr=False)    # (G,) kappa per grid pair
    _alp_flat: np.ndarray = field(default=None, repr=False)    # (G,) alpha per grid pair
    _belief0: np.ndarray = field(default=None, repr=False)

    # ----------------------------------------------------------------------
    def _joint_grid(self):
        """Flattened (κ, α) support pairs."""
        K, A = np.meshgrid(self.k_grid, self.a_grid, indexing="ij")
        return K.ravel(), A.ravel()

    def initial_belief(self) -> np.ndarray:
        kap, _ = self._joint_grid()
        n = kap.size
        return np.ones(n) / n

    # ----------------------------------------------------------------------
    def _prepare(self, directions: np.ndarray, coherences: np.ndarray):
        sig = (tuple(sorted(self.k_like.items())), float(self.k_motor),
               self.k_grid.size, self.a_grid.size)
        if self._prepared_for == sig and self._readout:
            return
        di = DIRECTION_SPACE.astype(float)
        dirs = np.unique(np.asarray(directions, dtype=int))
        cohs = np.unique(np.asarray(coherences, dtype=float))
        self._dir_col = {int(d): j for j, d in enumerate(dirs)}
        self._uniform = np.ones(360) / 360.0
        self._kap_flat, self._alp_flat = self._joint_grid()
        G = self._kap_flat.size

        motor = von_mises_pdfs(np.arange(0, 360), 0, self.k_motor,
                               normalize=True)                          # (360,1)

        # Mixed prior per grid pair. This same table IS the feedback-likelihood
        # table: p(f | κ, α) = prior density at f (feedback = a draw from prior).
        self._table = np.vstack([
            mixture_prior(self._kap_flat[g], self._alp_flat[g], self.prior_mean)
            for g in range(G)])                                         # (G,360)

        # Per-coherence read-out stack: one MAP read-out per grid pair (the
        # belief averages these — integrate-after, as in hb_integration).
        self._readout = {}
        for c in cohs:
            like_c = von_mises_pdfs(di, di, self.k_like[c], normalize=True)   # (360,360)
            m_pdfs_c = von_mises_pdfs(di, dirs, self.k_like[c], normalize=True)  # (360,D)
            stack = np.empty((G, 360, dirs.size))
            for g in range(G):
                # only the D displayed-direction columns are ever used; skip the
                # (360,360) scatter+nansum in _map_readout (~1.6x per-eval win,
                # verified identical to _map_readout(...)[:, dirs-1] to 0.0).
                L = _map_readout_cols(like_c, m_pdfs_c, self._table[g])        # (360,D)
                stack[g] = circular_convolution(L, motor)
            # Store direction-major and C-contiguous so the per-trial slice
            # self._readout[c][j] is a contiguous (G,360) matrix. The natural
            # (G,360,D) layout makes [:, :, j] a STRIDED view, which throws the
            # belief@readouts matmul off the BLAS fast path (~20x slower per
            # eval). This transpose is the single largest fit-speed lever.
            self._readout[c] = np.ascontiguousarray(
                np.transpose(stack, (2, 0, 1)))                               # (D,G,360)

        self._belief0 = self.initial_belief()
        self._prepared_for = sig

    # ----------------------------------------------------------------------
    def update_belief(self, belief: np.ndarray, feedback_dir: int) -> np.ndarray:
        """Predict/correct update of the joint belief over (κ, α) from feedback."""
        belief = forget(belief, self._belief0, self.lam)
        return bayes_correct(belief, self._table[:, feedback_dir - 1])

    def estimate_distribution(self, coherence: float, direction: int,
                              belief: np.ndarray) -> np.ndarray:
        """Belief-averaged read-out for one condition (integrate-after) + lapse."""
        j = self._dir_col[int(direction)]
        readouts = self._readout[coherence][j]              # (G,360) contiguous
        percept = belief @ readouts                         # (360,)
        dist = (percept + self.p_random * self._uniform) / (1.0 + self.p_random)
        s = dist.sum()
        return dist / s if s > 0 else self._uniform.copy()

    # ----------------------------------------------------------------------
    def filter(self, directions, coherences, feedback=None,
               sample: bool = False, rng: Optional[np.random.RandomState] = None,
               record_belief: bool = False):
        directions = np.asarray(directions, dtype=int)
        coherences = np.asarray(coherences, dtype=float)
        feedback = directions if feedback is None else np.asarray(feedback, dtype=int)
        self._prepare(directions, coherences)

        belief = self._belief0.copy()
        dists, responses, believed_sd, believed_alpha = [], [], [], []
        for t in range(directions.size):
            dist = self.estimate_distribution(coherences[t], directions[t], belief)
            dists.append(dist)
            if sample:
                responses.append(int(rng.choice(np.arange(1, 361), p=dist)))
            if record_belief:
                k_mean = float(np.sum(belief * self._kap_flat))
                a_mean = float(np.sum(belief * self._alp_flat))
                believed_sd.append(von_mises_std(max(k_mean, 1e-6), self.prior_mean))
                believed_alpha.append(a_mean)
            belief = self.update_belief(belief, feedback[t])

        out = {"dists": dists}
        if sample:
            out["responses"] = np.array(responses)
        if record_belief:
            out["believed_sd"] = np.array(believed_sd)
            out["believed_alpha"] = np.array(believed_alpha)   # the learned confidence
        return out

    # ----------------------------------------------------------------------
    N_PARAMS = 6  # 3 k_e + k_motor + p_random + lam  (alpha AND kappa are learned)

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


# ---------------------------------------------------------------------------
# Demonstration: the confidence α is LEARNED trial-by-trial (the abstract's ask)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    obs = HBAdaptiveConfidenceObserver()
    rng = np.random.RandomState(0)
    n = 400
    dirs = rng.randint(1, 361, n)
    cohs = rng.choice([0.06, 0.12, 0.24], n)

    # (1) INFORMED world: feedback drawn from a tight prior at 225 -> the
    #     observer should grow confident the prior is informed (E[α] rises).
    p_informed = von_mises_pdfs(DIRECTION_SPACE, 225.0, 8.0, normalize=True).ravel()
    fb_informed = rng.choice(np.arange(1, 361), size=n, p=p_informed)
    out_i = obs.filter(dirs, cohs, feedback=fb_informed, record_belief=True)

    # (2) NAIVE world: feedback uniform over directions -> the observer should
    #     lose confidence in the prior (E[α] falls).
    fb_uniform = rng.randint(1, 361, n)
    out_u = obs.filter(dirs, cohs, feedback=fb_uniform, record_belief=True)

    print("Prior CONFIDENCE E[α] is updated over trials (the abstract's requirement):")
    print(f"  informed feedback (tight prior): E[α] {out_i['believed_alpha'][0]:.2f} "
          f"-> {out_i['believed_alpha'][-1]:.2f}   (should RISE)")
    print(f"  naive feedback (uniform):        E[α] {out_u['believed_alpha'][0]:.2f} "
          f"-> {out_u['believed_alpha'][-1]:.2f}   (should FALL)")
    print("\nPrior WIDTH (believed SD, deg) is also learned:")
    print(f"  informed feedback: believed SD {out_i['believed_sd'][0]:.0f} "
          f"-> {out_i['believed_sd'][-1]:.0f}  (true prior tight ~ SD 20)")
    d = out_i["dists"][-1]
    print(f"\nresponse distribution valid: sum={d.sum():.6f}, min={d.min():.2e}")
    print(f"free parameters: {HBAdaptiveConfidenceObserver.N_PARAMS} "
          f"(alpha and kappa learned, not fitted)")
