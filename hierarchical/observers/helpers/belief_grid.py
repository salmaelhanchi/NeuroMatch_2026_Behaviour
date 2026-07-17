"""
belief_grid.py
==============

Numerical machinery for the belief over prior strength used by the learning
observers (``online_switching_observer.py`` and the models built on it). Nothing
here is model *logic* — these are the grid, table, and reference-computation
utilities kept out of the equation-level model files. Circular primitives are
reused from ``circular.py``.

Contents
--------
- make_k_grid            : the discretised support for the belief over k_prior.
- observation_likelihood_table : V(f; 225, k) for every (k, feedback direction).
- forget                 : the volatility / predict step (leak toward the base belief).
- bayes_correct          : the multiply-and-renormalise correct step.
- batch_posterior        : an independent brute-force belief, used as a cross-check
                           that the recursive update matches the batch posterior.
- expected_prior_weight  : E_belief[ k/(k+k_e) ], the belief-averaged switch weight.
"""

from __future__ import annotations

import numpy as np

from observers.helpers.circular import von_mises_pdfs, DIRECTION_SPACE


# ---------------------------------------------------------------------------
# The k-grid: the discrete support on which the belief b(k) lives.
# ---------------------------------------------------------------------------
def make_k_grid(k_min: float = 0.01, k_max: float = 60.0, n: int = 40):
    """Log-spaced grid of prior-concentration values k.

    Log spacing because the perceptually meaningful differences in a von Mises
    concentration are multiplicative (k = 1 vs 2 matters as much as 10 vs 20).
    Returns the grid values (used both as the belief support and to evaluate
    switch weights).
    """
    return np.geomspace(k_min, k_max, n)


# ---------------------------------------------------------------------------
# Precomputed observation likelihoods for the belief update (H1b).
# ---------------------------------------------------------------------------
def observation_likelihood_table(k_grid: np.ndarray,
                                 prior_mean: float = 225.0) -> np.ndarray:
    """Table ``T[i, d-1] = V(direction d ; prior_mean, k_grid[i])``.

    Row i is the likelihood, under a prior of strength ``k_grid[i]``, of seeing
    a feedback direction at each of the 360 possible directions. Updating the
    belief after feedback ``f`` is then just multiplying the current belief by
    column ``f-1`` of this table (H1b) — no per-trial von Mises evaluation.
    """
    # von_mises_pdfs returns (360, n_means); we want one column per k, so build
    # by treating each k as its own von Mises centred on the prior mean.
    cols = [von_mises_pdfs(DIRECTION_SPACE, prior_mean, k, normalize=True).ravel()
            for k in k_grid]
    # stack -> (n_k, 360)
    return np.vstack(cols)


# ---------------------------------------------------------------------------
# Belief dynamics (H1a / H1b)
# ---------------------------------------------------------------------------
def forget(belief: np.ndarray, belief0: np.ndarray, lam: float) -> np.ndarray:
    """Volatility / predict step (H1a): leak the belief toward the initial
    hyper-belief by fraction ``lam``. ``lam=0`` -> pure accumulation (no
    forgetting); ``lam=1`` -> belief resets to ``belief0`` every trial."""
    if lam <= 0:
        return belief
    return (1.0 - lam) * belief + lam * belief0


def bayes_correct(belief_pred: np.ndarray, obs_like_col: np.ndarray) -> np.ndarray:
    """Correct step (H1b): multiply the predicted belief by the observation
    likelihood for the feedback direction, then renormalise to sum 1."""
    post = belief_pred * obs_like_col
    s = post.sum()
    if s <= 0:
        # numerical underflow guard: fall back to the predicted belief
        return belief_pred / belief_pred.sum()
    return post / s


# ---------------------------------------------------------------------------
# Independent reference for Phase-4 verification
# ---------------------------------------------------------------------------
def batch_posterior(belief0: np.ndarray, obs_table: np.ndarray,
                    feedback_dirs) -> np.ndarray:
    """Brute-force belief over k from a whole feedback history (NO recursion).

    Computes ``b(k) ∝ b0(k) · Π_t V(f_t; 225, k)`` directly in log-space. This
    shares no code with the recursive ``forget``/``bayes_correct`` path, so
    matching it (in the λ=0 case) validates the online update independently —
    the Phase-4 cross-check demanded by the spec.
    """
    feedback_dirs = np.asarray(feedback_dirs, dtype=int)
    logb = np.log(np.clip(belief0, 1e-320, None))
    # sum log-likelihood contributions across trials for each k
    logL = np.log(np.clip(obs_table[:, feedback_dirs - 1], 1e-320, None)).sum(axis=1)
    logpost = logb + logL
    logpost -= logpost.max()
    post = np.exp(logpost)
    return post / post.sum()


# ---------------------------------------------------------------------------
# Switch weight under the belief (H2)
# ---------------------------------------------------------------------------
def expected_prior_weight(belief: np.ndarray, k_grid: np.ndarray,
                          k_e: float) -> float:
    """E_belief[ k / (k + k_e) ]  — the reliability-ratio switch weight (Eq. 6)
    averaged over the current belief about the prior strength (H2).

    If the evidence is perfectly flat (k_e = 0) the prior always wins the
    competition (weight 1), matching the static model's guard.
    """
    if k_e == 0:
        return 1.0
    ratio = k_grid / (k_grid + k_e)
    return float(np.sum(belief * ratio))