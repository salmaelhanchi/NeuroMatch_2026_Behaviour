"""Posterior readout rules shared by fitting and prediction."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


MAP_ROUND_DECIMALS = 6


def tie_aware_map_support(
    log_posterior_scores: np.ndarray,
    decimals: int = MAP_ROUND_DECIMALS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tied MAP bins and their counts after normalized-posterior rounding.

    The last axis is direction. Every tied maximum receives equal readout mass.
    """

    log_posterior_scores = np.asarray(log_posterior_scores, dtype=float)
    posterior = np.exp(
        log_posterior_scores
        - logsumexp(log_posterior_scores, axis=-1, keepdims=True)
    )
    np.round(posterior, decimals=decimals, out=posterior)
    maxima = np.max(posterior, axis=-1, keepdims=True)
    tied = posterior == maxima
    tie_counts = tied.sum(axis=-1)
    if np.any(tie_counts < 1):
        raise RuntimeError("Every posterior must have at least one MAP bin.")
    return tied, tie_counts
