"""Chronological held-out scoring utilities for participant-level comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .base import PredictionResult
from .data import PreparedSubject


@dataclass(frozen=True)
class HoldoutSplit:
    train_mask: np.ndarray
    test_mask: np.ndarray
    heldout_sessions: tuple[object, ...]


def last_sessions_holdout(subject: PreparedSubject, n_sessions: int = 2) -> HoldoutSplit:
    """Hold out complete final sessions while preserving chronological order."""

    session_starts = np.flatnonzero(subject.reset_before)
    sessions = subject.session_ids[session_starts]
    if n_sessions < 1 or n_sessions >= sessions.size:
        raise ValueError("n_sessions must leave at least one complete training session.")
    heldout_sessions = tuple(sessions[-n_sessions:].tolist())
    test_mask = np.isin(subject.session_ids, heldout_sessions)
    train_mask = ~test_mask
    if not np.any(train_mask & subject.response_valid):
        raise ValueError("The training split has no valid responses.")
    if not np.any(test_mask & subject.response_valid):
        raise ValueError("The held-out split has no valid responses.")
    return HoldoutSplit(
        train_mask=train_mask,
        test_mask=test_mask,
        heldout_sessions=heldout_sessions,
    )


def with_response_mask(subject: PreparedSubject, mask: np.ndarray) -> PreparedSubject:
    """Keep the full causal sequence but score responses only where mask is true."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (subject.n_trials,):
        raise ValueError("Response mask must contain one value per trial.")
    return replace(subject, response_valid=subject.response_valid & selected)


def observed_log_scores(
    subject: PreparedSubject,
    prediction: PredictionResult,
    mask: np.ndarray,
) -> np.ndarray:
    """Return one natural-log predictive score per selected valid response."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (subject.n_trials,):
        raise ValueError("Score mask must contain one value per trial.")
    if prediction.response_pmfs.shape[0] != subject.n_trials:
        raise ValueError("Prediction rows do not align with participant trials.")
    indices = np.flatnonzero(selected & subject.response_valid)
    probabilities = prediction.response_pmfs[indices, subject.response_bins[indices]]
    return np.log(np.maximum(probabilities, np.finfo(float).tiny))


def predictive_score_record(
    model_name: str,
    log_scores: np.ndarray,
    n_response_bins: int,
) -> dict[str, float | int | str]:
    """Summarize held-out log probability on interpretable scales."""

    values = np.asarray(log_scores, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Predictive log scores must be finite and nonempty.")
    mean_log_score = float(values.mean())
    uniform_log_score = -np.log(float(n_response_bins))
    return {
        "model": model_name,
        "heldout_trials": int(values.size),
        "heldout_elpd": float(values.sum()),
        "mean_log_score": mean_log_score,
        "bits_per_trial_over_uniform": float(
            (mean_log_score - uniform_log_score) / np.log(2.0)
        ),
    }


def paired_block_bootstrap_difference(
    first_log_scores: np.ndarray,
    second_log_scores: np.ndarray,
    block_ids: np.ndarray,
    *,
    n_draws: int = 5000,
    seed: int = 20260721,
) -> dict[str, float | int]:
    """Bootstrap the paired mean log-score difference by complete trial blocks."""

    first = np.asarray(first_log_scores, dtype=float)
    second = np.asarray(second_log_scores, dtype=float)
    blocks = np.asarray(block_ids)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Paired log-score arrays must have the same one-dimensional shape.")
    if blocks.shape[0] != first.size:
        raise ValueError("block_ids must align with the paired log scores.")
    if first.size == 0 or not np.all(np.isfinite(first - second)):
        raise ValueError("Paired log-score differences must be finite and nonempty.")
    if n_draws < 100:
        raise ValueError("n_draws must be at least 100.")

    block_matrix = blocks.reshape(-1, 1) if blocks.ndim == 1 else blocks
    _, inverse = np.unique(block_matrix.astype(str), axis=0, return_inverse=True)
    n_blocks = int(inverse.max()) + 1
    if n_blocks < 2:
        raise ValueError("Paired block bootstrap requires at least two blocks.")

    difference = first - second
    block_sums = np.bincount(inverse, weights=difference, minlength=n_blocks)
    block_counts = np.bincount(inverse, minlength=n_blocks)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_blocks, size=(n_draws, n_blocks))
    draws = block_sums[sampled].sum(axis=1) / block_counts[sampled].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "ci_2.5%": float(lower),
        "ci_97.5%": float(upper),
        "probability_first_better": float(np.mean(draws > 0.0)),
        "blocks": n_blocks,
        "bootstrap_draws": int(n_draws),
    }
