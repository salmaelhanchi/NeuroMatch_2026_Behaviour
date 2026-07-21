"""Shared circular probability and readout operations."""

from __future__ import annotations

import numpy as np
from scipy.special import i0e, logsumexp


MAP_ROUND_DECIMALS = 6


def wrap_signed_degrees(values: np.ndarray | float) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def wrap_unsigned_degrees(values: np.ndarray | float) -> np.ndarray:
    return np.asarray(values, dtype=float) % 360.0


def circular_difference_degrees(
    angle: np.ndarray | float,
    reference: np.ndarray | float,
) -> np.ndarray:
    return wrap_signed_degrees(np.asarray(angle) - np.asarray(reference))


def angle_to_bin(angle_degrees: np.ndarray | float, n_angles: int) -> np.ndarray:
    step = 360.0 / n_angles
    wrapped = wrap_unsigned_degrees(angle_degrees)
    return np.floor((wrapped + step / 2.0) / step).astype(int) % n_angles


def vm_pmf(
    support_degrees: np.ndarray,
    means_degrees: np.ndarray | float,
    kappas: np.ndarray | float,
) -> np.ndarray:
    """Return one normalized discrete von Mises row per mean/kappa pair."""

    means = np.atleast_1d(np.asarray(means_degrees, dtype=float))
    concentrations = np.atleast_1d(np.asarray(kappas, dtype=float))
    means, concentrations = np.broadcast_arrays(means, concentrations)
    difference = np.deg2rad(
        np.asarray(support_degrees, dtype=float)[None, :] - means.reshape(-1, 1)
    )
    log_weights = concentrations.reshape(-1, 1) * (np.cos(difference) - 1.0)
    return np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))


def vm_log_density_at(
    observations_degrees: np.ndarray,
    mean_degrees: float,
    kappas: np.ndarray,
) -> np.ndarray:
    """Continuous von Mises log density for each observation and kappa."""

    observations = np.asarray(observations_degrees, dtype=float)[:, None]
    concentrations = np.asarray(kappas, dtype=float)[None, :]
    difference = np.deg2rad(observations - float(mean_degrees))
    log_i0 = np.log(i0e(concentrations)) + concentrations
    return concentrations * np.cos(difference) - np.log(2.0 * np.pi) - log_i0


def sensory_log_likelihood(kappa: float, theta_degrees: np.ndarray) -> np.ndarray:
    """Log p(theta | internal measurement) with rows indexing measurements."""

    theta = np.asarray(theta_degrees, dtype=float)
    difference = np.deg2rad(theta[None, :] - theta[:, None])
    scores = float(kappa) * (np.cos(difference) - 1.0)
    return scores - logsumexp(scores, axis=1, keepdims=True)


def measurement_pmfs(
    directions_degrees: np.ndarray,
    sensory_kappa: float,
    theta_degrees: np.ndarray,
) -> np.ndarray:
    return vm_pmf(theta_degrees, np.asarray(directions_degrees), float(sensory_kappa))


def tie_aware_map_weights(
    log_posterior_scores: np.ndarray,
    decimals: int = MAP_ROUND_DECIMALS,
) -> np.ndarray:
    """Assign equal readout mass to posterior bins tied at the MAP."""

    posterior = np.exp(
        np.asarray(log_posterior_scores, dtype=float)
        - logsumexp(log_posterior_scores, axis=-1, keepdims=True)
    )
    np.round(posterior, decimals=decimals, out=posterior)
    is_mode = posterior == posterior.max(axis=-1, keepdims=True)
    counts = is_mode.sum(axis=-1, keepdims=True)
    if np.any(counts < 1):
        raise RuntimeError("Every posterior must have at least one MAP bin.")
    return is_mode / counts


def circular_convolve_rows(pmfs: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(pmfs, dtype=float)
    convolved = np.fft.ifft(
        np.fft.fft(values, axis=-1) * np.fft.fft(np.asarray(kernel, dtype=float)),
        axis=-1,
    ).real
    convolved = np.maximum(convolved, 0.0)
    totals = convolved.sum(axis=-1, keepdims=True)
    return convolved / np.where(totals > 0.0, totals, 1.0)


def circular_mean_degrees(angles_degrees: list[float] | np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)

