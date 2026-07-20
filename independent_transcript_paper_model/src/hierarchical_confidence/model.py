"""Discrete hidden-confidence observer with one integrated posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import i0e, logsumexp

from .circular import angle_to_bin
from .data import PRIOR_MEAN_DEGREES
from .readout import tie_aware_map_support


@dataclass(frozen=True)
class GridSpec:
    """Fixed numerical support points; these are not fitted parameters."""

    n_angles: int = 72
    n_positive_kappa: int = 15
    kappa_min: float = 0.05
    kappa_max: float = 50.0
    prior_mean_degrees: float = PRIOR_MEAN_DEGREES

    def __post_init__(self) -> None:
        if self.n_angles < 8 or 360 % self.n_angles != 0:
            raise ValueError("n_angles must be a divisor of 360 and at least 8.")
        if self.n_positive_kappa < 2:
            raise ValueError("At least two positive kappa support points are required.")
        if not 0.0 < self.kappa_min < self.kappa_max:
            raise ValueError("Require 0 < kappa_min < kappa_max.")

    @property
    def theta_degrees(self) -> np.ndarray:
        return np.linspace(0.0, 360.0, self.n_angles, endpoint=False)

    @property
    def kappa_values(self) -> np.ndarray:
        positive = np.geomspace(self.kappa_min, self.kappa_max, self.n_positive_kappa)
        return np.concatenate(([0.0], positive))


@dataclass(frozen=True)
class ModelParameters:
    rho: float
    sensory_kappas: np.ndarray
    motor_kappa: float
    lapse: float

    def validate(self, n_coherences: int) -> None:
        sensory = np.asarray(self.sensory_kappas, dtype=float)
        if sensory.shape != (n_coherences,):
            raise ValueError(f"Expected {n_coherences} sensory kappas; found shape {sensory.shape}.")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between zero and one.")
        if np.any(sensory <= 0.0) or self.motor_kappa <= 0.0:
            raise ValueError("All concentration parameters must be positive.")
        if not 0.0 <= self.lapse < 1.0:
            raise ValueError("lapse must lie in [0, 1).")

    def as_record(self, coherence_values: np.ndarray) -> dict[str, float]:
        record = {
            "rho": float(self.rho),
            "motor_kappa": float(self.motor_kappa),
            "lapse": float(self.lapse),
        }
        for coherence, kappa in zip(coherence_values, self.sensory_kappas):
            record[f"sensory_kappa_coherence_{coherence:g}"] = float(kappa)
        return record


@dataclass(frozen=True)
class PreparedSubject:
    subject_id: int
    directions: np.ndarray
    coherence_indices: np.ndarray
    coherence_values: np.ndarray
    response_bins: np.ndarray
    response_valid: np.ndarray
    prior_std: np.ndarray
    block_ids: np.ndarray
    trial_indices: np.ndarray

    @property
    def n_trials(self) -> int:
        return len(self.directions)


def prepare_subject(data: pd.DataFrame, subject_id: int, grid: GridSpec) -> PreparedSubject:
    """Convert one already-sorted participant table into model arrays."""

    rows = data.loc[data["subject_id"] == subject_id].copy()
    if rows.empty:
        raise ValueError(f"Subject {subject_id} is not present in the data.")
    coherences = np.sort(data["motion_coherence"].dropna().unique().astype(float))
    coherence_index = np.searchsorted(coherences, rows["motion_coherence"].to_numpy(dtype=float))
    if np.any(coherence_index >= len(coherences)):
        raise ValueError("A coherence value could not be indexed.")

    valid = rows["response_valid"].to_numpy(dtype=bool)
    response_bins = np.full(len(rows), -1, dtype=int)
    response_bins[valid] = angle_to_bin(rows.loc[valid, "response_angle"].to_numpy(), grid.n_angles)
    return PreparedSubject(
        subject_id=int(subject_id),
        directions=rows["motion_direction"].to_numpy(dtype=float),
        coherence_indices=coherence_index.astype(int),
        coherence_values=coherences,
        response_bins=response_bins,
        response_valid=valid,
        prior_std=rows["prior_std"].to_numpy(dtype=float),
        block_ids=rows["block_id"].to_numpy(),
        trial_indices=rows["trial_index"].to_numpy(dtype=int),
    )


def _normalize_log_rows(log_values: np.ndarray) -> np.ndarray:
    return np.exp(log_values - logsumexp(log_values, axis=-1, keepdims=True))


def _vm_pmf_grid(means_degrees: np.ndarray, kappas: np.ndarray, theta_degrees: np.ndarray) -> np.ndarray:
    means = np.atleast_1d(means_degrees).astype(float)
    kappas = np.atleast_1d(kappas).astype(float)
    means, kappas = np.broadcast_arrays(means, kappas)
    difference = np.deg2rad(theta_degrees[None, :] - means.reshape(-1, 1))
    log_weights = kappas.reshape(-1, 1) * (np.cos(difference) - 1.0)
    return _normalize_log_rows(log_weights)


class HierarchicalObserver:
    """Evaluate one participant while preserving trial order across blocks."""

    def __init__(self, subject: PreparedSubject, grid: GridSpec, batch_size: int = 128) -> None:
        self.subject = subject
        self.grid = grid
        self.batch_size = int(batch_size)
        self.theta = grid.theta_degrees
        self.kappa = grid.kappa_values
        self._prior_basis = _vm_pmf_grid(
            np.full_like(self.kappa, grid.prior_mean_degrees),
            self.kappa,
            self.theta,
        )
        self._feedback_log_likelihood = self._make_feedback_log_likelihood()

    def _make_feedback_log_likelihood(self) -> np.ndarray:
        difference = np.deg2rad(
            self.subject.directions[:, None] - self.grid.prior_mean_degrees
        )
        log_i0 = np.log(i0e(self.kappa)) + self.kappa
        return (
            self.kappa[None, :] * np.cos(difference)
            - np.log(2.0 * np.pi)
            - log_i0[None, :]
        )

    def confidence_trajectory(self, rho: float) -> tuple[np.ndarray, np.ndarray]:
        """Return pre-feedback H_t and the final post-feedback state.

        A uniform initial mass over the fixed kappa support is an explicit pilot
        implementation choice. The state is never reset at block boundaries.
        """

        if not 0.0 < rho < 1.0:
            raise ValueError("rho must lie strictly between zero and one.")
        n_trials = self.subject.n_trials
        n_kappa = len(self.kappa)
        trajectory = np.empty((n_trials, n_kappa), dtype=float)
        log_h_after = np.full(n_kappa, -np.log(n_kappa), dtype=float)
        for trial in range(n_trials):
            log_h_before = rho * log_h_after
            log_h_before -= logsumexp(log_h_before)
            trajectory[trial] = np.exp(log_h_before)
            log_h_after = log_h_before + self._feedback_log_likelihood[trial]
            log_h_after -= logsumexp(log_h_after)
        return trajectory, np.exp(log_h_after)

    def effective_priors(self, rho: float) -> tuple[np.ndarray, np.ndarray]:
        h_before, h_final = self.confidence_trajectory(rho)
        effective = h_before @ self._prior_basis
        effective /= effective.sum(axis=1, keepdims=True)
        return effective, h_final

    def _sensory_log_likelihood(self, kappa: float) -> np.ndarray:
        difference = np.deg2rad(self.theta[:, None] - self.theta[None, :])
        log_values = kappa * (np.cos(difference) - 1.0)
        return log_values - logsumexp(log_values, axis=1, keepdims=True)

    def _measurement_pmfs(self, directions: np.ndarray, kappa: float) -> np.ndarray:
        difference = np.deg2rad(self.theta[None, :] - directions[:, None])
        log_values = kappa * (np.cos(difference) - 1.0)
        return _normalize_log_rows(log_values)

    def _motor_kernel(self, kappa: float) -> np.ndarray:
        return _vm_pmf_grid(np.array([0.0]), np.array([kappa]), self.theta)[0]

    def negative_log_likelihood(self, parameters: ModelParameters) -> float:
        """Score observed response bins; missing responses still update H_t."""

        parameters.validate(len(self.subject.coherence_values))
        priors, _ = self.effective_priors(parameters.rho)
        log_priors = np.log(np.maximum(priors, np.finfo(float).tiny))
        motor = self._motor_kernel(parameters.motor_kappa)
        total_log_likelihood = 0.0

        for coherence_index, sensory_kappa in enumerate(parameters.sensory_kappas):
            eligible = np.flatnonzero(
                (self.subject.coherence_indices == coherence_index) & self.subject.response_valid
            )
            sensory_log_likelihood = self._sensory_log_likelihood(float(sensory_kappa))
            for start in range(0, len(eligible), self.batch_size):
                indices = eligible[start : start + self.batch_size]
                posterior_scores = sensory_log_likelihood[None, :, :] + log_priors[indices, None, :]
                tied_map, tie_counts = tie_aware_map_support(posterior_scores)
                measurement = self._measurement_pmfs(
                    self.subject.directions[indices], float(sensory_kappa)
                )
                response_bin = self.subject.response_bins[indices, None]
                motor_by_percept = motor[
                    (response_bin - np.arange(self.grid.n_angles)[None, :])
                    % self.grid.n_angles
                ]
                motor_probability = (
                    np.einsum("bmt,bt->bm", tied_map, motor_by_percept, optimize=True)
                    / tie_counts
                )
                probability = np.sum(measurement * motor_probability, axis=1)
                probability = (1.0 - parameters.lapse) * probability + parameters.lapse / self.grid.n_angles
                total_log_likelihood += np.log(np.maximum(probability, np.finfo(float).tiny)).sum()
        return float(-total_log_likelihood)

    def predict_response_pmfs(self, parameters: ModelParameters) -> np.ndarray:
        """Return one full response distribution per trial for diagnostics."""

        parameters.validate(len(self.subject.coherence_values))
        priors, _ = self.effective_priors(parameters.rho)
        log_priors = np.log(np.maximum(priors, np.finfo(float).tiny))
        motor = self._motor_kernel(parameters.motor_kappa)
        predictions = np.empty((self.subject.n_trials, self.grid.n_angles), dtype=float)

        for coherence_index, sensory_kappa in enumerate(parameters.sensory_kappas):
            eligible = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            sensory_log_likelihood = self._sensory_log_likelihood(float(sensory_kappa))
            for start in range(0, len(eligible), self.batch_size):
                indices = eligible[start : start + self.batch_size]
                posterior_scores = sensory_log_likelihood[None, :, :] + log_priors[indices, None, :]
                tied_map, tie_counts = tie_aware_map_support(posterior_scores)
                measurement = self._measurement_pmfs(
                    self.subject.directions[indices], float(sensory_kappa)
                )
                percept = np.einsum(
                    "bm,bmt->bt",
                    measurement / tie_counts,
                    tied_map,
                    optimize=True,
                )
                convolved = np.fft.ifft(
                    np.fft.fft(percept, axis=1) * np.fft.fft(motor)[None, :],
                    axis=1,
                ).real
                convolved = np.maximum(convolved, 0.0)
                convolved /= convolved.sum(axis=1, keepdims=True)
                predictions[indices] = (
                    (1.0 - parameters.lapse) * convolved
                    + parameters.lapse / self.grid.n_angles
                )
        return predictions

    def tie_diagnostics(self, parameters: ModelParameters) -> dict[str, float | int]:
        """Summarize how often measurement-conditional posteriors have tied MAP bins."""

        parameters.validate(len(self.subject.coherence_values))
        priors, _ = self.effective_priors(parameters.rho)
        log_priors = np.log(np.maximum(priors, np.finfo(float).tiny))
        total_pairs = 0
        tied_pairs = 0
        expected_tie_events = 0.0
        max_tie_count = 1

        for coherence_index, sensory_kappa in enumerate(parameters.sensory_kappas):
            eligible = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            sensory_log_likelihood = self._sensory_log_likelihood(float(sensory_kappa))
            for start in range(0, len(eligible), self.batch_size):
                indices = eligible[start : start + self.batch_size]
                posterior_scores = sensory_log_likelihood[None, :, :] + log_priors[indices, None, :]
                _, tie_counts = tie_aware_map_support(posterior_scores)
                tie_event = tie_counts > 1
                measurement = self._measurement_pmfs(
                    self.subject.directions[indices], float(sensory_kappa)
                )
                total_pairs += tie_event.size
                tied_pairs += int(tie_event.sum())
                expected_tie_events += float(np.sum(measurement * tie_event))
                max_tie_count = max(max_tie_count, int(tie_counts.max()))

        return {
            "trial_measurement_pairs": total_pairs,
            "tied_trial_measurement_pairs": tied_pairs,
            "unweighted_tie_rate": tied_pairs / max(total_pairs, 1),
            "mean_expected_tie_probability": expected_tie_events / self.subject.n_trials,
            "maximum_tied_map_bins": max_tie_count,
        }

    def state_summary(self, rho: float) -> pd.DataFrame:
        h_before, _ = self.confidence_trajectory(rho)
        expected_kappa = h_before @ self.kappa
        modal_kappa = self.kappa[np.argmax(h_before, axis=1)]
        entropy = -np.sum(h_before * np.log(np.maximum(h_before, np.finfo(float).tiny)), axis=1)
        return pd.DataFrame(
            {
                "subject_id": self.subject.subject_id,
                "subject_trial_position": np.arange(self.subject.n_trials),
                "trial_index": self.subject.trial_indices,
                "block_id": self.subject.block_ids,
                "prior_std_diagnostic_only": self.subject.prior_std,
                "expected_hidden_kappa": expected_kappa,
                "modal_hidden_kappa": modal_kappa,
                "hidden_state_entropy": entropy,
            }
        )
