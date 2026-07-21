"""One effective prior, one posterior, and one tie-aware MAP readout."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit, logsumexp

from .base import PredictionResult, StandardizedObserver
from .circular import (
    measurement_pmfs,
    sensory_log_likelihood,
    tie_aware_map_weights,
    vm_log_density_at,
)


class IntegratedPriorObserver(StandardizedObserver):
    """Standardized version of the independent hidden-confidence observer."""

    name = "integrated_prior"

    @property
    def raw_parameter_names(self) -> tuple[str, ...]:
        sensory = tuple(f"log_sensory_kappa_{c:g}" for c in self.subject.coherence_values)
        return ("logit_rho",) + sensory + ("log_motor_kappa", "logit_lapse")

    @property
    def raw_bounds(self) -> tuple[tuple[float, float], ...]:
        sensory = ((np.log(0.1), np.log(80.0)),) * len(self.subject.coherence_values)
        return (
            (logit(0.05), logit(0.995)),
        ) + sensory + (
            (np.log(0.5), np.log(200.0)),
            (logit(0.0005), logit(0.20)),
        )

    def default_raw_parameters(self) -> np.ndarray:
        sensory = np.geomspace(1.5, 12.0, len(self.subject.coherence_values))
        return np.concatenate(([logit(0.95)], np.log(sensory), [np.log(30.0), logit(0.02)]))

    def decode(self, raw: np.ndarray) -> dict[str, object]:
        values = np.asarray(raw, dtype=float)
        n = len(self.subject.coherence_values)
        return {
            "rho": float(expit(values[0])),
            "sensory_kappas": np.exp(values[1 : 1 + n]),
            "motor_kappa": float(np.exp(values[-2])),
            "lapse": float(expit(values[-1])),
        }

    def parameter_record(self, raw: np.ndarray) -> dict[str, float]:
        parameters = self.decode(raw)
        record = {
            "rho": parameters["rho"],
            "motor_kappa": parameters["motor_kappa"],
            "lapse": parameters["lapse"],
        }
        for coherence, kappa in zip(self.subject.coherence_values, parameters["sensory_kappas"]):
            record[f"sensory_kappa_{coherence:g}"] = float(kappa)
        return record

    def confidence_trajectory(self, rho: float) -> np.ndarray:
        n_kappa = self.kappa.size
        base_log = np.full(n_kappa, -np.log(n_kappa), dtype=float)
        feedback_log_likelihood = vm_log_density_at(
            self.subject.directions,
            self.grid.prior_mean_degrees,
            self.kappa,
        )
        trajectory = np.empty((self.subject.n_trials, n_kappa), dtype=float)
        log_after = base_log.copy()
        for trial in range(self.subject.n_trials):
            if self.subject.reset_before[trial]:
                log_after = base_log.copy()
            log_before = float(rho) * log_after
            log_before -= logsumexp(log_before)
            trajectory[trial] = np.exp(log_before)
            log_after = log_before + feedback_log_likelihood[trial]
            log_after -= logsumexp(log_after)
        return trajectory

    def predict(self, raw: np.ndarray) -> PredictionResult:
        parameters = self.decode(raw)
        confidence = self.confidence_trajectory(parameters["rho"])
        effective_priors = confidence @ self.prior_basis
        effective_priors /= effective_priors.sum(axis=1, keepdims=True)
        log_priors = np.log(np.maximum(effective_priors, np.finfo(float).tiny))
        percept_pmfs = np.empty((self.subject.n_trials, self.grid.n_angles), dtype=float)

        for coherence_index, sensory_kappa in enumerate(parameters["sensory_kappas"]):
            indices = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            log_likelihood = sensory_log_likelihood(float(sensory_kappa), self.theta)
            for start in range(0, indices.size, self.batch_size):
                batch = indices[start : start + self.batch_size]
                posterior = log_likelihood[None, :, :] + log_priors[batch, None, :]
                readout = tie_aware_map_weights(posterior)
                measurements = measurement_pmfs(
                    self.subject.directions[batch],
                    float(sensory_kappa),
                    self.theta,
                )
                percept_pmfs[batch] = np.einsum(
                    "bm,bmt->bt",
                    measurements,
                    readout,
                    optimize=True,
                )

        response_pmfs = self.motor_and_lapse(
            percept_pmfs,
            parameters["motor_kappa"],
            parameters["lapse"],
        )
        return PredictionResult(
            response_pmfs=response_pmfs,
            state=confidence @ self.kappa,
            state_label="expected prior kappa",
        )

