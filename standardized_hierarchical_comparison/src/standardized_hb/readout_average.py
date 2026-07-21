"""Kappa-conditional posterior readouts averaged at the response level."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from .base import PredictionResult, StandardizedObserver
from .circular import (
    measurement_pmfs,
    sensory_log_likelihood,
    tie_aware_map_weights,
    vm_log_density_at,
)


class ReadoutAverageObserver(StandardizedObserver):
    """Standardized version of `hierarchical/observers/models/hb_integration.py`."""

    name = "readout_average"

    @property
    def raw_parameter_names(self) -> tuple[str, ...]:
        sensory = tuple(f"log_sensory_kappa_{c:g}" for c in self.subject.coherence_values)
        return sensory + ("logit_alpha", "log_motor_kappa", "logit_lapse", "logit_lambda")

    @property
    def raw_bounds(self) -> tuple[tuple[float, float], ...]:
        sensory = ((np.log(0.1), np.log(80.0)),) * len(self.subject.coherence_values)
        return sensory + (
            (logit(0.05), logit(0.95)),
            (np.log(0.5), np.log(200.0)),
            (logit(0.0005), logit(0.20)),
            (logit(0.001), logit(0.50)),
        )

    def default_raw_parameters(self) -> np.ndarray:
        sensory = np.geomspace(1.5, 12.0, len(self.subject.coherence_values))
        return np.concatenate(
            (np.log(sensory), [logit(0.60), np.log(30.0), logit(0.02), logit(0.05)])
        )

    def decode(self, raw: np.ndarray) -> dict[str, object]:
        values = np.asarray(raw, dtype=float)
        n = len(self.subject.coherence_values)
        return {
            "sensory_kappas": np.exp(values[:n]),
            "alpha": float(expit(values[n])),
            "motor_kappa": float(np.exp(values[n + 1])),
            "lapse": float(expit(values[n + 2])),
            "lambda": float(expit(values[n + 3])),
        }

    def parameter_record(self, raw: np.ndarray) -> dict[str, float]:
        parameters = self.decode(raw)
        record = {
            "alpha": parameters["alpha"],
            "motor_kappa": parameters["motor_kappa"],
            "lapse": parameters["lapse"],
            "lambda": parameters["lambda"],
        }
        for coherence, kappa in zip(self.subject.coherence_values, parameters["sensory_kappas"]):
            record[f"sensory_kappa_{coherence:g}"] = float(kappa)
        return record

    def belief_trajectory(self, alpha: float, forgetting: float) -> np.ndarray:
        base = np.full(self.kappa.size, 1.0 / self.kappa.size)
        trajectory = np.empty((self.subject.n_trials, self.kappa.size), dtype=float)
        vm_density = np.exp(
            vm_log_density_at(
                self.subject.directions,
                self.grid.prior_mean_degrees,
                self.kappa,
            )
        )
        feedback_likelihood = alpha * vm_density + (1.0 - alpha) / (2.0 * np.pi)
        belief = base.copy()
        for trial in range(self.subject.n_trials):
            if self.subject.reset_before[trial]:
                belief = base.copy()
            trajectory[trial] = belief
            predicted = (1.0 - forgetting) * belief + forgetting * base
            belief = predicted * feedback_likelihood[trial]
            belief /= belief.sum()
        return trajectory

    def predict(self, raw: np.ndarray) -> PredictionResult:
        parameters = self.decode(raw)
        alpha = parameters["alpha"]
        beliefs = self.belief_trajectory(alpha, parameters["lambda"])
        priors = alpha * self.prior_basis + (1.0 - alpha) * self.grid.uniform
        log_priors = np.log(np.maximum(priors, np.finfo(float).tiny))
        percept_pmfs = np.empty((self.subject.n_trials, self.grid.n_angles), dtype=float)

        for coherence_index, sensory_kappa in enumerate(parameters["sensory_kappas"]):
            indices = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            log_likelihood = sensory_log_likelihood(float(sensory_kappa), self.theta)
            readout_by_kappa = np.stack(
                [tie_aware_map_weights(log_likelihood + log_prior[None, :]) for log_prior in log_priors],
                axis=0,
            )
            for start in range(0, indices.size, self.batch_size):
                batch = indices[start : start + self.batch_size]
                measurements = measurement_pmfs(
                    self.subject.directions[batch],
                    float(sensory_kappa),
                    self.theta,
                )
                conditional = np.einsum(
                    "bm,kmt->bkt",
                    measurements,
                    readout_by_kappa,
                    optimize=True,
                )
                percept_pmfs[batch] = np.einsum(
                    "bk,bkt->bt",
                    beliefs[batch],
                    conditional,
                    optimize=True,
                )

        response_pmfs = self.motor_and_lapse(
            percept_pmfs,
            parameters["motor_kappa"],
            parameters["lapse"],
        )
        return PredictionResult(
            response_pmfs=response_pmfs,
            state=beliefs @ self.kappa,
            state_label="expected prior kappa",
        )

