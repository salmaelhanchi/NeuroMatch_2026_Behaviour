"""Adaptive prior-versus-evidence response mixture without prior-width labels."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import expit, logit

from .base import PredictionResult, StandardizedObserver
from .circular import (
    circular_difference_degrees,
    circular_mean_degrees,
    measurement_pmfs,
    vm_pmf,
)


class ReliabilityMixtureObserver(StandardizedObserver):
    """Standardized version of the nested reliability-mixture implementation."""

    name = "reliability_mixture"

    @property
    def raw_parameter_names(self) -> tuple[str, ...]:
        sensory = tuple(f"log_sensory_kappa_{c:g}" for c in self.subject.coherence_values)
        return sensory + (
            "log_prior_kappa",
            "logit_learning_rate",
            "log_motor_kappa",
            "logit_lapse",
        )

    @property
    def raw_bounds(self) -> tuple[tuple[float, float], ...]:
        sensory = ((np.log(0.1), np.log(80.0)),) * len(self.subject.coherence_values)
        return sensory + (
            (np.log(0.05), np.log(80.0)),
            (logit(0.005), logit(0.50)),
            (np.log(0.5), np.log(200.0)),
            (logit(0.0005), logit(0.20)),
        )

    def default_raw_parameters(self) -> np.ndarray:
        sensory = np.geomspace(1.5, 12.0, len(self.subject.coherence_values))
        return np.concatenate(
            (np.log(sensory), [np.log(2.7), logit(0.10), np.log(30.0), logit(0.02)])
        )

    def decode(self, raw: np.ndarray) -> dict[str, object]:
        values = np.asarray(raw, dtype=float)
        n = len(self.subject.coherence_values)
        return {
            "sensory_kappas": np.exp(values[:n]),
            "prior_kappa": float(np.exp(values[n])),
            "learning_rate": float(expit(values[n + 1])),
            "motor_kappa": float(np.exp(values[n + 2])),
            "lapse": float(expit(values[n + 3])),
        }

    def parameter_record(self, raw: np.ndarray) -> dict[str, float]:
        parameters = self.decode(raw)
        record = {
            "prior_kappa": parameters["prior_kappa"],
            "learning_rate": parameters["learning_rate"],
            "motor_kappa": parameters["motor_kappa"],
            "lapse": parameters["lapse"],
        }
        for coherence, kappa in zip(self.subject.coherence_values, parameters["sensory_kappas"]):
            record[f"sensory_kappa_{coherence:g}"] = float(kappa)
        return record

    def reliance_trajectory(self, prior_kappa: float, learning_rate: float) -> np.ndarray:
        trajectory = np.empty(self.subject.n_trials, dtype=float)
        history: deque[float] = deque(maxlen=5)
        reliance = 0.5
        for trial, direction in enumerate(self.subject.directions):
            if self.subject.reset_before[trial]:
                history.clear()
                reliance = 0.5
            trajectory[trial] = reliance
            history.append(float(direction))
            smoothed_feedback = circular_mean_degrees(list(history))
            difference = circular_difference_degrees(
                smoothed_feedback,
                self.grid.prior_mean_degrees,
            )
            agreement = np.exp(float(prior_kappa) * (np.cos(np.deg2rad(difference)) - 1.0))
            reliance = float(
                np.clip(
                    reliance + float(learning_rate) * (agreement - reliance),
                    1e-4,
                    1.0 - 1e-4,
                )
            )
        return trajectory

    def predict(self, raw: np.ndarray) -> PredictionResult:
        parameters = self.decode(raw)
        reliance = self.reliance_trajectory(
            parameters["prior_kappa"],
            parameters["learning_rate"],
        )
        prior_component = vm_pmf(
            self.theta,
            self.grid.prior_mean_degrees,
            parameters["prior_kappa"],
        )[0]
        evidence = np.empty((self.subject.n_trials, self.grid.n_angles), dtype=float)
        for coherence_index, sensory_kappa in enumerate(parameters["sensory_kappas"]):
            indices = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            evidence[indices] = measurement_pmfs(
                self.subject.directions[indices],
                float(sensory_kappa),
                self.theta,
            )
        percept_pmfs = (
            reliance[:, None] * prior_component[None, :]
            + (1.0 - reliance[:, None]) * evidence
        )
        response_pmfs = self.motor_and_lapse(
            percept_pmfs,
            parameters["motor_kappa"],
            parameters["lapse"],
        )
        return PredictionResult(
            response_pmfs=response_pmfs,
            state=reliance,
            state_label="prior reliance",
        )

