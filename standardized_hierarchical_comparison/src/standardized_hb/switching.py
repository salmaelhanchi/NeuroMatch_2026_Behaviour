"""Standardized original Switching observer from the supplied starter notebook."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from .base import PredictionResult, StandardizedObserver
from .circular import angle_to_bin, measurement_pmfs


class SwitchingObserver(StandardizedObserver):
    """Reliability-ratio switch between prior and sensory response branches."""

    name = "switching_observer"

    @property
    def raw_parameter_names(self) -> tuple[str, ...]:
        sensory = tuple(f"log_sensory_kappa_{c:g}" for c in self.subject.coherence_values)
        return sensory + ("log_prior_kappa", "log_motor_kappa", "logit_lapse")

    @property
    def raw_bounds(self) -> tuple[tuple[float, float], ...]:
        sensory = ((np.log(0.1), np.log(80.0)),) * len(self.subject.coherence_values)
        return sensory + (
            (np.log(0.05), np.log(80.0)),
            (np.log(0.5), np.log(200.0)),
            (logit(0.0005), logit(0.20)),
        )

    def default_raw_parameters(self) -> np.ndarray:
        sensory = np.geomspace(1.5, 12.0, len(self.subject.coherence_values))
        return np.concatenate(
            (np.log(sensory), [np.log(2.7), np.log(30.0), logit(0.02)])
        )

    def decode(self, raw: np.ndarray) -> dict[str, object]:
        values = np.asarray(raw, dtype=float)
        n = len(self.subject.coherence_values)
        return {
            "sensory_kappas": np.exp(values[:n]),
            "prior_kappa": float(np.exp(values[n])),
            "motor_kappa": float(np.exp(values[n + 1])),
            "lapse": float(expit(values[n + 2])),
        }

    def parameter_record(self, raw: np.ndarray) -> dict[str, float]:
        parameters = self.decode(raw)
        record = {
            "prior_kappa": parameters["prior_kappa"],
            "motor_kappa": parameters["motor_kappa"],
            "lapse": parameters["lapse"],
        }
        for coherence, kappa in zip(
            self.subject.coherence_values,
            parameters["sensory_kappas"],
        ):
            record[f"sensory_kappa_{coherence:g}"] = float(kappa)
        return record

    def predict(self, raw: np.ndarray) -> PredictionResult:
        parameters = self.decode(raw)
        prior_kappa = float(parameters["prior_kappa"])
        prior_bin = int(angle_to_bin(self.grid.prior_mean_degrees, self.grid.n_angles))
        prior_component = np.zeros(self.grid.n_angles, dtype=float)
        prior_component[prior_bin] = 1.0

        percept_pmfs = np.empty((self.subject.n_trials, self.grid.n_angles), dtype=float)
        prior_reliance = np.empty(self.subject.n_trials, dtype=float)
        for coherence_index, sensory_kappa in enumerate(parameters["sensory_kappas"]):
            indices = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            sensory_kappa = float(sensory_kappa)
            switch_probability = prior_kappa / (prior_kappa + sensory_kappa)
            sensory_component = measurement_pmfs(
                self.subject.directions[indices],
                sensory_kappa,
                self.theta,
            )
            percept_pmfs[indices] = (
                switch_probability * prior_component[None, :]
                + (1.0 - switch_probability) * sensory_component
            )
            prior_reliance[indices] = switch_probability

        response_pmfs = self.motor_and_lapse(
            percept_pmfs,
            parameters["motor_kappa"],
            parameters["lapse"],
        )
        return PredictionResult(
            response_pmfs=response_pmfs,
            state=prior_reliance,
            state_label="prior-switch probability",
        )
