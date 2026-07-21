"""Shared grid, prediction result, scoring, and observer interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .circular import vm_pmf

if TYPE_CHECKING:
    from .data import PreparedSubject


@dataclass(frozen=True)
class GridSpec:
    n_angles: int = 72
    n_positive_kappa: int = 9
    kappa_min: float = 0.05
    kappa_max: float = 50.0
    prior_mean_degrees: float = 225.0

    def __post_init__(self) -> None:
        if self.n_angles < 12 or 360 % self.n_angles != 0:
            raise ValueError("n_angles must divide 360 and be at least 12.")
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

    @property
    def uniform(self) -> np.ndarray:
        return np.full(self.n_angles, 1.0 / self.n_angles)


@dataclass(frozen=True)
class PredictionResult:
    response_pmfs: np.ndarray
    state: np.ndarray
    state_label: str


class StandardizedObserver:
    """Base class enforcing one response likelihood across all observers."""

    name = "base"

    def __init__(self, subject: "PreparedSubject", grid: GridSpec, batch_size: int = 128) -> None:
        self.subject = subject
        self.grid = grid
        self.batch_size = int(batch_size)
        self.theta = grid.theta_degrees
        self.kappa = grid.kappa_values
        self.prior_basis = vm_pmf(
            self.theta,
            np.full(self.kappa.shape, grid.prior_mean_degrees),
            self.kappa,
        )

    @property
    def parameter_count(self) -> int:
        return len(self.raw_parameter_names)

    @property
    def raw_parameter_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def raw_bounds(self) -> tuple[tuple[float, float], ...]:
        raise NotImplementedError

    def default_raw_parameters(self) -> np.ndarray:
        raise NotImplementedError

    def decode(self, raw: np.ndarray) -> dict[str, object]:
        raise NotImplementedError

    def parameter_record(self, raw: np.ndarray) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, raw: np.ndarray) -> PredictionResult:
        raise NotImplementedError

    def negative_log_likelihood(self, raw: np.ndarray) -> float:
        prediction = self.predict(raw).response_pmfs
        valid_indices = np.flatnonzero(self.subject.response_valid)
        probabilities = prediction[
            valid_indices,
            self.subject.response_bins[valid_indices],
        ]
        return float(-np.log(np.maximum(probabilities, np.finfo(float).tiny)).sum())

    def motor_and_lapse(
        self,
        percept_pmfs: np.ndarray,
        motor_kappa: float,
        lapse: float,
    ) -> np.ndarray:
        from .circular import circular_convolve_rows

        motor = vm_pmf(self.theta, 0.0, float(motor_kappa))[0]
        convolved = circular_convolve_rows(percept_pmfs, motor)
        return (1.0 - float(lapse)) * convolved + float(lapse) * self.grid.uniform

