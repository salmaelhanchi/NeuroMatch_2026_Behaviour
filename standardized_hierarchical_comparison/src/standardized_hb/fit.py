"""One bounded multistart fitting routine shared by all three observers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from .base import StandardizedObserver


@dataclass(frozen=True)
class FitConfig:
    n_starts: int = 2
    max_evaluations_per_start: int = 60
    seed: int = 20260721


@dataclass
class FitResult:
    model_name: str
    raw_parameters: np.ndarray
    parameters: dict[str, float]
    nll: float
    aic: float
    bic: float
    elapsed_seconds: float
    evaluations: int
    best_start: int
    success: bool
    message: str
    history: pd.DataFrame
    raw_result: OptimizeResult | None = None

    def summary_record(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "nll": self.nll,
            "aic": self.aic,
            "bic": self.bic,
            "parameters": self.parameters,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluations": self.evaluations,
            "best_start": self.best_start,
            "success": self.success,
            "message": self.message,
        }


def _starts(observer: StandardizedObserver, config: FitConfig) -> list[np.ndarray]:
    if config.n_starts < 1:
        raise ValueError("n_starts must be at least one.")
    default = observer.default_raw_parameters()
    bounds = np.asarray(observer.raw_bounds, dtype=float)
    rng = np.random.default_rng(config.seed)
    starts = [default]
    width = bounds[:, 1] - bounds[:, 0]
    for _ in range(1, config.n_starts):
        jittered = default + rng.normal(0.0, 0.18, size=default.size) * width
        starts.append(np.clip(jittered, bounds[:, 0], bounds[:, 1]))
    return starts


def fit_model(observer: StandardizedObserver, config: FitConfig | None = None) -> FitResult:
    config = config or FitConfig()
    histories: list[dict[str, object]] = []
    best_result: OptimizeResult | None = None
    best_start = -1
    best_value = np.inf
    total_evaluations = 0
    started = perf_counter()

    for start_id, initial in enumerate(_starts(observer, config)):
        def objective(raw: np.ndarray) -> float:
            nonlocal total_evaluations
            total_evaluations += 1
            try:
                value = observer.negative_log_likelihood(raw)
            except (FloatingPointError, RuntimeError, ValueError):
                value = 1e12
            if not np.isfinite(value):
                value = 1e12
            histories.append(
                {
                    "model": observer.name,
                    "start": start_id,
                    "evaluation": total_evaluations,
                    "elapsed_seconds": perf_counter() - started,
                    "nll": float(value),
                }
            )
            return float(value)

        result = minimize(
            objective,
            initial,
            method="Powell",
            bounds=observer.raw_bounds,
            options={
                "maxfev": int(config.max_evaluations_per_start),
                "xtol": 1e-3,
                "ftol": 1e-3,
                "disp": False,
            },
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_result = result
            best_start = start_id

    if best_result is None:
        raise RuntimeError(f"No optimization result was produced for {observer.name}.")
    n = observer.subject.n_valid_responses
    k = observer.parameter_count
    return FitResult(
        model_name=observer.name,
        raw_parameters=np.asarray(best_result.x, dtype=float),
        parameters=observer.parameter_record(best_result.x),
        nll=best_value,
        aic=2.0 * k + 2.0 * best_value,
        bic=k * np.log(n) + 2.0 * best_value,
        elapsed_seconds=perf_counter() - started,
        evaluations=total_evaluations,
        best_start=best_start,
        success=bool(best_result.success),
        message=str(best_result.message),
        history=pd.DataFrame(histories),
        raw_result=best_result,
    )

