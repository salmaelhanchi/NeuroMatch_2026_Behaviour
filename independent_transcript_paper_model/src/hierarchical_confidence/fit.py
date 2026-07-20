"""Budgeted fitting for the first 30-minute exploratory run."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, logit

from .model import HierarchicalObserver, ModelParameters


class TimeBudgetReached(RuntimeError):
    pass


@dataclass
class FitResult:
    subject_id: int
    parameters: ModelParameters
    nll: float
    initial_nll: float
    elapsed_seconds: float
    evaluations: int
    success: bool
    status: str
    history: pd.DataFrame
    raw_result: OptimizeResult | None = None

    def summary_record(self, coherence_values: np.ndarray) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "nll": self.nll,
            "initial_nll": self.initial_nll,
            "nll_improvement": self.initial_nll - self.nll,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluations": self.evaluations,
            "success": self.success,
            "status": self.status,
            **self.parameters.as_record(coherence_values),
        }


class ParameterTransform:
    def __init__(self, coherence_values: np.ndarray) -> None:
        self.coherence_values = np.asarray(coherence_values, dtype=float)
        self.n_coherences = len(self.coherence_values)

    def encode(self, parameters: ModelParameters) -> np.ndarray:
        parameters.validate(self.n_coherences)
        return np.concatenate(
            (
                [logit(parameters.rho)],
                np.log(parameters.sensory_kappas),
                [np.log(parameters.motor_kappa), logit(parameters.lapse)],
            )
        )

    def decode(self, raw: np.ndarray) -> ModelParameters:
        raw = np.asarray(raw, dtype=float)
        return ModelParameters(
            rho=float(expit(raw[0])),
            sensory_kappas=np.exp(raw[1 : 1 + self.n_coherences]),
            motor_kappa=float(np.exp(raw[-2])),
            lapse=float(expit(raw[-1])),
        )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return (
            [(float(logit(0.05)), float(logit(0.995)))]
            + [(float(np.log(0.1)), float(np.log(80.0)))] * self.n_coherences
            + [
                (float(np.log(0.5)), float(np.log(200.0))),
                (float(logit(0.0005)), float(logit(0.20))),
            ]
        )

    def default_parameters(self) -> ModelParameters:
        sensory = np.geomspace(1.5, 12.0, self.n_coherences)
        return ModelParameters(rho=0.95, sensory_kappas=sensory, motor_kappa=30.0, lapse=0.02)


def fit_subject(
    observer: HierarchicalObserver,
    max_evaluations: int = 100,
    time_budget_seconds: float = 12.0 * 60.0,
    initial_parameters: ModelParameters | None = None,
) -> FitResult:
    """Run one bounded Powell start, stopping at evaluations or wall time."""

    transform = ParameterTransform(observer.subject.coherence_values)
    initial_parameters = initial_parameters or transform.default_parameters()
    initial_raw = transform.encode(initial_parameters)
    started = perf_counter()
    history: list[dict[str, float | int]] = []
    best_raw = initial_raw.copy()
    best_nll = np.inf

    def objective(raw: np.ndarray) -> float:
        nonlocal best_raw, best_nll
        if history and perf_counter() - started >= time_budget_seconds:
            raise TimeBudgetReached
        nll = observer.negative_log_likelihood(transform.decode(raw))
        elapsed = perf_counter() - started
        history.append({"evaluation": len(history) + 1, "elapsed_seconds": elapsed, "nll": nll})
        if nll < best_nll:
            best_nll = nll
            best_raw = np.asarray(raw, dtype=float).copy()
        return nll

    result: OptimizeResult | None = None
    status = ""
    success = False
    try:
        result = minimize(
            objective,
            initial_raw,
            method="Powell",
            bounds=transform.bounds,
            options={"maxfev": int(max_evaluations), "xtol": 1e-3, "ftol": 1e-3, "disp": False},
        )
        success = bool(result.success)
        status = str(result.message)
    except TimeBudgetReached:
        status = f"Stopped after reaching the {time_budget_seconds / 60.0:g}-minute subject budget."

    elapsed = perf_counter() - started
    if not history:
        raise RuntimeError("The optimizer stopped before completing one objective evaluation.")
    initial_nll = float(history[0]["nll"])
    return FitResult(
        subject_id=observer.subject.subject_id,
        parameters=transform.decode(best_raw),
        nll=float(best_nll),
        initial_nll=initial_nll,
        elapsed_seconds=elapsed,
        evaluations=len(history),
        success=success,
        status=status,
        history=pd.DataFrame(history),
        raw_result=result,
    )
