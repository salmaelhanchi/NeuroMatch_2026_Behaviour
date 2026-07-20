"""Deterministic initial values for participant-level multi-start fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .fit import ParameterTransform
from .model import ModelParameters


def _latin_hypercube(n_samples: int, n_dimensions: int, seed: int) -> np.ndarray:
    """Generate a small reproducible Latin hypercube without extra dependencies."""

    if n_samples < 1 or n_dimensions < 1:
        raise ValueError("Latin-hypercube dimensions and sample count must be positive.")
    rng = np.random.default_rng(seed)
    samples = (np.arange(n_samples)[:, None] + rng.random((n_samples, n_dimensions))) / n_samples
    for dimension in range(n_dimensions):
        rng.shuffle(samples[:, dimension])
    return samples


def generate_multistart_parameters(
    coherence_values: Sequence[float],
    n_starts: int,
    seed: int,
) -> list[ModelParameters]:
    """Return one declared default and stratified random valid parameter starts."""

    coherence_values = np.asarray(coherence_values, dtype=float)
    if coherence_values.ndim != 1 or len(coherence_values) < 1:
        raise ValueError("At least one coherence value is required.")
    if n_starts < 1:
        raise ValueError("n_starts must be at least one.")

    transform = ParameterTransform(coherence_values)
    starts = [transform.default_parameters()]
    if n_starts == 1:
        return starts

    random = _latin_hypercube(n_starts - 1, len(coherence_values) + 3, seed)
    for row in random:
        rho = float(expit(logit(0.15) + row[0] * (logit(0.98) - logit(0.15))))
        sensory = np.exp(
            np.log(0.5)
            + row[1 : 1 + len(coherence_values)] * (np.log(80.0) - np.log(0.5))
        )
        sensory.sort()
        motor = float(
            np.exp(
                np.log(1.0)
                + row[-2] * (np.log(100.0) - np.log(1.0))
            )
        )
        lapse = float(
            expit(logit(0.002) + row[-1] * (logit(0.15) - logit(0.002)))
        )
        parameters = ModelParameters(
            rho=rho,
            sensory_kappas=sensory,
            motor_kappa=motor,
            lapse=lapse,
        )
        parameters.validate(len(coherence_values))
        starts.append(parameters)
    return starts


def parameters_from_record(
    record: Mapping[str, object],
    coherence_values: Sequence[float],
    prefix: str = "",
) -> ModelParameters:
    """Reconstruct parameters from a saved summary or start-schedule record."""

    coherence_values = np.asarray(coherence_values, dtype=float)
    parameters = ModelParameters(
        rho=float(record[f"{prefix}rho"]),
        sensory_kappas=np.asarray(
            [record[f"{prefix}sensory_kappa_coherence_{value:g}"] for value in coherence_values],
            dtype=float,
        ),
        motor_kappa=float(record[f"{prefix}motor_kappa"]),
        lapse=float(record[f"{prefix}lapse"]),
    )
    parameters.validate(len(coherence_values))
    return parameters


def make_multistart_schedule(
    subject_ids: Sequence[int],
    coherence_values: Sequence[float],
    n_starts: int,
    base_seed: int,
) -> pd.DataFrame:
    """Create an auditable participant-by-start table of initial parameters."""

    records: list[dict[str, object]] = []
    for subject_id in subject_ids:
        subject_seed = int(base_seed + 1009 * int(subject_id))
        starts = generate_multistart_parameters(coherence_values, n_starts, subject_seed)
        for start_id, parameters in enumerate(starts):
            record: dict[str, object] = {
                "subject_id": int(subject_id),
                "start_id": int(start_id),
                "start_source": "declared_default" if start_id == 0 else "stratified_random",
                "subject_seed": subject_seed,
                "initial_rho": parameters.rho,
                "initial_motor_kappa": parameters.motor_kappa,
                "initial_lapse": parameters.lapse,
            }
            for coherence, kappa in zip(coherence_values, parameters.sensory_kappas):
                record[f"initial_sensory_kappa_coherence_{coherence:g}"] = float(kappa)
            records.append(record)
    return pd.DataFrame(records)
