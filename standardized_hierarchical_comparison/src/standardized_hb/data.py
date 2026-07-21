"""One shared data contract that never exposes experimental prior width."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .base import GridSpec
from .circular import angle_to_bin, wrap_unsigned_degrees


PRIOR_MEAN_DEGREES = 225.0
REQUIRED_COLUMNS = {
    "trial_index",
    "motion_direction",
    "motion_coherence",
    "estimate_x",
    "estimate_y",
    "prior_mean",
    "subject_id",
    "session_id",
    "run_id",
}
ORDER_COLUMNS = ["subject_id", "session_id", "run_id", "trial_index"]


@dataclass(frozen=True)
class PreparedSubject:
    subject_id: int
    directions: np.ndarray
    coherence_indices: np.ndarray
    coherence_values: np.ndarray
    response_bins: np.ndarray
    response_valid: np.ndarray
    session_ids: np.ndarray
    run_ids: np.ndarray
    trial_indices: np.ndarray
    reset_before: np.ndarray

    @property
    def n_trials(self) -> int:
        return int(self.directions.size)

    @property
    def n_valid_responses(self) -> int:
        return int(self.response_valid.sum())


def load_participant(
    csv_path: str | Path,
    subject_id: int,
    grid: GridSpec,
) -> tuple[PreparedSubject, dict[str, object]]:
    """Load one participant without retaining or using `prior_std`."""

    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    header = pd.read_csv(path, nrows=0)
    missing = sorted(REQUIRED_COLUMNS.difference(header.columns))
    if missing:
        raise ValueError(f"Raw data are missing required columns: {missing}")
    raw = pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS))
    duplicates = raw.duplicated(ORDER_COLUMNS, keep=False)
    if duplicates.any():
        raise ValueError("Composite participant/session/run/trial keys must be unique.")

    rows = raw.loc[raw["subject_id"] == int(subject_id)].copy()
    if rows.empty:
        raise ValueError(f"Participant {subject_id} is not present in {path}.")
    rows = rows.sort_values(ORDER_COLUMNS, kind="mergesort").reset_index(drop=True)

    prior_means = rows["prior_mean"].dropna().to_numpy(dtype=float)
    if prior_means.size == 0 or not np.allclose(prior_means, grid.prior_mean_degrees):
        raise ValueError(f"Expected a fixed prior mean of {grid.prior_mean_degrees:g} degrees.")

    x = pd.to_numeric(rows["estimate_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["estimate_y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    response_angles = np.full(len(rows), np.nan, dtype=float)
    response_angles[valid] = wrap_unsigned_degrees(np.rad2deg(np.arctan2(y[valid], x[valid])))
    response_bins = np.full(len(rows), -1, dtype=int)
    response_bins[valid] = angle_to_bin(response_angles[valid], grid.n_angles)

    coherence_values = np.sort(rows["motion_coherence"].dropna().unique().astype(float))
    coherence_indices = np.searchsorted(
        coherence_values,
        rows["motion_coherence"].to_numpy(dtype=float),
    )
    session_ids = rows["session_id"].to_numpy()
    reset_before = np.ones(len(rows), dtype=bool)
    reset_before[1:] = session_ids[1:] != session_ids[:-1]

    subject = PreparedSubject(
        subject_id=int(subject_id),
        directions=rows["motion_direction"].to_numpy(dtype=float),
        coherence_indices=coherence_indices.astype(int),
        coherence_values=coherence_values,
        response_bins=response_bins,
        response_valid=valid,
        session_ids=session_ids,
        run_ids=rows["run_id"].to_numpy(),
        trial_indices=rows["trial_index"].to_numpy(dtype=int),
        reset_before=reset_before,
    )
    audit = {
        "csv_path": str(path),
        "subject_id": int(subject_id),
        "trials": subject.n_trials,
        "valid_responses": subject.n_valid_responses,
        "sessions": int(np.unique(subject.session_ids).size),
        "runs": int(np.unique(np.column_stack((subject.session_ids, subject.run_ids)), axis=0).shape[0]),
        "coherences": subject.coherence_values.tolist(),
        "prior_mean_degrees": float(grid.prior_mean_degrees),
        "prior_std_used": False,
    }
    return subject, audit
