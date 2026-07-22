"""Data loading, transparent derived variables, and pilot selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .circular import circular_difference_degrees, wrap_degrees_unsigned


REQUIRED_COLUMNS = {
    "trial_index",
    "motion_direction",
    "motion_coherence",
    "estimate_x",
    "estimate_y",
    "prior_std",
    "prior_mean",
    "subject_id",
    "session_id",
    "run_id",
}

ORDER_COLUMNS = ["subject_id", "session_id", "run_id", "trial_index"]
PRIOR_MEAN_DEGREES = 225.0


def _block_phase(position: pd.Series, size: pd.Series) -> pd.Categorical:
    denominator = np.maximum(size.to_numpy(dtype=float) - 1.0, 1.0)
    fraction = position.to_numpy(dtype=float) / denominator
    labels = np.where(fraction < 1.0 / 3.0, "early", np.where(fraction < 2.0 / 3.0, "middle", "late"))
    return pd.Categorical(labels, categories=["early", "middle", "late"], ordered=True)


def load_and_prepare_data(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load raw trials and derive only variables justified by the project question."""

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Raw data are missing required columns: {missing_columns}")

    duplicate_mask = data.duplicated(ORDER_COLUMNS, keep=False)
    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, ORDER_COLUMNS].head().to_dict("records")
        raise ValueError(f"Composite trial keys are not unique. Examples: {examples}")

    prior_means = np.sort(data["prior_mean"].dropna().unique())
    if len(prior_means) != 1 or not np.isclose(prior_means[0], PRIOR_MEAN_DEGREES):
        raise ValueError(f"Expected one fixed prior mean at 225 degrees; found {prior_means.tolist()}")

    data = data.sort_values(ORDER_COLUMNS, kind="mergesort").reset_index(drop=True)
    data["block_id"] = (
        "S" + data["subject_id"].astype(str)
        + "_session" + data["session_id"].astype(str)
        + "_run" + data["run_id"].astype(str)
    )

    x = pd.to_numeric(data["estimate_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data["estimate_y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    response_angle = np.full(len(data), np.nan, dtype=float)
    response_angle[valid] = wrap_degrees_unsigned(np.degrees(np.arctan2(y[valid], x[valid])))

    direction = data["motion_direction"].to_numpy(dtype=float)
    data["response_valid"] = valid
    data["response_angle"] = response_angle
    data["stimulus_from_prior"] = circular_difference_degrees(direction, PRIOR_MEAN_DEGREES)
    data["response_from_prior"] = circular_difference_degrees(response_angle, PRIOR_MEAN_DEGREES)
    data["response_error"] = circular_difference_degrees(response_angle, direction)

    grouped = data.groupby("block_id", sort=False)
    data["block_trial_position"] = grouped.cumcount()
    data["block_n_trials"] = grouped["trial_index"].transform("size")
    data["block_phase"] = _block_phase(data["block_trial_position"], data["block_n_trials"])
    data["subject_trial_position"] = data.groupby("subject_id", sort=False).cumcount()

    audit = {
        "csv_path": str(csv_path),
        "rows": int(len(data)),
        "subjects": int(data["subject_id"].nunique()),
        "blocks": int(data["block_id"].nunique()),
        "valid_responses": int(data["response_valid"].sum()),
        "missing_responses": int((~data["response_valid"]).sum()),
        "coherences": sorted(data["motion_coherence"].dropna().unique().tolist()),
        "prior_widths": sorted(data["prior_std"].dropna().unique().tolist()),
        "fixed_prior_mean": float(prior_means[0]),
    }
    return data, audit


def pilot_selection_table(
    data: pd.DataFrame,
    conflict_min_degrees: float = 60.0,
    attraction_window_degrees: float = 20.0,
) -> pd.DataFrame:
    """Rank contrasting pilots using an observable, model-free dual-attraction score.

    The score is the geometric mean of response mass close to the prior and
    response mass close to the stimulus on low-coherence, high-conflict trials.
    It selects pilots only; it is not a bimodality test and never enters fitting.
    """

    lowest_coherence = float(data["motion_coherence"].min())
    conflict = np.abs(data["stimulus_from_prior"].to_numpy(dtype=float))
    subset = data.loc[
        data["response_valid"]
        & np.isclose(data["motion_coherence"], lowest_coherence)
        & (conflict >= conflict_min_degrees)
    ].copy()
    if subset.empty:
        raise ValueError("No valid low-coherence, high-conflict trials are available for pilot selection.")

    subset["near_prior"] = np.abs(subset["response_from_prior"]) <= attraction_window_degrees
    subset["near_stimulus"] = np.abs(subset["response_error"]) <= attraction_window_degrees
    table = (
        subset.groupby("subject_id", sort=True)
        .agg(
            eligible_trials=("response_valid", "size"),
            prior_mass=("near_prior", "mean"),
            stimulus_mass=("near_stimulus", "mean"),
        )
        .reset_index()
    )
    table["dual_attraction_score"] = np.sqrt(table["prior_mass"] * table["stimulus_mass"])
    table["lowest_coherence"] = lowest_coherence
    table["conflict_threshold_degrees"] = conflict_min_degrees
    table["attraction_window_degrees"] = attraction_window_degrees
    return table.sort_values("dual_attraction_score", ascending=False, kind="mergesort").reset_index(drop=True)
