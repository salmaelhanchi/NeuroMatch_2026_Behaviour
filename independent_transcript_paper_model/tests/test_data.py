from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_confidence.data import load_and_prepare_data, pilot_selection_table


def _raw_rows() -> pd.DataFrame:
    rows = []
    responses = [225.0, 5.0, np.nan, 225.0, 5.0, 225.0]
    for subject_id in (1, 2):
        for trial_index, angle in enumerate(responses, start=1):
            radians = np.deg2rad(angle) if np.isfinite(angle) else np.nan
            rows.append(
                {
                    "trial_index": trial_index,
                    "motion_direction": 5.0 if trial_index % 2 == 0 else 225.0,
                    "motion_coherence": 0.06,
                    "estimate_x": np.cos(radians) if np.isfinite(radians) else np.nan,
                    "estimate_y": np.sin(radians) if np.isfinite(radians) else np.nan,
                    "prior_std": 20.0,
                    "prior_mean": 225.0,
                    "subject_id": subject_id,
                    "session_id": 1,
                    "run_id": 1,
                }
            )
    return pd.DataFrame(rows)


def test_load_keeps_missing_response_in_sequence(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _raw_rows().sample(frac=1.0, random_state=4).to_csv(path, index=False)
    data, audit = load_and_prepare_data(path)
    assert audit["rows"] == 12
    assert audit["missing_responses"] == 2
    assert data.groupby("subject_id")["trial_index"].apply(list).tolist() == [list(range(1, 7))] * 2
    assert data.loc[~data["response_valid"], "response_angle"].isna().all()


def test_pilot_score_is_selection_only_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _raw_rows().to_csv(path, index=False)
    data, _ = load_and_prepare_data(path)
    table = pilot_selection_table(data, conflict_min_degrees=60.0)
    assert set(table["subject_id"]) == {1, 2}
    assert np.isfinite(table["dual_attraction_score"]).all()
