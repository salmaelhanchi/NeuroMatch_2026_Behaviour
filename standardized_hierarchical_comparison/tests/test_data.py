from __future__ import annotations

import pandas as pd

from standardized_hb import GridSpec, load_participant


def test_loader_drops_experimental_prior_width(tmp_path) -> None:
    path = tmp_path / "trials.csv"
    pd.DataFrame(
        {
            "trial_index": [1, 2],
            "motion_direction": [225.0, 85.0],
            "motion_coherence": [0.06, 0.24],
            "estimate_x": [-1.0, 1.0],
            "estimate_y": [-1.0, 1.0],
            "prior_std": [10.0, 80.0],
            "prior_mean": [225.0, 225.0],
            "subject_id": [1, 1],
            "session_id": [1, 1],
            "run_id": [1, 1],
        }
    ).to_csv(path, index=False)

    subject, audit = load_participant(path, 1, GridSpec(n_angles=12, n_positive_kappa=4))

    assert not hasattr(subject, "prior_std")
    assert audit["prior_std_used"] is False

