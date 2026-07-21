from __future__ import annotations

import numpy as np

from standardized_hb import (
    GridSpec,
    PredictionResult,
    last_sessions_holdout,
    observed_log_scores,
    paired_block_bootstrap_difference,
    predictive_score_record,
    with_response_mask,
)

from test_models import make_subject


def test_final_session_holdout_preserves_full_sequence() -> None:
    grid = GridSpec(n_angles=72, n_positive_kappa=4)
    subject = make_subject(grid)
    split = last_sessions_holdout(subject, n_sessions=1)
    training_subject = with_response_mask(subject, split.train_mask)

    assert split.heldout_sessions == (2,)
    np.testing.assert_array_equal(split.train_mask, [True, True, True, True, False, False])
    assert training_subject.n_trials == subject.n_trials
    assert training_subject.n_valid_responses == 4
    assert not np.any(training_subject.response_valid & split.test_mask)


def test_predictive_scores_and_paired_block_bootstrap() -> None:
    grid = GridSpec(n_angles=72, n_positive_kappa=4)
    subject = make_subject(grid)
    prediction = PredictionResult(
        response_pmfs=np.full((subject.n_trials, grid.n_angles), 1.0 / grid.n_angles),
        state=np.zeros(subject.n_trials),
        state_label="test state",
    )
    scores = observed_log_scores(subject, prediction, np.ones(subject.n_trials, dtype=bool))
    record = predictive_score_record("uniform", scores, grid.n_angles)

    np.testing.assert_allclose(scores, -np.log(grid.n_angles))
    np.testing.assert_allclose(record["bits_per_trial_over_uniform"], 0.0)

    comparison = paired_block_bootstrap_difference(
        np.array([-1.0, -1.1, -0.9, -1.2]),
        np.array([-1.5, -1.4, -1.6, -1.3]),
        np.array([[1, 1], [1, 1], [2, 1], [2, 1]]),
        n_draws=500,
    )
    assert comparison["mean_difference"] > 0.0
    assert comparison["ci_2.5%"] > 0.0
    assert comparison["blocks"] == 2
