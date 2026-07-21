from __future__ import annotations

import numpy as np

from standardized_hb import (
    GridSpec,
    IntegratedPriorObserver,
    PredictionResult,
    PreparedSubject,
    ReadoutAverageObserver,
    ReliabilityMixtureObserver,
    SwitchingObserver,
)
from standardized_hb.plotting import paper_bimodality_figure


def make_subject(grid: GridSpec) -> PreparedSubject:
    directions = np.array([225.0, 235.0, 85.0, 95.0, 225.0, 85.0])
    return PreparedSubject(
        subject_id=1,
        directions=directions,
        coherence_indices=np.array([0, 1, 0, 1, 0, 1]),
        coherence_values=np.array([0.06, 0.24]),
        response_bins=np.array([9, 9, 3, 4, 9, 3]),
        response_valid=np.array([True, True, True, True, False, True]),
        session_ids=np.array([1, 1, 1, 1, 2, 2]),
        run_ids=np.array([1, 1, 2, 2, 3, 3]),
        trial_indices=np.array([1, 2, 1, 2, 1, 2]),
        reset_before=np.array([True, False, False, False, True, False]),
    )


def test_all_models_share_normalized_response_contract() -> None:
    grid = GridSpec(n_angles=12, n_positive_kappa=4)
    subject = make_subject(grid)
    models = [
        ReadoutAverageObserver(subject, grid, batch_size=2),
        ReliabilityMixtureObserver(subject, grid, batch_size=2),
        IntegratedPriorObserver(subject, grid, batch_size=2),
        SwitchingObserver(subject, grid, batch_size=2),
    ]
    for model in models:
        raw = model.default_raw_parameters()
        prediction = model.predict(raw)
        assert prediction.response_pmfs.shape == (subject.n_trials, grid.n_angles)
        np.testing.assert_allclose(prediction.response_pmfs.sum(axis=1), 1.0, atol=1e-10)
        assert np.isfinite(model.negative_log_likelihood(raw))


def test_session_boundaries_reset_each_latent_state() -> None:
    grid = GridSpec(n_angles=12, n_positive_kappa=4)
    subject = make_subject(grid)

    readout = ReadoutAverageObserver(subject, grid)
    readout_params = readout.decode(readout.default_raw_parameters())
    belief = readout.belief_trajectory(readout_params["alpha"], readout_params["lambda"])
    np.testing.assert_allclose(belief[0], belief[4])

    integrated = IntegratedPriorObserver(subject, grid)
    confidence = integrated.confidence_trajectory(0.95)
    np.testing.assert_allclose(confidence[0], confidence[4])

    mixture = ReliabilityMixtureObserver(subject, grid)
    reliance = mixture.reliance_trajectory(prior_kappa=2.7, learning_rate=0.1)
    assert reliance[0] == reliance[4] == 0.5


def test_prepared_subject_has_no_prior_std_field() -> None:
    grid = GridSpec(n_angles=12, n_positive_kappa=4)
    subject = make_subject(grid)
    assert not hasattr(subject, "prior_std")


def test_switching_probability_follows_reliability_ratio() -> None:
    grid = GridSpec(n_angles=72, n_positive_kappa=4)
    subject = make_subject(grid)
    observer = SwitchingObserver(subject, grid)
    raw = observer.default_raw_parameters()
    parameters = observer.decode(raw)
    prediction = observer.predict(raw)

    expected = np.array(
        [
            parameters["prior_kappa"]
            / (
                parameters["prior_kappa"]
                + parameters["sensory_kappas"][coherence_index]
            )
            for coherence_index in subject.coherence_indices
        ]
    )
    np.testing.assert_allclose(prediction.state, expected)


def test_paper_diagnostic_uses_supplied_selector_and_wider_bins() -> None:
    grid = GridSpec(n_angles=72, n_positive_kappa=4)
    subject = make_subject(grid)
    selector = np.isclose(subject.directions, 85.0)
    prediction = PredictionResult(
        response_pmfs=np.full((subject.n_trials, grid.n_angles), 1.0 / grid.n_angles),
        state=np.zeros(subject.n_trials),
        state_label="test state",
    )

    figure = paper_bimodality_figure(
        subject,
        grid,
        {"readout_average": prediction},
        {85.0: selector},
        bin_width_degrees=15.0,
    )

    observed_line = figure.axes[0].lines[0]
    assert observed_line.get_xdata().size == 24
    np.testing.assert_allclose(observed_line.get_ydata().sum(), 1.0)
