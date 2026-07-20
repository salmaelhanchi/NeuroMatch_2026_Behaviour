import numpy as np

from hierarchical_confidence.model import (
    GridSpec,
    HierarchicalObserver,
    ModelParameters,
    PreparedSubject,
)


def _subject(grid: GridSpec) -> PreparedSubject:
    directions = np.array([225.0, 225.0, 5.0, 5.0])
    return PreparedSubject(
        subject_id=99,
        directions=directions,
        coherence_indices=np.array([0, 1, 0, 1]),
        coherence_values=np.array([0.06, 0.24]),
        response_bins=np.array([15, 15, 0, -1]) % grid.n_angles,
        response_valid=np.array([True, True, True, False]),
        prior_std=np.array([10.0, 10.0, 80.0, 80.0]),
        block_ids=np.array(["a", "a", "b", "b"]),
        trial_indices=np.array([1, 2, 1, 2]),
    )


def test_grid_contains_uniform_plus_positive_kappas() -> None:
    grid = GridSpec(n_angles=24, n_positive_kappa=5)
    assert len(grid.kappa_values) == 6
    assert grid.kappa_values[0] == 0.0
    assert np.all(np.diff(grid.kappa_values) > 0.0)


def test_hidden_state_updates_across_block_boundary() -> None:
    grid = GridSpec(n_angles=24, n_positive_kappa=5)
    observer = HierarchicalObserver(_subject(grid), grid)
    trajectory, final_state = observer.confidence_trajectory(0.95)
    np.testing.assert_allclose(trajectory.sum(axis=1), 1.0)
    np.testing.assert_allclose(final_state.sum(), 1.0)
    assert not np.allclose(trajectory[2], np.full(6, 1.0 / 6.0))


def test_predictions_normalize_and_match_nll_scoring() -> None:
    grid = GridSpec(n_angles=24, n_positive_kappa=5)
    subject = _subject(grid)
    observer = HierarchicalObserver(subject, grid, batch_size=2)
    parameters = ModelParameters(
        rho=0.9,
        sensory_kappas=np.array([2.0, 8.0]),
        motor_kappa=15.0,
        lapse=0.02,
    )
    predictions = observer.predict_response_pmfs(parameters)
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-10)
    valid = subject.response_valid
    direct_nll = -np.log(predictions[np.flatnonzero(valid), subject.response_bins[valid]]).sum()
    np.testing.assert_allclose(observer.negative_log_likelihood(parameters), direct_nll, atol=1e-10)
