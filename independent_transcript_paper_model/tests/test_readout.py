import numpy as np

from hierarchical_confidence.readout import tie_aware_map_support


def _weights(probabilities: np.ndarray) -> np.ndarray:
    tied, counts = tie_aware_map_support(np.log(probabilities))
    return tied / counts[..., None]


def test_unique_map_receives_all_readout_mass() -> None:
    weights = _weights(np.array([[0.1, 0.7, 0.2]]))
    np.testing.assert_allclose(weights, [[0.0, 1.0, 0.0]])


def test_two_equal_separated_modes_split_mass_equally() -> None:
    weights = _weights(np.array([[0.45, 0.05, 0.05, 0.45]]))
    np.testing.assert_allclose(weights, [[0.5, 0.0, 0.0, 0.5]])


def test_three_tied_modes_each_receive_one_third() -> None:
    weights = _weights(np.array([[0.3, 0.3, 0.1, 0.3]]))
    np.testing.assert_allclose(weights, [[1 / 3, 1 / 3, 0.0, 1 / 3]])


def test_six_decimal_rounding_removes_numerical_tie_breaking() -> None:
    probabilities = np.array([[0.5000004, 0.4999996]])
    weights = _weights(probabilities)
    np.testing.assert_allclose(weights, [[0.5, 0.5]])


def test_circular_endpoint_bins_can_both_be_map_modes() -> None:
    weights = _weights(np.array([[0.4, 0.1, 0.1, 0.4]]))
    assert weights[0, 0] == 0.5
    assert weights[0, -1] == 0.5
