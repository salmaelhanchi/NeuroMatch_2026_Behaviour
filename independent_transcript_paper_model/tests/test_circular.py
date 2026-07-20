import numpy as np

from hierarchical_confidence.circular import (
    angle_to_bin,
    circular_difference_degrees,
    wrap_degrees_unsigned,
)


def test_circular_boundary_is_short() -> None:
    assert circular_difference_degrees(1.0, 359.0) == 2.0
    assert circular_difference_degrees(359.0, 1.0) == -2.0


def test_unsigned_wrap_and_nearest_bins() -> None:
    np.testing.assert_allclose(wrap_degrees_unsigned([-1.0, 360.0, 361.0]), [359.0, 0.0, 1.0])
    np.testing.assert_array_equal(angle_to_bin([359.0, 0.0, 4.9, 5.1], 72), [0, 0, 1, 1])
