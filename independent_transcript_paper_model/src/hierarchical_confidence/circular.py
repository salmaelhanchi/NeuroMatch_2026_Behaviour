"""Small, explicit utilities for directions measured in degrees."""

from __future__ import annotations

import numpy as np


def wrap_degrees_signed(values: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [-180, 180)."""

    values = np.asarray(values, dtype=float)
    return (values + 180.0) % 360.0 - 180.0


def wrap_degrees_unsigned(values: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [0, 360)."""

    return np.asarray(values, dtype=float) % 360.0


def circular_difference_degrees(
    angle: np.ndarray | float,
    reference: np.ndarray | float,
) -> np.ndarray:
    """Return the signed shortest difference angle - reference."""

    return wrap_degrees_signed(np.asarray(angle) - np.asarray(reference))


def angle_to_bin(angle_degrees: np.ndarray | float, n_angles: int) -> np.ndarray:
    """Map angles to the nearest center on an evenly spaced circular grid."""

    step = 360.0 / n_angles
    wrapped = wrap_degrees_unsigned(angle_degrees)
    return np.floor((wrapped + step / 2.0) / step).astype(int) % n_angles
