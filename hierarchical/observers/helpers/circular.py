"""
circular.py
===========

Low-level circular-statistics utilities used by the Switching observer.

These are the "specialty" numerical routines that would clutter the main
model file (``switching_observer.py``) if inlined.  They are faithful ports
of the MATLAB helpers written by Steeve Laquitaine for

    Laquitaine & Gardner (2018), "A Switching Observer for Human Perceptual
    Estimation", Neuron 97(2), 462-474.

MATLAB sources ported here:
    - vmPdfs.m               -> von_mises_pdfs   (paper Eq. 1)
    - SLde2r.m / SLra2d.m    -> deg2rad_signed / rad2deg_360
    - SLcircConv.m           -> circular_convolution
    - SLcircWeightedMeanStd.m-> circular_weighted_mean_std
    - (paper Eq. 2)          -> von_mises_std

Everything works on the discretised circular direction space
``theta = 1, 2, ..., 360`` degrees, exactly as in the original code.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0e  # exponentially-scaled modified Bessel I0(k)*exp(-k)

# The whole model lives on a 1-degree grid of directions 1..360 deg.
DIRECTION_SPACE = np.arange(1, 361)  # 1..360 (matches MATLAB diSpace)


# ---------------------------------------------------------------------------
# Degree <-> radian conversions (SLde2r.m, SLra2d.m)
# ---------------------------------------------------------------------------
def deg2rad_signed(angle_deg: np.ndarray) -> np.ndarray:
    """Convert degrees to *signed* radians in (-pi, pi].

    Port of SLde2r(ang, 1).  Angles > 180 deg are mapped to their negative
    equivalent so that, e.g., 270 deg -> -pi/2.  The signed convention keeps
    the von Mises exponent ``cos(theta - mu)`` numerically well-behaved.
    """
    angle_deg = np.asarray(angle_deg, dtype=float)
    radians = (angle_deg / 360.0) * 2.0 * np.pi
    radians = np.where(angle_deg > 180.0,
                       (angle_deg - 360.0) * (2.0 * np.pi / 360.0),
                       radians)
    return radians


def rad2deg_360(theta_rad: np.ndarray) -> np.ndarray:
    """Convert radians to degrees on the 1..360 grid (SLra2d.m).

    Output angles lie in [1, 360]; 0 is wrapped to 360 because the model
    indexes directions from 1.
    """
    deg = np.asarray(theta_rad, dtype=float) / (2.0 * np.pi) * 360.0
    deg = np.mod(deg, 360.0)
    deg[deg < 0] += 360.0
    deg[deg == 0] = 360.0
    return deg


# ---------------------------------------------------------------------------
# Von Mises probability densities  (vmPdfs.m ; paper Eq. 1)
# ---------------------------------------------------------------------------
def von_mises_pdfs(x: np.ndarray,
                   means_deg,
                   kappas,
                   normalize: bool = True) -> np.ndarray:
    """Von Mises densities on the circular grid ``x`` (paper Eq. 1).

        V(theta; mu, k) = exp(k*cos(theta - mu) - k) / (2*pi * I0(k))

    Parameters
    ----------
    x : array of directions in degrees (the support, typically 1..360).
    means_deg : scalar or 1-D array of von Mises means mu (degrees).
    kappas : scalar or 1-D array of concentration parameters k
             (one per mean).  Larger k -> narrower distribution.
    normalize : if True, each column is scaled to sum to 1 (a proper pmf on
                the discrete grid).

    Returns
    -------
    (len(x), n_means) array; column j is V(x; means[j], kappas[j]).

    Numerical note (from the original vmPdfs.m)
    -------------------------------------------
    The naive form ``exp(k cos)/ (2 pi I0(k))`` overflows for k >~ 700 because
    both numerator and I0(k) explode.  We use the mathematically identical but
    stable form ``exp(k cos - k) / (2 pi * I0e(k))`` where
    ``I0e(k) = I0(k) * exp(-k)``.  As ``k -> inf`` the von Mises collapses to a
    delta at the mean, which we emit explicitly.
    """
    x = np.asarray(x, dtype=float).ravel()
    means_deg = np.atleast_1d(np.asarray(means_deg, dtype=float)).ravel()
    kappas = np.atleast_1d(np.asarray(kappas, dtype=float)).ravel()

    if kappas.size == 1 and means_deg.size > 1:
        kappas = np.repeat(kappas, means_deg.size)

    x_rad = deg2rad_signed(x)                     # (N,)
    mu_rad = deg2rad_signed(means_deg)            # (M,)

    out = np.zeros((x.size, means_deg.size), dtype=float)

    for j in range(means_deg.size):
        k = kappas[j]
        if k > 1e300:
            # Delta density: all mass on the grid point nearest the mean.
            col = np.zeros(x.size)
            col[np.argmin(np.abs(x - means_deg[j]))] = 1.0
            out[:, j] = col
            continue

        # Stable von Mises (unnormalised-by-shape but scaled by I0e).
        col = np.exp(k * np.cos(x_rad - mu_rad[j]) - k) / (2.0 * np.pi * i0e(k))
        if normalize:
            s = col.sum()
            if s > 0:
                col = col / s
        out[:, j] = col

    return out


# ---------------------------------------------------------------------------
# k -> circular standard deviation  (paper Eq. 2)
# ---------------------------------------------------------------------------
def von_mises_std(kappa: float, mean_deg: float = 225.0) -> float:
    """Circular standard deviation (degrees) of a von Mises with strength k.

    Implements paper Eq. 2 directly:

        s = sqrt( sum_theta  p(theta) * f(theta, mu)^2 )

    where ``p`` is the (normalised) von Mises density on the 1..360 grid and
    ``f(theta, mu)`` is the *signed* angular distance between each direction
    and the mean.  This is how the paper reports prior widths (e.g. an 80 deg
    prior) from the fitted concentration parameters k.
    """
    p = von_mises_pdfs(DIRECTION_SPACE, mean_deg, kappa, normalize=True).ravel()
    # signed angular distance f(theta, mu) in degrees, in (-180, 180]
    f = (DIRECTION_SPACE - mean_deg + 180.0) % 360.0 - 180.0
    return float(np.sqrt(np.sum(p * f ** 2)))


# ---------------------------------------------------------------------------
# Circular convolution  (SLcircConv.m)
# ---------------------------------------------------------------------------
def circular_convolution(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Column-wise circular convolution via the FFT (SLcircConv.m).

        cconv = ifft( fft(v1) .* fft(v2) )

    Used to smear the switching percept distribution with motor noise
    (a von Mises centred on 0).  Because convolution is distributive, the
    lapse/uniform term can be mixed in before convolving.  The tiny imaginary
    residue from the FFT round-trip is discarded.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    conv = np.fft.ifft(np.fft.fft(v1, axis=0) * np.fft.fft(v2, axis=0), axis=0)
    return np.real(conv)


# ---------------------------------------------------------------------------
# Circular weighted mean & std of a distribution over directions
# (SLcircWeightedMeanStd.m) -- used to summarise predicted distributions.
# ---------------------------------------------------------------------------
def circular_weighted_mean_std(directions_deg: np.ndarray,
                               weights: np.ndarray):
    """Circular mean and standard deviation (degrees) of a weighted set of
    directions -- the vector-average way the paper computes predicted
    estimate mean and variability (Figures 3B, 5C).

    Parameters
    ----------
    directions_deg : directions (degrees), e.g. 1..360.
    weights : non-negative weights / probabilities (need not sum to 1).

    Returns
    -------
    (mean_deg, std_deg)
    """
    directions_deg = np.asarray(directions_deg, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()

    ang = np.deg2rad(directions_deg)
    x = np.sum(w * np.cos(ang))
    y = np.sum(w * np.sin(ang))
    mean_deg = np.rad2deg(np.arctan2(y, x)) % 360.0
    if mean_deg == 0:
        mean_deg = 360.0

    # Circular std around the mean, matching SLcircWeightedMeanStd: signed
    # deviations folded into (-180, 180], weighted second moment.
    dev = (directions_deg - mean_deg + 180.0) % 360.0 - 180.0
    var = np.sum(w * dev ** 2)
    return mean_deg, float(np.sqrt(var))