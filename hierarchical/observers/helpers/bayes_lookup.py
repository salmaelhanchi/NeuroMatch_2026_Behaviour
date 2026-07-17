"""
bayes_lookup.py
===============

The Girshick Bayesian *MAP look-up table* -- the piece of machinery the
Switching observer reuses to turn a (likelihood strength, prior strength) pair
into a full distribution of read-out percepts for every displayed direction.

This is a faithful port of ``SLGirshickBayesLookupTable.m`` (Laquitaine &
Gardner, 2018), restricted to the case the Switching model actually uses:
a von Mises prior with the maximum-a-posteriori (MAP) read-out.

Why this lives in its own file
------------------------------
The switching *logic* (``switching_observer.py``) is short and maps almost
one-to-one onto the paper's equations.  The heavy lifting -- forming a
posterior for every possible internal measurement, extracting its mode(s),
handling ties, and converting P(percept | measurement) into
P(percept | displayed direction) -- is bookkeeping that would bury that logic.
It is isolated here behind a single function:

    girshick_map_lookup(k_like, prior_mode, k_prior, ...) -> (percepts, L)

where ``L[i, d]`` is  P(read-out percept_i | displayed direction d).

Background (Girshick et al., 2011; paper Eq. 3-4)
-------------------------------------------------
For an internal measurement ``m`` the observer forms the posterior

    posterior(theta | m)  ∝  likelihood(theta | m) * prior_cardinal(theta)
                                                   * prior_learnt(theta)

and reads out its mode (the MAP estimate, Eq. 4).  Sweeping ``m`` over all
360 possible measurements and weighting each by how often it occurs for a
given displayed direction ``d`` -- P(m | d), itself a von Mises centred on d
-- yields the percept distribution P(percept | d).
"""

from __future__ import annotations

import numpy as np

from observers.helpers.circular import (
    DIRECTION_SPACE,
    von_mises_pdfs,
    deg2rad_signed,
    rad2deg_360,
)

CARDINAL_MODES = np.array([90, 180, 270, 360])  # oblique/cardinal axes


def girshick_map_lookup(k_like: float,
                        prior_mode: float,
                        k_prior: float,
                        motion_dirs: np.ndarray,
                        k_cardinal: float = 0.0):
    """Percept distribution P(percept | displayed direction) via MAP read-out.

    Parameters
    ----------
    k_like : concentration of the sensory likelihood (``ke`` in the paper).
        Set ``k_like = 0`` to make the likelihood flat -- the read-out then
        collapses onto the prior mode (this is how the Switching model builds
        its "prior" component).
    prior_mode : mean of the learnt von Mises prior (degrees), 225 in the
        motion experiment.
    k_prior : concentration of the learnt prior (``kprior``).  Set
        ``k_prior = 0`` to make the prior flat -- the read-out then follows
        the sensory measurement (the Switching model's "evidence" component).
    motion_dirs : the displayed directions for which columns are computed
        (only these are filled to save work, as in the MATLAB code).
    k_cardinal : concentration of the fixed cardinal prior (0 = no cardinal
        prior; the winning "withoutCardinal" model uses 0).

    Returns
    -------
    L : (360, 360) array; ``L[percept-1, direction-1] =
        P(read-out percept | displayed direction)``.  Rows span the full
        percept grid 1..360 (percepts never produced get probability 0), so
        every call returns the *same* support and two read-outs can be added
        directly.  Columns for directions not in ``motion_dirs`` stay NaN,
        exactly like the original (only displayed directions are queried).
    """
    di = DIRECTION_SPACE.astype(float)          # posterior support 1..360
    m = DIRECTION_SPACE.astype(float)           # possible measurements 1..360
    n = di.size
    motion_dirs = np.atleast_1d(np.asarray(motion_dirs)).astype(int)

    # -- measurement distribution  P(m | d) ~ V(m; d, k_like) ----------------
    # Column j is the density of measurements when direction motion_dirs[j]
    # is shown.  With k_like = 0 this is uniform (all measurements equally
    # likely), which is what makes the prior-only read-out a clean delta.
    m_pdfs = von_mises_pdfs(m, motion_dirs, k_like, normalize=True)  # (360, D)

    # -- sensory likelihood  l[theta, m] = V(theta; m, k_like) ---------------
    # For each hypothesised measurement m (column) the likelihood over
    # directions theta (rows).
    like = von_mises_pdfs(di, di, k_like, normalize=True)            # (360,360)

    # -- learnt prior (same for every measurement column) --------------------
    prior_learnt = von_mises_pdfs(di, prior_mode, k_prior, normalize=True)
    prior_learnt = np.tile(prior_learnt, (1, n))                    # (360,360)

    # -- optional cardinal prior: mixture of 4 von Mises on the axes ---------
    if k_cardinal and not np.isnan(k_cardinal):
        card = von_mises_pdfs(di, CARDINAL_MODES, k_cardinal, normalize=True)
        prior_cardinal = 0.25 * card.sum(axis=1, keepdims=True)
        posterior = like * prior_cardinal * prior_learnt
    else:
        posterior = like * prior_learnt

    # normalise each measurement's posterior to a proper distribution
    with np.errstate(invalid="ignore", divide="ignore"):
        posterior = posterior / posterior.sum(axis=0, keepdims=True)
    # round to 6 decimals so near-ties in the mode are detected robustly
    # (matches the fix(...*1e6)/1e6 trick in the MATLAB source)
    posterior = np.round(posterior * 1e6) / 1e6

    # -- numerical rescue for very strong priors (Murray & Morgenstern 2010) -
    # When k_prior is huge the product likelihood*prior underflows to 0 for
    # every direction and normalisation yields NaN.  The posterior is then
    # replaced by its known closed-form von Mises (mode + concentration).
    _fix_degenerate_posteriors(posterior, di, m, prior_mode, k_like, k_prior)

    # -- MAP read-out: mode(s) of each measurement's posterior ---------------
    # A flat-ish posterior can have several equal maxima; each is an equally
    # likely percept for that measurement (handled by P(percept | m) below).
    percept_per_m = np.full((n, n), np.nan)     # rows = measurements
    for i in range(n):
        col = posterior[:, i]
        mx = col.max()
        modes = np.nonzero(col == mx)[0]        # indices into di (0-based)
        percept_per_m[i, :modes.size] = di[modes]

    max_modes = int(np.nanmax(np.sum(~np.isnan(percept_per_m), axis=1)))
    percept_per_m = percept_per_m[:, :max_modes]

    # -- P(percept | m): if a measurement yields q modes, each has prob 1/q --
    n_modes_each_m = np.sum(~np.isnan(percept_per_m), axis=1)          # (360,)
    p_percept_given_m = (1.0 / n_modes_each_m)[:, None]                # (360,1)
    p_percept_given_m = np.tile(p_percept_given_m, (max_modes, motion_dirs.size))

    # -- P(m | d) stacked to line up with the (measurement, mode) rows -------
    p_m_given_d = np.tile(m_pdfs, (max_modes, 1))                # (360*modes, D)

    # flatten measurements x modes into one long list of (percept, weight)
    percept_flat = percept_per_m.T.reshape(-1)   # column-major to match MATLAB
    # measurement index that produced each row (kept for parity; not needed)
    # combine: P(percept | d) = P(percept | m) * P(m | d)
    p_percept_given_d = p_percept_given_m * p_m_given_d          # (360*modes,D)

    # -- aggregate over measurements that produced the *same* percept --------
    valid = ~np.isnan(percept_flat)
    percept_valid = percept_flat[valid].astype(int)
    contrib = p_percept_given_d[valid, :]

    # Aggregate onto the FULL 1..360 percept grid.  Percepts never produced
    # keep probability 0 (the MATLAB "missing percepts" padding).  This makes
    # every read-out share one common 360-row support, so the Switching model
    # can add the evidence and prior read-outs element-wise.
    P = np.zeros((360, motion_dirs.size))
    for p in np.unique(percept_valid):
        P[p - 1, :] = contrib[percept_valid == p, :].sum(axis=0)

    # -- scatter into a full (360 percepts x 360 directions) matrix ----------
    L = np.full((360, 360), np.nan)
    L[:, motion_dirs - 1] = P

    # scale each displayed-direction column to a probability distribution
    with np.errstate(invalid="ignore", divide="ignore"):
        L = L / np.nansum(L, axis=0, keepdims=True)

    return L


def _fix_degenerate_posteriors(posterior, di, m, prior_mode, k_like, k_prior):
    """In-place Murray & Morgenstern (2010) closed form for NaN posteriors.

    For measurements whose posterior underflowed (all zeros -> NaN after
    normalising), rebuild the posterior analytically as a von Mises whose
    mode and concentration follow from combining a von Mises likelihood
    (strength k_like at the measurement) with a von Mises prior (strength
    k_prior at ``prior_mode``).
    """
    if k_like == 0 or k_prior == 0:
        return  # one factor is flat -> product never underflows to NaN here
    bad = np.nonzero(np.isnan(posterior[0, :]))[0]
    if bad.size == 0:
        return
    mi = m[bad]
    mi_rad = deg2rad_signed(mi)
    up_rad = deg2rad_signed(np.array([prior_mode]))[0]
    # posterior mode (combined direction)
    upo = mi_rad + np.arctan2(np.sin(up_rad - mi_rad),
                              (k_like / k_prior) + np.cos(up_rad - mi_rad))
    upo_deg = rad2deg_360(upo)
    # posterior concentration
    kpo = np.sqrt(k_like ** 2 + k_prior ** 2
                  + 2.0 * k_prior * k_like * np.cos(up_rad - mi_rad))
    rebuilt = von_mises_pdfs(di, upo_deg, np.full(upo_deg.shape, kpo)
                             if np.ndim(kpo) else kpo, normalize=True)
    posterior[:, bad] = rebuilt