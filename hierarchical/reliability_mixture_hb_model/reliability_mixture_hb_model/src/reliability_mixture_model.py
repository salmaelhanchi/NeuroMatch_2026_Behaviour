"""
reliability_mixture_model.py
=============================

The reliability-mixture hierarchical Bayesian observer: on each trial, the
percept is a genuine discrete mixture (never a multiplicative blend) between
a prior-driven component (centered on the fixed prior mean, 225 degrees) and
a likelihood-driven component (centered on that trial's true motion
direction). The mixture weight, `prior_reliance`, is the hyper-prior: it
updates trial-by-trial via a delta rule against a 5-trial smoothed window of
true feedback direction, with learning rate `alpha`.

This is deliberately NOT the same construction as
`hierarchical/observers/models/hb_integration.py` in this repo, which learns
prior concentration (kappa) trial-by-trial and holds its mixture weight
(alpha) fixed. See this folder's README.md for the full comparison and why
each choice was made.

Free parameters per subject (10 total):
    k_llh[coherence]     -- 3 values, sensory likelihood concentration
    k_prior[prior_width] -- 4 values, prior concentration per block width
    alpha                -- prior_reliance learning rate
    k_motor              -- motor/response noise concentration
    lapse_rate            -- lapse probability

Usage:
    from reliability_mixture_model import load_and_prepare_data, hb_mixture_nll

    trial_data = load_and_prepare_data("data01_direction4priors.csv")
    subject_4 = trial_data[trial_data.subject == 4].sort_values(['run_id', 'trial'])
    nll = hb_mixture_nll(theta, subject_4, cohs_u=[0.06, 0.12, 0.24], ps_u=[80, 40, 20, 10])
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy.special import i0e

DEG = np.arange(1, 361)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and build the trial-level dataset this model expects.

    Handles: circular estimate_deg from x/y coordinates, circular error,
    block_id = (subject_id, run_id), and the same_session_prev flag needed
    for the window-reset rule (reliance resets only at session boundaries,
    not every block, so carryover isn't assumed away by construction).
    """
    raw = pd.read_csv(csv_path)
    data = raw.copy()

    data['estimate_deg'] = np.degrees(np.arctan2(data.estimate_y, data.estimate_x)) % 360
    data.loc[data.estimate_deg == 0, 'estimate_deg'] = 360
    data = data.dropna(subset=['estimate_deg', 'motion_direction']).copy()

    data['error_deg'] = ((data.estimate_deg - data.motion_direction + 180) % 360) - 180
    data['block_id'] = data['subject_id'].astype(str) + '_' + data['run_id'].astype(str)

    trial_data = data.rename(columns={
        'subject_id': 'subject',
        'trial_index': 'trial',
        'prior_std': 'prior_width',
        'motion_coherence': 'coherence',
        'motion_direction': 'true_direction',
    })[['subject', 'block_id', 'trial', 'prior_width', 'coherence',
        'true_direction', 'estimate_deg', 'error_deg',
        'session_id', 'experiment_id']]

    block_context = (
        data.groupby(['subject_id', 'run_id'], as_index=False)
        .agg(session_id=('session_id', 'first'))
        .sort_values(['subject_id', 'run_id'])
        .reset_index(drop=True)
    )
    block_context['prev_session_id'] = block_context.groupby('subject_id')['session_id'].shift(1)
    block_context['same_session_prev'] = (
        block_context['prev_session_id'].notna()
        & (block_context['session_id'] == block_context['prev_session_id'])
    )
    block_context['block_id'] = block_context['subject_id'].astype(str) + '_' + block_context['run_id'].astype(str)

    trial_data = trial_data.merge(
        block_context[['block_id', 'run_id', 'same_session_prev']], on='block_id', how='left'
    )
    return trial_data


def vm_pdf(support_deg, mu_deg, k, norm=True):
    """Von Mises PMF over a discrete circular support (the circular analogue
    of a Gaussian; k is the concentration, higher = tighter)."""
    mu = np.atleast_1d(np.asarray(mu_deg, float))
    x = np.deg2rad(np.asarray(support_deg, float))[:, None]
    u = np.deg2rad(mu)[None, :]
    k = float(k)
    if np.isinf(k) or k > 1e300:
        out = np.zeros((len(support_deg), len(mu)))
        for j, mm in enumerate(mu):
            out[np.argmin(np.abs(np.asarray(support_deg) - mm)), j] = 1.0
    else:
        out = np.exp(k * np.cos(x - u) - k) / (2 * np.pi * i0e(k))
    if norm:
        out = out / out.sum(0, keepdims=True)
    return out


def wrap_signed_deg(diff_deg):
    return ((diff_deg + 180) % 360) - 180


def circular_mean_deg(angles_deg):
    angles_rad = np.deg2rad(np.asarray(angles_deg))
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    return float(np.degrees(np.arctan2(mean_y, mean_x)) % 360)


def prior_agreement(measurement_deg, prior_mean_deg, k_prior):
    """How well recent (smoothed) feedback agrees with the fixed prior mean.
    NOTE: this reuses k_prior, the same value that shapes the percept mixture
    for that condition -- see this folder's README for the identifiability
    consequence this has when reliance collapses toward zero."""
    delta_rad = np.deg2rad(wrap_signed_deg(measurement_deg - prior_mean_deg))
    return float(np.exp(k_prior * (np.cos(delta_rad) - 1.0)))


def trial_percept_distribution(prior_mean_deg, measurement_deg, k_prior, k_llh, prior_reliance):
    """The discrete either/or mixture: prior_reliance weight on the
    prior-centered component, (1 - prior_reliance) on the likelihood-centered
    component. This -- not a multiplicative blend -- is what lets the model
    reproduce genuinely bimodal response distributions."""
    prior_component = vm_pdf(DEG, prior_mean_deg, k_prior)[:, 0]
    llh_component = vm_pdf(DEG, measurement_deg, k_llh)[:, 0]
    mixture = prior_reliance * prior_component + (1 - prior_reliance) * llh_component
    return mixture / mixture.sum()


def circ_convolve_vec(p, kernel):
    return np.real(np.fft.ifft(np.fft.fft(p) * np.fft.fft(kernel)))


def hb_mixture_trial_loop(d: pd.DataFrame, params: dict, prior_mean_deg: float = 225.0,
                           window_size: int = 5):
    """Sequential forward pass over one subject's trials (must be pre-sorted
    by run_id, trial). Returns (p_obs, reliance_trace): the model's predicted
    probability of the actually-observed response on each trial, and the
    prior_reliance trajectory.

    params: dict with keys k_llh (dict by coherence), k_prior (dict by
    prior_width), alpha, k_motor, lapse_rate.
    """
    k_llh, k_prior = params['k_llh'], params['k_prior']
    alpha, k_motor, lapse_rate = params['alpha'], params['k_motor'], params['lapse_rate']

    reliance = 0.5
    window_buf = deque(maxlen=window_size)
    prev_block_id = None
    motor_kernel = vm_pdf(DEG, 360.0, k_motor)[:, 0]

    n = len(d)
    p_obs = np.empty(n)
    reliance_trace = np.empty(n)

    block_ids = d['block_id'].to_numpy()
    same_sess = d['same_session_prev'].to_numpy()
    true_dirs = d['true_direction'].to_numpy(dtype=float)
    coherences = d['coherence'].to_numpy()
    prior_widths = d['prior_width'].to_numpy()
    est_degs = d['estimate_deg'].to_numpy(dtype=float)

    for i in range(n):
        is_new_block = block_ids[i] != prev_block_id
        if is_new_block and not bool(same_sess[i]):
            window_buf.clear()  # reset only at session boundaries, not every block

        reliance_trace[i] = reliance
        kp = k_prior[prior_widths[i]]
        kl = k_llh[coherences[i]]

        mixture = trial_percept_distribution(prior_mean_deg, true_dirs[i], kp, kl, reliance)
        with_motor = circ_convolve_vec(mixture, motor_kernel)
        with_motor = np.clip(with_motor, 0, None)
        with_motor = with_motor / with_motor.sum()
        final = (1 - lapse_rate) * with_motor + lapse_rate * (1.0 / 360)

        e_idx = int(round(est_degs[i])) % 360
        if e_idx == 0:
            e_idx = 360
        p_obs[i] = final[e_idx - 1]

        # feedback is the true motion_direction, not the subject's own estimate
        # (avoids circularity: using the subject's estimate would let reliance
        # partly confirm itself through its own influence on the response)
        window_buf.append(true_dirs[i])
        smoothed = circular_mean_deg(list(window_buf))
        agreement = prior_agreement(smoothed, prior_mean_deg, kp)
        reliance = float(np.clip(reliance + alpha * (agreement - reliance), 1e-4, 1 - 1e-4))

        prev_block_id = block_ids[i]

    return p_obs, reliance_trace


def hb_mixture_nll(theta, d, cohs_u, ps_u, window_size=5):
    """Negative log-likelihood for scipy.optimize. theta is a flat array of
    10 transformed parameters: log(k_llh) x3, log(k_prior) x4, logit(alpha),
    log(k_motor), logit-scaled lapse_rate (see any of this folder's notebooks
    for the exact encode/decode convention used during fitting)."""
    i = 0
    k_llh = {c: float(np.exp(theta[i + j])) for j, c in enumerate(cohs_u)}; i += len(cohs_u)
    k_prior = {p: float(np.exp(theta[i + j])) for j, p in enumerate(ps_u)}; i += len(ps_u)
    alpha = 1.0 / (1.0 + np.exp(-theta[i])); i += 1
    k_motor = float(np.exp(theta[i])); i += 1
    lapse_rate = 0.3 / (1.0 + np.exp(-theta[i]))
    params = dict(k_llh=k_llh, k_prior=k_prior, alpha=alpha, k_motor=k_motor, lapse_rate=lapse_rate)
    p_obs, _ = hb_mixture_trial_loop(d, params, window_size=window_size)
    p_obs = np.clip(p_obs, 1e-320, None)
    return -np.log(p_obs).sum()
