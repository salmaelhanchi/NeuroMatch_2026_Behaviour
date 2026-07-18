"""
switching_observer_fit.py
=========================

Fit the paper's static Switching observer to one subject, and expose it with the
same CLI as the other per-model fitters:

    python -m observers.fitting.switching_observer_fit human 1

This is a THIN wrapper. The static-switch optimisation itself is not
reimplemented here -- it delegates to the already-tested multi-start static path
(`observers.fitting.online_recovery.fit_static`, given the fair multi-start
treatment in `observers.fitting.fair_refit`). This module only:
  * loads a subject,
  * calls that fitter,
  * rebuilds a proper `SwitchingObserver` from the fitted parameters, and
  * reports / returns NLL, AIC, BIC.

For the full fair AIC table across models, use `python -m
observers.fitting.fair_refit table` instead.
"""
from __future__ import annotations

import sys
import numpy as np

from observers.models.switching_observer import SwitchingObserver
from observers.fitting.online_recovery import fit_static, unpack_static
from observers.helpers.dataset import load_subject_design
from observers.helpers.paths import DATA_CSV

N_PARAMS = 9  # 3 k_like + 4 k_prior + k_motor + p_random


def observer_from_theta(theta) -> SwitchingObserver:
    """Rebuild a SwitchingObserver from the packed static-fit parameter vector.

    `unpack_static` returns an OnlineHierarchicalObserver (fast fitting path) plus
    the per-SD prior concentrations; here we map those onto the canonical
    SwitchingObserver, whose prior keys are string SD labels.
    """
    onl, k_prior_by_sd = unpack_static(theta)
    return SwitchingObserver(
        k_like=dict(onl.k_like),
        k_prior={str(int(sd)): float(k) for sd, k in k_prior_by_sd.items()},
        p_random=float(onl.p_random),
        k_motor=float(onl.k_motor),
    )


def fit(data, maxiter=500):
    """Fit the static switch to one subject's trial data.

    Returns (observer, nll, aic, bic).
    """
    theta, nll, aic = fit_static(data, maxiter=maxiter)
    obs = observer_from_theta(theta)
    n = len(np.asarray(data["estimates"]))
    bic = N_PARAMS * np.log(n) + 2 * nll
    return obs, float(nll), float(aic), float(bic)


def _load_subject(sid):
    """Load one subject as the dict the static fitter expects (mirrors the
    convention in hb_integration_fit / fair_refit): estimate_dir -> 'estimates'.
    """
    d = load_subject_design(str(DATA_CSV), int(sid))
    return dict(
        motion_direction=d.motion_direction.values.astype(int),
        motion_coherence=d.motion_coherence.values.astype(float),
        prior_std=d.prior_std.values.astype(int),
        estimates=d.estimate_dir.values.astype(int),
    )


def human(subject_ids, maxiter=500):
    out = {}
    for sid in subject_ids:
        data = _load_subject(int(sid))
        obs, nll, aic, bic = fit(data, maxiter=maxiter)
        n = len(np.asarray(data["estimates"]))
        out[int(sid)] = {"nll": nll, "aic": aic, "bic": bic, "n": n,
                         "k_like": obs.k_like, "k_prior": obs.k_prior,
                         "k_motor": obs.k_motor, "p_random": obs.p_random}
        print(f"subject {sid}: n={n}  NLL={nll:.1f}  AIC={aic:.1f}  BIC={bic:.1f}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "human":
        ids = [int(a) for a in sys.argv[2:]] or [1]
        human(ids)
    else:
        print(__doc__)
        print("usage: python -m observers.fitting.switching_observer_fit human <subject_id> [...]")
