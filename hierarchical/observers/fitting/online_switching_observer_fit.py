"""
online_switching_observer_fit.py
================================

Fit the online-learning Switching observer (:class:`OnlineHierarchicalObserver`)
to one subject, and expose it with the same CLI as the other per-model fitters:

    python -m observers.fitting.online_switching_observer_fit human 1

This is a THIN wrapper. The optimisation itself is not reimplemented here -- it
delegates to the already-tested online path
(``observers.fitting.online_recovery.fit_online``). This module only:
  * loads a subject,
  * calls that fitter,
  * reports / returns NLL, AIC, BIC.

The online observer is the switch family's learning member: it starts from the
paper's Switching observer and lets the prior's strength be learned online, with
a forgetting knob ``lam`` (setting ``lam=0`` recovers the static switch, which is
exactly how ``online_recovery.fit_static`` fits the paper's Switch model).
"""
from __future__ import annotations

import sys
import numpy as np

from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.fitting.online_recovery import fit_online, conv_info as _conv_info
from observers.helpers.dataset import load_subject_design
from observers.helpers.paths import DATA_CSV

N_PARAMS = 6  # 3 k_like + k_motor + p_random + lam


def fit(data, maxiter=500):
    """Fit the online switching observer to one subject's trial data.

    Returns (observer, nll, aic, bic).
    """
    obs, nll, aic = fit_online(data, maxiter=maxiter)
    n = len(np.asarray(data["estimates"]))
    bic = N_PARAMS * np.log(n) + 2 * nll
    return obs, float(nll), float(aic), float(bic)


def _load_subject(sid):
    """Load one subject as the dict the fitter expects (mirrors the convention
    in switching_observer_fit / hb_rachel_fit: estimate_dir -> 'estimates')."""
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
                         "k_like": obs.k_like, "k_motor": obs.k_motor,
                         "p_random": obs.p_random, "lam": obs.lam}
        print(f"subject {sid}: n={n}  NLL={nll:.1f}  AIC={aic:.1f}  BIC={bic:.1f}  lam={obs.lam:.3f}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "human":
        ids = [int(a) for a in sys.argv[2:]] or [1]
        human(ids)
    else:
        print(__doc__)
        print("usage: python -m observers.fitting.online_switching_observer_fit human <subject_id> [...]")
