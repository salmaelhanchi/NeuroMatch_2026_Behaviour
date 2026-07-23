"""
hb_salma.py
===========

House-API adapter for **HB - Salma**, the vendored hierarchical-confidence
observer in ``observers.models.salma_hierarchical_helpers`` (a byte-identical copy of
the branch model; see that subpackage's docstring).

Why an adapter and not a re-port
--------------------------------
Salma's model is written against a different contract from the other observers:
it is **subject-batch** (it consumes one participant's whole trial table at once
and returns all response PMFs together, because its confidence trajectory is a
single ordered pass) and it lives on a **72-bin (5 deg)** angle grid with a
tie-aware MAP read-out. The other observers here are **per-trial-sequence**
(``filter(directions, coherences, feedback)``) on a **360-bin (1 deg)** grid.

Re-deriving her mechanism in the house style on a 360-bin grid would produce
numbers that silently differ from the HB - Salma results already reported in the
comparison docs. So instead of re-porting, this adapter wraps her REAL code and
translates the two calling conventions, so team code can do::

    from observers.models.hb_salma import HBSalmaObserver
    obs = HBSalmaObserver()
    out = obs.filter(directions, coherences, feedback=directions)   # like the others
    nll = obs.negative_log_likelihood(estimates, directions, coherences)

Grid note (read before comparing NLL/AIC)
-----------------------------------------
Her native likelihood is on 72 bins; the other observers score on 360 bins.
NLL/AIC on different grids are NOT comparable (a coarser grid concentrates mass
into fewer bins and mechanically lowers NLL). This adapter therefore exposes:
  * ``negative_log_likelihood(..., grid="native")``  -> her own 72-bin score
    (matches the published HB - Salma numbers);
  * ``negative_log_likelihood(..., grid="deg360")``  -> her 72-bin PMFs
    up-sampled to 360 bins (each 1 deg direction takes its containing 5 deg
    bin's mass / 5), so it is directly comparable to the other observers.
Use ``grid="deg360"`` for any cross-model AIC; use ``grid="native"`` to
reproduce her package's own output.

Parameters (6, fitted): rho, 3 sensory kappas, motor kappa, lapse. No alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from observers.models.salma_hierarchical_helpers import (
    GridSpec,
    ModelParameters,
    HierarchicalObserver,
    prepare_subject,
    PRIOR_MEAN_DEGREES,
)


N_PARAMS = 6  # rho + 3 sensory kappas + motor kappa + lapse


def _pmf72_to_deg360(pmf72: np.ndarray, n_angles: int) -> np.ndarray:
    """Up-sample a 72-bin PMF (bin centres 0,5,...,355) to a length-360 array
    aligned to directions 1..360, spreading each bin's mass uniformly over the
    5 one-degree slots it covers so the result still sums to 1."""
    step = 360 // n_angles                      # 5
    # her bin b (centre b*step) covers integer degrees [b*step-2, b*step+2].
    deg = np.arange(1, 361)
    # map each degree to its nearest 72-bin centre (same rule as angle_to_bin)
    b = np.floor(((deg % 360) + step / 2.0) / step).astype(int) % n_angles
    return pmf72[b] / step


@dataclass
class HBSalmaObserver:
    """House-API wrapper around the vendored HB - Salma observer.

    Fields mirror the other observers' constructor style (defaults are the
    documented HB - Salma defaults). ``filter`` / ``estimate_distribution`` /
    ``negative_log_likelihood`` present the same interface as the other models;
    internally every call builds a one-subject table and drives the real
    ``HierarchicalObserver``.
    """

    rho: float = 0.9
    sensory_kappas: tuple = (1.5, 3.0, 8.0)     # per coherence, ascending
    motor_kappa: float = 40.0
    lapse: float = 0.02
    grid: GridSpec = field(default_factory=GridSpec)

    N_PARAMS = N_PARAMS

    # -- internals -----------------------------------------------------------
    def _build(self, directions, coherences, feedback):
        """Assemble a one-subject PreparedSubject + HierarchicalObserver from
        house-style per-trial arrays. ``feedback`` (the true direction shown at
        trial end) drives her confidence update, exactly as ``directions`` do in
        her own pipeline; response_angle is a placeholder (overwritten per call).
        """
        directions = np.asarray(directions, dtype=float)
        coherences = np.asarray(coherences, dtype=float)
        feedback = directions if feedback is None else np.asarray(feedback, dtype=float)
        n = directions.size
        # her confidence trajectory reads directions; feed the FEEDBACK sequence
        # in as the driving directions so learning matches the house convention.
        df = pd.DataFrame(dict(
            subject_id=0,
            motion_direction=feedback,
            motion_coherence=coherences,
            response_angle=np.zeros(n),           # placeholder; not used for PMFs
            response_valid=True,
            prior_std=80.0,
            block_id=0,
            trial_index=np.arange(n),
        ))
        # her sensory kappas are indexed by ascending unique coherence; make sure
        # the number of sensory kappas matches the coherence levels present.
        cohs = np.sort(pd.unique(coherences.astype(float)))
        sk = np.asarray(self.sensory_kappas, dtype=float)
        if sk.size != cohs.size:
            # broadcast / trim: use the first cohs.size kappas, or repeat last
            sk = (np.resize(sk, cohs.size) if sk.size < cohs.size else sk[:cohs.size])
        sub = prepare_subject(df, 0, self.grid)
        obs = HierarchicalObserver(sub, self.grid)
        params = ModelParameters(rho=float(self.rho), sensory_kappas=sk,
                                 motor_kappa=float(self.motor_kappa),
                                 lapse=float(self.lapse))
        return sub, obs, params

    # -- house API -----------------------------------------------------------
    def filter(self, directions, coherences, feedback=None,
               sample: bool = False, rng: Optional[np.random.RandomState] = None,
               record_belief: bool = False, grid: str = "deg360"):
        """Return per-trial response distributions (list of arrays), matching
        the other observers' ``filter``. ``grid="deg360"`` (default) returns
        length-360 PMFs aligned to directions 1..360; ``grid="native"`` returns
        her 72-bin PMFs. ``record_belief`` adds the believed prior SD trajectory.
        """
        sub, obs, params = self._build(directions, coherences, feedback)
        pmfs = obs.predict_response_pmfs(params)             # (n, 72)
        if grid == "deg360":
            dists = [_pmf72_to_deg360(pmfs[t], self.grid.n_angles)
                     for t in range(pmfs.shape[0])]
            support = np.arange(1, 361)
        else:
            dists = [pmfs[t] for t in range(pmfs.shape[0])]
            support = self.grid.theta_degrees
        out = {"dists": dists}
        if sample:
            rng = rng or np.random.RandomState()
            out["responses"] = np.array([int(rng.choice(support, p=d / d.sum()))
                                         for d in dists])
        if record_belief:
            h_before, _ = obs.confidence_trajectory(float(self.rho))
            # believed SD = circular SD of the effective prior each trial
            eff = h_before @ obs._prior_basis
            eff /= eff.sum(axis=1, keepdims=True)
            th = np.deg2rad(self.grid.theta_degrees)
            R = np.abs((eff * np.exp(1j * th)[None, :]).sum(axis=1))
            out["believed_sd"] = np.rad2deg(np.sqrt(np.maximum(-2.0 * np.log(np.clip(R, 1e-12, 1)), 0)))
        return out

    def estimate_distribution(self, coherence: float, direction: int,
                              feedback_history=None, grid: str = "deg360"):
        """Single-trial convenience: the response distribution for one trial.

        Because HB - Salma's percept depends on the whole preceding feedback
        sequence (its confidence trajectory), a truly isolated single-trial call
        is ill-defined. Pass ``feedback_history`` (the directions seen on all
        trials up to and including this one) to get the correct distribution for
        the LAST trial; with no history it returns the trial-0 distribution
        (uniform hyper-prior).
        """
        if feedback_history is None:
            dirs = np.array([direction]); cohs = np.array([coherence])
        else:
            dirs = np.asarray(feedback_history, dtype=float)
            cohs = np.full(dirs.size, coherence, dtype=float)
        out = self.filter(dirs, cohs, feedback=dirs, grid=grid)
        return out["dists"][-1]

    def negative_log_likelihood(self, estimates, directions, coherences,
                                feedback=None, grid: str = "native") -> float:
        """Negative log likelihood of observed ``estimates`` (integer degrees).

        ``grid="native"`` (default) uses her own 72-bin scoring — reproduces the
        published HB - Salma numbers. ``grid="deg360"`` scores the up-sampled
        360-bin PMFs and IS comparable to the other observers' NLL.
        """
        estimates = np.asarray(estimates, dtype=int)
        if grid == "native":
            sub, obs, params = self._build(directions, coherences, feedback)
            # inject the real responses so her own NLL scores them
            from observers.models.salma_hierarchical_helpers import prepare_subject as _ps
            directions = np.asarray(directions, dtype=float)
            feedback = directions if feedback is None else np.asarray(feedback, dtype=float)
            coherences = np.asarray(coherences, dtype=float)
            df = pd.DataFrame(dict(
                subject_id=0, motion_direction=feedback, motion_coherence=coherences,
                response_angle=estimates.astype(float), response_valid=True,
                prior_std=80.0, block_id=0, trial_index=np.arange(estimates.size)))
            cohs = np.sort(pd.unique(coherences.astype(float)))
            sk = np.asarray(self.sensory_kappas, dtype=float)
            if sk.size != cohs.size:
                sk = (np.resize(sk, cohs.size) if sk.size < cohs.size else sk[:cohs.size])
            sub = _ps(df, 0, self.grid)
            obs = HierarchicalObserver(sub, self.grid)
            params = ModelParameters(rho=float(self.rho), sensory_kappas=sk,
                                     motor_kappa=float(self.motor_kappa), lapse=float(self.lapse))
            return float(obs.negative_log_likelihood(params))
        # deg360: score the up-sampled PMFs at the integer-degree responses
        out = self.filter(directions, coherences, feedback=feedback, grid="deg360")
        ll = 0.0
        for t, d in enumerate(out["dists"]):
            ll += np.log(max(d[(estimates[t] - 1) % 360], 1e-320))
        return -float(ll)

    @staticmethod
    def aic(nll: float, n_params: int = N_PARAMS) -> float:
        return 2.0 * n_params + 2.0 * nll

    @staticmethod
    def bic(nll: float, n_trials: int, n_params: int = N_PARAMS) -> float:
        return n_params * np.log(n_trials) + 2.0 * nll
