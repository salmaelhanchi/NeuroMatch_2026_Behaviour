"""
dataset.py
==========

Data loading and synthetic-design generation for the observer models.

Two design sources:
    load_subject_design(csv, subject) : one subject's real trials, in order,
        from the four-prior motion experiment (data01_direction4priors.csv).
    make_synthetic_design(...)        : blocks with known prior widths, for
        parameter- and model-recovery tests.

For synthetic data only the *responses* are simulated; the design (directions,
coherences, block structure) is either drawn to mimic the experiment or taken
verbatim from the human CSV, so recovery tests exercise the pipeline exactly as
it runs on real data. A synthetic rollout also returns the ground-truth
belief-SD trajectory alongside the observable design + responses, so a recovery
test can check the fitted trajectory against the true one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from observers.helpers.circular import von_mises_pdfs, DIRECTION_SPACE

# PRIOR_MEAN is a fixed task constant (225 deg). ``OnlineHierarchicalObserver``
# lives in the gitignored ``other models/`` folder and is only used by the
# optional ``simulate()`` helper below — import it lazily so the core data
# loader (used by the whole comparison pipeline) works in a clean checkout.
PRIOR_MEAN = 225.0

# prior SD label -> concentration k used to GENERATE that block's directions.
# Obtained by numerically inverting circular.von_mises_std so the labels are
# honest: k below reproduces 80/40/20/10 deg circular SD to 2 decimals.
SD_TO_K = {80: 0.7485, 40: 2.7714, 20: 8.7488, 10: 33.3373}
COHERENCES = np.array([0.06, 0.12, 0.24])


def _draw_from_prior(k: float, n: int, rng) -> np.ndarray:
    """Draw n directions (1..360) from a von Mises prior of strength k at 225."""
    p = von_mises_pdfs(DIRECTION_SPACE, PRIOR_MEAN, k, normalize=True).ravel()
    return rng.choice(DIRECTION_SPACE, size=n, p=p)


def make_synthetic_design(block_sds=(80, 40, 20, 10), trials_per_block=250,
                          seed: int = 0):
    """Build a controlled experiment design: one block per prior width, each
    with directions drawn from that block's true prior and random coherences.

    Returns a DataFrame with the columns the model consumes.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for sd in block_sds:
        k = SD_TO_K[sd]
        dirs = _draw_from_prior(k, trials_per_block, rng)
        cohs = rng.choice(COHERENCES, size=trials_per_block)
        for d, c in zip(dirs, cohs):
            rows.append((int(d), float(c), int(sd)))
    df = pd.DataFrame(rows, columns=["motion_direction", "motion_coherence",
                                     "prior_std"])
    return df


def load_subject_design(csv_path: str, subject_id: int):
    """Load one human subject's trials in chronological order from the CSV.

    Returns a DataFrame with motion_direction, motion_coherence, prior_std and
    the subject's actual estimate (degrees, 1..360) computed from the recorded
    Cartesian response.
    """
    df = pd.read_csv(csv_path)
    df = df[df.subject_id == subject_id].copy()
    df = df.sort_values(["session_id", "run_id", "trial_index"]).reset_index(drop=True)
    # estimate: Cartesian (x,y) -> direction in 1..360
    ang = np.degrees(np.arctan2(df.estimate_y.values, df.estimate_x.values)) % 360.0
    ang[ang == 0] = 360.0
    df["estimate_dir"] = np.round(ang).astype(int).clip(1, 360)
    keep = df[["motion_direction", "motion_coherence", "prior_std",
               "estimate_dir"]].copy()
    keep = keep.dropna(subset=["motion_direction", "motion_coherence"])
    keep["motion_direction"] = keep["motion_direction"].round().astype(int).clip(1, 360)
    return keep.reset_index(drop=True)


def simulate(observer, design, seed: int = 0):
    """Roll the observer through a design, sampling a response per trial.

    Returns a dict with the design arrays, sampled 'estimates', and the HIDDEN
    'believed_sd' trajectory (ground truth for debugging / Phase-4 checks).
    """
    rng = np.random.RandomState(seed)
    directions = design["motion_direction"].values.astype(int)
    coherences = design["motion_coherence"].values.astype(float)
    out = observer.filter(directions, coherences, feedback=directions,
                          sample=True, rng=rng, record_belief=True)
    return {
        "motion_direction": directions,
        "motion_coherence": coherences,
        "prior_std": design["prior_std"].values if "prior_std" in design else None,
        "estimates": out["responses"],
        "believed_sd": out["believed_sd"],
    }


if __name__ == "__main__":
    # smoke test: simulate one dataset and show the belief learning the prior.
    # Requires the optional OnlineHierarchicalObserver (gitignored 'other models/').
    from observers.models.online_switching_observer import OnlineHierarchicalObserver
    obs = OnlineHierarchicalObserver(
        k_like={0.24: 8.0, 0.12: 3.0, 0.06: 1.0}, k_motor=40.0,
        p_random=0.02, lam=0.02)
    design = make_synthetic_design(trials_per_block=200, seed=1)
    sim = simulate(obs, design, seed=1)
    print("simulated trials:", sim["estimates"].size)
    # believed prior SD at the end of each block should approach the true SD
    sd = sim["believed_sd"]
    for i, block_sd in enumerate((80, 40, 20, 10)):
        end = (i + 1) * 200 - 1
        print(f"  block true SD={block_sd:>2}  believed SD at block end "
              f"= {sd[end]:5.1f} deg")