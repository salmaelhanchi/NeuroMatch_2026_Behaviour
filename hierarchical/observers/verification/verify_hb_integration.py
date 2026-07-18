"""
verify_hb_integration.py
========================

Verification for the Hierarchical Bayesian *integration* observer
(``hb_integration.py``) — the abstract's model. Same structure as
verify_online.py.

The point of this script is not "does it run" but "does it do what the abstract
claims, and is it actually *different* from the switching family?" The switch
model is bimodal too, so showing two peaks proves nothing on
its own. The discriminating checks (T5) are the ones that matter.

Checks
------
T1  Reduction: alpha=1 mixture read-out == girshick_map_lookup (single VM prior).
T2  All estimate distributions are proper (sum to 1).
T3  Emergence: bimodality appears with NO switch — far-from-prior + low coherence
    gives two peaks (near stimulus AND near prior); near-prior / high coherence
    is unimodal. "reproduce both unimodal and bimodal" (abstract).
T4  alpha -> 0: the prior vanishes, estimate follows the evidence (unimodal).
T5  DISCRIMINATOR vs the switch model: in the integration model prior reliance
    DECLINES as the stimulus moves away from 225 (a far measurement is poorly
    explained by the von Mises component, so the uniform floor wins); in the
    switch model the prior weight is flat across direction at fixed
    coherence/belief. This is the falsifiable difference.
T6  Learning: the belief over kappa converges to a known kappa from feedback;
    the recursive update equals the batch posterior (lambda=0).
T7  Cost: wall-time of one negative_log_likelihood call on a real subject
    (decide fit viability empirically before any batch).
"""

from __future__ import annotations

import time
import numpy as np

from observers.helpers.circular import von_mises_std, DIRECTION_SPACE
from observers.helpers.bayes_lookup import girshick_map_lookup
from observers.models.hb_integration import (HBIntegrationObserver, mixture_map_lookup, PRIOR_MEAN)
from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.helpers.paths import DATA_CSV

results = []


def _check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  — {detail}" if detail else ""))


def _sdist(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _peaks(dist, window=8, min_frac=0.02):
    """Local maxima of a circular pmf that carry non-trivial mass."""
    n = dist.size
    out = []
    for i in range(n):
        lo = dist[(np.arange(i - window, i)) % n]
        hi = dist[(np.arange(i + 1, i + window + 1)) % n]
        if dist[i] >= lo.max() and dist[i] >= hi.max() and dist[i] > min_frac * dist.max():
            out.append(i + 1)  # 1..360
    return out


# ---------------------------------------------------------------------------
def test_reduction():
    """alpha=1 must reproduce the single-von-Mises Girshick lookup exactly."""
    dirs = np.array([85, 155, 225, 300])
    for k_like, kappa in [(1.0, 2.7), (8.0, 8.7), (3.0, 0.5)]:
        L_mix = mixture_map_lookup(k_like, PRIOR_MEAN, kappa, alpha=1.0,
                                   motion_dirs=dirs)
        L_ref = girshick_map_lookup(k_like, PRIOR_MEAN, kappa, motion_dirs=dirs)
        d = np.nanmax(np.abs(np.nan_to_num(L_mix) - np.nan_to_num(L_ref)))
        _check(f"alpha=1 == Girshick (k_e={k_like}, kappa={kappa})", d < 1e-9,
               f"max|Δ|={d:.2e}")


def test_normalisation():
    obs = HBIntegrationObserver()
    obs._prepare(np.array([85, 155, 225, 300]),
                 np.array([0.06, 0.12, 0.24]))
    belief = obs.initial_belief()
    worst = 0.0
    for c in (0.06, 0.12, 0.24):
        for d in (85, 155, 225, 300):
            s = obs.estimate_distribution(c, d, belief).sum()
            worst = max(worst, abs(s - 1.0))
    _check("estimate distributions sum to 1", worst < 1e-9, f"max|sum-1|={worst:.2e}")


def test_emergence():
    """Bimodality emerges from integration (no switch code involved)."""
    obs = HBIntegrationObserver(alpha=0.6)
    dirs = np.array([85, 225])
    obs._prepare(dirs, np.array([0.06, 0.24]))
    belief = obs.initial_belief()

    # far from prior (85), low coherence -> expect two peaks (stim ~85, prior ~225)
    d_far = obs.estimate_distribution(0.06, 85, belief)
    pk = _peaks(d_far)
    near_stim = any(abs(_sdist(p, 85)) < 30 for p in pk)
    near_prior = any(abs(_sdist(p, 225)) < 40 for p in pk)
    _check("far+low-coh: bimodal (stimulus AND prior peaks)",
           near_stim and near_prior and len(pk) >= 2,
           f"peaks={pk}")

    # near prior (225) -> unimodal at the prior/stimulus
    d_near = obs.estimate_distribution(0.06, 225, belief)
    pk_near = _peaks(d_near)
    _check("near-prior: unimodal", len(pk_near) == 1, f"peaks={pk_near}")

    # far from prior BUT high coherence -> evidence dominates, ~unimodal at stim
    d_hi = obs.estimate_distribution(0.24, 85, belief)
    pk_hi = _peaks(d_hi)
    main_at_stim = abs(_sdist(int(np.argmax(d_hi)) + 1, 85)) < 20
    _check("far+high-coh: evidence dominates (peak at stimulus)", main_at_stim,
           f"argmax={int(np.argmax(d_hi))+1}, peaks={pk_hi}")


def test_alpha_zero():
    """alpha -> 0: prior has no structure, estimate follows evidence."""
    obs = HBIntegrationObserver(alpha=1e-6)
    obs._prepare(np.array([85]), np.array([0.06]))
    belief = obs.initial_belief()
    d = obs.estimate_distribution(0.06, 85, belief)
    pk = _peaks(d)
    at_stim = abs(_sdist(int(np.argmax(d)) + 1, 85)) < 20
    _check("alpha->0: unimodal at the stimulus (prior gone)",
           at_stim and len(pk) == 1, f"argmax={int(np.argmax(d))+1}, peaks={pk}")


def test_discriminator():
    """Integration vs switch: prior reliance vs stimulus-to-prior distance.

    Match both models at a FIXED prior strength (point-mass belief for the
    integration model; fixed k_prior for the switch model) and a narrow
    (high-coherence) likelihood so the evidence peak does not leak into the
    prior window. Measure response mass within +/-18 deg of 225 for stimuli at
    growing distance from 225.

    Prediction: the integration model's prior-window mass DECLINES with
    distance; the switch model's is roughly FLAT (its prior read-out is a delta
    at 225 with a direction-independent weight).
    """
    k_prior = 2.7          # prior SD ~40 deg
    coh = 0.24             # k_e = 8 -> fairly narrow evidence
    stim_dirs = [165, 145, 125, 105]     # distances 60,80,100,120 from 225
    win = (225 - 18, 225 + 18)

    integ = HBIntegrationObserver(alpha=0.6)
    integ._prepare(np.array(stim_dirs), np.array([coh]))
    # point-mass belief at the grid kappa closest to k_prior
    ki = int(np.argmin(np.abs(integ.k_grid - k_prior)))
    b = np.zeros(integ.k_grid.size); b[ki] = 1.0

    switch = OnlineHierarchicalObserver(
        k_like={0.24: 8.0, 0.12: 3.0, 0.06: 1.0}, k_motor=40.0, p_random=0.01)
    switch._prepare(np.array(stim_dirs), np.array([coh]))

    def window_mass(dist):
        idx = np.arange(win[0] - 1, win[1])  # 0-based
        return float(dist[idx].sum())

    integ_curve, switch_curve = [], []
    for d in stim_dirs:
        integ_curve.append(window_mass(integ.estimate_distribution(coh, d, b)))
        switch_curve.append(window_mass(
            switch.estimate_distribution_fixedk(coh, d, k_prior)))

    ic = np.array(integ_curve); sc = np.array(switch_curve)
    # integration should fall monotonically with distance and vary much more
    integ_declines = np.all(np.diff(ic) < 0)
    integ_cv = ic.std() / ic.mean()
    switch_cv = sc.std() / (sc.mean() + 1e-12)
    print(f"    stim dirs {stim_dirs} (dist 60..120 from 225)")
    print(f"    integration prior-window mass: "
          + ", ".join(f"{v:.3f}" for v in ic) + f"   (CV={integ_cv:.2f})")
    print(f"    switch      prior-window mass: "
          + ", ".join(f"{v:.3f}" for v in sc) + f"   (CV={switch_cv:.2f})")
    _check("DISCRIMINATOR: integration prior-reliance declines with distance "
           "while switch stays ~flat",
           integ_declines and integ_cv > 3 * switch_cv,
           f"integ CV={integ_cv:.2f} vs switch CV={switch_cv:.2f}")


def test_prior_mode_offset():
    """Secondary discriminator: the integration 'prior' peak is pulled off 225
    toward the stimulus, and the pull grows as the prior widens (small kappa).
    Reported (not asserted hard) — it can be small once motor noise blurs it."""
    obs = HBIntegrationObserver(alpha=0.7)
    obs._prepare(np.array([105]), np.array([0.12]))
    print("    prior-peak location for stimulus=105 (prior mean 225):")
    for kappa, label in [(0.5, "wide ~SD80"), (2.7, "mid ~SD40"), (8.7, "tight ~SD10")]:
        ki = int(np.argmin(np.abs(obs.k_grid - kappa)))
        b = np.zeros(obs.k_grid.size); b[ki] = 1.0
        d = obs.estimate_distribution(0.12, 105, b)
        seg = d[205 - 1:245]                       # window around the prior mode
        loc = 205 + int(np.argmax(seg))
        print(f"      kappa={kappa:>4} ({label}): prior peak at {loc} deg "
              f"(offset {loc-225:+d} from prior mean)")


def test_learning_and_batch():
    """Belief over kappa converges to the true kappa; recursive == batch (lam=0)."""
    rng = np.random.RandomState(0)
    true_sd = 40.0
    # find the kappa whose circular SD ~ true_sd, draw feedback from it
    from observers.helpers.belief_grid import make_k_grid
    grid = make_k_grid(n=15)
    true_k = grid[int(np.argmin([abs(von_mises_std(k) - true_sd) for k in grid]))]
    from observers.helpers.circular import von_mises_pdfs
    prior_pmf = von_mises_pdfs(DIRECTION_SPACE, PRIOR_MEAN, true_k, normalize=True).ravel()
    feedback = rng.choice(np.arange(1, 361), size=600, p=prior_pmf)

    obs = HBIntegrationObserver(alpha=0.999, lam=0.0, k_grid=grid)
    obs._prepare(np.array([225]), np.array([0.12]))  # builds obs table
    b = obs.initial_belief()
    for f in feedback:
        b = obs.update_belief(b, int(f))
    learned_sd = von_mises_std(float(np.sum(b * grid)))
    _check("belief converges to true kappa (SD~40)", abs(learned_sd - true_sd) < 8,
           f"true SD={true_sd:.0f}, learned SD={learned_sd:.1f}")

    # batch cross-check (lam=0): recursive product == batch posterior
    logb = np.log(np.clip(obs.initial_belief(), 1e-320, None))
    logL = np.log(np.clip(obs._obs_table[:, feedback - 1], 1e-320, None)).sum(axis=1)
    lp = logb + logL; lp -= lp.max()
    batch = np.exp(lp); batch /= batch.sum()
    _check("recursive belief == batch posterior (lambda=0)",
           np.max(np.abs(b - batch)) < 1e-10,
           f"max|Δ|={np.max(np.abs(b - batch)):.2e}")


def test_timing():
    """Time ONE full-sequence NLL eval on a real subject (advisor point 1)."""
    try:
        from observers.helpers.dataset import load_subject_design
        d = load_subject_design(DATA_CSV, 1)
    except Exception as e:  # data not present / pandas missing
        print(f"    [skip timing — could not load subject data: {e}]")
        return
    data = dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                estimates=d.estimate_dir.values.astype(int))
    obs = HBIntegrationObserver()
    t0 = time.time()
    nll = obs.negative_log_likelihood(data["estimates"], data["motion_direction"],
                                      data["motion_coherence"])
    dt = time.time() - t0
    n = data["estimates"].size
    print(f"    subject 1: n={n} trials, NLL={nll:.1f}, "
          f"AIC={HBIntegrationObserver.aic(nll):.1f}, "
          f"BIC={HBIntegrationObserver.bic(nll, n):.1f}")
    print(f"    one NLL eval = {dt:.2f} s (grid={obs.k_grid.size} kappa). "
          f"NOTE: a full 7-dim Nelder-Mead fit ran ~3-5 min/subject in practice "
          f"(many more than 200 evals, plus slow NaN regions the optimiser "
          f"probes) — budget accordingly and prefer warm/multi-start for a batch.")
    _check("single NLL eval completes", np.isfinite(nll), f"{dt:.2f}s")


def run():
    """Run every check, print a PASS/FAIL line each, and a summary.

    Returns (passed, total) so a notebook can assert on it. Safe to call
    repeatedly (the results tally is reset on each call).
    """
    results.clear()
    print("=== Verification: Hierarchical Bayesian INTEGRATION observer ===")
    print("\n-- T1 reduction --");            test_reduction()
    print("\n-- T2 normalisation --");        test_normalisation()
    print("\n-- T3 emergence of bimodality --"); test_emergence()
    print("\n-- T4 alpha->0 --");             test_alpha_zero()
    print("\n-- T5 discriminator vs switch --"); test_discriminator()
    print("\n-- (secondary) prior-mode offset --"); test_prior_mode_offset()
    print("\n-- T6 online learning --");      test_learning_and_batch()
    print("\n-- T7 fit-cost timing --");      test_timing()
    print("\n" + "=" * 60)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} checks passed"
          + ("  ✅ ALL PASS" if passed == total else "  ❌ FAILURES PRESENT"))
    return passed, total


if __name__ == "__main__":
    run()
