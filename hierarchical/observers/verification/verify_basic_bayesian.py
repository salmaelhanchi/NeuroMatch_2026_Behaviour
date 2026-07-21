"""
verify_basic_bayesian.py
========================

Verify the Basic Bayesian observer against oracles with KNOWN answers before any
fit is trusted.

  1. reduction: prior off (k_prior->0) => estimate follows the evidence (stimulus)
  2. reduction: likelihood off (k_like->0) => estimate collapses onto the prior (225)
  3. ORACLE: equals hb_integration at alpha=1 with the belief pinned to a delta at
     a grid kappa (an independently-verified basic-Bayesian ground truth)
  4. all condition distributions are valid probability vectors
  5. NLL is lower at the generating parameters than at deliberately-wrong ones
  (observation, not a gate) the combined percept is unimodal, unlike the switch

Run:  python -m observers.verification.verify_basic_bayesian
"""
from __future__ import annotations

import numpy as np

from observers.helpers.circular import DIRECTION_SPACE
from observers.models.basic_bayesian import BasicBayesianObserver, PRIOR_MEAN
from observers.models.hb_integration import HBIntegrationObserver

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _sdist(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _peak(dist):
    return int(np.argmax(dist)) + 1


def test_reduction_evidence():
    # prior switched off -> the integrated MAP is just the measurement -> peak at
    # the stimulus direction, far from the prior mean.
    d = 85
    obs = BasicBayesianObserver(k_like={0.24: 8.0, 0.12: 3.0, 0.06: 8.0},
                                k_prior={"80": 0.0, "40": 0.0, "20": 0.0, "10": 0.0},
                                p_random=0.0, k_motor=1e6)
    peak = _peak(obs.estimate_distribution(d, 0.06, "80"))
    check("k_prior->0 => estimate at the stimulus", abs(_sdist(peak, d)) <= 2,
          f"peak={peak}, stimulus={d}")


def test_reduction_prior():
    # likelihood switched off -> every measurement maps to the prior mode -> 225.
    d = 85
    obs = BasicBayesianObserver(k_like={0.24: 0.0, 0.12: 0.0, 0.06: 0.0},
                                k_prior={"80": 8.7, "40": 8.7, "20": 8.7, "10": 8.7},
                                p_random=0.0, k_motor=1e6)
    peak = _peak(obs.estimate_distribution(d, 0.06, "80"))
    check("k_like->0 => estimate at the prior mean (225)",
          abs(_sdist(peak, PRIOR_MEAN)) <= 2, f"peak={peak}, prior_mean=225")


def test_matches_hb_alpha1():
    # The strong oracle: hb_integration at alpha=1 IS the basic Girshick Bayesian
    # (verified to 1e-17 in verify_hb_integration). With its belief pinned to a
    # delta at a grid kappa and no lapse, it must equal this model with
    # k_prior=that kappa. A different code path arriving at the same numbers.
    k_like = {0.06: 1.0, 0.12: 3.0, 0.24: 8.0}
    k_motor = 40.0
    coh, direction = 0.06, 85

    hb = HBIntegrationObserver(k_like=k_like, alpha=1.0, lam=0.0,
                               k_motor=k_motor, p_random=0.0)
    hb._prepare(np.array([direction]), np.array([coh]))
    i = 8  # an interior grid point
    kap = float(hb.k_grid[i])
    belief = np.zeros(hb.k_grid.size)
    belief[i] = 1.0
    hb_dist = hb.estimate_distribution(coh, direction, belief)

    bb = BasicBayesianObserver(k_like=k_like,
                               k_prior={"80": kap, "40": kap, "20": kap, "10": kap},
                               k_motor=k_motor, p_random=0.0)
    bb_dist = bb.estimate_distribution(direction, coh, "80")

    maxdiff = float(np.max(np.abs(hb_dist - bb_dist)))
    check("equals hb_integration(alpha=1) at a delta belief (kappa=%.3f)" % kap,
          maxdiff < 1e-8, f"max|Δ|={maxdiff:.2e}")


def test_valid_distributions():
    obs = BasicBayesianObserver()
    bad = 0
    for d in range(5, 360, 40):
        for c in (0.06, 0.12, 0.24):
            for pl in ("80", "40", "20", "10"):
                dist = obs.estimate_distribution(d, c, pl)
                if abs(dist.sum() - 1) > 1e-9 or np.any(dist < 0):
                    bad += 1
    check("all 108 condition distributions are valid", bad == 0,
          f"invalid distributions = {bad}")


def test_nll_responds_to_data():
    obs = BasicBayesianObserver()
    rng = np.random.RandomState(0)
    n = 200
    dd = rng.choice([25, 85, 145, 205, 265, 325], n)
    cc = rng.choice([0.06, 0.12, 0.24], n)
    pp = rng.choice(["80", "40", "20", "10"], n)
    est = []
    for d, c, pl in zip(dd, cc, pp):
        dist = obs.estimate_distribution(int(d), float(c), str(pl))
        est.append(rng.choice(np.arange(1, 361), p=dist))
    est = np.array(est)
    nll_true = obs.negative_log_likelihood(est, dd, cc, pp)
    obs_bad = BasicBayesianObserver(k_like={0.24: 0.1, 0.12: 0.1, 0.06: 0.1},
                                    k_prior={"80": 50, "40": 50, "20": 50, "10": 50})
    nll_bad = obs_bad.negative_log_likelihood(est, dd, cc, pp)
    check("NLL lower at generating than at bad params", nll_true < nll_bad,
          f"NLL(true)={nll_true:.1f} < NLL(bad)={nll_bad:.1f}")


def observe_unimodality():
    # Not a gate: report whether the combined percept is unimodal (it should be —
    # a product of two von Mises is a single von Mises — and this is the property
    # that distinguishes it from the bimodal Switching observer).
    obs = BasicBayesianObserver(p_random=0.0)
    multi = 0
    checked = 0
    for d in (25, 85, 145, 265, 325):
        for c in (0.06, 0.12, 0.24):
            for pl in ("80", "40", "20", "10"):
                dist = obs.estimate_distribution(d, c, pl)
                # count local maxima on the circle above a small floor
                peaks = 0
                for k in range(360):
                    v = dist[k]
                    if v > dist[k - 1] and v >= dist[(k + 1) % 360] and v > 1e-4:
                        peaks += 1
                checked += 1
                multi += (peaks > 1)
    print(f"[note] unimodality: {checked - multi}/{checked} conditions single-peaked "
          f"(observation, not a pass/fail gate)")


def run():
    """Run every check; print PASS/FAIL and a summary. Returns (passed, total)."""
    results.clear()
    print("=== Verification: Basic Bayesian observer ===")
    test_reduction_evidence()
    test_reduction_prior()
    test_matches_hb_alpha1()
    test_valid_distributions()
    test_nll_responds_to_data()
    observe_unimodality()
    print("=" * 55)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} checks passed"
          + ("  ✅ ALL PASS" if passed == total else "  ❌ FAILURES PRESENT"))
    return passed, total


if __name__ == "__main__":
    run()
