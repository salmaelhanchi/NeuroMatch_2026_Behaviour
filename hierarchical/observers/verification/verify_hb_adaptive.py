"""
verify_hb_adaptive.py
=====================

Verification for the HB-Adaptive-Confidence observer
(``hb_adaptive_confidence.py``, ``HBAdaptiveConfidenceObserver``) — **the
abstract's model, taken literally**: a Bayesian observer that learns BOTH the
prior confidence α and the prior width κ online, so neither is fitted (k = 6).

The point of this script is not "does it run" but "does it do the specific thing
the abstract claims for THIS model, and is it structurally distinct from
HB-Rachel (which fits a fixed α) and the switch family?" The discriminating
checks are T5 and T6.

Checks
------
T1  Reduction: at a belief pinned to a single (κ, α=1) grid pair the mixed prior
    equals the single von Mises V(θ;225,κ); at α=0 it equals the uniform floor.
    (mixture_prior identity — the atom the joint belief is built from.)
T2  All estimate distributions are proper pmfs (sum to 1, non-negative).
T3  Emergence: with a spread belief, far-from-prior + low coherence gives two
    peaks (near stimulus AND near 225); near-prior / high coherence is unimodal.
    "reproduce both unimodal and bimodal" (abstract prediction i), no switch.
T4  Six free parameters, α and κ absent from the fitted vector: the fitter's
    pack/unpack round-trips exactly six numbers and none of them is α or κ.
T5  DISCRIMINATOR vs HB-Rachel — **confidence is learned, not fitted**. Feeding
    feedback from a tight prior at 225 makes E[α] RISE over trials; feeding
    uniform feedback makes E[α] FALL. HB-Rachel's α is a constant, so it cannot
    move. This is the abstract's "confidence updated over trials" clause.
T6  Width is learned too: feedback from a known-width prior drives the believed
    prior SD toward that width (E[κ] → the generating SD), recovered from
    feedback alone (prior_std is never an input).
T7  Causal ordering (no leakage): the response distribution at trial t is
    computed from the belief BEFORE trial t's feedback is folded in — verified
    by showing trial-0's distribution is the flat-belief read-out, independent
    of what feedback[0] is.
T8  Cost: wall-time of one full-sequence NLL eval on a real subject.
"""

from __future__ import annotations

import time
import numpy as np

from observers.helpers.circular import von_mises_pdfs, von_mises_std, DIRECTION_SPACE
from observers.models.hb_rachel import mixture_prior, PRIOR_MEAN
from observers.models.hb_adaptive_confidence import (
    HBAdaptiveConfidenceObserver, make_alpha_grid)
from observers.helpers.paths import DATA_CSV

results = []


def _check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _sdist(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _peaks(dist, window=8, min_frac=0.02):
    """Local maxima of a circular pmf carrying non-trivial mass."""
    n = dist.size
    out = []
    for i in range(n):
        lo = dist[(np.arange(i - window, i)) % n]
        hi = dist[(np.arange(i + 1, i + window + 1)) % n]
        if dist[i] >= lo.max() and dist[i] >= hi.max() and dist[i] > min_frac * dist.max():
            out.append(i + 1)
    return out


def _pinned_belief(obs, kappa, alpha):
    """A belief vector putting all mass on the (κ, α) grid pair nearest the
    requested (kappa, alpha). The joint grid is meshgrid(k_grid, a_grid,
    indexing='ij'), so κ is the slow axis and α the fast axis."""
    kap, alp = obs._joint_grid()
    ki = int(np.argmin(np.abs(obs.k_grid - kappa)))
    ai = int(np.argmin(np.abs(obs.a_grid - alpha)))
    idx = ki * obs.a_grid.size + ai
    b = np.zeros(kap.size); b[idx] = 1.0
    return b, kap[idx], alp[idx]


# ---------------------------------------------------------------------------
def test_reduction():
    """The mixed-prior atom reduces to the known limits (this IS the model's
    per-grid-pair prior p(θ|κ,α) = α·V(θ;225,κ) + (1-α)/360)."""
    for kappa in (0.5, 2.7, 8.7):
        vm = von_mises_pdfs(DIRECTION_SPACE, PRIOR_MEAN, kappa, normalize=True).ravel()
        d1 = np.max(np.abs(mixture_prior(kappa, 1.0) - vm))
        _check(f"α=1 mixed prior == V(θ;225,κ={kappa})", d1 < 1e-12, f"max|Δ|={d1:.2e}")
    d0 = np.max(np.abs(mixture_prior(8.0, 0.0) - 1.0 / 360.0))
    _check("α=0 mixed prior == uniform floor", d0 < 1e-12, f"max|Δ|={d0:.2e}")


def test_normalisation():
    obs = HBAdaptiveConfidenceObserver()
    obs._prepare(np.array([85, 155, 225, 300]), np.array([0.06, 0.12, 0.24]))
    belief = obs.initial_belief()
    worst_sum, any_neg = 0.0, False
    for c in (0.06, 0.12, 0.24):
        for d in (85, 155, 225, 300):
            dist = obs.estimate_distribution(c, d, belief)
            worst_sum = max(worst_sum, abs(dist.sum() - 1.0))
            any_neg = any_neg or bool(np.any(dist < 0))
    _check("estimate distributions are proper pmfs", worst_sum < 1e-9 and not any_neg,
           f"max|sum-1|={worst_sum:.2e}, any_negative={any_neg}")


def test_emergence():
    """Bimodality emerges from graded integration with a spread belief (no switch)."""
    obs = HBAdaptiveConfidenceObserver(p_random=0.0)
    dirs = np.array([85, 225]); cohs = np.array([0.06, 0.24])
    obs._prepare(dirs, cohs)
    belief = obs.initial_belief()   # flat over the (κ,α) grid: genuinely spread

    d_far = obs.estimate_distribution(0.06, 85, belief)
    pk = _peaks(d_far)
    near_stim = any(abs(_sdist(p, 85)) < 30 for p in pk)
    near_prior = any(abs(_sdist(p, 225)) < 45 for p in pk)
    _check("far+low-coh: bimodal (stimulus AND prior peaks)",
           near_stim and near_prior and len(pk) >= 2, f"peaks={pk}")

    d_near = obs.estimate_distribution(0.06, 225, belief)
    _check("near-prior: unimodal", len(_peaks(d_near)) == 1, f"peaks={_peaks(d_near)}")

    d_hi = obs.estimate_distribution(0.24, 85, belief)
    main_at_stim = abs(_sdist(int(np.argmax(d_hi)) + 1, 85)) < 20
    _check("far+high-coh: evidence dominates (peak at stimulus)", main_at_stim,
           f"argmax={int(np.argmax(d_hi)) + 1}")


def test_param_count():
    """Exactly six fitted params, and α/κ are NOT among them (they are learned)."""
    from observers.fitting import hb_adaptive_confidence_fit as F
    _check("N_PARAMS == 6", F.N_PARAMS == 6, f"N_PARAMS={F.N_PARAMS}")
    theta = F.pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 30.0, 0.05, 0.07)
    _check("packed θ has length 6", theta.size == 6, f"len={theta.size}")
    obs = F.unpack(theta)
    # the six params are the three k_like, k_motor, p_random, lam — reconstructed
    round_trip = F.pack(obs.k_like, obs.k_motor, obs.p_random, obs.lam)
    _check("pack/unpack round-trips the 6 params", np.allclose(theta, round_trip, atol=1e-9),
           f"max|Δ|={np.max(np.abs(theta - round_trip)):.2e}")
    _check("observer exposes learned α/κ grids (not fitted scalars)",
           hasattr(obs, "a_grid") and hasattr(obs, "k_grid")
           and not hasattr(obs, "alpha"),
           f"a_grid={obs.a_grid.size} pts, k_grid={obs.k_grid.size} pts, "
           f"has fixed alpha attr={hasattr(obs, 'alpha')}")


def test_confidence_is_learned():
    """DISCRIMINATOR vs HB-Rachel: E[α] moves with the feedback statistics.
    Informed (tight-prior) feedback -> E[α] rises; uniform feedback -> E[α] falls.
    HB-Rachel's α is a fixed fitted constant and cannot do this."""
    rng = np.random.RandomState(0)
    n = 500
    dirs = rng.randint(1, 361, n)
    cohs = rng.choice([0.06, 0.12, 0.24], n)
    obs = HBAdaptiveConfidenceObserver(lam=0.05)

    p_informed = von_mises_pdfs(DIRECTION_SPACE, 225.0, 8.0, normalize=True).ravel()
    fb_inf = rng.choice(np.arange(1, 361), size=n, p=p_informed)
    a_inf = obs.filter(dirs, cohs, feedback=fb_inf, record_belief=True)["believed_alpha"]

    fb_unif = rng.randint(1, 361, n)
    a_unif = obs.filter(dirs, cohs, feedback=fb_unif, record_belief=True)["believed_alpha"]

    rose = a_inf[-1] - a_inf[0] > 0.10
    fell = a_unif[-1] - a_unif[0] < -0.05
    print(f"    informed feedback: E[α] {a_inf[0]:.2f} -> {a_inf[-1]:.2f} (should rise)")
    print(f"    uniform  feedback: E[α] {a_unif[0]:.2f} -> {a_unif[-1]:.2f} (should fall)")
    _check("DISCRIMINATOR: prior confidence E[α] is LEARNED over trials "
           "(rises for informed feedback, falls for uniform)", rose and fell,
           f"Δα_informed={a_inf[-1] - a_inf[0]:+.2f}, Δα_uniform={a_unif[-1] - a_unif[0]:+.2f}")


def test_width_is_learned():
    """Width is learned too: feedback from a known-SD prior drives the believed
    prior SD toward that SD (recovered from feedback, prior_std never supplied)."""
    rng = np.random.RandomState(1)
    n = 600
    dirs = rng.randint(1, 361, n)
    cohs = rng.choice([0.06, 0.12, 0.24], n)
    obs = HBAdaptiveConfidenceObserver(lam=0.0)   # no forgetting: converge cleanly

    true_sd = 20.0
    # kappa whose circular SD ~ 20 deg
    ks = np.linspace(0.1, 60, 4000)
    true_k = ks[int(np.argmin([abs(von_mises_std(k) - true_sd) for k in ks]))]
    p = von_mises_pdfs(DIRECTION_SPACE, 225.0, true_k, normalize=True).ravel()
    fb = rng.choice(np.arange(1, 361), size=n, p=p)
    sd_traj = obs.filter(dirs, cohs, feedback=fb, record_belief=True)["believed_sd"]
    learned = float(np.mean(sd_traj[-100:]))
    print(f"    believed prior SD: {sd_traj[0]:.0f} -> {learned:.0f} deg "
          f"(true generating SD ~ {true_sd:.0f})")
    _check("believed prior WIDTH converges toward the generating SD",
           abs(learned - true_sd) < 12, f"learned SD={learned:.1f}, true={true_sd:.0f}")


def test_causal_ordering():
    """No leakage: trial-0's response distribution uses the flat initial belief
    and is therefore independent of feedback[0] (feedback is applied AFTER the
    response is read out)."""
    obs = HBAdaptiveConfidenceObserver()
    dirs = np.array([85, 200, 300]); cohs = np.array([0.06, 0.12, 0.24])
    d_a = obs.filter(dirs, cohs, feedback=np.array([10, 200, 300]))["dists"][0]
    d_b = obs.filter(dirs, cohs, feedback=np.array([350, 200, 300]))["dists"][0]
    # trial-0 dist must be identical regardless of trial-0 feedback value
    same = np.max(np.abs(d_a - d_b)) < 1e-12
    # and it must equal the flat-belief read-out for that condition
    obs._prepare(dirs, cohs)
    d_flat = obs.estimate_distribution(0.06, 85, obs.initial_belief())
    matches_flat = np.max(np.abs(d_a - d_flat)) < 1e-12
    _check("trial-0 response is pre-feedback (no within-trial leakage)",
           same and matches_flat,
           f"max|Δ(fb variants)|={np.max(np.abs(d_a - d_b)):.2e}, "
           f"max|Δ(flat)|={np.max(np.abs(d_a - d_flat)):.2e}")


def test_timing():
    try:
        from observers.helpers.dataset import load_subject_design
        d = load_subject_design(str(DATA_CSV), 1)
    except Exception as e:
        print(f"    [skip timing — could not load subject data: {e}]")
        return
    data = dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                estimates=d.estimate_dir.values.astype(int))
    obs = HBAdaptiveConfidenceObserver()
    t0 = time.time()
    nll = obs.negative_log_likelihood(data["estimates"], data["motion_direction"],
                                      data["motion_coherence"])
    dt = time.time() - t0
    n = data["estimates"].size
    print(f"    subject 1: n={n} trials, NLL={nll:.1f}, "
          f"AIC={HBAdaptiveConfidenceObserver.aic(nll):.1f}, "
          f"BIC={HBAdaptiveConfidenceObserver.bic(nll, n):.1f}")
    print(f"    one NLL eval = {dt:.2f} s (joint grid {obs.k_grid.size}×{obs.a_grid.size} "
          f"= {obs.k_grid.size * obs.a_grid.size} (κ,α) pairs).")
    _check("single full-sequence NLL eval completes", np.isfinite(nll), f"{dt:.2f}s")


def run():
    """Run every check, print a PASS/FAIL line each, and a summary.
    Returns (passed, total). Safe to call repeatedly."""
    results.clear()
    print("=== Verification: HB-Adaptive-Confidence observer (learns α AND κ) ===")
    print("\n-- T1 reduction (mixed-prior atom) --");     test_reduction()
    print("\n-- T2 normalisation --");                    test_normalisation()
    print("\n-- T3 emergence of bimodality --");          test_emergence()
    print("\n-- T4 parameter count (α,κ not fitted) --");  test_param_count()
    print("\n-- T5 DISCRIMINATOR: confidence α is learned --"); test_confidence_is_learned()
    print("\n-- T6 width κ is learned --");                test_width_is_learned()
    print("\n-- T7 causal ordering (no leakage) --");     test_causal_ordering()
    print("\n-- T8 fit-cost timing --");                  test_timing()
    print("\n" + "=" * 60)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} checks passed"
          + ("  ✅ ALL PASS" if passed == total else "  ❌ FAILURES PRESENT"))
    return passed, total


if __name__ == "__main__":
    run()
