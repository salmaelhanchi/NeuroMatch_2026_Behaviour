"""
verify_hb_salma.py
==================

Verification for **HB-Salma** (``hb_salma.py``, ``HBSalmaObserver``) — a
house-API adapter over the vendored ``salma_hierarchical_helpers`` package. It
is the most structurally distinct model in the registry:

  * parameters are ``rho`` (confidence-update rate), three ``sensory_kappas``
    (per coherence), ``motor_kappa`` and ``lapse`` — 6 fitted, NO α;
  * it is natively a **72-bin (5°)** subject-batch model, wrapped to the
    house per-trial ``filter(directions, coherences, feedback)`` API;
  * confidence is forgotten GEOMETRICALLY (via ``rho``), not with Rachel's
    linear leak; and prior_std is NOT a model input (hardcoded to 80 in the
    adapter — the width is recovered from feedback).

Because NLL on 72 bins is not comparable to the other observers' 360-bin NLL,
the adapter exposes both ``grid="native"`` (72-bin, reproduces the published
HB-Salma numbers) and ``grid="deg360"`` (up-sampled, cross-model comparable).
Getting that contract right is the thing most likely to break silently, so it
is checked explicitly (T4, T5).

Checks
------
T1  Parameters: exposes rho / sensory_kappas / motor_kappa / lapse, no α;
    registry k == 6, learns == True.
T2  Native PMFs live on 72 bins and are proper; deg360 PMFs are length-360,
    proper, and sum to 1.
T3  Up-sampling conserves probability: each 5° native bin's mass is spread over
    its 5 degrees, so the deg360 pmf sums to 1 and its coarse-binned marginal
    matches the native pmf.
T4  Grid contract: native NLL and deg360 NLL are BOTH finite, DIFFER (different
    grids), and native is the lower of the two (coarser grid concentrates mass)
    — the documented reason the two must never be mixed in one AIC.
T5  Sensory tuning: the read-out is a tie-aware MAP percept convolved with the
    motor kernel, so response *width* is set by motor_kappa, not the sensory
    kappa. The sensory kappa instead sets how far the percept is pulled OFF the
    prior toward an off-prior stimulus: with low prior confidence, raising the
    sensory kappa moves the MAP mode from near the prior (225) to the stimulus.
T6  Confidence learning: feedback clustered at 225 drives the believed prior SD
    DOWN over trials (the geometric-forget confidence trajectory), and rho
    controls how fast — larger rho tracks faster. prior_std is never supplied.
T7  Cost: one native NLL eval on a real subject.
"""

from __future__ import annotations

import time
import numpy as np

from observers.models.hb_salma import HBSalmaObserver, _pmf72_to_deg360, N_PARAMS
from observers.helpers.circular import von_mises_pdfs, DIRECTION_SPACE
from observers.helpers.paths import DATA_CSV

results = []


def _check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _sdist(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _circ_sd_deg(pmf360):
    """Circular SD (deg) of a length-360 pmf on directions 1..360."""
    th = np.deg2rad(np.arange(1, 361))
    R = abs((pmf360 * np.exp(1j * th)).sum() / pmf360.sum())
    return float(np.rad2deg(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))))


def _synth(n, seed=0, prior_k=8.0):
    """A short synthetic session: random directions/coherences, feedback drawn
    from a tight prior at 225 (so confidence should grow)."""
    rng = np.random.RandomState(seed)
    dirs = rng.randint(1, 361, n)
    cohs = rng.choice([0.06, 0.12, 0.24], n)
    p = von_mises_pdfs(DIRECTION_SPACE, 225.0, prior_k, normalize=True).ravel()
    fb = rng.choice(np.arange(1, 361), size=n, p=p)
    return dirs, cohs, fb


# ---------------------------------------------------------------------------
def test_parameters():
    obs = HBSalmaObserver()
    has = all(hasattr(obs, a) for a in ("rho", "sensory_kappas", "motor_kappa", "lapse"))
    _check("exposes rho / sensory_kappas / motor_kappa / lapse", has)
    _check("has NO α parameter (unlike the HB-Rachel family)", not hasattr(obs, "alpha"))
    _check("N_PARAMS == 6", N_PARAMS == 6, f"N_PARAMS={N_PARAMS}")
    _check("3 sensory kappas (one per coherence)", len(obs.sensory_kappas) == 3,
           f"sensory_kappas={obs.sensory_kappas}")
    from observers.comparison.registry import build_registry
    spec = build_registry(["hb_salma"])["hb_salma"]
    _check("registry k == 6, learns == True", spec.n_params == 6 and spec.learns is True,
           f"k={spec.n_params}, learns={spec.learns}")


def test_grids_valid():
    obs = HBSalmaObserver()
    dirs, cohs, fb = _synth(40, seed=1)
    nat = obs.filter(dirs, cohs, feedback=fb, grid="native")["dists"]
    deg = obs.filter(dirs, cohs, feedback=fb, grid="deg360")["dists"]
    _check("native PMFs are 72-bin", nat[0].size == 72, f"len={nat[0].size}")
    _check("deg360 PMFs are length-360", deg[0].size == 360, f"len={deg[0].size}")
    worst_n = max(abs(d.sum() - 1.0) for d in nat)
    worst_d = max(abs(d.sum() - 1.0) for d in deg)
    _check("all native PMFs sum to 1", worst_n < 1e-9, f"max|sum-1|={worst_n:.2e}")
    _check("all deg360 PMFs sum to 1", worst_d < 1e-9, f"max|sum-1|={worst_d:.2e}")


def test_upsample_conserves_mass():
    """The 72->360 up-sample must conserve probability and, re-binned to 72,
    return the native pmf (each 5° bin's mass / 5 over its degrees)."""
    obs = HBSalmaObserver()
    dirs, cohs, fb = _synth(30, seed=2)
    nat = obs.filter(dirs, cohs, feedback=fb, grid="native")["dists"]
    d360 = _pmf72_to_deg360(nat[-1], 72)
    _check("up-sampled pmf sums to 1", abs(d360.sum() - 1.0) < 1e-9,
           f"sum={d360.sum():.6f}")
    # re-bin the 360 back to 72 using the SAME degree indexing the up-sampler
    # uses (directions 1..360, nearest 5° bin centre).
    step = 360.0 / 72
    deg = np.arange(1, 361)
    nb = np.floor(((deg % 360) + step / 2.0) / step).astype(int) % 72
    rebinned = np.array([d360[nb == b].sum() for b in range(72)])
    d = np.max(np.abs(rebinned - nat[-1]))
    _check("re-binning deg360 back to 72 returns the native pmf", d < 1e-9,
           f"max|Δ|={d:.2e}")


def test_grid_contract():
    """Native and deg360 NLL are both finite, differ, and native < deg360
    (a coarser grid concentrates mass into fewer bins -> lower NLL). This is the
    documented reason the two grids must never be mixed in one AIC."""
    obs = HBSalmaObserver()
    rng = np.random.RandomState(3)
    dirs, cohs, fb = _synth(200, seed=3)
    est = np.array([int(rng.choice(np.arange(1, 361))) for _ in range(dirs.size)])
    nll_native = obs.negative_log_likelihood(est, dirs, cohs, feedback=fb, grid="native")
    nll_deg360 = obs.negative_log_likelihood(est, dirs, cohs, feedback=fb, grid="deg360")
    print(f"    NLL native (72-bin) = {nll_native:.1f}   NLL deg360 = {nll_deg360:.1f}")
    both_finite = np.isfinite(nll_native) and np.isfinite(nll_deg360)
    _check("native and deg360 NLL are both finite and DIFFER", both_finite
           and abs(nll_native - nll_deg360) > 1.0,
           f"native={nll_native:.1f}, deg360={nll_deg360:.1f}")
    _check("native NLL < deg360 NLL (coarser grid concentrates mass)",
           nll_native < nll_deg360, f"native={nll_native:.1f} vs deg360={nll_deg360:.1f}")


def test_sensory_tuning():
    """The MAP+motor read-out sets response width by motor_kappa; the sensory
    kappa controls how far the percept is pulled off the prior. Under LOW prior
    confidence (uniform feedback), raising the 0.24-coherence sensory kappa moves
    the MAP mode for an off-prior stimulus (135°) from near the prior (225°)
    toward the stimulus."""
    rng = np.random.RandomState(0)
    n = 150
    dirs = rng.randint(1, 361, n)
    cohs = np.full(n, 0.24)
    fb = rng.randint(1, 361, n)   # uniform feedback -> low learned prior confidence
    stim = 135
    modes = []
    for sk in [(0.5, 1.0, 2.0), (2.0, 8.0, 40.0), (8.0, 40.0, 300.0)]:
        obs = HBSalmaObserver(sensory_kappas=sk, lapse=0.0, rho=0.5)
        d = obs.estimate_distribution(0.24, stim, feedback_history=list(fb) + [stim],
                                      grid="deg360")
        modes.append(int(np.argmax(d)) + 1)
    to_stim = [abs(_sdist(m, stim)) for m in modes]
    print(f"    MAP mode by sensory kappa (stim=135, prior=225): {modes}  "
          f"(dist-to-stim {[round(x) for x in to_stim]})")
    _check("higher sensory kappa pulls the percept off the prior toward the stimulus",
           to_stim[0] > to_stim[1] > to_stim[2] and to_stim[2] < 15,
           f"dist-to-stim={[round(x) for x in to_stim]}")


def test_confidence_learning():
    """Feedback clustered at 225 drives believed prior SD DOWN; larger rho
    tracks faster. prior_std is never supplied — width is recovered from
    feedback via the geometric-forget confidence trajectory."""
    dirs, cohs, fb = _synth(400, seed=4, prior_k=16.0)   # tight prior feedback
    sd_fast = HBSalmaObserver(rho=0.95).filter(
        dirs, cohs, feedback=fb, record_belief=True)["believed_sd"]
    sd_slow = HBSalmaObserver(rho=0.60).filter(
        dirs, cohs, feedback=fb, record_belief=True)["believed_sd"]
    print(f"    believed prior SD (rho=0.95): {sd_fast[0]:.0f} -> {sd_fast[-1]:.0f} deg")
    print(f"    believed prior SD (rho=0.60): {sd_slow[0]:.0f} -> {sd_slow[-1]:.0f} deg")
    declined = sd_fast[-1] < sd_fast[0] - 3
    faster = sd_fast[-1] <= sd_slow[-1] + 1e-6
    _check("believed prior SD declines under tight-prior feedback (width learned)",
           declined, f"Δ={sd_fast[-1] - sd_fast[0]:+.1f} deg (rho=0.95)")
    _check("larger rho tracks confidence at least as fast",
           faster, f"final SD rho0.95={sd_fast[-1]:.1f} <= rho0.60={sd_slow[-1]:.1f}")


def test_timing():
    try:
        from observers.helpers.dataset import load_subject_design
        d = load_subject_design(str(DATA_CSV), 1)
    except Exception as e:
        print(f"    [skip timing — could not load subject data: {e}]")
        return
    dirs = d.motion_direction.values.astype(int)
    cohs = d.motion_coherence.values.astype(float)
    est = d.estimate_dir.values.astype(int)
    obs = HBSalmaObserver()
    t0 = time.time()
    nll = obs.negative_log_likelihood(est, dirs, cohs, grid="native")
    dt = time.time() - t0
    print(f"    subject 1: n={est.size} trials, native NLL={nll:.1f} (one eval {dt:.2f}s)")
    _check("single native NLL eval completes", np.isfinite(nll), f"{dt:.2f}s")


def run():
    """Run every check; return (passed, total). Safe to call repeatedly."""
    results.clear()
    print("=== Verification: HB-Salma observer (72-bin, geometric-forget) ===")
    print("\n-- T1 parameters (rho/kappas/motor/lapse, no α) --"); test_parameters()
    print("\n-- T2 grids valid (72-bin native, 360 deg) --");      test_grids_valid()
    print("\n-- T3 up-sample conserves mass --");                  test_upsample_conserves_mass()
    print("\n-- T4 grid contract (native vs deg360 NLL) --");      test_grid_contract()
    print("\n-- T5 sensory tuning (coherence -> tightness) --");   test_sensory_tuning()
    print("\n-- T6 confidence learning (rho) --");                 test_confidence_learning()
    print("\n-- T7 fit-cost timing --");                           test_timing()
    print("\n" + "=" * 60)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} checks passed"
          + ("  ✅ ALL PASS" if passed == total else "  ❌ FAILURES PRESENT"))
    return passed, total


if __name__ == "__main__":
    run()
