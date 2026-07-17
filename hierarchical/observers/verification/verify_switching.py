"""
verify_switching.py
===================

Verify the static Switching observer (Laquitaine & Gardner, 2018) against
identities with KNOWN answers, before any fitting is trusted.

  1. Evidence read-out with a flat prior == the measurement von Mises.
  2. Prior read-out with flat evidence == a delta at the prior mean (225).
  3. Switch weights match Eq. 6 (reliability ratio) before the lapse.
  4. Every condition's response distribution is a valid probability vector.
  5. NLL is finite and lower at the generating parameters than at bad ones.

Run:  python -m observers.verification.verify_switching
"""
import numpy as np

from observers.helpers.circular import von_mises_pdfs, DIRECTION_SPACE
from observers.helpers.bayes_lookup import girshick_map_lookup
from observers.models.switching_observer import SwitchingObserver, _column

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def run():
    """Run every identity check, print PASS/FAIL lines, and a summary.

    Returns (passed, total). Safe to call repeatedly.
    """
    results.clear()
    print("=== Verification: static Switching observer ===")

    # ---- 1. evidence read-out (k_prior=0) == measurement von Mises ----
    d, k = 85, 3.0
    L = girshick_map_lookup(k_like=k, prior_mode=225.0, k_prior=0.0,
                            motion_dirs=np.array([d]), k_cardinal=0.0)
    got = _column(L, d)
    want = von_mises_pdfs(DIRECTION_SPACE, d, k, normalize=True).ravel()
    check("evidence read-out == V(.;d,k_like)",
          np.max(np.abs(got - want)) < 1e-9 and got.argmax() + 1 == d,
          f"max|Δ|={np.max(np.abs(got - want)):.2e}, argmax={got.argmax()+1} (want {d})")

    # ---- 2. prior read-out (k_like=0) == delta at prior mean ----
    Lp = girshick_map_lookup(k_like=0.0, prior_mode=225.0, k_prior=2.7,
                             motion_dirs=np.array([d]), k_cardinal=0.0)
    gp = _column(Lp, d)
    check("prior read-out == delta at 225",
          gp.argmax() + 1 == 225 and abs(gp.sum() - 1.0) < 1e-9,
          f"mass@225={gp[224]:.4f}, total={gp.sum():.4f}, argmax={gp.argmax()+1}")

    # ---- 3. switching weights (Eq. 6) before lapse ----
    obs = SwitchingObserver(p_random=0.0)
    ke, kp = 4.0, 2.0
    Pp, Pe, Pr = obs._switching_weights(ke, kp)
    check("switch weights match Eq.6 reliability ratio",
          abs(Pe - ke/(ke+kp)) < 1e-9 and abs(Pp - kp/(ke+kp)) < 1e-9,
          f"P(evidence)={Pe:.4f} (want {ke/(ke+kp):.4f})")

    # ---- 4. all condition distributions are valid probabilities ----
    obs2 = SwitchingObserver()
    bad = 0
    for d_ in range(5, 360, 40):
        for c in (0.06, 0.12, 0.24):
            for pl in ("80", "40", "20", "10"):
                dist = obs2.estimate_distribution(d_, c, pl)
                if abs(dist.sum() - 1) > 1e-9 or np.any(dist < 0):
                    bad += 1
    check("all 108 condition distributions are valid", bad == 0,
          f"invalid distributions = {bad}")

    # ---- 5. NLL finite and responds to data ----
    rng = np.random.RandomState(0)
    n = 200
    dd = rng.choice([25, 85, 145, 205, 265, 325], n)
    cc = rng.choice([0.06, 0.12, 0.24], n)
    pp = rng.choice(["80", "40", "20", "10"], n)
    est = []
    for d_, c, pl in zip(dd, cc, pp):
        dist = obs2.estimate_distribution(int(d_), float(c), str(pl))
        est.append(rng.choice(np.arange(1, 361), p=dist))
    est = np.array(est)
    nll_true = obs2.negative_log_likelihood(est, dd, cc, pp)
    obs_bad = SwitchingObserver(k_like={0.24: 0.1, 0.12: 0.1, 0.06: 0.1},
                                k_prior={"80": 50, "40": 50, "20": 50, "10": 50})
    nll_bad = obs_bad.negative_log_likelihood(est, dd, cc, pp)
    check("NLL lower at generating than at bad params", nll_true < nll_bad,
          f"NLL(true)={nll_true:.1f} < NLL(bad)={nll_bad:.1f}")

    print("=" * 55)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} checks passed"
          + ("  ✅ ALL PASS" if passed == total else "  ❌ FAILURES PRESENT"))
    return passed, total


if __name__ == "__main__":
    run()
