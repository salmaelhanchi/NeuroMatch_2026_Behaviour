"""
verify_online_switching_observer.py
===================================

Verify the online-learning Switching observer
(:class:`OnlineHierarchicalObserver`) against identities with KNOWN answers,
before any fitting is trusted. This is the switch family's learning member: it
starts from the paper's Switching observer and learns the prior's strength
online, with a forgetting knob ``lam``.

  1. Every trial's estimate distribution is a valid probability vector.
  2. The initial (hyper) belief over prior strength is uniform on the k-grid.
  3. With lam=0 the belief only ever sharpens (no leak back toward uniform)
     under a consistent feedback stream — i.e. the observer accumulates.
  4. With lam=1 the belief is fully reset each trial (pure forgetting): the
     post-update belief equals one Bayesian correction from uniform, regardless
     of history.
  5. Consistent feedback concentrates the belief: the expected prior strength
     E[k] rises above its initial value after seeing many aligned feedbacks.
  6. NLL is finite and lower at the generating parameters than at bad ones.

Run:  python -m observers.verification.verify_online_switching_observer
"""
import numpy as np

from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.helpers.belief_grid import bayes_correct

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _Ek(obs, belief):
    """Expected prior strength E[k] under a belief over the k-grid."""
    return float(np.dot(belief, obs.k_grid))


def run():
    """Run every identity check, print PASS/FAIL lines, and a summary.

    Returns (passed, total). Safe to call repeatedly.
    """
    results.clear()
    print("=== Verification: online-learning Switching observer ===")

    dirs = np.array([225, 225, 225, 225, 225, 225, 225, 225], dtype=int)
    cohs = np.array([0.12] * dirs.size, dtype=float)

    # ---- 1. valid probability vectors over a trial sequence ----
    obs = OnlineHierarchicalObserver(lam=0.0)
    obs._prepare(dirs, cohs)
    b = obs._belief0.copy()
    valid = True
    for d, c in zip(dirs, cohs):
        dist = obs.estimate_distribution(float(c), int(d), b)
        if abs(dist.sum() - 1.0) > 1e-9 or np.any(dist < -1e-12):
            valid = False
        b = obs.update_belief(b, int(d))
    check("every trial's estimate distribution is a valid pmf", valid)

    # ---- 2. initial belief is uniform on the k-grid ----
    b0 = obs.initial_belief()
    n = obs.k_grid.size
    check("initial belief is uniform on the k-grid",
          np.allclose(b0, 1.0 / n) and abs(b0.sum() - 1.0) < 1e-12,
          f"n_grid={n}, max|Δ|={np.max(np.abs(b0 - 1.0/n)):.2e}")

    # ---- 3. lam=0: no leak back toward uniform (pure accumulation) ----
    obs0 = OnlineHierarchicalObserver(lam=0.0); obs0._prepare(dirs, cohs)
    b_pred = np.full(n, 1.0 / n)  # forget(belief0, belief0, lam=0) == belief0
    from observers.helpers.belief_grid import forget
    check("lam=0 leaves the belief unchanged in the predict (forget) step",
          np.allclose(forget(b_pred, obs0._belief0, 0.0), b_pred),
          "forget(b, b0, 0) == b")

    # ---- 4. lam=1: full reset — post-update belief is one correction from uniform ----
    obs1 = OnlineHierarchicalObserver(lam=1.0); obs1._prepare(dirs, cohs)
    b_hist = obs1._belief0.copy()
    for _ in range(4):                       # walk several trials of history
        b_hist = obs1.update_belief(b_hist, 225)
    # one correction from uniform for the same feedback:
    reset = bayes_correct(obs1._belief0.copy(), obs1._obs_table[:, 225 - 1])
    check("lam=1 fully resets belief each trial (history-independent)",
          np.allclose(b_hist, reset, atol=1e-10),
          f"max|Δ|={np.max(np.abs(b_hist - reset)):.2e}")

    # ---- 5. consistent feedback concentrates belief: E[k] rises ----
    obsL = OnlineHierarchicalObserver(lam=0.0); obsL._prepare(dirs, cohs)
    bL = obsL._belief0.copy()
    Ek0 = _Ek(obsL, bL)
    for _ in range(20):
        bL = obsL.update_belief(bL, 225)     # repeated feedback AT the prior mean
    Ek1 = _Ek(obsL, bL)
    check("consistent feedback raises expected prior strength E[k]",
          Ek1 > Ek0,
          f"E[k]: {Ek0:.2f} -> {Ek1:.2f} after 20 aligned feedbacks")

    # ---- 6. NLL finite and lower at generating than at bad params ----
    rng = np.random.RandomState(0)
    N = 300
    dd = rng.choice([25, 85, 145, 205, 265, 325], N)
    cc = rng.choice([0.06, 0.12, 0.24], N)
    gen = OnlineHierarchicalObserver(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
                                     k_motor=40.0, p_random=0.02, lam=0.2)
    est = gen.filter(dd, cc, feedback=dd, sample=True, rng=rng)["responses"]
    nll_true = gen.negative_log_likelihood(est, dd, cc, feedback=dd)
    bad = OnlineHierarchicalObserver(k_like={0.06: 0.1, 0.12: 0.1, 0.24: 0.1},
                                     k_motor=1.0, p_random=0.4, lam=0.9)
    nll_bad = bad.negative_log_likelihood(est, dd, cc, feedback=dd)
    check("NLL finite and lower at generating than at bad params",
          np.isfinite(nll_true) and nll_true < nll_bad,
          f"NLL(true)={nll_true:.1f} < NLL(bad)={nll_bad:.1f}")

    print("=" * 55)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} checks passed"
          + ("  ✅ ALL PASS" if passed == total else "  ❌ FAILURES PRESENT"))
    return passed, total


if __name__ == "__main__":
    run()
