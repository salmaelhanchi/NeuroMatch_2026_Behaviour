"""
hb_rachel_fit.py
================

Fit the HB-Rachel observer (hb_rachel.py, HBRachelObserver) and
compare it to the switching family — the abstract's core question:

    "test whether switching-like perceptual behavior requires an explicit
     selection mechanism or can emerge from adaptive hierarchical inference."

Modes (Task 5 / Task 3 deliverables):
  recover   parameter recovery on simulated data, incl. the alpha-vs-lapse
            identifiability check.
  human N.. fit subject(s), report NLL / AIC / BIC for integration, and pull
            the static + online numbers from human_fit_results.json (adding
            BIC), so all three models are compared on the same trials.
  cv    N.. sequence-preserving K-fold cross-validation for a subject.

Usage:
  python hb_rachel_fit.py recover
  python hb_rachel_fit.py human 1 3
  python hb_rachel_fit.py cv 1
"""

from __future__ import annotations

import json, sys
import numpy as np
from scipy.optimize import minimize

from observers.models.hb_rachel import HBRachelObserver
from observers.helpers.dataset import load_subject_design, make_synthetic_design
from observers.helpers.paths import DATA_CSV, HUMAN_FITS, HB_FITS
from observers.fitting.online_recovery import conv_info as _conv_info

CSV = DATA_CSV
COHS = [0.06, 0.12, 0.24]


# --------------------------- param transforms ------------------------------
def _sig(x):   return 1.0 / (1.0 + np.exp(-x))
def _logit(p): return np.log(p / (1.0 - p))


def pack(k_like, alpha, k_motor, p_random, lam):
    return np.array([np.log(k_like[0.06]), np.log(k_like[0.12]), np.log(k_like[0.24]),
                     _logit(alpha), np.log(k_motor), _logit(p_random), _logit(lam)])


def unpack(theta) -> HBRachelObserver:
    k_like = {0.06: np.exp(theta[0]), 0.12: np.exp(theta[1]), 0.24: np.exp(theta[2])}
    return HBRachelObserver(k_like=k_like, alpha=_sig(theta[3]),
                                 k_motor=np.exp(theta[4]), p_random=_sig(theta[5]),
                                 lam=_sig(theta[6]))


# --------------------------- likelihood helpers ----------------------------
def _trial_logliks(obs: HBRachelObserver, data):
    """Per-trial log p(estimate_t | ...) with the belief filter over the FULL
    ordered sequence (so learning/order is preserved even under a CV mask)."""
    d, c, e = data["motion_direction"], data["motion_coherence"], data["estimates"]
    out = obs.filter(d, c, feedback=d, sample=False)
    ll = np.array([np.log(max(out["dists"][t][(e[t] - 1) % 360], 1e-320))
                   for t in range(len(e))])
    return ll


def nll_masked(obs, data, mask=None):
    ll = _trial_logliks(obs, data)
    return -float(ll.sum() if mask is None else ll[mask].sum())


def fit(data, x0=None, maxiter=400, mask=None):
    def obj(theta):
        try:
            v = nll_masked(unpack(theta), data, mask)
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12
    if x0 is None:
        x0 = pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.6, 30.0, 0.05, 0.05)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    obs = unpack(res.x)
    nll = float(res.fun)
    obs._fit_info = _conv_info(res, maxiter)   # convergence diagnostics
    return obs, nll, res.x


# ------------------------------ recover ------------------------------------
def _simulate(obs, design, seed):
    rng = np.random.RandomState(seed)
    d = design["motion_direction"].values.astype(int)
    c = design["motion_coherence"].values.astype(float)
    out = obs.filter(d, c, feedback=d, sample=True, rng=rng)
    return {"motion_direction": d, "motion_coherence": c, "estimates": out["responses"]}


def recover():
    print("=== parameter recovery (integration model) ===")
    truth = dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, alpha=0.6,
                 k_motor=30.0, p_random=0.03, lam=0.05)
    gen = HBRachelObserver(**truth)
    design = make_synthetic_design(trials_per_block=200, seed=1)
    rec = {k: [] for k in ["k06", "k12", "k24", "alpha", "k_motor", "p_random", "lam"]}
    for seed in (1, 2, 3):
        data = _simulate(gen, design, seed)
        obs, nll, x = fit(data, maxiter=500)
        rec["k06"].append(obs.k_like[0.06]); rec["k12"].append(obs.k_like[0.12])
        rec["k24"].append(obs.k_like[0.24]); rec["alpha"].append(obs.alpha)
        rec["k_motor"].append(obs.k_motor); rec["p_random"].append(obs.p_random)
        rec["lam"].append(obs.lam)
        print(f"  seed {seed}: alpha={obs.alpha:.3f} p_rand={obs.p_random:.3f} "
              f"lam={obs.lam:.3f} k=({obs.k_like[0.06]:.2f},{obs.k_like[0.12]:.2f},"
              f"{obs.k_like[0.24]:.2f}) k_motor={obs.k_motor:.1f}")
    tvals = dict(k06=1.0, k12=3.0, k24=8.0, alpha=0.6, k_motor=30.0,
                 p_random=0.03, lam=0.05)
    print("  --- recovered mean vs truth ---")
    ok = True
    for k in rec:
        m = float(np.mean(rec[k])); rel = abs(m - tvals[k]) / abs(tvals[k])
        tag = "ok" if rel < 0.35 else "WEAK"
        ok = ok and (k in ("lam",) or rel < 0.35)
        print(f"    {k:9s}: recovered {m:8.3f}  truth {tvals[k]:7.3f}  "
              f"rel {rel:4.2f}  [{tag}]")
    # alpha identifiability: does recovered alpha track truth without collapsing
    # into the lapse? Report the (alpha, p_random) pairs — if alpha were
    # unidentified it would wander while p_random absorbed the uniform mass.
    a = np.array(rec["alpha"]); p = np.array(rec["p_random"])
    print(f"  alpha identifiability: recovered alpha={a.mean():.3f}±{a.std():.3f} "
          f"(truth 0.6), p_random={p.mean():.3f}±{p.std():.3f} (truth 0.03)")
    print(f"  -> alpha {'CLUSTERS near truth (identifiable)' if abs(a.mean()-0.6)<0.15 and a.std()<0.12 else 'is UNSTABLE (check alpha–lapse tradeoff)'}")
    return ok


# ------------------------------ human fits ---------------------------------
def _load_subject(sid):
    d = load_subject_design(CSV, sid)
    return dict(motion_direction=d.motion_direction.values.astype(int),
                motion_coherence=d.motion_coherence.values.astype(float),
                prior_std=d.prior_std.values.astype(int),
                estimates=d.estimate_dir.values.astype(int))


def _starts_for(onl, sid):
    """Multi-start x0 list (warm start tried early so an interrupted run still
    keeps the informative fit): cold default, warm-from-online, one variant.
    Guards against the 7-dimensional Nelder-Mead simplex settling in a local
    basin, which can leave the learning-rate parameter (lambda) badly off."""
    starts = [("cold", pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.6, 30.0, 0.05, 0.05))]
    if str(sid) in onl:
        o = onl[str(sid)]
        kl = {0.06: o["k_like"][0], 0.12: o["k_like"][1], 0.24: o["k_like"][2]}
        starts.append(("warm", pack(kl, 0.6, o["k_motor"],
                                    min(max(o["p_random"], 1e-3), 0.2), 0.05)))
    starts.append(("var", pack({0.06: 1.0, 0.12: 3.0, 0.24: 8.0}, 0.75, 25.0, 0.03, 0.10)))
    return starts


def _row(obs, nll, sid, n, onl):
    """Result row + rival AIC/BIC for one subject.  BIC = k*ln(n) + 2*NLL, and
    for a rival AIC_r = 2k + 2*NLL so BIC_r = k*ln(n) + (AIC_r - 2k)."""
    aic = HBRachelObserver.aic(nll); bic = HBRachelObserver.bic(nll, n)
    row = {"subject": sid, "n_trials": n,
           "nll_integration": nll, "aic_integration": aic, "bic_integration": bic,
           "alpha": obs.alpha, "k_motor": obs.k_motor, "p_random": obs.p_random,
           "lam": obs.lam,
           "k_like": [obs.k_like[0.06], obs.k_like[0.12], obs.k_like[0.24]]}
    if str(sid) in onl:
        o = onl[str(sid)]
        row.update(aic_online=o["aic_online"], aic_static=o["aic_static"],
                   bic_online=6 * np.log(n) + (o["aic_online"] - 12),
                   bic_static=9 * np.log(n) + (o["aic_static"] - 18))
    return row


def human(subject_ids):
    onl = json.load(open(HUMAN_FITS)) if HUMAN_FITS.exists() else {}
    res = json.load(open(HB_FITS)) if HB_FITS.exists() else {}
    for sid in subject_ids:
        sid = int(sid)
        data = _load_subject(sid)
        n = data["estimates"].size
        print(f"fitting subject {sid} (n={n}, multi-start) ...", flush=True)
        best_obs, best_nll = None, np.inf
        for name, x0 in _starts_for(onl, sid):
            o_j, nll_j, _ = fit(data, x0=x0, maxiter=400)
            better = nll_j < best_nll
            print(f"    start[{name}]: NLL={nll_j:.1f} (alpha={o_j.alpha:.2f} "
                  f"lam={o_j.lam:.3f}){'  <- best so far' if better else ''}", flush=True)
            if better:
                best_obs, best_nll = o_j, nll_j
                # persist best-so-far AFTER EACH START -> a kill can't lose it.
                # Re-read + merge before writing so parallel per-subject jobs do
                # not clobber each other's rows (they share this file).
                disk = json.load(open(HB_FITS)) if HB_FITS.exists() else {}
                disk[str(sid)] = _row(best_obs, best_nll, sid, n, onl)
                json.dump(disk, open(HB_FITS, "w"), indent=2)

        row = res[str(sid)]
        rivals = [("integration", row["aic_integration"])]
        for m, disp in (("online", "online"), ("static", "static")):
            if f"aic_{m}" in row:
                rivals.append((disp, row[f"aic_{m}"]))
        bst = min(rivals, key=lambda t: t[1])
        others = "  ".join(f"{nm}={a:.1f}" for nm, a in rivals if nm != "integration")
        print(f"  subj {sid}: integration AIC={row['aic_integration']:.1f} "
              f"BIC={row['bic_integration']:.1f} (alpha={best_obs.alpha:.2f}, "
              f"lam={best_obs.lam:.3f})  | {others} -> best={bst[0]} "
              f"(ΔAIC vs best={row['aic_integration'] - bst[1]:+.1f})", flush=True)


# ------------------- sequence-preserving cross-validation ------------------
def cv(subject_ids, K=5):
    """K-fold CV that never shuffles trials: folds are CONTIGUOUS segments, and
    the belief filter always runs over the full ordered sequence, so the
    sequential learning is preserved. We fit on the train mask and score
    held-out predictive NLL per trial on the test mask.

    Caveat: because feedback (the true direction) is available on every trial
    regardless of the mask, the belief still 'sees' held-out trials' feedback;
    this tests predictive fit of the RESPONSES with the order intact, not a
    strictly causal forecast. Documented as the fork it is.
    """
    for sid in subject_ids:
        sid = int(sid)
        data = _load_subject(sid)
        n = data["estimates"].size
        folds = np.array_split(np.arange(n), K)
        tot_test = 0.0
        print(f"subject {sid}: {K}-fold sequence-preserving CV (n={n})")
        for f, test_idx in enumerate(folds):
            test = np.zeros(n, dtype=bool); test[test_idx] = True
            train = ~test
            obs, _, x = fit(data, maxiter=400, mask=train)
            test_nll = nll_masked(obs, data, mask=test)
            tot_test += test_nll
            print(f"  fold {f+1}/{K}: test trials={test.sum():4d}  "
                  f"held-out NLL={test_nll:8.1f}  "
                  f"(per-trial {test_nll/test.sum():.3f})")
        print(f"  total held-out NLL = {tot_test:.1f}  "
              f"(per-trial {tot_test/n:.3f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "recover"
    if mode == "recover":
        recover()
    elif mode == "human":
        human(sys.argv[2:])
    elif mode == "cv":
        cv(sys.argv[2:])
    else:
        print(__doc__)
