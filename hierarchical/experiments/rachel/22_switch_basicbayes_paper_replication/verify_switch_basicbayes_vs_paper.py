"""
verify_switch_basicbayes_vs_paper.py
====================================

PAPER-REPLICATION check for the Switch and Basic-Bayesian observers.

Unlike the other verify_*.py suites (which test internal mathematical
identities: reduction limits, Eq.6 weights, valid probability vectors), this
module cross-checks our production models against an INDEPENDENT re-port of
Laquitaine & Gardner's own MATLAB reference code, translated straight from:

    reference/laquitaine_gardner_matlab/codes/assets/
        SLGirshickBayesLookupTable.m   (Girshick posterior + MAP readout)
        SLgetLoglBayesianModel.m       (lapse mixture + motor conv + NLL)
        vmPdfs.m                        (von Mises density)
        SLcircConv.m                    (circular convolution)

The re-port below imports NOTHING from observers/ -- it is a second, independent
implementation, so agreement is evidence the production code reproduces the
paper's model rather than merely agreeing with itself.

Run:  python -m observers.verification.verify_switch_basicbayes_vs_paper

Finding (see module docstring of the report): production == reference to
machine precision (max|Δ| ~ 2e-17) once a single, documented reparameterization
is accounted for -- the lapse mixture. The paper writes the guess mixture as
    (1 - p) * P  +  p * U            [SLgetLoglBayesianModel.m lines 353-354]
while the production models use the algebraically-equivalent, HB-comparable
    (P + p * U) / (1 + p)
These span the SAME one-parameter family of distributions (effective lapse
lambda = p/(1+p) vs lambda = p); since p_random is a free fitted parameter the
MLE distribution, NLL, AIC and BIC are identical -- only the reported numeric
value of p_random differs. Under the production convention the match is exact.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e

DEG = np.arange(1, 361)

# --------------------------------------------------------------------------- #
#  Independent re-port of the reference MATLAB primitives
# --------------------------------------------------------------------------- #
def _de2r(ang):
    ang = np.asarray(ang, dtype=float)
    r = (ang / 360.0) * (2 * np.pi)
    return np.where(ang > 180, (ang - 360.0) * (2 * np.pi / 360.0), r)


def vmpdfs(x, u, k, normalize=True):
    x = np.atleast_1d(np.asarray(x, float)); u = np.atleast_1d(np.asarray(u, float))
    xr = _de2r(x)[:, None]; ur = _de2r(u)[None, :]
    if np.ndim(k) == 0:
        k = float(k)
        if k > 300:
            M = np.zeros((x.size, u.size))
            for j in range(u.size):
                M[np.argmin(np.abs(x - u[j])), j] = 1.0
            return M
        M = np.exp(k * np.cos(xr - ur) - k) / (2 * np.pi * i0e(k))
    else:
        k = np.atleast_1d(np.asarray(k, float))[None, :]
        M = np.exp(k * np.cos(xr - ur) - k) / (2 * np.pi * i0e(k))
    return M / M.sum(axis=0, keepdims=True) if normalize else M


def circconv(v1, v2):
    return np.real(np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)))


def girshick_map(k_like, prior_mode, k_prior, motion_dirs, k_cardinal=0.0):
    """Port of SLGirshickBayesLookupTable.m (vonMisesPrior, MAPReadout)."""
    di = DEG.astype(float); m = DEG.astype(float); n = di.size
    motion_dirs = np.atleast_1d(np.asarray(motion_dirs)).astype(int)
    mPdfs = vmpdfs(m, motion_dirs.astype(float), k_like, True)
    like = vmpdfs(di, di, k_like, True)
    prior = np.tile(vmpdfs(di, np.array([float(prior_mode)]), k_prior, True), (1, n))
    if k_cardinal and not np.isnan(k_cardinal):
        card = vmpdfs(di, np.array([90, 180, 270, 360.]), k_cardinal, True)
        po = like * (0.25 * card.sum(axis=1, keepdims=True)) * prior
    else:
        po = like * prior
    with np.errstate(invalid="ignore", divide="ignore"):
        po = po / po.sum(axis=0, keepdims=True)
    po = np.round(po * 1e6) / 1e6
    bad = np.nonzero(np.isnan(po[0, :]))[0]
    if k_like != 0 and k_prior != 0 and bad.size:
        mir = _de2r(m[bad]); upr = _de2r(np.array([prior_mode]))[0]
        upo = mir + np.arctan2(np.sin(upr - mir), (k_like / k_prior) + np.cos(upr - mir))
        upo_deg = np.round(np.degrees(upo) % 360); upo_deg[upo_deg == 0] = 360
        kpo = np.sqrt(k_like**2 + k_prior**2 + 2 * k_prior * k_like * np.cos(upr - mir))
        for jj, col in enumerate(bad):
            po[:, col] = vmpdfs(di, np.array([upo_deg[jj]]), float(kpo[jj])).ravel()
    ppm = np.full((n, n), np.nan)
    for i in range(n):
        c = po[:, i]; modes = np.nonzero(c == c.max())[0]; ppm[i, :modes.size] = di[modes]
    mm = int(np.nanmax((~np.isnan(ppm)).sum(axis=1))); ppm = ppm[:, :mm]
    nmodes = (~np.isnan(ppm)).sum(axis=1)
    p_pm = np.tile((1.0 / nmodes)[:, None], (mm, motion_dirs.size))
    p_md = np.tile(mPdfs, (mm, 1))
    flat = ppm.T.reshape(-1); p_pd = p_pm * p_md
    v = ~np.isnan(flat); pv = flat[v].astype(int); contrib = p_pd[v, :]
    P = np.zeros((360, motion_dirs.size))
    for p in np.unique(pv):
        P[p - 1, :] = contrib[pv == p, :].sum(axis=0)
    L = np.full((360, 360), np.nan); L[:, motion_dirs - 1] = P
    with np.errstate(invalid="ignore", divide="ignore"):
        return L / np.nansum(L, axis=0, keepdims=True)


def _lapse(P, p_random, convention):
    U = np.ones(360) / 360.0
    if convention == "paper":
        return (1.0 - p_random) * P + p_random * U
    return (P + p_random * U) / (1.0 + p_random)   # production / HB-comparable


def _motor(P, k_motor):
    pmot = vmpdfs(np.arange(0, 360), np.array([0.0]), k_motor, True).ravel()
    est = np.clip(circconv(P, pmot), 1e-320, None)
    return est / est.sum()


def ref_basic_bayes(theta, coh, pl, kl, kp, k_motor, p_random, convention="prod"):
    P = np.nan_to_num(girshick_map(kl[coh], 225.0, kp[pl], [theta])[:, theta - 1])
    return _motor(_lapse(P / P.sum(), p_random, convention), k_motor)


def ref_switch(theta, coh, pl, kl, kp, k_motor, p_random, convention="prod"):
    k_e, k_p = kl[coh], kp[pl]
    ev = np.nan_to_num(girshick_map(k_e, 225.0, 0.0, [theta])[:, theta - 1])
    pr = np.nan_to_num(girshick_map(0.0, 225.0, k_p, [theta])[:, theta - 1])
    mix = (k_e / (k_e + k_p)) * ev + (k_p / (k_e + k_p)) * pr
    return _motor(_lapse(mix, p_random, convention), k_motor)


# --------------------------------------------------------------------------- #
#  The check
# --------------------------------------------------------------------------- #
def run(tol=1e-10):
    """Compare production Switch/Basic-Bayes to the independent reference.
    Returns (passed, total). Uses the production lapse convention (exact match)."""
    from observers.models.switching_observer import SwitchingObserver
    from observers.models.basic_bayesian import BasicBayesianObserver

    kl = {0.24: 8.0, 0.12: 3.0, 0.06: 1.0}
    kp = {"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7}
    km, pr = 40.0, 0.03
    sw = SwitchingObserver(k_like=kl, k_prior=kp, p_random=pr, k_motor=km)
    bb = BasicBayesianObserver(k_like=kl, k_prior=kp, p_random=pr, k_motor=km)
    conds = [(d, c, p) for d in (5, 85, 145, 225, 265, 325)
             for c in (0.06, 0.12, 0.24) for p in ("80", "40", "20", "10")]

    results = []
    for name, prod_fn, ref_fn in [
        ("Switch      == paper reference (MAP + Eq.6 mixture + motor)",
         lambda d, c, p: sw.estimate_distribution(d, c, p),
         lambda d, c, p: ref_switch(d, c, p, kl, kp, km, pr, "prod")),
        ("Basic-Bayes == paper reference (Girshick joint MAP + motor)",
         lambda d, c, p: bb.estimate_distribution(d, c, p),
         lambda d, c, p: ref_basic_bayes(d, c, p, kl, kp, km, pr, "prod")),
    ]:
        md = max(np.max(np.abs(prod_fn(*k) - ref_fn(*k))) for k in conds)
        ok = md < tol; results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  (max|Δ|={md:.2e} over {len(conds)} conditions)")

    print("=" * 60)
    passed, total = sum(results), len(results)
    print(f"{passed}/{total} paper-replication checks passed"
          + ("  ALL PASS" if passed == total else "  FAILURES"))
    return passed, total


if __name__ == "__main__":
    run()
