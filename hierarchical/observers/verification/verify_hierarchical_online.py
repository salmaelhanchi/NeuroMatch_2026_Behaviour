"""
verify_hierarchical_online.py
=============================

The build spec's §8 sanity checks for the Hierarchical Online observer, as a
runnable test. All must pass for the ported model to be numerically faithful.

    python -m observers.verification.verify_hierarchical_online

Checks
------
1. readout 'select'  ==  girshick MAP     (weight_tail = 1 - pi)
2. readout 'sample'  ==  sampling_lookup  (weight_tail = 1 - pi)
3. readout 'average' ==  girshick BLS
4. the sequential (online) log-likelihood is finite on a real subject
5. alpha -> 0 with fixed R0 collapses to the STATIC hierarchical readout
   (exact at the model's own discretisation of the prior mode)
"""
import numpy as np

from observers.models import hierarchical_online as HO
from observers.fitting import hierarchical_online_fit as F
from observers.comparison.registry import load_subject


def main():
    md = np.arange(5, 356, 10); pi = 0.6
    ok = True

    c1 = np.allclose(HO.hierarchical_lookup(md, 3., 2., pi, 225, 'select'),
                     HO.girshick_lookup(md, 3., 225, 2., weight_tail=1 - pi, readout='MAP'))
    c2 = np.allclose(HO.hierarchical_lookup(md, 3., 2., pi, 225, 'sample'),
                     HO.sampling_lookup(md, 3., 225, 2., weight_tail=1 - pi))
    c3 = np.allclose(HO.hierarchical_lookup(md, 3., 2., pi, 225, 'average'),
                     HO.girshick_lookup(md, 3., 225, 2., weight_tail=1 - pi, readout='BLS'))
    print(f"[1] select  == girshick MAP : {c1}")
    print(f"[2] sample  == sampling     : {c2}")
    print(f"[3] average == girshick BLS : {c3}")
    ok &= c1 and c2 and c3

    data = load_subject(1)
    p = dict(k_llh={0.06: 1., 0.12: 3., 0.24: 8.}, pi=0.6, p_rand=0.03,
             k_motor=30., alpha=0.05, R0=0.2, mode_init=225)
    nll = F.online_negll(data, p)
    c4 = np.isfinite(nll)
    print(f"[4] sequential LL finite    : {c4}  (negLL={nll:.1f})")
    ok &= bool(c4)

    # alpha -> 0: prior frozen at (225, k0); compare to a static readout at the
    # SAME discretised mode the model's cache key uses.
    R0 = 0.2; k0 = R0 * (2 - R0 * R0) / (1 - R0 * R0)
    p0 = dict(k_llh={0.06: 1., 0.12: 3., 0.24: 8.}, pi=0.6, p_rand=0.03,
              k_motor=30., alpha=0.0, R0=R0, mode_init=225)
    dirs = data["motion_direction"].astype(float); cohs = data["motion_coherence"]
    out = HO.replay_dists(dirs, cohs, dirs, p0, readout='sample', session=None)
    udirs = np.array(sorted(np.unique(dirs))); didx = {int(d): i for i, d in enumerate(udirs)}
    ker = HO.vm_pdf(HO.DEG, 360, p0['k_motor'])[:, 0]
    mode_disc = round(225 / 4) * 4
    cache = {}
    def static_dist(coh, d):
        M = cache.get(coh)
        if M is None:
            L = HO.hierarchical_lookup(udirs, p0['k_llh'][coh], k0, p0['pi'], mode_disc, 'sample')
            L = (1 - p0['p_rand']) * L + p0['p_rand'] / 360; L = HO.circ_convolve(L, ker)
            M = np.clip(L / L.sum(0, keepdims=True), 1e-320, None); cache[coh] = M
        return M[:, didx[int(d)]]
    maxd = max(np.abs(out["dists"][t] - static_dist(cohs[t], dirs[t])).max()
               for t in range(0, len(dirs), 137))
    c5 = maxd < 1e-12
    print(f"[5] alpha->0 static limit   : {c5}  (max diff {maxd:.2e})")
    ok &= bool(c5)

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
