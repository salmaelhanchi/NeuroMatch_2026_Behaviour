"""
run_replication.py — regenerate the Switch / Basic-Bayes paper-replication result.

Runs the 72-condition comparison of our production models against an independent
NumPy re-port of Laquitaine & Gardner's MATLAB reference (see
verify_switch_basicbayes_vs_paper.py, which this reuses), under BOTH lapse
conventions, and writes:

    replication_maxdiff.csv   per-(model, lapse-convention) max|Δ|
    replication.png           visual proof of the match

Run from the repo's hierarchical/ folder:
    PYTHONPATH=. python experiments/rachel/22_switch_basicbayes_paper_replication/run_replication.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the independent reference re-port lives beside this script AND in
# observers/verification/; import whichever is importable.
try:
    from observers.verification.verify_switch_basicbayes_vs_paper import (
        ref_switch, ref_basic_bayes, girshick_map)
except Exception:
    import importlib.util, sys
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        "_ref", os.path.join(here, "verify_switch_basicbayes_vs_paper.py"))
    _ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(_ref)
    ref_switch, ref_basic_bayes, girshick_map = (
        _ref.ref_switch, _ref.ref_basic_bayes, _ref.girshick_map)

from observers.models.switching_observer import SwitchingObserver
from observers.models.basic_bayesian import BasicBayesianObserver
from observers.helpers.bayes_lookup import girshick_map_lookup as prod_lookup

HERE = os.path.dirname(os.path.abspath(__file__))
KL = {0.24: 8.0, 0.12: 3.0, 0.06: 1.0}
KP = {"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7}
KM, PR = 40.0, 0.03
CONDS = [(d, c, p) for d in (5, 85, 145, 225, 265, 325)
         for c in (0.06, 0.12, 0.24) for p in ("80", "40", "20", "10")]


def compare():
    sw = SwitchingObserver(k_like=KL, k_prior=KP, p_random=PR, k_motor=KM)
    bb = BasicBayesianObserver(k_like=KL, k_prior=KP, p_random=PR, k_motor=KM)
    rows = []
    # shared Girshick engine (no lapse/motor): bit-exact check
    eng = max(np.max(np.abs(
        np.nan_to_num(prod_lookup(k_like=KL[c], prior_mode=225.0, k_prior=KP[p],
                                  motion_dirs=np.array([d]), k_cardinal=0.0)[:, d - 1])
        - np.nan_to_num(girshick_map(KL[c], 225.0, KP[p], [d])[:, d - 1])))
        for d, c, p in CONDS)
    rows.append(dict(component="Girshick MAP engine (shared)", lapse="n/a", max_abs_diff=eng))
    for lap in ("prod", "paper"):
        md_sw = max(np.max(np.abs(sw.estimate_distribution(*k)
                                  - ref_switch(*k, KL, KP, KM, PR, lap))) for k in CONDS)
        md_bb = max(np.max(np.abs(bb.estimate_distribution(*k)
                                  - ref_basic_bayes(*k, KL, KP, KM, PR, lap))) for k in CONDS)
        rows.append(dict(component="Switch full distribution", lapse=lap, max_abs_diff=md_sw))
        rows.append(dict(component="Basic-Bayes full distribution", lapse=lap, max_abs_diff=md_bb))
    return sw, bb, pd.DataFrame(rows)


def figure(sw, bb, df, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(1, 361)
    # (a) Switch overlay, (b) Basic-Bayes overlay for one telling condition
    d, c, p = 85, 0.06, "80"   # low coherence, wide prior: bimodal switch vs unimodal bayes
    for ax, obs, ref_fn, name, col in [
        (axes[0], sw, ref_switch, "Switch", "#30638e"),
        (axes[1], bb, ref_basic_bayes, "Basic-Bayes", "#a0a0a0")]:
        prod = obs.estimate_distribution(d, c, p)
        ref = ref_fn(d, c, p, KL, KP, KM, PR, "prod")
        ax.plot(x, prod, color=col, lw=3.0, label="our implementation")
        ax.plot(x, ref, color="k", lw=1.0, ls=(0, (4, 3)), label="paper reference (re-port)")
        ax.axvline(225, color="#c17817", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(f"{name}: P(estimate) | dir={d}, coh={c}, prior={p}", fontsize=9)
        ax.set_xlabel("estimate (deg)"); ax.set_xlim(1, 360)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("probability")
    # (c) max|diff| per component (log scale), production convention
    d2 = df[df.lapse.isin(["prod", "n/a"])].copy()
    labels = ["Girshick\nengine", "Switch\n(full)", "Basic-Bayes\n(full)"]
    vals = [d2[d2.component.str.startswith("Girshick")].max_abs_diff.iloc[0],
            d2[d2.component.str.startswith("Switch")].max_abs_diff.iloc[0],
            d2[d2.component.str.startswith("Basic")].max_abs_diff.iloc[0]]
    vals = [max(v, 1e-18) for v in vals]
    axes[2].bar(range(3), vals, color=["#7a7a7a", "#30638e", "#a0a0a0"])
    axes[2].axhline(1e-10, color="crimson", lw=1.0, ls="--", label="tol 1e-10")
    axes[2].set_yscale("log"); axes[2].set_ylim(1e-18, 1e-3)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("max |Δ| vs reference"); axes[2].set_title("Agreement (72 conditions)", fontsize=9)
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    sw, bb, df = compare()
    df.to_csv(os.path.join(HERE, "replication_maxdiff.csv"), index=False)
    figure(sw, bb, df, os.path.join(HERE, "replication.png"))
    print(df.to_string(index=False))
    prod_max = df[df.lapse.isin(["prod", "n/a"])].max_abs_diff.max()
    print(f"\nProduction-convention max|Δ| across all components: {prod_max:.2e}")
    print("PASS" if prod_max < 1e-10 else "FAIL")


if __name__ == "__main__":
    main()
