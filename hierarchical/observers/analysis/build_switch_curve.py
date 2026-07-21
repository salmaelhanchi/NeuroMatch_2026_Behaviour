"""
Within-block switch-probability learning curve (RT-independent).

Idea: the online model's switch weight w_prior(t) evolves within a block as the
prior is learned, whereas the static model's is constant. The switch can be
observed directly — without reaction time — on trials where the stimulus is far
from the prior, by labelling each estimate as 'prior' (near 225) or 'evidence'
(near the stimulus). Plotting the fraction of prior-chosen trials against
within-block trial index gives an empirical learning curve for the switch, which
is compared to the fitted online model's prediction.

Reads the fitted subjects from results/fits/human_fit_results.json.

Run:  python -m observers.analysis.build_switch_curve
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.helpers.dataset import load_subject_design
from observers.helpers.paths import DATA_CSV, HUMAN_FITS, FIGURES_DIR

CSV = DATA_CSV
DIRS = np.arange(1, 361)

def sdist(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def within_block_index(prior_std):
    idx = np.zeros(len(prior_std), dtype=int)
    c = 0
    for i in range(len(prior_std)):
        c = 0 if (i == 0 or prior_std[i] != prior_std[i-1]) else c + 1
        idx[i] = c
    return idx

# region masks depend on the trial's stimulus direction
def region_masks(stim):
    dprior = np.abs(sdist(DIRS, 225.0))
    dstim = np.abs(sdist(DIRS, stim))
    prior_region = (dprior < 25) & (dstim > 40)
    evid_region = (dstim < 25) & (dprior > 40)
    return prior_region, evid_region


def build(save=False):
    """Compute and plot the within-block switch-probability learning curve.

    Reads the fitted online-model parameters from human_fit_results.json, labels
    far-from-prior trials as prior- or evidence-chosen by the estimate location,
    and plots the empirical vs. model-predicted P(prior-chosen) across
    within-block position. Returns (fig, R) where R is the per-trial DataFrame.
    Set save=True to also write results/figures/switch_probability_curve.png.
    """
    fits = json.load(open(HUMAN_FITS))
    subjects = sorted(int(s) for s in fits)

    # accumulate per far-from-prior, cleanly labelled trial:
    #   within-block index, empirical label, model P(prior|labelled)
    rows = []
    for s in subjects:
        r = fits[str(s)]
        obs = OnlineHierarchicalObserver(
            k_like={0.06: r["k_like"][0], 0.12: r["k_like"][1], 0.24: r["k_like"][2]},
            k_motor=r["k_motor"], p_random=r["p_random"], lam=r["lam"])
        d = load_subject_design(CSV, s)
        dirs = d.motion_direction.values.astype(int)
        cohs = d.motion_coherence.values.astype(float)
        est = d.estimate_dir.values.astype(int)
        pstd = d.prior_std.values.astype(int)
        wbi = within_block_index(pstd)

        out = obs.filter(dirs, cohs, feedback=dirs)  # per-trial estimate dists
        dists = out["dists"]

        stim_far = np.abs(sdist(dirs, 225.0))
        for t in range(len(dirs)):
            if stim_far[t] <= 60:      # need stimulus far from prior to label
                continue
            e_p = abs(sdist(est[t], 225.0))
            e_s = abs(sdist(est[t], dirs[t]))
            if e_p < 25 and e_s > 40:
                emp = 1               # prior
            elif e_s < 25 and e_p > 40:
                emp = 0               # evidence
            else:
                continue              # ambiguous
            pr_mask, ev_mask = region_masks(dirs[t])
            p_pr = dists[t][pr_mask].sum()
            p_ev = dists[t][ev_mask].sum()
            model_p = p_pr / (p_pr + p_ev) if (p_pr + p_ev) > 0 else np.nan
            rows.append((wbi[t], emp, model_p))

    R = pd.DataFrame(rows, columns=["wbi", "emp_prior", "model_p"])
    print(f"labelled far-from-prior trials pooled over {len(subjects)} subjects: {len(R)}")

    # bin by within-block index
    edges = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 10000]
    R["bin"] = pd.cut(R.wbi, bins=edges, right=False)
    g = R.groupby("bin", observed=True)
    centers, emp, emp_lo, emp_hi, mod, ns = [], [], [], [], [], []
    for b, gg in g:
        n = len(gg)
        if n < 30:
            continue
        p = gg.emp_prior.mean()
        se = np.sqrt(p * (1 - p) / n)
        centers.append(gg.wbi.median())
        emp.append(p); emp_lo.append(max(0, p - 1.96*se)); emp_hi.append(min(1, p + 1.96*se))
        mod.append(gg.model_p.mean()); ns.append(n)

    centers = np.array(centers)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(centers, emp_lo, emp_hi, color="C0", alpha=0.2)
    ax.plot(centers, emp, "o-", color="C0", label="empirical (estimate-location labels)")
    ax.plot(centers, mod, "s--", color="C3", label="online model prediction")
    ax.set_xscale("symlog")
    ax.set_xlabel("trials since block change (within-block position)")
    ax.set_ylabel("P(prior-chosen | cleanly labelled)")
    ax.set_title("Within-block switch-probability learning curve\n"
                 f"(far-from-prior trials, {len(R)} labelled, {len(subjects)} subjects)")
    ax.legend(); ax.grid(alpha=0.3)
    for x, y, n in zip(centers, emp, ns):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center", color="0.4")
    fig.tight_layout()

    if save:
        out_png = FIGURES_DIR / "switch_probability_curve.png"
        fig.savefig(out_png, dpi=120)
        print(f"saved {out_png}")

    # --- trend test: early (first 5 within-block) vs later ---
    early = R[R.wbi < 5].emp_prior.values
    late = R[R.wbi >= 20].emp_prior.values
    print(f"\nP(prior) early (wbi<5, n={len(early)}) = {early.mean():.3f}")
    print(f"P(prior) late  (wbi>=20, n={len(late)}) = {late.mean():.3f}")
    u, p = mannwhitneyu(early, late)
    print(f"early-vs-late difference: Δ={early.mean()-late.mean():+.3f}, Mann-Whitney p={p:.4f}")
    print(f"model early={R[R.wbi<5].model_p.mean():.3f}  model late={R[R.wbi>=20].model_p.mean():.3f}")

    return fig, R


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # headless CLI: render to file, no display needed
    build(save=True)
