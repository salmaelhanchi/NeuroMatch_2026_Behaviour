"""
bimodality_test.py — pre-fit screen for bimodal-recovery capability
===================================================================

Bimodality is the abstract's central qualitative claim: in the far-from-prior,
low-coherence, wide-prior regime, human responses split into TWO clusters — one
near the stimulus, one at the 225 deg prior. Matching the *total* prior-cluster
mass is necessary but NOT sufficient: a single broad hump centred between
stimulus and prior has the same mass but no valley. This screen measures the
thing that matters — whether the predicted distribution actually has two
separated peaks — and it does so BEFORE the expensive fits, so a model that
architecturally cannot split is caught early.

It runs each registered model at MATCHED, UNFITTED parameters (same k_like,
k_motor, lapse, volatility) on the real stimulus sequence of a subject. Matched
params isolate *architecture* — combination rule (integrate-before vs -after),
alpha treatment (fixed / learned / none), grid resolution — from fit quality,
which is what a capability screen should test.

Metric: **valley-depth** between the stimulus mode and the prior mode,

    valley_depth = 1 - min_between_modes / min(height_stim, height_prior)

1.0 = two clean separated peaks (bimodal); ~0 = a single filled hump. Computed
per far stimulus direction (single direction, never pooled across directions,
which would manufacture apparent bimodality), then averaged. The observed data
gets the same metric as the reference target.

Usage:
  python -m observers.comparison.bimodality_test --subject 1
  python -m observers.comparison.bimodality_test --subject 1 --models switch hb_adaptive recombined hb_salma
  python -m observers.comparison.bimodality_test --subject 1 --plot
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from observers.comparison.registry import build_registry, load_subject
from observers.helpers.paths import FITS_DIR, FIGURES_DIR

OUT_DIR = FITS_DIR / "comparison_bimodality"
PRIOR_MEAN = 225

# matched, unfitted parameters — the same reliability regime for every model so
# only architecture differs.
MATCHED = dict(k_like={0.06: 1.0, 0.12: 3.0, 0.24: 8.0},
               k_motor=40.0, p_random=0.02, lam=0.10)


def valley_depth(dist, stim, prior=PRIOR_MEAN, mode_hw=8):
    """1 - (deepest point between the two modes) / (shorter of the two mode
    heights). ~1 = deep valley (two peaks); ~0 = filled single hump."""
    dist = np.asarray(dist, float)
    a, b = sorted([int(stim), int(prior)])
    # segment strictly between the two modes (short arc, since |stim-prior|<180 here)
    seg = dist[a: b - 1] if (b - a) <= 180 else np.r_[dist[b - 1:], dist[:a]]
    h_stim = dist[max(0, stim - mode_hw): stim + mode_hw].max()
    h_prior = dist[max(0, prior - mode_hw): prior + mode_hw].max()
    pm = min(h_stim, h_prior)
    if pm <= 0 or seg.size == 0:
        return 0.0
    return float(np.clip(1.0 - seg.min() / pm, 0.0, 1.0))


def _band_mass(dist, ctr, hw=15):
    idx = np.arange(1, 361)
    m = np.abs(((idx - ctr + 180) % 360) - 180) <= hw
    d = np.asarray(dist, float)
    return float(d[m].sum())


def _build_matched(spec):
    """Construct an observer at the matched params for this model family."""
    name = spec.name
    if name == "switch":
        from observers.models.switching_observer import SwitchingObserver
        # switch needs per-width k_prior; use the paper defaults (reliabilities)
        return SwitchingObserver(k_like=MATCHED["k_like"],
                                 k_prior={"80": 0.5, "40": 1.4, "20": 2.7, "10": 8.7},
                                 k_motor=MATCHED["k_motor"], p_random=MATCHED["p_random"])
    if name == "hb_rachel":
        from observers.models.hb_rachel import HBRachelObserver
        return HBRachelObserver(alpha=0.6, **MATCHED)
    if name == "recombined":
        from observers.models.hb_integrate_before import HBIntegrateBeforeObserver
        return HBIntegrateBeforeObserver(alpha=0.6, **MATCHED)
    if name == "hb_adaptive":
        from observers.models.hb_adaptive_confidence import HBAdaptiveConfidenceObserver
        return HBAdaptiveConfidenceObserver(**MATCHED)
    if name == "hb_salma":
        from observers.models.hb_salma import HBSalmaObserver
        sk = tuple(MATCHED["k_like"][c] for c in (0.06, 0.12, 0.24))
        return HBSalmaObserver(rho=0.9, sensory_kappas=sk,
                               motor_kappa=MATCHED["k_motor"], lapse=MATCHED["p_random"])
    raise KeyError(name)


def _full_prediction(spec, obs, data):
    """Full (N,360) predicted distributions for one model — computed ONCE and
    cached, since the learning models' filter over the whole sequence is the
    expensive step. For the non-learning Switch this is condition-driven and
    fast, but predict_distributions handles it uniformly."""
    return spec.predict_distributions(obs, data)


def _predict_far(pred, data, stim, coh_max=0.12, sd=80):
    """Mean predicted distribution over the far-stimulus, wide-prior, low-coh
    trials for one stimulus direction (using each trial's own belief state).
    ``pred`` is the pre-computed (N,360) array for this model."""
    d = np.asarray(data["motion_direction"], int)
    c = np.asarray(data["motion_coherence"], float)
    ps = np.asarray(data["prior_std"], int)
    sel = (d == stim) & (c <= coh_max) & (ps == sd)
    if sel.sum() < 15:
        return None, int(sel.sum())
    dist = pred[sel].mean(axis=0)
    dist = dist / dist.sum()
    return dist, int(sel.sum())


def run(subject=1, models=None, plot=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or ["switch", "hb_adaptive", "hb_rachel", "recombined", "hb_salma"]
    specs = build_registry(models)
    data = load_subject(subject)
    d = np.asarray(data["motion_direction"], int)
    c = np.asarray(data["motion_coherence"], float)
    e = np.asarray(data["estimates"], int)
    ps = np.asarray(data["prior_std"], int)

    # far stimulus directions (>= 60 deg from the prior) with enough wide-prior
    # low-coherence trials
    far_dirs = []
    for stim in sorted(set(d.tolist())):
        if abs(((stim - PRIOR_MEAN + 180) % 360) - 180) < 60:
            continue
        sel = (d == stim) & (c <= 0.12) & (ps == 80)
        if sel.sum() >= 15:
            far_dirs.append(int(stim))

    obs_cache = {m: _build_matched(specs[m]) for m in models}
    # compute each model's full-sequence prediction ONCE (the expensive step for
    # the learning models), then slice per far direction below.
    pred_cache = {m: _full_prediction(specs[m], obs_cache[m], data) for m in models}
    results = {"subject": subject, "far_dirs": far_dirs, "matched_params": _json_params(),
               "per_dir": [], "mean_valley": {}, "observed_mean_valley": None}
    valleys = {m: [] for m in models}
    obs_valleys = []
    example = None

    for stim in far_dirs:
        sel = (d == stim) & (c <= 0.12) & (ps == 80)
        oh = np.zeros(360); np.add.at(oh, (e[sel] - 1) % 360, 1.0); oh /= oh.sum()
        v_obs = valley_depth(oh, stim)
        obs_valleys.append(v_obs)
        row = {"stim": stim, "n": int(sel.sum()), "observed_valley": round(v_obs, 3),
               "observed_mass_stim": round(_band_mass(oh, stim), 3),
               "observed_mass_prior": round(_band_mass(oh, PRIOR_MEAN), 3), "models": {}}
        preds_for_plot = {"observed": oh.tolist()}
        for m in models:
            dist, n = _predict_far(pred_cache[m], data, stim)
            if dist is None:
                continue
            v = valley_depth(dist, stim)
            valleys[m].append(v)
            row["models"][m] = {"valley": round(v, 3),
                                "mass_stim": round(_band_mass(dist, stim), 3),
                                "mass_prior": round(_band_mass(dist, PRIOR_MEAN), 3)}
            preds_for_plot[m] = dist.tolist()
        results["per_dir"].append(row)
        # keep the most illustrative example: the far direction closest to 90 deg
        # from the prior (the canonical two-cluster case) with a clean observed
        # valley and enough trials.
        offset90 = abs(abs(((stim - PRIOR_MEAN + 180) % 360) - 180) - 90)
        score = (v_obs, -offset90, int(sel.sum()))
        if example is None or score > example[0]:
            example = (score, stim, preds_for_plot)

    results["observed_mean_valley"] = round(float(np.mean(obs_valleys)), 3) if obs_valleys else None
    for m in models:
        results["mean_valley"][m] = round(float(np.mean(valleys[m])), 3) if valleys[m] else None

    json.dump(results, open(OUT_DIR / f"bimodality_subject{subject}.json", "w"), indent=2)

    # report
    print(f"=== bimodality-recovery screen (subject {subject}, matched unfitted params) ===")
    print(f"far stimulus directions tested: {far_dirs}")
    print(f"valley-depth: 1.0 = two clean peaks, 0 = single filled hump\n")
    print(f"  {'OBSERVED':<14s} mean valley = {results['observed_mean_valley']}")
    ranked = sorted([m for m in models if results['mean_valley'][m] is not None],
                    key=lambda m: -results['mean_valley'][m])
    for m in ranked:
        print(f"  {specs[m].label:<14s} mean valley = {results['mean_valley'][m]}")
    print(f"\n  (closer to OBSERVED = better bimodal recovery)")

    if plot and example is not None:
        _, ex_stim, ex_preds = example
        _plot_example(specs, models, subject, ex_stim, ex_preds)
    return results


def _json_params():
    p = dict(MATCHED)
    p["k_like"] = {str(k): v for k, v in p["k_like"].items()}
    return p


def _plot_example(specs, models, subject, stim, preds):
    import matplotlib as mpl, matplotlib.pyplot as plt
    mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    x = np.arange(1, 361)
    # light circular smoothing of the observed histogram so the panel reads as a
    # distribution rather than raw spikes (the metric itself uses the raw hist).
    obs = np.asarray(preds["observed"], float)
    kern = np.exp(-0.5 * (np.arange(-30, 31) / 8.0) ** 2); kern /= kern.sum()
    obs_s = np.convolve(np.r_[obs[-30:], obs, obs[:30]], kern, mode="same")[30:-30]
    obs_s = obs_s / obs_s.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, obs_s, color="0.78", label="observed (smoothed)", zorder=1)
    ymax = obs_s.max()
    for m in models:
        if m in preds:
            ax.plot(x, preds[m], color=specs[m].color, lw=1.6, label=specs[m].label)
            ymax = max(ymax, np.asarray(preds[m]).max())
    ax.set_ylim(0, ymax * 1.15)
    ax.axvline(stim, color="k", ls="--", lw=0.7); ax.axvline(PRIOR_MEAN, color="k", ls=":", lw=0.7)
    ax.set_xlim(1, 360); ax.set_xticks([90, stim, PRIOR_MEAN, 360])
    ax.set_xlabel("reported direction (deg)"); ax.set_ylabel("density")
    ax.set_title(f"Bimodality screen — subject {subject}, stimulus {stim}°"
                 f" (matched params)", loc="left")
    ax.legend(fontsize=7)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"bimodality_screen_subject{subject}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(f"bimodality_screen_subject{subject}.png", dpi=150, bbox_inches="tight")
    print(f"\nexample figure -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    run(subject=a.subject, models=a.models, plot=a.plot)
