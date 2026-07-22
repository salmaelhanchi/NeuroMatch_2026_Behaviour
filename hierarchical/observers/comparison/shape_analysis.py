"""
shape_analysis.py — distribution-shape reproduction + bimodality
================================================================

The qualitative axis of the abstract's comparison: does each model reproduce the
SHAPE of the observed response distribution, cell by cell (coherence x
prior-width), and specifically the bimodality in the far-from-prior / low-
coherence regime?

For each subject it rebuilds each model's fitted observer (from the batch-stage
JSON), computes per-trial predicted distributions, pools them within each
(coherence, prior-width) cell, and compares the pooled predicted density to the
observed response histogram by total-variation distance. It also computes a
far-band prior-cluster-mass metric (response mass near 225 deg when the stimulus
is far from 225) — the bimodality signature — for observed and each model.

Outputs (saved as .npz + a summary .json under results/fits/comparison_shape/):
  * per-cell TV distance, observed vs each model, for every subject
  * pooled observed & predicted histograms for the representative cells the
    figure will draw (so the figure is a plotting step, not a re-run)
  * far-band prior-cluster mass: observed vs each model, per subject
  * per-subject bimodality flag (is the observed far-band distribution bimodal?)

Usage:
  python -m observers.comparison.shape_analysis --subjects 1 2
  python -m observers.comparison.shape_analysis            # all 12
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from observers.comparison.registry import build_registry, load_subject, ALL_SUBJECTS, DEFAULT_MODELS
from observers.comparison import fit_batch
from observers.helpers.paths import FITS_DIR

OUT_DIR = FITS_DIR / "comparison_shape"
COHS = [0.06, 0.12, 0.24]
SDS = [80, 40, 20, 10]
PRIOR_MEAN = 225


def _circ_dist(a, b):
    return np.abs(((np.asarray(a) - b + 180) % 360) - 180)


def _hist_360(values, weights=None):
    """Normalised histogram over 1..360 (as a length-360 density)."""
    h = np.zeros(360)
    v = (np.asarray(values, int) - 1) % 360
    if weights is None:
        np.add.at(h, v, 1.0)
    else:
        np.add.at(h, v, weights)
    s = h.sum()
    return h / s if s > 0 else h


def _tv(p, q):
    """Total-variation distance between two length-360 densities."""
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return 0.5 * np.abs(p - q).sum()


def _prior_cluster_mass(dist_or_hist, half_width=18):
    """Fraction of mass within +/- half_width of the 225 deg prior mode."""
    idx = np.arange(1, 361)
    near = _circ_dist(idx, PRIOR_MEAN) <= half_width
    d = dist_or_hist / dist_or_hist.sum() if dist_or_hist.sum() > 0 else dist_or_hist
    return float(d[near].sum())


def _valley_depth(dist, stim, prior=PRIOR_MEAN, mode_hw=8):
    """Two-peak structure between the stimulus mode and the prior mode:
    1 - (deepest point between them)/(shorter mode height). ~1 = two clean
    separated peaks (bimodal); ~0 = single filled hump. Prior-cluster MASS is
    necessary but not sufficient for bimodality (a broad hump has the same
    mass); this measures the valley that mass alone misses."""
    dist = np.asarray(dist, float)
    a, b = sorted([int(stim), int(prior)])
    seg = dist[a:b - 1] if (b - a) <= 180 else np.r_[dist[b - 1:], dist[:a]]
    h_stim = dist[max(0, stim - mode_hw):stim + mode_hw].max()
    h_prior = dist[max(0, prior - mode_hw):prior + mode_hw].max()
    pm = min(h_stim, h_prior)
    if pm <= 0 or seg.size == 0:
        return 0.0
    return float(np.clip(1.0 - seg.min() / pm, 0.0, 1.0))


def _mean_valley(dist_source, d, c, ps, e=None):
    """Mean valley-depth over single far stimulus directions (never pooled
    across directions, which manufactures apparent bimodality). ``dist_source``
    is either a callable idx->mean predicted dist, or (if e given) None to use
    the observed histogram per direction."""
    vals = []
    for stim in sorted(set(int(x) for x in d)):
        if abs(((stim - PRIOR_MEAN + 180) % 360) - 180) < 60:
            continue
        sel = (d == stim) & (c <= 0.12) & (ps == 80)
        if sel.sum() < 15:
            continue
        if e is not None:
            dist = _hist_360(e[sel])
        else:
            dist = dist_source(sel)
        vals.append(_valley_depth(dist, stim))
    return round(float(np.mean(vals)), 3) if vals else None


def analyse_subject(sid, specs, fits):
    data = load_subject(sid)
    d, c, e, ps = (data["motion_direction"], data["motion_coherence"],
                   data["estimates"], data["prior_std"])
    n = e.size
    # predicted per-trial dists for each model (rebuilt from saved fit)
    pred = {}
    for name, spec in specs.items():
        if sid not in fits.get(name, {}):
            continue
        obs = spec.rebuild(fits[name][sid]["params"])
        pred[name] = spec.predict_distributions(obs, data)   # (N,360)

    cells, far_mass = [], {"observed": [], **{k: [] for k in pred}}
    # per (coherence, prior-width) cell: pooled observed vs pooled predicted, TV
    for coh in COHS:
        for sd in SDS:
            sel = (c == coh) & (ps == sd)
            if sel.sum() < 20:
                continue
            obs_hist = _hist_360(e[sel])
            row = {"coh": coh, "sd": sd, "n": int(sel.sum())}
            for name in pred:
                pooled = pred[name][sel].mean(axis=0)
                row[f"tv_{name}"] = _tv(obs_hist, pooled)
            cells.append(row)

    # far-band prior-cluster mass (bimodality signature): stimulus far from 225,
    # low coherence, widest prior — the regime where subjects go bimodal.
    far = (_circ_dist(d, PRIOR_MEAN) >= 90) & (c <= 0.12) & (ps == 80)
    if far.sum() >= 20:
        far_mass["observed"] = _prior_cluster_mass(_hist_360(e[far]))
        for name in pred:
            far_mass[name] = _prior_cluster_mass(pred[name][far].mean(axis=0))
    else:
        far_mass = None

    # valley-depth (two-peak structure), the metric that mass alone misses.
    # Per single far stimulus direction, averaged. Observed is the target.
    valley = {"observed": _mean_valley(None, d, c, ps, e=e)}
    for name in pred:
        valley[name] = _mean_valley(lambda sel, _n=name: pred[_n][sel].mean(axis=0),
                                    d, c, ps)

    # representative cells for the figure: a bimodal cell (low coh, wide prior,
    # far stimulus) and a unimodal cell (high coh, narrow prior)
    rep = {}
    for tag, (coh, sd, far_stim) in {"bimodal": (0.06, 80, True),
                                     "unimodal": (0.24, 10, False)}.items():
        if far_stim:
            sel = (c == coh) & (ps == sd) & (_circ_dist(d, PRIOR_MEAN) >= 90)
        else:
            sel = (c == coh) & (ps == sd) & (_circ_dist(d, PRIOR_MEAN) <= 30)
        if sel.sum() < 15:
            continue
        rep[tag] = {"coh": coh, "sd": sd, "n": int(sel.sum()),
                    "observed": _hist_360(e[sel]).tolist()}
        for name in pred:
            rep[tag][name] = pred[name][sel].mean(axis=0).tolist()

    return {"subject": sid, "n_trials": int(n), "cells": cells,
            "far_band": far_mass, "valley_depth": valley, "representative": rep,
            "models": list(pred.keys())}


def run(models=None, subjects=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    specs = build_registry(models)
    fits = fit_batch.load_all(models, subjects)
    summary = {}
    for sid in subjects:
        res = analyse_subject(int(sid), specs, fits)
        np.savez(OUT_DIR / f"shape_subject{sid}.npz",
                 **{"json": json.dumps(res)})
        # compact per-subject summary line
        mean_tv = {m: float(np.mean([cc[f"tv_{m}"] for cc in res["cells"]]))
                   for m in res["models"] if res["cells"]}
        summary[int(sid)] = {"mean_tv": mean_tv, "far_band": res["far_band"],
                             "valley_depth": res["valley_depth"]}
        fb = res["far_band"]
        fb_str = ("far-band prior-mass " +
                  ", ".join(f"{k}={fb[k]:.3f}" for k in fb)) if fb else "far-band n/a"
        vd = res["valley_depth"]
        vd_str = ("valley-depth " +
                  ", ".join(f"{k}={vd[k]}" for k in vd if vd[k] is not None)) if vd else "valley n/a"
        print(f"subject {sid}: mean-TV " +
              ", ".join(f"{m}={mean_tv[m]:.3f}" for m in mean_tv) +
              f" | {fb_str} | {vd_str}", flush=True)
    json.dump(summary, open(OUT_DIR / "shape_summary.json", "w"), indent=2)
    print(f"\nshape analysis complete -> {OUT_DIR}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects)
