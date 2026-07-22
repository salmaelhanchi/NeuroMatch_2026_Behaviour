"""
make_figure.py — multi-panel results figure (standard layout, Panels A-E)
=========================================================================

Assembles the main results figure from arrays already written by the analysis
stages (batch fit, cross-validation, shape analysis). It does NOT fit or
simulate anything — it is a pure plotting step, so it is fast and rerunnable.

Panels (conventional model-comparison layout):
  A  schematic: the two model structures side by side (text schematic)
  B  distribution reproduction: observed histogram (grey) + each model's
     predicted density, for a bimodal cell and a unimodal cell
  C  per-subject quantitative comparison: Delta(CV-NLL) or Delta(AIC),
     model minus reference, sorted bars (below zero = model wins)
  D  learned-latent trajectory: E[kappa] / E[alpha] over trials vs block
     structure, for an example subject (only for models that learn)
  E  bimodality scatter: observed vs predicted far-band prior-cluster mass

Colours are pulled from each model's registry ModelSpec.color and threaded
across every panel; grey is always the observed data.

Usage:
  python -m observers.comparison.make_figure --subjects 1 2 --example-subject 9
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")   # non-interactive: never block on a display when run as a subprocess
import matplotlib.pyplot as plt

from observers.comparison.registry import build_registry, load_subject, ALL_SUBJECTS, DEFAULT_MODELS
from observers.comparison import fit_batch, cross_validate
from observers.helpers.paths import FITS_DIR, FIGURES_DIR

SHAPE_DIR = FITS_DIR / "comparison_shape"


# Self-contained publication style (the assembler must run as a standalone
# script, so we set rcParams here rather than depend on an interactive plugin):
# clean despined frame, legible fonts, no chartjunk.
_STYLE = {
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titleweight": "bold",
    "lines.linewidth": 1.4, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
}


def _load_shape(sid):
    p = SHAPE_DIR / f"shape_subject{sid}.npz"
    if not p.exists():
        return None
    return json.loads(str(np.load(p, allow_pickle=True)["json"]))


def _ref_model(models):
    """Reference model for Delta metrics: the Switch if present, else first."""
    return "switch" if "switch" in models else models[0]


def build_figure(models, subjects, example_subject):
    mpl.rcParams.update(_STYLE)
    specs = build_registry(models)
    colors = {m: specs[m].color for m in models}
    labels = {m: specs[m].label for m in models}
    fits = fit_batch.load_all(models, subjects)
    cvs = cross_validate.load_all(models, subjects)
    ref = _ref_model(models)

    fig = plt.figure(figsize=(12, 7.8))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42,
                          left=0.07, right=0.98, top=0.9, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")  # spare / legend

    # ---- Panel A: schematic (text) ----
    axA.axis("off")
    axA.set_title("Model structures", loc="left")
    lines = []
    for m in models:
        s = specs[m]
        mech = {
            "hb_adaptive": "learns kappa+alpha online, integrate-after",
            "switch":      "selects prior OR evidence (no integration)",
            "basic_bayes": "always integrates, fixed reliabilities",
            "hb_salma":    "learns kappa online, integrate-before (72-bin)",
            "recombined":  "learns kappa online, integrate-before",
            "hb_rachel":   "fixed alpha, integrate-after",
        }.get(m, "integrator")
        lines.append(f"{labels[m]} (k={s.n_params}): {mech}")
    axA.text(0.02, 0.85, "\n\n".join(lines), va="top", ha="left",
             fontsize=8, transform=axA.transAxes)

    # ---- Panel B: distribution reproduction (bimodal + unimodal cell) ----
    shp = _load_shape(example_subject) or (
        next((_load_shape(s) for s in subjects if _load_shape(s)), None))
    axB.set_title("Response distributions", loc="left")
    if shp and shp.get("representative"):
        rep = shp["representative"]
        tag = "bimodal" if "bimodal" in rep else next(iter(rep))
        cell = rep[tag]
        x = np.arange(1, 361)
        axB.fill_between(x, cell["observed"], color="0.7", step="mid",
                         label="observed", zorder=1)
        for m in models:
            if m in cell:
                axB.plot(x, cell[m], color=colors[m], lw=1.5, label=labels[m])
        axB.set_xlabel("reported direction (deg)")
        axB.set_ylabel("density")
        axB.set_xlim(1, 360); axB.set_xticks([90, 180, 270, 360])
        axB.axvline(225, color="k", ls=":", lw=0.6)
        axB.legend(frameon=False, fontsize=6, loc="upper left")
    else:
        axB.text(0.5, 0.5, "shape arrays not yet computed", ha="center",
                 transform=axB.transAxes, fontsize=7)

    # ---- Panel C: per-subject Delta metric bars ----
    axC.set_title(f"Δ vs {labels[ref]}", loc="left")
    have_cv = all(cvs.get(m) for m in models)
    metric = "CV-NLL" if have_cv else "AIC"
    subs = sorted(s for s in subjects if s in fits.get(ref, {}))
    deltas = {m: [] for m in models if m != ref}
    ok_subs = []
    for s in subs:
        if have_cv:
            base = cvs[ref].get(s, {}).get("cv_nll")
            vals = {m: cvs[m].get(s, {}).get("cv_nll") for m in models}
        else:
            base = fits[ref].get(s, {}).get("aic")
            vals = {m: fits[m].get(s, {}).get("aic") for m in models}
        if base is None or any(vals[m] is None for m in models):
            continue
        ok_subs.append(s)
        for m in deltas:
            deltas[m].append(vals[m] - base)
    if ok_subs:
        xpos = np.arange(len(ok_subs))
        w = 0.8 / max(len(deltas), 1)
        for i, m in enumerate(deltas):
            axC.bar(xpos + i * w, deltas[m], width=w, color=colors[m], label=labels[m])
        axC.axhline(0, color="k", lw=0.7)
        axC.set_xticks(xpos + 0.4 - w / 2); axC.set_xticklabels(ok_subs, fontsize=6)
        axC.set_xlabel("subject"); axC.set_ylabel(f"Δ {metric}")
        axC.legend(frameon=False, fontsize=6)
    else:
        axC.text(0.5, 0.5, "fit results not yet available", ha="center",
                 transform=axC.transAxes, fontsize=7)

    # ---- Panel D: learned-latent trajectory (example subject) ----
    axD.set_title("Learned prior width vs block", loc="left")
    learner = next((m for m in models if specs[m].learns), None)
    traj_drawn = False
    if learner and example_subject in fits.get(learner, {}):
        data = load_subject(example_subject)
        obs = specs[learner].rebuild(fits[learner][example_subject]["params"])
        out = obs.filter(data["motion_direction"], data["motion_coherence"],
                         feedback=data["motion_direction"], record_belief=True)
        if "believed_sd" in out:
            t = np.arange(len(out["believed_sd"]))
            axD.plot(t, out["believed_sd"], color=colors[learner], lw=1.0,
                     label=f"{labels[learner]} E[SD]")
            axD.plot(t, data["prior_std"], color="k", lw=0.7, ls="--",
                     label="true block SD")
            axD.set_xlabel("trial"); axD.set_ylabel("prior SD (deg)")
            axD.legend(frameon=False, fontsize=6, loc="lower left", ncol=1)
            traj_drawn = True
    if not traj_drawn:
        axD.text(0.5, 0.5, "trajectory needs a learning model fit", ha="center",
                 transform=axD.transAxes, fontsize=7)

    # ---- Panel E: bimodality scatter (far-band prior-cluster mass) ----
    axE.set_title("Far-band prior-cluster mass", loc="left")
    obs_v, drawn = [], False
    pts = {m: [] for m in models}
    for s in subjects:
        sh = _load_shape(s)
        if not sh or not sh.get("far_band"):
            continue
        fb = sh["far_band"]
        obs_v.append(fb["observed"])
        for m in models:
            pts[m].append(fb.get(m, np.nan))
    if obs_v:
        for m in models:
            axE.scatter(obs_v, pts[m], color=colors[m], s=20, label=labels[m])
        lim = [0, max(max(obs_v), *(max(pts[m]) for m in models if pts[m])) * 1.1 + 1e-6]
        axE.plot(lim, lim, "k:", lw=0.6)
        axE.set_xlabel("observed prior-cluster mass")
        axE.set_ylabel("predicted")
        axE.legend(frameon=False, fontsize=6)
        drawn = True
    if not drawn:
        axE.text(0.5, 0.5, "far-band arrays not yet computed", ha="center",
                 transform=axE.transAxes, fontsize=7)

    for ax, L in zip([axA, axB, axC, axD, axE], "ABCDE"):
        ax.text(-0.14, 1.14, L, transform=ax.transAxes, fontweight="bold", fontsize=13)
    return fig


def run(models=None, subjects=None, example_subject=9, out="model_comparison_figure.png"):
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    fig = build_figure(models, subjects, example_subject)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / out
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(out, dpi=150, bbox_inches="tight")   # also to cwd for artifact save
    print(f"figure -> {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--example-subject", type=int, default=9)
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects, example_subject=a.example_subject)
