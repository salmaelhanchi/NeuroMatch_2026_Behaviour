"""
make_table.py — model-comparison table + supplementary recovery figure
=======================================================================

Reads the batch, CV, and recovery arrays and produces:

  * model_comparison_table.md and .csv — models as ROWS, columns:
      k, ΣNLL, ΔAIC, ΔBIC, CV-NLL, per-subject wins (AIC), per-subject wins (CV)
    Δ values are relative to the reference model (Switch if present). ΣNLL and
    per-subject wins are summed / tallied across the fitted subjects.

  * model_recovery_confusion.png — the supplementary figure: parameter-recovery
    scatter (recovered vs truth) + the model-recovery confusion matrix heatmap.

Pure assembly from saved arrays — no fitting.

Usage:
  python -m observers.comparison.make_table --subjects 1 2
"""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")   # non-interactive: never block on a display when run as a subprocess
import matplotlib.pyplot as plt

from observers.comparison.registry import build_registry, ALL_SUBJECTS, DEFAULT_MODELS
from observers.comparison import fit_batch, cross_validate
from observers.helpers.paths import FITS_DIR, FIGURES_DIR

REC_DIR = FITS_DIR / "comparison_recovery"


def _ref(models):
    return "switch" if "switch" in models else models[0]


def _farband_nll(specs, models, common):
    """Regime-split NLL: total held-out-of-nothing NLL restricted to the
    bimodality regime (stimulus >=60 deg from the 225 prior, low coherence,
    wide prior). This is ~16% of trials but it is the ONLY regime where the
    abstract's shape claim lives — a model can win the aggregate NLL while
    losing here. Computed from stored params (rebuild + per-trial score)."""
    import numpy as np
    from observers.comparison.registry import load_subject
    fits = fit_batch.load_all(models, common)
    out = {m: 0.0 for m in models}
    ok = {m: True for m in models}
    for m in models:
        spec = specs[m]
        for s in common:
            if s not in fits.get(m, {}):
                ok[m] = False; continue
            data = load_subject(s)
            d = np.asarray(data["motion_direction"], int)
            c = np.asarray(data["motion_coherence"], float)
            ps = np.asarray(data["prior_std"], int)
            far = (np.abs(((d - 225 + 180) % 360) - 180) >= 60) & (c <= 0.12) & (ps == 80)
            obs = spec.rebuild(fits[m][s]["params"])
            ll = spec.trial_logliks(obs, data)
            out[m] += float(-ll[far].sum())
    return {m: (round(out[m], 1) if ok[m] else None) for m in models}


def build_table(models, subjects):
    specs = build_registry(models)
    fits = fit_batch.load_all(models, subjects)
    cvs = cross_validate.load_all(models, subjects)
    ref = _ref(models)

    # subjects for which the reference AND every model has a fit
    common = [s for s in subjects
              if all(s in fits.get(m, {}) for m in models)]
    # per-subject AIC / CV winners. CV wins are contested only among the models
    # that actually HAVE a CV score (a fit-only model such as Recombined is
    # simply not in the CV race — it must not zero out everyone else's wins).
    cv_models = [m for m in models if cvs.get(m)]
    aic_wins = {m: 0 for m in models}
    cv_wins = {m: 0 for m in models}
    n_cv_subj = 0
    for s in common:
        aic_vals = {m: fits[m][s]["aic"] for m in models}
        aic_wins[min(aic_vals, key=aic_vals.get)] += 1
        cv_here = [m for m in cv_models if s in cvs.get(m, {})]
        if cv_here:
            cv_vals = {m: cvs[m][s]["cv_nll"] for m in cv_here}
            cv_wins[min(cv_vals, key=cv_vals.get)] += 1
            n_cv_subj += 1

    sum_nll = {m: sum(fits[m][s]["nll"] for s in common) for m in models}
    sum_aic = {m: sum(fits[m][s]["aic"] for s in common) for m in models}
    sum_bic = {m: sum(fits[m][s]["bic"] for s in common) for m in models}
    sum_cv = {m: (sum(cvs[m][s]["cv_nll"] for s in common
                      if s in cvs.get(m, {})) if cvs.get(m) else None)
              for m in models}

    # across-subject mean +/- SD of the per-trial CV-NLL (error bars in the writeup)
    import numpy as _np
    cv_per_trial = {}
    for m in models:
        vals = [cvs[m][s]["cv_per_trial"] for s in common
                if s in cvs.get(m, {}) and "cv_per_trial" in cvs[m][s]]
        cv_per_trial[m] = ((round(float(_np.mean(vals)), 4),
                            round(float(_np.std(vals)), 4)) if vals else None)

    far_nll = _farband_nll(specs, models, common)
    far_ref = far_nll.get(ref)

    rows = []
    for m in models:
        rows.append({
            "model": specs[m].label, "k": specs[m].n_params,
            "sum_nll": round(sum_nll[m], 1),
            "delta_aic": round(sum_aic[m] - sum_aic[ref], 1),
            "delta_bic": round(sum_bic[m] - sum_bic[ref], 1),
            "cv_nll": (round(sum_cv[m], 1) if sum_cv[m] is not None else None),
            # per-trial CV-NLL mean and across-subject SD (None for fit-only models)
            "cv_per_trial_mean": (cv_per_trial[m][0] if cv_per_trial[m] else None),
            "cv_per_trial_sd": (cv_per_trial[m][1] if cv_per_trial[m] else None),
            # regime-split: NLL in the bimodality regime, and Δ vs reference
            "farband_nll": far_nll[m],
            "delta_farband": (round(far_nll[m] - far_ref, 1)
                              if (far_nll[m] is not None and far_ref is not None) else None),
            "wins_aic": aic_wins[m], "wins_cv": cv_wins[m],
        })
    return rows, common, ref, specs, n_cv_subj


def write_table(rows, common, ref, specs, models, n_cv_subj):
    FITS_DIR.mkdir(parents=True, exist_ok=True)
    ref_label = specs[ref].label
    ncv = n_cv_subj or len(common)
    # markdown
    md = []
    md.append("# Model comparison\n")
    md.append(f"Fitted on {len(common)} subject(s): {common}. "
              f"Δ values are relative to **{ref_label}**. All models are on the "
              f"360° grid, so NLLs are directly comparable (no grid artifact).\n")
    md.append("| Model | k | ΣNLL | ΔAIC | ΔBIC | CV-NLL | CV/trial (mean±SD) | ΔFarband-NLL | Wins (AIC) | Wins (CV) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cv = r["cv_nll"] if r["cv_nll"] is not None else "—"
        cvpt = (f"{r['cv_per_trial_mean']}±{r['cv_per_trial_sd']}"
                if r["cv_per_trial_mean"] is not None else "—")
        dfb = f"{r['delta_farband']:+}" if r["delta_farband"] is not None else "—"
        md.append(f"| {r['model']} | {r['k']} | {r['sum_nll']} | "
                  f"{r['delta_aic']:+} | {r['delta_bic']:+} | {cv} | {cvpt} | {dfb} | "
                  f"{r['wins_aic']}/{len(common)} | {r['wins_cv']}/{ncv} |")
    md.append("\n*Negative Δ favours that model over the reference. Lead with "
              "CV-NLL (overfitting-proof); AIC/BIC corroborate.*\n")
    md.append("**ΔFarband-NLL** is the NLL gap vs the reference restricted to the "
              "bimodality regime (stimulus ≥60° from the 225° prior, low coherence, "
              "wide prior — ~16% of trials). The aggregate ΣNLL is dominated by the "
              "easy near-prior trials; this column tests the abstract's shape claim "
              "on the trials where shape actually varies. A model can win ΣNLL while "
              "losing here.\n")
    md_path = FITS_DIR / "model_comparison_table.md"
    md_path.write_text("\n".join(md))
    # csv
    csv_path = FITS_DIR / "model_comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # also drop copies in cwd for artifact save
    (Path("model_comparison_table.md")).write_text("\n".join(md))
    with open("model_comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"table -> {md_path} , {csv_path}", flush=True)
    return md_path, csv_path


def supplementary_figure(models, out="model_recovery_confusion.png"):
    pr_path = REC_DIR / "parameter_recovery.json"
    mr_path = REC_DIR / "model_recovery.json"
    if not mr_path.exists():
        print("recovery arrays not found; skip supplementary figure", flush=True)
        return None
    mr = json.load(open(mr_path))
    names = mr["names"]
    conf = np.array([[mr["confusion_aic"][g][f] for f in names] for g in names], float)
    conf_n = conf / conf.sum(axis=1, keepdims=True).clip(min=1)

    ncol = 2 if pr_path.exists() else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4.5))
    axes = np.atleast_1d(axes)

    axm = axes[-1]
    im = axm.imshow(conf_n, vmin=0, vmax=1, cmap="Blues")
    axm.set_xticks(range(len(names))); axm.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axm.set_yticks(range(len(names))); axm.set_yticklabels(names, fontsize=7)
    axm.set_xlabel("selected (AIC)"); axm.set_ylabel("true generator")
    axm.set_title("Model-recovery confusion", loc="left")
    for i in range(len(names)):
        for j in range(len(names)):
            axm.text(j, i, f"{conf_n[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="k" if conf_n[i, j] < 0.6 else "w")
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)

    if pr_path.exists():
        pr = json.load(open(pr_path))
        axp = axes[0]
        tvals, rvals, cols = [], [], []
        cmap = plt.get_cmap("tab10")
        for ci, (name, blk) in enumerate(pr.items()):
            truth = blk["truth"]
            for rec in blk["recovered"]:
                for key, tv in truth.items():
                    if isinstance(tv, dict):
                        for kk, vv in tv.items():
                            rv = rec.get(key, {}).get(kk)
                            if rv is not None:
                                tvals.append(float(vv)); rvals.append(float(rv)); cols.append(cmap(ci))
                    elif isinstance(tv, (list, tuple)):
                        # vector param (e.g. Salma's sensory_kappas): plot per element
                        rvv = rec.get(key)
                        if isinstance(rvv, (list, tuple)):
                            for a, b in zip(tv, rvv):
                                tvals.append(float(a)); rvals.append(float(b)); cols.append(cmap(ci))
                    else:
                        rv = rec.get(key)
                        if rv is not None and not isinstance(rv, (list, tuple, dict)):
                            tvals.append(float(tv)); rvals.append(float(rv)); cols.append(cmap(ci))
        axp.scatter(tvals, rvals, c=cols, s=18)
        lim = [0, max(max(tvals), max(rvals)) * 1.1]
        axp.plot(lim, lim, "k:", lw=0.6)
        axp.set_xlabel("true parameter"); axp.set_ylabel("recovered")
        axp.set_title("Parameter recovery", loc="left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / out
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"supplementary figure -> {path}", flush=True)
    return path


def run(models=None, subjects=None):
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    rows, common, ref, specs, n_cv_subj = build_table(models, subjects)
    write_table(rows, common, ref, specs, models, n_cv_subj)
    ncv = n_cv_subj or len(common)
    print("\n| Model | k | ΣNLL | ΔAIC | ΔBIC | CV-NLL | ΔFar-NLL | Wins(AIC) | Wins(CV) |  (CV/trial mean±SD in md)")
    for r in rows:
        dfb = f"{r['delta_farband']:+}" if r['delta_farband'] is not None else "—"
        print(f"| {r['model']:12s} | {r['k']} | {r['sum_nll']:>9} | "
              f"{r['delta_aic']:>+8} | {r['delta_bic']:>+8} | "
              f"{str(r['cv_nll']):>9} | {dfb:>8} | {r['wins_aic']}/{len(common)} | "
              f"{r['wins_cv']}/{ncv} |")
    supplementary_figure(models)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects)
