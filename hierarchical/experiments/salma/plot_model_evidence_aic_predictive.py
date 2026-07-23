from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
HIERARCHICAL_ROOT = HERE.parents[1]
sys.path.insert(0, str(HIERARCHICAL_ROOT))

from observers.comparison.registry import ALL_MODELS, build_registry


FIT_ROOT = HIERARCHICAL_ROOT / "results" / "fits" / "comparison"
CV_ROOT = HIERARCHICAL_ROOT / "results" / "fits" / "comparison_cv"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
OUT_CSV = RESULTS_DIR / "model_evidence_aic_predictive_summary.csv"
OUT_SUBJECT_CSV = RESULTS_DIR / "model_evidence_subject_scores.csv"
OUT_FIG = FIGURES_DIR / "model_evidence_aic_predictive.png"

MODEL_ORDER = [
    "switch",
    "basic_bayes",
    "hb_adaptive",
    "hb_rachel",
    "hb_salma",
    "recombined",
    "hierarchical_online",
]

COLOR_OVERRIDES = {
    "switch": "#30638e",
    "basic_bayes": "#a0a0a0",
    "hb_adaptive": "#d1495b",
    "hb_rachel": "#edae49",
    "hb_salma": "#8e6c8a",
    "recombined": "#66a182",
    "hierarchical_online": "#3a7d44",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_fit_records(model: str) -> dict[int, dict]:
    folder = FIT_ROOT / model
    if not folder.exists():
        return {}
    out = {}
    for path in folder.glob("subject*.json"):
        stem = path.stem.replace("subject", "")
        if stem.isdigit():
            out[int(stem)] = load_json(path)
    return out


def load_cv_records(model: str) -> dict[int, dict]:
    folder = CV_ROOT / model
    if not folder.exists():
        return {}
    out = {}
    for path in folder.glob("subject*_cv.json"):
        sid = path.stem.replace("subject", "").replace("_cv", "")
        if sid.isdigit():
            out[int(sid)] = load_json(path)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    registry = build_registry(ALL_MODELS)
    rows = []
    subject_rows = []

    for model in MODEL_ORDER:
        fits = load_fit_records(model)
        cvs = load_cv_records(model)
        if not fits:
            continue
        label = fits[min(fits)].get("label", registry[model].label)
        color = COLOR_OVERRIDES.get(model, registry[model].color)

        total_nll = sum(float(item["nll"]) for item in fits.values())
        total_aic = sum(float(item["aic"]) for item in fits.values())
        total_bic = sum(float(item["bic"]) for item in fits.values())
        total_trials = sum(int(item.get("n_trials", 0)) for item in fits.values())
        train_nll_per_trial = total_nll / total_trials
        train_bits_over_uniform = math.log2(360.0) - train_nll_per_trial / math.log(2.0)

        cv_trials = sum(int(item.get("n_trials", 0)) for item in cvs.values())
        cv_nll = sum(float(item.get("cv_nll", np.nan)) for item in cvs.values()) if cvs else np.nan
        cv_nll_per_trial = cv_nll / cv_trials if cvs and cv_trials else np.nan
        cv_bits_over_uniform = (
            math.log2(360.0) - cv_nll_per_trial / math.log(2.0)
            if np.isfinite(cv_nll_per_trial)
            else np.nan
        )
        cv_subject_bits = []
        for sid, item in cvs.items():
            per_trial = float(item["cv_per_trial"])
            bits = math.log2(360.0) - per_trial / math.log(2.0)
            cv_subject_bits.append(bits)

        rows.append(
            {
                "model": model,
                "label": label,
                "n_fit_subjects": len(fits),
                "n_cv_subjects": len(cvs),
                "n_fit_trials": total_trials,
                "n_cv_trials": cv_trials,
                "sum_nll": total_nll,
                "sum_aic": total_aic,
                "sum_bic": total_bic,
                "train_nll_per_trial": train_nll_per_trial,
                "train_bits_per_trial_over_uniform": train_bits_over_uniform,
                "cv_nll": cv_nll,
                "cv_nll_per_trial": cv_nll_per_trial,
                "cv_bits_per_trial_over_uniform": cv_bits_over_uniform,
                "cv_bits_subject_mean": float(np.mean(cv_subject_bits)) if cv_subject_bits else np.nan,
                "cv_bits_subject_sem": (
                    float(np.std(cv_subject_bits, ddof=1) / math.sqrt(len(cv_subject_bits)))
                    if len(cv_subject_bits) > 1
                    else np.nan
                ),
                "color": color,
            }
        )

        for sid, item in fits.items():
            subject_rows.append(
                {
                    "model": model,
                    "label": label,
                    "subject": sid,
                    "n_trials": int(item.get("n_trials", 0)),
                    "nll": float(item["nll"]),
                    "aic": float(item["aic"]),
                    "bic": float(item["bic"]),
                    "cv_nll": float(cvs[sid]["cv_nll"]) if sid in cvs else np.nan,
                    "cv_per_trial": float(cvs[sid]["cv_per_trial"]) if sid in cvs else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    summary["delta_aic_from_best"] = summary["sum_aic"] - summary["sum_aic"].min()
    summary["delta_bic_from_best"] = summary["sum_bic"] - summary["sum_bic"].min()
    summary = summary.sort_values("delta_aic_from_best").reset_index(drop=True)
    summary.to_csv(OUT_CSV, index=False)
    pd.DataFrame(subject_rows).sort_values(["subject", "model"]).to_csv(
        OUT_SUBJECT_CSV, index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    ax = axes[0]
    aic_rows = summary.sort_values("delta_aic_from_best", ascending=True)
    bars = ax.barh(
        aic_rows["label"],
        aic_rows["delta_aic_from_best"],
        color=aic_rows["color"],
        alpha=0.92,
    )
    ax.invert_yaxis()
    ax.axvline(0, color="0.2", lw=1.0)
    ax.set_xlabel("Delta AIC from best, summed over participants")
    ax.set_title("In-sample evidence with complexity penalty", loc="left")
    ax.grid(axis="x", alpha=0.18)
    for bar, (_, row) in zip(bars, aic_rows.iterrows()):
        text = f"{row['delta_aic_from_best']:.0f}"
        ax.text(
            bar.get_width() + max(aic_rows["delta_aic_from_best"].max() * 0.01, 20),
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            fontsize=8,
        )

    ax = axes[1]
    train_rows = summary.sort_values("train_bits_per_trial_over_uniform", ascending=True)
    ax.barh(
        train_rows["label"],
        train_rows["train_bits_per_trial_over_uniform"],
        color=train_rows["color"],
        alpha=0.92,
    )
    ax.set_xlabel("In-sample predictive information, bits/trial over uniform")
    ax.set_title("Training likelihood, not held out", loc="left")
    ax.grid(axis="x", alpha=0.18)

    ax = axes[2]
    cv_rows = summary[np.isfinite(summary["cv_bits_per_trial_over_uniform"])].copy()
    cv_rows = cv_rows.sort_values("cv_bits_per_trial_over_uniform", ascending=True)
    ax.barh(
        cv_rows["label"],
        cv_rows["cv_bits_per_trial_over_uniform"],
        xerr=cv_rows["cv_bits_subject_sem"],
        color=cv_rows["color"],
        alpha=0.92,
        capsize=3,
    )
    ax.set_xlabel("Held-out predictive information, bits/trial over uniform")
    ax.set_title("Cross-validated predictive likelihood", loc="left")
    ax.grid(axis="x", alpha=0.18)
    missing_cv = summary.loc[
        ~np.isfinite(summary["cv_bits_per_trial_over_uniform"]), "label"
    ].tolist()
    if missing_cv:
        ax.text(
            0.01,
            -0.18,
            "No saved CV record for: " + ", ".join(missing_cv),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="0.25",
        )

    fig.suptitle("Model evidence: AIC and predictive likelihood", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")

    print(OUT_CSV)
    print(OUT_SUBJECT_CSV)
    print(OUT_FIG)
    print(summary[
        [
            "model",
            "label",
            "n_fit_subjects",
            "n_cv_subjects",
            "delta_aic_from_best",
            "train_bits_per_trial_over_uniform",
            "cv_bits_per_trial_over_uniform",
        ]
    ].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
