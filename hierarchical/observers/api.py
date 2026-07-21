"""observers.api — the one import surface for notebooks.

This module is the *only* thing a notebook needs to import. Everything is a
plain function that returns a DataFrame or a matplotlib Figure (which renders
inline in Colab), so notebook cells stay one line long::

    from observers import api

    api.verify_all()                      # confirm every model behaves correctly
    api.subjects_with_data()              # which subjects are in the dataset
    trials = api.load_subject(1)          # one subject's trials as a DataFrame
    table  = api.comparison_table()       # fair AIC per model per subject
    fig    = api.plot_model_comparison()  # the 3-panel comparison figure
    fig    = api.plot_switch_curve()      # within-block switch learning curve

No model logic lives here — this file only wraps the code in observers/ so the
notebooks never contain analysis logic (which would drift from the package).
The four model classes are re-exported for anyone who wants to build an observer
by hand. Fitting functions are marked SLOW (minutes per subject).
"""
from __future__ import annotations

import json

import pandas as pd

from observers.helpers.paths import DATA_CSV, HUMAN_FITS, HB_FITS, BASIC_FITS
from observers.helpers.dataset import (
    load_subject_design, make_synthetic_design, simulate,
)

# Re-export the four observer models for advanced/manual use.
from observers.models.switching_observer import SwitchingObserver
from observers.models.basic_bayesian import BasicBayesianObserver
from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.models.asymptote_transient import AsymptoteTransientObserver
from observers.models.hb_integration import HBIntegrationObserver

# Re-export the four extension models (Anirban's spec) — experimental, not yet
# in the fair all-12 comparison. See docs/new_models_manifest.md and
# docs/new_models_build_report.md. Verify with, e.g.:
#   python -m observers.verification.verify_causal_inference
from observers.models.anirban_variants.causal_inference_mixture import CausalInferenceObserver
from observers.models.anirban_variants.logistic_mixture import LogisticMixtureObserver
from observers.models.anirban_variants.bimodal_likelihood import BimodalLikelihoodObserver
from observers.models.anirban_variants.finite_sample import FiniteSampleObserver

COHERENCES = (0.06, 0.12, 0.24)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def subjects_with_data():
    """List the subject IDs present in the experimental dataset."""
    df = pd.read_csv(DATA_CSV)
    return sorted(int(s) for s in df.subject_id.unique())


def load_subject(subject_id: int) -> pd.DataFrame:
    """One subject's trials in chronological order (a pandas DataFrame).

    Columns: motion_direction, motion_coherence, prior_std, estimate_dir.
    """
    return load_subject_design(DATA_CSV, int(subject_id))


# --------------------------------------------------------------------------- #
#  Verification — confirm each model behaves as its spec requires
# --------------------------------------------------------------------------- #
def verify_switching():
    """Run the static Switching observer checks. Returns (passed, total)."""
    from observers.verification import verify_switching as m
    return m.run()


def verify_online():
    """Run the online switching observer checks. Returns (passed, total)."""
    from observers.verification import verify_online as m
    return m.run()


def verify_hb_integration():
    """Run the hierarchical Bayesian integration checks. Returns (passed, total)."""
    from observers.verification import verify_hb_integration as m
    return m.run()


def verify_basic_bayesian():
    """Run the basic Bayesian baseline checks. Returns (passed, total)."""
    from observers.verification import verify_basic_bayesian as m
    return m.run()


def verify_all():
    """Run every model verification suite. Returns {name: (passed, total)}."""
    out = {
        "switching": verify_switching(),
        "basic_bayesian": verify_basic_bayesian(),
        "online": verify_online(),
        "hb_integration": verify_hb_integration(),
    }
    passed = sum(p for p, _ in out.values())
    total = sum(t for _, t in out.values())
    print(f"\n==== TOTAL {passed}/{total} checks passed across all models ====")
    return out


# --------------------------------------------------------------------------- #
#  Model comparison
# --------------------------------------------------------------------------- #
def comparison_table() -> pd.DataFrame:
    """Fair AIC per subject per model as a tidy DataFrame.

    Read live from results/fits/ (fair_fit + at_fit + hb_integration), so it
    never drifts from the committed fits. Columns: subject, model, AIC,
    dAIC_from_best, is_best. Lower AIC is better.
    """
    from observers.analysis.plot_model_comparison import load_comparison_aics
    aic = load_comparison_aics()
    rows = []
    for s, per_model in aic.items():
        best = min(per_model, key=per_model.get)
        for model, val in per_model.items():
            rows.append({
                "subject": int(s), "model": model, "AIC": round(val, 1),
                "dAIC_from_best": round(val - per_model[best], 1),
                "is_best": model == best,
            })
    return (pd.DataFrame(rows)
            .sort_values(["subject", "AIC"])
            .reset_index(drop=True))


def plot_model_comparison():
    """The 3-panel comparison figure (fit / dynamics / read-out). Returns a Figure."""
    from observers.analysis.plot_model_comparison import make_figure
    return make_figure()


def plot_switch_curve():
    """Within-block switch-probability learning curve. Returns a Figure."""
    from observers.analysis.build_switch_curve import build
    fig, _ = build()
    return fig


# --------------------------------------------------------------------------- #
#  Fitted observers — load stored parameters as a ready-to-use model
# --------------------------------------------------------------------------- #
def load_fitted_online(subject_id: int) -> OnlineHierarchicalObserver:
    """An online switching observer with subject `subject_id`'s fitted parameters.

    Call `.estimate_distribution(...)` or `.filter(...)` on the result to
    generate predictions. Reads results/fits/human_fit_results.json.
    """
    r = json.load(open(HUMAN_FITS))[str(int(subject_id))]
    return OnlineHierarchicalObserver(
        k_like={0.06: r["k_like"][0], 0.12: r["k_like"][1], 0.24: r["k_like"][2]},
        k_motor=r["k_motor"], p_random=r["p_random"], lam=r["lam"])


def load_fitted_hb(subject_id: int) -> HBIntegrationObserver:
    """A hierarchical Bayesian integration observer with subject `subject_id`'s
    fitted parameters. Reads results/fits/hb_integration_results.json.
    """
    r = json.load(open(HB_FITS))[str(int(subject_id))]
    return HBIntegrationObserver(
        k_like={0.06: r["k_like"][0], 0.12: r["k_like"][1], 0.24: r["k_like"][2]},
        alpha=r["alpha"], k_motor=r["k_motor"], p_random=r["p_random"], lam=r["lam"])


def load_fitted_basic_bayesian(subject_id: int) -> BasicBayesianObserver:
    """A basic Bayesian observer with subject `subject_id`'s fitted parameters.
    Reads results/fits/basic_bayesian_results.json.
    """
    r = json.load(open(BASIC_FITS))[str(int(subject_id))]
    return BasicBayesianObserver(
        k_like={0.06: r["k_like"][0], 0.12: r["k_like"][1], 0.24: r["k_like"][2]},
        k_prior={k: float(v) for k, v in r["k_prior"].items()},
        k_motor=r["k_motor"], p_random=r["p_random"])


# --------------------------------------------------------------------------- #
#  Fitting — SLOW (minutes per subject). For exploration; the committed
#  results/fits/ were produced with multi-start via the CLI fitting modules.
# --------------------------------------------------------------------------- #
def _subject_data(subject_id: int) -> dict:
    d = load_subject_design(DATA_CSV, int(subject_id))
    return {
        "motion_direction": d.motion_direction.values.astype(int),
        "motion_coherence": d.motion_coherence.values.astype(float),
        "prior_std": d.prior_std.values.astype(int),
        "estimates": d.estimate_dir.values.astype(int),
    }


def fit_switching_models(subject_id: int) -> dict:
    """Fit BOTH switch models (static + online) to one subject. SLOW (~1-2 min).

    Returns a dict of NLL/AIC and fitted parameters. Does not write to disk.
    """
    from observers.fitting.online_fit_human import fit_subject
    return fit_subject(int(subject_id))


def fit_hb(subject_id: int, maxiter: int = 400) -> dict:
    """Single-start fit of the HB integration model to one subject. SLOW (~3-5 min).

    Returns NLL/AIC/BIC and fitted parameters. Does not write to disk. This is a
    quick single-start fit for exploration; the committed fits used multi-start
    (run `python -m observers.fitting.hb_integration_fit human <id>` for that).
    """
    from observers.fitting.hb_integration_fit import fit
    data = _subject_data(subject_id)
    obs, nll, _ = fit(data, maxiter=maxiter)
    n = int(data["estimates"].size)
    return {
        "subject": int(subject_id), "n_trials": n, "nll": nll,
        "aic": HBIntegrationObserver.aic(nll),
        "bic": HBIntegrationObserver.bic(nll, n),
        "alpha": obs.alpha, "k_motor": obs.k_motor,
        "p_random": obs.p_random, "lam": obs.lam,
        "k_like": [obs.k_like[c] for c in COHERENCES],
    }


def fit_basic_bayesian(subject_id: int, maxiter: int = 800) -> dict:
    """Single-start fit of the basic Bayesian baseline to one subject. SLOW (~1 min).

    Returns NLL/AIC/BIC and fitted parameters. Does not write to disk.
    """
    from observers.fitting.basic_bayesian_fit import fit, _load_subject
    data = _load_subject(subject_id)
    obs, nll, aic, bic = fit(data, maxiter=maxiter)
    return {
        "subject": int(subject_id), "n_trials": int(data["estimates"].size),
        "nll": nll, "aic": aic, "bic": bic,
        "k_motor": obs.k_motor, "p_random": obs.p_random,
        "k_like": [obs.k_like[c] for c in COHERENCES], "k_prior": obs.k_prior,
    }
