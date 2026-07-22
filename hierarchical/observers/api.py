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
from observers.models.hb_rachel import HBRachelObserver

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
    """Run the online-learning Switching observer checks. Returns (passed, total)."""
    from observers.verification import verify_online_switching_observer as m
    return m.run()


def verify_hb_rachel():
    """Run the HB-Rachel checks. Returns (passed, total)."""
    from observers.verification import verify_hb_rachel as m
    return m.run()


# Backward-compat alias (old public name).
verify_hb_integration = verify_hb_rachel


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
        "hb_rachel": verify_hb_rachel(),
    }
    passed = sum(p for p, _ in out.values())
    total = sum(t for _, t in out.values())
    print(f"\n==== TOTAL {passed}/{total} checks passed across all models ====")
    return out


# --------------------------------------------------------------------------- #
#  Model comparison
# --------------------------------------------------------------------------- #
def comparison_table() -> pd.DataFrame:
    """AIC per subject per model as a tidy DataFrame.

    Read live from the per-model fit folders
    (results/fits/comparison/<model>/subject<N>.json) for every registered
    model, so it never drifts from the committed fits. Columns: subject, model,
    AIC, dAIC_from_best, is_best. Lower AIC is better. For a fuller table with
    NLL/BIC/CV, use results_table().
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


def load_fitted_hb(subject_id: int) -> HBRachelObserver:
    """A hierarchical Bayesian integration observer with subject `subject_id`'s
    fitted parameters. Reads results/fits/hb_integration_results.json.
    """
    r = json.load(open(HB_FITS))[str(int(subject_id))]
    return HBRachelObserver(
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
    (run `python -m observers.fitting.hb_rachel_fit human <id>` for that).
    """
    from observers.fitting.hb_rachel_fit import fit
    data = _subject_data(subject_id)
    obs, nll, _ = fit(data, maxiter=maxiter)
    n = int(data["estimates"].size)
    return {
        "subject": int(subject_id), "n_trials": n, "nll": nll,
        "aic": HBRachelObserver.aic(nll),
        "bic": HBRachelObserver.bic(nll, n),
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


# =========================================================================== #
#  Registry-backed generic accessors — support ALL registered models
#
#  These dispatch through observers.comparison.registry.MODEL_REGISTRY, so a
#  newly registered model is supported automatically with no edits here. They
#  read the SAME fit files the batch pipeline writes
#  (results/fits/comparison/<model>/subject<N>.json), so the API and the
#  comparison pipeline never disagree. The older named functions above remain
#  as thin, backward-compatible conveniences.
# =========================================================================== #
def list_models() -> list:
    """Registry keys for every registered model, e.g. ['switch', 'hb_rachel', ...]."""
    return _all_model_names()


def _all_model_names() -> list:
    """Every registered model key, read from the registry's single source of truth."""
    from observers.comparison.registry import ALL_MODELS
    return list(ALL_MODELS)


def model_info() -> "pd.DataFrame":
    """One row per registered model: key, label, #params, what it learns."""
    from observers.comparison.registry import build_registry
    reg = build_registry(_all_model_names())
    rows = [{"key": k, "label": s.label, "n_params": s.n_params,
             "learns": getattr(s, "learns", ""), "color": s.color}
            for k, s in reg.items()]
    return pd.DataFrame(rows).set_index("key")


def get_model(key: str, params: dict):
    """Rebuild a model observer from a parameter dict, dispatched by registry key."""
    from observers.comparison.registry import build_registry
    reg = build_registry([key])
    if key not in reg:
        raise KeyError(f"unknown model {key!r}; known: {_all_model_names()}")
    return reg[key].rebuild(params)


def load_fitted(key: str, subject_id: int):
    """Load the batch-fitted observer for one model x subject.

    Reads results/fits/comparison/<model>/subject<N>.json (the file the fit
    pipeline writes) and rebuilds the observer via the registry. Raises
    FileNotFoundError if that subject has not been fit for this model.
    """
    import json
    from observers.comparison.fit_batch import _result_path
    from observers.comparison.registry import build_registry
    p = _result_path(key, int(subject_id))
    if not p.exists():
        raise FileNotFoundError(
            f"no fit for model {key!r} subject {subject_id} at {p}. "
            f"Run: python -m observers.comparison.fit_batch --models {key} "
            f"--subjects {subject_id}")
    rec = json.load(open(p))
    obs = build_registry([key])[key].rebuild(rec["params"])
    return obs, rec


def fitted_subjects(key: str) -> list:
    """Subject IDs that have a saved point fit for this model."""
    from observers.comparison.fit_batch import _result_path
    from observers.comparison.registry import ALL_SUBJECTS
    return [s for s in ALL_SUBJECTS if _result_path(key, s).exists()]


def load_fitted_cv(key: str, subject_id: int) -> dict:
    """Load the cross-validation record for one model x subject (held-out NLL).

    Reads results/fits/comparison_cv/<model>/subject<N>_cv.json.
    """
    import json
    from observers.comparison.cross_validate import _result_path as _cv_path
    p = _cv_path(key, int(subject_id))
    if not p.exists():
        raise FileNotFoundError(
            f"no CV for model {key!r} subject {subject_id} at {p}")
    return json.load(open(p))


def predict(key: str, subject_id: int) -> "np.ndarray":
    """Per-trial predicted response distributions for a batch-fitted model x subject.

    Loads the saved fit, rebuilds the observer, and returns
    spec.predict_distributions(obs, subject_data).
    """
    from observers.comparison.registry import build_registry, load_subject
    obs, _rec = load_fitted(key, subject_id)
    return build_registry([key])[key].predict_distributions(obs, load_subject(subject_id))


def fit_model(key: str, subject_id: int, maxiter: int = 400) -> dict:
    """Fit ANY registered model to one subject via the registry (does not write to disk).

    Returns NLL/AIC/BIC + start_spread. For persisted, resumable, multi-start
    batch fits use `python -m observers.comparison.fit_batch --models <key>`.
    """
    import math
    from observers.comparison.registry import build_registry, load_subject
    spec = build_registry([key])[key]
    data = load_subject(int(subject_id))
    res = spec.fit(data, maxiter=maxiter)
    n = int(len(data["estimates"]))
    nll = float(res.nll); k = int(res.n_params)
    return {"model": key, "label": spec.label, "subject": int(subject_id),
            "n_trials": n, "nll": nll, "k": k,
            "aic": 2 * k + 2 * nll, "bic": k * math.log(n) + 2 * nll,
            "start_spread": float(getattr(res, "start_spread", float("nan")))}


# =========================================================================== #
#  Notebook helpers — processed data, results tables, and learned trajectories
#
#  Built for team Colab notebooks: everything is registry-backed and reads the
#  batch pipeline's own files, so a notebook shows exactly the committed fits.
# =========================================================================== #
def results_table(models: list = None, include_cv: bool = True) -> "pd.DataFrame":
    """Tidy model-comparison table across all registered models x subjects.

    One row per (model, subject), read from the batch pipeline's fit JSONs
    (results/fits/comparison/<model>/subject<N>.json). Columns: model, label,
    subject, n_trials, k, nll, aic, bic, and (if include_cv and available)
    cv_nll. Missing fits are simply absent (e.g. hb_salma subject 12), so the
    table honestly reflects what has been fit.
    """
    import json
    from observers.comparison.fit_batch import _result_path
    from observers.comparison.cross_validate import _result_path as _cv_path
    from observers.comparison.registry import ALL_SUBJECTS
    names = models or _all_model_names()
    rows = []
    for key in names:
        for sid in ALL_SUBJECTS:
            p = _result_path(key, sid)
            if not p.exists():
                continue
            d = json.load(open(p))
            row = {"model": key, "label": d.get("label", key), "subject": sid,
                   "n_trials": d.get("n_trials"), "k": d.get("k"),
                   "nll": d.get("nll"), "aic": d.get("aic"), "bic": d.get("bic")}
            if include_cv:
                cp = _cv_path(key, sid)
                if cp.exists():
                    cv = json.load(open(cp))
                    row["cv_nll"] = cv.get("cv_nll", cv.get("nll"))
            rows.append(row)
    return pd.DataFrame(rows)


def observed_distribution(subject_id: int, direction: int = None,
                          coherence: float = None, prior_std: int = None,
                          bins: int = 360) -> "np.ndarray":
    """Empirical response histogram (density over 1..360 deg) for one subject.

    Optionally condition on a single stimulus cell by passing any of
    ``direction`` (displayed motion direction), ``coherence``, or ``prior_std``.
    Returns a length-``bins`` density vector (sums to 1) aligned with the
    per-direction axis that :func:`predict` returns, so the two overlay directly.
    Raises ValueError if the filter selects no trials.
    """
    import numpy as np
    df = load_subject(subject_id)
    m = np.ones(len(df), dtype=bool)
    if direction is not None:
        m &= (df["motion_direction"].values == int(direction))
    if coherence is not None:
        m &= np.isclose(df["motion_coherence"].values, float(coherence))
    if prior_std is not None:
        m &= (df["prior_std"].values == int(prior_std))
    est = df["estimate_dir"].values[m]
    if est.size == 0:
        raise ValueError(f"no trials for subject {subject_id} with the given "
                         f"direction/coherence/prior_std filter")
    edges = np.linspace(0.5, bins + 0.5, bins + 1)
    counts, _ = np.histogram(est, bins=edges)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def belief_trajectory(key: str, subject_id: int) -> "pd.DataFrame":
    """Learned prior-belief trajectory for a LEARNING model, replayed over one
    subject's real trials (feedback = the displayed direction, as in the task).

    Loads the batch-fitted observer, rolls it forward, and returns a DataFrame
    with one row per trial: trial (1-based), believed_sd (deg), and, when the
    model tracks it, believed_alpha (prior confidence). Raises ValueError for a
    non-learning model (switch, basic_bayes), which has no evolving belief.
    """
    from observers.comparison.registry import build_registry, load_subject as _ls
    spec = build_registry([key])[key]
    if not getattr(spec, "learns", False):
        raise ValueError(f"model {key!r} does not learn — it has no belief "
                         f"trajectory (only learning models do)")
    obs, _rec = load_fitted(key, subject_id)
    data = _ls(subject_id)
    dirs = data["motion_direction"]; cohs = data["motion_coherence"]
    out = obs.filter(dirs, cohs, feedback=dirs, record_belief=True)
    cols = {"trial": range(1, len(out["believed_sd"]) + 1),
            "believed_sd": out["believed_sd"]}
    if "believed_alpha" in out:
        cols["believed_alpha"] = out["believed_alpha"]
    return pd.DataFrame(cols)


def trial_logliks(key: str, subject_id: int, obs=None) -> "np.ndarray":
    """Per-trial log-likelihood vector for a model on one subject's real trials.

    The escape hatch for custom goodness-of-fit: sum it for the total NLL, slice
    it by condition, average it for a per-trial score, or hold out trials for a
    custom cross-validation. By default uses the batch-fitted observer; pass your
    own ``obs`` (e.g. from ``get_model`` / ``fit_model``) to score arbitrary
    parameters. Returns a length-n_trials array aligned with ``load_subject``.
    """
    from observers.comparison.registry import build_registry, load_subject as _ls
    spec = build_registry([key])[key]
    if obs is None:
        obs, _rec = load_fitted(key, subject_id)
    return spec.trial_logliks(obs, _ls(int(subject_id)))


def simulate(key: str, design, seed: int = 0, obs=None) -> dict:
    """Draw synthetic responses from a model over a trial design (generative mode).

    ``design`` is either a subject id (uses that subject's real trial sequence)
    or a DataFrame with columns motion_direction/motion_coherence/prior_std. By
    default the model is the batch-fitted observer for ``key``; pass ``obs`` to
    simulate from arbitrary parameters (the basis of parameter recovery: simulate
    from known params, refit, compare). Returns a data dict (motion_direction,
    motion_coherence, prior_std, estimates) that ``fit_model``/``trial_logliks``
    consume directly.
    """
    from observers.comparison.registry import build_registry
    spec = build_registry([key])[key]
    design_is_subject = not hasattr(design, "columns")
    if obs is None:
        if not design_is_subject:
            raise ValueError("pass obs= when design is a DataFrame (there is no "
                             "subject id to load a fitted observer from)")
        obs, _rec = load_fitted(key, int(design))
    design_df = load_subject(int(design)) if design_is_subject else design
    return spec.simulate(obs, design_df, seed)


def bias_variability(subject_id: int, prior_mean: float = 225.0) -> "pd.DataFrame":
    """Estimation bias and circular SD by condition — the Laquitaine Fig-3 core.

    For each (motion_coherence, prior_std, motion_direction) cell of one
    subject's data, computes the circular-mean estimate, the signed bias toward
    the prior mean (positive = pulled toward ``prior_mean``), and the circular SD
    of the estimates. Uses the paper's vector-average circular statistics
    (``circular_weighted_mean_std``), so the 360-deg wraparound is handled
    correctly. Returns a tidy DataFrame — group/plot it however you like.

    Columns: coherence, prior_std, direction, n, mean_estimate, bias, circ_sd.
    """
    import numpy as np
    from observers.helpers.circular import circular_weighted_mean_std
    df = load_subject(subject_id)
    d = df["motion_direction"].values.astype(int)
    c = df["motion_coherence"].values.astype(float)
    p = df["prior_std"].values.astype(int)
    e = df["estimate_dir"].values.astype(int)

    def _signed_toward(mean_est, direction):
        # signed error of the mean estimate relative to the true direction,
        # projected onto the sign that points toward the prior mean.
        err = (mean_est - direction + 180.0) % 360.0 - 180.0        # est - true
        to_prior = (prior_mean - direction + 180.0) % 360.0 - 180.0  # prior - true
        return err * np.sign(to_prior) if to_prior != 0 else abs(err)

    rows = []
    for cc in np.unique(c):
        for pp in np.unique(p):
            for dd in np.unique(d):
                m = (c == cc) & (p == pp) & (d == dd)
                n = int(m.sum())
                if n == 0:
                    continue
                est = e[m]
                mean_est, sd = circular_weighted_mean_std(est, np.ones(n))
                rows.append({"coherence": float(cc), "prior_std": int(pp),
                             "direction": int(dd), "n": n,
                             "mean_estimate": round(mean_est, 2),
                             "bias": round(_signed_toward(mean_est, dd), 2),
                             "circ_sd": round(sd, 2)})
    return pd.DataFrame(rows)
