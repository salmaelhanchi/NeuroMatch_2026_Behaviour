# Building, fitting, and validating a model — the pipeline, and the tools for ours

This document has two parts:
- **Part A** — the general pipeline: the stages of building, training, and
  validating a computational model, in the order you actually do them.
- **Part B** — for each stage, the specific tool / file / function in *our*
  switching-observer project, and why it is the right one.

---

# Part A — The general pipeline

## 1. Specify the model (the generative story)
Before any code, write down in equations how the model turns a stimulus into a
response: the latent variables, what is observed, where the noise enters. The
output of this stage is the **likelihood function** `P(response | stimulus,
parameters)`. Everything downstream depends on getting this right.

## 2. Implement it and prove the code matches the equations (verification)
Translate the equations to code, then prove the code computes what you wrote —
not by eye, but by checking **reductions to known answers**: set a parameter to
a limit where the answer is analytically known and confirm the code returns it.
Verification is about *correctness of implementation*, and it happens before you
touch data.

## 3. Get the data into model-facing form (preprocessing)
Load trials and convert them to the variables the model consumes (circular
directions, coherence, block width, the reported estimate), then validate the
table — no NaNs where the model can't handle them, condition labels matching the
model's parameter keys. Silent bugs here masquerade as model failures later.

## 4. Fit — estimate parameters (training)
Choose an objective (here: **maximum likelihood** — minimise the negative
log-likelihood of the observed estimates) and an optimiser. Two things matter a
lot: **multi-start** (non-convex likelihoods have local optima; a single cold
start can settle in a bad basin), and **sensible starting values and bounds**.
For interpretable cognitive models the fitted *parameters themselves* are the
scientific product, not just predictive accuracy.

## 5. Validate — three distinct questions, kept separate
- **Parameter recovery (identifiability):** simulate from *known* parameters,
  refit, check you get them back. If you can't, no fitted number means anything.
- **Model recovery:** simulate from model A, fit both A and B, confirm A wins —
  proves the models are *distinguishable* before comparing them on real data.
- **Goodness of fit + out-of-sample:** does it fit held-out data and beat
  competitors? Use cross-validation or a complexity-penalised criterion
  (AIC/BIC); an in-sample win can be overfitting.

## 6. Compare models and interpret
Rank competing models on the same footing (identical preprocessing, matched
fitting effort, held-out likelihood), then interpret the *winning parameters*
scientifically.

**The throughline:** verification asks "is the code right?", recovery asks "are
the fitted numbers trustworthy?", and comparison asks "is it the best account of
the data?" — three separate checks people routinely blur into "does it work".

> A related but distinct activity: confirming a model **is a published model**
> (structural correspondence to the paper) verifies the *specification* in Step 1
> against an external authority, before fitting enters. That is what notebook
> `04_switching_matches_paper.ipynb` does for our switching observer.

---

# Part B — The tools for our model, step by step

Paths are relative to `hierarchical/`. "Run from repo root with `PYTHONPATH=.`"

## Step 1 — Specify → the model modules + equation docs
- **`observers/models/switching_observer.py`**, **`hb_integration.py`** — the
  model classes; each docstring states the equations (H1–H5 / M1–M5) it
  implements.
- **`docs/generative-model.md`**, **`docs/model_verification_guide.md`**, and the
  paper PDF `docs/Laquitaine_Gardner_2018_switching_observer.pdf`.

*Why these:* the specification must live next to the code and be traceable to the
paper. The class docstrings are the equations; the paper PDF is the external
authority the spec is checked against.

## Step 2 — Verify → the `verification/` suites
- **`observers/verification/verify_switching.py`** (`run()`, 5 checks) and
  **`verify_hb_integration.py`** (`run()`, 12 checks).
- **`notebooks/04_switching_matches_paper.ipynb`** — the step-by-step
  paper-correspondence walkthrough for the switch.

*Why these:* they check **reductions to known answers** (evidence read-out ==
pure von Mises; switch weights == the reliability ratio; α=1 == the exact
Girshick estimator to machine precision) — the correct evidence for "the code
computes the equations", far stronger than a plot that looks right.

## Step 3 — Preprocess → the `helpers/`
- **`observers/helpers/dataset.py`** — `load_subject_design`, `make_model_frame`,
  `simulate`, `make_synthetic_design`.
- **`observers/helpers/circular.py`** — von Mises pdfs, circular distance, the
  1..360° direction space (a faithful port of the paper's MATLAB primitives).
- **`observers/helpers/paths.py`** — canonical data / results paths.

*Why these:* the model consumes **circular** variables and a strict condition
grid; `circular.py` guarantees the angle maths matches the paper's, and
`dataset.py` produces exactly the columns the fitters expect (so a wrong column
name can't silently corrupt a fit).

## Step 4 — Fit → the `fitting/` wrappers + scipy
- **`observers/fitting/switching_observer_fit.py`** — `fit(data, maxiter)` and the
  `human N` CLI; the switch's dedicated fitter.
- **`observers/fitting/online_recovery.py`** — `fit_static` (the shared,
  multi-start static-switch optimiser the switch fitter delegates to).
- **`observers/fitting/fair_refit.py`** — gives every model the *same* multi-start
  treatment so the comparison is like-for-like.
- Optimiser: **scipy `minimize`, Nelder-Mead** (derivative-free — the likelihood
  is a grid look-up with no clean gradient).

*Why these:* Nelder-Mead suits a non-smooth grid likelihood; the multi-start
wrappers guard against local optima (a real effect here — cold vs warm starts
moved fits by hundreds of AIC); `fair_refit.py` enforces equal fitting effort so
no model is flattered by being fit harder.

## Step 5 — Validate → the recovery + CV tools
- **Parameter recovery:** `online_recovery.static_parameter_recovery()` (switch),
  `online_recovery.parameter_recovery()` (online), and
  `hb_integration_fit.py recover` (HB, incl. the α-vs-lapse identifiability check).
- **Model recovery:** `online_recovery.model_recovery()` (simulate one, fit both,
  confirm the generator wins) and `crosscheck_static_matches_switching()`.
- **Goodness of fit / out-of-sample:** **AIC/BIC** returned by every fitter, and
  **K-fold cross-validation on held-out per-trial NLL** (the method used in the
  broader project to catch AT's in-sample overfitting; the CV harness lives in
  the project's analysis working copy, not this minimal branch).

*Why these:* recovery is the check that a fitted parameter is *trustworthy* —
these routines simulate from known truth and confirm it comes back, and the
α-vs-lapse test specifically rules out the classic mixture-model confound. CV /
AIC / BIC separate genuine fit from overfitting, which single in-sample NLL
cannot.

## Step 6 — Compare & interpret → the `analysis/` tools
- **`observers/analysis/plot_model_comparison.py`** — assembles the per-subject
  AIC/BIC comparison figure across models.
- **`observers/analysis/build_switch_curve.py`** — the empirical
  prior-vs-evidence learning curve, for interpreting *what the winning switch
  parameters mean* behaviourally.
- **`notebooks/02_model_comparison.ipynb`**, **`notebooks/03_validate_models*.ipynb`**
  — the team-facing comparison and validation walkthroughs.

*Why these:* comparison must be on one footing (same preprocessing via
`helpers/`, same fitting effort via `fair_refit.py`), and interpretation needs a
behavioural read-out (`build_switch_curve.py`) not just a winning AIC number.

---

## One-line map

| Step | Tool in our project | Why |
|---|---|---|
| 1 Specify | `models/*.py` docstrings, `docs/generative-model.md`, paper PDF | spec traceable to the paper |
| 2 Verify | `verification/verify_*.py`, notebook `04` | reductions to known answers |
| 3 Preprocess | `helpers/dataset.py`, `circular.py`, `paths.py` | correct circular vars + condition grid |
| 4 Fit | `fitting/switching_observer_fit.py`, `online_recovery.fit_static`, scipy Nelder-Mead, `fair_refit.py` | non-smooth likelihood, multi-start, equal effort |
| 5 Validate | `online_recovery.static_parameter_recovery/model_recovery`, `hb_integration_fit recover`, AIC/BIC + K-fold CV | trustworthy params, overfitting caught |
| 6 Compare | `analysis/plot_model_comparison.py`, `build_switch_curve.py`, notebooks `02`/`03` | one footing + behavioural interpretation |
