# Placement manifest — four new observer models (Anirban's spec)

## Repository layout (abstract-focused split)

`observers/models/` holds exactly the **two models the abstract and task list
compare**: the **switching observer** and the **HB integration** model
(with their fitters and verification). Everything else — the online switching
observer, the asymptote-transient learner, and the four experimental extensions
(causal-inference, logistic, bimodal-likelihood, finite-sample) — lives in the
repo-root **`other models/`** folder.

Rationale: `docs/project_tasks.md` is built around HB (tasks 1, 3), the switching
observer (task 4), and their comparison (task 5); the abstract asks HB-integration
vs. switch. So the two contenders sit in the main folder and the team's starting
point is uncluttered. The learning variants and extensions remain available but
out of the way.

Nothing about how you import changed: each `observers/{models,fitting,verification}/__init__.py`
appends the matching `other models/<sub>/` folder to its package `__path__`, so
the moved files keep resolving at their original module names — e.g.
`from observers.models.hb_integration import HBIntegrationObserver` and
`python -m observers.verification.verify_switching` both work unchanged. This is
necessary because the new models are built on the originals (causal-inference
imports `hb_integration`; finite-sample and logistic import
`online_switching_observer`), and `observers/helpers/dataset.py` imports
`online_switching_observer` too.

```
hierarchical/
├─ observers/
│  ├─ models/         switching_observer, hb_integration     (+ __init__ path-extend)
│  │  └─ anirban_variants/   causal_inference_mixture, logistic_mixture,
│  │                         bimodal_likelihood, finite_sample   (real subpackage)
│  ├─ fitting/        hb_integration_fit, fair_refit, online_recovery  (+ __init__ path-extend)
│  ├─ verification/   verify_switching, verify_hb_integration (+ __init__ path-extend)
│  ├─ helpers/        circular, bayes_lookup, belief_grid, dataset, paths  (unchanged)
│  └─ api.py          re-exports all 8 model classes          (additive edit)
└─ other models/
   ├─ models/         online_switching_observer, asymptote_transient
   ├─ fitting/        online_fit_human, asymptote_transient_fit,
   │                  causal_inference_fit, logistic_mixture_fit,
   │                  bimodal_likelihood_fit, finite_sample_fit
   └─ verification/   verify_online, verify_causal_inference,
                      verify_logistic_mixture, verify_bimodal_likelihood,
                      verify_finite_sample
```

The four extension **models** are grouped in a real subpackage
`observers/models/anirban_variants/` (its name is a valid Python identifier, so
it's a normal package with its own `__init__.py`, not a path-magic folder). Import
them as `from observers.models.anirban_variants import CausalInferenceObserver` or
`from observers.models.anirban_variants.finite_sample import FiniteSampleObserver`.
Their fitters/verification stay under `other models/` and resolve unchanged via
the `__path__` extension.

Import paths are unchanged regardless of which folder holds a file: every
`observers/{models,fitting,verification}/__init__.py` appends the matching
`other models/<sub>/` folder to its package `__path__`, so
`from observers.models.hb_integration import ...` and
`python -m observers.verification.verify_finite_sample` both work from either
location. Cross-folder imports resolve in both directions (the extensions in
`other models/` import `hb_integration` from `observers/`; `asymptote_transient`
imports `online_switching_observer`, both in `other models/`).

---


All files below were written **directly into the package tree** and match the
existing `observers/` conventions (dataclass models, `observers.helpers.*` for
shared math, `N_PARAMS` + static `aic`/`bic`, `pack`/`unpack`/`fit` fitters,
`test_*` verification scripts). At build time no existing model/helper/fitter
file was modified — every model entry is a new file. (Subsequently, during the
"other models" split above, the three subpackage `__init__.py` files —
`observers/{models,fitting,verification}/__init__.py` — were given a `__path__`
extension so the moved core files still resolve; and `observers/api.py` got an
additive re-export block. The four original model/fitter/verification source
files themselves were relocated unchanged, not edited.)

## Files created (12 new files)

| model | model file | verification | fitter |
|---|---|---|---|
Current locations (after the abstract-focused split and the `anirban_variants`
subpackage grouping). The **model** files were built in `observers/models/`, then
relocated to `observers/models/anirban_variants/`; their fitters and verification
suites now live under `other models/`. Import module paths track these locations
(the model modules are `observers.models.anirban_variants.<name>`); the fitter/
verification module names are unchanged and still resolve via the `__path__`
extension.

| Causal-inference mixture (inferred responsibility) | `observers/models/anirban_variants/causal_inference_mixture.py` | `other models/verification/verify_causal_inference.py` | `other models/fitting/causal_inference_fit.py` |
| Logistic-covariate mixture (Deck B) | `observers/models/anirban_variants/logistic_mixture.py` | `other models/verification/verify_logistic_mixture.py` | `other models/fitting/logistic_mixture_fit.py` |
| Bimodal-likelihood control (Chetverikov & Jehee) | `observers/models/anirban_variants/bimodal_likelihood.py` | `other models/verification/verify_bimodal_likelihood.py` | `other models/fitting/bimodal_likelihood_fit.py` |
| Finite-sample readout (resource-rational nest) | `observers/models/anirban_variants/finite_sample.py` | `other models/verification/verify_finite_sample.py` | `other models/fitting/finite_sample_fit.py` |

## Class / parameter registry

| model | class | N_PARAMS | parameters |
|---|---|---|---|
| causal-inference | `CausalInferenceObserver` | 7 | 3 k_like + kappa + pi + p_random + k_motor |
| logistic mixture | `LogisticMixtureObserver` | 11 | 3 k_like + k_motor + p_random + 6 logistic coeffs (b0, b_pstd, b_coh, b_int, b_err, b_hist) |
| bimodal-likelihood | `BimodalLikelihoodObserver` | 10 | 3 k_like + 4 k_prior + k_motor + p_random + g (streak weight) |
| finite-sample | `FiniteSampleObserver` | 10 | 3 k_like + 4 k_prior + k_motor + p_random + n_samples |

## How to run (from the repo root, `hierarchical/`)

```
export PYTHONPATH=.
python -m observers.verification.verify_causal_inference
python -m observers.verification.verify_logistic_mixture
python -m observers.verification.verify_bimodal_likelihood
python -m observers.verification.verify_finite_sample
```

Each prints `[PASS]`/`[FAIL]` lines and an `N/N checks passed` summary.

## Helper-script coverage (fitter + verification per model)

Every model now has both a fitter and a verification suite. Two gaps were closed:

- **`other models/verification/verify_asymptote_transient.py`** (new) — AT was the
  only model with no verification script. 5/5 checks: conservation; exact
  structural reductions (`carryover=False` ⇒ `k_eff≡k_asym`; `τ→0`); monotone
  within-block relaxation; and cross-path equivalence to the static
  `SwitchingObserver` (reported at its true ~6e-5 grid tolerance — AT reads out on
  the online fixed-k grid, the switch via the Girshick look-up, so this reduction
  is grid-approximate, not machine-precision).
- **`observers/fitting/switching_observer_fit.py`** (new) — the switch was the only
  model without a dedicated fitter CLI. A thin wrapper delegating to the tested
  `online_recovery.fit_static` (no duplicated optimisation logic), giving
  `python -m observers.fitting.switching_observer_fit human <sid>` parity with the
  other models. For the full fair multi-start AIC table use
  `python -m observers.fitting.fair_refit table`.

Full verification set (all pass): `verify_switching` 5/5, `verify_hb_integration`
12/12, `verify_online` 9/9, `verify_asymptote_transient` 5/5,
`verify_causal_inference` 7/7, `verify_logistic_mixture` 4/4,
`verify_bimodal_likelihood` 6/6, `verify_finite_sample` 5/5.

## Removed
- **`observers/analysis/rt_switch_analysis.py`** — a standalone reaction-time
  diagnostic. The task list explicitly recommends focusing on estimation error
  rather than reaction time, so it is out of scope; moved to Trash (recoverable).
  Nothing imported it. `build_switch_curve.py` and `plot_model_comparison.py` were
  kept — both feed the abstract's model-comparison and learning-curve analyses.
- Regenerable `__pycache__/` bytecode caches were cleared.

## Existing model source unchanged
The four pre-existing model source files —
`switching_observer.py`, `online_switching_observer.py`,
`asymptote_transient.py`, `hb_integration.py` — and all helpers/fitters/
verification scripts have unchanged *contents*: they were relocated to
`other models/` (see layout above), not edited. Confirmed by import check (all
four still import and expose their original `N_PARAMS`). The only files edited
this session are the three subpackage `__init__.py` (path extension) and
`observers/api.py` (additive re-exports); the `observers/helpers/` files were
not touched at all. Note: this tree is not a git repo, so the guarantee rests on
the import check and inventory, not `git diff`.
