# Fitting, cross-validation & model-validation procedure

**Repo:** `NeuroMatch_2026_Behaviour/hierarchical`
**Purpose:** the exact, model-agnostic steps this repo uses to fit any observer, cross-validate it,
and validate the comparison. Every step below iterates the model **registry**, so the procedure is
identical for `switch`, `hb_adaptive`, `hb_salma`, or any model added later — you name models, never
edit stage code.

All commands are run from the repo root (`hierarchical/`) with the package importable:

```bash
cd hierarchical
export PYTHONPATH=.        # or run inside .venv where the package is installed
```

Canonical layout (from `observers/helpers/paths.py`):
- data → `data/data01_direction4priors.csv`
- point fits → `results/fits/comparison/<model>/subject<N>.json`
- CV → `results/fits/comparison_cv/<model>/subject<N>_cv.json`
- shape → `results/fits/comparison_shape/`
- recovery → `results/fits/comparison_recovery/`
- figures/tables → `results/figures/`, `results/`

---

## The one contract every model satisfies — the registry

`observers/comparison/registry.py` is the single source of truth for "which models exist" and how each
is fit/scored/simulated. Each model is one `ModelSpec` exposing a uniform interface:

```
spec.fit(data, maxiter=, mask=)     -> FitResult(obs, nll, x, n_params, start_spread)
spec.trial_logliks(obs, data)       -> per-trial log-likelihood array (length N)
spec.simulate(obs, design, seed)    -> synthetic data dict   (for recovery)
spec.rebuild(params)                -> observer from a saved params dict (for shape)
spec.predict_distributions(obs,data)-> (N, 360) per-trial predicted response distributions
spec.n_params, spec.learns, spec.grid_deg, spec.color
```

`build_registry(names)` returns the active subset; `ALL_MODELS` lists every registered key; the current
set is `switch, basic_bayes, hb_adaptive, hb_rachel, hb_salma, recombined, hierarchical_online,
reliability_mixture`. **To add a model:** write its observer + a house-style fitter, add one `ModelSpec`
entry, and every stage picks it up with no other change.

**Shared fitting conventions (identical across models, so a fit-quality difference is never an
optimizer-effort difference):**
- Optimizer: **Nelder–Mead** (scipy), tolerances `xatol=fatol=1e-2`, `maxiter=400` by default.
  This matches Laquitaine & Gardner (2018), who fit every observer with Nelder–Mead. *(The paper used a
  strict tolerance of 1e-4 and fell back to CMA-ES for the ill-conditioned 9-parameter switch — see the
  methods-review for where the repo currently diverges.)*
- Likelihood: **per-trial log-probability of the subject's actual estimate** on a common **360°** grid,
  so all NLLs are directly comparable (no grid-resolution artifact).
- Multi-start: **`N_STARTS = 10`** on every reported point fit (base start + Gaussian-jittered
  perturbations in transformed parameter space), keeping the lowest-NLL result. `start_spread`
  (NLL range across starts) is recorded as a convergence diagnostic. **Inside CV folds, a single start**
  is used (`_starts_for(mask)`), on the argument that per-fold global search is unnecessary.
- Learning models replay their belief filter over the **full ordered sequence**; trial order (and thus
  learning) is always preserved, including under a CV mask.

### The per-model files behind each registry entry

The stages are generic, but each model contributes up to **three dedicated files**. The `ModelSpec`
in `registry.py` is the seam that wires them into the uniform interface; the pipeline never imports them
directly. When adding or debugging a model, these are the files you touch:

| registry key | k | learns | model class (`observers/models/`) | fitter (`observers/fitting/`) | verifier (`observers/verification/`) |
|---|---|---|---|---|---|
| `switch` | 9 | no | `switching_observer.py` | `switching_observer_fit.py` † | `verify_switching.py` |
| `basic_bayes` | 9 | no | `basic_bayesian.py` | `basic_bayesian_fit.py` | `verify_basic_bayesian.py` |
| `hb_adaptive` | 6 | yes | `hb_adaptive_confidence.py` | `hb_adaptive_confidence_fit.py` | — (see below) |
| `hb_rachel` | 7 | yes | `hb_rachel.py` | `hb_rachel_fit.py` | `verify_hb_rachel.py` |
| `recombined` | 7 | yes | `hb_integrate_before.py` | `hb_rachel_fit.py` (reused) | — |
| `hb_salma` | 6 | yes | `hb_salma.py` (+ `salma_hierarchical_helpers/`) | none — inline fitter in `registry.py` | — |
| `hierarchical_online` | 8 | yes | `hierarchical_online.py` | `hierarchical_online_fit.py` (+ `hierarchical_online_backfill.py`) | `verify_hierarchical_online.py` |
| `reliability_mixture` | 10 | yes | `reliability_mixture.py` | `reliability_mixture_fit.py` | `verify_reliability_mixture.py` |

Each per-model **fitter** module exposes (via the registry) at least `N_PARAMS`, `pack`/`unpack`,
`fit(data, x0=, maxiter=, mask=)`, a per-trial log-lik function, and usually `_simulate`. Several also
carry a standalone CLI for solo debugging (not used by the pipeline):
- `recover` / `human N..` / `cv N..` → `hb_adaptive_confidence_fit.py`, `hb_rachel_fit.py`
- `human N..` only → `switching_observer_fit.py`, `basic_bayesian_fit.py`, `online_switching_observer_fit.py`
  e.g. `python -m observers.fitting.hb_adaptive_confidence_fit recover`.

**Shared / infrastructure fitters (not a single model's):**
- `online_recovery.py` — the multi-start static/online switch optimiser (`fit_static`, `fit_online`,
  `pack_*`/`unpack_*`, `conv_info`). **† The `switch` registry entry fits *through* this**, then rebuilds
  a canonical `SwitchingObserver` from θ; `switching_observer_fit.py` is a thin wrapper over it.
- `fair_refit.py` — the like-for-like multi-start refit driver for the switch family.
- `hierarchical_online_backfill.py` — a maintenance/backfill helper for the online model's outputs.

**Special cases worth knowing (all documented in the methods-review):**
1. **`hb_salma` has no fitter module.** Its `_hb_salma_spec` deliberately refits with the *house*
   optimiser (Nelder–Mead, 10 starts, `maxiter=400`) inline in `registry.py`, **not** with its native
   `salma_hierarchical_helpers` package (which used Powell / 2 rho-starts). This is intentional: it puts
   Salma on the identical fitting protocol as every other model so a fit difference can't be an optimiser
   difference. The `salma_hierarchical_helpers/` subpackage supplies the model's numerics only.
2. **`recombined` reuses `hb_rachel_fit.py`** — same engine, different integration rule
   (`hb_integrate_before.py`), so no separate fitter is needed.
3. **Every registry model now has a dedicated verifier.** `verify_hb_adaptive.py`, `verify_recombined.py`,
   and `verify_hb_salma.py` close the former gap; each asserts *model-specific* identities, not boilerplate:
   - `verify_hb_adaptive.py` (16 checks): mixed-prior atom reduces to VM at α=1 / uniform at α=0; α and κ
     are **learned, not fitted** (E[α] rises under informed feedback and falls under uniform; E[κ]→the
     generating SD); trial-0 response is pre-feedback (no within-trial leakage).
   - `verify_recombined.py` (11 checks): integrate-before **==** integrate-after under a point-mass belief
     but **diverges** under a spread belief (the defining axis vs HB-Rachel).
   - `verify_hb_salma.py` (17 checks): 6 params (rho/3 sensory kappas/motor/lapse, no α); the native-72-bin
     vs up-sampled-deg360 grid contract (native NLL < deg360 NLL, never mix in one AIC); higher sensory
     kappa pulls the MAP percept off the prior toward the stimulus; rho controls confidence-decline speed.

   `api.verify_all()` runs all seven suites (**72 checks total**).
4. Two model files are **not** wired into the current registry: `hb_integration.py` /
   `hb_integration_fit.py` (superseded by `hb_adaptive_confidence` / `hb_rachel`) and
   `online_switching_observer.py` (the online-switch model, still exercised by
   `verify_online_switching_observer.py` but not a registry key). Don't cite them as part of the live
   comparison.

**Model-agnostic rule for adding a model:** create `models/<m>.py` and `fitting/<m>_fit.py` (exposing the
fitter API above), add a `verify_<m>.py`, add one `ModelSpec` entry, and — per house convention — prove
the new model's shared-code touches leave every other model's fits bit-identical before relying on them.

---

## Step 0 — Validate model correctness *before* trusting any fit

Identity checks with known answers, per model, under `observers/verification/`. Run the whole suite:

```python
from observers import api
api.verify_all()          # prints PASS/FAIL per check; returns {model: (passed, total)}
```

or one model directly: `python -m observers.verification.verify_switching`.

Each script asserts model-specific identities that must hold regardless of data — e.g. for the Switch
(`verify_switching.py`): evidence read-out with a flat prior equals the measurement von Mises; prior
read-out with flat evidence is a delta at 225°; switch weights match Eq. 6; all 108 condition
distributions are valid probability vectors; and NLL is lower at generating than at bad parameters
(**the minimal identifiability sanity check**: the objective responds to the data in the right
direction). A model that fails Step 0 must not proceed to fitting.

**Model-agnostic rule:** a new model needs a `verify_<model>.py` asserting (a) its read-out reduces to
the known closed form in limiting cases, (b) every predicted distribution is a valid probability vector,
(c) NLL(true) < NLL(bad) on self-simulated data, and (d) **at least one discriminating check** that would
FAIL for the nearest sibling model (e.g. integrate-before vs -after under a spread belief; learned vs
fitted α). Wire it into `api.verify_all()`. As of now **all seven registry models have a verifier**
(72 checks total).

---

## Step 1 — Point fit (maximum likelihood), per model × subject

Driver: `observers/comparison/fit_batch.py`. Resumable, one JSON per model × subject.

```bash
# default 2 headline models, all 12 subjects
python -m observers.comparison.fit_batch

# any registered model(s) / subject subset / iteration budget
python -m observers.comparison.fit_batch --models hb_adaptive switch --subjects 1 2
python -m observers.comparison.fit_batch --models hb_salma --maxiter 1500
python -m observers.comparison.fit_batch --force            # refit even if JSON exists
```

Per fit it calls `spec.fit(data, maxiter)` (which multi-starts the reported fit), then computes
`AIC = 2k + 2·NLL` and `BIC = k·ln(n) + 2·NLL` where **n = trial count** and **k = `spec.n_params`**.
It writes NLL/AIC/BIC/k, fitted `params`, raw `theta`, `start_spread`, and the scipy `convergence`
block to `results/fits/comparison/<model>/subject<N>.json`.

**Resume semantics (important):** a run skips an existing file **only if it was fit at ≥ the requested
`maxiter`**; a stale lower-budget file (e.g. a `maxiter=15` smoke result) is refit. So bumping `--maxiter`
upgrades stale fits automatically; use `--force` to refit everything.

**Convergence judgement (repo caveat):** the scipy `converged` boolean is *not* a reliable stop
criterion here — the `k_e[0.24]` identifiability ridge keeps the likelihood surface flat, so
Nelder–Mead can report non-convergence (or hit `maxiter`) while the NLL is already stable. **Judge
convergence by NLL stability across a `maxiter` increase, and by a small `start_spread`, not by the
boolean.** A large `start_spread` (starts landing at genuinely different NLLs) is the real
under-convergence flag.

---

## Step 2 — Cross-validation (held-out per-trial NLL)

Driver: `observers/comparison/cross_validate.py`. Resumable, one JSON per model × subject.

```bash
python -m observers.comparison.cross_validate --models hb_adaptive switch --subjects 1 2 --folds 5
python -m observers.comparison.cross_validate --maxiter 400 --ticker-interval 30
```

Procedure per model × subject:
1. **Block-aligned folds** (`_block_folds`): a "block" is a maximal run of constant `prior_std`; whole
   blocks are grouped into K (default 5) contiguous folds, so a held-out fold is one or more *complete*
   prior-width blocks (never a split block). If there are fewer blocks than folds, it falls back to plain
   contiguous trial folds via `np.array_split`.
2. For each fold: fit on the training mask (`spec.fit(data, maxiter, mask=train)` — **single start**),
   then score held-out per-trial NLL with `spec.trial_logliks(obs, data)[test]`. The belief filter runs
   over the **full ordered sequence** so learning/order is preserved.
3. Writes `cv_nll`, `cv_per_trial` (= cv_nll / N), per-fold detail, k, folds, seconds to
   `results/fits/comparison_cv/<model>/subject<N>_cv.json`.

A background health ticker prints a machine-health beat every `--ticker-interval` seconds (a long CV run
is otherwise silent through each minutes-long fold fit); set `--ticker-interval 0` to disable.

**What this metric does and does not test (read the methods-review):**
- The belief filter runs over the **full ordered sequence** even under a CV mask, so the belief that
  predicts a held-out trial was updated from earlier trials' true directions. **This is not response
  leakage:** the belief is driven only by the exogenous stimulus (`feedback = motion_direction`, an
  input), the subject's *estimate* (the scored quantity) is never used to predict itself or any other
  trial, and the update is applied *after* each trial's response is scored. Both learning and non-learning
  models get the same full stimulus stream, so there is no tilt toward the learning hypothesis.
- Consequence: block-fold CV scores **parameter generalization** (do train-fitted params predict held-out
  responses?), **not forecasting** of an unseen block. To test the *learning dynamics* per the abstract,
  add a **forward-chaining / expanding-window** split (train early blocks → predict later ones).
- **Single-start folds** can leave an ill-conditioned model (e.g. the 9-parameter switch) under-fit in
  a fold, which surfaces as in-sample-better / out-of-sample-worse. Multi-start the fold fits for
  poorly-conditioned models if this appears.

---

## Step 3 — Distribution-shape reproduction (the qualitative axis)

Driver: `observers/comparison/shape_analysis.py`.

```bash
python -m observers.comparison.shape_analysis --subjects 1 2      # or no --subjects for all 12
```

For each subject it rebuilds every model's fitted observer from the Step-1 JSON (`spec.rebuild`),
computes per-trial predicted distributions (`spec.predict_distributions`), pools them within each
**(coherence × prior-width) cell**, and compares pooled predicted vs observed response histograms by
**total-variation distance**. It also computes the **far-band prior-cluster mass** (response mass near
225° when the stimulus is far from 225° — the bimodality signature) for observed and each model, and a
per-subject **bimodality flag** on the observed far-band distribution. Outputs (`.npz` + summary `.json`)
under `results/fits/comparison_shape/`, including the pooled histograms the figure will draw (so the
figure is a plotting step, not a re-run).

This is the stage that tests whether a model reproduces **both unimodal and bimodal** shapes, cell by
cell — the qualitative comparison that fit indices alone cannot adjudicate.

---

## Step 4 — Identifiability: parameter & model recovery

Driver: `observers/comparison/recovery.py`. This is the check that *licenses* the whole comparison.

```bash
python -m observers.comparison.recovery --models hb_adaptive switch --n-sim 2 --maxiter 400
```

1. **Parameter recovery** (per model): simulate from known ground-truth params (`spec.simulate`), refit
   the *same* model, and check the fitted params return. For models with a known trade-off (e.g.
   HB-Adaptive's κ–α ridge) this maps the ridge so any reported latent is interpretable.
2. **Model recovery** (confusion matrix): simulate from each generator, fit **all** models, tally the
   AIC/BIC winner into a rows=generator × cols=selected matrix. **The diagonal must dominate** — if the
   metrics can't recover the true generator on synthetic data, the comparison on real data is
   meaningless.

Both iterate the registry, so adding a model extends the confusion matrix automatically. Outputs under
`results/fits/comparison_recovery/`.

**Model-agnostic guidance:** `--n-sim 2` is a smoke default; use **≥20–50 sims per generator** to
estimate off-diagonal confusion rates. Any parameter that governs the mechanism under test (e.g. a
learning-rate/volatility term) must be *demonstrated* recoverable, not exempted from the pass criterion.

---

## Step 5 — Assemble figure & comparison table

- `observers/comparison/make_figure.py` → multi-panel results figure (fit / dynamics / read-out / shape).
- `observers/comparison/make_table.py` → the comparison table. Per model it reports k, summed NLL, ΔAIC
  and ΔBIC vs a reference model, summed CV-NLL and mean±sd CV-per-trial, a **far-band NLL** regime split
  (NLL restricted to the bimodality regime: stimulus ≥60° from 225°, low coherence — a model can win the
  aggregate NLL while losing where the abstract's shape claim lives), and per-subject **AIC-win / CV-win**
  counts. CV wins are contested only among models that actually have CV results.

```bash
python -m observers.comparison.make_figure --subjects ... --example-subject 9
python -m observers.comparison.make_table  --subjects ...
```

---

## Running the whole thing

**Serial, end-to-end** (`run_all.py`) — fit → CV → shape → recovery → figure → table, all registry-driven:

```bash
python -m observers.comparison.run_all                                  # default models, all subjects
python -m observers.comparison.run_all --models hb_adaptive switch hb_rachel
python -m observers.comparison.run_all --skip-fit --skip-cv --skip-recovery   # re-assemble outputs only
```

Stages 1–2 are resumable; stages 3–6 are cheap assembly and always rerun. `--skip-*` flags re-build
figures/tables from existing fits without refitting.

**Parallel / resumable / analyze-as-you-go** (`run_parallel.py`) — recommended for the heavy models,
which are multi-hour per subject:

```bash
python -m observers.comparison.run_parallel --workers 4 \
       --fit-models switch hb_adaptive hb_salma recombined \
       --cv-models  switch hb_adaptive hb_salma
```

- Runs each (model × subject) point fit as an independent worker subprocess (each writes its own JSON on
  completion — no file changes until a subject finishes is expected, not a stall).
- `--fit-models` / `--cv-models` use `nargs='*'`: passing the flag **with no values** means *skip that
  stage* (e.g. `--cv-models` alone = fit-only, no CV; used for fit-only models like `recombined`).
- `--no-shared` skips the shared stages (shape/recovery/figure/table); run those separately via
  `shape_analysis.run` / `recovery.run` / `make_figure.run` / `make_table.run` on whatever models have
  finished — the analyze-as-you-go workflow.

---

## Checklist for validating a comparison result

1. **Step 0 passes** for every model in the comparison (`api.verify_all()` all-green).
2. **All models fit to the same converged standard** — compare `start_spread` and NLL-stability across a
   `maxiter` bump across models; no model is systematically hitting `maxiter` while others converge.
3. **AIC/BIC** computed with each model's own k and the same n; report ΔAIC/ΔBIC vs a fixed reference.
4. **CV-per-trial** reported alongside, scoped correctly: it measures parameter generalization, not
   block forecasting (no response leakage; the belief runs on the exogenous stimulus). Use a
   forward-chaining split if the goal is to test the learning dynamics.
5. **Shape reproduction** (TV distance + far-band mass + bimodality flag) reported as the qualitative
   axis, including the far-band NLL regime split.
6. **Recovery** at adequate `--n-sim`: parameter recovery clean (mechanism parameters demonstrated
   recoverable), model-recovery confusion matrix diagonal-dominant.

Only when 1–6 hold is a "model A beats model B" statement defensible.
