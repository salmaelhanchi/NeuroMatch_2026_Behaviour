# Repo guide for LLM assistants

**Read this first.** You are helping a NeuroMatch 2026 team member analyze
human perceptual-estimation data with a set of Bayesian / switching observer
models. This document tells you where everything is, how to reach the models
and data through one Python API, and the exact schema of every file you will
touch. Everything here is verified against the live repo — prefer it over
guessing, and prefer the **`observers.api` module** over reading model source.

The science in one sentence: subjects estimate motion direction; a peaked
**prior** (mean 225°, block-varying width) competes with **sensory evidence**
(reliability set by motion coherence), and we compare models of how the two
are combined — from the paper's **Switch** (flip between prior and evidence)
to **hierarchical Bayesian** observers that *learn* the prior's width online.

---

## 1 · Repository layout

Everything lives under `hierarchical/` (the package root — all paths below are
relative to it; [`observers/helpers/paths.py`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/observers/helpers/paths.py) resolves them at runtime).

```
hierarchical/
├── observers/                 # THE PACKAGE — import this
│   ├── api.py                 # <- the front door. Almost everything is here.
│   ├── models/                # one file per observer model (the math)
│   ├── fitting/               # one *_fit.py per model (maximum-likelihood fitters)
│   ├── verification/          # verify_*.py — behavioral/identity checks per model
│   ├── comparison/            # the pipeline: registry, batch fit, CV, figures, tables
│   │   ├── registry.py        # model registry — single source of truth for models
│   │   ├── fit_batch.py       # fit one model×subject -> results/fits/comparison/
│   │   ├── cross_validate.py  # 5-fold block CV -> results/fits/comparison_cv/
│   │   ├── run_parallel.py    # runs many model×subject jobs across worker processes
│   │   ├── make_figure.py     # assembles the multi-panel comparison figure
│   │   └── make_table.py      # assembles the model-comparison table
│   └── helpers/               # circular stats, grids, dataset loader, paths
├── data/
│   └── data01_direction4priors.csv   # ALL trial data, 12 subjects, 83,213 rows
├── results/
│   └── fits/
│       ├── comparison/<model>/subject<N>.json      # point fits (per model, per subject)
│       └── comparison_cv/<model>/subject<N>_cv.json # cross-validation records
├── docs/                      # this file, the abstract, model notes
└── experiments/<person>/      # per-person analysis notebooks (read shared resources above)
```

**Rules of the repo:**
- Shared, reusable things (models, fits, data) live in `observers/`, `results/`,
  `data/`. Per-person analysis lives in `experiments/<person>/NN_<name>/` and
  *reads* the shared resources — it never copies the fit database or defines
  new models.
- A new model is added in **one place**: a spec function in
  [`observers/comparison/registry.py`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/observers/comparison/registry.py). Every stage (fitting, CV, figures,
  tables, the API) iterates the registry, so registering a model makes it
  visible everywhere.

---

## 1b · Background reading — the abstract and the paper

Two documents in `docs/` define what this project is doing and why. **Read
them (or have your user point you at them) before making scientific
judgments** — they are the ground truth for what the models are meant to show.

- **[`docs/project_abstract.md`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/docs/project_abstract.md)** — the team's NeuroMatch 2026 abstract (group
  "Posterior Motives"), title *"Hierarchical Bayesian Learning as a Mechanism
  for Perceptual Switching."* This is the project's own thesis: whether a
  Bayesian observer that **learns how much to trust its prior, trial by
  trial** can reproduce the bimodal estimates the paper attributed to a
  discrete prior-vs-likelihood switch. It states the two hypotheses the
  analyses test — the model should (i) reproduce both unimodal and bimodal
  estimate distributions, and (ii) recover the block-specific prior widths that
  change through learning. When a teammate asks "what are we trying to show?",
  this file is the answer.

- **[`docs/Laquitaine_Gardner_2018_switching_observer.pdf`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/docs/Laquitaine_Gardner_2018_switching_observer.pdf)** — the source paper
  (Laquitaine & Gardner, 2018, *Neuron*, "A Switching Observer for Human
  Perceptual Estimation"). This is the work the project builds on and argues
  with: it introduced the **Switch** model (`switch` in this repo) and the
  motion-direction estimation dataset. Consult it for the original model
  definition, the experimental design, and the results the team is reframing.

(There is also a scaffold of related papers and notes under
[`docs/anirban-modelling-scaffold/Papers/`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/tree/model-verification/hierarchical/docs/anirban-modelling-scaffold/Papers/) — including a second copy of the
Laquitaine & Gardner paper — if broader background is needed.)

---

## 2 · The models

Eight models are registered. Six are **fully fit** (usable in comparisons now);
two are **held out** — code-complete and verified, but not yet fit (do not use
them in a results table until someone runs the fitter).

| key | label | params | learns? | color | status |
|---|---|---|---|---|---|
| `switch` | Switch | 9 | no | `#30638e` | fit + CV (the paper's model) |
| `basic_bayes` | Basic-Bayes | 9 | no | `#a0a0a0` | fit + CV (always-integrate baseline) |
| `hb_adaptive` | HB-Adaptive | 6 | yes | `#d1495b` | fit + CV |
| `hb_rachel` | HB-Rachel | 7 | yes | `#edae49` | fit + CV (our main HB observer) |
| `hb_salma` | HB-Salma | 6 | yes | `#8e6c8a` | fit only (no CV) |
| `recombined` | Recombined | 7 | yes | `#66a182` | fit only (no CV) |
| `hierarchical_online` | Hier-Online | 8 | yes | `#3a7d44` | **held out — 0 fits** |
| `reliability_mixture` | Reliability-Mixture | 10 | yes | `#c17817` | **held out — 0 fits** |

What "learns" means: a learning model updates its belief about the prior's
width **trial by trial** from feedback; a non-learning model uses fixed
per-block parameters. `switch` and `basic_bayes` are the two non-learning
models.

Key distinctions to keep straight:
- **Switch** flips between using the prior OR the evidence each trial (a
  discrete heuristic). **HB models** integrate them, but *grade* the reliance
  on the prior — and the hierarchical ones learn the prior width online.
- **HB-Rachel** is the team's primary hierarchical observer (integrate-after,
  learns width, fixed mixing weight α). **HB-Salma** and **Recombined** are
  integrate-before variants. **HB-Adaptive** learns both α and the width.
- Bimodal vs unimodal response shapes come from the *combination rule* plus a
  uniform-floor component in the mixed prior — not from any switch. Broad
  evidence leaves the prior peak standing (two peaks); sharp evidence
  overwhelms it (one peak).

To see this table at runtime: `api.model_info()` (DataFrame) and
`api.list_models()` (the key list).

---

## 3 · The API — `from observers import api`

**This is the front door. Reach data and models through these functions
rather than importing model classes or reading JSON yourself.** The API owns
loading, prediction, and per-trial quantities; your notebook owns aggregation
and plotting.

### Load
```python
api.load_subject(2)          # -> DataFrame of subject 2's trials.
                             #    columns: motion_direction, motion_coherence,
                             #    prior_std, estimate_dir, session_id
api.load_fitted('hb_rachel', 2)   # -> (observer, record_dict) with fitted params
api.observed_distribution(2, direction=335, coherence=0.06, prior_std=80)
                             # -> empirical response histogram, density over 1..360.
                             #    any of direction/coherence/prior_std may be omitted
                             #    to pool over that factor.
```

### Inspect
```python
api.list_models()           # -> list of the 8 model keys
api.model_info()            # -> DataFrame: label, n_params, learns, color
api.fitted_subjects('hb_rachel')   # -> [subject ids that have a fit]
```

### Predict (per-trial model output)
```python
api.predict('hb_rachel', 2)         # -> np.ndarray (n_trials, 360): the predicted
                                    #    response distribution for every trial.
                                    #    ROW-ALIGNED with api.load_subject(2).
api.belief_trajectory('hb_rachel', 2)
                                    # -> DataFrame trial, believed_sd  (the LEARNED
                                    #    prior width per trial). hb_adaptive also
                                    #    returns believed_alpha. Raises ValueError
                                    #    for non-learning models (switch, basic_bayes).
```

### Fit / simulate
```python
api.fit_model('hb_rachel', 2, maxiter=400)   # refit from scratch (SLOW: seconds-minutes)
api.simulate('hb_rachel', 2, seed=0)         # generative: synthetic responses from the
                                             # fitted model (basis of parameter recovery).
                                             # design can be a subject id OR a DataFrame.
```

### Evaluate / compare
```python
api.results_table()                  # -> tidy DataFrame, ONE ROW per (model, subject):
                                     #    model, label, subject, n_trials, k, nll, aic, bic, cv_nll.
                                     #    Missing fits are simply absent. This is the
                                     #    main comparison object — group it yourself.
api.results_table(models=['switch','hb_rachel'], include_cv=True)
api.trial_logliks('hb_rachel', 2)    # -> per-trial log-likelihood array (n_trials,).
                                     #    ROW-ALIGNED with load_subject. Your escape
                                     #    hatch for any custom goodness-of-fit.
api.bias_variability(2)              # -> per-(coherence, prior_std, direction) DataFrame:
                                     #    n, mean_estimate, bias (signed toward prior),
                                     #    circ_sd. Uses correct 360-degree circular stats.
api.load_fitted_cv('hb_rachel', 2)   # -> the raw CV record dict (see schema below)
```

### Verify (does a model behave correctly?)
```python
api.verify_all()        # runs every model's identity/behavior checks -> {name: (passed, total)}
api.verify_switching(); api.verify_hb_rachel(); api.verify_basic_bayesian(); api.verify_online()
```

**Alignment guarantee.** `predict`, `trial_logliks`, and `belief_trajectory`
all return arrays row-aligned with `api.load_subject(subject_id)`. To attach
conditions to any per-trial array, just pull the columns from `load_subject`
and assign — same order, same length.

---

## 4 · The data

One CSV holds every trial for all 12 subjects:
[`data/data01_direction4priors.csv`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/data/data01_direction4priors.csv) — **83,213 rows**, prior mean fixed at 225°.

**Use `api.load_subject(id)`**, which returns a clean per-subject DataFrame
with the columns you actually need (and renames the raw estimate to a single
angle):

| column (from `load_subject`) | meaning |
|---|---|
| `motion_direction` | true stimulus direction, degrees (1..360) |
| `motion_coherence` | 0.06 / 0.12 / 0.24 — sets sensory reliability (higher = sharper evidence) |
| `prior_std` | block prior width: 10 / 20 / 40 / 80 degrees |
| `estimate_dir` | the subject's reported direction, degrees |
| `session_id` | session label (learning resets at session boundaries) |

The raw CSV has more columns (`trial_index`, `estimate_x/y`, `reaction_time`,
`prior_mean`, `subject_id`, …); `load_subject` derives `estimate_dir` from
`estimate_x/estimate_y` and keeps only what the models use. **Prefer
`load_subject` over reading the CSV directly.**

The experimental design: prior is a von Mises bump centered at 225° whose width
changes in blocks (SD 10/20/40/80). Feedback (the true direction) is shown each
trial, which is what lets a learning model recover the block width online.

---

## 5 · Fits and cross-validation files

### Point fits — `results/fits/comparison/<model>/subject<N>.json`
Maximum-likelihood fit of one model to one subject. Schema (real keys):

| key | type | meaning |
|---|---|---|
| `model`, `label` | str | model key and display label |
| `subject` | int | subject id |
| `n_trials` | int | trials this subject contributed |
| `nll` | float | negative log-likelihood at the fitted parameters (lower = better fit) |
| `aic`, `bic` | float | information criteria (penalize params: AIC=2k+2·NLL, BIC=k·ln(n)+2·NLL) |
| `k` | int | number of free parameters |
| `learns` | bool | whether this model learns online |
| `grid_deg` | int | angular grid resolution used for scoring (360 for all — comparable) |
| `params` | dict | the fitted parameters (model-specific keys, e.g. k_like, k_prior, alpha, lam) |
| `theta` | list | the raw parameter vector (fitter-internal) |
| `convergence` | dict | `converged`, `n_iter`, `n_feval`, `hit_maxiter` |
| `start_spread` | float | spread of NLL across multi-start seeds (0 = single start / all agreed) |
| `maxiter`, `seconds` | int/float | optimizer budget and wall time |

### Cross-validation — `results/fits/comparison_cv/<model>/subject<N>_cv.json`
5-fold **block** cross-validation (whole prior-width blocks held out), scoring
held-out per-trial NLL. Schema:

| key | type | meaning |
|---|---|---|
| `model`, `label`, `subject`, `k` | | as above |
| `n_trials` | int | total trials |
| `folds` | int | number of CV folds (5) |
| `cv_nll` | float | summed held-out NLL across folds (lower = better generalization) |
| `cv_per_trial` | float | `cv_nll / n_trials` — use THIS to compare across subjects |
| `fold_detail` | list | per fold: `fold`, `n_test`, `held_out_nll`, `per_trial` |
| `maxiter`, `seconds` | | budget and wall time |

### Coverage right now
| model | point fits | CV |
|---|---|---|
| switch, basic_bayes, hb_adaptive, hb_rachel | 12/12 | 12/12 |
| hb_salma, recombined | 12/12 | **none** (point-fit only) |
| hierarchical_online, reliability_mixture | **0** | **0** (held out) |

**Consequences for analysis:**
- In-sample comparison (NLL / AIC / BIC) can use all six fit models.
- Out-of-sample comparison (CV) is limited to the four with CV files. Do not
  put hb_salma / recombined in a CV table.
- All fits are on the **360° grid**, so their NLL/AIC/BIC are directly
  comparable across models.
- **Lead with CV** for model selection: in-sample AIC rewards Switch's extra
  parameters, but held-out NLL is the honest test and is where the hierarchical
  models are competitive.

---

## 5b · How the models were fitted and cross-validated (methodology)

This is *how* the numbers in those JSON files were produced — useful when a
teammate asks "how were these fit?" or needs to defend the method.

### Likelihood
Every model defines a **per-trial log-likelihood**: `log p(estimate_t |
stimulus_t, coherence_t, prior_block_t, params)`. The trial log-likelihoods are
summed to a total log-likelihood, and fitting **minimizes the negative
log-likelihood (NLL)** — i.e. maximum-likelihood estimation. There is no
binning of subjects or conditions: the fit uses each trial individually, in
its real temporal order.

For **learning** models this ordering matters. The observer runs a **belief
filter** over the whole ordered trial sequence, updating its believed prior
width from the feedback (the true direction, shown each trial) as it goes. So
the likelihood of trial *t* depends on everything the observer learned from
trials 1…*t*−1. Non-learning models (`switch`, `basic_bayes`) have no such
state — each trial's likelihood is independent given the fixed parameters.

### Optimizer
- **Nelder-Mead simplex** (`scipy.optimize.minimize`), tolerances
  `xatol = fatol = 1e-2`, default **`maxiter = 400`** function iterations.
- **Multi-start: 10 starts** for the reported point fit (`N_STARTS = 10`,
  matching Laquitaine & Gardner's ~10 initial points). Starts are jittered
  around a sensible base guess; the best (lowest-NLL) result is kept. The
  spread of NLL across starts is recorded as **`start_spread`** in the fit JSON
  — `0.0` means all starts agreed (a good sign the optimum is not a fluke of
  initialization).
- All models are scored on the **360° angular grid**, so their NLL / AIC / BIC
  are directly comparable regardless of any coarser grid a model might use
  internally for speed.

### AIC / BIC
Computed from the fitted NLL and the parameter count *k*:
`AIC = 2k + 2·NLL`, `BIC = k·ln(n_trials) + 2·NLL` (both verified against the
stored values). Lower is better; both penalize extra parameters, BIC more
harshly. Parameter counts differ (Switch 9, HB-Rachel 7, HB-Adaptive 6, …), so
these criteria — not raw NLL — are the fair in-sample comparison.

### Cross-validation (the out-of-sample test)
- **5-fold, blocked by prior width.** A "block" is a maximal run of constant
  `prior_std`; whole blocks are grouped into 5 contiguous folds
  (`_block_folds`). This holds out **entire prior-context blocks**, not random
  trials — the honest test of whether a model generalizes to a prior width it
  did not fit to, rather than interpolating within a block it has already seen.
- For each fold: refit on the training blocks (**single start** in CV, not 10 —
  10 starts × 5 folds × 12 subjects would be prohibitive), then score the
  **held-out** trials' summed NLL. `cv_nll` sums the held-out NLL across folds;
  `cv_per_trial = cv_nll / n_trials` is the number to compare across subjects.
- **Subtlety for learning models:** even on held-out trials, the belief filter
  still runs over the *full* ordered sequence, so the observer's learned state
  is realistic at every trial. The fold only controls which trials'
  likelihoods are *scored* (test) vs *optimized* (train) — it does not blind
  the learner to feedback it would have had in real time.

### What this means when reading results
- **In-sample AIC/BIC** rewards flexible models; **CV** is the honest test.
  Lead model selection with CV. Switch's 9 parameters win many subjects
  in-sample but that edge shrinks or reverses out-of-sample — the reframing
  the project argues for.
- `convergence.hit_maxiter = true` on a fit means the optimizer ran out of
  iterations before converging — treat that fit's NLL as an upper bound and
  consider a higher `--maxiter`.
- Only `switch`, `basic_bayes`, `hb_adaptive`, `hb_rachel` have CV files;
  `hb_salma` and `recombined` are point-fit only, so they belong in in-sample
  tables but not CV tables.

---

## 6 · Custom calls (when no API helper fits)

The API covers the common needs; for anything else, reach the model object
through the registry. A `ModelSpec` is the per-model handle.

```python
from observers.comparison.registry import build_registry, load_subject

spec = build_registry(['hb_rachel'])['hb_rachel']   # a ModelSpec
# ModelSpec fields:   name, label, n_params, learns, grid_deg, color
# ModelSpec methods:  fit(data, maxiter, mask) -> FitResult
#                     rebuild(params_dict)      -> observer
#                     predict_distributions(obs, data) -> (n_trials, 360)
#                     trial_logliks(obs, data)  -> (n_trials,)
#                     simulate(obs, design, seed) -> dict of synthetic trial arrays

obs, rec = api.load_fitted('hb_rachel', 2)           # obs = fitted observer
data = load_subject(2)                               # NOTE: registry.load_subject
                                                     # returns a DICT of arrays;
                                                     # api.load_subject returns a DataFrame
dists = spec.predict_distributions(obs, data)        # same as api.predict, but you
                                                     # control the input trials
```

Gotcha: **`registry.load_subject` returns a dict of numpy arrays**, while
**`api.load_subject` returns a DataFrame**. The `ModelSpec` methods and
`trial_logliks` consume the dict form; `simulate` with a custom design wants
the DataFrame. The `api.*` wrappers bridge this for you — only drop to the
registry when you need trial-level control the API doesn't expose.

---

## 7 · Running the pipeline (rarely needed from a notebook)

These are batch/CLI tools, run from the repo root with `PYTHONPATH` set — not
usually called inside a notebook, but here for completeness:

```bash
# fit one model×subject (writes results/fits/comparison/<model>/subject<N>.json)
python -m observers.comparison.fit_batch --model hb_rachel --subject 2 --maxiter 400
# cross-validate (writes results/fits/comparison_cv/...)
python -m observers.comparison.cross_validate --model hb_rachel --subject 2 --folds 5
# many jobs across workers
python -m observers.comparison.run_parallel --workers 4
# assemble the figure / table from existing fits
python -m observers.comparison.make_figure
python -m observers.comparison.make_table
```

Fitting is the expensive step (seconds to minutes per model×subject;
`hierarchical_online` is minutes). Everything downstream reads the saved JSON,
so once fits exist, analysis and plotting are fast.

---

## 7b · If functionality is missing: edit the repo on a branch for your user

If your user needs something this codebase does not yet do — a new API helper,
a model tweak, a new analysis, a bug fix — you can **make the change for them
on a fresh git branch**, rather than only working around it inside the
notebook. The repo is already a full git checkout (the notebook's setup cell
cloned it), so you have everything you need.

**Do this when** the gap is in the shared code (a missing [`observers/api.py`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/observers/api.py)
helper, a model that needs a fix, a pipeline option). **Don't do this** for
one-off analysis that belongs in the user's own notebook — write that inline.

Workflow (run these in a notebook cell or terminal, from the repo root):

```bash
cd /content/NeuroMatch_2026_Behaviour        # wherever the repo was cloned
git checkout -b feature/<short-description>   # NEVER edit model-verification/main directly
# ... make your edits to observers/… with the user watching ...
python -c "from observers import api; api.verify_all()"   # prove you didn't break anything
git add -p                                    # stage deliberately, review each hunk
git commit -m "Add <thing> (assisted)"        # note assistance in the message
```

Then **hand control back to the user to publish** — they have the git
credentials, you do not:

```bash
git push -u origin feature/<short-description>
# then open a pull request against model-verification on GitHub
```

Ground rules that keep this safe and reviewable:
- **Always a new branch**, never a direct commit to `model-verification` or
  `main`. The user reviews before anything merges.
- **Keep the change small and single-purpose.** One branch = one feature or
  fix. Mechanical reorganization (file moves, path edits) goes in a *separate*
  commit from substantive code changes, so history stays legible.
- **Verify before you commit:** run `api.verify_all()` and, if you touched a
  fitted model, refit one subject and confirm it still scores. Never commit a
  change that leaves `verify_all()` failing.
- **You cannot push.** The push and the pull request are the user's action.
  Prepare the branch and the commit; tell them the exact `git push` command and
  that they should open a PR. Do not attempt to configure credentials.
- If your edit changes how a model fits or scores, say so explicitly — existing
  fit/CV JSON files were produced by the *old* code and may need regenerating.

---

## 7c · Where to save outputs — the `experiments/` directory

Your user has a dedicated place for analysis outputs: **`experiments/<user>/`**
(for the repo owner, `experiments/rachel/`). Put figures, tables, notebooks,
CSVs, and short write-ups there — **not** in the repo root, and **not** mixed
into `observers/`, `results/`, or `data/` (those hold shared, reusable
resources only).

Convention:
- One **numbered, self-contained subfolder per piece of work**:
  `experiments/<user>/NN_<short_name>/` (e.g. `03_prior_recovery/`). Numbering
  keeps them ordered.
- Inside it, keep that experiment's own outputs — put images in a `figures/`
  subfolder beside the notebook, not loose.
- Each experiment folder **reads** the shared resources by relative path
  (`../../observers/`, `../../results/fits/`, `../../data/`,
  `../../docs/`) — it never copies the fit database or defines new models. New
  models always go in `observers/` + the registry (see §7b), never inside an
  experiment folder.
- A short `README.md` in each folder (one-line summary + a `Tags:` line) makes
  the tree self-documenting.

So when you generate a figure or a results table for your user, save it under
`experiments/<user>/NN_<name>/` and reference the shared data through the API
— that keeps their experiment reproducible and their repo clean.

---

## 8 · Quick reference — "how do I…?"

| task | call |
|---|---|
| list the models | `api.list_models()` / `api.model_info()` |
| get one subject's trials | `api.load_subject(id)` |
| model's predicted distribution per trial | `api.predict(key, id)` |
| observed response histogram | `api.observed_distribution(id, ...)` |
| the learned prior width over trials | `api.belief_trajectory(key, id)` |
| per-trial log-likelihood | `api.trial_logliks(key, id)` |
| bias & variability by condition | `api.bias_variability(id)` |
| the big comparison table | `api.results_table()` |
| cross-validation numbers | `api.load_fitted_cv(key, id)` or `results_table(include_cv=True)` |
| refit a model | `api.fit_model(key, id)` |
| simulate synthetic data | `api.simulate(key, design, seed)` |
| check a model behaves right | `api.verify_all()` |
| read the project's thesis / hypotheses | [`docs/project_abstract.md`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/docs/project_abstract.md) |
| read the source paper (Switch model, dataset) | [`docs/Laquitaine_Gardner_2018_switching_observer.pdf`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/docs/Laquitaine_Gardner_2018_switching_observer.pdf) |
| add missing functionality for the user | new branch → edit `observers/` → verify → user pushes (§7b) |
| save a figure/table/notebook | `experiments/<user>/NN_<name>/` (§7c) |

**When unsure:** call `api.model_info()` and `api.results_table()` first to see
what exists, load one subject with `api.load_subject`, and build up from the
per-trial arrays. Everything is row-aligned; the API returns numbers, and you
aggregate and plot.
