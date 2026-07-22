# Hierarchical Bayesian Learning as a Mechanism for Perceptual Switching

**Team Posterior Motives** — Bhattacharya, Elhanchi, Wise, Ruamkonthong, Kolchina
(NeuroMatch Academy group project)

We model how people estimate the direction of briefly presented random-dot
motion when a learned *prior* over directions competes with noisy *sensory
evidence*, and ask what mechanism produces the **bimodal** responses seen in the
data — peaks near both the prior mean (225°) and the true direction.

The submitted abstract is in [`docs/project_abstract.md`](docs/project_abstract.md).
The empirical verdicts (which model wins on which subject) live in the
`experiments/` record, because they depend on fitting choices and are still being
finalised. **This README is the map:** the question, the models in the
comparison, how to reproduce the fits, and where everything lives.

---

## TL;DR for teammates

- **Code you import** lives in [`observers/`](observers/) — one installable
  package. Notebooks and scripts call it; they never redefine model logic.
- **The model comparison** (fit → cross-validate → table → figures) is the
  [`observers/comparison/`](observers/comparison/) sub-package. One command
  reruns it end-to-end: `python -m observers.comparison.run_parallel`.
- **Fitted results** are the shared database under
  [`results/fits/`](results/fits/) — one JSON per model × subject, expensive to
  regenerate, so committed and kept in one place.
- **Each person's exploratory work** is a self-contained numbered folder under
  [`experiments/rachel/`](experiments/rachel/) (see its
  [README](experiments/rachel/README.md) for the index).
- **To add a model:** write its `observers/models/` file + fitter, then add one
  `ModelSpec` entry to `observers/comparison/registry.py`. Every pipeline stage
  (fits, CV, table, figures, `observers.api`) picks it up automatically.

---

## The question

Laquitaine & Gardner (2018) explained the bimodality with a **Switching
observer**: on each trial the observer *selects* either the prior mean or the
sensory estimate rather than blending them. That account fits the prior as a
fixed, already-learned quantity and never models how the prior is acquired.

Our abstract asks the opposite:

> Can a Bayesian observer that **learns how much to trust its prior, trial by
> trial**, reproduce the bimodal estimates previously attributed to an explicit
> prior-versus-likelihood switch?

The model the abstract proposes holds two priors at once as a mixed hyper-prior
(a peaked von Mises at 225° plus a uniform floor) and adds a hyperparameter,
updated over trials, for how much confidence to place in the informed vs. the
naive component. Bimodality then falls out of ordinary Bayesian integration — no
hand-coded switch — and the switch is reframed as *graded* integration.

Two design axes separate the models, and the comparison is built to attribute
any difference to the right one:

1. **Is the prior strength fixed, or learned online from feedback?**
2. **Is the read-out a *selection* (commit to prior *or* evidence) or an
   *integration* (one blended posterior)?** For the integrators, a further
   distinction matters — whether the confidence belief is folded in
   *before* the Bayesian read-out (**integrate-before**) or the posterior is
   read out first and combined *after* (**integrate-after**).

---

## The models in the comparison

These are the models the pipeline actually fits and scores — the registry in
[`observers/comparison/registry.py`](observers/comparison/registry.py) is the
single source of truth, and its insertion order is the canonical display order.

| Key | Label | Learns prior? | Read-out | Params | Role in the story |
|---|---|---|---|---|---|
| `switch` | Switch | no (fixed per block) | **selection** | 9 | Laquitaine & Gardner's incumbent; the model to beat |
| `basic_bayes` | Basic-Bayes | no (fixed per block) | integration (always) | 9 | Paper baseline; unimodal, no switch |
| `hb_adaptive` | HB-Adaptive | **yes — α and κ** | integration (integrate-after) | 6 | Joint α+κ online learning, integrate-after |
| `hb_rachel` | HB-Rachel | yes — κ (α fixed) | integration (integrate-after) | 7 | Fixed-confidence integrator (was `hb_integration`) |
| `hb_salma` | HB-Salma | yes — geometric forget | integration (integrate-before) | 6 | 72-bin belief, scored on the 360° grid |
| `recombined` | Recombined | yes | integration (integrate-before) | 7 | Rachel's engine + Salma's integrate-before read-out |
| `hierarchical_online` | Hier-Online | yes — mean **and** width | integration (mixture prior) | 8 | Mixture prior with online-learned mean *and* width |

Placed on the 2×2 of *learning* × *read-out*:

|                          | **Selection (switch)** | **Integration (no switch)** |
|--------------------------|------------------------|-----------------------------|
| **Prior fixed per block**| Switch                 | Basic-Bayes                 |
| **Prior learned online** | *(none currently)*     | HB-Adaptive · HB-Rachel · HB-Salma · Recombined · Hier-Online |

The point of carrying several integrators is that they all produce bimodality,
so a raw bimodality count cannot separate switch from integration. The abstract's
comparison rests on a learning Bayesian observer accounting for **inter-subject
variability**, and specifically predicts it will (i) **reproduce both unimodal
and bimodal** estimate distributions, and (ii) **recover the block-specific prior
widths** that change through learning across blocks — a principled account of how
confidence in the prior evolves, reframing the switch's discrete strategy as
graded integration. A useful mechanistic discriminator: under integration +
online learning each *trial* is unimodal and bimodality emerges only when trials
are **pooled across a block's learning transient**, whereas a switch lands at
prior-or-evidence within individual trials.

Model equations and derivations:
[`experiments/rachel/03_model_equations/`](experiments/rachel/03_model_equations/).

## Repository layout

```
hierarchical/
├── README.md                     ← you are here
├── pyproject.toml                makes `observers` a pip-installable package
├── requirements.txt
│
├── observers/                    THE PACKAGE — all model logic
│   ├── api.py                    one curated surface for notebooks (see `help(api)`)
│   ├── models/                   one file per observer
│   ├── fitting/                  per-model maximum-likelihood fitters
│   ├── verification/             self-checks that each model matches its spec
│   ├── analysis/                 behavioural analyses & comparison plots
│   ├── helpers/                  shared math, data loading, path constants
│   └── comparison/               the model-comparison pipeline (see below)
│
├── results/
│   ├── fits/
│   │   ├── comparison/<model>/subject<N>.json       point fits (one per model×subject)
│   │   ├── comparison_cv/<model>/subject<N>_cv.json block-fold cross-validation
│   │   ├── run_manifest.json                        provenance of the latest run
│   │   └── manifests/                               immutable per-run archive
│   ├── figures/                  generated comparison figures
│   └── logs/                     run logs (git-ignored; audit trail only)
│
├── experiments/<name>/           one folder per teammate; inside, numbered
│                                 self-contained exploration folders (e.g.
│                                 experiments/rachel/01_model_review …);
│                                 see each experiments/<name>/README.md for its index
│
├── docs/                         abstract, source paper, Anirban scaffold, model notes
├── data/                         the motion-direction trial data
└── notebooks/                    Colab notebooks (thin — they import observers)
```

**Shared vs. experiment-specific.** Top-level directories hold only *shared,
reusable* resources: the `observers/` code, the `results/fits/` parameter
database, `docs/`, `data/`, `notebooks/`. Anything exploratory — one-off
analyses, per-experiment figures and reports — belongs in a numbered folder
inside your own `experiments/<name>/` space (see below).

`observers/helpers/` holds everything the models lean on but that is not itself a
model: circular / von Mises math, the Girshick MAP posterior look-up, the
trial-by-trial belief-grid update, data loading, and `paths.py` (every
data/results location in one place).

---

## Adding your own experiments

Each teammate gets a top-level space under `experiments/<name>/` (so far only
`experiments/rachel/` exists — Anirban, Salma, Romi, and Valeria: make yours).
Keep exploratory work there, not at the repo root, so the shared directories
stay clean. The convention, mirroring `experiments/rachel/`:

1. **Make your space** the first time: `experiments/<yourname>/`, with a short
   `README.md` that indexes your folders (copy the structure of
   [`experiments/rachel/README.md`](experiments/rachel/README.md)).
2. **One folder per piece of work**, numbered and named:
   `experiments/<yourname>/01_short_description/`. Each folder is
   **self-contained** — it holds *all* of its own output (figures, tables,
   reports, notebooks) and any experiment-specific derived fits in a local
   `results/` subfolder.
3. **Give each folder a `README.md`** with a one-line summary and a `Tags:` line
   (facets `type:`, `claim:` — see `experiments/rachel/README.md` for the
   vocabulary), so the work is discoverable without opening it.
4. **Read shared resources, don't copy them.** From inside an experiment folder
   the shared code and data are two levels up: `../../observers/`,
   `../../results/fits/comparison/<model>/subject<N>.json`, `../../data/`,
   `../../docs/`. The shared fitted-parameter database is expensive to
   regenerate — read it, don't duplicate it.
5. **Never edit another person's `experiments/<name>/` folder.** Model code and
   the fit database are shared and edited together; each person's `experiments/`
   space is theirs alone.

New models still go in `observers/` + the registry (see the top of this README),
not in an experiment folder — an experiment *uses* the shared models, it doesn't
redefine them.

---

## The comparison pipeline (`observers/comparison/`)

The registry-driven pipeline that produces the AIC/BIC/CV comparison. Every
stage iterates the registry, so all registered models flow through automatically.

| Module | What it does | Output |
|---|---|---|
| `registry.py` | defines every model as a `ModelSpec` (fit, NLL, simulate) | — |
| `fit_batch.py` | maximum-likelihood point fit, per model × subject (resumable) | `results/fits/comparison/<model>/subject<N>.json` |
| `cross_validate.py` | 5-fold **block** CV — holds out whole prior-width blocks | `results/fits/comparison_cv/<model>/subject<N>_cv.json` |
| `shape_analysis.py` | response-distribution shape / bimodality metrics | — |
| `recovery.py` | parameter-recovery / model-recovery checks | — |
| `make_table.py` | AIC/BIC/CV leaderboard | `results/fits/model_comparison_table.{md,csv}` |
| `make_figure.py` | the multi-panel comparison figure | `results/figures/` |
| `bimodality_test.py` | conditioned bimodality test | — |
| `run_all.py` | serial end-to-end orchestrator | all of the above |
| `run_parallel.py` | parallel driver (subjects across workers) + provenance manifest | all of the above |
| `fit_monitor.py` | live health/stall dashboard for long fit jobs | `results/logs/monitor/` |

**Fit data currently in the repo:** point fits for all six committed models ×
12 subjects; block-fold CV for `switch`, `basic_bayes`, `hb_adaptive`, and
`hb_rachel` × 12 subjects.

**Reproduce the whole comparison** (resumable — skips any model×subject whose
JSON already exists; delete a JSON or pass `--force` to refit):

```bash
# from this directory, with the package installed (see Setup):
PYTHONPATH="$(pwd)" python -m observers.comparison.run_parallel --workers 3
```

`run_parallel` writes an immutable manifest per run under `results/fits/manifests/`
(git commit, library versions, config, grids) plus `run_manifest.json` as the
"latest" pointer — so a Methods section or a reproducer can always read exactly
what produced a given batch.

---

## Notebooks (Google Colab)

`notebooks/` is the no-setup way to use the models — everything runs in the
browser, no command line or local install needed. **Open a notebook in Colab and
run it top to bottom**, starting with
`00_start_here.ipynb`, then `01_explore_data.ipynb`, `02_model_comparison.ipynb`,
`03_validate_models.ipynb`, and `04_switching_matches_paper.ipynb`.

The rule that keeps this from rotting: **notebooks contain no model logic.** The
first cell pulls the latest code and installs the package; every later cell just
calls `observers.api`:

```python
from observers import api
api.verify_all()               # confirm every model behaves as specified
api.subjects_with_data()       # which subjects are in the dataset
trials = api.load_subject(1)   # one subject's trials (a DataFrame)
api.results_table()            # AIC/BIC/CV leaderboard
api.plot_model_comparison()    # the comparison figure (renders inline)
```

Because the logic lives in `observers/` and notebooks only import it, "syncing"
is just re-running the setup cell. There is exactly one copy of every model, so
no one can accidentally run a stale one. Read `observers/api.py` or run
`help(api)` for the full surface. Fits run inside Colab are **not** saved back to
GitHub (Colab's filesystem is temporary) — push from a local checkout to keep them.

---

## Setup (local / command line)

```bash
cd hierarchical
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # installs deps and makes `import observers` work anywhere
```

(or `uv venv && uv pip install -e .`.) `pip install -r requirements.txt` installs
the dependencies without the package.

Scripts are modules of the `observers` package, so **run them from this project
root with `python -m`** (not `python path/to/file.py`); data and results paths
resolve absolutely, so outputs always land under `results/`.

```bash
# Verify each model behaves as its spec requires (fast, no fitting):
python -m observers.verification.verify_switching
python -m observers.verification.verify_hb_rachel
python -m observers.verification.verify_basic_bayesian

# Fit one model for one subject (writes to results/fits/comparison/…):
python -m observers.comparison.fit_batch --models hb_adaptive --subjects 1

# Rebuild the leaderboard and figures from existing fits:
python -m observers.comparison.make_table
python -m observers.comparison.make_figure
```

---

## Reference

Laquitaine, S. & Gardner, J. L. (2018). *A Switching Observer for Human
Perceptual Estimation.* **Neuron** 97(2), 462–474. PDF in
[`docs/`](docs/Laquitaine_Gardner_2018_switching_observer.pdf).
