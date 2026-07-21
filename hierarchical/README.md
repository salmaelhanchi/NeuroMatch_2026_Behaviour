# Hierarchical Bayesian learning of prior confidence during perceptual estimation

Code for the NMA group project of the same name (Bhattacharya, Elhanchi, Wise,
Ruamkonthong, Kolchina). We model how people estimate the direction of briefly
presented random-dot motion when a learned *prior* over directions competes with
noisy *sensory evidence* — and ask what mechanism produces the **bimodal**
responses seen in the data (peaks near both the prior mean and the true
direction).

The full scientific abstract is in [`docs/project_abstract.md`](docs/project_abstract.md);
the per-task breakdown in [`docs/project_tasks.md`](docs/project_tasks.md). This
README is the map: what each model *is*, how they contrast, and how they bear on
the abstract's question. The empirical verdicts (which model wins on which
subject) live in the model docs under `docs/`, because they are still contested
and depend on fitting choices.

---

## The question the project asks

Laquitaine & Gardner (2018) explained the bimodality with a **Switching
observer**: on each trial the observer *selects* either the prior mean or the
sensory estimate rather than blending them. That account fits the prior as a
fixed, already-learned quantity and never models how the prior is acquired.

Our abstract asks the opposite:

> Does switching-like bimodal behaviour **require** an explicit selection
> mechanism, or can it **emerge** from adaptive hierarchical Bayesian inference
> once the observer is allowed to learn its prior online?

To answer that cleanly, the code implements four observer models, plus the
classical **Basic Bayesian** baseline they are all measured against. Two
questions are being separated, and each of the four is one cell in that 2×2:

1. **Is the prior strength fixed, or learned online from trial-to-trial feedback?**
2. **Is the cause-level read-out a *selection* (commit to prior *or* evidence) or
   an *integration* (one blended posterior)?**

---

## The models at a glance

The Basic Bayesian is the paper's baseline — likelihood × prior into one
posterior, MAP read-out, no switch and no learning; it's what the Switching
observer was built to beat. The other four are the switch-vs-integration /
fixed-vs-learned contrast above.

| Model | File (`observers/models/`) | Prior strength | Read-out | Free params | Spec doc |
|---|---|---|---|---|---|
| **Basic Bayesian** (baseline) | [`basic_bayesian.py`](observers/models/basic_bayesian.py) | fixed, fitted per block (4 values) | **integration** (no switch) | 9 | Laquitaine & Gardner (2018), [`docs/`](docs/Laquitaine_Gardner_2018_switching_observer.pdf) |
| **Switching observer** (the paper) | [`switching_observer.py`](observers/models/switching_observer.py) | fixed, fitted per block (4 values) | **selection** (switch) | 9 | Laquitaine & Gardner (2018), [`docs/`](docs/Laquitaine_Gardner_2018_switching_observer.pdf) |
| **Online switching observer** | [`online_switching_observer.py`](observers/models/online_switching_observer.py) | **learned online** (belief over strength) | **selection** (switch) | 6 | [`docs/generative-model.md`](docs/generative-model.md) |
| **Asymptote + transient** | [`asymptote_transient.py`](observers/models/asymptote_transient.py) | per-block levels **+ within-block transient** | **selection** (switch) | 11 | [`docs/asymptote_transient.md`](docs/asymptote_transient.md) |
| **Hierarchical Bayesian integration** | [`hb_integration.py`](observers/models/hb_integration.py) | **learned online** (belief over strength) | **integration** (no switch) | 7 | [`docs/hb_integration.md`](docs/hb_integration.md) |

Each model file is self-contained source logic for one observer; shared math
(circular statistics, the Girshick posterior look-up, the belief-grid update)
lives in [`observers/helpers/`](observers/helpers/) so the model files stay
focused on the mechanism.

---

## What each model does

### 1. Switching observer — the incumbent

The Laquitaine & Gardner (2018) model. On each trial the observer draws a noisy
sensory measurement, then **commits to one cause**: with a probability set by the
reliability ratio `k_prior / (k_prior + k_e)` it reports the prior mean (225°),
otherwise it reports the sensory estimate. It never averages the two. Add a lapse
rate and von Mises motor noise and you get the response distribution.

Because the observer *selects*, the two outcomes land in two places, so response
distributions are **bimodal within a single condition**. The prior strength
`k_prior` is a fixed fitted parameter — one per block width (SD 10/20/40/80°), so
four of them — and how the observer *came to know* the prior is not modelled.

### 2. Online switching observer — same switch, learned prior

Keeps the switch of model 1 but replaces the four fitted `k_prior` values with a
**belief distribution over prior strength that is updated trial by trial** from
the feedback direction revealed at the end of each trial (a predict/correct
filter with a volatility/forgetting knob `λ`). The switch weight is then the
reliability ratio evaluated under the *current* belief.

This isolates the *learning* ingredient: it is still a selection model, but the
prior strengths are now **emergent, not fitted**, so it has fewer parameters (6)
than the static model (9). Its signature is a **within-block learning transient**
in how often the prior is chosen. Full generative spec:
[`docs/generative-model.md`](docs/generative-model.md).

### 3. Asymptote + transient — a bridge model

A hybrid pointed to by the empirical learning curve. It keeps the static model's
flexible **per-block prior levels** (four asymptotes) *and* adds the online
model's **within-block transient** toward them: the effective prior strength
relaxes exponentially across a block,

```
k_eff(t) = k_asym[block] + (k_start − k_asym[block]) · exp(−t / τ)
```

with an asymmetric time constant (tightening vs loosening). It is still a switch
model; turning the transient off recovers the static observer exactly (it *nests*
model 1). 11 parameters. Details:
[`docs/asymptote_transient.md`](docs/asymptote_transient.md).

### 4. Hierarchical Bayesian integration — the abstract's proposal

The model the abstract actually proposes, and the one that imposes **no switch**.
It puts a **mixed hyper-prior** on the true direction — a peaked von Mises at the
prior mean plus a uniform floor,

```
p(θ | κ, α) = α · V(θ; 225°, κ)  +  (1 − α) · (1/360)
```

— and simply reads out the MAP of the ordinary Bayesian posterior (evidence
likelihood × this prior). Bimodality falls out **for free**: a measurement near
225° is best explained by the von Mises component and gets pulled toward the
prior; a measurement far from 225° is best explained by the uniform floor and
sits at the sensory evidence. The "switch" is a *derived responsibility* — which
mixture component explains the measurement — not a hand-coded competition. The
prior precision `κ` is learned online exactly as in model 2. 7 parameters.
Details: [`docs/hb_integration.md`](docs/hb_integration.md).

---

## How they contrast — and the falsifiable test

Placed on the 2×2 (learning × read-out):

|                       | **Selection read-out (switch)** | **Integration read-out (no switch)** |
|-----------------------|--------------------------------|--------------------------------------|
| **Prior fixed per block** | Switching observer (1) · Asymptote+transient (3, levels + transient) | — |
| **Prior learned online**  | Online switching observer (2) | HB integration (4) |

The switch family (1–3) and the integration model (4) *both* produce
bimodality, so a raw bimodality count cannot separate them. The discriminating
prediction the abstract names is about **where the two peaks come from**:

- A **switch** predicts two peaks *within individual trials* — the observer
  genuinely lands at the prior on some trials and at the evidence on others,
  regardless of the stimulus's distance from the prior.
- **Integration + online learning** predicts each *trial* is unimodal, and
  bimodality appears only when trials are **pooled across a block's learning
  transient**: early in a block the believed prior is weak (mass at the
  stimulus), late in a block it has sharpened (mass at the prior). A second,
  finer discriminator: under integration, reliance on the prior should **fall as
  the stimulus moves away from 225°** (a far measurement is poorly explained by
  the von Mises component), whereas the switch's prior weight is flat across
  direction at fixed reliability.

That is the crux of the project: models 2 and 3 exist so the comparison can
attribute any difference to *selection vs integration* rather than to *whether
the prior was learned*, holding the other ingredient fixed.

---

## Repository layout

```
hierarchical/
├── README.md                     ← you are here
├── pyproject.toml                makes `observers` a pip-installable package
├── requirements.txt
├── data/                         experimental trial data (motion-direction task)
├── docs/                         abstract, task list, model specs, source paper
├── notebooks/                    Colab notebooks (thin — they import observers)
├── results/
│   ├── figures/                  generated .png figures
│   └── fits/                     fitted-parameter JSON (one file per model family)
└── observers/                    the importable Python package
    ├── api.py                    ← one curated surface for notebooks
    ├── models/                   ← the four observer models, one file each
    ├── helpers/                  shared math + data loading + path constants
    ├── fitting/                  parameter estimation, recovery, human fits
    ├── verification/             self-checks that each model behaves as specified
    └── analysis/                 comparison plots and behavioural analyses
```

`observers/helpers/` holds everything the models lean on but that is not itself a
model: `circular.py` (von Mises / circular-distance math), `bayes_lookup.py` (the
Girshick MAP posterior look-up), `belief_grid.py` (the trial-by-trial belief
update over prior strength), `dataset.py` (load a subject's trials or simulate
synthetic ones), and `paths.py` (every data/results file location in one place).

---

## Notebooks (Google Colab)

The `notebooks/` folder is how less-technical group members use the models — no
command line needed. **Open a notebook in Colab and run it top to bottom.** Start
with `00_start_here.ipynb`; then `01_explore_data.ipynb` and
`02_model_comparison.ipynb`.

The one rule that keeps this from rotting: **notebooks contain no model logic.**
Every notebook's first cell pulls the latest code and installs the package; after
that, cells just call `observers.api`:

```python
from observers import api
api.verify_all()               # confirm every model behaves correctly
api.subjects_with_data()       # which subjects are in the dataset
trials = api.load_subject(1)   # one subject's trials (a DataFrame)
api.comparison_table()         # fair AIC per model per subject
api.plot_model_comparison()    # the 3-panel comparison figure (renders inline)
api.plot_switch_curve()        # within-block switch learning curve
```

Because the logic lives in `observers/` and the notebooks only import it,
"syncing" a notebook to newer code is just re-running the setup cell (it does a
`git pull`). There is exactly one copy of every model, so no one can accidentally
run a stale one. `observers/api.py` is the whole surface — read it or run
`help(api)` to see what's available.

Fits run inside Colab are **not** saved back to GitHub (Colab's filesystem is
temporary) — download them or push from a local checkout to keep them.

---

## Setup (local, for command-line use)

```bash
cd hierarchical
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # installs deps and makes `import observers` work anywhere
```

(or `uv venv && uv pip install -e .`.) `pip install -r requirements.txt` still
works if you only want the dependencies without installing the package.

## Running (command line)

Scripts are modules of the `observers` package, so **run them from this project
root with `python -m`** (not `python path/to/file.py`). Paths to data and results
are resolved absolutely, so results always land in `results/`.

```bash
# Verify each model behaves as its spec requires (fast, no fitting):
python -m observers.verification.verify_switching
python -m observers.verification.verify_online
python -m observers.verification.verify_hb_integration

# Inspect existing fits and regenerate the comparison figure:
python -m observers.fitting.online_fit_human --summary   # static + online AIC table
python -m observers.fitting.fair_refit table             # fair 4-model AIC table
python -m observers.analysis.plot_model_comparison       # writes results/figures/model_comparison.png

# Fit a subject (writes to results/fits/…):
python -m observers.fitting.hb_integration_fit human 1
python -m observers.fitting.asymptote_transient_fit human 1
```

---

## Extension models (Anirban's spec) — experimental

Beyond the four core models above, four further observers implement the
alternatives laid out in `anirban-modelling/Hypotheses_critique_and_alternatives.md`.
They are **built and verified but not yet in the fair all-12-subject comparison**
(a full multi-start batch is deferred). Full detail:
[`docs/new_models_manifest.md`](docs/new_models_manifest.md) (paths, params) and
[`docs/new_models_build_report.md`](docs/new_models_build_report.md) (verification
+ smoke-fits).

| Model | File (`observers/models/`) | What it adds | Params |
|---|---|---|---|
| **Causal-inference mixture** | `causal_inference_mixture.py` | mixing weight = per-measurement posterior responsibility `p(z=peaked\|m)` → switch rate depends on displayed direction (a falsifiable signature the switch lacks) | 7 |
| **Logistic-covariate mixture** | `logistic_mixture.py` | weight = fitted logistic of prior_std, coherence, interaction, recent error, cumulative prior reliance → can be non-monotonic in prior width + carry history/hysteresis | 11 |
| **Bimodal-likelihood control** | `bimodal_likelihood.py` | bimodal sensory likelihood (two lobes 180° apart), ordinary Bayesian integration, **no switch** — the Chetverikov & Jehee (2023) competitor | 10 |
| **Finite-sample readout** | `finite_sample.py` | report the mean of `n` posterior samples; `n=1` ≈ switch, `n→∞` = Basic Bayes — a resource-rational nest with `n` fitted | 10 |

```bash
# Verify each extension model (fast, no fitting):
python -m observers.verification.verify_causal_inference
python -m observers.verification.verify_logistic_mixture
python -m observers.verification.verify_bimodal_likelihood
python -m observers.verification.verify_finite_sample
```

Each nests a known baseline (verified to machine precision): causal-inference →
fixed-ratio mixture / switch; bimodal-likelihood `g=1` → Girshick Basic Bayes;
finite-sample `n=1` → the static Switching observer, `n→∞` → the posterior mean.
