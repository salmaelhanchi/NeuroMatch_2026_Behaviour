# NeuroMatch 2026 Behaviour

Current modeling scaffold root:
[repository root](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour)

This repository currently contains a scaffold for building and validating a hierarchical Bayesian observer model for perceptual estimation behavior.

The core modeling question is:

```text
previous block context
-> initial prior confidence at current block start
-> within-block confidence learning
-> predicted estimate distribution
```

The scaffold is intentionally modular. The goal is to keep the data path, equation path, and modeling path easy to inspect before moving into full model fitting and comparison.

## Main Files

| File | Purpose |
|---|---|
| [hierarchical_observer_scaffold.ipynb](hierarchical_observer_scaffold.ipynb) | Main notebook. Loads the remote behavioral CSV, builds analysis tables, defines model components, and runs scaffold checks. |
| [scaffold_recent_changes.md](scaffold_recent_changes.md) | Change-tracking file. Records what changed, what names are stable, and what to check if something breaks. |
| [scaffold_design_guide.md](scaffold_design_guide.md) | Design guide. Explains the scaffold vision, stable functions, and how future model edits should be organized. |
| [simple_simulation_checks_interpretation.md](simple_simulation_checks_interpretation.md) | Interpretation of the simple simulation cell. Explains why the output supports the precision-not-mean update check. |
| [abstract_draft.md](abstract_draft.md) | Current project abstract draft. |
| [tasks_draft.md](tasks_draft.md) | Draft task checklist for HB implementation, fitting, validation, switching observer, comparison, behavioral validation, and presentation. |

The scaffold files are intentionally tracked at the repository root. They should not be nested under an extra folder in GitHub.

## Current Checklist Coverage

| Checklist item | Current implementation status | Primary notebook cells |
|---|---|---|
| **HB implementation** | Partially implemented as a scaffold. The notebook defines model pieces and fitting contracts, but does not yet perform full hierarchical inference. | Cells 22-30 |
| **Build the hierarchical Bayesian model so that it learns prior confidence across trials and generates predicted perceptual estimates.** | Implemented at scaffold level. Prior confidence is represented trial-by-trial and used to generate one-trial predicted estimate distributions. Full parameter fitting is still future work. | Cells 12, 15, 17, 23, 25, 26, 28, 30 |
| **Define prior, sensory likelihood, posterior, motor noise, and lapse components.** | Implemented through named circular model functions and `build_trial_components(...)`. | Cells 23, 26 |
| **Implement trial-by-trial updating of prior confidence.** | Implemented through `add_prior_confidence_scaffold(...)` for data-derived proxy columns and `update_online_prior_state(...)` for the explicit online update. | Cells 12, 25, 28 |
| **Preserve circular angle calculations.** | Implemented through angle wrapping, circular means, von Mises-like densities, circular convolution, and circular estimate probability lookup. | Cells 12, 23 |
| **Test the model on simple simulated conditions.** | Implemented in the `Simple simulation checks` section using a toy transition from `prior_std = 10` to `prior_std = 80`. | Cells 27-28 |
| **Confirm that the update affects prior precision, not only prior mean.** | Implemented with assertions showing `prior_mean_unchanged == True` and `prior_precision_changed == True`. | Cell 28 |

## Notebook Cell Map

| Cell | Section or main action | Checklist connection |
|---|---|---|
| 1 | Notebook title and scaffold purpose | Provides context for the HB scaffold. |
| 2 | Package availability check | Setup only. Not a modeling checklist item. |
| 3 | Imports | Setup only. |
| 4 | Simple local path setup heading | Setup only. |
| 5 | Local path setup for PDFs and outputs | Keeps local paper paths available while data loads remotely. |
| 6 | Scaffold contract heading | Defines the notebook contract. |
| 7 | Input/output columns, predictor groups, and `EQUATION_MAP` | Maps paper steps to dataset columns and model functions. Supports HB implementation traceability. |
| 8 | Load and inspect raw data heading | Data setup. |
| 9 | Remote CSV loader | Loads the behavioral data from GitHub raw URL. |
| 10 | Required-column checks and raw-data summary | Confirms the expected dataset columns exist before modeling. |
| 11 | Feature builders heading | Starts data-to-model feature construction. |
| 12 | Feature builder functions | Preserves circular angle calculations and creates prior-confidence proxy columns. |
| 13 | Prepared-trial validation | Confirms derived columns exist after feature building. |
| 14 | Analysis-facing tables heading | Starts model-ready table creation. |
| 15 | `model_df` and hierarchical index construction | Defines the model table used for later fitting and hierarchical design. |
| 16 | Carryover-to-estimate path heading | Focuses on between-block carryover. |
| 17 | Block transition and confidence-flow previews | Shows previous block context -> current block confidence path. |
| 18 | Descriptive checks before fitting heading | Data inspection before modeling. |
| 19 | Descriptive summaries and plots | Helps inspect estimation error by condition before fitting. |
| 20 | Experimental Distribution Compiler heading | Starts empirical target construction. |
| 21 | Distribution compiler functions and outputs | Builds experimental distributions that later model predictions should match. |
| 22 | Core circular model functions heading | Starts model equation implementation. |
| 23 | Circular model functions | Defines prior, sensory likelihood, posterior, readout, motor noise, lapse, and circular probability lookup. |
| 24 | Hierarchical and switching hooks heading | Starts model variants and learning hooks. |
| 25 | `ObserverParams`, model registry, responsibility, and online update | Defines model parameters and implements confidence-only online updating through `update_online_prior_state(...)`. |
| 26 | One-trial prediction path | Uses trial columns to build model components and generate probability of the observed estimate. |
| 27 | Simple simulation checks heading | Starts the direct scaffold test section. |
| 28 | Simple simulation code and assertions | Tests simulated conditions and confirms prior precision changes while prior mean stays fixed. |
| 29 | Fitting interface placeholder heading | Marks where future fitting should be added. |
| 30 | Balanced subset and fit contract | Describes what will be estimated first; this is not a fit result yet. |
| 31 | Notebook editing guide | Explains which functions to change when testing new modeling theories. |

## How To Interpret The Current State

The scaffold has enough structure to show how the dataset connects to the proposed model:

```text
raw behavioral CSV
-> prepared trial table
-> block and transition context
-> prior confidence proxy path
-> circular model components
-> predicted estimate distribution
```

The notebook does not yet prove that the HB model fits the real data.

Evidence of a real fitted model would require outputs such as:

```text
optimized or sampled parameters
negative log-likelihood from fitted parameters
AIC/BIC or cross-validation
observed-vs-predicted estimate distributions
parameter recovery
convergence diagnostics
```

The current final fitting-contract cell should be interpreted as a plan for fitting, not as fit evidence.

## Explanation Of Tracking Files

`scaffold_recent_changes.md` is the change log for the scaffold. Use it to see what was recently added, what names should stay stable, and what to check first when cells fail.

`scaffold_design_guide.md` explains the intended architecture. Use it when deciding where a new function should go or when transferring the project to a new session.

`simple_simulation_checks_interpretation.md` explains the output of the simulation-check cell. It is specifically tied to these checklist items:

```text
Test the model on simple simulated conditions.
Confirm that the update affects prior precision, not only prior mean.
```

## Stable Data Source

The notebook loads data from:

```text
https://raw.githubusercontent.com/steevelaquitaine/projInference/refs/heads/gh-pages/data/csv/data01_direction4priors.csv
```

The scaffold should continue to load from the remote CSV unless there is a clear reason to change the data source.

## Next Modeling Steps

The next substantial implementation step is to turn the scaffold into a real fitted model:

```text
fit basic Bayesian model
fit switching observer model
fit hierarchical prior-confidence model
compare models using likelihood and predictive error
run parameter recovery
compare observed and predicted estimate distributions
```
