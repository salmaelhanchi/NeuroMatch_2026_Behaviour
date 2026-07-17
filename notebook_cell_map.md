# Notebook Cell Map

Notebook:
[hierarchical_observer_scaffold.ipynb](hierarchical_observer_scaffold.ipynb)

This file maps each notebook cell to its plain-language purpose, main inputs, main outputs, and current status.

## Status Labels

```text
Setup = environment or paths
Data prep = creates or checks trial-level data
Descriptive = summarizes behavior before fitting
Model scaffold = defines model pieces, but does not fit parameters yet
Sanity check = tests that a scaffold component behaves correctly
Future fitting = documents planned fitting steps
```

## Cell-By-Cell Map

| Cell | Status | What it does | Main input | Main output |
|---:|---|---|---|---|
| 1 | Setup | Explains the purpose of the notebook and the current modeling question. | None | Reader context |
| 2 | Setup | Checks whether optional Python packages are available. | Python environment | Package availability table |
| 3 | Setup | Imports libraries used by the notebook. | Installed packages | Imported modules |
| 4 | Setup | Introduces local path setup. | None | Reader context |
| 5 | Setup | Defines repository paths, output folder, and reference PDF paths. | Current working directory | `PROJECT_ROOT`, `RESEARCH_DIR`, `OUTPUT_DIR`, `PAPER_PATHS` |
| 6 | Data prep | Introduces the scaffold contract. | None | Reader context |
| 7 | Data prep | Defines stable column names, predictor groups, outcomes, and the paper-to-data map. | Project assumptions | `RAW_INPUT_COLUMNS`, `DERIVED_COLUMNS`, `PREDICTOR_SETS`, `EQUATION_MAP` |
| 8 | Data prep | Introduces raw data loading and inspection. | None | Reader context |
| 9 | Data prep | Defines the remote CSV URL and the data-loading function. | GitHub raw CSV URL | `load_trials(...)` |
| 10 | Data prep | Loads the raw dataset, checks required columns, and summarizes the table. | Remote CSV | `raw`, column summary |
| 11 | Data prep | Introduces feature-building functions. | None | Reader context |
| 12 | Data prep | Builds circular angles, block IDs, previous-block context, confidence proxy, history columns, and trial bins. | `raw` dataframe | `prepare_trials(...)` and helper functions |
| 13 | Data prep | Runs the feature builder and validates the prepared table. | `raw` | `trials` |
| 14 | Data prep | Introduces model-facing tables. | None | Reader context |
| 15 | Data prep | Creates the narrow model table and numeric subject/block indexes. | `trials` | `model_df`, `hierarchical_design` |
| 16 | Model scaffold | Introduces the carryover-to-estimate model path. | None | Reader context |
| 17 | Descriptive | Summarizes block transitions and previews confidence flow across trials. | `trials` | `block_transition_summary`, `confidence_flow_preview` |
| 18 | Descriptive | Introduces descriptive checks before fitting. | None | Reader context |
| 19 | Descriptive | Summarizes condition counts and circular estimation error before fitting. | `trials` | `condition_error_summary`, plots |
| 20 | Data prep | Introduces the experimental distribution compiler. | None | Reader context |
| 21 | Data prep | Creates empirical response-distribution targets for later model fitting and comparison. | `model_df` | `distribution_df`, `model_targets`, summaries, histograms |
| 22 | Model scaffold | Introduces circular model functions. | None | Reader context |
| 23 | Model scaffold | Defines circular probability functions: prior, sensory likelihood, posterior, readout, motor noise, lapse, and observed response probability. | Angle grid and model parameters | Reusable circular model functions |
| 24 | Model scaffold | Introduces hierarchical and switching hooks. | None | Reader context |
| 25 | Model scaffold | Defines observer parameters, model registry, switching responsibility, and confidence-only updating. | `model_df`, starting assumptions | `ObserverParams`, `MODEL_REGISTRY`, `update_online_prior_state(...)` |
| 26 | Model scaffold | Builds one-trial predicted estimate distributions and computes a smoke-test likelihood. | One row of `model_df`, `ObserverParams` | `build_trial_components(...)`, `negative_log_likelihood(...)` smoke output |
| 27 | Sanity check | Introduces simple simulation checks. | None | Reader context |
| 28 | Sanity check | Simulates a toy block and confirms prior precision changes while prior mean stays fixed. | Simulated 10 deg -> 80 deg transition | `sim_confidence_path`, `confidence_update_check`, `simulation_component_summary` |
| 29 | Future fitting | Introduces the fitting interface placeholder. | None | Reader context |
| 30 | Future fitting | Creates a balanced subset and documents the future per-subject fitting contract. | `model_df` | `balanced`, `fit_contract` |
| 31 | Future fitting | Explains where to edit the notebook when testing new theories. | Notebook structure | Editing guide |

## Most Important Cells For Non-Technical Review

Start with these:

| Cell | Why it matters |
|---:|---|
| 1 | Gives the project purpose. |
| 7 | Shows what columns and model pieces are considered stable. |
| 17 | Shows the proposed previous-block to current-confidence path. |
| 21 | Shows the experimental distributions future models must explain. |
| 23 | Defines the model's math components in code. |
| 26 | Shows how one trial becomes a predicted estimate probability. |
| 28 | Shows the simple test that confidence changes but prior mean does not. |
| 30 | Shows what fitting will estimate later; it is not yet a fit result. |

## Common Misreadings To Avoid

```text
The fitting contract is not fitted model evidence.
The simple simulation is not real participant fitting.
The confidence proxy is not the final fitted hyperprior.
The switching proxy is not the final Switching observer model.
The notebook currently prepares and checks the scaffold; it does not yet complete model comparison.
```

