# NeuroMatch 2026 Behaviour

Repository root: [NeuroMatch_2026_Behaviour](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour)

## Research Question

Participants estimated motion direction after seeing motion stimuli with different sensory reliability and different prior widths. Their responses can show peaks near both the true motion direction and the prior mean.

The project asks whether this behavior requires an explicit **Switching observer**, or whether it can emerge from a **Hierarchical Bayesian observer** that learns **prior confidence / prior precision** over trials.

Current HB modeling path:

```text
previous block context
-> initial prior confidence at current block start
-> within-block confidence learning
-> posterior estimate prediction
-> motor noise and lapse
-> predicted response distribution
```

Important terminology: the changing hidden quantity is treated as **prior confidence / precision** (`prior_kappa_t`), not a changing prior mean.

## Where We Are Now

This repository currently has a working HB modeling scaffold plus smoke-test notebooks. These are useful for checking whether the model code, variable mapping, and result diagnostics behave sensibly.

Current status:

| Area | Status |
|---|---|
| Data loading and trial preparation |  |
| HB model components | Implemented: prior, sensory likelihood, posterior, readout, motor noise, lapse. |
| Trial-by-trial prior-confidence learning | Implemented as `prior_kappa_t`. |
| Per-subject smoke fitting | Implemented on a block-balanced subset. |
| Observed vs predicted distribution check |  |
| Parameter recovery | Implemented as a smoke diagnostic, with result-oriented plots and summaries. |
| Final model comparison | Not done yet. Switching observer and full HB comparison still need implementation. |

The current outputs are **diagnostics**, not final proof that the model is correct.

## Main Notebooks

| Notebook | What it does |
|---|---|
| [hierarchical_observer_scaffold.ipynb](hierarchical_observer_scaffold.ipynb) | Main scaffold. Shows how raw data columns become model inputs and how the HB model pieces connect. |
| [hb_verified_model_implementation.ipynb](hb_verified_model_implementation.ipynb) | Verified implementation copy. Fits a first per-subject HB prior-confidence model on a smoke subset. |
| [hb_smoke_fit_user_guide.ipynb](hb_smoke_fit_user_guide.ipynb) | Helper guide for collaborators. Compares observed participant errors with model-simulated errors. |
| [hb_parameter_recovery_smoke.ipynb](hb_parameter_recovery_smoke.ipynb) | Parameter recovery smoke test. Simulates data from fitted parameters, refits the model, and checks what recovers well or poorly. |
| [Switching_Bayesian_Observer_starter.ipynb](Switching_Bayesian_Observer_starter.ipynb) | Starter notebook for the future switching observer implementation. |
| [Data_Preperation_Notebook_.ipynb](Data_Preperation_Notebook_.ipynb) | Earlier data preparation notebook retained for reference. |

## Markdown Guide Files

| File | What it explains |
|---|---|
| [abstract_draft.md](abstract_draft.md) | Current project abstract draft. |
| [tasks_draft.md](tasks_draft.md) | Draft project checklist from HB implementation through validation, model comparison, and presentation. |
| [collaborator_guide.md](collaborator_guide.md) | Non-technical overview of the model, current status, and how collaborators should inspect progress. |
| [notebook_cell_map.md](notebook_cell_map.md) | Plain-language map of the scaffold notebook cells. |
| [model_variable_verification.md](model_variable_verification.md) | Paper-to-data variable mapping and the prior-confidence vs prior-mean clarification. |
| [scaffold_recent_changes.md](scaffold_recent_changes.md) | Change-tracking file for scaffold edits, stable names, and debugging notes. |
| [scaffold_design_guide.md](scaffold_design_guide.md) | Scaffold design logic: function groups, expected flow, and where future edits should go. |
| [simple_simulation_checks_interpretation.md](simple_simulation_checks_interpretation.md) | Interpretation of the simple simulation showing that prior precision changes while prior mean stays fixed. |
| [data01_direction4priors_column_map_and_exploration.md](data01_direction4priors_column_map_and_exploration.md) | Data column notes and exploratory mapping. |
| [first_look_data_columns.md](first_look_data_columns.md) | First-pass description of available dataset columns. |
| [explatory_data_discovery_observations_and_research_notes.md](explatory_data_discovery_observations_and_research_notes.md) | Exploratory observations about behavior and possible research directions. |
| [varsha_by_idea_1.md](varsha_by_idea_1.md) | Extra idea notes kept for reference. |

Conversation/archive files such as [gpt_conversation_markdown.md](gpt_conversation_markdown.md), [markdown_conversation.md](markdown_conversation.md), and [markdown_conversation_work_continuation.md](markdown_conversation_work_continuation.md) are not required to run the model.

## What You Need To Use The Model

Required:

- Python notebook environment.
- Python libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `nbformat`, `nbclient`.
- Internet access for the remote CSV, unless `data01_direction4priors.csv` is already present locally.

Data source used by the notebooks:

```text
https://raw.githubusercontent.com/steevelaquitaine/projInference/refs/heads/gh-pages/data/csv/data01_direction4priors.csv
```

Recommended run order:

1. Open [hierarchical_observer_scaffold.ipynb](hierarchical_observer_scaffold.ipynb) to understand the data-to-model structure.
2. Run [hb_verified_model_implementation.ipynb](hb_verified_model_implementation.ipynb) to produce smoke-fit parameter and prediction outputs.
3. Run [hb_smoke_fit_user_guide.ipynb](hb_smoke_fit_user_guide.ipynb) to compare observed vs model-predicted response distributions.
4. Run [hb_parameter_recovery_smoke.ipynb](hb_parameter_recovery_smoke.ipynb) to check parameter recovery and prior-confidence trajectories.

Key output files are saved in [outputs](outputs/), including smoke-fit results, trial predictions, observed-vs-predicted summaries, and parameter-recovery diagnostics.

## Next Steps

The next work should move from smoke diagnostics to full model testing:

1. Fit the HB model on the full trial-level dataset for each subject.
2. Implement the Switching observer model.
3. Compare HB vs Switching using NLL, AIC, BIC, and cross-validation.
4. Run stronger parameter recovery with more trials, more subjects, and multiple optimizer restarts.
5. Create condition-specific observed-vs-predicted plots around sensory and prior modes.
