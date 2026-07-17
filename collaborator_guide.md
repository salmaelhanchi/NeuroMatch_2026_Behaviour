# Collaborator Guide

This file is the simple starting point for collaborators who want to understand the project without reading every line of code.

## The Project In One Sentence

We are testing whether people update their confidence in a prior across trials when estimating motion direction, and whether that learning can explain response patterns that look like switching between the prior and the sensory evidence.

## The Current Model Story

The working model path is:

```text
previous block context
-> initial prior confidence at the start of the current block
-> confidence learning within the block
-> predicted distribution of reported estimates
```

In plain language:

```text
What happened in the previous block may affect how strongly a participant trusts the prior at the start of the next block.
As trials continue, the participant may adjust that trust.
That changing trust should change where their estimates fall.
```

## What The Notebook Currently Does

The main notebook is:

[hierarchical_observer_scaffold.ipynb](hierarchical_observer_scaffold.ipynb)

It currently does these things:

```text
load the remote behavioral dataset
check that required columns exist
create trial-level model variables
calculate circular estimation error
track previous-block context
create a proxy for prior confidence across trials
compile empirical response distributions
define circular Bayesian model components
run a simple simulation check
document the future fitting contract
```

## What The Notebook Does Not Prove Yet

The notebook does not yet prove that the hierarchical Bayesian model fits the real data.

That will require:

```text
fitted parameters for each subject
observed-vs-predicted response distributions
negative log-likelihood from fitted parameters
AIC/BIC or cross-validation
parameter recovery
comparison against the Switching observer
```

## How To Read The Notebook

Use this order:

1. Read the notebook title cell.
2. Read [notebook_cell_map.md](notebook_cell_map.md) beside the notebook.
3. Run or inspect cells from top to bottom.
4. Treat the final fitting-contract cell as a plan, not as proof of model fit.

Each code cell now starts with:

```python
# Cell purpose: ...
```

That sentence says what the cell does before the code begins.

## Change Tracking System

Use these files:

| File | Use it for |
|---|---|
| [tasks_draft.md](tasks_draft.md) | Project-level task checklist. |
| [notebook_cell_map.md](notebook_cell_map.md) | Cell-by-cell explanation of the notebook. |
| [scaffold_recent_changes.md](scaffold_recent_changes.md) | Detailed change history and stable names. |
| [simple_simulation_checks_interpretation.md](simple_simulation_checks_interpretation.md) | Explanation of the simple simulation output. |
| [abstract_draft.md](abstract_draft.md) | Current abstract draft. |

## How To Review A Change

When a new change is made, check these four questions:

```text
What question does the change answer?
Which notebook cell changed?
Which dataset columns or model variables changed?
Does this produce a real fit result, or only a scaffold/check result?
```

## Safe Interpretation Rule

If an output is a table of inputs, predictor names, or parameter names, it is probably a scaffold or fitting plan.

If an output contains fitted parameters, NLL, AIC/BIC, cross-validation, or observed-vs-predicted plots, it is model evidence.

The current notebook is mostly scaffold plus sanity checks.

