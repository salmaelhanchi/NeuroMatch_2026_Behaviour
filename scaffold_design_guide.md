# Hierarchical Observer Scaffold Guide

Date: 2026-07-16

Notebook:
`hierarchical_observer_scaffold.ipynb`

Purpose of this file:
Explain the scaffold in simple terms so the work can be resumed, shared, or debugged in future sessions.

## Main Vision

The scaffold is a bridge between:

```text
paper equations
dataset columns
derived behavioral variables
experimental distributions
future model fitting
```

The research question can still change, but the data path should stay clear.

The current direction is:

```text
Adaptive prior confidence shapes perceptual estimation through hierarchical Bayesian inference.
```

In simpler terms:

```text
Subjects see noisy motion directions.
They also know, or learn, that directions come from a prior distribution.
The model asks how much confidence subjects place in that prior on each trial.
That confidence may depend on current block, previous block, sensory reliability, trial order, and subject.
```

## What The Model Tries To Explain

Observed data:

```text
motion_direction
motion_coherence
prior_std
prior_mean
estimate_x
estimate_y
subject_id
session_id
run_id
trial_index
```

Main derived response:

```text
estimate_deg
```

Main behavioral outcomes:

```text
error_from_motion
abs_error_from_motion
signed_prior_pull
closer_to_prior_than_motion
```

The model tries to predict:

```text
the distribution of estimate_deg
```

not only average error.

This matters because the paper's key claim is about distribution shape: human estimates can look bimodal, with one mode near the sensory evidence and another near the prior.

## High-Level Sequence

The notebook is organized as a sequence:

```text
1. Load data
2. Build trial and block identifiers
3. Add circular response variables
4. Add previous-trial history
5. Add previous-block context
6. Add prior-confidence scaffold variables
7. Build analysis-facing table
8. Compile empirical distributions
9. Define circular model functions
10. Define model registry and fitting contract
```

Some parts are sequential. Some are parallel summaries of the same table.

## Important Functions By Step

### 1. `load_trials`

Reads the remote CSV.

Output:

```text
raw
```

This is the untouched trial table.

### 2. `add_block_id`

Creates:

```text
block_id
trials_into_block
```

Why it helps:
The prior condition is fixed inside a block. Learning and carryover must be tracked relative to block order.

### 3. `build_block_context`

Creates one row per block.

Adds previous-block information:

```text
prev_prior_std
same_session_prev
prior_std_change
transition_direction
transition_type
```

Why it helps:
This is the foundation for carryover. It tells us what the subject experienced before the current block.

### 4. `add_block_context`

Merges the block-level context back into every trial.

Why it helps:
Each trial now knows both its current condition and its previous block condition.

### 5. `add_circular_columns`

Converts response coordinates into angles and circular errors.

Adds:

```text
estimate_deg
motion_offset_from_prior
estimate_offset_from_prior
error_from_motion
abs_error_from_motion
signed_prior_pull
distance_to_motion
distance_to_prior
closer_to_prior_than_motion
```

Why it helps:
Angles wrap around 360 degrees. This avoids wrong errors like treating 355 and 5 degrees as 350 degrees apart.

### 6. `add_history_columns`

Adds previous-trial variables inside each block:

```text
previous_motion_direction
previous_motion_coherence
previous_error_from_motion
```

Why it helps:
The task gives feedback after each trial, so previous trial information can influence the next trial.

### 7. `add_trial_bins`

Adds:

```text
trial_bin
```

Current bins:

```text
early
middle
late
```

Why it helps:
It gives a simple first view of within-block learning before fitting a full continuous model.

### 8. `add_prior_confidence_scaffold`

Adds the current scaffold version of the hidden confidence path:

```text
previous_prior_confidence_proxy
current_prior_confidence_target
initial_prior_confidence_proxy
within_block_learning_progress
prior_confidence_t_proxy
```

Why it helps:
This makes the proposed HB model visible in the data before final fitting.

Interpretation:

```text
previous block context
-> initial prior confidence at current block start
-> within-block confidence learning
-> trial-level prior confidence
```

Important:
These are proxy variables, not final estimated parameters.

### 9. `prepare_trials`

Runs the main feature-building sequence:

```python
out = add_block_id(df)
out = add_block_context(out)
out = add_circular_columns(out)
out = add_history_columns(out)
out = add_trial_bins(out)
out = add_prior_confidence_scaffold(out)
```

Output:

```text
trials
```

This is the full prepared table.

### 10. `make_model_frame`

Creates a narrower analysis table.

Output:

```text
model_df
```

Why it helps:
Modeling functions should not depend on every raw column. They should use a clean table with the stable modeling variables.

### 11. `make_hierarchical_design`

Creates numeric indices:

```text
subject_idx
block_idx
coherence_idx
prior_std_idx
transition_idx
```

Why it helps:
Hierarchical models need integer indices for subjects, blocks, coherence levels, prior widths, and transition types.

### 12. Experimental Distribution Compiler

The compiler turns `model_df` into empirical targets.

Main outputs:

```text
distribution_df
count_summary
subject_condition_summary
group_condition_summary
empirical_histograms
transition_distribution_summary
model_targets
```

Why it helps:
Models need something stable to fit and compare against. The compiler defines those targets before model fitting starts.

## Compiler Function Sequence

```text
model_df
-> add_switching_proxy_columns
-> define_distribution_bins
-> make_distribution_frame
-> compile_trial_counts
-> compile_subject_condition_summary
-> compile_group_condition_summary
```

Parallel branch:

```text
distribution_df
-> compile_empirical_histograms
```

Carryover branch:

```text
distribution_df
-> compile_transition_distributions
```

Final object:

```text
compile_model_targets
-> model_targets
```

## Why Subject-Aware Summaries Matter

Do not only pool trials.

Better order:

```text
trial -> block -> subject -> group
```

Reason:
Some subjects or blocks have more trials. Subject-aware summaries prevent high-trial subjects from dominating the group result.

## Core Model Functions

These define the circular model pieces:

```text
vm_pdf_deg
build_prior_grid
build_likelihood_grid
combine_likelihood_and_prior
readout_distribution
apply_motor_and_lapse
probability_at_observed_estimate
negative_log_likelihood
```

How they connect:

```text
motion_direction + motion_coherence
-> likelihood

prior_mean + prior_confidence_t
-> prior

likelihood x prior
-> posterior

posterior readout
-> percept distribution

motor noise + lapse
-> predicted estimate distribution

actual estimate_deg
-> likelihood / negative log likelihood
```

## Model Families In The Registry

Current named model families:

```text
basic_bayesian_bls
switching_map
sampling
hierarchical_switching
online_prior_learning
hierarchical_prior_confidence
```

Main target now:

```text
hierarchical_prior_confidence
```

Question:

```text
Does learned prior confidence explain estimate distributions better than fixed-prior or switching-only models?
```

## What Helps Debugging

Check these in order:

1. `raw.shape`
2. `trials.shape`
3. `transition_type` counts
4. `model_df.shape`
5. `distribution_df.shape`
6. `model_targets.keys()`
7. `empirical_histograms.head()`
8. `transition_distribution_summary.head()`

If a later model fails, first check whether these objects exist and contain the stable columns.

## What The Next Session Should Do

Recommended next step:

```text
Run the full scaffold notebook top to bottom in a notebook environment with pandas/numpy/matplotlib.
```

Then inspect:

```text
count_summary
subject_condition_summary
group_condition_summary
empirical_histograms
transition_distribution_summary
```

After that, implement the first real model comparison:

```text
sensory-only / maximum likelihood baseline
Basic Bayesian BLS
Switching MAP
hierarchical prior-confidence model
```
