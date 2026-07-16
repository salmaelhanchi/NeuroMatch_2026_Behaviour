# Scaffold Recent Changes

Date: 2026-07-16

Notebook:
`hierarchical_observer_scaffold.ipynb`

Purpose of this file:
Keep a simple record of what changed in the scaffold so future sessions can quickly see the current state, what is stable, and what should be checked if something breaks.

## Repository Layout Change

Date: 2026-07-16

The scaffold files were moved out of the old nested folder and into the repository root.

Current tracked scaffold files:

```text
README.md
hierarchical_observer_scaffold.ipynb
scaffold_recent_changes.md
scaffold_design_guide.md
simple_simulation_checks_interpretation.md
```

Reason:
The GitHub repository should show the main notebook and docs directly at the top level, not nested inside an extra project folder.

## Current Modeling Direction

Working model idea:

```text
previous block context
-> initial prior confidence at current block start
-> within-block confidence learning
-> predicted estimate distribution
```

The model is not only asking whether subjects use the prior. It is asking how strongly they trust the prior on each trial, how that trust changes inside a block, and whether confidence carries over from the previous block.

## Main Data Source Change

The scaffold now loads the behavioral CSV from the remote GitHub raw URL:

```python
REMOTE_CSV_URL = "https://raw.githubusercontent.com/steevelaquitaine/projInference/refs/heads/gh-pages/data/csv/data01_direction4priors.csv"

def load_trials(csv_url: str = REMOTE_CSV_URL) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    return df
```

The loading cell is intentionally minimal. Column checks and raw-data summaries are in the next cell.

Local path resolution is still used for:

```text
Model_explainer.pdf
Hierarchical_explicit_stepwise.pdf
rt_to_adaptation_explorarion_data_walkthrough.pdf
outputs/
```

## Between-Block Context Added

The scaffold now builds a block-level table before trial-level modeling.

Added function:

```python
build_block_context(df)
```

It creates one row per block and adds:

```text
block_order_within_subject
prev_block_id
prev_prior_std
prev_prior_mean
prev_session_id
prev_run_id
same_session_prev
prior_std_change
prior_width_changed
transition_direction
transition_type
```

Important labels:

```text
repeat_80
transition_to_80
same_session_repeat_prior
same_session_prior_transition
cross_session_previous
first_block
```

Why this matters:
The TA advised tracking between-block structure. The RT/adaptation walkthrough showed that RT is not strong as a main perceptual outcome, but it does show a block-start adjustment pattern. This makes previous block context important for initial prior confidence.

## Prior Confidence Scaffold Added

Added function:

```python
add_prior_confidence_scaffold(df, carryover_weight=0.35, learning_tau=45.0)
```

It adds proxy columns:

```text
previous_prior_confidence_proxy
current_prior_confidence_target
initial_prior_confidence_proxy
within_block_learning_progress
prior_confidence_t_proxy
```

Important interpretation:
These are scaffold proxies, not final fitted parameters.

They make the intended model path visible before fitting:

```text
previous block prior width
-> starting confidence in current block
-> exponential within-block learning progress
-> trial-level prior confidence
```

Later, these should be estimated from the response likelihood:

```text
carryover_weight
learning_tau
subject-specific initial confidence
subject-specific asymptotic confidence
```

## Switching Proxy Added

Added trial-level descriptive columns:

```text
distance_to_motion
distance_to_prior
closer_to_prior_than_motion
```

Meaning:

```text
closer_to_prior_than_motion = response is closer to prior_mean than to motion_direction
```

This is not the final switching model. It is a simple empirical proxy that helps describe whether a response looks prior-like or motion-like.

## Experimental Distribution Compiler Added

New section:

```text
Experimental Distribution Compiler
```

Stable outputs:

```text
distribution_df
count_summary
subject_condition_summary
group_condition_summary
empirical_histograms
transition_distribution_summary
model_targets
```

Compiler functions:

```python
add_switching_proxy_columns(df)
define_distribution_bins(df)
make_distribution_frame(df)
compile_trial_counts(distribution_df)
compile_subject_condition_summary(distribution_df)
compile_group_condition_summary(subject_condition_summary)
compile_empirical_histograms(distribution_df)
compile_transition_distributions(distribution_df)
compile_model_targets(...)
```

Why this matters:
The compiler creates the empirical targets that later models must explain. It does not fit the HB model yet.

It supports:

```text
trial-level model fitting
subject-aware summaries
group-level summaries
distribution-shape comparison
transition/carryover summaries
```

## Stable Names Not To Rename Casually

These names should be treated as stable references:

```text
distribution_df
subject_condition_summary
group_condition_summary
empirical_histograms
transition_distribution_summary
model_targets
```

Core column names:

```text
motion_direction
motion_coherence
prior_std
prior_mean
estimate_deg
error_from_motion
abs_error_from_motion
signed_prior_pull
subject_id
session_id
run_id
block_id
trials_into_block
trial_bin
prev_prior_std
same_session_prev
transition_type
transition_direction
prior_confidence_t_proxy
closer_to_prior_than_motion
```

If these names change, later model-fitting and comparison code will likely break.

## Current Model Registry Target

Added model target:

```python
hierarchical_prior_confidence
```

Main question:

```text
Does learned prior confidence explain estimate distributions better than fixed-prior or switching-only models?
```

The fitting contract now includes:

```text
k_llh_by_coherence
k_prior_by_std
carryover_weight
learning_tau
initial_prior_confidence_by_subject
asymptotic_prior_confidence_by_prior_std
k_motor
lapse_rate
```

## Checklist Implementation: HB Components And Confidence Update

Date: 2026-07-16

This update implements the current checklist items in the notebook scaffold.

Implemented model components:

```python
build_trial_components(...)
```

This function now creates named one-trial model pieces:

```text
sensory_likelihood
prior
posterior
readout_pdf
estimate_pdf
```

Meaning:

```text
dataset condition columns
-> sensory likelihood from motion direction/coherence
-> prior from prior mean/width/confidence
-> posterior from likelihood x prior
-> readout distribution
-> final estimate distribution after motor noise and lapse
```

Implemented confidence update:

```python
update_online_prior_state(previous_mean, previous_kappa, target_kappa, learning_rate)
```

Important behavior:

```text
prior_mean stays tied to the block prior_mean
prior_kappa changes toward the current block target
```

So the update now affects prior precision/confidence, not just the prior mean.

Preserved circular calculations:

The scaffold still uses circular helpers for angle wrapping and circular probability calculations:

```text
wrap_signed_deg
wrap_unsigned_deg
vm_pdf_deg
circular_mean_deg
circular_convolve
probability_at_observed_estimate
```

Added simple simulation test:

```python
simulate_prior_confidence_learning(...)
```

The simulation checks a toy block where confidence carries over from the previous prior width and learns toward the current prior width. It creates:

```text
sim_confidence_path
confidence_update_check
simulation_component_summary
```

The notebook now asserts:

```text
all component probability distributions sum to 1
prior_mean is unchanged across the toy block
prior_kappa changes across the toy block
```

This is a scaffold test only. It does not yet complete full model validation, parameter recovery, or model comparison.

## Validation Done

The notebook JSON loads successfully.

Syntax check:

```text
31 cells
17 code cells
0 syntax errors
```

The full notebook was not executed in the shell because the shell Python environment does not have the notebook data stack installed.

## Bug Fix: Compiler Missing Offset Columns

Date: 2026-07-16

Issue:
The Experimental Distribution Compiler raised:

```text
KeyError: 'motion_offset_from_prior'
```

Cause:
`add_circular_columns()` correctly created `motion_offset_from_prior` and `estimate_offset_from_prior`, but `make_model_frame()` accidentally dropped those columns when it created the narrow `model_df`.

Fix:
`make_model_frame()` now keeps:

```text
motion_offset_from_prior
estimate_offset_from_prior
```

The compiler also now checks for required derived columns and gives a clearer message:

```text
Re-run prepare_trials(raw), then model_df = make_model_frame(trials).
```

## Cleanup: Removed Root Helper

Date: 2026-07-16

Issue:
The CSV loading cell had been simplified, but the notebook still had a separate root-detection helper in the local path section.

Fix:
Removed the root-detection helper and replaced it with simple local path setup:

```python
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR
RESEARCH_DIR = PROJECT_ROOT
```

The remote data-loading cell remains minimal.

## What To Check If Something Breaks

1. If loading fails:
   Check `REMOTE_CSV_URL`.

2. If derived columns are missing:
   Run cells from the top. `prepare_trials(raw)` must run before compiler cells.

3. If transition columns are missing:
   Check `build_block_context()` and `add_block_context()`.

4. If confidence columns are missing:
   Check `add_prior_confidence_scaffold()`.

5. If compiler outputs are missing:
   Run the full `Experimental Distribution Compiler` section.

6. If model fitting later fails:
   Check that `model_targets["trial_table"]` exists and contains the stable columns listed above.
