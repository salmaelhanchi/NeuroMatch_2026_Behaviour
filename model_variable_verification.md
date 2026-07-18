# Model Variable Verification

This file records the variable check before implementing the verified HB model notebook.

## Source Materials Checked

```text
A-Switching-Observer-for-Human-Perceptual-Estimati.pdf
Model_explainer.pdf
Hierarchical_explicit_stepwise.pdf
rt_to_adaptation_explorarion_data_walkthrough.pdf
data01_direction4priors.csv
```

## Paper-To-Data Mapping

| Paper/model term | Meaning | Dataset or implementation name |
|---|---|---|
| `theta_true` | Presented motion direction on the trial | `motion_direction` |
| motion coherence | Sensory reliability condition | `motion_coherence` |
| sensory reliability / likelihood precision | Concentration of sensory likelihood | fitted `k_llh_by_coherence` |
| prior mean / prior mode | Direction favored by the prior | `prior_mean`, usually 225 |
| prior width condition | Experimental prior standard deviation in the block | `prior_std` |
| prior confidence / prior precision | Strength of the prior used by the observer | fitted and trial-varying `prior_kappa_t` |
| hyper-prior in hierarchy notes | Belief about prior parameters | represented here as changing confidence/precision, not changing prior mean |
| estimate / reported direction | Participant response angle | `estimate_x`, `estimate_y` converted to `estimate_deg` |
| motor precision | Response noise after perceptual readout | fitted `k_motor` |
| lapse rate | Probability of a random response | fitted `lapse_rate` |

## Important Terminology Decision

The PDFs sometimes describe the top-level learning variable as a hyper-prior over the prior.

For this project, the first implemented HB model treats that hidden top-level learning as:

```text
trial-by-trial prior confidence / prior precision
```

It does not update the prior mean.

The prior mean remains tied to the task condition:

```text
prior_mean = 225 deg
```

The changing quantity is:

```text
prior_kappa_t
```

where larger `prior_kappa_t` means stronger confidence in the prior and smaller `prior_kappa_t` means weaker confidence.

## Verified Model Flow

The implemented model follows this path:

```text
previous block prior width
-> initial prior confidence at current block start
-> within-block confidence learning toward current block target
-> prior distribution for each trial
-> sensory likelihood for each trial
-> posterior distribution
-> readout estimate
-> motor noise and lapse
-> probability of the participant's observed estimate
```

## Current Implementation Scope

The verified implementation notebook fits a first tractable HB model separately by subject.

It preserves:

```text
circular angle calculations
prior distribution
sensory likelihood
posterior
readout
motor noise
lapse
trial-by-trial confidence updating
per-subject fitting
NLL output
predicted estimate output
```

The current fitting likelihood uses the presented motion direction as the sensory measurement center for speed and clarity. The paper's full Basic Bayesian observer marginalizes over possible noisy sensory measurements. That measurement-marginalized version is the next precision upgrade after this first per-subject HB fit is stable.

