# Source-Only Execution Plan

## Goal

Execute the project from the raw dataset using only the model requirements established by the TA transcripts, the original paper, and the two secondary explanatory PDFs.

## Stage 1: establish the raw-data contract

### Input

- `data01_direction4priors.csv`
- paper-defined task variables

### Process

1. Read the CSV without modifying it.
2. Verify required columns for subject, session, run, trial, direction, coherence, prior condition, prior mean, and response coordinates.
3. Sort by subject, session, run, and original trial index.
4. Create a permanent block identifier from subject, session, and run.
5. Convert response coordinates to circular degrees.
6. Mark missing responses instead of removing their trials from the feedback sequence.
7. Verify that prior mean remains 225 degrees.
8. Create signed circular stimulus, response, and error fields.

### Output

- chronological trial table;
- response-validity mask;
- block summary;
- data-quality report.

### Gate

No model work begins until ordering, circular conversion, missing responses, and block boundaries are verified.

## Stage 2: reproduce behavioral observations

### Input

- chronological trial table

### Process

For each subject, compile distributions by:

- prior width;
- motion coherence;
- true direction relative to 225 degrees;
- trial position within block.

Calculate:

- circular response mean;
- circular response variability;
- signed estimation error;
- estimate-versus-direction slope;
- response mass near the prior;
- response mass near the true direction;
- full response histograms in large-conflict conditions;
- first-100 versus last-100 block behavior.

### Output

- subject-level behavioral figures;
- group summary figures;
- list of conditions showing one or two modes;
- early-versus-late stability summary.

### Gate

Confirm that the raw data show the paper's basic effects before fitting models. If bimodality cannot be located under a predeclared condition definition, resolve that analysis first.

## Stage 3: build common circular probability tools

### Input

- 1-to-360-degree grid;
- von Mises means and concentrations.

### Process

Implement and test:

- circular wrapping;
- stable von Mises PMFs;
- normalization;
- circular mean and SD;
- circular convolution;
- MAP tie handling;
- response-grid indexing;
- uniform lapse mixture;
- latent-measurement marginalization.

### Output

- tested probability functions shared by every model.

### Gate

All PMFs must be finite, nonnegative, circularly wrapped, and normalized. Rotating all angles together must leave probabilities unchanged.

## Stage 4: implement the paper comparators

### Input

- shared circular tools;
- trial direction and coherence;
- candidate parameter values.

### Process

Implement independently:

1. Sensory-only observer.
2. Basic Bayesian observer with latent measurement integration.
3. Original Switching observer.
4. Switching-prior-sampling variant if time permits.

Use the same motor-noise and lapse layer for every model.

### Output

- `P(response | trial, parameters)` for each model;
- simulation function for each model;
- subject-level NLL function.

### Gate

The Basic observer should produce a compromise mode. The Switching observer should be able to produce separate prior-like and sensory-like modes.

## Stage 5: implement the hierarchical hidden-confidence observer

### Input

- previous hyper-state `H_t(kappa)`;
- current direction and coherence;
- fixed mean 225 degrees;
- subject parameters `rho`, sensory concentrations, motor concentration, and lapse.

### Process

1. Discount/diffuse the previous hyper-state using `rho`.
2. Form one effective prior by averaging von Mises priors over `H_t`.
3. For each latent measurement, multiply the sensory likelihood by that effective prior.
4. Normalize one posterior over direction.
5. Apply one posterior readout.
6. Integrate the readout over possible sensory measurements.
7. Apply motor noise and lapse.
8. Score the current response if it is valid.
9. Update `H_t(kappa)` using the revealed direction.
10. Carry the updated state into the next trial and across block boundaries.

### Output

- trial response PMF;
- trial log likelihood;
- pre-feedback and post-feedback hyper-state;
- expected concentration trajectory;
- effective-prior and posterior snapshots for diagnostic conditions.

### Gate

Changing only `prior_std` while keeping experienced directions identical must not change predictions or state. Changing feedback on trial `t` must affect trial `t+1`, not trial `t`. The primary model must contain no branch-selection or responsibility variable.

### Optional extension after this gate

If the validated and recoverable hidden-confidence observer cannot reproduce bimodality, test a two-regime distribution over `kappa`, including possible mass near zero. Continue to marginalize over `kappa` before forming one posterior. Any model that reads out confidence regimes separately must be named and compared as a distinct gating model.

## Stage 6: validate model behavior by simulation

### Input

- controlled synthetic directions and coherences;
- known model parameters.

### Process

Simulate at least these cases:

1. High coherence, weak confidence: sensory-dominated single peak.
2. Low coherence, strong confidence: prior-dominated single peak.
3. Large prior-sensory conflict: test whether the prediction remains unimodal or separates into two modes.
4. Narrow block followed by broad block: confidence should fall gradually.
5. Broad block followed by narrow block: confidence should rise gradually.
6. Same sequence with different block labels: identical model behavior.

### Output

- predicted distributions;
- confidence trajectories;
- effective-prior and posterior trajectories;
- simulation validation report.

### Gate

The model must be numerically valid and respond sensibly to confidence and feedback. Whether it generates two modes is an empirical result: success supports the abstract's hypothesis, while failure shows that hidden confidence alone is insufficient.

## Stage 7: parameter recovery

### Input

- synthetic datasets generated from known parameter combinations.

### Process

1. Generate data using realistic subject trial sequences.
2. Fit the model without exposing the generating values.
3. Use multiple starting points.
4. Repeat across random seeds.
5. Compare generated and recovered values.
6. Inspect tradeoffs among `rho`, sensory concentration, motor noise, and lapse.
7. Compare generated and recovered response distributions and state trajectories.

### Output

- generated-versus-recovered plots;
- parameter error table;
- convergence table;
- identifiability decision for every parameter.

### Gate

Parameters that cannot be recovered are fixed, removed, or reparameterized before real-data interpretation.

## Stage 8: fit the 12 subjects

### Input

- validated model;
- each subject's chronological sequence.

### Process

1. Pilot one subject with clear bimodality.
2. Pilot one subject with weaker or noisier modes.
3. Inspect predictions and state trajectories.
4. Fit all 12 subjects separately using multiple starts.
5. Save every fit, start, convergence message, and prediction.

### Output

- one parameter record per subject and model;
- trial-wise predicted PMFs;
- subject-level observed/predicted distributions;
- subject-level confidence trajectories.

### Gate

No group conclusion is made if important subject-level failures are hidden by averaging.

## Stage 9: model validation

### Input

- fitted models and observed subject responses.

### Process

Compare observed and predicted:

- full response histograms;
- circular means and variability;
- near-prior and near-motion mass;
- peak separation;
- early and late behavior;
- behavior around block transitions;
- convergence versus noisy state fluctuation.

### Output

- posterior-predictive panels for every subject;
- failure-condition table;
- successful-condition table.

### Gate

This stage must be accepted before AIC is treated as meaningful, following the TA instruction.

## Stage 10: model comparison

### Input

- validated subject-level fits for the sensory, Basic, Switching, and hierarchical models.

### Process

1. Use identical response rows and observation conventions.
2. Calculate NLL and AIC for continuity with the paper.
3. Add sequence-preserving held-out prediction when feasible.
4. Calculate within-subject score differences.
5. Report individual subjects before group summaries.
6. Interpret scalar scores together with distribution-shape validation.

### Output

- per-subject comparison table;
- AIC difference plots;
- held-out prediction table;
- conclusion about where each model succeeds or fails.

### Gate

A model is not declared superior from AIC alone if it fails to reproduce the observed response distributions.

## Stage 11: test the new learning prediction

### Input

- fitted hierarchical trajectories;
- empirical within-block error or slope trajectories.

### Process

Test whether:

- estimated confidence changes systematically within blocks;
- confidence change differs across experienced prior widths;
- confidence trajectories relate to behavioral learning trajectories;
- block-end confidence predicts behavior at the next block's start;
- effects differ across motion coherences.

These tests follow the TA's proposed progression from model validation to model prediction.

### Output

- learning-trajectory correlations;
- transition analyses;
- coherence-stratified results;
- clearly labeled exploratory findings.

## Deliverables

- reproducible data audit;
- empirical paper-replication figures;
- tested model functions;
- simulation validation report;
- parameter recovery report;
- 12 subject-level model fits;
- observed-versus-predicted distribution report;
- model comparison table;
- learning-prediction analysis;
- final interpretation with limitations.

## Stop conditions

Stop and repair the model if:

- it updates the prior mean;
- it uses `prior_std` to update the observer;
- it treats true direction as the sensory measurement;
- bimodality is forced by adding an unreported branch-selection rule;
- its hyper-state changes before feedback;
- its parameters fail recovery;
- only AIC, means, or standard deviations are inspected;
- subject-level failures are hidden by pooling.

