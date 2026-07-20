# Simplified Step-by-Step Technical Implementation

## How to read this guide

Every step has four parts:

- **Input**: what enters the step.
- **Process**: what the code does.
- **Output**: what the next step receives.
- **Check**: how to know the step is correct.

The explanation is intentionally sequential. A later step should not silently redo or change an earlier step.

## Step 0: define the angle grids

**Input**

- no data yet.

**Process**

Create:

```text
theta_grid = [1, 2, ..., 360]
kappa_grid = a positive grid from almost uniform to sharply concentrated
```

A practical implementation choice is 61 log-spaced kappa values from 0.05 to 50, plus 0 for a uniform distribution.

**Output**

- direction grid;
- concentration grid.

**Check**

- direction grid has exactly 360 values;
- 1 and 360 are circular neighbors;
- kappa values are ordered and nonnegative.

## Step 1: load and order trials

**Input**

- raw CSV.

**Process**

Read the file and sort by:

```text
subject_id -> session_id -> run_id -> trial_index
```

Create a block key:

```text
block_id = (subject_id, session_id, run_id)
```

Do not sort by direction, coherence, prior width, or response.

**Output**

- one chronological table.

**Check**

- every composite trial key is unique;
- original trial order is retained within every block.

## Step 2: convert responses to angles

**Input**

- `estimate_x` and `estimate_y`.

**Process**

```text
estimate_deg = atan2(estimate_y, estimate_x) in degrees
estimate_deg = estimate_deg mod 360
```

Create:

```text
response_valid = estimate_x and estimate_y are both present
```

**Output**

- response angle;
- validity mask.

**Check**

- every valid response lies on the circular range;
- missing responses remain as sequence rows.

## Step 3: create circular differences

**Input**

- true direction;
- response angle;
- fixed prior mean 225 degrees.

**Process**

Use:

```text
wrap(x) = ((x + 180) mod 360) - 180
```

Calculate:

```text
stimulus_from_prior = wrap(true_direction - 225)
response_from_prior = wrap(response_angle - 225)
response_error = wrap(response_angle - true_direction)
```

**Output**

- circular analysis variables.

**Check**

- examples near 1 and 359 degrees give small, not 358-degree, differences.

## Step 4: make the observed distributions

**Input**

- valid responses and condition labels.

**Process**

For each subject, group by prior width, coherence, and true direction relative to 225 degrees. Count responses in circular bins and normalize counts to probabilities.

Also create early, middle, and late block summaries.

**Output**

- observed response PMFs;
- means, variability, and mode-mass summaries.

**Check**

- each response PMF sums to one;
- high-conflict, low-coherence conditions are not hidden by broad pooling.

## Step 5: create a stable von Mises PMF

**Input**

- mean direction `mu`;
- concentration `kappa`;
- direction grid.

**Process**

For each grid angle:

```text
value(theta) = exp[kappa * (cos(theta - mu) - 1)]
```

Normalize the 360 values. Use a stable Bessel-function implementation or direct grid normalization to avoid overflow.

**Output**

- normalized circular PMF.

**Check**

- `kappa = 0` is uniform;
- larger kappa is narrower;
- moving `mu` rotates the PMF without changing shape.

## Step 6: construct sensory measurements

**Input**

- true direction `d`;
- coherence `c`;
- sensory concentration `kappa_sensory[c]`.

**Process**

Construct:

```text
P(m | d, c) = VM(m; d, kappa_sensory[c])
```

This is the probability of each possible internal measurement.

**Output**

- 360-value measurement PMF.

**Check**

- high coherence gives a narrower measurement PMF;
- the PMF is centered on the true direction only as a distribution, not as a guaranteed measurement.

## Step 7: implement the Basic Bayesian trial

**Input**

- each possible measurement `m`;
- sensory concentration;
- fixed prior mean;
- fixed prior concentration for the condition.

**Process**

For every possible measurement:

```text
likelihood(theta | m) = VM(theta; m, kappa_sensory)
prior(theta) = VM(theta; 225, kappa_prior)
posterior(theta | m) proportional to likelihood * prior
percept(m) = posterior MAP
```

Then integrate over measurement:

```text
P(percept | d) = sum_m P(percept | m) P(m | d)
```

**Output**

- Basic Bayesian percept PMF for the trial.

**Check**

- conflict generally produces one compromise mode.

## Step 8: implement the Switching trial

**Input**

- measurement distribution;
- sensory and prior concentrations;
- fixed prior mean.

**Process**

Calculate:

```text
w_prior = kappa_prior / (kappa_prior + kappa_sensory)
w_sensory = 1 - w_prior
```

Create a prior branch at 225 degrees and a sensory branch from the measurement distribution. Mix them using the two weights.

**Output**

- Switching percept PMF.

**Check**

- large conflict can produce one peak near 225 and another near the true direction;
- strengthening the prior increases prior-branch mass.

## Step 9: initialize hierarchical confidence

**Input**

- `kappa_grid`;
- fixed broad initial hyper-state.

**Process**

Set:

```text
H_1(kappa) = normalized broad probability over kappa_grid
```

Do not initialize from the first block's `prior_std`.

**Output**

- pre-trial concentration belief.

**Check**

- changing the first block label does not alter initialization.

## Step 10: predict the confidence state before a trial

**Input**

- previous post-feedback state `H_t`;
- memory parameter `rho`.

**Process**

Discount accumulated certainty:

```text
H_t_minus(kappa) proportional to H_t(kappa)^rho
```

Normalize.

**Output**

- pre-feedback state for the current trial.

**Check**

- `rho = 1` preserves the state;
- lower `rho` makes it broader;
- no current-trial feedback has been used.

## Step 11: construct the effective prior

**Input**

- pre-feedback hyper-state;
- fixed mean 225 degrees.

**Process**

Average the direction priors represented by every possible concentration:

```text
P_effective(theta)
  = sum_kappa H_t_minus(kappa) VM(theta; 225, kappa)
```

This marginalizes the hidden concentration. It does not select a concentration regime.

**Output**

- one effective prior over direction.

**Check**

- the PMF sums to one;
- shifting `H_t` toward larger kappa narrows the effective prior;
- no confidence regime or prior component has been selected.

## Step 12: calculate one integrated posterior

**Input**

- one possible measurement `m`;
- likelihood;
- effective prior.

**Process**

```text
posterior(theta | m, H_t)
  proportional to likelihood(theta | m) * P_effective(theta)
```

Normalize the posterior, then apply one predeclared readout such as MAP, circular posterior mean, or posterior sampling. Do not read out different kappa regimes separately.

**Output**

- one posterior conditional on measurement;
- one percept PMF conditional on measurement.

**Check**

- posterior sums to one;
- only one readout operation is applied;
- changing the distribution over kappa changes the posterior through the effective prior.

## Step 13: integrate over sensory uncertainty

**Input**

- integrated-posterior readout PMF for every measurement;
- `P(m | d, c)`.

**Process**

```text
P(percept | d, c, H_t) = sum_m P(percept | m, H_t) P(m | d, c)
```

**Output**

- one percept PMF for the current trial.

**Check**

- result is normalized;
- replacing measurement integration with the true direction changes the result, proving the latent path is active.

## Step 14: add motor noise and lapse

**Input**

- percept PMF;
- motor concentration;
- lapse probability.

**Process**

1. Circularly convolve the percept PMF with zero-mean motor noise.
2. Mix the result with a uniform PMF using the lapse probability.

**Output**

- final response PMF.

**Check**

- PMF sums to one;
- motor noise broadens without shifting;
- lapse adds uniform mass.

## Step 15: score the observed response

**Input**

- response PMF;
- observed response angle;
- response-validity mask.

**Process**

If response is valid:

```text
trial_log_likelihood = log P(observed response)
```

If response is missing, do not add a likelihood term.

**Output**

- trial score or masked score.

**Check**

- no valid response receives zero probability;
- a missing response does not remove the trial from the state sequence.

## Step 16: update confidence after feedback

**Input**

- pre-feedback state;
- revealed true direction;
- fixed mean 225 degrees.

**Process**

For every candidate concentration:

```text
feedback_likelihood(kappa) = VM(true_direction; 225, kappa)
H_next(kappa) proportional to feedback_likelihood(kappa) * H_t_minus(kappa)
```

Normalize.

**Output**

- post-feedback state for the next trial.

**Check**

- feedback near 225 shifts mass toward larger concentration;
- far feedback shifts mass toward smaller concentration;
- current response prediction does not change retroactively.

## Step 17: carry state across a block boundary

**Input**

- final post-feedback state of the old block;
- first trial of the new block.

**Process**

Use the final old-block state as the initial state for the new block. Apply the ordinary prediction and feedback updates. Do not set the state from the new block label.

**Output**

- continuous cross-block confidence trajectory.

**Check**

- the state is continuous at the boundary;
- adaptation occurs only as directions from the new block are revealed.

## Step 18: calculate the subject objective

**Input**

- all chronological trial scores for one subject.

**Process**

```text
NLL = -sum(valid trial log likelihoods)
```

Run the complete sequence from the same initial state for every tested parameter set.

**Output**

- one subject-level NLL.

**Check**

- shuffling trials changes the hierarchical NLL;
- shuffling trials does not change a correctly implemented static model NLL when condition labels remain attached.

## Step 19: fit parameters

**Input**

- one subject's trials;
- parameter bounds;
- multiple starting values.

**Process**

Optimize NLL using transformed parameters:

```text
log(kappa) for positive concentrations
logit(rho) and logit(lapse) for probabilities
```

Retain every start and select the best converged result.

**Output**

- fitted parameters;
- convergence record;
- fitted trial predictions.

**Check**

- several starts converge to comparable solutions;
- fitted values are not trapped at bounds without explanation.

## Step 20: validate by simulation

**Input**

- chosen known parameters;
- controlled or real trial sequences.

**Process**

Generate responses and test one-peak, two-peak, narrow-to-broad, and broad-to-narrow scenarios.

**Output**

- simulation plots and trajectories.

**Check**

- the model behaves numerically and directionally as intended;
- whether two modes appear is recorded as a result rather than forced by adding a branch-selection mechanism.

## Step 21: recover parameters

**Input**

- synthetic responses with known generating parameters.

**Process**

Fit the model exactly as for real data and compare fitted values with generating values across many seeds and settings.

**Output**

- recovery plots and error table.

**Check**

- important parameters recover consistently;
- nonrecoverable parameters are not interpreted in real subjects.

## Step 22: fit and validate all subjects

**Input**

- validated fitting pipeline;
- 12 subject sequences.

**Process**

Fit separately, generate predictions, and plot observed versus predicted distributions for every subject.

**Output**

- 12 fit records;
- 12 diagnostic reports;
- group summary built from subject-level results.

**Check**

- no subject disappears from the report;
- noisy or failed fits are shown explicitly.

## Step 23: compare models

**Input**

- validated fits from sensory-only, Basic Bayesian, Switching, and hierarchical confidence models.

**Process**

Compare subject-level NLL, AIC, held-out prediction, and distribution-shape accuracy.

**Output**

- model comparison table and plots.

**Check**

- identical data rows and response models are used;
- AIC is interpreted only after distribution validation and recovery.

## End-to-end pseudocode

```text
for subject in subjects:
    trials = chronological_trials(subject)
    H = broad_initial_hyper_state()
    total_log_likelihood = 0

    for trial in trials:
        H_before = discount(H, rho)
        effective_prior = average_prior_over_kappa(H_before, mean=225)

        response_pmf = predict_response(
            true_direction=trial.direction,
            coherence=trial.coherence,
            prior=effective_prior,
            posterior_readout="one shared readout",
            latent_measurement=True,
            motor_noise=kappa_motor,
            lapse=lapse,
        )

        if trial.response_valid:
            total_log_likelihood += log_probability(
                response_pmf,
                trial.response_angle,
            )

        H = update_confidence_after_feedback(
            H_before,
            feedback_direction=trial.direction,
            fixed_mean=225,
        )

    subject_nll = -total_log_likelihood
```

The ordering in this pseudocode is part of the scientific model. Prediction comes before feedback update.

