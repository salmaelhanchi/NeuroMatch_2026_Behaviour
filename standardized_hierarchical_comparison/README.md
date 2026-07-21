# Standardized hierarchical observer comparison

This folder compares four participant-level observer models under one shared
observation and fitting contract. The comparison deliberately excludes the
experimenter's `prior_std` label from every model input.

## Shared contract

- Participant 1 and participant 3 have dedicated comparison notebooks.
- Trials are ordered by session, run, and trial index.
- Fixed prior mean: 225 degrees.
- True-direction feedback updates each stateful model after response prediction.
- Latent state resets at session boundaries and carries across blocks within a
  session.
- 72 circular response bins (5 degrees each) for every model.
- Identical motor-noise convolution, direct lapse mixture, response binning,
  negative log likelihood, AIC, BIC, optimizer, and multistart schedule.
- `prior_std` is neither required by the loader nor retained in the prepared
  participant object.

## Models

1. `SwitchingObserver`: the original Switching response model from
   `Switching_Bayesian_Observer_starter2.ipynb`, standardized to one fitted
   participant-level prior concentration. It switches between a point mass at
   the prior mean and a sensory response using the prior/sensory reliability
   ratio.
2. `ReadoutAverageObserver`: the first-level `hierarchical` implementation.
   It forms a posterior and MAP readout separately for each possible prior
   concentration, then averages response distributions over the belief in
   concentration.
3. `ReliabilityMixtureObserver`: the nested reliability-mixture implementation.
   It learns a scalar prior-reliance gate, but now uses one participant-level
   prior concentration instead of four `prior_std`-indexed concentrations.
4. `IntegratedPriorObserver`: the independent implementation. It first
   marginalizes concentration uncertainty into one effective prior, then forms
   one posterior and one tie-aware MAP readout.

These are observer models with maximum-likelihood parameter fitting. They are
not population-level hierarchical Bayesian parameter models.

## Run

Open the required comparison notebook and run all cells:

- `notebooks/01_participant_1_three_model_comparison.ipynb` retains the first
  standardized pilot.
- `notebooks/02_participant_3_bimodal_comparison.ipynb` uses participant 3,
  gates score interpretation on optimizer convergence, and reproduces the
  paper's 6%-coherence, 80-degree-prior, five-direction diagnostic with
  15-degree response bins. The prior-width label is used only to select trials
  for this post-fit plot and is never passed to an observer. Participant 3 was
  selected because the paper's Figure 5F shows `sub03` as a clear bimodality
  example and Figure 5E reports a 670-point AIC advantage for switching. This
  notebook defaults to two starts and 60 objective evaluations per start for
  interactive exploration. Set `HB_MAX_EVALUATIONS_PER_START=250` only for a
  slower convergence check.
- `notebooks/03_participant_3_predictive_switching_comparison.ipynb` fits all
  four models to sessions 1-7 and scores complete held-out sessions 8-9. It
  reports trial-level predictive log likelihood and paired run-block bootstrap
  intervals for every model pair.

The predictive notebook defaults to two starts and 250 objective evaluations
per start because it is intended as the more decisive held-out comparison.
Override this with `HB_MAX_EVALUATIONS_PER_START`. Set `HB_SMOKE_TEST=1` before
execution for a short end-to-end check using the same full trial sequence.

Outputs are written to the participant-specific in-sample, predictive, or smoke
directory under `outputs/`.

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```
