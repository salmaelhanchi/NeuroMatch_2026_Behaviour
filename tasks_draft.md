# Tasks Draft

This task list is a planning draft for the project workflow. It separates what is already scaffolded from the next modeling and validation work.

The name next to each task is the primary lead, but everyone is welcome to get hands-on with the data, models, or any other tasks. 

## 1. HB Implementation - Salma

Build the hierarchical Bayesian model so that it learns prior confidence across trials and generates predicted perceptual estimates.

- [x] Define prior, sensory likelihood, posterior, motor noise, and lapse components.
- [x] Implement trial-by-trial updating of prior confidence.
- [x] Preserve circular angle calculations.
- [x] Test the model on simple simulated conditions.
- [x] Confirm that the update affects prior precision, not only prior mean.

## 2. Data Preparation And Model Fitting - Romi

Organize the experimental data and estimate HB model parameters separately for each participant.

- [x] Create a trial-level dataset for all 12 subjects.
- [x] Include subject, block, trial, prior width, coherence, true direction, and estimate.
- [x] Calculate circular estimation error.
- [ ] Fit the HB model to each subject's trial-level responses.
- [ ] Save fitted parameters, predicted distributions, NLL, and convergence status.

## 3. Model Validation - Anirban

Check whether the HB model behaves correctly and whether its fitted parameters and predictions are reliable.

- [ ] Compare observed and model-predicted response distributions.
- [ ] Plot posterior and prior-confidence trajectories.
- [ ] Check whether the hyperprior converges across trials.
- [ ] Perform parameter recovery using simulated data: simulate data from fitted parameters, refit the model, and check whether the original parameters are recovered.
- [ ] Identify unstable or poorly recoverable parameters.

## 4. Switching Observer Implementation - Anirban

Reproduce the paper's model that switches between prior-based and sensory-based estimates instead of integrating them.

- [ ] Implement prior and sensory likelihood distributions.
- [ ] Calculate the probability of selecting the prior versus sensory evidence.
- [ ] Add motor noise and lapse rate.
- [ ] Fit the model separately to each subject.
- [ ] Verify that it can generate bimodal response distributions.

## 5. Model Comparison - Rachel

Determine whether the HB model or the Switching observer better explains and predicts participant responses.

- [ ] Fit both models using the same trials and preprocessing.
- [ ] Calculate NLL, AIC, and BIC.
- [ ] Perform block- or session-based cross-validation.
- [ ] Compare predicted and observed distribution shapes.
- [ ] Report results separately for each subject and summarize group patterns.

## 6. Behavioral Validation Optional - Romi

Test whether the learning predicted by the HB model is reflected in actual behavioral changes. Focus on estimation error rather than reaction time and test whether learning differs across prior and coherence conditions.

- [ ] Measure estimation error across trials within each block.
- [ ] Compare early and late trials.
- [ ] Estimate behavioral learning rates.
- [ ] Separate analyses by prior width and motion coherence.
- [ ] Test whether behavioral learning correlates with hyperprior learning.

## 7. Result Integration And Presentation - Valeria

Combine the modeling and behavioral results into one clear research conclusion and prepare the final abstract, figures, and presentation.

- [ ] Summarize HB validation, model comparison, and behavioral findings.
- [ ] Select representative subject-level and group-level results.
- [ ] Create final figures and tables with consistent labels and metrics.
- [ ] Identify the main conclusion, limitations, and unresolved questions.
- [ ] Prepare the abstract, presentation slides, and speaking roles.

