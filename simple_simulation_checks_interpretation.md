# Simple Simulation Checks Interpretation

Notebook cell:

```text
Simple simulation checks
```

Checklist items covered:

```text
Test the model on simple simulated conditions.
Confirm that the update affects prior precision, not only prior mean.
```

## What This Cell Is

This cell is a scaffold test, not a fit to the experimental data.

Its purpose is to check that the model pieces behave in the expected direction before we start fitting real subject responses.

The simulated situation is:

```text
previous block prior_std = 10
current block prior_std = 80
```

Interpretation:

```text
previous block had a narrow, high-confidence prior
current block has a wide, low-confidence prior
```

So the model should begin the new block with strong carried-over prior confidence, then gradually reduce prior confidence toward the current block target.

## Confidence Path Output

The first displayed tables show the simulated trial-by-trial confidence path.

Important columns:

```text
trial_in_block
prior_mean
prior_kappa
target_kappa
previous_prior_std
current_prior_std
```

Interpretation:

```text
prior_mean = the center/location of the prior
prior_kappa = the precision/confidence of the prior
target_kappa = the confidence implied by the current block prior width
```

The output shows:

```text
prior_mean stays at 225.0
prior_kappa starts high and moves downward toward target_kappa
```

This is the expected behavior for a transition from a narrow prior to a wide prior.

## Precision-Not-Mean Check

The middle table is the direct checklist confirmation.

Important output:

```text
mean_change_deg = 0.0
prior_mean_unchanged = True
prior_precision_changed = True
```

Interpretation:

```text
the prior location did not move
the prior confidence/precision changed
```

This confirms that the online update affects `prior_kappa`, not `prior_mean`.

In model terms:

```text
prior_mean remains tied to the block prior_mean condition
prior_kappa is the hidden confidence state that learns across trials
```

## Component Distribution Output

The final table checks the named model components at two points:

```text
block_start
late_block
```

The components are:

```text
sensory_likelihood
prior
posterior
readout_pdf
estimate_pdf
```

Each row has:

```text
probability_sum = 1.0
```

This means each distribution is normalized correctly.

The simulated sensory evidence peaks at:

```text
sensory_likelihood peak = 235
```

The prior peaks at:

```text
prior peak = 225
```

So the simulated trial creates a clear conflict:

```text
sensory evidence points near 235
prior points at 225
```

At block start, the posterior/estimate peak is closer to the prior because carried-over confidence is strong.

At late block, the posterior/estimate peak moves closer to the sensory likelihood because confidence in the wide current prior has weakened.

## Behavioral Interpretation

This simulated output supports the intended scaffold logic:

```text
previous block context
-> initial prior confidence at current block start
-> within-block confidence learning
-> predicted estimate distribution
```

For this specific toy transition:

```text
10-degree prior block -> 80-degree prior block
```

the model predicts:

```text
early trials: stronger prior pull
later trials: weaker prior pull, estimates closer to sensory evidence
```

## What This Does Not Prove Yet

This cell does not show that the model fits the real experimental data.

It does not yet provide:

```text
optimized parameters
negative log-likelihood from fitted parameters
AIC/BIC
cross-validation
parameter recovery
observed-vs-predicted experimental distributions
```

It only confirms that the scaffold model can run on a simple simulated condition and that the confidence update changes prior precision rather than prior mean.

