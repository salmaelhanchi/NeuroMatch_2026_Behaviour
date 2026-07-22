# How to confirm the two abstract models work

Applies to the two models the abstract compares: **`switching_observer`** (the
paper's model) and **`hb_integration`** (the abstract's hierarchical Bayesian
model). "Does it work" has three separate meanings, each needing its own
evidence. All three pass for both models.

## Level 1 — Correctness (does the code compute the equations?)

Each model ships a verification suite that checks it against **oracles with
known answers** — not just "does it run". Run from the repo root with
`PYTHONPATH=.`:

```
python -m observers.verification.verify_switching        # 5/5 PASS
python -m observers.verification.verify_hb_integration   # 12/12 PASS
```

What these actually prove:

**switching_observer (5/5):**
- evidence read-out equals a pure von Mises V(·; d, k_like)  (max|Δ| ~ 1e-17)
- prior read-out is a delta at 225 (all mass at the prior mean)
- switch weights match the paper's Eq. 6 reliability ratio exactly
- all 108 condition distributions are valid (normalised, non-negative)
- NLL is lower at the generating params than at deliberately-wrong ones

**hb_integration (12/12):**
- **reduction:** with alpha=1 the model collapses to the exact Girshick
  Bayesian estimator (max|Δ| ~ 1e-17, machine precision) — proves the
  integration read-out is implemented correctly.
- **emergence:** bimodality (a peak near the stimulus AND near the prior)
  arises from Bayesian inference, not a hand-coded switch.
- **online learning:** the belief over prior strength converges to the true
  kappa (learned SD 40.9 vs true 40), and the recursive update equals the
  batch posterior (max|Δ| ~ 1e-36).
- **discriminator (the scientifically important one):** integration's
  prior-reliance DECLINES as the stimulus moves away from the prior
  (CV=1.34), while the switch's stays flat (CV=0.04). This proves the two
  models make *different, distinguishable* predictions — without it, a model
  comparison would be meaningless.

## Level 2 — Fittability (can it be fit to a real subject?)

```
python -m observers.fitting.switching_observer_fit human 1
python -m observers.fitting.hb_integration_fit human 1      # ~3-5 min/subject
```

Subject 1 (n=8562 trials):

| model | NLL | AIC | notable fitted params |
|---|---|---|---|
| switching_observer | 38869.7 | 77757.4 | implied prior SDs 96/78/36/8 deg track the 80/40/20/10 deg blocks |
| hb_integration | ~38950 (capped) | ~77915 | alpha ~ 0.53, ordered k_like 1.0/10/49 |

Both return sensible, correctly-ordered parameters. **Known quirk:** the
switch's `k_like[0.24]` runs to a very large value (~1e5) — at high coherence
the likelihood is nearly a delta, so its exact concentration is weakly
identified and the optimiser pushes it to the ceiling. It means "very
reliable"; it is not a bug, but do not over-interpret that one number.

## Level 3 — Parameter recovery (is a fitted number trustworthy?)

The decisive test: simulate data from KNOWN parameters, refit, and check the
fit returns them. If recovery fails, no parameter reported from real data means
anything.

```
python -c "from observers.fitting.online_recovery import parameter_recovery; parameter_recovery()"
python -m observers.fitting.hb_integration_fit recover
```

**switching_observer:** k_like recovers nearly exactly (2.89 vs 3.0; 7.73 vs
8.0). k_prior recovers the correct ordering and magnitudes; the tightest prior
(SD10) is the noisiest (6.5 vs 8.7) because a tight prior yields few
informative trials — an expected, not pathological, effect. Sensory
reliabilities recover better than prior reliabilities.

**hb_integration:** every parameter recovers within ~10% across three seeds.
Critically, **alpha** — the model's defining mixture weight — clusters at
0.602 +/- 0.006 (truth 0.6) and stays separated from the lapse rate. That
alpha-vs-lapse confusion is the classic mixture-model failure mode; recovery
confirms it does not occur here, so a fitted alpha is meaningful.

## Summary

| | switching_observer | hb_integration |
|---|---|---|
| Correctness (verification) | 5/5 PASS | 12/12 PASS |
| Fittability (real subject) | yes | yes |
| Parameter recovery | yes (k_like tight; k_prior ordered) | yes (all <~10%; alpha identifiable) |

Both models are verified, fittable, and identifiable. The remaining work for
the abstract is the head-to-head comparison (Task 5): fit both to every subject
under identical preprocessing and compare NLL/AIC/BIC plus cross-validated
held-out likelihood.
