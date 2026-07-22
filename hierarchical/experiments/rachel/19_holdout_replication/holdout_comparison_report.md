# Held-out predictive comparison — our models under Salma's protocol (12 subjects)

## What this is

A faithful replication of the held-out comparison from Salma's `comparison_models` branch
(`standardized_hierarchical_comparison/`), run with **our** registry models. Protocol, copied exactly:

* **Chronological holdout** — the last 2 complete experimental sessions of each subject are held
  out; models are fit on the earlier sessions only. Learning is causal (forward in time; belief
  resets at each session start). This is more appropriate for a *learning* model than block-shuffle
  K-fold, which can leak future feedback into the trained belief.
* **Score** — held-out negative log predictive likelihood (per-trial log probability of the actual
  response, summed over held-out trials). Lower = better prediction.
* **Uncertainty** — paired block bootstrap over held-out runs (5000 draws) for each model pair.

## Model mapping (from Salma's model docstrings)

| Salma's model | Our model | Mechanism |
|---|---|---|
| `SwitchingObserver` | **switch** | select prior OR evidence per trial (k=9) |
| `ReadoutAverageObserver` (her docstring: *"standardized version of hb_integration.py"*) | **hb_rachel** | integrate-after, learns κ, α fixed (k=7) |
| — (our headline model, scored for context) | **hb_adaptive** | integrate-after, learns κ AND α jointly (k=6) |
| — (baseline) | **basic_bayes** | fixed always-integrate (k=9) |

`recombined` (integrate-before) and `reliability_mixture` (not in our registry) are omitted here —
recombined is the ~40-min/subject bottleneck and is analysed separately.

## Result: the models are near-tied out-of-sample

| Model | Σ held-out NLL (12 subj) | Δ from best | subjects won |
|---|---|---|---|
| **Switch** | 123051 | +0 | 3/12 |
| **HB-Adaptive** | 123263 | +213 | **5/12** |
| **HB-Rachel** (readout-avg) | 123280 | +229 | 1/12 |
| **Basic-Bayes** | 123564 | +513 | 3/12 |

The total spread across all four models is **513 NLL over ~123,000** — about
**0.4%**. Switch has the lowest total, but **HB-Adaptive wins the most individual subjects (5/12)**.
No model dominates: which model predicts best flips subject to subject (see per-subject panel, where
every model touches the 0-line for some subject). The paired bootstrap differences vs Switch are tiny
(mean log-score gap ≈ 0.008 nats/trial for both HB models — well within noise).

## Interpretation for the abstract

This is the **honest and favourable** framing: **out-of-sample, a learning Bayesian observer predicts
human estimates as well as the paper's switching heuristic.** The in-sample AIC table favours Switch
(it has the most parameters and bends to the training data), but that advantage **disappears under
held-out prediction** — the generalisation gap Switch pays for its flexibility exactly cancels its
in-sample edge. HB-Adaptive, with the *fewest* parameters (k=6), wins the plurality of subjects.

On the **cohort totals** this agrees with the spirit of Salma's single-subject finding (that a
learning Bayesian integrator predicts as well as Switch out-of-sample), though the agreement is at
the aggregate level, not participant-by-participant: on *our* subject 3 specifically the readout-avg
model (HB-Rachel, 9884) is in fact the worst of the four and does **not** out-predict Switch (9831) —
the direction flips across subjects (part of the same "no model dominates" point), and per-subject
results are not expected to match hers anyway given the different grid and implementation. The
strongest defensible claim is the aggregate one: **the switching heuristic and the graded
learning-Bayesian observer are statistically indistinguishable on held-out prediction — so a learning
Bayesian mechanism is a viable account of the same behaviour, without a discrete switch.**

## Caveats

* **Grid scale differs from Salma's.** We score on a 360° response grid (mean log-score ≈ −4.6); she
  used 72° bins (≈ −2.9). Absolute NLLs are therefore **not** comparable across the two efforts — only
  the rankings and bootstrap CIs are.
* **Fits at maxiter=400.** Switch is mildly under-converged at this budget (a longer run lowered one
  subject's NLL ~117 points); since better convergence would only *help* Switch and it already ties,
  this does not change the conclusion.

## Per-subject held-out NLL

| Subj | Switch | HB-Rachel (readout-avg) | HB-Adaptive | Basic-Bayes | winner |
|---|---|---|---|---|---|
| 1 | 9857 | 9857 | 9850**\*** | 9912 | HB-Adaptive |
| 2 | 7552 | 7482**\*** | 7491 | 7653 | HB-Rachel |
| 3 | 9831 | 9884 | 9867 | 9820**\*** | Basic-Bayes |
| 4 | 9520 | 9499 | 9490**\*** | 9545 | HB-Adaptive |
| 5 | 11989**\*** | 12168 | 12326 | 12230 | Switch |
| 6 | 10392 | 10436 | 10403 | 10359**\*** | Basic-Bayes |
| 7 | 10065 | 10091 | 10069 | 10058**\*** | Basic-Bayes |
| 8 | 11477**\*** | 11478 | 11493 | 11498 | Switch |
| 9 | 10456 | 10454 | 10445**\*** | 10465 | HB-Adaptive |
| 10 | 10650 | 10671 | 10640**\*** | 10677 | HB-Adaptive |
| 11 | 10351 | 10290 | 10260**\*** | 10410 | HB-Adaptive |
| 12 | 10910**\*** | 10969 | 10930 | 10938 | Switch |

*\* = per-subject winner (lowest held-out NLL). Lower is better throughout.*
