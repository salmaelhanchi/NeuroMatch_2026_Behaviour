# Cross-validation: is AT's win overfitting?

**Question.** AT (11 params) won the in-sample AIC/BIC comparison. AIC/BIC
penalize parameter *count* but score fit on the same data used to fit — they
cannot detect overfitting that generalizes poorly. Cross-validation can:
fit on training trials, score held-out trials. If AT's extra parameters absorb
non-replicable noise, its held-out prediction degrades relative to simpler
models.

**Design.** Sequence-preserving 5-fold CV (contiguous trial segments; belief
filters over the full ordered sequence so learning is preserved). Metric:
held-out NLL, no complexity penalty needed. Four models compared: static (9p),
online (6p), AT (11p), adaptive-volatility (6p). Warm-started from each model's
full-data MLE. All 12 subjects.

Caveat (same as the codebase's `cv` mode): feedback = the exogenous stimulus
direction is present on every trial, so the belief "sees" held-out feedback.
This scores predictive fit of *responses* with order intact, not a strictly
causal forecast of *learning*.

## Verdict — AT is NOT globally overfitting, but CV sharpens the picture

**AT wins out-of-sample on 6/12 subjects** — where extra parameters can only
help if they capture real structure. On those subjects the within-block
learning transient is genuine, replicable structure, not noise-fitting.

**But on 4/12 subjects AT's in-sample AIC win was overfitting — and CV caught
it.** On subjects 2, 3, 4, 12, AT was the best *in-sample* model (among these 4)
yet a *simpler* model predicts held-out data better. That is exactly the
signature of overfitting the in-sample criteria missed.

## Held-out NLL: AT − best simpler model (per subject)

Negative = AT generalizes better; positive = a simpler model wins out-of-sample.

| subject | CV winner | AT − best-simpler | reading |
|---|---|---|---|
| 6 | **AT** | −68.7 | AT far better OOS |
| 10 | **AT** | −24.0 | AT better |
| 9 | **AT** | −11.0 | AT better |
| 5 | **AT** | −9.5 | AT better |
| 1 | **AT** | −9.4 | AT better |
| 11 | **AT** | −2.8 | AT marginally better |
| 2 | static | +0.6 | tie (AT ≈ static OOS) |
| 3 | adaptive-volatility | +10.9 | simpler generalizes better |
| 7 | static | +11.3 | simpler generalizes better |
| 4 | online | +31.1 | simpler generalizes better |
| 12 | online | +61.9 | simpler generalizes better |
| 8 | online | +104.7 | simpler clearly better OOS |

**CV winner tally: AT 6, online 3, static 2, adaptive-volatility 1.**

## In-sample → out-of-sample flips (the overfitting diagnostic)

Among the 4 shared models, 4 subjects change winner from AIC to CV:
- S2: AT (in-sample) → static (CV)
- S3: AT → adaptive-volatility (CV)
- S4: AT → online (CV)
- S12: AT → online (CV)

Every flip is AT → a simpler model. None go the other way. This is the clean
statement: **AT's in-sample advantage is partly real (survives CV on half the
subjects, decisively on subjects 6, 10) and partly overfitting (reverses on a
third of subjects).**

## What this means for the paper

1. **The learning-transient mechanism is validated, not a fitting artefact.**
   On the subjects where AT wins in-sample by a large margin (5: ΔNLL 106; 6: 73;
   10: 73), it also wins out-of-sample (−9.5, −68.7, −24.0). Large in-sample wins
   are real. That is the defensible positive result.

2. **AT should NOT be presented as a uniform winner.** Out-of-sample it wins
   only half the subjects. The honest framing: AT is the best model for a
   subset with strong within-block transients; for others (notably the
   online-preferring 4, 8, 12) a 6-parameter learner generalizes better and AT's
   11 parameters are excess.

3. **The 6-parameter models earn their place.** online wins CV on 3 subjects and
   adaptive-volatility on 1 — 4/12 subjects are best described out-of-sample by a
   parsimonious 6-param learner. adaptive-volatility, the boundary-agnostic
   normative model, generalizes better than AT on subject 3 despite losing
   in-sample — vindicating it as the better *generalizer* where it matters.

4. **Report CV, not just AIC/BIC.** The all-12 AIC table (AT wins 7/12) overstates
   AT's standing; the CV table (AT wins 6/12, and *loses* the AIC→CV flip on 4
   subjects) is the honest one. Lead with CV.

## Bottom line
AT performed well **mostly for a real reason** — a replicable within-block
learning transient, confirmed out-of-sample on the subjects where it wins big.
It is **partly overfitting** on a third of subjects, which cross-validation
detected and the in-sample criteria did not. The correct headline is not "AT is
the model" but "a switch with a within-block learning transient is the best
account for subjects with strong transients; simpler online learners suffice for
the rest — and only out-of-sample testing tells them apart."

## Files
- `cv_verdict.png` — per-subject AT−best-simpler (held-out) + AIC→CV flip diagram.
- `cv_holdout_nll_all12.csv` — held-out NLL, all 12 subjects × 4 models, winners.
- `cv_switch_family.py` — the sequence-preserving CV harness (resumable).
