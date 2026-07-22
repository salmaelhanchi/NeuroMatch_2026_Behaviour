# All-12-subject model comparison — the batch that settles the ordering

Six observer models fit to all 12 subjects, each model fairly multi-started,
reporting AIC and BIC. This replaces the earlier 2-subject (1–1) result, which
was too thin to rank anything.

Models: **static** (9p), **online** (6p), **asymptote+transient / AT** (11p),
**adaptive-volatility** (6p) — the *switch family*; **integration free-α** (7p)
and **integration derived-α** (6p) — the *integration family*.

## Headline result

**The switch family wins 9 of 12 subjects on AIC; integration wins 3.** On BIC
(heavier complexity penalty) the split is **8–4**: subject 7 flips from static
(AIC winner) to integration derived-α, because BIC's heavier penalty on static's
9 parameters loses to derived-α's 6. So integration wins S1, S2, S6, S7 on BIC.
Either way, switching-style models are the majority account — the *opposite* of
what the abstract originally hoped.

**AT is the single best model overall.** It wins 7/12 subjects on AIC (5/12 on
BIC) and has the lowest summed AIC and BIC across the group. The paper's static
switch observer is second. So the within-block *learning transient* AT adds to
the switch is carrying real explanatory weight — the strongest positive finding
of the project.

## Per-subject AIC winner

| subject | AIC winner | Δ to 2nd best | note |
|---|---|---|---|
| 1 | integration free-α | 91.6 | integration-like |
| 2 | integration free-α | 285.4 | integration-like (strong) |
| 3 | AT | 6.2 | switch, near-tie with adaptive/static |
| 4 | AT | 10.9 | switch |
| 5 | AT | 208.0 | switch (strong) |
| 6 | integration derived-α | 6.1 | integration, near-tie with AT |
| 7 | static | 1.0 | switch, tie with derived-α |
| 8 | online | 2.0 | switch, near-tie (adaptive/AT) |
| 9 | AT | 37.1 | switch |
| 10 | AT | 141.8 | switch (strong) |
| 11 | AT | 45.7 | switch |
| 12 | AT | 23.8 | switch |

AIC winner tally: **AT 7, integration-free 2, integration-derived 1, static 1,
online 1.** BIC winner tally: AT 5, integration-derived 2, integration-free 2,
adaptive 1, static 1, online 1.

## Group-level fit (summed across 12 subjects, Δ from best)

| model | ΣAIC − best | ΣBIC − best |
|---|---|---|
| **AT** | **0** | **0** |
| static | 683 | 519 |
| adaptive-volatility | 2284 | 1874 |
| online | 2320 | 1911 |
| integration derived-α | 2869 | 2460 |
| integration free-α | 2973 | 2646 |

## What each new model taught us

**AT (learning transient) is validated at scale.** Not a 2-subject fluke — it is
the best model on the majority of subjects and at the group level. The within-
block transient in prior reliance is a real, generalisable effect the paper's
static observer cannot produce.

**Adaptive-volatility did NOT beat AT on the full sample.** On 2 subjects it had
edged AT; across 12 it trails AT by ~40–950 AIC per subject (0/12 wins vs AT,
within 10 AIC on only 2/12). Interpretation: the *extra freedom* of AT's per-
block asymptotes and asymmetric τ genuinely fits these subjects better than the
single-hazard adaptive learner — i.e. subjects' within-block dynamics are richer
than a pure change-point rule with one hazard. Adaptive-volatility remains the
more principled model (boundary-agnostic, 6p) and the better *scientific* story,
but AT is the better *fit*. Both belong in the paper: AT as the best descriptive
account, adaptive-volatility as the normative one that shows boundary knowledge
is not strictly required.

**Derived-α vs free-α integration is subject-dependent.** Derived-α (the
authors' reliability-ratio formulation) beats free-α on 6/12 subjects and loses
on 6 — and where integration wins the subject outright (1, 2, 6), it is free-α on
1 & 2 but derived-α on 6. So neither integration variant dominates; the mixture
weight's treatment matters per individual. Consistent with the paper's remark
that the reliability-weighted mixture is a "reformulation" that behaves much like
the switch.

**Individual differences are real.** 3 subjects (1, 2, 6) are genuinely
integration-like; the other 9 are switch-like. This is exactly the "substantial
individual differences" the abstract predicted — just resolved in favour of
switching for the majority rather than integration.

## Bottom line for the abstract / paper

- The honest headline is **not** "integration replaces the switch." It is:
  *switching-style models, and especially a switch with an added within-block
  learning transient (AT), give the best account of most subjects; a minority are
  better explained by Bayesian integration.*
- The genuinely new, defensible contribution stands: **the within-block learning
  transient** (AT wins the group) and the **boundary-agnostic normative version**
  (adaptive-volatility) that shows the transient does not require knowing block
  boundaries.
- The original abstract's mechanism (mixed hyper-prior integration replacing the
  switch) is **not** supported at scale — which is itself a clean, publishable
  result, and matches the paper's own finding that the switch beats mixture
  priors.

## Caveats
- Single fit per (subject, model) with fair multi-start; AIC noise is ~tens of
  AIC, so near-ties (subjects 3, 4, 6, 7, 8) should be read as ties, not
  rankings. The strong wins (2, 5, 10) are robust.
- `k_e[0.24]` still hits the motor-noise ridge in several subjects (shared across
  all models; does not affect model *ordering* since it is common).
- No cross-validation yet (AIC/BIC only). PPCs and the late-block within-trial
  bimodality test remain the qualitative complements to this quantitative ranking.

## Files
- `all12_model_comparison.png` — ΔAIC heatmap (per subject) + group-level ΣAIC/ΣBIC.
- `all12_model_comparison.csv` — full AIC & BIC, every subject × model.
- `all12_summary_by_model.csv` — summed AIC/BIC per model.
- `all12_fit_results.json` — all fitted parameters, every subject × model.
- `batch_fit_all.py` — the resumable harness (one subject per process).
