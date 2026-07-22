# Recombined observer (HB - Rachel × HB - Salma): build + tests

## What it is
A "best-of" cross built to serve the abstract's three targets (always
integrate; learn prior confidence online; emergent, not imposed, bimodality).
File: `observers/models/hb_integrate_before.py` — a subclass of
`HBIntegrationObserver` that touches none of the existing model's code.

Component choices (each serves one abstract target):

| Component | Taken from | Why |
|---|---|---|
| Combination rule: integrate belief into ONE effective prior, then ONE read-out | HB - Salma | the honest single-posterior version of "the observer integrates" |
| Explicit prior floor  α·V(θ;225,κ) + (1−α)/360 | HB - Rachel | separates "how much to trust prior" (α) from "how tight" (κ); drives emergent bimodality |
| Learned belief b_t(κ) over prior concentration | both (shared) | the hierarchical latent |
| Forgetting: linear leak toward anchor  (1−λ)b + λ·b₀ | HB - Rachel | re-opens ruled-out widths in one trial (suits abrupt block changes); anchor can be informative |
| Grid: 360-bin (1°), plain MAP read-out | HB - Rachel | smooth predictions, no tie-aware step needed |

Parameter count: **7** (3 k_like + α + k_motor + p_random + λ) — unchanged from
HB - Rachel, because the combination-rule swap adds no parameters. AIC
comparisons against the switching observer stay clean.

## Test 1 — does bimodality still emerge? YES
Concern: moving the α-floor inside a single integrated posterior (rather than
averaging read-outs) might kill the second mode. Verdict: it survives.

Using a mass-split criterion (genuine bimodality = substantial mass near BOTH
the stimulus and the prior mean 225°), with the belief settled on a wide
(SD80-like) prior, coherence 0.06:

| stimulus | dist. from prior | mass near stim | mass near prior | bimodal |
|---:|---:|---:|---:|:--:|
| 225 | 0°  | 0.70 | 0.70 | no (unimodal at prior — correct) |
| 185 | 40° | 0.19 | 0.63 | yes |
| 145 | 80° | 0.12 | 0.45 | yes |
| 105 | 120°| 0.20 | 0.27 | yes |

Unimodal exactly at the prior mean, bimodal as the stimulus moves away — the
signature the abstract promises, produced by integration with no switching rule.

Numerical caveat: integrate-then-read-out on a single broad prior leaves
low-amplitude discretisation ripple in the MAP read-out (~0.002 vs a ~0.027
main peak). It is cosmetic — it does not affect the mass split — but a naive
local-maxima mode counter will over-count it. HB - Rachel's average-over-
read-outs washes this ripple out; this is the one numerical cost of the swap.
(Fixable with light smoothing or a BLS rather than MAP read-out if it matters.)

## Test 2 — are α and λ jointly recoverable? YES
Concern: both modulate effective prior reliance and could trade off. Verdict:
they are jointly identifiable.

Simulated 2400 trials (4 blocks SD80/40/20/10, 3 interleaved coherences) from
the recombined model with true α=0.60, λ=0.15; mapped the NLL over the (α, λ)
plane with the other parameters at truth.

- NLL minimum lands EXACTLY on truth: α=0.60, λ=0.15.
- The ΔNLL surface is a ROUND BASIN, not a diagonal ridge — moving either
  parameter away from truth costs likelihood regardless of the other. No
  confound. (Figure: `recombined_alpha_lambda_recovery.png`.)
- The λ=0 column is 400–700 ΔNLL worse than the optimum: the forgetting term
  is strongly identified — the data cannot be explained without it.
- The basin is shallower along λ than α (α∈[0.55,0.65]×λ∈[0.10,0.15] all within
  ΔNLL≈9), so λ is estimated less precisely than α — but the JOINT minimum is
  unique and on-truth, which is what identifiability requires.

## Status
Design recommendation, now backed by two passing checks on simulated data.
Not yet fit to human data. Next step to make it abstract-ready: fit all 12
subjects and AIC-compare against the switching observer and against HB - Rachel
(does integrate-before + linear-forget beat integrate-after empirically?).
