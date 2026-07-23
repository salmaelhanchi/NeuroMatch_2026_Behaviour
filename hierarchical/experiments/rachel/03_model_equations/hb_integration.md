# Hierarchical Bayesian *integration* observer — the abstract's model

This is the model our project abstract actually proposes, and the piece the
folder was missing. The switching family (`switching_observer.py`,
`online_switching_observer.py`, `asymptote_transient.py`) all *impose* a selection between
prior and evidence. The abstract asks the opposite question:

> test whether switching-like perceptual behavior **requires** an explicit
> selection mechanism or can **emerge** from adaptive hierarchical inference.

So this model imposes no switch. Files: `models/hb_integration.py` (model),
`verification/verify_hb_integration.py` (checks), `fitting/hb_integration_fit.py`
(recover / human / cv) — all under the `observers/` package.

## The idea

Put a **mixed hyper-prior** on the true direction — the abstract's "peaked von
Mises at the mean and uniform prior":

    p(θ | κ, α) = α · V(θ; 225, κ)  +  (1 − α) · (1/360)

and read out the **MAP of the ordinary Bayesian posterior** `V(θ; m, k_e) ·
p(θ | κ, α)`. Bimodality then falls out for free:

- a measurement near 225 → the von Mises component dominates → estimate pulled
  toward the prior;
- a measurement far from 225 → the uniform floor dominates → estimate sits at
  the sensory evidence.

Sweeping the internal measurement therefore produces a response distribution
with a peak near the stimulus **and** a peak near the prior — with no selection
rule. The "switch" is a *derived responsibility* (which mixture component
explains the measurement), not a hand-coded competition.

## Equations (M1–M5, paralleling the switch model's H1–H5)

- **Prior:** the mixture above.
- **(M1) Learning.** A belief `b_t(κ)` over prior precision, updated by the same
  predict/correct filter as the switch model, but the feedback likelihood is the
  *mixture* `α·V(f;225,κ) + (1−α)/360`. The block manipulation is a width
  manipulation (SD 10/20/40/80), so κ is exactly what changes and must be
  tracked; α ("does the prior ever apply") is held fixed across blocks.
- **(M2) Read-out.** MAP push-forward of the mixture posterior — a vectorised
  variant of the paper's Girshick lookup (`mixture_map_lookup`).
- **(M3–M4) Estimate.** Motor-convolve each per-κ read-out, then **average over
  the belief** `Σ_κ b_t(κ)·readout(κ)`, then fold in the lapse. (Read-out-then-
  average — the same H4 convention `online_switching_observer` uses, so any fit difference
  is switch-vs-integration, not a mismatched marginalisation rule.)
- **(M5) Likelihood.** The belief path is deterministic given feedback, so
  `NLL = −Σ_t log p(θ̂_t | …)`; no particle filter. AIC/BIC included.

## Free parameters — 7

`k_e(0.06/0.12/0.24)` (3) · `α` prior confidence (1) · `k_motor` (1) ·
`p_random` lapse (1) · `λ` volatility (1). Compare: static switching 9, online
switching 6, asymptote+transient 11. The four block prior strengths are
*emergent* (learned via κ), not fitted.

## Verification (`verify_hb_integration.py` — 12/12 pass)

- **Reduction:** `α=1` reproduces `girshick_map_lookup` exactly (max|Δ| ≈ 1e-17)
  — the model genuinely nests the paper's single-von-Mises machinery.
- **Emergence:** far-from-prior + low coherence → two peaks (near stimulus and
  near 225); near-prior and high-coherence → unimodal. Reproduces **both**
  unimodal and bimodal responses, as the abstract claims — with no switch.
- **α→0:** the prior vanishes, the estimate follows the evidence (unimodal).
- **Discriminator vs the switch model** (the key check — the switch model is
  bimodal too): the integration model's prior reliance **declines with the
  stimulus-to-225 distance** (window mass 0.062 → 0.001 over distances 60–120°),
  because a far measurement is poorly explained by the von Mises component; the
  switch model's prior weight is **flat** across direction (0.266 → 0.240) at
  fixed coherence/belief. This is a falsifiable behavioural difference, testable
  against the data.
- **Learning:** the belief over κ converges to a known κ (learned SD 40.9 vs
  true 40); the recursive update equals the batch posterior at λ=0.
- **Cost:** one full-subject likelihood ≈ 0.16 s ⇒ ~0.5 min/subject to fit.

## Parameter recovery (`hb_integration_fit.py recover`)

All 7 parameters recover from data simulated with known parameters (3 seeds):
k_e, k_motor, p_random within ~10%, and **λ recovers well** (0.045 vs 0.05) —
better than the switch model's weakly-identified λ. Crucially **α is
identifiable** (0.602 ± 0.006 vs truth 0.6) and does **not** trade off with the
lapse (p_random 0.029 vs 0.03) — the two uniform-mass mechanisms (in the prior
vs in the response) are separable in this design.

## Human fits + model comparison (all four models multi-started)

Fit on the same trials, **every model given equal multi-start effort**
(integration + AT: cold/warm/variant; online + static refit fairly via
`fair_refit.py`, warm-started from the AT solution which nests static). AIC:

| subject | integration | online | static | AT | best (Δ over 2nd) |
|---|---|---|---|---|---|
| 1 | **76937.8** | 77130.5 | 77072.1 | 77251.6 | integration (+134 over static) |
| 3 | 84305.0 | 83484.6 | **83474.8** | 83488.3 | static (+9.7; integration +810 behind) |

**The fair comparison split 1–1, and it vindicated the fairness worry.** Giving
the switch models equal multi-start effort improved them a lot on subject 1 —
static 77791→**77072** (~719), online 78100→**77131** (~969) — exactly the
local-basin headroom integration's own multi-start had revealed. Integration
still wins subject 1, but its lead over the next model collapsed from ~314 (vs a
single-start rival) to **~134** (vs a fairly-fit static). On subject 3 the three
switch models are effectively tied (83475 / 83485 / 83488, a ~14-AIC spread) and
integration is clearly worst (+810). So: **integration is the best account for
subject 1 and among the worst for subject 3.**

**Subject 1 — the win is real, but the abstract's *mechanism* lost.** The fit
drove α→1.0, switching **off** the uniform component — by the reduction (T1)
that is the classic single-von-Mises Girshick integration observer, *not* the
mixed hyper-prior the abstract proposes. So what wins is **plain Bayesian
integration + online κ-learning**, and the data *reject* the uniform-mixture
ingredient for this subject. (`k_e[0.24]` runs to a large value on a flat ridge —
once `k_e ≫ k_motor` the motor noise dominates and the likelihood is insensitive
to it; treat it as unidentified, not a finding.) Subject 3's fit, by contrast,
*kept* the mixture (α≈0.60) and still lost — the mixture doesn't rescue it.

Crucially, a **posterior-predictive check** shows the win is not a bulk/precision
artifact. Simulating from the fitted model and pooling the far-from-prior,
low-coherence trials reproduces the bimodality in the data:

    near-stimulus mode   near-prior mode
    data                 0.45                 0.18
    fitted model         0.38                 0.18

Per *trial* the prediction is unimodal, but the believed prior SD swings 8→100°
across each block, so pooled over trials the model places mass at **both** the
stimulus (early block, weak belief) and the prior (late block, sharpened belief).
That is bimodality **emerging from online inference with no explicit switch** —
direct support for the abstract's hypothesis, achieved by κ-learning rather than
the uniform mixture.

**Subject 3 — AT (switch-family) still wins (~817).** Here the fit *kept* the
mixture (α≈0.60, λ≈0.05, close to recovery), yet integration trails. So subject 3
is better described by a switch.

Net: integration is a real competitor — it fairly beats AT on subject 1, and does
so by capturing the bimodality through learning, not a switch. But it is not
uniformly preferred (loses on subject 3), and on subject 1 it wins by *shedding*
the abstract's uniform-mixture mechanism. The abstract's core claim (bimodality
need not require a switch) gets **one supporting subject**; its specific
mechanistic proposal (the mixed hyper-prior) is not what carried it.

## Honest status / next steps

Built, verified (12/12), recoverable (incl. α), fast (95 ms/eval after the
`_map_readout` caching fix), multi-start + crash-safe, and now compared **fairly**
(all four models multi-started, `fair_refit.py`). It is the best account for
subject 1 (by ~134, with a posterior-predictive-validated bimodality mechanism)
and clearly worst for subject 3. One supporting subject, one against — the
abstract's hypothesis has real but partial support on this 2-subject sample.

Still owed before any real verdict — these are the team's calls:
1. **All 12 subjects**, background batch (~3 min/subject each; per-subject jobs
   to avoid the results-file race — now guarded by a re-read-merge on write; the
   switch models need `fair_refit.py`, not the single-start `online_fit_human`).
2. **Cross-model CV** (the `cv` mode currently scores integration only).
3. **Decide α**: since the winning subject-1 fit sits at the α=1 boundary,
   profile/compare α-free vs α-fixed=1 to state cleanly whether the uniform
   mixture ever helps, or whether κ-learning alone is the story.
4. Both subject-1 leads and subject-3 gaps are single-fit AICs; fit noise is
   ~tens of AIC, so the subject-1 +134 is meaningful but the subject-3 switch
   ordering (~14 spread) is a tie, not a ranking.

Verification note: the 12/12 suite characterised α=0.6; the subject-1 winner sits
at α=1, covered by T1 (reduction) and the posterior-predictive above, but a
dedicated check at the winning configuration would close the gap. CV caveat: it
preserves trial order but the belief still sees held-out feedback (the exogenous
stimulus direction), so it scores predictive fit of responses with order intact,
not a strictly causal forecast.
