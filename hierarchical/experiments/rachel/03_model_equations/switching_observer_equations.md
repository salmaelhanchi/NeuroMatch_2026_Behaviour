# switching_observer — implemented equations (H1–H5)

The Switching observer of Laquitaine & Gardner (2018) — the model that won
their comparison. Labelled H1–H5 to parallel hb_integration's M1–M5, so the two
documents line up block-for-block. Everything lives on the direction grid
**θ ∈ {1, …, 360}°**, prior mode **μ = 225°**. Building block — the (normalised)
von Mises pmf:

    V(θ; μ, κ) = exp(κ · cos(θ − μ)) / Σ_θ' exp(κ · cos(θ' − μ))

The one-paragraph idea: the Basic Bayesian observer MULTIPLIES likelihood and
prior into one posterior and reports its mode → unimodal estimates. Subjects
were BIMODAL (one peak at the stimulus, one at the prior). The Switching
observer reproduces this by NOT combining them: on each trial it commits to
EITHER the sensory evidence OR the prior mean, choosing between them in
proportion to reliability. Because the choice is stochastic across trials, the
estimate distribution has two peaks.

---

## H1 — Evidence read-out  (prior switched OFF, k_prior = 0)

Girshick MAP push-forward with a flat prior. With no prior pull, the MAP of each
measurement is the measurement itself, so this is the sensory percept
distribution — a bump at the stimulus whose width shrinks as coherence
(k_like) grows:

    p_e(est | θ_true, c) = GirshickMAP( k_like = k_like(c), k_prior = 0 )

Source: `girshick_map_lookup(..., k_prior=0)` → column at θ_true.

---

## H2 — Prior read-out  (likelihood switched OFF, k_like = 0)

Girshick MAP with a flat likelihood. Every measurement now maps to the prior
mode, giving a spike at 225°:

    p_prior(est) = GirshickMAP( k_like = 0, k_prior = k_prior(prior) )
                 ≈ delta(est − 225)

Source: `girshick_map_lookup(k_like=0, ...)` → column at θ_true.

---

## H3 — Switching weights  (paper Eq. 6)  +  lapse

Reliability-proportional competition between the two representations. With
sensory reliability k_e = k_like(c) and prior reliability k_p = k_prior(prior):

    w_prior    = k_p / (k_p + k_e)
    w_evidence = k_e / (k_p + k_e)

Then the two competition weights and the FIXED lapse p_random are renormalised
so the three mixture probabilities sum to exactly 1:

    total = w_prior + w_evidence + p_random
    P_prior    = 1 − (p_random + w_evidence) / total
    P_evidence = 1 − (p_random + w_prior)    / total
    P_random   = 1 − (w_evidence + w_prior)  / total

High coherence (large k_e) → evidence usually wins → veridical estimates.
Low coherence → prior wins more → estimates biased toward 225°.
Source: `_switching_weights()`.

---

## H4 — Percept mixture  (commit to one representation, per trial)

The percept distribution is the reliability-weighted MIXTURE of the two
read-outs plus the lapse floor — NOT a product:

    P(percept) = P_evidence · p_e  +  P_prior · p_prior  +  P_random · (1/360)

(Guard: if k_e = k_p = 0 the observer is at chance → uniform.) The stochastic
per-trial commitment to evidence-or-prior is what makes this a MIXTURE, and the
two components are what make the estimate distribution bimodal.
Source: `estimate_distribution()` step 4.

---

## H5 — Motor noise  (paper Eq. 7)

Circularly convolve the percept with von Mises motor noise centred on 0:

    p(est | θ_true, c, prior) = P(percept)  ⊛  V(· ; 0, k_motor)

then renormalise to a probability vector. Source: `_apply_motor_noise()`.

---

## Fitted parameters (9)  — the SAME set as the Basic Bayesian observer

    k_like(0.06), k_like(0.12), k_like(0.24)         sensory reliability, per coherence
    k_prior(80), k_prior(40), k_prior(20), k_prior(10)  prior reliability, per block width
    k_motor                                          motor-noise concentration
    p_random                                         lapse rate

(N_PARAMS = 9.) Because the parameter set is identical to the Basic Bayesian
(integration) observer, switching buys the bimodality "for free" — no extra
parameters relative to the model it beats.

---

## How this contrasts with hb_integration (M1–M5)

Same two ingredients (a sensory read-out and a prior at 225°), combined by
OPPOSITE rules:

- Switching (H4): MIXTURE — commit to evidence OR prior each trial, weights from
  the reliability ratio k_p/(k_p+k_e). Bimodality is IMPOSED by the stochastic
  commitment.
- hb_integration (M4): INTEGRATION — one posterior per κ (likelihood × mixed
  prior), MAP read-out, averaged over the learned belief b_t(κ). Bimodality
  EMERGES from the uniform floor of the mixed prior, with no switch.

Two structural differences beyond the combination rule:

1. Learned latent. Switching's k_prior is a FIXED fitted constant per block
   width (no trial-by-trial dynamics). hb_integration LEARNS the prior
   concentration online as a belief b_t(κ) updated from feedback (its M5) — the
   promotion of a fixed parameter into an evolving latent is the whole point of
   the hb_integration extension.
2. Weight origin. Switching's reliance on the prior is a DERIVED ratio
   k_p/(k_p+k_e) (Eq. 6), not a free parameter. hb_integration's mixture weight
   α IS a fitted parameter, held fixed across blocks.
