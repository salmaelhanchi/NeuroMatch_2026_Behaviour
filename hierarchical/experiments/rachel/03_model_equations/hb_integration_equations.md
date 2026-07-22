# HB - Rachel (hb_integration observer) — implemented equations (M1–M5)

Everything lives on the discrete direction grid **θ ∈ {1, …, 360}°**, with the
prior mode **μ = 225°**. The building block is the (normalised) von Mises pmf:

    V(θ; μ, κ) = exp(κ · cos(θ − μ)) / Σ_θ' exp(κ · cos(θ' − μ))

κ (kappa) is the concentration — large κ = narrow bump, small κ = broad.

---

## M1 — Mixed hyper-prior  (the model's defining object)

A peaked von Mises at the prior mean, unioned with a uniform floor:

    p(θ | κ, α) = α · V(θ; 225, κ)  +  (1 − α) · (1/360)

- α  = mixture weight ("does the prior structure apply at all")
- κ  = prior concentration (how tight the belief about 225° is)

Source: `mixture_prior()`.

---

## M2 — Sensory likelihood + measurement model  (per coherence c)

With sensory concentration k_like(c) (one per coherence 0.06 / 0.12 / 0.24):

    measurement:   P(m | θ_true) = V(m; θ_true, k_like(c))
    likelihood:    P(θ | m)      = V(θ; m,      k_like(c))

---

## M3 — Posterior, MAP read-out, motor noise  (per fixed κ)

For each internal measurement m, form the posterior, take its mode (ties shared
equally), then marginalise over the measurement to get the read-out for that κ:

    posterior(θ | m)  ∝  P(θ | m) · p(θ | κ, α)

    R_κ(est | θ_true) = Σ_m  P( est = mode[ posterior(· | m) ] ) · P(m | θ_true)

Each per-κ read-out is then circularly convolved with von Mises motor noise
V(· ; 0, k_motor). Source: `_map_readout()` + motor convolution.

---

## M4 — Estimate distribution = belief-average of read-outs, then lapse

THE key structural choice (read-out-then-average): average the per-κ read-outs
over the current belief b_t(κ), then mix in the uniform lapse:

                    [ Σ_κ b_t(κ) · R_κ(est | θ_true) ]  +  p_random · (1/360)
    p_t(est | ·) =  --------------------------------------------------------
                                        1 + p_random

Source: `estimate_distribution()`  ( percept = belief @ readouts ).

---

## M5 — Trial-by-trial belief update over κ  (the "learning of prior confidence")

Between trials the belief leaks toward its initial state (volatility), then is
corrected by the feedback direction θ_fb shown at trial end:

    predict (forget):   b~_t(κ) = (1 − λ) · b_{t−1}(κ)  +  λ · b_0(κ)

    correct (Bayes):    b_t(κ)  ∝  b~_t(κ) · [ α · V(θ_fb; 225, κ) + (1 − α)/360 ]
                        then renormalise so Σ_κ b_t(κ) = 1

- λ (lambda) = volatility / forgetting.  λ = 0 → pure accumulation;
  λ = 1 → belief resets every trial.
Source: `update_belief()` = `forget()` then `bayes_correct()`.

---

## Fitted parameters (7)

    k_like(0.06), k_like(0.12), k_like(0.24)   sensory reliability, per coherence
    α                                          prior mixture weight (fixed across blocks)
    k_motor                                    motor-noise concentration
    p_random                                   lapse rate
    λ                                          volatility / forgetting

κ is NOT fitted — it is the *learned latent*, carried as the belief b_t(κ) over
a ~15-point κ grid and updated every trial by M5.  (N_PARAMS = 7.)

---

## Two things the equations encode that are easy to miss

1. Bimodality is EMERGENT, not switched. In M3, when the measurement lands near
   225° the von Mises term of M1 dominates (estimate pulled to the prior); when
   it lands far, the uniform floor dominates (estimate at the evidence).
   Sweeping m yields two modes with NO switch statement — the "switch" is a
   *derived responsibility* (which mixture component of M1 explains m).

2. Why M4 averages read-outs, not priors. It marginalises κ AFTER the read-out
   (Σ_κ b(κ)·R_κ), not by collapsing the belief into one prior first. This is
   deliberate: it matches the online switching observer's convention so that the
   switch-vs-integration comparison differs ONLY in the read-out rule. (This is
   exactly the axis on which the branch's independent model diverges — it
   integrates into ONE posterior before reading out.)
