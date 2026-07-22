# HB - Salma (branch hierarchical model) — implemented equations (B1–B5)

The independent "hidden-confidence" observer from the branch
`new_implementaion_with_hiearchial-_fiting`
(`independent_transcript_paper_model/src/hierarchical_confidence/model.py`).
Class docstring: "Discrete hidden-confidence observer with one integrated
posterior." Labelled B1–B5 to line up with hb_integration's M1–M5 and the
switch's H1–H5, so all three documents read block-for-block.

**Numerical grid (fixed support, NOT fitted).** Unlike our 1° / 360-point grid:

    θ grid:   72 bins, linspace(0, 360, endpoint=False) = 0, 5, 10, …, 355°
    κ grid:   [0.0]  ∪  geomspace(0.05, 50, 15)   = 16 support points
              (note: κ = 0 IS in the support → a uniform-prior component is
               reachable through the belief, with no explicit mixture weight)

Everything is computed in LOG space with logsumexp for stability. Building
block — von Mises pmf on the grid, written stably as exp(κ(cos·−1)) then
row-normalised:  V(θ; μ, κ) ∝ exp(κ · (cos(θ − μ) − 1)).

---

## B1 — Prior basis  (pure von Mises per κ — NO mixture floor)

For every κ on the grid, a normalised von Mises at the prior mean 225°:

    prior_basis[κ, θ] = V(θ; 225, κ)

There is no `α·VM + (1−α)/360` mixture here. The only route to a "flat" prior
is the κ = 0 support point (which makes V uniform). Source: `_prior_basis`.

---

## B2 — Hidden-confidence trajectory over κ  (geometric forget + feedback Bayes)

A belief H_t(κ) over the κ grid, carried across ALL trials (never reset at block
boundaries — an explicit pilot choice). Initialised uniform, H_0(κ) = 1/n_κ.
Each trial, in log space:

    predict (forget):   log H~_t = ρ · log H_{t−1}         (then renormalise)
    record H~_t as the PRE-feedback state used for the estimate
    correct (Bayes):    log H_t  = log H~_t + logV(θ_fb; 225, κ)   (then renormalise)

- ρ (rho), 0 < ρ < 1: geometric forgetting. Raising the log-belief to power ρ
  and renormalising shrinks it toward uniform — the smaller ρ, the faster the
  leak toward "no confidence". (Contrast: our M5 forgets LINEARLY toward b_0.)
- The feedback term is the von Mises log-likelihood of the shown direction θ_fb
  under prior mean 225° at each κ. Source: `confidence_trajectory()`,
  `_make_feedback_log_likelihood()`.

---

## B3 — Effective prior = INTEGRATE the belief into ONE prior  (THE key difference)

Marginalise κ BEFORE the read-out: collapse the per-κ prior basis over the
current pre-feedback belief into a single effective prior pmf for the trial:

    effective_prior_t(θ) = Σ_κ H~_t(κ) · V(θ; 225, κ)      (then renormalise)

This is the "one integrated posterior" of the docstring, and it is the exact
axis on which this model differs from ours: hb_integration marginalises κ AFTER
the read-out (average of per-κ read-outs, M4); this model marginalises κ BEFORE
(one prior → one posterior → one read-out). Source: `effective_priors()`.

---

## B4 — Posterior + tie-aware MAP read-out  (one integrated posterior)

With sensory concentration k_sensory(c) (one per coherence), form the log
posterior as sensory likelihood × the single effective prior, then take the MAP
— but ties are handled explicitly (every tied maximum bin shares the read-out
mass equally, rather than an arbitrary argmax):

    log posterior_t(θ | m) = logP(θ | m, k_sensory(c))  +  log effective_prior_t(θ)

    MAP read-out: tied_map = { bins achieving the posterior max }, each weight 1/(#ties)

The coarse 72-bin grid makes exact MAP ties common, which is why the tie-aware
treatment matters here (it is a genuine numerical difference from our 360-bin
argmax). Source: `tie_aware_map_support()`, used in `predict_response_pmfs()` /
`negative_log_likelihood()`.

---

## B5 — Percept push-forward + motor noise + lapse

Marginalise the MAP read-out over the internal measurement distribution, convolve
with von Mises motor noise (via FFT), and mix in the lapse floor:

    percept_t(est | θ_true) = Σ_m P(m | θ_true, k_sensory) · MAP(est | m)

    p_t(est) = (1 − lapse) · [ percept ⊛ V(· ; 0, k_motor) ]  +  lapse · (1/n_θ)

Source: `predict_response_pmfs()` (FFT convolution) and the lapse line.

---

## Fitted parameters (6)  — leaner than ours

    k_sensory(0.06), k_sensory(0.12), k_sensory(0.24)   sensory reliability, per coherence
    ρ                                                   geometric forgetting / volatility
    k_motor                                             motor-noise concentration
    lapse                                               lapse rate

(No α, no per-block prior-width parameter — 6 total vs our 7.) As in ours, the
prior concentration κ is NOT fitted: it is the learned latent, carried as the
belief H_t(κ) and updated every trial by B2.

---

## How this contrasts with hb_integration (M1–M5)

Same hierarchical idea — learn a belief over the prior concentration κ from
feedback, trial by trial — but three real definitional differences:

1. Combination rule (B3 vs M4). This model INTEGRATES the belief into one prior
   before the read-out (Σ_κ H(κ)·V, then one posterior, one MAP). Ours AVERAGES
   per-κ read-outs after the fact (Σ_κ b(κ)·R_κ). These are different
   distributions in general; both still produce Girshick bimodality, but the
   far-from-prior shape differs (this is the divergence the block-phase
   comparison figure shows, mean TV ≈ 0.16 in the far band).

2. Prior parameterisation. This model has NO mixture weight α: its prior is a
   pure von Mises whose width is entirely emergent from the learned belief (a
   uniform component is reachable only via the κ = 0 grid point). Ours keeps α
   as a fitted mixture weight on top of the learned κ. → 6 params vs 7.

3. Forgetting law. This model forgets GEOMETRICALLY (log H ← ρ·log H, shrink
   toward uniform). Ours forgets LINEARLY toward the initial belief
   (b ← (1−λ)b + λ·b_0).

Engineering (not model definition): 72-bin grid vs our 360-bin; full log-space
with logsumexp; tie-aware MAP vs plain argmax; Powell optimiser with a time
budget and Latin-hypercube multi-start — a deliberately multistart/GPU-friendly
rewrite.
