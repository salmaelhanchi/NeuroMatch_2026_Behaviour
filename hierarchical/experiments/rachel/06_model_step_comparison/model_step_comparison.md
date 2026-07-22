# Observer models — step-by-step comparison

A single-view comparison of the observer models built for this project, aligned
stage by stage. Each model's own equation labels are given in full in its
individual reference:

| Model | File | Equation labels |
|---|---|---|
| **Switch** — paper's Switching observer (Laquitaine & Gardner 2018) | `switching_observer.py` | H1–H5 |
| **Basic Bayesian** — paper's baseline integrator (Laquitaine & Gardner 2018) | `basic_bayesian.py` | paper Eq 1–7 |
| **HB - Rachel** — our hb_integration observer (the abstract's model) | `hb_integration.py` | M1–M5 |
| **HB - Salma** — independent hierarchical observer | `hierarchical_confidence/model.py` | B1–B5 |
| **Recombined** — Rachel × Salma cross | `hb_integrate_before.py` | (inherits M1–M5) |
| **HB - Adaptive** — the abstract's model taken literally (learns α *and* κ) | `hb_adaptive_confidence.py` | (inherits M1–M5) |
| **Rommana** — reliability-mixture observer (learned mixture weight) | `reliability_mixture_model.py` | R1–R5 |

The models do not number 1:1 (the Switch keeps evidence and prior as two
separate read-outs; Basic Bayesian multiplies them into one posterior; the HB
models integrate them *and* learn), so the tables below align on **conceptual
stages** rather than on each model's internal step number.

Two of the seven are the paper's own **non-learning** models — the **Switch**
(prior and evidence kept separate, committed to one or the other per trial) and
**Basic Bayesian** (prior and evidence multiplied into a single posterior). The
other four are our **learning** models: **HB - Rachel**, **HB - Salma**,
**Recombined**, and **HB - Adaptive** all learn a prior latent trial-by-trial.
Basic Bayesian is the key foil — "the Switch minus the switch": same nine
parameters, but it *integrates* instead of *selecting*, so its estimate is
**unimodal**. It is the model the Switch was shown to beat, and the starting
point our HB models add learning on top of.

**Recombined** is defined by one axis only: it is exactly HB - Rachel with the
*combination rule* swapped to HB - Salma's (integrate the belief into ONE prior
*before* reading out, rather than averaging per-κ read-outs *after*). Every
other stage is inherited byte-for-byte from HB - Rachel — same α floor, same
learned belief, same linear forgetting, same 360-bin grid, same 7 parameters.
Read it as "HB - Rachel's engine with HB - Salma's combination rule."

**HB - Adaptive** is also defined by one axis relative to HB - Rachel, but a
different one: the prior-confidence weight **α, which Rachel *fits* as a fixed
constant, is here *learned* trial-by-trial** alongside the width κ. The observer
carries a **joint belief `b_t(κ, α)`** over a 2-D grid of (width, confidence)
pairs and updates both latents from feedback with Rachel's own predict/correct
filter. Everything else — integrate-after combination, linear forgetting,
360-bin grid, motor+lapse, and the underlying `mixture_prior`/`_map_readout`
kernels — is inherited from HB - Rachel (it imports them directly). Because α
moves from the *parameter budget* into the *latent state*, it has **one fewer
free parameter than Rachel** (6 vs 7). Read it as "HB - Rachel with the
confidence weight promoted from a fitted constant to a learned latent." It is
the only **integrating** model in which *how much to trust the prior* is itself
learned — the abstract's literal requirement. (Rommana also learns a
trust-the-prior weight, but within a *mixture* architecture, not graded
integration — see its framing below.)

**Rommana** sits apart from the four HB models on the *defining* axis: like the
**Switch**, its percept is a genuine **discrete either/or mixture** — a
prior-centered component *or* a likelihood-centered component, never a
multiplicative blend — so it does not integrate. But unlike the Switch, whose
mixture weight is *derived* from the reliability ratio and fixed per block,
Rommana **learns its mixture weight `prior_reliance` trial-by-trial** with a
delta rule: after each trial the weight moves toward how well recent (smoothed)
feedback agrees with the 225° prior mean, at learning rate α. So it is the one
model that keeps the Switch's *architecture* (select, don't integrate) while
adding the HB models' *learning* (the reliance on the prior is acquired from
feedback, not set by hand). Read it as "the Switch, but the coin's bias is
learned." What it learns is the *weight* (how much to trust the prior), whereas
the HB models learn the prior's *width* κ; it does not marginalise over a κ
belief and has no MAP read-out — the mixture density itself is the response
distribution. It is committed on `main` (Rommana's `reliability_mixture_hb_model/`).

---

## Stage-by-stage comparison

### Stage 1 — Prior representation

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Form | Delta at 225° | Pure V(θ;225,k_prior) | α·V(θ;225,κ) + (1−α)/360 | Pure V(θ;225,κ), per κ | α·V(θ;225,κ) + (1−α)/360 | α·V(θ;225,κ) + (1−α)/360, per (κ,α) pair | V(θ;225,k_prior) — the prior-centered mixture component |
| Mixture floor (uniform) | via lapse only | via lapse only | **yes, weight (1−α)** | **no** (κ=0 grid point) | **yes, weight (1−α)** | **yes, (1−α) — and α is learned** | via lapse only |
| Prior width set by | fitted `k_prior`/block | fitted `k_prior`/block | learned belief b_t(κ) | learned belief H_t(κ) | learned belief b_t(κ) | **joint learned belief b_t(κ,α)** | fitted `k_prior`/block (width fixed; *weight* is learned) |
| Label | H2 | Eq 2 | M1 | B1 | M1 (inherited) | M1 (inherited) | R1 |

### Stage 2 — Sensory / likelihood component

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Form | V(·;θ_true,k_like(c)) | V(·;θ_true,k_like(c)) | V(·;θ_true,k_like(c)) | V(·;θ_true,k_sensory(c)) | V(·;θ_true,k_like(c)) | V(·;θ_true,k_like(c)) | V(·;θ_true,k_llh(c)) — likelihood-centered mixture component |
| Measurement noise | marginalised (Girshick) | marginalised (Girshick) | marginalised (Girshick) | marginalised (Girshick) | marginalised (Girshick) | marginalised (Girshick) | **not marginalised** — VM centered on true direction |
| Per-coherence reliability | 3 `k_like` | 3 `k_like` | 3 `k_like` | 3 `k_sensory` | 3 `k_like` | 3 `k_like` | 3 `k_llh` |
| Label | H1 | Eq 1 | M2 | B2 (grid) | M2 (inherited) | M2 (inherited) | R2 |

### Stage 3 — How prior and evidence are combined  (the defining axis)

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Rule | **MIXTURE** — commit per trial | **INTEGRATE** — one posterior | **INTEGRATE, avg read-outs** | **INTEGRATE into one prior** | **INTEGRATE into one prior** | **INTEGRATE, avg read-outs** | **MIXTURE** — weighted either/or |
| Formula | P_e·evid + P_pr·prior + P_r·unif | MAP[V(θ_e,k_e)·V(225,k_prior)] | Σ_κ b(κ)·R_κ (avg after) | MAP(Σ_κ H(κ)·V) (one before) | MAP(Σ_κ b(κ)·V) (one before) | Σ_(κ,α) b(κ,α)·R_(κ,α) (avg after) | r·prior + (1−r)·llh, r = learned reliance |
| κ marginalised… | n/a | n/a | **AFTER** read-out | **BEFORE** read-out | **BEFORE** read-out | **AFTER** read-out (over κ *and* α) | n/a (no κ belief) |
| Bimodality is… | **imposed** by commit | **none — unimodal** | **emergent** (floor) | **emergent** (posterior) | **emergent** (floor) | **emergent** (floor) | **imposed** by mixture |
| Label | H3+H4 | Eq 3–5 | M4 | B3+B4 | **B3-style + M's α floor** | M4 (inherited) | R3 |

### Stage 4 — Reliance-on-prior weight

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| What sets it | **derived** k_prior/(k_prior+k_e) | **emergent** (precision-weight) | **fitted** α (fixed) | emergent from belief | **fitted** α (fixed) | **learned** α (latent, tracked) | **learned** reliance r (latent, tracked) |
| Free parameter? | no | no | **yes** (α) | no | **yes** (α) | **no — α is a learned latent** | **no — r is a learned latent** (α is its learning *rate*) |
| Label | H3 | Eq 3–5 | M1 | — | M1 (inherited) | M1 (inherited) | R4 |

### Stage 5 — Learned latent (trial-by-trial dynamics)  ← where the HB models sit

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Learns across trials? | **no** (fixed/block) | **no** (fixed/block) | **yes** — b_t(κ) | **yes** — H_t(κ) | **yes** — b_t(κ) | **yes** — b_t(κ,α) | **yes** — reliance r_t (scalar) |
| WHAT is learned | — | — | prior concentration κ | prior concentration κ | prior concentration κ | **prior concentration κ AND confidence α** | **prior-reliance weight r** (mixture weight) |
| Update rule | none | none | forget → Bayes-correct | forget → Bayes-correct | forget → Bayes-correct | forget → Bayes-correct (joint) | **delta rule** r ← r + α·(agreement − r) |
| Forgetting form | — | — | **linear** (1−λ)b+λ·b0 | **geometric** logH←ρ·logH | **linear** (1−λ)b+λ·b0 | **linear** (1−λ)b+λ·b0 | EMA over feedback window (no explicit b0) |
| Reset | n/a | n/a | no (carried) | no (carried) | no (carried) | no (carried) | **yes — at session boundaries** |
| Label | — | — | M5 | B2 | M5 (inherited) | M5 (inherited, over κ×α) | R5 |

### Stage 6 — Read-out type

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Read-out | MAP (Girshick) | MAP on joint posterior (once) | MAP per κ (Girshick) | MAP, **tie-aware** | MAP on effective prior (once) | MAP per (κ,α) pair (Girshick) | **none** — mixture density *is* the response pmf |
| Tie handling | argmax | argmax | argmax | **ties share 1/#ties** | argmax | argmax | n/a (no argmax) |
| Label | H1/H2 | Eq 3–5 | M3 | B4 | M3 (inherited) | M3 (inherited) | R3 |

### Stage 7 — Motor noise + lapse

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Motor | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) FFT | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) |
| Lapse | into Eq. 6 weights | (p+p_r·unif)/(1+p_r) | (p+p_r·unif)/(1+p_r) | (1−lapse)·p + lapse/n_θ | (p+p_r·unif)/(1+p_r) | (p+p_r·unif)/(1+p_r) | (1−lapse)·p + lapse/360 |
| Label | H5 | Eq 7 | M4 | B5 | M4 (inherited) | M4 (inherited) | R5 |

### Numerical grid & fitting

| Aspect | Switch | Basic Bayesian | HB - Rachel | HB - Salma | Recombined | HB - Adaptive | Rommana |
|---|---|---|---|---|---|---|---|
| Direction grid | 360 (1°) | 360 (1°) | 360 (1°) | **72 (5°)** | 360 (1°) | 360 (1°) | 360 (1°) |
| κ grid | n/a | n/a | ~15 points | 16 points (incl. κ=0) | ~15 points | **~15 κ × 9 α = 135 pairs** | n/a (no κ belief; scalar reliance) |
| Compute | linear | linear | linear | **log (logsumexp)** | linear | linear | linear |
| Optimiser | (fitter-dependent) | (fitter-dependent) | (fitter-dependent) | Powell + LH multistart | (fitter-dependent) | (fitter-dependent) | (fitter-dependent) |

---

## Parameter counts

| Model | N | Parameters |
|---|---|---|
| **Switch** | 9 | 3 k_like + 4 k_prior + k_motor + p_random |
| **Basic Bayesian** | 9 | 3 k_like + 4 k_prior + k_motor + p_random |
| **HB - Rachel** | 7 | 3 k_like + α + k_motor + p_random + λ |
| **HB - Salma** | 6 | 3 k_sensory + ρ + k_motor + lapse |
| **Recombined** | 7 | 3 k_like + α + k_motor + p_random + λ |
| **HB - Adaptive** | 6 | 3 k_like + k_motor + p_random + λ  (α *and* κ are learned latents, not fitted) |
| **Rommana** | 10 | 3 k_llh + 4 k_prior + α (reliance learning rate) + k_motor + lapse  (reliance r is a learned latent) |

**Switch and Basic Bayesian share the identical nine-parameter set** — the only
difference between them is Stage 3 (select vs multiply), exactly the paper's
comparison. **Recombined and HB - Rachel share the identical seven-parameter
set** — the only difference between *them* is also Stage 3 (average per-κ
read-outs after vs integrate into one prior before). The combination rule is
thus the axis that separates *both* matched-parameter pairs. **HB - Adaptive and
HB - Rachel share the identical combination rule and engine** — the only
difference between *them* is Stage 4/5: Rachel *fits* α as a fixed constant,
while Adaptive *learns* it as a latent, dropping it from the parameter budget (7
→ 6). **α appears explicitly in HB - Rachel and Recombined** as the fixed fitted
prior mixture weight; in **HB - Adaptive** α exists but is learned, not fitted;
the remaining two models have no α.

---

## One-line summary of the differences

- **The two paper baselines (Switch, Basic Bayesian) both fix the prior width**
  per block — neither learns trial-by-trial. They differ only in how they
  combine prior and evidence: the **Switch selects** (commits per trial → the
  **bimodal** estimates subjects showed), while **Basic Bayesian multiplies**
  (one posterior → a single **unimodal** bump). Basic Bayesian is the model the
  Switch was shown to beat.
- **The four HB models all add learning** on top of integration: they learn a
  prior latent trial-by-trial from feedback instead of fixing it per block. They
  differ along two axes:
  - **Combination rule (Stage 3):** HB - Rachel and **HB - Adaptive** average
    per-κ read-outs *after*; HB - Salma and **Recombined** integrate the belief
    into *one* prior *before* reading out.
  - **What is learned (Stage 5):** HB - Rachel, HB - Salma, and Recombined learn
    only the prior *width* κ (α is fixed or absent). **HB - Adaptive** learns the
    prior *confidence* α **as well as** κ, via a joint belief b_t(κ,α) — the only
    model in which *how much to trust the prior* is itself a learned latent.
  - **Everything else (α floor, forgetting form, grid):** HB - Rachel,
    **Recombined**, and **HB - Adaptive** share the α-mixture floor, linear
    forgetting, and 360-bin grid; HB - Salma differs (no α, geometric forgetting,
    72-bin log-space).
  - So two "one-axis-from-Rachel" crosses sit either side of her:
    **Recombined = HB - Rachel's engine + HB - Salma's combination rule**
    (isolates integrate-before vs integrate-after), and
    **HB - Adaptive = HB - Rachel with α promoted from a fitted constant to a
    learned latent** (isolates fitting-confidence vs learning-confidence).
- **Rommana is the bridge between the two families.** It keeps the **Switch's
  architecture** — a discrete either/or mixture that commits to prior *or*
  evidence per trial (so its bimodality is *imposed*, like the Switch's, not
  *emergent* like the HB models') — but adds the **HB models' learning**: the
  mixture weight (reliance on the prior) is acquired trial-by-trial from how well
  feedback agrees with the prior mean, via a delta rule, and reset each session.
  It is the only model that learns the *weight* (how much to trust the prior)
  rather than the prior's *width* κ. Read it as "the Switch with a learned coin
  bias." It carries the most free parameters (10) because it keeps the Switch's
  full 3+4 concentration set *and* adds a reliance learning rate.
- **The through-line:** Basic Bayesian (integrate, no learning, unimodal) is the
  baseline; the Switch (select, no learning) explains the bimodality the
  baseline misses; the HB models (integrate *with* learning) recover that
  bimodality *without* a switch rule — as an emergent property of a learned,
  mixture-floored prior. HB - Rachel/Salma/Recombined explain how prior *width*
  is acquired; **HB - Adaptive** goes one step further and learns the prior
  *confidence* too, matching the abstract's literal claim that the observer
  learns how much to trust its prior.

*(Empirical note: HB - Rachel and Recombined agree near the prior and diverge in
the far-from-prior bimodality regime — the integrate-before rule leaves faint
discretisation ripple that the average-after rule washes out; mean
total-variation between the two ≈ 0.16 far, ≈ 0.06 near. HB - Adaptive shares
Rachel's integrate-after read-out, so it inherits her far-band shape; its
distinctive behaviour is dynamic — E[α] rises toward 1 when feedback clusters
near 225° and falls toward 0 when feedback is spread, verified in a 9/9 test
battery: bit-exact reduction to a single mixed-prior read-out when the belief is
pinned, correct confidence-learning direction, and clean recovery of its six
fitted parameters.)*
