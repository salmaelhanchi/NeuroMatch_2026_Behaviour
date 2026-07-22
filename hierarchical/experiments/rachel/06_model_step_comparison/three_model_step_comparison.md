# Three observer models — step-by-step comparison

A single-view comparison of the three observer models built for this project,
aligned stage by stage. Each model's own equation labels are given in full in
its individual reference:

| Model | File | Equation labels |
|---|---|---|
| **Switch** — paper's Switching observer (Laquitaine & Gardner 2018) | `switching_observer.py` | H1–H5 |
| **HB - Rachel** — our hb_integration observer (the abstract's model) | `hb_integration.py` | M1–M5 |
| **HB - Salma** — independent hierarchical observer | `hierarchical_confidence/model.py` | B1–B5 |

The three do not number 1:1 (the Switch keeps evidence and prior as two separate
read-outs; the two hierarchical models integrate them), so the tables below
align on **conceptual stages** rather than on each model's internal step number.

---

## Stage-by-stage comparison

### Stage 1 — Prior representation

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Form | Delta at 225° (prior read-out) | α·V(θ;225,κ) + (1−α)/360 | Pure V(θ;225,κ), per κ |
| Mixture floor (uniform) | via lapse only | **yes, weight (1−α)** | **no** (κ=0 grid point only) |
| Prior width set by | fitted `k_prior` per block | learned belief b_t(κ) | learned belief H_t(κ) |
| Label | H2 | M1 | B1 |

### Stage 2 — Sensory likelihood

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Form | V(·; θ_true, k_like(c)) | V(·; θ_true, k_like(c)) | V(·; θ_true, k_sensory(c)) |
| Per-coherence reliability | 3 `k_like` | 3 `k_like` | 3 `k_sensory` |
| Label | H1 | M2 | B2 (grid) |

### Stage 3 — How prior and evidence are combined  (the defining axis)

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Rule | **MIXTURE** — commit to prior OR evidence per trial | **INTEGRATE, then average read-outs** | **INTEGRATE into one prior, then read out** |
| Formula | P_e·evidence + P_prior·prior + P_rand·unif | Σ_κ b(κ)·R_κ  (avg of per-κ read-outs) | one posterior from Σ_κ H(κ)·V, then MAP |
| κ marginalised… | n/a (no belief) | **AFTER** read-out | **BEFORE** read-out |
| Bimodality is… | **imposed** by stochastic commit | **emergent** from uniform floor | **emergent** from integrated posterior |
| Label | H3+H4 | M4 | B3+B4 |

### Stage 4 — Reliance-on-prior weight

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| What sets it | **derived** ratio k_prior/(k_prior+k_e) (Eq. 6) | **fitted** α (fixed across blocks) | emergent from belief (no explicit weight) |
| Free parameter? | no (derived) | **yes** (α) | no |
| Label | H3 | M1 | — |

### Stage 5 — Learned latent (trial-by-trial dynamics)

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Learns across trials? | **no** — k_prior fixed per block | **yes** — belief b_t(κ) | **yes** — belief H_t(κ) |
| Update rule | none | forget then Bayes-correct on feedback | forget then Bayes-correct on feedback |
| Forgetting law | — | **linear** toward b_0: (1−λ)b + λ·b_0 | **geometric**: log H ← ρ·log H |
| Reset at block edge? | n/a | no (carried) | no (carried) |
| Label | — | M5 | B2 |

### Stage 6 — Read-out type

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Read-out | MAP (Girshick push-forward) | MAP per κ (Girshick) | MAP, **tie-aware** |
| Tie handling | argmax | argmax | **ties share mass 1/#ties** |
| Label | H1/H2 | M3 | B4 |

### Stage 7 — Motor noise + lapse

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Motor | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) | ⊛ V(·;0,k_motor) via FFT |
| Lapse | renormalised into Eq. 6 weights | (percept + p_random·unif)/(1+p_random) | (1−lapse)·percept + lapse/n_θ |
| Label | H5 | M4 | B5 |

### Numerical grid & fitting

| Aspect | Switch | HB - Rachel | HB - Salma |
|---|---|---|---|
| Direction grid | 360 (1°) | 360 (1°) | **72 (5°)** |
| κ grid | n/a | ~15 points | 16 points (incl. κ=0) |
| Compute | linear space | linear space | **log space (logsumexp)** |
| Optimiser | (fitter-dependent) | (fitter-dependent) | Powell + Latin-hypercube multistart |

---

## Parameter counts

| Model | N | Parameters |
|---|---|---|
| **Switch** | 9 | 3 k_like + 4 k_prior + k_motor + p_random |
| **HB - Rachel** | 7 | 3 k_like + α + k_motor + p_random + λ |
| **HB - Salma** | 6 | 3 k_sensory + ρ + k_motor + lapse |

In all three, the sensory reliabilities (k_like / k_sensory) and motor+lapse are
shared in spirit. They differ in the prior machinery: the Switch spends 4
parameters on fixed per-block prior widths; HB - Rachel replaces those with 1 learned
belief + α + λ; HB - Salma replaces them with 1 learned belief + ρ and drops α.

---

## One-line summary of the differences

- **Switch vs the two hierarchical models:** the Switch *commits* to prior-or-
  evidence each trial (mixture, imposed bimodality) and its prior width is a
  *fixed* fitted constant per block. Both hierarchical models *integrate* and
  *learn* the prior width online.
- **HB - Rachel vs HB - Salma (the two hierarchical models):** same idea, three definitional
  differences — (1) HB - Rachel averages per-κ read-outs *after* the fact while
  HB - Salma integrates the belief into *one* prior *before* reading out; (2) HB - Rachel
  keeps a fitted mixture weight α while HB - Salma's prior is a pure von Mises
  (no α); (3) HB - Rachel forgets linearly, HB - Salma geometrically. Empirically these
  agree near the prior and diverge in the far-from-prior (bimodality) regime —
  mean total-variation ≈ 0.16 in the far band, ≈ 0.06 near.
