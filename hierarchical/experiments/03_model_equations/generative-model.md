# Hierarchical Online Switching Observer — generative model spec (Phase 0)

This document fixes the modelling commitments *before* any code, so that every
later phase has a written oracle to check against. It extends the static
Switching observer of Laquitaine & Gardner (2018) by making the prior itself a
**latent quantity the observer learns online**, trial by trial, from feedback.

The static model answers "given the prior, what do you report?". This model
answers "how do you come to know the prior while using it?"

---

## 1. What the observer already had (static switching, recap)

On a trial with displayed direction `θ_true` and coherence `c`:

- draws a sensory measurement `θ_e ~ V(θ_true, k_e(c))`,
- commits to **either** the sensory evidence **or** the prior mean (the switch),
  with switch probability set by relative reliability (paper Eq. 6):

  `p_prior = k_prior / (k_prior + k_e)`,  `p_e = 1 − p_prior`,

- lapses to a uniform guess with probability `p_random`,
- adds von Mises motor noise `V(·;0,k_motor)` (paper Eq. 7).

Here `k_prior` was a **fixed fitted parameter** (one per prior block width).

---

## 2. The extension: the prior strength is latent and learned

We replace the fixed `k_prior` with a **belief distribution** `b_t(k)` over the
prior concentration `k`, carried across trials. This is the added level of the
hierarchy:

```
   Level 3 (hyper):     initial belief b_0(k)  +  volatility λ
                                 │  (governs how belief may change)
   Level 2 (prior):     belief b_t(k) over prior strength k  ← learned online
                                 │
   Level 1 (trial):     θ_true → θ_e ~ V(θ_true,k_e) ; latent switch s_t ; estimate θ̂
```

### Modelling forks — decided here

1. **Learning signal = feedback.** The task reveals the true direction at the
   end of each trial. The observer treats each revealed direction as one sample
   from the prior and updates its belief about the prior's strength. (The
   alternative — learning from the noisy measurement — is left as future work.)

2. **Cause-level readout = selection.** The switch commits to evidence *or*
   prior on each trial (probability matching), never averaging — this is what
   produces the bimodal estimate distributions. (Marginalising the switch would
   collapse back to unimodal integration.)

3. **Switch weight = reliability ratio (Eq. 6), evaluated under the current
   belief.** We keep the paper's winning competition rule and make it depend on
   what the observer *currently believes* the prior strength to be. The fully
   "derived responsibility" variant is a separate refinement.

4. **Belief readout for the estimate = marginalise over `b_t`.** The observer is
   uncertain about `k`, so its estimate distribution integrates the switch over
   its belief. This is exact and cheap (see §5).

---

## 3. Equations

Notation: directions live on the 1..360° grid; `μ = 225°` is the (fixed) prior
mean; `V(θ; μ, k)` is the von Mises pmf (paper Eq. 1).

### (H1) Belief update — predict/correct filter per trial

Before seeing trial `t`'s feedback, **forget** slightly toward the initial
belief (volatility, lets the observer re-adapt when the block changes):

    b_t^-(k) = (1 − λ) · b_{t−1}(k) + λ · b_0(k)                      (H1a)

After feedback direction `f_t` (= the revealed true direction), **correct** by
Bayes' rule, treating `f_t` as a draw from `V(·; μ, k)`:

    b_t(k) ∝ b_t^-(k) · V(f_t; μ, k)                                 (H1b)

renormalised so `Σ_k b_t(k) = 1`. With `λ = 0` this is exact sequential Bayesian
accumulation; (H1a)+(H1b) together must equal the batch posterior
`b_0(k)·Π_{s≤t} V(f_s; μ, k)` (this is the Phase-4 cross-check).

### (H2) Expected switch weight (Eq. 6 under the belief)

    w_prior(t) = E_{b_t}[ k / (k + k_e) ] = Σ_k b_t(k) · k/(k + k_e)  (H2)
    w_e(t)     = 1 − w_prior(t)

As the belief sharpens onto the true `k`, `w_prior(t)` approaches the static
value `k_true/(k_true + k_e)`. Early-block uncertainty ⇒ different weight ⇒ the
online learning signature.

### (H3) Lapse renormalisation (matches the static code exactly)

Because `w_prior + w_e = 1`, the paper's renormalisation collapses to a linear
form, so marginalising over the belief is exact:

    P_prior(t) = w_prior(t) / (1 + p_random)
    P_e(t)     = w_e(t)     / (1 + p_random)
    P_rand     = p_random   / (1 + p_random)     (constant)          (H3)

### (H4) Estimate distribution (Eq. 7 with belief-derived weights)

    percept_t = P_e(t)·p(θ_e | θ_true)  +  P_prior(t)·δ(θ − μ)  +  P_rand·(1/360)
    p(θ̂ | θ_true, b_t) = V(·;0,k_motor) ⊛ percept_t                 (H4)

- `p(θ_e | θ_true)`: sensory read-out = Girshick MAP look-up with the prior off
  (reused from the switching layer; a bump at `θ_true` of width set by `k_e`).
- `δ(θ − μ)`: prior read-out = look-up with the likelihood off (a spike at 225°;
  its position is independent of `k` — `k` only enters the *weight*).
- `⊛`: circular convolution.

### (H5) Generative response and likelihood

Generative (simulation): sample `θ̂_t ~ p(θ̂ | θ_true, b_t)`.

Likelihood (fitting): the belief path is **deterministic** given the feedback
sequence and parameters, so

    NLL(Θ) = − Σ_t log p(θ̂_t^{obs} | θ_true,t, b_t(Θ))              (H5)

with `Θ = { k_e(0.06/0.12/0.24), k_motor, p_random, λ }` and `b_0` = uniform over
the `k`-grid (no extra parameter in the base model). No particle filter is
needed because nothing on the belief path is sampled — a key tractability win.

---

## 4. Free parameters

| symbol | meaning | count |
|--------|---------|-------|
| `k_e(c)` | sensory strength per coherence | 3 |
| `k_motor` | motor precision | 1 |
| `p_random` | lapse rate | 1 |
| `λ` | volatility / forgetting (the learning knob) | 1 |
| **total** | | **6** |

Compare: static switching used 9 (three `k_e` + **four** `k_prior` + lapse +
motor). The online model *replaces* the four fitted prior strengths with a
single learning process — it has **fewer** parameters, and the prior strengths
become emergent, not fitted.

---

## 5. Why this is tractable (design notes for the implementation)

- The prior read-out is a delta at 225° regardless of `k`; `k` only scales the
  scalar switch weight. So the belief marginalisation (H4) reduces to a scalar
  expectation (H2) — no per-`k` distribution table.
- The evidence read-out depends only on `k_e`, so it is computed once per
  coherence (3 Girshick calls), then convolved with motor noise once.
- The belief update (H1b) is a multiply-and-renormalise against a precomputed
  table `V(f; 225, k)` over the `k`-grid.
- Consequently one full-experiment likelihood ≈ 3 Girshick calls + O(trials ×
  grid) scalar work — fast enough for Nelder-Mead / CMA-ES.

---

## 6. Verification plan (what each later phase checks against this spec)

- **Phase 4** — (H1) convergence to a known `k`; degenerate limits (`λ→0`,
  `k_e→∞`, sharp belief); (H1a+b) equals the batch posterior; distributions
  normalise.
- **Phase 6** — recover `Θ` from data simulated with known `Θ` (H5).
- **Phase 7** — data simulated from this model vs. static / no-learning models;
  fitting must prefer the true generator (AIC).
- **Phase 8** — fit to human CSV; compare learning-curve signatures and AIC to
  the static switching observer, with Phase-7 identifiability caveats.