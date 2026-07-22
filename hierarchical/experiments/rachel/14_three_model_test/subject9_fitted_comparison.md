# Subject 9 — fitted three-model comparison

The three models fit by maximum likelihood to subject 9's full trial sequence
(8632 trials, correct block order: session → run → trial). This is the
adjudicating step the earlier simulated tests could not provide.

## The grid-comparability trap (important)
Rachel and Recombined predict over a **360-bin** (1°) response grid; Salma over
a **72-bin** (5°) grid. Raw NLL/AIC across different grids are NOT comparable —
a coarser grid concentrates probability into fewer, larger bins and mechanically
lowers NLL. On the raw numbers Salma looked ~27,000 AIC "better," which was pure
grid artifact.

**Fix:** all three likelihoods recomputed on the common 72-bin grid (Rachel's and
Recombined's fitted predictions coarsened to Salma's bins, evaluated at the same
response bins). Everything below is on that fair footing.

## Fair result (72-bin grid)

| model | NLL | k | AIC | ΔAIC | fitted α | fitted forgetting | far-band prior cluster |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rachel** (integrate-after) | 28502 | 7 | **57017** | **0 (best)** | 0.445 | λ=0.517 | 20.9% |
| **Salma** (integrate-before, no α) | 28651 | 6 | 57314 | +297 | — | ρ=0.581 | 5.7% |
| **Recombined** (integrate-before + α) | 29182 | 7 | 58379 | +1362* | 0.436 | λ=0.519 | 20.4% |
| Observed | | | | | | | 13.8% |

*Recombined is UNDER-CONVERGED — see caveat below; its AIC is an upper bound.

## What the fit says

**1. Rachel wins on AIC — but narrowly over Salma (ΔAIC ≈ 297).**
The best-fitting model on subject 9 is Rachel (integrate-after + α floor + linear
forgetting). Salma is close behind and does it with ONE FEWER PARAMETER (no α),
so on a per-parameter basis Salma is remarkably efficient — 297 AIC is a real but
not overwhelming gap. The two represent genuinely different, both-viable accounts.

**2. The far-band behaviour splits exactly as predicted, and NEITHER model nails it.**
- Rachel (and Recombined) still OVERSHOOT the human prior cluster even after
  fitting: 21% predicted vs 14% observed. Fitting pulled α down (0.6 default →
  0.445) and shrank the overshoot (25% → 21%), but the α floor still imposes more
  prior-cluster mass than subject 9 shows.
- Salma UNDERSHOOTS: 5.7% vs 14%. With no α floor, once ρ is fit her prior
  cluster is too thin far from the prior.
- The human value (14%) sits BETWEEN the two model families. This is the single
  most interesting scientific result: **subject 9's far-band behaviour is
  intermediate** — more prior-clustering than a no-floor model, less than an
  α=0.44 floor produces. It hints the true floor is small but nonzero, or that
  the floor should itself be learned rather than fixed.

**3. Fitted forgetting is high (λ ≈ 0.52 for both α-floor models).**
Both α-floor models want a large forgetting rate — the belief leaks ~half-way to
the anchor each trial. This is consistent with the task's frequent block changes
(subject 9 has 40 blocks of ~215 trials): fast forgetting is needed to re-learn
each new block's width. It also confirms λ is doing real work (the fit does not
drive it to zero).

## Caveats (do not over-read this)
- **ONE subject.** Subject 9 was chosen for having the most far-band data; the
  ranking may differ on others. The paper found the switch beat basic Bayesian on
  8/12 — model rankings are subject-dependent here too.
- **Recombined is under-converged.** Its per-trial full 360×360 read-out makes it
  ~15× slower to fit than Rachel (which caches a read-out stack). It got only 60
  warm-started Nelder-Mead iterations (from Rachel's solution) vs full runs for
  the others; its NLL is an UPPER BOUND and its true converged AIC is somewhere
  below 58379 — possibly much closer to Rachel. Its +1362 gap is NOT a fair
  verdict, only a "not yet converged" flag. To compare it fairly it needs the
  read-out caching optimisation or a longer optimiser budget.
- **Salma's fit used her own Powell optimiser** (max 200 evals, 2 rho starts);
  Rachel/Recombined used Nelder-Mead. Different optimisers, though both converged
  on Rachel/Salma.
- These are per-trial NLL fits; no cross-validation. AIC penalises parameter
  count but not the integrate-after model's implicit flexibility.

## Bottom line
On subject 9, at fair common-grid AIC, **Rachel (integrate-after) fits best,
with Salma a close and more parsimonious second; Recombined is not yet fairly
converged.** The scientifically interesting finding is behavioural, not the
ranking: the human far-band prior cluster (14%) lands BETWEEN the α-floor
models' overshoot (~21%) and Salma's no-floor undershoot (6%) — suggesting the
prior floor is real but smaller than a fixed α≈0.44, or should itself be learned.
