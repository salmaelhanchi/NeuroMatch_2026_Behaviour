# Build report — Anirban's four models, implemented

This closes the four gaps identified in the Anirban-vision comparison. All four
models Anirban specified but that had not been built are now implemented,
verified, and smoke-fit. They live in the `observers/` package (see
`docs/new_models_manifest.md` for exact paths).

## What was built vs. Anirban's spec

| Anirban's model | built as | status |
|---|---|---|
| **Inferred-responsibility causal-inference H1** (weight = per-measurement posterior p(z=peaked\|m), stimulus-dependent) | `CausalInferenceObserver` (7p) | built, 7/7 verify, smoke-fit S1 |
| **Logistic-covariate mixture** (Deck B; weight = fitted logistic of prior_std, coh, interaction, recent error, cumulative prior reliance) | `LogisticMixtureObserver` (11p) | built, 4/4 verify, smoke-fit S1 (flat-basin, needs multi-start) |
| **Bimodal-likelihood control** (Chetverikov & Jehee; two-lobe likelihood, no switch) | `BimodalLikelihoodObserver` (10p) | built, 6/6 verify, smoke-fit S1 |
| **Finite-sample readout** (n posterior samples; n->inf = Basic Bayes, n=1 ~ switch) | `FiniteSampleObserver` (10p) | built, 5/5 verify, smoke-fit S1 |

## Verification (independently re-run) — 22/22 checks pass

- **Causal-inference:** normalisation (1e-16); reduces to hb-mixture-MAP (Δ 3e-17)
  and to the fixed reliability ratio (Δ 2e-16); **the falsifiable signature** —
  inferred switch rate DECLINES with distance from 225 deg (prior-window mass
  0.525 -> 0.311 across displayed 225 -> 105), while the fixed-ratio analogue is
  FLAT at 0.73. This is the stimulus-dependence the switch family cannot produce.
- **Bimodal-likelihood:** g=1 reduces to Girshick single-prior Basic Bayes exactly
  (Δ < 5e-17, three parameter pairs); produces two lobes ~180 deg apart with NO
  switch (mass 0.42/0.43 at 135/315 for a 135 deg stimulus).
- **Finite-sample:** n=1 reproduces the static Switching observer estimate
  distribution exactly (Δ 2e-17); n=200 concentrates on the posterior mean
  (unimodal, SD 78 -> 14); SD decreases monotonically with n. A true nest:
  Basic-Bayes and switch are its two limits.
- **Logistic mixture:** flat weight 0.5 when all betas 0; represents a
  NON-MONOTONIC prior_std dependence (coh-averaged prior-reliance
  [0.203, 0.312, 0.370, 0.368], an interior turning point impossible for a
  concentration ratio — matches Deck B); reproduces hysteresis (b_hist>0 raises
  w_prior by +0.33 over a prior-reliant run).

## Smoke-fits (subject 1, single start, maxiter<=150 — viability, NOT comparison)

| model | NLL | AIC | note |
|---|---|---|---|
| finite-sample | 38412 | 76845 | fitted n≈3.4 — an interior resource-rational value between switch (1) and Bayes (inf) |
| bimodal-likelihood | 38749 | 77518 | fitted g≈0.99 — second lobe barely used (coherent-motion task is not ambiguous) |
| causal-inference | 39632 | 79278 | fitted pi≈0.5, kappa≈2.8 |
| logistic mixture | 40480 | 80982 | cold start settled at flat basin — needs multi-start; AIC not comparable yet |

These single-start AICs are **run-viability snapshots**, not a model comparison —
they prove each model fits real trial data end-to-end. A proper comparison needs
the fair multi-start all-12-subject batch (deferred: local compute is saturated).

## Scientifically notable early signals
- **finite-sample n≈3.4** is the most interesting: subject 1 is neither a pure
  switcher (n=1) nor a full integrator (n=inf) but uses ~3 samples — a concrete,
  fittable resource-rational estimate.
- **bimodal-likelihood g≈0.99** means the no-switch competitor does NOT lean on
  its streak lobe for this subject — an early hint the Chetverikov-Jehee account
  may not by itself explain the bimodality here (but needs the full batch).

## Not yet done (deferred, compute-bound)
- Fair multi-start fits of all four models across all 12 subjects.
- AIC/BIC/CV placement of the new models against the existing six.
- Logistic model needs multi-start (or covariate-informed initialisation) to
  escape the flat-weight basin before its fit is meaningful.
