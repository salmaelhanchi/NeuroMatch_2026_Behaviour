# Model comparison: results guide

*How to read the outputs of the `observers.comparison` pipeline, and which test
each panel and table cell reports. Written for the team — no prior familiarity
with the pipeline internals assumed.*

The pipeline compares perceptual-observer models on the Laquitaine & Gardner
motion-direction data. Its headline question is whether **HB-Adaptive** — the
hierarchical Bayesian observer that learns *both* the prior width (κ) and the
prior confidence (α) online — reproduces human behaviour as well as or better
than the paper's **Switching observer**, which selects between prior and
evidence each trial. Every model in the comparison is registered in
`observers/comparison/registry.py`; adding a model is a one-entry change there,
and every stage below picks it up automatically.

All models are evaluated on the same 360°direction grid, so the negative
log-likelihoods (NLL) are **directly comparable** — there is no grid-resolution
artifact to correct for (unlike the earlier HB-Salma 72-bin comparison, where
raw NLLs were not comparable).

---

## The two output objects

1. **`results/figures/model_comparison_figure.png`** — the main multi-panel
   results figure (Panels A–E).
2. **`results/fits/model_comparison_table.md` / `.csv`** — the model-comparison
   table (models as rows).

Plus a **supplementary** figure,
`results/figures/model_recovery_confusion.png`, that licenses the comparison
(parameter recovery + model-recovery confusion matrix).

---

## What each panel tests

| Panel | Shows | Test tier | What a "pass" looks like |
|---|---|---|---|
| **A** | Model structures (schematic) | — | orientation only: Switch selects prior *or* evidence (k=9); HB-Adaptive learns κ+α online (k=6) |
| **B** | Observed vs predicted response distributions, for a bimodal cell (low coherence, wide prior, far stimulus) and a unimodal cell (high coherence, narrow prior) | Tier 2: distribution-shape reproduction | the model's density tracks the grey observed histogram — crucially, reproduces the bimodal split without a switch rule |
| **C** | Per-subject Δ (model − reference), sorted. Uses CV-NLL when available, else AIC | Tier 1: quantitative fit | bars below zero = model beats the Switch on that subject |
| **D** | Learned prior width E[SD] over trials vs the true block SD | Tier 2: latent recovery | the learned width steps down/up with the block structure — the abstract's "recovers block-specific prior widths" |
| **E** | Far-band prior-cluster mass, predicted vs observed | Tier 2: bimodality signature | points on the diagonal = the model reproduces how much mass subjects actually put near the prior when the stimulus is far from it |

## What each table column reports

| Column | Meaning |
|---|---|
| **k** | number of *fitted* free parameters (HB-Adaptive = 6; α and κ are learned latents, not fitted) |
| **ΣNLL** | summed negative log-likelihood across fitted subjects (lower = better raw fit) |
| **ΔAIC** | AIC relative to the reference model (Switch); negative = better after the complexity penalty |
| **ΔBIC** | BIC relative to reference; a stricter penalty than AIC — the two can disagree |
| **CV-NLL** | summed block-fold held-out per-trial NLL — the **overfitting-proof** metric; **lead with this** |
| **Wins (AIC)** | number of subjects on which the model has the best AIC |
| **Wins (CV)** | number of subjects on which the model has the best held-out CV-NLL |

**Reading order:** lead with **CV-NLL** and **Wins (CV)** — held-out data
penalises over-flexible models directly, so no complexity assumption is needed.
AIC/BIC corroborate on the in-sample fit. Then Panels B/D/E show the fit is for
the *right reasons* (shape, learned width, bimodality), not just a lower number.

---

## The supplementary figure (why the comparison is trustworthy)

- **Parameter recovery** (left): simulate from known parameters, refit, check
  they return. Points on the diagonal = identifiable parameters. For
  HB-Adaptive this also covers the κ–α ridge (both are learned latents that can
  trade off from a single feedback draw).
- **Model-recovery confusion matrix** (right): simulate from each model, fit
  *all* models, record which one AIC selects. The diagonal must dominate — if
  the metrics cannot recover the true generator on synthetic data, the
  comparison on real data would be meaningless.

---

## Rerunning

```
# full pipeline (long — HB-Adaptive fit is CPU-heavy, ~1.1 s/eval over its
# 135-pair κ×α grid; a converged all-12 run is multi-hour):
python -m observers.comparison.run_all

# scope to a subset:
python -m observers.comparison.run_all --subjects 1 2 --maxiter 400

# re-assemble figures/tables from existing fits without refitting (fast):
python -m observers.comparison.run_all --skip-fit --skip-cv --skip-recovery

# add a model (once its registry entry is enabled):
python -m observers.comparison.run_all --models hb_adaptive switch hb_rachel
```

Stages 1–2 (fit, CV) are **resumable**: one JSON per model×subject, existing
files are skipped unless `--force`, so an interrupted run picks up where it left
off. Stages 3–6 (shape, recovery, figure, table) are cheap assembly.

---

## Smoke-test findings (subject 1, under-converged budget)

A validation pass on subject 1 at a deliberately tiny optimiser budget (not
converged — for path-checking only) already shows the expected direction:

- **AIC:** HB-Adaptive 78 336 vs Switch 80 601 — HB-Adaptive better by ~2 265
  with 3 fewer parameters.
- **Held-out CV-NLL:** HB-Adaptive 39 770 vs Switch 40 294 (per-trial 4.645 vs
  4.706) — HB-Adaptive wins out-of-sample too.
- **Far-band prior-cluster mass:** observed 0.084, HB-Adaptive 0.079 (near
  match), Switch 0.268 (over-commits to the prior 3×).

These are directional only; the converged all-12-subject numbers replace them
when the full run completes.
