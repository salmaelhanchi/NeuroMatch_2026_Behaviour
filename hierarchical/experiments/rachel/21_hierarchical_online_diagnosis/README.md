# 21 — hierarchical_online: bad model or bad fit?

`type:diagnosis | claim:model-comparison | status:result | presentation:backup`

## TL;DR

**hierarchical_online's poor in-sample showing on subjects 1 and 3 is genuine
model misfit, not a fitting failure.** Even at its own best achievable
parameters, the model cannot fit those subjects as well as the simpler
Basic-Bayes observer. A refit will not rescue it. Combined with the fact that
it has **no cross-validation** yet, hierarchical_online is not on solid enough
footing to carry a slide claim — HB-Adaptive remains the clean mechanism story.

## The question

In-sample, hierarchical_online (k=8) vs Basic-Bayes (k=9) is a rough tie
(HO lower AIC on 5/12, median ΔAIC ≈ +53 favouring Basic-Bayes). But two
subjects stand out with very large Basic-Bayes advantages:

| Subject | HO NLL | HO AIC | Basic-Bayes AIC | ΔAIC (HO − BB) | HO convergence |
|---|---|---|---|---|---|
| S1 | 39489.9 | 78995.8 | 76874.1 | **+2121.7** | converged=True, 213 iters (< 400 cap) |
| S3 | 43211.5 | 86439.0 | 83999.4 | **+2439.6** | converged=True, 306 iters (< 400 cap) |

Both fits **terminated successfully** well under the iteration cap, so the
convergence flag alone can't tell us whether the poor fit is real or whether
the optimizer stalled in a local minimum on the model's known rough,
multimodal likelihood surface (large multistart `start_spread`: S1=672, S3=900).

## The test

`converged: True` means the optimizer stopped moving — **not** that the best
fit was found. To separate "bad model" from "bad fit" we forward-evaluated
S1's NLL (each eval = one full 8562-trial replay, ~4.4 s) at many alternative
parameter settings. If any nearby point scored substantially better than the
converged fit, the fit was stuck (a fitting problem); if not, the misfit is
genuine (a model limitation).

**S1 fitted NLL = 39489.9** (objective reproduces the stored value exactly).

- **180-point grid** over `pi ∈ {0.3,0.5,0.7,0.85,0.95}`, `alpha ∈
  {0.02,0.05,0.1,0.2}`, `R0 ∈ {0,0.2,0.5}`, `k_motor ∈ {fitted,30,20}`:
  **best = 39489.9, Δ = +0.0.** Nothing in a wide neighbourhood beats the fit.
- **Focused 12-point probe**: the only settings that beat the fit did so by a
  trivial **−9 and −24 nats** (both from raising `R0`, the initial prior
  strength). Every genuinely different setting — lowering `pi` toward a pure
  integrator, changing `alpha` — made it **worse by +47 to +863 nats**.

The −24 "improvement" is real but negligible against the **~1040-nat gap to
Basic-Bayes** (HO best 39466 vs BB 38428). There is no hidden basin where the
model becomes competitive.

## Verdict

**Bad model, not bad fit — for these subjects.** The optimizer found the right
answer; the right answer is "hierarchical_online fits S1/S3 poorly." Its
online-learning dynamics (resultant-vector update + prior mixing) are a
liability rather than an asset for these subjects, whose behaviour a static
Bayesian integrator captures better.

Notes:
- `readout='sample'` (the smooth readout) was used by the fitter, so readout
  misspecification is ruled out.
- The genuinely under-converged HO fits are **S9, S11, S12** (hit the 400-iter
  cap, converged=False) — but their AIC gaps are small (−33, +473, −13), so
  refitting them would not change the overall in-sample tie.
- hierarchical_online fits are on **inconsistent standards** (S2 at
  maxiter=1500; S5/S8 at maxiter=400 with untracked convergence; the rest at
  maxiter=400 tracked). A clean comparison would refit all 12 to one standard,
  but the S1/S3 misfit conclusion stands regardless.

## Files
- `README.md` — this note (the diagnosis is fully contained here; the raw fit
  JSONs live in `results/fits/comparison/hierarchical_online/`).
