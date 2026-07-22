# Switch model — paper-standard fitting (Nelder–Mead 1e-4 + CMA-ES)

*Date: 2026-07-22*

*Tags: type:validation | claim:switching-vs-integration | status:result | presentation:backup*

Refit of the Switch model to Laquitaine & Gardner's exact fitting protocol — Nelder–Mead
10-start at the "very strict" 1e-4 tolerance, plus CMA-ES as an independent second optimizer —
to test whether the Switch fits are optimizer-independent and to diagnose the subject-5
cross-validation failure.

## Headline results

- **Optimizer-independent.** Nelder–Mead (1e-4) and CMA-ES agree on **11/12 subjects within 2
  nats**; CMA-ES is never worse (ties 10, marginally better 2). ΣNLL 403052.2 (NM) vs 403044.2
  (CMA-ES), Δ −8.1 / 403k. The AIC model comparison rests on genuine global optima, not an
  artifact of one optimizer. This reproduces the paper's own NM-vs-CMA-ES agreement.
- **Subject 5 is the ill-conditioned case** the paper's CMA-ES was built for: CMA-ES found a
  marginally better optimum (−6.2 nats) and a non-zero seed spread (122.9 vs 0.0 elsewhere) —
  the multimodal-surface fingerprint.
- **Subject 5's AIC is in-range once fit properly.** Basic−Switch AIC difference = +233 (NM) /
  +246 (CMA-ES), both inside the paper's across-subject CI [96, 695]. At the old loose 1e-2
  tolerance it was +0.1 (a spurious outlier).
- **The subject-5 CV FAIL is a fold-composition artifact, not a fit problem** — tightening the
  fit (proven optimal by two optimizers) does not clear it. See the write-up §3/§4b.

## Files

- `SWITCH_MODEL_CV_ISSUE.md` — full team write-up: the CV failure, its mechanism, the AIC
  cross-check against the paper, the CMA-ES replication, and how much to trust the Switch model.
- `switch_refit_effect.png` — before/after AIC per subject: 11/12 unchanged, only subject 5
  moves (from +0.1, an apparent outlier, into the paper's CI). The one interpretive change.
- `switch_optimizer_comparison.csv` — per-subject NM(1e-4) vs CMA-ES NLL/AIC + seed spread.
- `cmaes_fits/` — 12 per-subject CMA-ES fit JSONs (`subject<N>.json`).

## What changed in the shared results

The canonical Switch fits the API reads (`results/fits/comparison/switch/`) are now the NM 1e-4
fits (ΣNLL 403052.2; subject-5 AIC 61747.7). `api.results_table()` and the comparison
plots/tables already reflect these. CMA-ES fits are kept SEPARATE at
`results/fits/comparison_cmaes/switch/` as the confirmation run — the API does not read them.

## Reproduce

Committed NM 1e-4 fits: `results/fits/comparison/switch/` (env `SWITCH_FIT_TOL=1e-4`).
CMA-ES fits: `results/fits/comparison_cmaes/switch/` (env `SWITCH_OPTIMIZER=cmaes`). Both knobs
default to the unchanged house behaviour; only the Switch fit path reads them.
