# Results storage layout

Every pipeline stage reads and writes through `observers/helpers/paths.py`, so the
layout is defined in exactly one place and a script finds it no matter where it is
launched from. This document is the map of what lands where, and the tracking /
provenance rules.

```
results/
├── fits/                                  # FITS_DIR — all fitted-parameter output
│   ├── comparison/                        # Step 1: point fits
│   │   └── <model>/
│   │       ├── subject<N>.json            #   one point fit per model × subject
│   │       ├── validation.md              #   PER-MODEL validation record (validate_model)
│   │       └── validation.json            #   same, machine-readable
│   ├── comparison_cv/                     # Step 2: cross-validation
│   │   └── <model>/subject<N>_cv.json     #   one file per model × subject
│   ├── comparison_shape/                  # Step 3: distribution-shape analysis
│   │   ├── shape_subject<N>.npz           #   per-subject arrays
│   │   └── shape_summary.json             #   pooled summary
│   ├── comparison_recovery/               # Step 4: parameter + model recovery
│   │   ├── parameter_recovery.json
│   │   └── model_recovery.json            #   confusion matrix
│   ├── manifests/                         # immutable per-run provenance (tracked)
│   │   └── <stage-slug>_maxiter<N>_<UTC>.json
│   ├── VALIDATION_REPORT.md               # validate_all record (human-readable)
│   └── VALIDATION_REPORT.json             # validate_all record (machine-readable)
├── figures/                               # FIGURES_DIR — generated .png (Step 5)
└── logs/                                  # stdout transcripts — NOT version-controlled
    └── <model>/, jobs/, monitor/, ...
```

## Naming conventions (fixed — scripts depend on them)

| output | path | written by |
|---|---|---|
| point fit | `comparison/<model>/subject<N>.json` | `fit_batch.py` (`OUT_DIR = FITS_DIR/"comparison"`) |
| cross-validation | `comparison_cv/<model>/subject<N>_cv.json` | `cross_validate.py` |
| shape (per subject) | `comparison_shape/shape_subject<N>.npz` | `shape_analysis.py` |
| shape (summary) | `comparison_shape/shape_summary.json` | `shape_analysis.py` |
| parameter recovery | `comparison_recovery/parameter_recovery.json` | `recovery.py` |
| model recovery | `comparison_recovery/model_recovery.json` | `recovery.py` |
| run manifest (archive) | `manifests/<stage-slug>_maxiter<N>_<UTC>.json` | `manifest.write_manifest` |
| validation report (comparison-wide) | `VALIDATION_REPORT.{md,json}` | `validate_all.py` |
| validation record (per-model) | `comparison/<model>/validation.{md,json}` | `validate_model.py` |

Two deliberate conventions:
- **Per-model subfolders, model-name prefix dropped from the filename** — the folder
  already identifies the model, so it is `comparison_cv/switch/subject3_cv.json`, not
  `comparison_cv/switch_subject3.json`.
- **CV and point fits share one parent** (`results/fits/`) so a Methods section
  points at a single tree.

**Two validation scopes, by design.** A model comparison produces two kinds of
validation record, and they live in different places because they answer different
questions:
- **Per-model** (`comparison/<model>/validation.{md,json}`, from `validate_model.py`)
  — is *this one model* sound on its own? Model-specific Step-0 verify, a per-subject
  fit table (nll/aic/bic/**start_spread**/convergence), and its own convergence /
  AIC-BIC / CV / shape checks. Filed in the model's folder like every other per-model
  artifact.
- **Comparison-wide** (`VALIDATION_REPORT.{md,json}`, from `validate_all.py`) — is the
  *whole comparison* (a set of models vs a reference) certifiable? Run-level, so it
  sits at the top of `results/fits/` next to the `manifests/` archive.
  Both reuse the SAME check functions (one definition of "valid"); the per-model
  record just scopes them to one model and adds the per-subject fit detail the
  comparison report pools away.

## Provenance: the run manifest

Written by the shared `observers/comparison/manifest.py`, called by **both**
drivers (`run_all.py` and `run_parallel.py`) so a serial run and a parallel run
leave *identical* provenance. It is written **at launch, before any fit**, so it
survives an interrupted run.

Each run writes **one** file — an **immutable archive** manifest at
`manifests/<stage-slug>_maxiter<N>_<UTC>.json`, never overwritten. The
`stage-slug` is stage-aware and honest: a fit-only run is `fit-<models>_...`,
a CV-only run `cv-<models>_...`, a combined run `fit-<...>_cv-<...>_...`. So a
CV-only pass is never archived under a misleading `fit_` name, and the run is
recognisable from the filename without opening it. There is no separate
latest-pointer file: the `<UTC>` timestamp in each filename sorts
chronologically, so "the most recent run" is the last-sorted archive entry.

Each manifest records: UTC timestamp, `driver` (`run_all`/`run_parallel`), git
commit + dirty flag, python + numpy + scipy versions, platform, the full config
(models per stage, subjects, maxiter, folds, plus driver-specific flags), the live
κ / α grid definitions (read from the model modules, so they cannot drift from what
ran), and `prior_mean_deg`.

**Handling rule:** never hand-edit or delete an archive manifest — it is the record
of what actually ran. If you need "which run produced these fits", read the archive
manifest whose config matches; the git SHA there pins the exact code.

## What is tracked vs local

- **Tracked:** everything under `results/fits/**` — the fit/CV/shape/recovery JSONs,
  the `manifests/` archive, and the `VALIDATION_REPORT.*`. These
  are the durable, reproducible record.
- **Not tracked:** `results/logs/` (git-ignored). These are stdout transcripts —
  progress chatter and false-start/killed runs — redundant with the real provenance
  (per-fit convergence/timing live in the fit JSONs; environment + git SHA live in
  the manifest). Kept locally for debugging only.

## Resume / overwrite semantics

- `fit_batch` and `cross_validate` are **resumable**: an existing output is skipped
  unless it was produced at a smaller `maxiter` than requested (or `--force` is
  passed). So re-running a batch only recomputes what is missing or under-budget.
- Archive manifests accumulate (one per run); the `VALIDATION_REPORT.*` is
  refreshed in place.
