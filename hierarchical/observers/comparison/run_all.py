"""
run_all.py — orchestrator for the whole model-comparison pipeline
=================================================================

Reruns the entire comparison end-to-end, in order:

    1. fit_batch        fit every registered model x subject   (resumable)
    2. cross_validate   block-fold held-out NLL                (resumable)
    3. shape_analysis   distribution-shape + bimodality
    4. recovery         parameter + model recovery
    5. make_figure      multi-panel results figure (Panels A-E)
    6. make_table       comparison table + supplementary recovery figure

Every stage reads the registry, so the set of models is chosen ONCE here (or on
the command line) and flows through every stage. Stages 1-2 are resumable
(skip existing JSON), so an interrupted run resumes cheaply; stages 3-6 are
cheap assembly and always rerun.

Because HB-Adaptive's fit is CPU-heavy (~1.1 s per likelihood eval over its
135-pair kappa-alpha grid), a full converged all-12 run is a multi-hour job.
Use --skip-fit / --skip-cv to re-assemble figures/tables from existing fits
without refitting, and --subjects / --maxiter to scope a run.

Usage:
  # full pipeline, default 2 models, all 12 subjects (long):
  python -m observers.comparison.run_all

  # scope to a smoke subset:
  python -m observers.comparison.run_all --subjects 1 2 --maxiter 400

  # re-assemble outputs from existing fits (fast):
  python -m observers.comparison.run_all --skip-fit --skip-cv --skip-recovery

  # add a third model (once its registry entry is enabled):
  python -m observers.comparison.run_all --models hb_adaptive switch hb_rachel
"""

from __future__ import annotations

import argparse, time

from observers.comparison import (fit_batch, cross_validate, shape_analysis,
                                  recovery, make_figure, make_table)
from observers.comparison.registry import DEFAULT_MODELS, ALL_SUBJECTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--example-subject", type=int, default=9)
    ap.add_argument("--rec-nsim", type=int, default=2)
    ap.add_argument("--rec-maxiter", type=int, default=300)
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-shape", action="store_true")
    ap.add_argument("--skip-recovery", action="store_true")
    ap.add_argument("--skip-figure", action="store_true")
    ap.add_argument("--with-table", action="store_true",
                    help="regenerate the pooled model_comparison_table (retired by "
                         "default; per-model validation.{md,json} records supersede it)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    models = a.models or DEFAULT_MODELS
    subjects = a.subjects or ALL_SUBJECTS
    print(f"=== pipeline: models={models} subjects={subjects} ===\n", flush=True)

    # Provenance manifest (written at launch, before any fit, so it survives an
    # interruption). Same shared writer as run_parallel, so serial and parallel
    # runs leave identical provenance. run_all fits AND cross-validates the same
    # model set, so both stage lists are `models` (minus whichever stage is
    # skipped) — an honest record of what this invocation actually ran.
    from observers.helpers.paths import FITS_DIR
    from observers.comparison.manifest import write_manifest
    write_manifest(
        FITS_DIR,
        driver="run_all",
        fit_models=[] if a.skip_fit else list(models),
        cv_models=[] if a.skip_cv else list(models),
        subjects=list(subjects),
        maxiter=a.maxiter,
        folds=a.folds,
        extra_config={
            "force": a.force, "rec_nsim": a.rec_nsim, "rec_maxiter": a.rec_maxiter,
            "example_subject": a.example_subject, "with_table": a.with_table,
            "skip": {k: getattr(a, f"skip_{k}") for k in
                     ("fit", "cv", "shape", "recovery", "figure")},
        },
    )

    def stage(name, fn):
        t0 = time.time()
        print(f"\n----- {name} -----", flush=True)
        fn()
        print(f"----- {name} done ({time.time()-t0:.0f}s) -----", flush=True)

    if not a.skip_fit:
        stage("1. batch fit",
              lambda: fit_batch.run(models, subjects, maxiter=a.maxiter, force=a.force))
    if not a.skip_cv:
        stage("2. cross-validation",
              lambda: cross_validate.run(models, subjects, folds=a.folds,
                                         maxiter=a.maxiter, force=a.force))
    if not a.skip_shape:
        stage("3. shape analysis",
              lambda: shape_analysis.run(models, subjects))
    if not a.skip_recovery:
        stage("4. recovery",
              lambda: recovery.run(models, n_sim=a.rec_nsim, maxiter=a.rec_maxiter))
    if not a.skip_figure:
        stage("5. results figure",
              lambda: make_figure.run(models, subjects, example_subject=a.example_subject))
    # The pooled model_comparison_table is retired by default: it silently
    # excluded incomplete/broken models (missing = absent row, not flagged) and
    # printed metrics for models with no CV, so it read as "complete" when it
    # wasn't. The per-model results/fits/comparison/<model>/validation.{md,json}
    # records (validate_model) carry the honest per-model fit/CV/convergence
    # health instead. Pass --with-table to regenerate the pooled table anyway.
    if a.with_table:
        stage("6. table + supplementary",
              lambda: make_table.run(models, subjects))

    print("\n=== pipeline complete ===", flush=True)


if __name__ == "__main__":
    main()
