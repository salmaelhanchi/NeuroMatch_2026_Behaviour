"""
refit_model.py — one-touch, forced re-training of a single model, end to end.

Use this when a model's committed fits are stale (e.g. produced by an
under-converged code path) and must be regenerated through the CURRENT code. It
chains the whole workflow so nothing is forgotten:

    Stage 0  verify        api.verify_<model>()  must be green BEFORE spending compute
    Stage 1  refit + CV     forced (ignores resume-skip), multi-start point fit + K-fold CV
    Stage 2  comparison     regenerate shape / figure / table over the FULL comparison set
    Stage 3  validate       stamp results/fits/VALIDATION_REPORT.{md,json}

Why a wrapper instead of a hand-typed run_parallel line? Two footguns it removes:
  1. WITHOUT --force, run_parallel/run_all SKIP any fit whose file already exists at
     maxiter >= requested. Refitting stale files at the same maxiter would silently
     no-op. This script always forces the target refit.
  2. run_parallel's shared stages (figure/table) run over --fit-models ONLY, so
     `run_parallel --fit-models hb_rachel` would rebuild the comparison figure with
     hb_rachel ALONE. This script refits the target with --no-shared, then rebuilds
     the shared stages separately over the full --compare set, so the comparison is
     preserved.

Examples
--------
    # Regenerate hb_rachel and refresh the standard 5-model comparison + report:
    python -m observers.comparison.refit_model --model hb_rachel

    # Same, fit-only model (no CV), custom comparison set, more workers:
    python -m observers.comparison.refit_model --model hb_salma --no-cv \
        --compare switch basic_bayes hb_adaptive hb_rachel hb_salma --workers 4

    # Just the refit, skip the comparison rebuild and the report:
    python -m observers.comparison.refit_model --model hb_rachel --skip-compare --skip-validate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from observers.comparison.registry import ALL_SUBJECTS
from observers.comparison.validate_all import CORE5


def _run(argv: list[str], fatal: bool = True) -> int:
    """Run a pipeline module as a subprocess so each stage is isolated. With
    fatal=True (default) a non-zero exit aborts the chain; with fatal=False the
    exit code is returned for the caller to note but the chain continues (used
    for a validation record whose FAIL is informative, not a run failure)."""
    print(f"\n$ python -m {' '.join(argv)}", flush=True)
    r = subprocess.run([sys.executable, "-m", *argv])
    if r.returncode != 0 and fatal:
        raise SystemExit(f"[refit_model] stage failed ({argv[0]}), aborting.")
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="One-touch forced refit of a single model.")
    ap.add_argument("--model", required=True,
                    help="registry key to refit, e.g. hb_rachel")
    ap.add_argument("--subjects", nargs="+", type=int, default=None,
                    help="subjects to refit (default: all 12)")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cv", action="store_true",
                    help="fit-only model (e.g. hb_salma, recombined): skip CV")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="model set for the comparison figure/table/report "
                         "(default: the standard CORE5). MUST include --model.")
    ap.add_argument("--rec-nsim", type=int, default=2)
    ap.add_argument("--rec-maxiter", type=int, default=300)
    ap.add_argument("--ref", default="switch",
                    help="reference model for validate_all (default: switch)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the Stage-0 correctness gate (not recommended)")
    ap.add_argument("--skip-compare", action="store_true",
                    help="skip Stage-2 comparison rebuild (shape/figure/table)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip Stage-3 validation report")
    a = ap.parse_args()

    model = a.model
    subjects = a.subjects or ALL_SUBJECTS
    compare = a.compare or CORE5
    if model not in compare and not a.skip_compare:
        # Keep the comparison honest: the model we just refit must be in the set
        # whose figure/table we rebuild, or the refresh would drop it.
        compare = [*compare, model]
    t0 = time.time()

    print(f"=== refit_model: {model} | subjects={list(subjects)} | "
          f"maxiter={a.maxiter} folds={a.folds} | compare={compare} ===", flush=True)

    # ---- Stage 0: verify the code is correct before spending compute ----
    if not a.skip_verify:
        print(f"\n----- Stage 0: verify {model} -----", flush=True)
        from observers import api
        fn = getattr(api, f"verify_{model}", None)
        if fn is None:
            print(f"[refit_model] no verify_{model} in api; skipping Stage 0. "
                  f"(add one — see docs/ADDING_A_NEW_MODEL.md)", flush=True)
        else:
            passed, total = fn()
            if passed != total:
                raise SystemExit(f"[refit_model] verify_{model} FAILED "
                                 f"({passed}/{total}); fix the model before refitting.")
            print(f"[refit_model] verify_{model}: {passed}/{total} OK", flush=True)

    # ---- Stage 1: forced refit (+ CV), target model only, no shared stages ----
    print(f"\n----- Stage 1: forced refit{'' if a.no_cv else ' + CV'} -----", flush=True)
    subj_args = ["--subjects", *[str(s) for s in subjects]]
    rp = ["observers.comparison.run_parallel",
          "--fit-models", model,
          *(([] if a.no_cv else ["--cv-models", model])),
          *subj_args,
          "--workers", str(a.workers),
          "--maxiter", str(a.maxiter),
          "--folds", str(a.folds),
          "--force",          # <-- the load-bearing flag: overwrite the stale fits
          "--no-shared"]      # <-- don't let run_parallel rebuild the figure over one model
    _run(rp)

    # ---- Stage 2: rebuild the comparison-wide shared stages over the FULL set ----
    if not a.skip_compare:
        print(f"\n----- Stage 2: rebuild comparison (shape/figure/table) -----", flush=True)
        ra = ["observers.comparison.run_all",
              "--models", *compare,
              *subj_args,
              "--maxiter", str(a.maxiter),
              "--folds", str(a.folds),
              "--rec-nsim", str(a.rec_nsim),
              "--rec-maxiter", str(a.rec_maxiter),
              "--skip-fit", "--skip-cv"]   # Stage 1 already fit/CV'd the target;
        #  every OTHER model in `compare` is resume-skipped anyway (fits already exist
        #  at >= maxiter), so --skip-fit/--skip-cv just makes that explicit and fast.
        _run(ra)

    # ---- Stage 3: stamp the reproducible validation records ----
    if not a.skip_validate:
        # 3a: the comparison-wide record (all models vs ref) at the top of results/fits/
        print(f"\n----- Stage 3a: validate_all (comparison) -----", flush=True)
        va = ["observers.comparison.validate_all",
              "--models", *compare,
              "--ref", a.ref,
              "--folds", str(a.folds)]
        _run(va)
        # 3b: the PER-MODEL record, filed in this model's own folder
        # (results/fits/comparison/<model>/validation.{md,json}). Non-fatal: a
        # FAIL here (e.g. a known CV-fold issue) should not abort the run after
        # the fits themselves succeeded — the record captures it for review.
        print(f"\n----- Stage 3b: validate_model ({model}) -----", flush=True)
        vm = ["observers.comparison.validate_model",
              "--model", model, *subj_args, "--folds", str(a.folds)]
        if a.no_cv:
            vm.append("--no-cv")
        _run(vm, fatal=False)

    print(f"\n=== refit_model DONE: {model} ({(time.time()-t0)/60:.1f} min) ===", flush=True)
    print("Check: start_spread should now be NON-zero in "
          f"results/fits/comparison/{model}/subject*.json "
          "(judge convergence by NLL stability + start_spread, not the boolean flag).",
          flush=True)


if __name__ == "__main__":
    main()
