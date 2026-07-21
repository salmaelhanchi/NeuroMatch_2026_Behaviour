"""
run_parallel.py — resumable parallel driver for the full comparison run
=======================================================================

Runs the per-subject fit and cross-validation stages across a pool of worker
processes (subjects in parallel), then the fast shared stages (shape, recovery,
figure, table) once. Everything is resumable: each stage skips subject/model
cells whose output JSON already exists, so an interrupted run is restarted by
simply re-invoking this script.

Model cost is very uneven — the integrate-BEFORE model (Recombined) re-does a
MAP read-out every trial (~8s/eval) because its combined prior changes as the
belief learns, whereas the integrate-after models precompute per-kappa readouts
(~1.1s/eval). So Recombined is fit (its bimodal SHAPE is the reason it is in the
analysis) but by default excluded from CV, whose cost it would dominate and
whose metric (overfitting check) is already characterised by its 7-param twin
HB-Rachel.

Usage:
  python -m observers.comparison.run_parallel                 # default set, 6 workers
  python -m observers.comparison.run_parallel --workers 4
  python -m observers.comparison.run_parallel --fit-models switch hb_adaptive hb_salma recombined \\
         --cv-models switch hb_adaptive hb_salma
"""

from __future__ import annotations

import argparse, subprocess, sys, time, json, platform
from datetime import datetime, timezone
from pathlib import Path

ALL_SUBJECTS = list(range(1, 13))
MOD = "observers.comparison"


def _git_sha():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, cwd=Path(__file__).parent)
        sha = r.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip()
        return {"commit": sha or None, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def write_manifest(args, subjects, out_dir):
    """Provenance manifest for the run — the one place a Methods section and a
    reproducer can read WHAT was run, WITH WHICH code and library versions, on
    WHICH config. Written at launch (before any fit) so it exists even if the
    run is interrupted. Grid definitions are read from the live model modules,
    not hardcoded, so they cannot drift from what actually ran."""
    import numpy, scipy
    grid = {}
    try:
        from observers.helpers.belief_grid import make_k_grid
        from observers.models.hb_adaptive_confidence import make_alpha_grid
        grid["k_grid"] = [float(x) for x in make_k_grid(n=15)]
        grid["alpha_grid"] = [float(x) for x in make_alpha_grid(9)]
    except Exception as e:
        grid["error"] = str(e)
    man = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__},
        "config": {
            "fit_models": args.fit_models, "cv_models": args.cv_models,
            "subjects": subjects, "maxiter": args.maxiter, "folds": args.folds,
            "workers": args.workers, "force": args.force,
            "rec_nsim": args.rec_nsim, "rec_maxiter": args.rec_maxiter,
            "example_subject": args.example_subject,
        },
        "grid": grid,
        "prior_mean_deg": 225.0,
        "notes": "Recombined is fit-only (excluded from CV/model-recovery): its "
                 "integrate-before read-out is ~8s/eval, and its overfitting "
                 "behaviour is characterised by its 7-param twin HB-Rachel.",
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(man, indent=2))
    print(f"[run_parallel] manifest -> {path}", flush=True)
    return man


def _cmd(stage, subject, models, maxiter, folds=None, force=False):
    cmd = [sys.executable, "-u", "-m", f"{MOD}.{stage}",
           "--subjects", str(subject), "--models", *models, "--maxiter", str(maxiter)]
    if folds is not None:
        cmd += ["--folds", str(folds)]
    if force and stage in ("fit_batch", "cross_validate"):
        cmd += ["--force"]
    return cmd


def _build_jobs(subjects, fit_models, cv_models, maxiter, folds, force=False):
    """One job = (label, command). fit runs before CV for a subject, but across
    subjects everything is independent, so we schedule all fit jobs then all CV
    jobs (each stage is itself resumable and idempotent)."""
    jobs = []
    for s in subjects:
        jobs.append((f"fit[s{s}]", _cmd("fit_batch", s, fit_models, maxiter, force=force)))
    for s in subjects:
        if cv_models:
            jobs.append((f"cv[s{s}]", _cmd("cross_validate", s, cv_models, maxiter, folds, force=force)))
    return jobs


def _run_pool(jobs, workers):
    """Manual subprocess pool (ProcessPoolExecutor is unavailable in the
    sandbox — SC_SEM_NSEMS_MAX is not queryable). Keeps <=workers procs live,
    logs each as it finishes, and never blocks the whole pool on one slow job.

    Each job's stdout/stderr is redirected to a per-job LOG FILE, not an OS
    pipe. This is load-bearing: a PIPE has a ~64KB buffer, and a child that
    prints more than that blocks on write until someone reads — but we only
    read after poll() reports done, so a chatty child would deadlock the whole
    pool (child waits for a reader, parent waits for the child to exit). Files
    have no such backpressure, so children run to completion regardless of how
    much they print, and the tails are available for debugging."""
    import collections
    from pathlib import Path
    from observers.helpers.paths import FITS_DIR
    job_log_dir = Path(FITS_DIR).parent / "logs" / "jobs"
    job_log_dir.mkdir(parents=True, exist_ok=True)

    pending = collections.deque(list(enumerate(jobs)))
    running = {}   # Popen -> (label, t0, logpath, fh)
    logs = []
    # Pin each worker to a SINGLE BLAS/OMP thread. The belief loop is not
    # BLAS-bound, so multi-threaded BLAS gives no speedup here; left unpinned,
    # each of the `workers` subprocesses spawns up to n_core BLAS threads
    # (workers x n_core threads on n_core cores -> load ~= workers*n_core and a
    # time-slicing collapse). One thread per worker keeps load == workers.
    import os
    worker_env = dict(os.environ,
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1")
    while pending or running:
        while pending and len(running) < workers:
            idx, (label, cmd) = pending.popleft()
            safe = label.replace(" ", "_").replace("/", "_")
            logpath = job_log_dir / f"{idx:03d}_{safe}.log"
            fh = open(logpath, "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 text=True, env=worker_env)
            running[p] = (label, time.time(), logpath, fh)
        done = [p for p in running if p.poll() is not None]
        if not done:
            time.sleep(2.0); continue
        for p in done:
            label, t0, logpath, fh = running.pop(p)
            fh.close()
            dt = time.time() - t0
            if p.returncode != 0:
                tail = ""
                try:
                    tail = logpath.read_text().strip()[-300:]
                except Exception:
                    pass
                line = f"FAIL {label} ({dt:.0f}s): {tail}"
            else:
                line = f"ok   {label} ({dt:.0f}s)"
            print("  " + line, flush=True)
            logs.append(line)
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--fit-models", nargs="+",
                    default=["basic_bayes", "switch", "hb_adaptive", "hb_salma", "recombined"])
    ap.add_argument("--cv-models", nargs="*",
                    default=["basic_bayes", "switch", "hb_adaptive", "hb_salma"],
                    help="models to cross-validate; pass with no values "
                         "(--cv-models) to skip CV entirely (e.g. fit-only recombined)")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--example-subject", type=int, default=9)
    ap.add_argument("--rec-nsim", type=int, default=2)
    ap.add_argument("--rec-maxiter", type=int, default=300)
    ap.add_argument("--force", action="store_true",
                    help="refit even if an output JSON already exists (clears stubs)")
    ap.add_argument("--no-shared", action="store_true",
                    help="run ONLY the per-subject fit+CV stages; skip the shared "
                         "shape/recovery/figure/table. Use when launching one model "
                         "per process so each finishes independently; run the shared "
                         "stages separately (make_figure/make_table/etc.) on whatever "
                         "models have completed so far.")
    a = ap.parse_args()
    subjects = a.subjects or ALL_SUBJECTS

    print(f"[run_parallel] fit={a.fit_models} cv={a.cv_models} "
          f"subjects={subjects} workers={a.workers} maxiter={a.maxiter}", flush=True)
    t0 = time.time()

    # ---- provenance manifest (written before any fit; survives interruption) ----
    from observers.helpers.paths import FITS_DIR
    write_manifest(a, subjects, FITS_DIR)

    # ---- per-subject fit + CV, parallel across subjects ----
    jobs = _build_jobs(subjects, a.fit_models, a.cv_models, a.maxiter, a.folds, force=a.force)
    _run_pool(jobs, a.workers)

    if a.no_shared:
        print(f"[run_parallel] fit+CV done ({(time.time()-t0)/60:.1f} min). "
              f"--no-shared: skipping shared stages (run them separately).",
              flush=True)
        return

    print(f"[run_parallel] fit+CV done ({(time.time()-t0)/60:.1f} min). "
          f"Running shared stages...", flush=True)

    # ---- shared stages (fast; run once, in-process) ----
    from observers.comparison import shape_analysis, recovery, make_figure, make_table
    shape_analysis.run(models=a.fit_models, subjects=subjects)
    # recovery on the affordable models only (Recombined excluded — 8s/eval makes
    # its model-recovery fan-out prohibitive; it is characterised by HB-Rachel's twin)
    recovery.run(models=a.cv_models, n_sim=a.rec_nsim, maxiter=a.rec_maxiter)
    make_figure.run(models=a.fit_models, subjects=subjects,
                    example_subject=a.example_subject)
    make_table.run(models=a.fit_models, subjects=subjects)

    print(f"[run_parallel] ALL DONE ({(time.time()-t0)/60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
