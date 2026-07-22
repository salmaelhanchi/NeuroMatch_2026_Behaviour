"""
manifest.py — the one place a run's provenance is written.

Both drivers (`run_all.py`, `run_parallel.py`) call `write_manifest(...)` so a
serial run and a parallel run leave IDENTICAL provenance. The manifest records
WHAT was run, WITH WHICH code and library versions, on WHICH config — enough for
a Methods section and a reproducer.

One file is written per run:
  * an IMMUTABLE archive manifest at
    results/fits/manifests/<stage-slug>_maxiter<N>_<UTC>.json  (never overwritten)

There is no separate 'latest' pointer file: the archive filenames carry a
UTC timestamp (…_YYYYMMDDTHHMMSSZ.json), so the most recent run is recovered
by sorting them. (The old results/fits/run_manifest.json pointer was retired —
nothing read it, so it was pure redundancy.)

Grid definitions are read from the live model modules, not hardcoded, so they
cannot drift from what actually ran.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_sha() -> dict:
    """Current commit + dirty flag; {commit: None} if git is unavailable.

    Runs with the global/system git config neutralised (GIT_CONFIG_GLOBAL /
    GIT_CONFIG_SYSTEM -> os.devnull). rev-parse and status need no user config,
    and in constrained environments (CI containers, sandboxes) an unreadable
    ~/.gitconfig otherwise makes git exit 128 and the SHA silently go null — the
    one place a stamped commit matters most. import os is local to keep the
    module-level import block minimal."""
    import os
    try:
        here = Path(__file__).parent
        env = {**os.environ,
               "GIT_CONFIG_GLOBAL": os.devnull,
               "GIT_CONFIG_SYSTEM": os.devnull}
        sha = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, cwd=here, env=env).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, cwd=here, env=env).stdout.strip()
        return {"commit": sha or None, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def _grid() -> dict:
    grid = {}
    try:
        from observers.helpers.belief_grid import make_k_grid
        from observers.models.hb_adaptive_confidence import make_alpha_grid
        grid["k_grid"] = [float(x) for x in make_k_grid(n=15)]
        grid["alpha_grid"] = [float(x) for x in make_alpha_grid(9)]
    except Exception as e:
        grid["error"] = str(e)
    return grid


def _stage_slug(fit_models, cv_models) -> str:
    """Stage-aware slug so a fit-only run is named fit-..., a CV-only run cv-...,
    and a combined run fit-..._cv-... — a CV-only pass is never archived under a
    misleading 'fit_' name."""
    parts = []
    if fit_models:
        parts.append("fit-" + "-".join(fit_models))
    if cv_models:
        parts.append("cv-" + "-".join(cv_models))
    return "_".join(parts) or "none"


def write_manifest(out_dir, *, driver, fit_models, cv_models, subjects,
                   maxiter, folds, extra_config=None, notes=None) -> dict:
    """Write the archive manifest + refresh the latest pointer. Returns the dict.

    Call this at launch, BEFORE any fit, so the manifest exists even if the run
    is interrupted. `driver` is a short string identifying the caller
    ('run_all' / 'run_parallel'). `fit_models`/`cv_models` are the model-name
    lists for each stage (either may be empty). `extra_config` merges into the
    config block (driver-specific flags: workers, rec_nsim, ...).
    """
    import numpy, scipy

    config = {
        "fit_models": list(fit_models or []),
        "cv_models": list(cv_models or []),
        "subjects": list(subjects),
        "maxiter": maxiter,
        "folds": folds,
    }
    if extra_config:
        config.update(extra_config)

    man = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "driver": driver,
        "git": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__},
        "config": config,
        "grid": _grid(),
        "prior_mean_deg": 225.0,
    }
    if notes:
        man["notes"] = notes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(man, indent=2)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = out_dir / "manifests"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{_stage_slug(config['fit_models'], config['cv_models'])}_maxiter{maxiter}_{stamp}.json"
    archive_path.write_text(payload)

    # No latest-pointer file: every run writes its own immutable manifest under
    # manifests/, and "latest" is recovered by sorting those UTC-stamped
    # filenames (…_YYYYMMDDTHHMMSSZ.json). The old run_manifest.json pointer was
    # redundant (nothing read it) and has been retired.
    print(f"[{driver}] manifest -> {archive_path}", flush=True)
    return man
