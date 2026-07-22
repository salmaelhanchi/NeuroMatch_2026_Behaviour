"""
cross_validate.py — block-fold held-out per-trial NLL
=====================================================

Sequence-preserving cross-validation for every registered model. Folds are
CONTIGUOUS segments aligned to block (prior-width) boundaries where possible, so
a whole block is held out at a time and the learning dynamics are tested
out-of-sample rather than interpolated. For each fold we fit on the training
trials and score the held-out per-trial NLL; the model's belief filter always
runs over the full ordered sequence so trial order / learning is preserved.

Cross-validated NLL is the overfitting-proof metric and the one to LEAD with:
it needs no complexity penalty because held-out data penalises over-flexible
models directly. AIC/BIC (in the batch stage) corroborate.

Caveat (documented, same as the house fitters): feedback (true direction) is
available on every trial regardless of the mask, so the belief still 'sees'
held-out feedback. This tests predictive fit of the RESPONSES with order
intact, not a strictly causal forecast.

Writes one JSON per model x subject under results/fits/comparison_cv/<model>/subject<N>_cv.json.
Resumable (skip existing unless --force).

Usage:
  python -m observers.comparison.cross_validate --subjects 1 2 --folds 5
  python -m observers.comparison.cross_validate --models hb_adaptive switch
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from observers.comparison.registry import (
    build_registry, load_subject, ALL_SUBJECTS, DEFAULT_MODELS)
from observers.helpers.paths import FITS_DIR

OUT_DIR = FITS_DIR / "comparison_cv"


# --------------------------------------------------------------------------- #
#  Live health ticker — sub-fold liveness for a long CV run
# --------------------------------------------------------------------------- #
# A CV fold is one full Nelder-Mead fit (minutes), so a fold-only log is silent
# for the whole fit. A background daemon thread prints a machine-health beat
# every INTERVAL seconds (load, free RAM, this process' rss, elapsed, and which
# fold is in flight) so a watcher (or fit_monitor.py) sees the job is alive and
# can flag a stall — exactly the health signals hier_online's per-eval heartbeat
# carries, but model-agnostic (it wraps the shared CV loop, touches no fitter).
import threading


def _mem_free_mb():
    """Best-effort available memory (MB): Linux /proc, macOS vm_stat, else nan."""
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 1024
    except OSError:
        pass
    try:
        import re, subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        free = re.search(r"Pages free:\s+(\d+)", out)
        spec = re.search(r"Pages speculative:\s+(\d+)", out)
        n = (int(free.group(1)) if free else 0) + (int(spec.group(1)) if spec else 0)
        return n * 4096 / (1024 ** 2)
    except Exception:
        return float("nan")


def _rss_mb():
    import resource, sys as _sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2) if _sys.platform == "darwin" else rss / 1024


class _HealthTicker(threading.Thread):
    """Daemon thread printing a machine-health beat every ``interval`` seconds.

    ``state`` is a dict the CV loop mutates in place (``fold``, ``n_folds``,
    ``fold_t0``) so each beat reports which fold is running and for how long.
    Reads only OS stats + shared ints; never touches the fit numerics.
    """

    def __init__(self, tag, state, interval=30.0):
        super().__init__(daemon=True)
        self.tag = tag
        self.state = state
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = time.time()

    def stop(self):
        self._stop.set()

    def run(self):
        import os as _os
        while not self._stop.wait(self.interval):
            try:
                load1 = _os.getloadavg()[0]
            except OSError:
                load1 = float("nan")
            st = self.state
            fold_age = time.time() - st.get("fold_t0", self._t0)
            print(f"[cv{self.tag}] ..alive  fold {st.get('fold', 0)}/{st.get('n_folds', 0)} "
                  f"(fitting {fold_age:4.0f}s)  t={time.time() - self._t0:5.0f}s  "
                  f"rss={_rss_mb():5.0f}MB  free={_mem_free_mb():6.0f}MB  load={load1:4.1f}",
                  flush=True)


def _result_path(model: str, sid: int) -> Path:
    return OUT_DIR / model / f"subject{sid}_cv.json"


def _block_folds(prior_std: np.ndarray, K: int) -> list:
    """Contiguous folds aligned to block boundaries. Blocks are maximal runs of
    a constant prior_std; we group whole blocks into K contiguous folds so a
    held-out fold is one or more complete blocks (never a block split)."""
    ps = np.asarray(prior_std)
    # block id increments whenever prior_std changes
    change = np.concatenate([[True], ps[1:] != ps[:-1]])
    block_id = np.cumsum(change) - 1
    n_blocks = int(block_id.max()) + 1
    if n_blocks < K:              # fall back to plain contiguous trial folds
        idx = np.arange(len(ps))
        return [f for f in np.array_split(idx, K)]
    block_groups = np.array_split(np.arange(n_blocks), K)
    folds = []
    for grp in block_groups:
        mask = np.isin(block_id, grp)
        folds.append(np.flatnonzero(mask))
    return folds


def cv_one(spec, data, sid: int, K: int, maxiter: int, ticker_interval=30.0) -> dict:
    n = int(data["estimates"].size)
    folds = _block_folds(data["prior_std"], K)
    t0 = time.time()
    fold_nlls, tot = [], 0.0

    # live health/progress: a background ticker prints a machine-health beat
    # every ticker_interval s (sub-fold liveness), and each fold prints a
    # convergence line on completion. state is shared with the ticker.
    tag = f"[{spec.name} s{sid}]"
    state = {"fold": 0, "n_folds": len(folds), "fold_t0": time.time()}
    ticker = (_HealthTicker(tag, state, interval=ticker_interval)
              if ticker_interval and ticker_interval > 0 else None)
    print(f"[cv{tag}] START  {len(folds)}-fold CV  n={n}  maxiter={maxiter}  "
          f"k={spec.n_params}  learns={spec.learns}", flush=True)
    if ticker is not None:
        ticker.start()
    try:
        for f, test_idx in enumerate(folds):
            state["fold"] = f + 1
            state["fold_t0"] = time.time()
            test = np.zeros(n, dtype=bool); test[test_idx] = True
            train = ~test
            ft0 = time.time()
            fr = spec.fit(data, maxiter=maxiter, mask=train)
            # score held-out per-trial NLL using the model's own trial-loglik fn
            ll = spec.trial_logliks(fr.obs, data)
            test_nll = float(-ll[test].sum())
            per_trial = test_nll / max(int(test.sum()), 1)
            fold_nlls.append({"fold": f + 1, "n_test": int(test.sum()),
                              "held_out_nll": test_nll,
                              "per_trial": per_trial})
            tot += test_nll
            # convergence gauge from the fitter's own diagnostics, when present
            # (conv_info: converged / n_iter / hit_maxiter). A fold that hit the
            # iteration cap is the reviewer-relevant flag.
            info = getattr(fr.obs, "_fit_info", None) or {}
            if not info:
                conv = "?"
            elif info.get("converged"):
                conv = f"converged({info.get('n_iter', '?')}it)"
            elif info.get("hit_maxiter"):
                conv = f"⚠hit-maxiter({info.get('n_iter', '?')}it)"
            else:
                conv = f"stopped({info.get('n_iter', '?')}it)"
            print(f"[cv{tag}] fold {f + 1}/{len(folds)}  n_test={int(test.sum()):4d}  "
                  f"train_nll={fr.nll:9.1f}  held_out={test_nll:8.1f}  "
                  f"(per-trial {per_trial:.3f})  spread={fr.start_spread:6.1f}  "
                  f"{conv}  {time.time() - ft0:4.0f}s", flush=True)
    finally:
        if ticker is not None:
            ticker.stop()
    print(f"[cv{tag}] DONE  CV-NLL={tot:.1f}  (per-trial {tot / n:.3f})  "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return {
        "model": spec.name, "label": spec.label, "subject": sid, "n_trials": n,
        "k": int(spec.n_params), "folds": K,
        "cv_nll": float(tot), "cv_per_trial": float(tot / n),
        "fold_detail": fold_nlls,
        "maxiter": int(maxiter), "seconds": round(time.time() - t0, 1),
    }


def run(models=None, subjects=None, folds=5, maxiter=400, force=False,
        ticker_interval=30.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    reg = build_registry(models)
    done, skipped = [], []
    for sid in subjects:
        data = load_subject(int(sid))
        for name in models:
            path = _result_path(name, int(sid))
            if path.exists() and path.stat().st_size > 0 and not force:
                skipped.append(path.name)
                print(f"skip  {name:12s} subject {sid} (exists)", flush=True)
                continue
            row = cv_one(reg[name], data, int(sid), folds, maxiter,
                         ticker_interval=ticker_interval)
            path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(row, open(path, "w"), indent=2)
            done.append(path.name)
            print(f"done  {name:12s} subject {sid}  CV-NLL={row['cv_nll']:.1f} "
                  f"(per-trial {row['cv_per_trial']:.3f}) ({row['seconds']}s)", flush=True)
    print(f"\nCV complete: {len(done)} run, {len(skipped)} skipped -> {OUT_DIR}", flush=True)
    return done, skipped


def load_all(models=None, subjects=None) -> dict:
    models = models or DEFAULT_MODELS
    subjects = subjects or ALL_SUBJECTS
    out = {m: {} for m in models}
    for m in models:
        for sid in subjects:
            p = _result_path(m, int(sid))
            if p.exists() and p.stat().st_size > 0:
                out[m][int(sid)] = json.load(open(p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ticker-interval", type=float, default=30.0,
                    help="seconds between background health beats (0 disables)")
    a = ap.parse_args()
    run(models=a.models, subjects=a.subjects, folds=a.folds, maxiter=a.maxiter,
        force=a.force, ticker_interval=a.ticker_interval)
