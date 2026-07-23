"""
fit_heartbeat.py — model-agnostic intra-fit progress heartbeat for point fits.

The per-subject point fitters (basic_bayesian_fit, hb_adaptive_confidence_fit,
the switch static fit) each run one or more Nelder-Mead searches that can take
many minutes at the paper-standard 1e-4 tolerance with ~10 starts. Without a
heartbeat the job log is SILENT from launch to completion, so a watcher cannot
tell a live-but-slow fit from a stalled one, nor see whether the NLL is still
improving.

This module wraps a fit's objective so that every ``interval`` seconds it prints
one beat carrying the live numbers a watcher wants:

    [basic_bayes s5] start 3/10  eval  742  best=  30984.1  0.03s/eval  t= 22s  rss= 210MB load= 6.1

Design constraints (mirrors the CV HealthTicker already in cross_validate.py):
  * READS only OS stats + an eval counter it owns; never perturbs the fit maths
    (the wrapped objective returns the identical value it computed).
  * Zero-overhead and OFF by default: if no interval is configured (env
    FIT_HB_INTERVAL unset / 0) or no tag is set, ``wrap`` returns the objective
    unchanged, so non-fit_batch callers (CV, recovery, ad-hoc) are untouched.
  * ``configure`` is called once per (model, subject) by fit_batch; each new
    ``wrap`` bumps a start counter so the multistart sequence is legible.
"""
from __future__ import annotations

import os
import sys
import time

# Module-level context, set by fit_batch before each subject/model fit. Reading
# it here (rather than threading a param through every fitter signature) keeps
# the fitters' public APIs unchanged for every other caller.
_CTX = {
    "tag": None,
    "interval": float(os.environ.get("FIT_HB_INTERVAL", "0") or 0),
    "start_counter": 0,
    "n_starts": None,
}


def configure(tag=None, interval=None, n_starts=None):
    """Set the heartbeat context for the fit about to run. Resets the start
    counter so ``start k/n`` numbers each multistart afresh."""
    if tag is not None:
        _CTX["tag"] = tag
    if interval is not None:
        _CTX["interval"] = float(interval)
    _CTX["n_starts"] = n_starts
    _CTX["start_counter"] = 0


def reset():
    """Clear the context (heartbeat becomes a no-op again)."""
    _CTX.update(tag=None, start_counter=0, n_starts=None)


def _rss_mb():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2) if sys.platform == "darwin" else rss / 1024


def _load1():
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("nan")


class _Beating:
    """Callable wrapping an objective: counts evals, tracks best-so-far, prints
    a beat every ``interval`` seconds. The returned value is the objective's own
    value, unmodified."""

    def __init__(self, fn, maxiter, tag, interval, start_idx, n_starts):
        self.fn = fn
        self.maxiter = maxiter
        self.tag = tag
        self.interval = interval
        self.start_idx = start_idx
        self.n_starts = n_starts
        self.n = 0
        self.best = float("inf")
        self._t0 = time.time()
        self._last = self._t0

    def __call__(self, theta):
        v = self.fn(theta)
        self.n += 1
        if v < self.best:
            self.best = v
        now = time.time()
        if self.interval and (now - self._last) >= self.interval:
            elapsed = now - self._t0
            spe = elapsed / max(self.n, 1)
            startpos = (f"start {self.start_idx}/{self.n_starts}"
                        if self.n_starts else f"start {self.start_idx}")
            print(f"[{self.tag}] {startpos}  eval {self.n:5d}  "
                  f"best={self.best:10.1f}  {spe:.3f}s/eval  t={elapsed:4.0f}s  "
                  f"rss={_rss_mb():4.0f}MB  load={_load1():.1f}", flush=True)
            self._last = now
        return v


def wrap(obj, maxiter):
    """Wrap ``obj`` with a heartbeat if a context tag + positive interval are
    set; otherwise return ``obj`` unchanged (the default, zero-overhead path)."""
    tag = _CTX["tag"]
    interval = _CTX["interval"]
    if not tag or not interval or interval <= 0:
        return obj
    _CTX["start_counter"] += 1
    return _Beating(obj, maxiter, tag, interval,
                    _CTX["start_counter"], _CTX["n_starts"])
