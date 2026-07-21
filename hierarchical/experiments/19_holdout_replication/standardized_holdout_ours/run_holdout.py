"""Replicate Salma's standardized held-out comparison protocol, driven by OUR
registry models (switch / hb_adaptive / recombined / basic_bayes).

Protocol (mirrors standardized_hierarchical_comparison):
  * Chronological holdout: the last N complete SESSIONS are held out; the model
    only ever learns forward in time (causal). Belief resets at each session start.
  * Score = held-out per-trial negative log predictive score (natural log),
    summed over valid held-out responses -> held-out NLL / ELPD.
  * Paired block bootstrap over held-out RUNS gives a CI on each model-pair
    difference in mean log score.

We fit each model on the TRAINING trials only (mask), then score the held-out
trials under the model's own per-trial predicted distribution. For the learning
models the belief is advanced through the full causal sequence (train+test) so
the held-out predictions use everything learned up to each trial - identical in
spirit to her "reset at session start; causal feedback updates only".
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observers.comparison.registry import build_registry, load_subject

CSV = Path(__file__).resolve().parents[1] / "data" / "data01_direction4priors.csv"
OUT = Path(__file__).resolve().parent / "outputs"
# Faithful mapping to Salma's roles (from her model docstrings):
#   switching        -> our switch                (same model)
#   readout_average  -> our hb_rachel (=hb_integration; her docstring names it) — integrate-after, learns kappa, alpha fixed
# Plus two of our own models scored under the same protocol for context:
#   hb_adaptive  — the abstract's headline (joint kappa+alpha learning, integrate-after)
#   basic_bayes  — fixed always-integrate baseline
# recombined (integrate-before) is DROPPED here: it is the ~40min/subject bottleneck
# and is run separately. Add it back via --with-recombined if needed.
MODELS = ["switch", "readout_average", "hb_adaptive", "basic_bayes"]
OUR = {"switch": "switch", "readout_average": "hb_rachel",
       "hb_adaptive": "hb_adaptive", "basic_bayes": "basic_bayes"}


def session_layout(subject_id: int):
    """Return (session_ids, run_ids) per trial in our loader's order."""
    df = pd.read_csv(CSV, usecols=["subject_id", "session_id", "run_id", "trial_index"])
    s = df[df.subject_id == subject_id].sort_values(
        ["session_id", "run_id", "trial_index"], kind="mergesort").reset_index(drop=True)
    return s.session_id.to_numpy(), s.run_id.to_numpy()


def holdout_mask(session_ids: np.ndarray, n_sessions: int):
    sess = sorted(pd.unique(session_ids))
    held = sess[-n_sessions:]
    test = np.isin(session_ids, held)
    return ~test, test, held


def run(subject_id: int, n_sessions: int, maxiter: int, n_boot: int, seed: int):
    data = load_subject(subject_id)
    sess, runs = session_layout(subject_id)
    n = len(data["estimates"])
    assert len(sess) == n, f"layout {len(sess)} != data {n}"
    train, test, held = holdout_mask(sess, n_sessions)
    reg = build_registry([OUR[m] for m in MODELS])

    per_trial_ll = {}        # model -> per-trial loglik over ALL trials (causal)
    fits = {}
    for name in MODELS:
        spec = reg[OUR[name]]
        t0 = time.time()
        fr = spec.fit(data, maxiter=maxiter, mask=train)   # fit on TRAIN only
        ll = spec.trial_logliks(fr.obs, data)              # causal per-trial loglik
        sec = time.time() - t0
        k = int(spec.n_params)
        n_valid = int(np.isfinite(ll).sum())
        ho_nll = float(-ll[test].sum())
        per_trial_ll[name] = ll
        fits[name] = dict(model=name, role=OUR[name], k=k,
                          train_nll=float(-ll[train].sum()),
                          heldout_nll=ho_nll,
                          heldout_trials=int(test.sum()),
                          mean_log_score=float(ll[test].mean()),
                          aic=float(2*k - 2*ll[train].sum()),
                          seconds=round(sec, 1))
        print(f"  {name:20s} heldout_NLL={ho_nll:.1f} mean_log={ll[test].mean():.4f} ({sec:.0f}s)")

    # rank by held-out NLL
    order = sorted(fits, key=lambda m: fits[m]["heldout_nll"])
    best = order[0]
    for m in fits:
        fits[m]["elpd_diff_from_best"] = -(fits[m]["heldout_nll"] - fits[best]["heldout_nll"])

    # paired block bootstrap over held-out RUNS
    test_idx = np.flatnonzero(test)
    test_runs = np.array([f"{sess[i]}_{runs[i]}" for i in test_idx])
    uruns = pd.unique(test_runs)
    rng = np.random.default_rng(seed)
    pairs = []
    for i, a in enumerate(order):
        for b in order[i+1:]:
            diff = per_trial_ll[a][test_idx] - per_trial_ll[b][test_idx]
            boots = np.empty(n_boot)
            for k in range(n_boot):
                pick = rng.choice(uruns, size=len(uruns), replace=True)
                sel = np.concatenate([np.flatnonzero(test_runs == r) for r in pick])
                boots[k] = diff[sel].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pairs.append(dict(first=a, second=b,
                              mean_log_diff=float(diff.mean()),
                              ci_lo=float(lo), ci_hi=float(hi),
                              prob_first_better=float((boots > 0).mean()),
                              supported=("inconclusive" if lo < 0 < hi
                                         else (a if lo > 0 else b))))

    result = dict(
        run_config=dict(subject_id=subject_id, heldout_final_sessions=n_sessions,
                        heldout_sessions=[int(x) for x in held],
                        maxiter=maxiter, bootstrap_draws=n_boot,
                        models={m: OUR[m] for m in MODELS}),
        data_audit=dict(trials=n, train_trials=int(train.sum()),
                        heldout_trials=int(test.sum()),
                        sessions=int(len(set(sess)))),
        fits=fits, ranking=order, predictive_pairs=pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"holdout_subject{subject_id}.json"
    json.dump(result, open(p, "w"), indent=1)
    print(f"  -> {p}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=[3])
    ap.add_argument("--n-sessions", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260721)
    a = ap.parse_args()
    for sid in a.subjects:
        print(f"subject {sid}:")
        run(sid, a.n_sessions, a.maxiter, a.n_boot, a.seed)