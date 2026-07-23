"""
paper_stationarity.py — reproduce Laquitaine & Gardner's (2018) parameter-
stability check (Neuron 97, p.466), the paper's ACTUAL first/last-half analysis.

This is NOT the cross-prediction test in split_half_bimodality.py. The paper did
an IN-SAMPLE stationarity check: fit the Switching observer separately to the
FIRST and LAST (temporal) halves of each subject's data, then test whether the
goodness of fit (AIC) and the prior-strength parameters differ between halves. A
non-significant difference means learning has plateaued and the fit does not
drift across the session.

Paper's reported result (n=12 direction-task subjects):
  * AICs did not differ between halves:            Wilcoxon Z=32, p=0.62
  * prior-strength params did not differ:          Z=37, p=0.90 (80 deg prior)
                                                    Z=25, p=0.30 (40 deg prior)
                                                    Z=23, p=0.23 (20 deg prior)

We refit with the repo's paper-standard Switch fitter and run the same Wilcoxon
signed-rank tests, reporting our Z/p beside the paper's.

Usage:
  PYTHONPATH=. python experiments/rachel/23_split_half_bimodality/paper_stationarity.py
"""
from __future__ import annotations
import json, os, time
import numpy as np
from scipy.stats import wilcoxon

from observers.comparison.registry import build_registry, load_subject, ALL_SUBJECTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
PRIOR_KEYS = ["80", "40", "20", "10"]
# paper's reported values, for side-by-side reporting (direction task, n=12)
PAPER = {"aic": (32, 0.62), "kp80": (37, 0.90), "kp40": (25, 0.30), "kp20": (23, 0.23)}


def _first_last_masks(n):
    """Temporal split: first half vs last half (contiguous), as in the paper."""
    h = n // 2
    first = np.zeros(n, bool); first[:h] = True
    last = np.zeros(n, bool);  last[n - h:] = True
    return first, last


def _fit_half(spec, data, mask, maxiter):
    res = spec.fit(data, maxiter=maxiter, mask=mask)
    obs = res.obs
    kp = getattr(obs, "k_prior", {}) or {}
    kp = {str(k): float(v) for k, v in kp.items()}
    n_used = int(mask.sum())
    k = int(res.n_params)
    aic = 2 * k + 2 * float(res.nll)   # AIC on the half it was fit to (in-sample)
    return {"nll": float(res.nll), "aic": aic, "n": n_used, "k_prior": kp}


def run(subjects=None, maxiter=400, model="switch"):
    subjects = subjects or ALL_SUBJECTS
    os.makedirs(OUT, exist_ok=True)
    spec = build_registry([model])[model]
    per = {}
    print(f"paper stationarity check ({model}): first vs last half, "
          f"{len(subjects)} subjects (serial)", flush=True)
    for sid in subjects:
        t0 = time.time()
        data = load_subject(int(sid))
        n = data["estimates"].size
        first, last = _first_last_masks(n)
        rec = {"first": _fit_half(spec, data, first, maxiter),
               "last": _fit_half(spec, data, last, maxiter)}
        rec["seconds"] = round(time.time() - t0, 1)
        per[int(sid)] = rec
        json.dump(rec, open(os.path.join(OUT, f"stationarity_subject{sid}.json"), "w"), indent=2)
        print(f"  subject {sid}: AIC first={rec['first']['aic']:.0f} "
              f"last={rec['last']['aic']:.0f} ({rec['seconds']}s)", flush=True)

    # Wilcoxon signed-rank: first vs last, matched by subject
    def _paired(getter):
        a = np.array([getter(per[s]["first"]) for s in per])
        b = np.array([getter(per[s]["last"]) for s in per])
        keep = np.isfinite(a) & np.isfinite(b)
        return a[keep], b[keep]

    tests = {}
    a, b = _paired(lambda h: h["aic"])
    W, p = wilcoxon(a, b)
    tests["aic"] = {"W": float(W), "p": float(p), "n": int(len(a))}
    for pk in ["80", "40", "20"]:   # paper reports these three (10 deg often degenerate)
        a, b = _paired(lambda h, _k=pk: h["k_prior"].get(_k, np.nan))
        if len(a) >= 3:
            W, p = wilcoxon(a, b)
            tests[f"kp{pk}"] = {"W": float(W), "p": float(p), "n": int(len(a))}

    summary = {"model": model, "n_subjects": len(per), "tests": tests, "paper": PAPER}
    json.dump(summary, open(os.path.join(OUT, "stationarity_summary.json"), "w"), indent=2)

    print("\nWilcoxon signed-rank, first vs last half (ours vs paper):")
    labels = {"aic": "AIC", "kp80": "k_prior[80]", "kp40": "k_prior[40]", "kp20": "k_prior[20]"}
    for key, lab in labels.items():
        if key in tests:
            t = tests[key]; pz, pp = PAPER.get(key, (None, None))
            paper_str = f"paper W~{pz}, p={pp}" if pz is not None else ""
            verdict = "n.s. (stable)" if t["p"] > 0.05 else "DIFFERS"
            print(f"  {lab:14s}: ours W={t['W']:.0f}, p={t['p']:.2f}, n={t['n']}  "
                  f"[{verdict}]   {paper_str}")
    print(f"\ndone -> {OUT}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--model", default="switch")
    a = ap.parse_args()
    run(subjects=a.subjects, maxiter=a.maxiter, model=a.model)
