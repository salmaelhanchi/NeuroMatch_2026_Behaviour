"""
validate_cv.py — post-hoc validity checks for cross-validation result files.

A companion to ``fit_monitor.py`` (which watches a run live); this runs AFTER a
CV pass and audits the written ``results/fits/comparison_cv/<model>/subject<N>_cv.json``
files. It is model-agnostic: point it at any registered model.

Two layers of checks per subject:

STRUCTURAL (the CV bookkeeping is internally consistent)
    S1  fold held-out NLLs sum to the reported cv_nll        (round-trip < 1e-6)
    S2  folds cover every trial exactly once  (Σ n_test == n_trials)
    S3  k and folds match the model spec / requested folds
    S4  every per-trial NLL is finite and in a sane band

STATISTICAL (the fit actually generalises)
    G1  CV per-trial NLL beats the uniform baseline (ln 360 ≈ 5.886)
    G2  generalisation gap = cv_per_trial − insample_per_trial sits in
        (GAP_LO, GAP_HI): a negative gap suggests a train/test leak (held-out
        scoring better than the trained fit), a large positive gap suggests
        overfitting. Needs the matching point fit in results/fits/comparison/.

Usage
-----
    python -m observers.comparison.validate_cv --models hb_salma
    python -m observers.comparison.validate_cv --models hb_salma switch --subjects 1 3
    python -m observers.comparison.validate_cv --models hb_salma --json report.json

Exit status is non-zero if any subject fails any check, so it can gate a
pipeline. A subject with no CV file yet is reported as PENDING (not a failure).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from observers.helpers.paths import FITS_DIR

CV_DIR = FITS_DIR / "comparison_cv"
PT_DIR = FITS_DIR / "comparison"

UNIFORM_PER_TRIAL = math.log(360.0)   # 5.886 — a model must beat chance
PT_LO, PT_HI = 0.5, UNIFORM_PER_TRIAL + 0.5   # sane per-trial NLL band
GAP_LO, GAP_HI = -0.05, 0.6                    # cv − insample generalisation gap


def _cv_path(model, sid):
    return CV_DIR / model / f"subject{sid}_cv.json"


def _pt_path(model, sid):
    return PT_DIR / model / f"subject{sid}.json"


def check_subject(model: str, sid: int, expect_folds: int | None = None) -> dict:
    """Run all checks for one (model, subject). Returns a report dict with a
    per-check pass/fail map and an overall ``ok`` (None => pending, no file)."""
    cvp = _cv_path(model, sid)
    if not cvp.exists() or cvp.stat().st_size == 0:
        return {"model": model, "subject": sid, "ok": None, "status": "PENDING",
                "checks": {}, "detail": {}}

    r = json.load(open(cvp))
    fd = r.get("fold_detail", [])
    checks, notes = {}, {}

    # ---- structural ----
    foldsum = sum(x["held_out_nll"] for x in fd)
    checks["S1_foldsum==cv_nll"] = abs(foldsum - r["cv_nll"]) < 1e-6
    notes["foldsum"] = round(foldsum, 6)

    cover = sum(x["n_test"] for x in fd)
    checks["S2_cover==n_trials"] = cover == r["n_trials"]
    notes["cover"] = cover

    ok_k = (expect_folds is None or r["folds"] == expect_folds)
    checks["S3_k_folds_shape"] = bool(r.get("k") and r["folds"] == len(fd) and ok_k)
    notes["k"], notes["folds"], notes["n_folds_detail"] = r.get("k"), r["folds"], len(fd)

    pts = [x["per_trial"] for x in fd]
    checks["S4_per_trial_finite_sane"] = all(
        math.isfinite(p) and PT_LO < p < PT_HI for p in pts)
    notes["per_trial_range"] = [round(min(pts), 3), round(max(pts), 3)] if pts else None

    # ---- statistical ----
    cv_pt = r["cv_per_trial"]
    checks["G1_beats_uniform"] = cv_pt < UNIFORM_PER_TRIAL
    notes["cv_per_trial"], notes["uniform"] = round(cv_pt, 4), round(UNIFORM_PER_TRIAL, 4)

    ptp = _pt_path(model, sid)
    if ptp.exists():
        pt = json.load(open(ptp))
        in_pt = pt["nll"] / pt["n_trials"]
        gap = cv_pt - in_pt
        checks["G2_gap_in_band"] = GAP_LO < gap < GAP_HI
        notes["insample_per_trial"], notes["gap"] = round(in_pt, 4), round(gap, 4)
    else:
        notes["gap"] = "no point fit"   # G2 not evaluable; not counted as fail

    ok = all(checks.values())
    return {"model": model, "subject": sid, "ok": ok,
            "status": "PASS" if ok else "FAIL", "checks": checks, "detail": notes}


def run(models, subjects, expect_folds=None, verbose=True) -> dict:
    from observers.comparison.registry import ALL_SUBJECTS
    subjects = subjects or ALL_SUBJECTS
    report = {"uniform_per_trial": round(UNIFORM_PER_TRIAL, 3),
              "gap_band": [GAP_LO, GAP_HI], "rows": []}
    n_fail = n_pend = n_pass = 0
    for model in models:
        for sid in subjects:
            row = check_subject(model, int(sid), expect_folds)
            report["rows"].append(row)
            if row["ok"] is None:
                n_pend += 1
            elif row["ok"]:
                n_pass += 1
            else:
                n_fail += 1
            if verbose:
                d = row["detail"]
                if row["status"] == "PENDING":
                    print(f"[{model} s{sid}] PENDING (no CV file yet)")
                else:
                    failed = [k for k, v in row["checks"].items() if not v]
                    tail = ("  cv_pt=%.3f gap=%s" % (
                        d.get("cv_per_trial", float('nan')), d.get("gap")))
                    flag = "" if row["ok"] else "  FAILED: " + ",".join(failed)
                    print(f"[{model} s{sid}] {row['status']}{tail}{flag}")
    report["summary"] = {"pass": n_pass, "fail": n_fail, "pending": n_pend}
    if verbose:
        print(f"\n{n_pass} pass, {n_fail} fail, {n_pend} pending  "
              f"(uniform/trial={UNIFORM_PER_TRIAL:.3f}, gap band {GAP_LO}..{GAP_HI})")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None,
                    help="expected fold count (S3 flags a mismatch)")
    ap.add_argument("--json", type=str, default=None,
                    help="also write the full report to this path")
    a = ap.parse_args()
    report = run(a.models, a.subjects, expect_folds=a.folds)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
        print(f"report -> {a.json}")
    import sys
    sys.exit(1 if report["summary"]["fail"] else 0)


if __name__ == "__main__":
    main()
