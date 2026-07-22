"""
validate_model.py — a PER-MODEL validation record, filed in the model's own folder.

`validate_all.py` certifies a *comparison* (a set of models vs a reference) and
writes one run-level report at the top of results/fits/. This is its per-model
companion: it validates ONE model on its own terms and writes the record INTO that
model's folder, matching the per-model layout of every other artifact:

    results/fits/comparison/<model>/validation.md     (human-readable)
    results/fits/comparison/<model>/validation.json    (machine-readable)

It reuses `validate_all`'s check functions scoped to the single model, so the
thresholds and pass/fail/warn semantics are IDENTICAL to the comparison report —
there is one source of truth for what "valid" means. What it adds is (a) a
model-specific Step-0 verify (not the whole suite) and (b) a per-subject fit table
(nll / aic / bic / start_spread / convergence / seconds), which the comparison
report pools away.

`refit_model.py` calls this automatically at the end of a refit, so every model
gets a fresh per-model record whenever its fits are regenerated.

    python -m observers.comparison.validate_model --model hb_rachel
    python -m observers.comparison.validate_model --model hb_salma --no-cv
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from observers.helpers.paths import FITS_DIR
from observers.comparison.registry import ALL_SUBJECTS, ALL_MODELS
from observers.comparison import validate_all as VA
from observers.comparison.manifest import git_sha

# Registry key -> observers.api verify function name. Kept explicit because the
# api wrapper names do not all match the registry keys (switch -> verify_switching,
# basic_bayes -> verify_basic_bayesian). A key absent here (or whose function is
# missing) yields a SKIP, not a crash.
_VERIFY_FN = {
    "switch": "verify_switching",
    "basic_bayes": "verify_basic_bayesian",
    "hb_adaptive": "verify_hb_adaptive",
    "hb_rachel": "verify_hb_rachel",
    "hb_salma": "verify_hb_salma",
    "recombined": "verify_recombined",
    "hierarchical_online": "verify_online",
    "reliability_mixture": "verify_reliability_mixture",
}


def _verify_one(model: str) -> dict:
    """Run only this model's verify suite. Returns {status, detail}."""
    from observers import api
    fn_name = _VERIFY_FN.get(model)
    fn = getattr(api, fn_name, None) if fn_name else None
    if fn is None:
        return {"status": "SKIP",
                "detail": {"note": f"no api.{fn_name or 'verify_'+model} — "
                                   "model has no dedicated verifier"}}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            passed, total = fn()
    except Exception as e:
        return {"status": "FAIL", "detail": {"error": repr(e)}}
    return {"status": "PASS" if passed == total else "FAIL",
            "detail": {"passed": passed, "total": total}}


def _fit_table(model: str, subjects) -> dict:
    """Per-subject point-fit health for this model. Reads the fit JSONs and
    surfaces the fields the pooled comparison report hides."""
    pt_dir = FITS_DIR / "comparison" / model
    rows = []
    for sid in subjects:
        f = pt_dir / f"subject{sid}.json"
        if not f.exists():
            rows.append({"subject": sid, "present": False})
            continue
        j = json.loads(f.read_text())
        conv = j.get("convergence", {}) or {}
        rows.append({
            "subject": sid, "present": True,
            "nll": j.get("nll"), "aic": j.get("aic"), "bic": j.get("bic"),
            "k": j.get("k"), "n_trials": j.get("n_trials"),
            "start_spread": j.get("start_spread"),
            "maxiter": j.get("maxiter"), "seconds": j.get("seconds"),
            "converged": conv.get("converged"),
            "hit_maxiter": conv.get("hit_maxiter"),
        })
    present = [r for r in rows if r.get("present")]
    # A model refit through the shared multistart() helper should show a NON-zero
    # start_spread; an all-zero column is the fingerprint of a single-start (stale
    # or mis-wired) fit path. Reported as a warning, not a hard fail.
    spreads = [r.get("start_spread") for r in present if r.get("start_spread") is not None]
    single_start = bool(spreads) and all(s == 0.0 for s in spreads)
    return {
        "n_subjects": len(present),
        "single_start_suspect": single_start,
        "rows": rows,
    }


def run(model: str, subjects=None, folds: int = 5, cv: bool = True) -> dict:
    subjects = subjects or ALL_SUBJECTS
    ms = [model]  # scope validate_all's list-taking checks to just this model

    rep = {
        "kind": "per_model_validation",
        "model": model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_sha(),
        "subjects": list(subjects),
        "folds": folds,
        "checks": {
            "0_verify":         _verify_one(model),
            "1_fit_table":      _fit_table(model, subjects),
            "2_fit_convergence": VA.item2_fit_convergence(ms),
            "3_aic_bic":        VA.item3_aic_bic(ms),
            "5_shape":          VA.item5_shape(ms),
        },
    }
    if cv:
        rep["checks"]["4_cv_validity"] = VA.item4_cv_validity(ms, folds)
    else:
        rep["checks"]["4_cv_validity"] = {"status": "SKIP",
            "detail": {"note": "fit-only model (--no-cv); no CV expected"}}

    # Overall verdict: any FAIL among the status-bearing checks fails the model.
    statuses = [c.get("status") for c in rep["checks"].values() if isinstance(c, dict) and "status" in c]
    rep["summary"] = {
        "fail": sum(s == "FAIL" for s in statuses),
        "pass": sum(s == "PASS" for s in statuses),
        "skip": sum(s == "SKIP" for s in statuses),
        "verdict": "FAIL" if any(s == "FAIL" for s in statuses) else "PASS",
    }
    return rep


def to_markdown(rep: dict) -> str:
    def _v(s): return {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "SKIP": "⏭️ SKIP"}.get(s, s)
    c = rep["checks"]
    L = [f"# Validation — `{rep['model']}`",
         f"_{rep['timestamp_utc']}_ · git `{(rep['git'].get('commit') or '?')[:8]}`"
         f"{' (dirty)' if rep['git'].get('dirty') else ''} · "
         f"verdict **{rep['summary']['verdict']}**\n",
         "| check | status |", "|---|---|",
         f"| 0. verify (model-specific) | {_v(c['0_verify']['status'])} |",
         f"| 2. fit convergence / spread | {_v(c['2_fit_convergence']['status'])} |",
         f"| 3. AIC/BIC well-formed | {_v(c['3_aic_bic']['status'])} |",
         f"| 4. CV validity | {_v(c['4_cv_validity']['status'])} |",
         f"| 5. shape reproduction | {_v(c['5_shape']['status'])} |",
         ""]
    ft = c["1_fit_table"]
    if ft["single_start_suspect"]:
        L.append("> ⚠️ **start_spread is 0 for every subject** — these fits look "
                 "single-start (stale or a mis-wired fit path). Refit through "
                 "`multistart()`.\n")
    L += [f"## Per-subject fits ({ft['n_subjects']}/{len(rep['subjects'])} present)",
          "| subj | NLL | AIC | k | start_spread | conv | hit_max | s |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ft["rows"]:
        if not r.get("present"):
            L.append(f"| {r['subject']} | — missing — |||||||")
            continue
        L.append("| {subject} | {nll:.1f} | {aic:.1f} | {k} | {ss} | {cv} | {hm} | {sec} |".format(
            subject=r["subject"], nll=r["nll"] or float('nan'), aic=r["aic"] or float('nan'),
            k=r["k"], ss=("%.1f" % r["start_spread"]) if r.get("start_spread") is not None else "?",
            cv=r.get("converged"), hm=r.get("hit_maxiter"),
            sec=("%.0f" % r["seconds"]) if r.get("seconds") is not None else "?"))
    return "\n".join(L)


def write(rep: dict) -> tuple[Path, Path]:
    out_dir = FITS_DIR / "comparison" / rep["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "validation.md", out_dir / "validation.json"
    md.write_text(to_markdown(rep))
    js.write_text(json.dumps(rep, indent=2))
    return md, js


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-model validation record.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true",
                    help="fit-only model: skip the CV-validity check")
    a = ap.parse_args()
    if a.model not in ALL_MODELS:
        raise SystemExit(f"[validate_model] unknown model '{a.model}'. "
                         f"Registry keys: {ALL_MODELS}")
    rep = run(a.model, subjects=a.subjects, folds=a.folds, cv=not a.no_cv)
    md, js = write(rep)
    print(to_markdown(rep))
    print(f"\nrecord -> {md}\n         {js}")
    sys.exit(1 if rep["summary"]["verdict"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
