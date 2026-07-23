"""
refit_delta.py — before/after report for a parity refit.

Compares a snapshot of point (and optionally CV) fit JSONs taken BEFORE a refit
against the current fits, and prints per-subject NLL / AIC changes per model plus
the effect on the Switch-vs-competitor AIC gap — the numbers that tell you
whether tightening the competitors' tolerance did what we predicted (their NLL
drops slightly; Switch's in-sample lead narrows).

    python -m observers.comparison.refit_delta \
        --before results/fits/_prerefit_backup_<ts> \
        --models basic_bayes hb_adaptive --reference switch

`--before` must contain point/<model>/subjectN.json (and cv/<model>/subjectN_cv.json
if --cv). The "after" side is read from the live results tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from observers.helpers.paths import FITS_DIR

POINT_AFTER = Path(FITS_DIR) / "comparison"       # results/fits/comparison/<model>
CV_AFTER = Path(FITS_DIR) / "comparison_cv"        # results/fits/comparison_cv/<model>


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _point(root_model: Path, sid: int):
    return _load(root_model / f"subject{sid}.json")


def _cv(root_model: Path, sid: int):
    return _load(root_model / f"subject{sid}_cv.json")


def _cv_nll(rec):
    """Held-out total NLL from a CV record, tolerant of field naming."""
    if rec is None:
        return None
    for k in ("cv_nll", "heldout_nll", "total_nll", "nll"):
        if k in rec:
            return float(rec[k])
    return None


def _fmt(x, w=9, p=1):
    return f"{x:{w}.{p}f}" if x is not None else " " * (w - 3) + "n/a"


def report(before_dir: Path, models, reference, subjects, do_cv):
    before_dir = Path(before_dir)
    lines = []
    lines.append(f"# Refit delta report")
    lines.append(f"before snapshot: {before_dir}")
    lines.append(f"models refit:    {', '.join(models)}   reference (unchanged): {reference}")
    lines.append("")

    # ---- point fits: per-subject NLL / AIC before -> after ----
    for m in models:
        b_root = before_dir / "point" / m
        a_root = POINT_AFTER / m
        lines.append(f"## {m} — point fit (NLL, AIC, tol/maxiter/hit_maxiter)")
        lines.append(f"{'subj':>4} {'NLL_before':>11} {'NLL_after':>10} {'dNLL':>8} "
                     f"{'AIC_before':>11} {'AIC_after':>10} {'dAIC':>8} "
                     f"{'iter_a':>7} {'hitmax_a':>8}")
        sum_db = sum_da = 0.0
        for s in subjects:
            b, a = _point(b_root, s), _point(a_root, s)
            if b is None or a is None:
                lines.append(f"{s:>4}   (missing before/after)")
                continue
            dnll = a["nll"] - b["nll"]
            daic = a["aic"] - b["aic"]
            sum_db += dnll
            sum_da += daic
            ci = a.get("convergence") or {}
            lines.append(f"{s:>4} {_fmt(b['nll'],11)} {_fmt(a['nll'],10)} {dnll:>8.1f} "
                         f"{_fmt(b['aic'],11)} {_fmt(a['aic'],10)} {daic:>8.1f} "
                         f"{ci.get('n_iter','?'):>7} {str(ci.get('hit_maxiter','?')):>8}")
        lines.append(f"{'SUM':>4} {'':>11} {'':>10} {sum_db:>8.1f} "
                     f"{'':>11} {'':>10} {sum_da:>8.1f}")
        lines.append("")

    # ---- Switch-vs-competitor AIC gap: before vs after ----
    ref_root = POINT_AFTER / reference  # reference unchanged, so before==after
    lines.append(f"## In-sample AIC gap vs {reference}  (positive => {reference} better)")
    lines.append(f"{'model':>13} {'gap_before':>11} {'gap_after':>10} {'change':>9} "
                 f"{'wins_before':>12} {'wins_after':>11}")
    for m in models:
        b_root = before_dir / "point" / m
        a_root = POINT_AFTER / m
        gb = ga = 0.0
        wb = wa = 0
        n = 0
        for s in subjects:
            rref = _point(ref_root, s)
            b, a = _point(b_root, s), _point(a_root, s)
            if rref is None or b is None or a is None:
                continue
            n += 1
            gb += b["aic"] - rref["aic"]     # competitor - reference (before)
            ga += a["aic"] - rref["aic"]     # competitor - reference (after)
            wb += int(rref["aic"] < b["aic"])   # reference beats competitor?
            wa += int(rref["aic"] < a["aic"])
        # report gap as reference-favoring = -(competitor - reference)/... keep sign explicit:
        # gap_before below is mean (competitor-ref); positive => reference better.
        mgb = gb / n if n else float("nan")
        mga = ga / n if n else float("nan")
        lines.append(f"{m:>13} {mgb:>11.1f} {mga:>10.1f} {mga-mgb:>9.1f} "
                     f"{wb:>10}/{n} {wa:>9}/{n}")
    lines.append("")
    lines.append("Interpretation: gap = mean(AIC_competitor - AIC_switch). Positive means "
                 "Switch has the lower AIC (better). If tightening the competitors' tolerance "
                 "helped them, the gap should SHRINK (move toward 0) after the refit.")

    if do_cv:
        lines.append("")
        for m in models:
            b_root = before_dir / "cv" / m
            a_root = CV_AFTER / m
            lines.append(f"## {m} — CV held-out NLL")
            lines.append(f"{'subj':>4} {'before':>11} {'after':>10} {'delta':>9}")
            for s in subjects:
                nb = _cv_nll(_cv(b_root, s))
                na = _cv_nll(_cv(a_root, s))
                d = (na - nb) if (nb is not None and na is not None) else None
                dstr = f"{d:>9.1f}" if d is not None else f"{'n/a':>9}"
                lines.append(f"{s:>4} {_fmt(nb,11)} {_fmt(na,10)} {dstr}")
            lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="pre-refit backup dir (has point/ and cv/)")
    ap.add_argument("--models", nargs="+", default=["basic_bayes", "hb_adaptive"])
    ap.add_argument("--reference", default="switch")
    ap.add_argument("--subjects", nargs="+", type=int,
                    default=list(range(1, 13)))
    ap.add_argument("--cv", action="store_true", help="also report CV held-out NLL")
    ap.add_argument("--out", default=None, help="write report to this path too")
    a = ap.parse_args()
    txt = report(Path(a.before), a.models, a.reference, a.subjects, a.cv)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
