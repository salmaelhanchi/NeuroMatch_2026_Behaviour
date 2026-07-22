# Asymptote + Transient Switching observer — build + results

The model the switch-probability learning curve pointed to: keep the static
model's flexible per-block prior LEVELS and add the online model's within-block
TRANSIENT toward them. Files: `asymptote_transient.py` (model),
`asymptote_transient_fit.py` (verify/recover/fit), `build_switch_curve_at.py`
(targeted test), `switch_curve_at.png`.

## Model (Eq. AT1)
Within a block the effective prior strength relaxes exponentially:

    k_eff(t) = k_asym[block] + (k_start − k_asym[block])·exp(−t/τ)

- k_asym[block]: per-block asymptote (4 free params) — static model's flexibility.
- k_start: effective strength carried over from the previous block's end.
- τ: asymmetric — τ_tighten (prior must strengthen) vs τ_loosen (must weaken).
  Reuses the verified `estimate_distribution_fixedk` (scalar k per trial), so the
  likelihood is deterministic. 11 params (static 9, online 6).

## Verification
- Reduction: carryover-off ⇒ EXACT static observer (ΔNLL = 0; static path already
  == switching_observer.py to 1e-17).
- k_eff limits correct (t0 = carryover, t∞ = asymptote).
- Parameter recovery: sensory/motor/lapse/asymptotes recover well;
  **asymmetry τ_loosen>τ_tighten recovered every run** (τ_loosen nailed to ~1%,
  τ_tighten weakly identified — fast transients leave little trace).

## Human fits (subjects 1, 3, 5)
- **Subject 1: AT BEST** — AIC 77717 vs static 77757 vs online 78099 (beats both).
  Genuine per-block transient (tightening, τ_tighten≈1.3).
- Subject 3: online best; AT correctly collapses to static (τ→0). Subject 3's
  learning is GLOBAL (session-start belief sharpening, online's λ≈0 mechanism),
  not per-block — so AT's per-block mechanism finds nothing.
- Subject 5: inconclusive — 11-dim Nelder-Mead from a cold start got stuck
  (reported AT worse than static, impossible since AT nests static). Needs
  warm-start from the static fit; warm-started refit exceeded the 10-min compute
  cap here. A robust all-subject fit is a batch job, not finished this session.

## Targeted test — reproduce the switch curve (subjects 1, 3; no new fitting)
`switch_curve_at.png`: AT tracks the empirical within-block P(prior) curve
**better than online — SSE 0.094 → 0.051 (roughly halved)**, correcting online's
over-prediction of prior reliance at both early and late positions. Both models
still over-predict somewhat and neither perfectly matches the empirical monotone
decline, so AT is a clear improvement, not a perfect fit.

## Honest bottom line
The AT model is validated, can win per-subject (subject 1 beats both rivals), has
an identifiable slow-to-loosen asymmetry, and reproduces the switch curve better
than online — the "levels + transient" direction is confirmed to have merit. BUT
it's heterogeneous across subjects (per-block transient for some, global learning
for others), the 11-dim fit is fragile (needs warm-start/multi-start), and the
transient's exact shape isn't fully captured. Promising improvement, not a final
model.

## Next steps
- Robust all-12-subject fits: warm-start AT from each subject's static fit +
  multi-start; run as a background batch (each ~5–8 min).
- The residual shape mismatch suggests refining the transient form (e.g., let
  k_start be the actual carried belief, or a two-timescale relaxation).
- Per-subject RT significance test (still queued) to label the switch on
  near-prior trials and further constrain the transient.