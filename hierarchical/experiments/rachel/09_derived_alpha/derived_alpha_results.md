# Derived-α integration observer — the authors' formulation, built and fit

`hb_integration_derived.py`. The change we discussed: replace the free, fitted
mixture weight `α` with the value the paper's Discussion actually specified —
the reliability ratio — and let the online-learned prior strength drive it.

## What changed, in one line

| | free-α (`hb_integration.py`) | derived-α (`hb_integration_derived.py`) |
|---|---|---|
| mixture weight `α` | **fitted**, constant across trials | **computed**: `α(κ,k_e)=κ/(κ+k_e)` (paper Eq. 6), evolves as κ is learned |
| # free params | 7 | **6** (α removed) |
| what the paper says | tested a *static* mixture, it lost | *"probabilities determined by the ratio of likelihood and prior strengths ... a reformulation of our Switching observer"* |

κ is still learned online (belief `b_t(κ)`, forgetting `λ`); the belief update
uses the pure von Mises feedback likelihood, matching `online_learner` so the
two learning models stay point-for-point comparable.

## Results (subjects 1 & 3)

**AIC (lower = better):**

| model | params | subj 1 | subj 3 |
|---|---|---|---|
| static (fair) | 9 | 77072.1 | **83474.8** |
| online (fair) | 6 | 77130.5 | 83484.6 |
| asymptote+transient | 11 | 77251.6 | 83488.3 |
| integration, free α | 7 | **76937.8** | 84305.0 |
| **integration, derived α** | **6** | 77023.2 | 83935.2 |

Three things to take from this:

1. **Derived-α beats fair static and online on subject 1** (77023 vs 77072 /
   77131) — and does it with the **fewest parameters of any model (6)** and
   **no free α**. So the subject-1 integration win is *not* an artifact of the
   extra α knob or of α running to its boundary; a principled,
   reliability-weighted mixture with online learning genuinely fits this subject
   better than the switch family.

2. **The free-α model's extra AIC edge on subject 1 (~85) buys almost nothing
   scientifically.** Free-α wins by shedding the mixture (α→1, plain
   integration); derived-α keeps the mechanism and pays one parameter less. For a
   *mechanistic* claim, derived-α is the stronger result even though its raw AIC
   is slightly higher.

3. **Subject 3 still prefers the switch family** (static 83475 vs derived-α
   83935). Consistent with everything else: subject 3 is switch-like, subject 1
   is integration-like. The 1–1 split holds.

## The discriminator (right panel of the figure)

The reason this matters beyond AIC: derived-α makes a **falsifiable behavioural
prediction the switch model does not**. Its prior reliance *declines* with how
far the stimulus is from the prior mean (window mass 0.59 → 0.44 over 30–120°),
because a far-off measurement is poorly explained by the peaked component. The
switch model's Eq. 6 weight is **flat** across direction (~0.27 → 0.24 at fixed
coherence/belief). So the two models — which the paper says are "reformulations"
of each other — are in fact **distinguishable from the data**, and the test is a
direction-resolved analysis of prior bias, no new fitting required.

This is the sharpest thing derived-α gives you: it implements the paper's own
conjecture *and* exposes a place where the conjecture is testable rather than a
mere relabeling.

## Caveats

- Subjects 1 & 3 only (1–1 split); single-fit AICs, noise ~tens of AIC. Needs
  the all-12 fair batch before any of this is a finding.
- `k_e[0.24]` runs to a huge value on subject 1 again (the motor-noise ridge) —
  same unidentifiability as every other model; report as motor-limited.
- Read-out is still MAP-per-κ-then-average (the shared H4/M4 convention), not a
  single-posterior marginalisation. Deliberate, for comparability; noted.

## Files
- `hb_integration_derived.py` — the model (subclass of `HBIntegrationObserver`;
  run it directly for the built-in discriminator smoke test).
- `derived_alpha_fit_results.json` — subject 1 & 3 fits.
- `derived_alpha_comparison.png` — AIC per subject + the discriminator.
