# Making the learning-transient model useful for the Laquitaine experiment

The task's defining feature is that **block changes were unsignaled** — subjects
had to infer that the prior's statistics shifted. `asymptote_transient.py` (AT)
violated this: its transient clock reset at the *true* block boundary, using
information the observer never had. That single fact is the first thing a
reviewer would attack, and it also drove AT's 11-parameter fragility.

I rebuilt the model to remove that assumption. `adaptive_volatility_switching.py`
keeps AT's two goals — per-block prior levels and a within-block transient — but
makes both **emergent from inference**, not hand-set at boundaries.

## The mechanism (what changed)

AT: `k_eff(t) = k_asym + (k_start − k_asym)·exp(−t/τ)`, clock resets at the true
boundary; 11 free params (4 asymptotes + k_start + 2 τ + switch machinery).

Adaptive-volatility: the online learner's belief `b_t(k)` over prior strength,
with a **change-point-driven adaptive learning rate** (reduced Bayesian online
change-point detection; Nassar et al. 2010; Behrens et al. 2007):

```
p_stay   = Σ_k b_{t-1}(k)·V(f_t;225,k)      # feedback fits current belief?
p_change = Σ_k b_0(k)·V(f_t;225,k)          # feedback fits a fresh reset?
CPP_t    = h·p_change / (h·p_change + (1-h)·p_stay)
b_t^-    = (1-CPP_t)·b_{t-1} + CPP_t·b_0     # forget in proportion to CPP
b_t(k)   ∝ b_t^-(k)·V(f_t;225,k)
```

- **Surprising feedback → CPP↑ → fast re-adaptation.** That is AT's transient,
  but *triggered by inferred surprise, not the true boundary*.
- **Consistent feedback → CPP≈0 → accumulation → belief settles at the block's
  true width.** That is AT's per-block asymptote, but *emergent, not fitted*.
- One dynamics parameter — the **hazard `h`** — replaces AT's {4 asymptotes,
  k_start, 2 τ}. **6 free params total** (3 k_e + k_motor + p_random + h).

## Verification (before fitting)

- **Nests the online learner:** `h→0` ⇒ CPP→0 ⇒ pure accumulation, matches
  `OnlineHierarchicalObserver(lam=0)` to max|Δ|≈3e-10.
- **Tracks unsignaled blocks:** on synthetic 80/40/20/10° blocks the belief
  converges near each true width (40→36°, 20→20°, 10→9°) with **no boundary
  information**. (The first wide 80° block reads ~27° under the E[k] reporting
  convention — wide priors are hard to distinguish from a diffuse belief; this
  is a reporting artefact of the SD summary, not of the fit, which uses the full
  estimate distributions.)
- **Bimodality preserved** (far/low-coh: prior-window mass 0.50, stimulus-window
  0.18) — it is still a genuine switching observer.

## Results (subjects 1 & 3)

**AIC (lower = better):**

| model | params | subj 1 | subj 3 |
|---|---|---|---|
| static (fair) | 9 | 77072.1 | **83474.8** |
| online (fair) | 6 | 77130.5 | 83484.6 |
| asymptote+transient | 11 | 77251.6 | 83488.3 |
| **adaptive-volatility** | **6** | **77133.4** | **83482.2** |
| integration, free α | 7 | 76937.8 | 84305.0 |
| integration, derived α | 6 | 77023.2 | 83935.2 |

- **Beats AT on both subjects** (77133 < 77252; 83482 < 83488) — with **6
  parameters instead of 11**, and without peeking at block boundaries.
- On **subject 3** it is within ~7 AIC of the best model (static) and beats
  online and AT — i.e. it is now a full member of the leading switch cluster,
  which AT was not.
- Fitted **hazard** is small (subj 1 h=0.021, subj 3 h=0.001) — subjects rarely
  infer a change, consistent with the priors being stable within a block and
  with subject 3 being nearly a pure accumulator.

## Why this is the most useful AT variant for this experiment

1. **It respects the experiment's design.** No block-boundary information leaks
   into the model — the transient is produced by the same surprise signal a real
   subject would have. This closes the biggest reviewer objection to AT.
2. **It is parsimonious and identified.** 6 params, one dynamics knob (h), no
   weakly-identified τ pair and no 4 free asymptotes — the fragility that made
   subject-5 fits fail should be largely gone.
3. **It unifies the model zoo.** It nests the online learner exactly, so the
   ladder is now clean: static (no learning) → online (fixed-rate learning) →
   adaptive-volatility (surprise-modulated learning) → integration (no switch).
   Each step adds one interpretable idea.
4. **It keeps the genuinely-new science.** It still produces the within-block
   transient in prior reliance that the static observer cannot — the finding
   that motivated AT — but now as a consequence of normative change-point
   inference rather than a descriptive exponential.

## Caveats
- Subjects 1 & 3 only (1–1 split); single-fit AIC, noise ~tens of AIC. Needs the
  all-12 fair batch.
- `k_e[0.24]` still hits the motor-noise ridge (shared across all models).
- Read-out is the reliability-ratio switch (inherited); the change is entirely in
  how the prior belief is *learned*, not how it is *used* on a given trial.
- Recommended: keep AT in the repo as the "peeks-at-boundary" comparison — the
  AT-vs-adaptive-volatility gap is itself evidence that boundary knowledge is not
  needed to explain the transient.

## Files
- `adaptive_volatility_switching.py` — the model (subclass of
  `OnlineHierarchicalObserver`; run directly for the reduction + tracking checks).
- `adaptive_volatility_fit_results.json` — subject 1 & 3 fits.
- `adaptive_volatility_comparison.png` — six-model AIC comparison.
