# Critical review — fitting & cross-validation strategy vs. the project abstract

**Repo:** `NeuroMatch_2026_Behaviour/hierarchical` (branch `model-verification`)
**Scope of this review:** the parameter-fitting and model-comparison machinery
(`observers/fitting/*`, `observers/comparison/{fit_batch,cross_validate,recovery,shape_analysis}.py`,
`registry.py`), judged against what the abstract actually claims.
**Not re-run here** — this is a methods audit of the existing code and the fits already on disk
(12 subjects, 8 models fit, 5 with CV).

---

## 0. What the abstract commits the methods to prove

The abstract makes one framing claim and two concrete, testable predictions:

- **Framing:** a Bayesian observer that *learns how much to trust its prior, trial by trial*
  can reproduce bimodality previously attributed to an explicit prior-vs-likelihood **switch** —
  i.e. reframe a discrete strategy as **graded** integration.
- **Prediction (i):** the HB model reproduces **both unimodal and bimodal** estimate distributions.
- **Prediction (ii):** it **recovers block-specific prior widths that change through learning**,
  giving a principled account of how prior-confidence evolves.
- **Comparison protocol stated in the abstract:** compare HB vs Switch by **penalized likelihood
  (AIC/BIC)** and by **ability to reproduce individual response distributions**.

The methods must therefore (a) put HB and Switch on a genuinely like-for-like footing for AIC/BIC,
(b) test distributional shape per subject, and (c) validate the *learned-width trajectory* — the
mechanism, not just the fit index. The review is organized around whether each is actually delivered.

---

## 1. Headline finding: the comparison is not yet decidable, and the current numbers cut *against* the abstract

Reading the fits on disk (`results/fits/comparison/` + `comparison_cv/`), HB-Adaptive (k=6) vs
Switch (k=9):

| metric | result on the 12 fitted subjects |
|---|---|
| **AIC** | Switch lower (better) in **10/12**; HB better in 2/12 |
| **BIC** | Switch lower in 10/12 (same split) |
| **CV per-trial NLL** | HB better in 7/12, but by tiny margins (≈0.001–0.01 nats/trial), except subject 5 |
| **Switch convergence** | hit the Nelder–Mead iteration cap in **10/12**; multi-start NLL spread up to **677 nats** |
| **HB-Adaptive convergence** | hit the cap in **1/12**; median start-spread ≈ 66 nats |

Two things follow, and they pull in opposite directions:

1. On the abstract's own stated metric (AIC/BIC), the current evidence **favours the Switch**, not
   the graded-Bayesian account. That is a substantive result the write-up has to confront, not a bug.
2. **But the comparison is not clean**, because the Switch is systematically *under-converged*
   (§2). Its reported NLL is an upper bound; a properly converged Switch would very likely widen its
   AIC lead. So the honest current status is: *the head-to-head cannot be adjudicated until both
   models are converged to the same standard.* Neither "HB wins" nor "Switch wins" is defensible yet.

The pipeline deserves credit for making this visible: `fit_batch.py` records `start_spread` and the
scipy convergence block on every fit precisely so a win "can never be quietly attributed to one model
being under-converged." The instrumentation is right; the fits shipped before it was acted on.

---

## 2. Fitting strategy

### 2.1 The convergence standard, not the algorithm, is the threat
**Nelder–Mead is paper-faithful** — Laquitaine & Gardner (2018) fit every observer with Nelder–Mead
(Methods, "We used the Nelder-mead algorithm to search for the model parameters that minimized the
negative sum of the log probability..."), so the repo's choice of optimizer is *correct* and should
be kept. The problem is that the repo runs the paper's algorithm but **not to the paper's convergence
standard, and skips the paper's own fallback for the hard case.** Three specifics from the paper's
Methods:

1. **Tolerance.** The paper accepted fits only when parameters stopped changing above "the very strict
   function and parameter tolerances of **1e-4**," and explicitly reports that "more relaxed tolerance
   worsened the fit." The repo uses `xatol=fatol=1e-2` — 100× looser than the value the paper found
   was already the loosest acceptable.
2. **Reaching tolerance at all.** More important than the tolerance value: the 9-parameter Switch
   **hits `maxiter=400` before satisfying even the loose 1e-2 criterion** in 10/12 subjects, so it
   never meets any convergence test. HB-Adaptive (k=6) hits the cap in 1/12. The two models are thus
   held to *different effective convergence standards*, which contaminates AIC, BIC and CV at once.
3. **The paper's fallback is exactly for this model.** The paper adds that results hold under CMA-ES,
   which "can fit the data better than Nelder-Mead when the objective function space is noisy and
   ill-conditioned, that is when the negative log likelihood has many local maxima" — and that
   paragraph is introducing the **nine-parameter switching observer**. So the paper reached for CMA-ES
   on precisely the ill-conditioned k=9 model that is blowing up in the repo.

**Caveat on the evidence** (per the known `k_e[0.24]` identifiability ridge): *hitting maxiter* is by
itself weak evidence of under-convergence, because a flat ridge can leave Nelder–Mead reporting
non-convergence while the NLL is already stable — judge convergence by NLL stability across a maxiter
increase, not the boolean flag. The load-bearing evidence here is instead the **~677-nat spread in
best-NLL across starts** for the Switch: different starts land at genuinely different likelihoods,
which a same-NLL/different-params ridge does not explain. *That* is what says the Switch point fit is
not yet at the optimum.

**Fix (priority 1) — keep Nelder–Mead, meet the paper's bar:**
(a) raise `maxiter` enough that the Switch *reaches* its tolerance instead of the cap, and judge
convergence by NLL stability across a maxiter increase (not the scipy boolean);
(b) tighten tolerances toward the paper's 1e-4, or at least verify the 1e-2 optimum is stable under
tightening;
(c) for the Switch specifically, add the paper's **CMA-ES** fallback (`pip install cma`) and confirm
it lands at the same or lower NLL — this is what the paper did for the ill-conditioned 9-parameter fit.
Re-fit both models to the same converged standard before reporting any AIC/BIC.

### 2.2 Multi-start is applied unevenly, and specifically shortchanges the model that needs it
`fair_refit.py` and `registry.multistart` give the reported point fits several starts (N_STARTS=10,
"matches Laquitaine & Gardner's ~10 starting points"). Good. But the Switch's own start-spreads
(up to 677 nats) show 10 jittered starts are **not enough to tame its 9-D landscape** — the point fit
is still start-dependent. The HB model, by contrast, is nearly start-independent (median spread 66,
and its bespoke 3-start scheme suffices). So the model that most needs global search is the one still
under-searched. Increase starts *and* fix the optimizer together; one without the other won't close it.

### 2.3 The learning models learn the prior from the veridical direction — a modeling idealization, NOT a statistical leak
`HBAdaptiveConfidenceObserver` learns its prior-confidence α (and width κ) by treating **the revealed
true motion direction as one draw from the prior** — `feedback` defaults to `directions` (the
`motion_direction` column) in every `filter()` call and in `_trial_logliks`. The true directions ARE the
per-block draws from the prior, so this is a coherent way to learn the prior. **Two clarifications that
correct an earlier, stronger version of this critique:**

1. **The timing is causally clean — there is no leakage into the fitted likelihood.** In the trial loop,
   the response distribution for trial *t* is computed from the belief *before* trial *t*'s feedback is
   applied (`dist = estimate_distribution(...); ...; belief = update_belief(belief, feedback[t])`). Trial
   *t*'s true direction shapes only trials *t+1* onward. So the point-fit NLL is a proper one-step-ahead
   predictive likelihood — the response is never predicted using its own trial's feedback. Calling this
   "leakage" (as an earlier draft did) was wrong: leakage means the dependent variable (the subject's
   *estimate*) informs its own prediction, and that never happens — the belief is driven only by the
   exogenous stimulus direction, and always post-response. (The true direction is of course also the
   *stimulus* inside `estimate_distribution`; that is correct and universal — every model, incl. the
   Switch, centres its likelihood on it.)

2. **The real concern is mechanistic plausibility, not validity.** The participant received **no
   trial-wise correctness feedback** (the paper explicitly withheld reward feedback so as not to
   interfere with prior learning), so a real observer could only infer the prior from its **own noisy
   percept** — very noisy at 6% coherence. The model instead infers it from the *veridical* direction,
   which yields a cleaner, faster-converging width trajectory than any real observer could achieve. This
   is a legitimate **idealization to flag**, not a bug: it means the learned-width claim is tested under a
   best-case learning signal. A stricter version would drive the belief from the observer's own
   percept/estimate rather than `motion_direction`; worth reporting as a robustness check, but it does
   **not** invalidate the fits. Its main consequence is for CV *power* (§3.2), not for the point-fit NLL.

---

## 3. Cross-validation strategy

### 3.1 Block-fold design is the right idea, but doesn't isolate the abstract's learning claim
`cross_validate._block_folds` builds contiguous folds aligned to prior-width block boundaries — good
reasoning: hold out whole blocks so learning is tested out-of-sample rather than interpolated. **But
the real design interleaves the four prior widths** (~33–36 short blocks cycling through all of
{80,40,20,10}° across 8 sessions; verified from the CSV), so each contiguous 5-fold chunk spans
*multiple* widths and every width has already been presented many times before any held-out fold.
Holding out one contiguous time-window therefore tests general predictive fit, **not** the abstract's
specific claim that the model *forecasts* block-specific widths as they change. To test that claim you
want either (a) a *forward-chaining* / expanding-window split (train on early blocks, forecast later
ones) or (b) a direct trajectory validation (§4).

### 3.2 CV tests parameter generalization, not forecasting — a power limit, not a leak
The belief filter runs over the **full ordered sequence including held-out trials**, so the belief that
predicts a held-out trial was updated using the true directions of earlier trials, some of which are
themselves held-out. An earlier draft of this review called that "asymmetric feedback leakage biasing
toward the abstract's model." **That was wrong on both counts, and the correction matters:**

- **It is not leakage.** The quantity being scored is the subject's *estimate*; that estimate is never
  used to predict itself or any other trial. The belief is driven only by the **exogenous stimulus
  direction** (feedback = `motion_direction`), which is an input, not the dependent variable. Held-out
  *responses* never inform any prediction. So the held-out NLL is a legitimate measure of how well the
  **train-fitted parameters** predict held-out responses.
- **It is not asymmetric in a way that favours the learning model.** Both models are handed the full
  stimulus stream (direction/coherence/prior_std) for every trial — the Switch needs it to predict too.
  Neither model sees held-out responses. There is no differential access to the dependent variable, hence
  no structural tilt toward the learning hypothesis. Withdrawn.

The **real** limitation is one of *power*, and it overlaps with §3.1: because the latent belief evolves
from the exogenous stimulus sequence rather than from fitted quantities, running it over the whole
sequence means CV is scoring *parameter* generalization (do `k_like`, `lam`, … fit on train predict
held-out responses?) and **not** the model's ability to *forecast* a not-yet-seen block. For the
abstract's learning claim you want the latter, which needs a **forward-chaining / expanding-window**
split (train on early blocks, predict later ones) — the belief legitimately runs forward through the
held-out tail there, and that version *does* test learning. Current block-fold CV is a sound
generalization check; it just isn't a test of the learning dynamics per se.

### 3.3 Single-start per-fold fits make CV unreliable for exactly the model that needs multistart
`_starts_for(mask)` uses **1 start inside CV folds** (multistart only for the reported point fit). The
docstring argues per-fold global search is unnecessary. That reasoning holds for the near-convex HB
model but is **false for the Switch**, whose point fit is demonstrably start-dependent. The fingerprint
is subject 5: Switch has *better* AIC (61981 < 62886) yet *worse* CV per-trial (6.267 vs 5.451), with a
point-fit spread of 677. In-sample-better/out-of-sample-worse normally reads as overfitting — but here
it is almost certainly a **failed single-start CV fold fit**, not genuine generalization failure. So the
CV numbers for the Switch are not trustworthy, and the "HB better on CV in 7/12" tally is partly measuring
optimizer luck per fold, not predictive fit. **Fix:** multi-start the CV fold fits for the poorly-conditioned
models (or fix the optimizer per §2.1), then recompute.

---

## 4. The abstract's two predictions — instrumentation gap

| abstract prediction | tested? | by what |
|---|---|---|
| (i) reproduce **both unimodal and bimodal** distributions | **Yes — well covered** | `shape_analysis.py`: per-cell TV distance (coherence × width), far-band prior-cluster mass, per-subject bimodality flag. This is the right qualitative test and targets the core claim directly. |
| (ii) **recover block-specific prior widths that change through learning** | **No — not tested on real data** | Nothing validates the model's *learned width/α trajectory* against the known block widths on human data. `recovery.py` checks parameter/model recovery on **synthetic** data only. |

Prediction (ii) is the abstract's headline *mechanistic* claim ("a principled account of how confidence
in the prior evolves"), and it currently rests only on fit indices, not on any direct test that the
learned latent tracks the true block widths. The dataset supports this test — block widths are known per
trial — so the natural experiment is: run the fitted observer, extract E[κ_t] / E[α_t], and check it
steps toward each block's true width when the block changes. `dataset.make_synthetic_design` even returns
the ground-truth belief-SD trajectory, so the synthetic half of this is already scaffolded; the real-data
half is missing. **This is the single most important gap for the abstract's story.**

---

## 5. Identifiability / recovery

- **Model recovery** (`recovery.py`) implements the right structure — simulate from each generator, fit
  *all* models, tally AIC/BIC winner into a confusion matrix — and correctly frames it as the check that
  "makes the comparison trustworthy." Two weaknesses:
  - Default **`n_sim=2`** per generator is far too few to estimate off-diagonal confusion rates; a
    2×2 or 8×8 confusion matrix built from 2 draws per row is noise. Needs ≥20–50 sims per generator.
  - It is only wired to run on the default 2 models unless invoked with `--models`; the confusion matrix
    that licenses the *8-model* registry has not been established at adequate n.
- **Parameter recovery** for HB-Adaptive uses a **35% relative-error tolerance** and **explicitly exempts
  `lam` and `p_random` from the pass criterion** (`ok = ok and (k in ("lam","p_random") or rel<0.35)`).
  But `lam` is the **volatility / learning-rate** parameter — the quantity that operationalizes
  "trial-by-trial" learning. Allowing it to be unrecoverable means the mechanism the abstract is selling
  may not be identifiable from the data. Combined with the documented **κ–α ridge** (a wide prior and a
  low-confidence prior look alike from a single feedback draw), this is a real identifiability threat:
  the learned latents whose trajectory the abstract wants to interpret may be only weakly constrained.
  Recovery of `lam` (and the κ/α trade-off) must be demonstrated, not exempted, before interpreting the
  learning dynamics.

---

## 6. Smaller / lower-priority points

- **BIC's sample size** uses n = trial count (~6000–8500). Trials within a subject are autocorrelated
  (strongly so for a learning model), so effective n ≪ raw n and the `k·ln n` penalty is inflated. This
  over-penalizes the higher-k model — i.e. it *helps* the k=6 HB against the k=9 Switch, a small tilt
  toward the abstract. Worth a sentence; not decisive.
- **Grid comparability** is handled correctly — all models scored on the same 360° grid, noted explicitly
  in `fit_batch`, so NLLs are directly comparable with no resolution artifact. Good.
- **Per-subject (not hierarchical) fitting** is appropriate for the "inter-subject variability" claim.
  Note the "hierarchical" in the abstract refers to the *within-model prior hierarchy*, not a
  cross-subject random-effects fit — no partial pooling is done, and none is required by the abstract.
  Just don't let a reader conflate the two.
- **Fit/score object identity for the Switch:** `switching_observer_fit` optimizes via `fit_static`
  (which uses an `OnlineHierarchicalObserver` proxy) then rebuilds a `SwitchingObserver` from θ. Worth a
  one-line unit check that the NLL of the rebuilt `SwitchingObserver` equals the optimizer's reported NLL,
  so the object that is *scored* is exactly the object that was *fit*.

---

## 7. Prioritized recommendations

1. **Converge the Switch to HB's standard before reporting any AIC/BIC.** Robust optimizer and/or
   ≥2000 iterations + more starts; drive `hit_maxiter` and start-spread down to HB's level. Until then the
   head-to-head is undecidable — and note that the *current* numbers favour the Switch, so this is not a
   formality.
2. **Report the learning-signal idealization (optional robustness check, not a fix).** The belief is
   updated from the veridical `motion_direction` — causally clean (post-response) but a best-case
   learning signal the participant never had (no correctness feedback in the task). Optionally add a
   variant that drives the belief from the observer's own percept/estimate and confirm the width claim
   survives. This is a plausibility check on the mechanism, not a correction to the likelihood.
3. **Add the learned-width trajectory validation on real data** (abstract prediction ii): extract E[κ_t]/E[α_t]
   from the fitted observer and test it tracks the known per-block widths. This is the missing headline test.
4. **Re-scope CV to test learning:** current block-fold CV is a sound *parameter-generalization* check
   (no leakage — it does not need de-biasing), but it does not test forecasting. Add a **forward-chaining /
   expanding-window** split (train early blocks, predict later ones) to test the learning dynamics, and
   multi-start the fold fits for ill-conditioned models.
5. **Strengthen recovery:** n_sim ≥ 20–50 for the confusion matrix across the full registry; demonstrate
   (don't exempt) recovery of `lam` and map the κ–α ridge before interpreting learned latents.
6. Keep leading with **CV as the abstract intends** *only after* 1 and 4 — otherwise lead with AIC/BIC on
   converged fits, and report shape reproduction (§4-i) as the qualitative pillar, which is already sound.

## 8. What is already done well
- Uniform registry so every model is fit/scored/plotted through one interface — the comparison is
  structurally like-for-like by construction.
- Convergence diagnostics (`start_spread`, scipy status) recorded per fit — the under-convergence was
  *detectable* from the outputs, which is why this review could find it.
- Same optimizer budget and same 360° grid across models; resumable, one-JSON-per-model×subject layout.
- Distribution-shape test (`shape_analysis`) is a faithful instrument for the abstract's prediction (i).
- Caveats (belief runs over full sequence, κ–α ridge) are documented in-code rather than hidden — the honesty is real;
  the issue is that some documented caveats are load-bearing for the main claim and need fixing, not just noting.
