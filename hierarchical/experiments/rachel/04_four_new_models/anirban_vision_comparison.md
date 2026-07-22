# Anirban's modelling folder vs. the abstract and our built models

Reviewer: Switching Observer specialist. Scope: `hierarchical/anirban-modelling/`
— the original design vision that inspired the abstract — compared against (1)
the project abstract and (2) the six observer models we built and fit this
session.

## What is actually in Anirban's folder

It is a **design-and-scaffold package, not fitted models.** Concretely:

- **`Model_explainer.md`** — a careful reverse-engineering of the original
  Laquitaine MATLAB code (`projInference-gh-pages/`, the real repo, included):
  what the code implements (one Bayesian estimator with MAP / BLS / Sampling
  readouts), what is *missing* (no sampling file `SLBayesSamplingLookupTable.m`;
  **no online/sequential model anywhere**), and a MATLAB→Python porting guide.
- **`Hypotheses_critique_and_alternatives.md`** — the intellectual core. States
  two hypotheses precisely, grounds them in the paper's Discussion and recent
  literature (causal inference, changepoint detection, HGF, sampling), critiques
  each on identifiability, and lays out a model-comparison plan. Also digests two
  **prior Neuromatch decks** (Deck A: online Bayesian prior-learning; Deck B:
  logistic-mixture switching) as precedents.
- **`Switching_Bayesian_Observer_starter.ipynb`** — a runnable Python port:
  `vm_pdf`, `girshick_lookup` (MAP + BLS), `sampling_lookup`, `trial_loglike`,
  data loading, bias-curve reproduction, and an **online-prior scaffold** (§8,
  `online_prior_stats`, a delta rule) that is explicitly a stub to be replaced.
- PDFs, slide decks, the reference papers, and a copy of the trial CSV.

So: strong specification and a working single-trial estimator, but the
learning models and the fits are left as "new work" — which is exactly the work
we did this session.

---

## Comparison 1 — Does Anirban's vision match the abstract?

**Yes, closely — Anirban's folder *is* the design document the abstract was
written from.** The two hypotheses in `Hypotheses_critique_and_alternatives.md`
map directly onto the abstract's proposal:

| abstract element | Anirban's document |
|---|---|
| mixed hyper-prior (peaked von Mises + uniform), mixing set by prior/likelihood strengths | **Hypothesis 1** — verbatim the paper's Discussion reformulation; the "peaked OR uniform per trial" latent-cause mixture |
| online / trial-by-trial learning of prior confidence | **Hypothesis 2** — Switching-posterior observer driven by an *online* running estimate of the prior, contrasted against the paper's exponential-τ sequential model |
| fit per subject by max-likelihood; validate by parameter recovery; compare to Switching via NLL/AIC/BIC/CV | the document's **model-comparison plan** and "validate every new model by parameter recovery" |

**One important refinement Anirban's document adds that the abstract glosses.**
The abstract treats the mixed hyper-prior as the novelty. Anirban's critique
correctly flags that this is only novel **if the mixing weight is *inferred*, not
hard-coded** — because with the weight fixed to the ratio `k_prior/(k_prior+k_e)`
the mixture model *is* the Switching observer written differently ("will fit
identically and yield no new prediction"). This is exactly the point that drove
our reframe this session, and Anirban reached it independently and earlier. His
document also notes the paper *already tested* a von-Mises-plus-uniform prior and
Switching beat it — the same "don't reproduce a result the authors already have"
caution.

**So the abstract is a faithful — if compressed — distillation of Anirban's
vision.** Where they differ, Anirban's document is *more* careful: it names the
identifiability trap the abstract walks into, and it proposes the escape (infer
the weight / parameterise it by covariates) that the abstract does not mention.

---

## Comparison 2 — Anirban's vision vs. the models we built

Anirban lays out a **menu** of models. Here is which ones we actually built and
fit, which we built in a stronger form than he specified, and which we have not
touched.

### Direct matches — we built and fit these

| Anirban's model | our model | match quality |
|---|---|---|
| **Switching observer** (MAP readout, fixed `p_prior=k_prior/(k_prior+k_e)`) | **static switching** (`switching_observer.py`) | **exact** — same Eq. 6/7, same 9 params, MAP-of-bimodal-posterior switch |
| **H1 fixed-ratio mixture** ("identity check against Switching") | **hb_integration derived-α** (`α=κ/(κ+k_e)`) | **exact in spirit** — this *is* the fixed-ratio mixture; we confirmed it reproduces the switch, the identity check he prescribes |
| **H1 with a *free* mixing weight** | **hb_integration free-α** (α fitted, 0.29–0.97 across subjects) | **strong** — our free-α is the "let the weight float" version; fitted, per-subject |
| **H2 online prior learning** (delta-rule / leaky integrator, learning rate α) | **online_learner** (`online_learner.py`, belief `b_t(κ)`, forget λ) | **strong** — his §8 scaffold, built out into a full Bayesian filter over κ and fit |
| **H2 stronger version: Bayesian online *changepoint* detection** (Adams & MacKay; resets learning at block boundaries) | **adaptive_volatility_switching** (CPP-driven adaptive learning rate, single hazard) | **strong** — this is precisely the changepoint model he flags as "the normatively-matched online model"; we built and fit it |
| paper's exponential-τ **Prior Learning Model** (`k_prior·(1−e^{−t/τ})`) | **asymptote_transient** (`k_eff(t)=k_asym+(k_start−k_asym)e^{−t/τ}`) | **close** — same exponential-transient idea, generalised (per-block asymptotes, asymmetric τ, carryover) |

**On his central methodological demand — "infer the weight / learn the prior
online, and compare on NLL/AIC/BIC/CV with parameter recovery" — we delivered:**
all six models fit to all 12 subjects, AIC/BIC comparison, and 5-fold
sequence-preserving cross-validation. That is the model-comparison plan his
document specifies, executed.

### Where we went *beyond* his spec

- **The changepoint model.** He names Adams & MacKay changepoint detection as
  the "stronger version worth adopting" but leaves it as a recommendation. We
  built it (`adaptive_volatility_switching.py`), verified it reduces to the online
  learner at hazard→0, and found it competitive (beats AT in-sample on the 2-subject
  set, wins CV on subject 3).
- **Cross-validation as the arbiter.** His plan lists CV; we ran it and it
  *changed the verdict* — AT wins in-sample AIC on 7/12 but only 6/12 out-of-sample,
  with 4 AIC→CV flips exposing overfitting the in-sample criteria missed. That is a
  result his design could not have anticipated without the fits.

### Where his vision goes beyond what we built — the genuine gaps

These are models in his menu we have **not** built, and they are worth knowing:

1. **The inferred-responsibility (causal-inference) H1.** This is his *preferred*
   version of H1, and we did **not** build it. Our derived-α model sets the weight
   to the reliability ratio `κ/(κ+k_e)`, which depends only on coherence and prior
   width. His causal-inference version sets it to the **per-measurement posterior**
   `p(z=peaked | m)`, which *also depends on where the measurement falls* (near vs.
   far from 225°). That predicts the switch rate varies **with the displayed
   direction**, not just the condition — a falsifiable signature neither our switch
   nor our derived-α model produces. **This is the single most valuable unbuilt
   model in his folder**, and it connects to our unrun C2 late-block/within-trial
   bimodality test.
2. **The logistic-covariate mixture (Deck B).** Mixing weight = fitted logistic of
   prior_std, coherence, their interaction, recent error, and *cumulative prior
   reliance*. This is a data-driven H1 that can capture the **non-monotonic
   prior_std effect** and **hysteresis** Deck B found — two empirical targets he
   says any winning model must reproduce, and which our concentration-ratio models
   *cannot* express (they are monotonic in prior width by construction).
3. **The bimodal-likelihood control (Chetverikov & Jehee 2023).** A Basic Bayesian
   observer with a *bimodal sensory likelihood* — bimodality from multiplicative
   integration of velocity + orientation-streak cues, **no switch at all**. He
   correctly calls this "the strongest current competitor." We have not built it,
   and it is the model that could undercut the whole switching interpretation.
4. **The finite-sample readout** (draw *n* posterior samples; n→∞ = Basic Bayesian,
   n=1 ≈ switch). A resource-rational nest that turns "how many samples" into a
   fitted question. Not built.

### One conceptual difference in the H2 we built

Anirban's H2 is a **Switching-posterior** observer: form the full posterior, then
switch between the *posterior mode* and the *prior mean*. Our online/AT/adaptive
models switch between the *sensory read-out* and the prior spike (the paper's
plain Switching observer with a learned prior), not the Switching-*posterior*
variant. Behaviourally close, but if you want to match his spec exactly, the
readout should be the posterior mode, not the raw sensory percept.

---

## Bottom line

1. **Abstract vs. Anirban: a faithful match.** The abstract is a compressed
   version of Anirban's two-hypothesis design; Anirban's document is the more
   rigorous source and already contains the identifiability caveat that reshaped
   the project.
2. **Our models vs. Anirban: we built and fit the *backbone* of his menu** — the
   switch, the fixed-ratio mixture (derived-α), the free-weight mixture (free-α),
   the online learner, the changepoint learner, and the exponential-transient
   learner — across all 12 subjects with AIC/BIC/CV. That covers Hypotheses 1
   (fixed-ratio and free) and 2 (online + changepoint + transient) as he framed
   them.
3. **The four things he specified that we had not built — now built.** As of this
   session all four are implemented, verified, and smoke-fit in the `observers/`
   package (see `docs/new_models_manifest.md`, `docs/new_models_build_report.md`):
   the **inferred-responsibility causal-inference H1** (`CausalInferenceObserver`,
   7p — stimulus-dependent switch rate confirmed: prior-window mass declines
   0.525→0.311 across displayed direction while the fixed-ratio analogue is flat),
   the **logistic-covariate mixture** (`LogisticMixtureObserver`, 11p — non-monotonic
   prior_std dependence and hysteresis both verified), the **bimodal-likelihood
   control** (`BimodalLikelihoodObserver`, 10p — g=1 reduces to Girshick exactly;
   two lobes 180° apart with no switch), and the **finite-sample readout**
   (`FiniteSampleObserver`, 10p — n=1 reproduces the static switch exactly, n→∞ the
   posterior mean; subject-1 smoke-fit gives an interior n≈3.4). 22/22 verification
   checks pass. **Still deferred (compute-bound):** the fair multi-start all-12-subject
   fits that would place these four against the existing six on AIC/BIC/CV, and a
   multi-start for the logistic model (its cold single-start settles at the flat-weight
   basin). The causal-inference H1 remains the one that would most sharply test
   whether a switch is really needed.

## Files reviewed
- `anirban-modelling/Model_explainer.md`, `Hypotheses_critique_and_alternatives.md`
- `anirban-modelling/Switching_Bayesian_Observer_starter.ipynb`
- `anirban-modelling/projInference-gh-pages/` (original Laquitaine MATLAB)
- our models: `switching_observer.py`, `online_learner.py`, `asymptote_transient.py`,
  `hb_integration.py`, `hb_integration_derived.py`, `adaptive_volatility_switching.py`
