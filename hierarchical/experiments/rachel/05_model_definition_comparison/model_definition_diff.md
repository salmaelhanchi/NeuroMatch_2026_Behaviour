# Do the root notebook models and the hierarchical/ models compute the same thing?

**Short answer: No — not for either model.** In both cases the two versions
share the same building blocks (von Mises priors/likelihoods, a Girshick-style
MAP readout) but assemble them into a **different estimate distribution**. This
was tested by *running* each definition on identical trial inputs and comparing
the output distributions — not by reading the code.

This note is only about the **model definition** — i.e. the function
`P(estimate | stimulus, condition)`. Fitting, notebooks, and packaging are out
of scope here.

| role | root notebook (initial) | hierarchical/ package |
|---|---|---|
| switching | `Switching_Bayesian_Observer_starter.ipynb` → `girshick_lookup` | `observers/models/switching_observer.py` |
| HB | `hb_verified_model_implementation.ipynb` → `build_trial_components` | `observers/models/hb_integration.py` |

The test: feed both the same conditions (coherence, prior width, true
direction), read out the estimate distribution with motor noise and lapse
switched off (so only the *core model* is compared), and measure where the mass
lands.

---

## 1. Switching — these are TWO DIFFERENT MODELS

The notebook's `girshick_lookup` and the package's `SwitchingObserver` are
**not the same model**, despite both being called "switching".

**Evidence — sweep the stimulus, watch the peak of the estimate distribution**
(prior mode = 225°, mid reliability):

| true dir | notebook peak | package peak |
|---:|---:|---:|
| 90  | 150 | 225 |
| 120 | 169 | 225 |
| 150 | 185 | 225 |
| 180 | 201 | 225 |
| 210 | 217 | 225 |
| 225 | 224 | 225 |

- **Notebook = INTEGRATION.** Its peak moves *smoothly* with the stimulus and
  always sits *between* the evidence and the prior. That is what a Bayesian
  observer that multiplies likelihood × peaked-prior and takes the MAP does — a
  single, attracted compromise estimate. There is no "switch": the notebook's
  `girshick_lookup` builds one posterior and reads out its mode.
- **Package = TRUE SWITCH.** Its estimate mass sits *either* at the sensory
  evidence *or* exactly at the prior mode (225°), never at a compromise
  location — the peak is pinned at 225 across the whole sweep because on these
  conditions the prior arm dominates. This is the paper's Eq. 6 mechanism: a
  reliability-weighted **mixture** of an evidence read-out and a delta at the
  prior mode.

Numerically, on matched trials the two distributions have a total-variation
distance of **0.20–0.90** (0 = identical, 1 = disjoint) and near-zero
correlation — the divergence is largest when the prior is strong (TV≈0.70–0.90)
and smallest for the weakest prior (SD80: TV≈0.20), where the switch mostly
follows the evidence and the two models nearly coincide. Their circular *means*
look similar — averaging a bimodal switch
gives a number close to the integrated compromise — but the **distributions are
qualitatively different**: unimodal-and-attracted (notebook) vs
bimodal-with-mass-at-the-prior (package).

**This is exactly the integration-vs-switching distinction the paper is about.**
The notebook labelled "switching starter" actually implements the *integration*
(Basic Bayesian) observer; the package `switching_observer.py` implements the
paper's *switching* observer. If the team's impression was that the root
switching model and the package switching model are the same, that is the most
important correction in this document.

---

## 2. HB — same prior, but a DIFFERENT estimate distribution

The HB pair is closer — they genuinely agree on the model *structure* — but they
still do not compute the same estimate distribution.

**What matches (verified by running):**
- **The mixed prior is identical.** Both build
  `alpha * vonMises(225, kappa) + (1 - alpha) * uniform`. Running both prior
  constructions on the same kappa/alpha gives total-variation distance
  **0.006** and correlation **0.9997** — the same curve.
- Same readout rule (MAP), same motor/lapse tail structure.

**What differs — the measurement model:**

Running both estimate distributions at a fixed trial (dir 135, no motor/lapse):

| | notebook `build_trial_components` | package `hb_integration` |
|---|---|---|
| effective support | **1.0 direction** (a near-spike) | **~144 directions** (broad) |
| mass within ±15° of mode | 1.000 | 0.340 |
| secondary prior bump at 225 | none | present |

- **Notebook uses the true stimulus direction as the internal measurement.**
  Its own code comment says so: *"Scaffold simplification: use true motion
  direction as the measurement center. A fuller port marginalizes over noisy
  measurements m sampled from P(m | motion_direction)."* With no measurement
  noise, the posterior collapses to a single point and the predicted estimate
  is essentially deterministic — a spike.
- **Package marginalizes over the noisy measurement** `m ~ P(m | theta_true)`,
  which is the paper's actual generative model. That produces a realistic broad
  estimate distribution, and it is what makes the emergent prior bump (the
  bimodality) appear.

So even though the prior and readout are the same, the notebook HB predicts a
sharp near-deterministic estimate while the package HB predicts the full noisy
distribution. Their per-trial estimate distributions have TV ≈ 1.0 (the spike
and the broad distribution barely overlap) even when their peaks nearly
coincide. For likelihood-based fitting this matters a lot: the notebook version
cannot represent response scatter from sensory noise, only from motor noise and
lapses.

Second, smaller difference: **online learning.** The package learns prior
precision kappa as a fitted Bayesian belief updated trial-by-trial; the notebook
carries kappa as a deterministic delta-rule proxy that is not estimated through
the response likelihood. (Out of scope for "same estimate distribution" but
relevant to the abstract's learning claim.)

---

## Verdict for the team

| | same thing? | why |
|---|---|---|
| **switching** | **No — different model class** | notebook = integration (smooth compromise); package = true switch (mixture with prior-mode delta) |
| **HB** | **No — same prior, different measurement model** | notebook uses true dir as measurement (spike); package marginalizes measurement noise (broad, bimodal) |

Your working impression is right to be cautious about the two root models. The
package versions are the ones that (a) implement the paper's actual mechanisms —
a genuine switch, and measurement-marginalized integration — and (b) pass the
numerical verification suites (switching 5/5, HB 12/12, including the exact
alpha=1 → Girshick reduction). The root notebooks are readable first-pass
scaffolds: the "switching" one is actually an integration observer, and the HB
one is a spike-approximation of the package HB with the same prior.

Recommendation: for anything the abstract depends on, use the `hierarchical/`
package definitions. If you want to keep the root notebooks as teaching
scaffolds, relabel the switching starter as an *integration/Basic-Bayesian*
demo to avoid the naming confusion.

---

### How this was tested (reproducible)

Both notebook model cores were extracted and run against the package classes on
identical inputs, with motor noise and lapse disabled so only the core model was
compared:
- switching: `girshick_lookup(dir, k_llh, 225, k_prior, readout="MAP")` vs
  `SwitchingObserver.estimate_distribution(dir, coh, prior_label)`, swept over
  true direction.
- HB: `build_trial_components(...).estimate_pdf` vs
  `HBIntegrationObserver.estimate_distribution(coh, dir, belief)` with a
  degenerate belief on a single kappa, plus a direct comparison of
  `build_prior_grid` vs `mixture_prior`.
Metrics: total-variation distance, correlation, argmax location, and effective
support (exp of entropy).
