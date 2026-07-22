# Project memory — Switching Observer specialist

_Durable notes for future sessions in project `proj_e85b260078c3`. Search:
"project memory switching observer"._

## The user
- **First-time modeler.** This is their first Bayesian model / first model of any
  kind. Calibrate explanations accordingly: define terms (likelihood, generative
  model, parameter recovery), explain *why* each step matters, avoid unexplained
  jargon. They are capable and want to understand, not just be handed results.
- Part of a 5-person Neuromatch team (Anirban, Salma, Rachel, Rommana/Romi,
  Valeria). Team-authored code in `hierarchical/` is already quite advanced —
  the USER personally is new, so guidance should help them understand and
  contribute/present, not assume they wrote the models.

## The project
- Neuromatch project extending Laquitaine & Gardner (2018, Neuron), "A Switching
  Observer for Human Perceptual Estimation." Motion-direction estimation; 12
  subjects; coherence 6/12/24%; prior SD 10/20/40/80° around mean 225°.
- Shared host path: `/Users/vestige/code/heartwood/nma-prep/group-project/hierarchical`
  (ro). Trial data: `data/data01_direction4priors.csv` (83k rows, all subjects).
  Key cols: trial_index, motion_direction (= feedback), motion_coherence,
  prior_std, subject_id, estimate_x/estimate_y (response as unit vector).
- **Feedback is real** (confirmed in paper Methods p16): true direction shown at
  end of every trial to help learn priors; reward feedback withheld. So the
  online-learning models' learning signal is part of the task, not an assumption.

## Models built (see model_review.md artifact)
1. `switching_observer.py` — paper's static winner (9 params). Faithful MATLAB port.
2. `online_learner.py` — online-learned prior strength k, belief b_t(k), λ (6).
3. `asymptote_transient.py` — per-block asymptote + within-block transient (11).
4. `hb_integration.py` — the abstract's model: mixed hyper-prior
   α·VM(225,κ)+(1−α)/360, MAP read-out, online κ (7).
- All three extensions provably nest the static observer (verified ~1e-17).
- Fit only on subjects 1 & 3 so far (1–1 split). k_e[0.24] unidentified (ridge).

## CRITICAL scientific context (paper Discussion, p11–12)
The paper **already anticipated and tested the abstract's exact model**:
- It proposes a hierarchical Bayesian observer with a hyper-prior that each trial
  the direction is drawn from "either a uniform or a peaked distribution ... with
  probabilities determined by the ratio of likelihood and prior strengths" and
  says this "would effectively amount to a reformulation of our Switching observer
  in Bayesian terminology."
- It reports: "We have similarly explored ... mixtures of uniform and von Mises
  distributions, and found that while these could produce bimodal estimate
  distributions, the Switching model provided better fits."
- Caveat it leaves open: "We cannot preclude the possibility that there may be
  some distributional form that would allow a Bayesian model with multiplicative
  integration to better explain the data."
⇒ The team's defensible novelty is NOT the static mixture (done, lost) but the
**online / trial-by-trial learning of prior confidence** — which the paper flagged
as a possible formulation but never implemented. Reframe the abstract around the
learning dynamics.
