# Hierarchical Online observer — integration notes

A 7th model in the comparison registry, built from the build spec
`IMPLEMENT_hierarchical_online_FULL.md`. This note covers what it is, what was
adapted from the spec, the guarantee that the adaptation does not change its
numerics, and how to fit all 12 subjects.

## What the model is

Each trial the observer treats the true direction as coming from a **mixture
prior** — a peaked von Mises (weight `pi`) *or* a flat uniform (weight `1-pi`)
— and reads out the responsibility-weighted posterior:

    post(θ | m) ∝ VM(θ; m, k_llh) · [ pi · VM(θ; μ, κ) + (1-pi)/360 ]

The prior's **mean μ and concentration κ are learned online, trial by trial,
from the feedback** (the true direction shown after each response) via a leaky
resultant-vector delta rule:

    μ_t = atan2(Cy, Cx);  R_t = min(hypot(Cx,Cy), 0.9999);  κ_t = R_t(2-R_t²)/(1-R_t²)
    Cx ← (1-α)Cx + α·cos(dir_t);   Cy ← (1-α)Cy + α·sin(dir_t)

So the prior **width is learned, not fitted**, and — uniquely among our models —
so is the prior **mean**. Trials run in chronological order; the belief resets
at each session boundary (`session_id`).

### Readouts (`readout=`)
- `sample` (default) — probability-matching draw; the closest analogue to the
  paper's Switching observer (draw the latent cause `z∼Bernoulli(responsibility)`
  and report the chosen component). Reproduces the switching bimodality.
- `select` — deterministic winner-take-all (posterior argmax).
- `average` — posterior mean (the unimodal Basic-Bayesian limit).

### Parameters (8)
| idx | name | meaning | bound |
|---|---|---|---|
| 0–2 | `k_llh[coh]` | sensory reliability per coherence (0.06/0.12/0.24) | ≥ 0 |
| 3 | `pi` | prior mixing weight = P(peaked cause) | [0, 1] |
| 4 | `p_rand` | lapse rate | [0, 1] |
| 5 | `k_motor` | motor precision (von Mises concentration) | ≥ 0 |
| 6 | `alpha` | online learning rate (feedback delta rule) | (0, 1] |
| 7 | `R0` | initial prior strength (resultant length) | [0, 1) |

`mode_init = 225°`. Special cases: `pi→1` → Basic Bayesian; `alpha→0` with fixed
`R0` → static hierarchical observer; frozen responsibility → online Switching.

## How it relates to the other models

Same mixture-prior *form* as `hb_adaptive`, but two mechanisms are new:
1. A **point-estimate** resultant-vector learner (not a Bayesian belief grid).
2. It **learns the prior mean μ** (every other model fixes it at 225°).
Also, `pi` is a fitted constant here (vs. `hb_adaptive` learning its α), and the
readout is a tunable knob rather than fixed to integration.

## What was adapted from the spec — and the no-change guarantee

The **numerics are the spec's, verbatim**; only the interface was adapted.

- **Circular helpers.** `vm_pdf` / `circ_convolve` delegate to the repo's
  `observers.helpers.circular` (`von_mises_pdfs` / `circular_convolution`),
  verified numerically identical to the spec's own copies (max abs diff
  ~1.7e-15 on von Mises; convolution bit-identical). One source of truth, zero
  numerical change.
- **Fitter packing (`packing=`).** Two options, **both optimising the identical
  negLL**:
  - `penalty` (**default**) — the spec's exact penalty-box Nelder-Mead with his
    4 start points. Fits reproduce his run bit-for-bit.
  - `house` — unconstrained sigmoid/logit space like the rest of the registry
    (`online_recovery.py` style), seeded from the same starts. More robust
    search; a different optimiser path, so fitted params can differ slightly.
    Opt-in. Verified to evaluate the same negLL as `penalty` at a shared point
    (Δ = 0).
- **`session_id` threading.** `load_subject()` now carries `session_id` as an
  additive 5th key so this model can reset its belief per session. The other six
  models never read it; verified non-interfering — `hb_adaptive` scores
  subject 1 at negLL 38450.6251 with the key present, exactly its saved fit.

The learning rule, readout math, caching key (`round(mu/4)*4`, log-κ to 1 dp),
readout default (`sample`), 8-parameter semantics, and AIC/BIC formulas
(`AIC=2K+2·nll`, `BIC=K·ln N+2·nll`, K=8) are all the spec's, unchanged.

## Correctness checks (spec §8)

    python observers/verification/verify_hierarchical_online.py   # (PYTHONPATH=.)

All pass: `select≡girshick MAP`, `sample≡sampling_lookup`, `average≡girshick
BLS` (with `weight_tail=1-pi`), finite sequential LL, and the `alpha→0` static
limit (exact, 0.0, at the model's own mode discretisation).

## Files

- `observers/models/hierarchical_online.py` — observer + readouts + learner.
- `observers/fitting/hierarchical_online_fit.py` — fitter (both packings),
  `_trial_logliks`, `_simulate`.
- `observers/comparison/registry.py` — `_hierarchical_online_spec()`, registered
  as `hierarchical_online` (label `Hier-Online`, green `#3a7d44`).
- `observers/verification/verify_hierarchical_online.py` — §8 checks.

## Fitting all 12 subjects

It uses the standard batch driver — no new script needed. Writes to
`results/fits/comparison/hierarchical_online/subject{S}.json`, resumable.

    # production fit, all subjects (uses the spec's maxiter=1500 via fit_multistart's 4 starts)
    PYTHONPATH=. python -m observers.comparison.fit_batch \
        --models hierarchical_online --maxiter 1500

    # one subject, for a quick check
    PYTHONPATH=. python -m observers.comparison.fit_batch \
        --models hierarchical_online --subjects 1 --maxiter 1500

**Cost.** ~20–25 s per subject-evaluation × 4 starts × Nelder-Mead iterations →
roughly **10–40 min per subject**, so ~2–8 h for all 12 serially. Launch it on
your own compute (Kaggle background, or a workstation left running); it is
resumable, so a killed run picks up where it left off. Once the JSONs exist,
`hierarchical_online` drops into every downstream comparison (AIC/BIC tables,
the presentation notebook) exactly like the other models.

**A note on reading the result.** Because the prior form matches `hb_adaptive`,
if the converged fit looks like `hb_adaptive`, that is a *finding* (the two
learning schemes converge), not a bug. The genuinely new degrees of freedom to
inspect are (a) the point-estimate learner's sharper block-width tracking (the
smoke fit already recovered 80°→~70° vs `hb_adaptive`'s ~25°), (b) whether the
learned mean μ moves off 225° for any subject, and (c) the `sample` readout's
switching bimodality.
