# Reliability-mixture hierarchical Bayesian model

Fitted and validated across all 12 subjects. Built by Romi with Claude (Anthropic)
as a coding/analysis collaborator.

**Read this before assuming this is the same thing as
`hierarchical/observers/models/hb_integration.py`. It isn't.** Both are
legitimate "hierarchical Bayesian" proposals, but they make opposite choices
about what's actually learned. See "How this differs from `hb_integration.py`"
below before using either one in a write-up, abstract, or comparison.

## What this model is

On each trial, the percept is a genuine discrete **either/or mixture** (never
a multiplicative blend) between:
- a **prior-driven** component: a von Mises centered on the fixed prior mean
  (225 degrees)
- a **likelihood-driven** component: a von Mises centered on that trial's
  true motion direction

The mixture weight, `prior_reliance`, is the hyper-prior. It updates
trial-by-trial via a delta rule against a 5-trial smoothed window of true
feedback direction, with learning rate `alpha`. `alpha` is the model's answer
to "how fast does this subject update their trust in the prior."

This follows the specific framing from the team's 13 Jul meeting (recorded in
`HB_model_handoff.md`, not currently in this repo): *"The observer first
estimates how reliable the prior itself is, and this estimate (hyper-prior)
controls switching... the thing that's learned is confidence in the prior,
not the prior's mean."*

The discrete-mixture choice (over a continuous multiplicative blend) is
deliberate: a product of two von Mises distributions is always a single von
Mises, so a multiplicative construction cannot reproduce the bimodal estimate
distributions the whole project is trying to explain. This was checked both
algebraically and against real data before committing to this design (see
`OLD_/hb_verified_model_implementation.ipynb` for the ruled-out multiplicative
version this superseded).

## How this differs from `hb_integration.py`

| | This model | `hb_integration.py` |
|---|---|---|
| What's learned trial-by-trial | `prior_reliance` (mixture weight, alpha) | prior concentration (`kappa`) |
| What's held fixed | `k_prior` per block width | mixture weight (`alpha`) |
| Mixture components | prior-centered **and stimulus-centered** | prior-centered **and uniform** |
| Why bimodality can emerge | genuine two-location mixture | MAP readout of a Bayesian posterior, mixture responsibility is "derived," not hand-coded |

Neither is more "correct" by construction, they're different hypotheses
about what a subject is actually uncertain about. **Note:** the current
`abstract_draft.md` describes `hb_integration.py`'s architecture (peaked +
uniform mixture, learned kappa), not this one. Worth resolving with whoever
wrote `hb_integration.py` before treating either version as final.

## What's in this folder

```
notebooks/
  kaggle_hb_multisubject_fit.ipynb        Main fit, all 12 subjects, run on Kaggle
  kaggle_hb_followup_flagged_subjects.ipynb  Re-fit for 6 subjects that hit parameter
                                            ceilings in the main run, with widened bounds
  alpha_pull_correlation_traceable.ipynb   Self-contained: rebuilds pull_increase from
                                            raw data and reproduces the alpha correlation
  bimodality_analysis.ipynb                Self-contained: for every subject and trial,
                                            computes the predicted response distribution
                                            and checks whether it's genuinely bimodal
                                            (same peak-finding methodology as the GPU
                                            project's own multimodality screen, for
                                            direct comparability). The example plots at
                                            the end visualize real trials captured
                                            during that same calculation, not a separate
                                            search. Needs the CSV and all 12 checkpoints
                                            attached as inputs; runs in a couple of
                                            minutes, no chunking needed.

results/
  checkpoints/subject_N.pkl                Best-fit parameters, all 8-start results,
                                            NLL, per-trial predicted probability, and
                                            prior_reliance trajectory, per subject
  alpha_by_subject.csv                     alpha + NLL per subject, with source checkpoint
  alpha_pull_merged_traceable.csv          alpha merged with pull_increase per subject
  our_model_multimodality_screen.csv       fraction of trials with genuinely bimodal
                                            predicted distributions, per subject
                                            (produced by bimodality_analysis.ipynb)
  our_model_multimodality_by_condition.csv  same, broken down by prior width x coherence

findings/
  hb_model_findings_summary.md             Two findings: (1) k_prior becomes
                                            unidentified when prior_reliance collapses
                                            in wide-prior blocks, a real design issue
                                            traced to k_prior's dual role; (2) alpha
                                            correlates with independent behavioral pull
                                            specifically at low coherence

src/
  reliability_mixture_model.py             The model itself: load_and_prepare_data,
                                            the von Mises mixture, the trial loop, NLL.
                                            Verified to reproduce the exact NLL values
                                            used in fitting. bimodality_analysis.ipynb
                                            inlines these same functions so it runs
                                            standalone on Kaggle/Colab.
```

## Key results at a glance

- **10 free parameters per subject** (3 `k_llh` + 4 `k_prior` + `alpha` + `k_motor` + `lapse_rate`)
- `alpha` ranges 0.027 to 0.248 across subjects, real individual variation
- `alpha` correlates with an independent behavioral measure (`pull_increase`)
  specifically at the lowest sensory coherence (r=0.67, p=0.017; does not
  survive strict Bonferroni correction given n=12 and 4 comparisons tested,
  treat as a promising signal, not a settled result)
- **The model genuinely produces bimodal per-trial predictions** (14.3%
  of trials on average, up to 30% in wide-prior/high-coherence conditions),
  and does so exactly where the paper predicts it should (bimodality climbs
  sharply as prior width increases, near-zero for narrow priors)
- A real identifiability issue: `k_prior` becomes meaningless whenever
  `prior_reliance` collapses toward zero in a condition (mostly the 80-degree
  blocks, 8 of 12 subjects affected). Full mechanism and evidence in
  `findings/hb_model_findings_summary.md`.

## Reproducing this

1. `results/checkpoints/*.pkl` are the source of truth for fitted parameters,
   already validated (see `src/reliability_mixture_model.py`'s NLL check).
2. To refit from scratch: run `notebooks/kaggle_hb_multisubject_fit.ipynb` on
   Kaggle (needs `data01_direction4priors.csv` attached as an input, already
   in this repo at `hierarchical/data/`). Expect roughly 7-9 hours total
   across all 12 subjects; the notebook supports chunked commits with
   checkpoint recovery if run in smaller pieces.
3. `notebooks/alpha_pull_correlation_traceable.ipynb` is fully self-contained
   and reproduces every number in the `alpha`/`pull_increase` finding from
   raw data.

## Status relative to the rest of the project

Not yet done: the AIC comparison against the paper's Switching observer
(`hierarchical/observers/models/switching_observer.py` in this repo already
has a working fitter, `N_PARAMS = 9`). That's the next step once the team
has aligned on which hierarchical model(s) to carry forward.
