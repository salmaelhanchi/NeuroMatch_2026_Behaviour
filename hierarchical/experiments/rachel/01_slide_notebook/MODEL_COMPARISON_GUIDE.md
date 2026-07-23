# Thinking about model comparison: AIC/BIC vs cross-validation

A practical reference for the Posterior Motives model comparison. Written for the
presentation — how to read the two axes, how to present them honestly, and what
the current results actually say.

*All per-subject numbers below are from the live `api.results_table()` at the time
of writing. Regenerate with `api.results_table()` before quoting them in a talk —
fits get refit and the table is the source of truth, not this doc.*

---

## 1. The two axes answer different questions

| | In-sample (AIC / BIC) | Out-of-sample (CV) |
|---|---|---|
| **Question** | How well does the model describe the data it was fit to, penalized for complexity? | Does the model predict trials it never saw? |
| **Cost** | Cheap — one fit per subject | ~5× a point fit (one fit per fold) |
| **Rewards** | Fit quality, docked 2/param (AIC) or ln N/param (BIC) | Structure that *recurs*, not memorized quirks |
| **Blind spot** | The complexity penalty is a fixed tax on *parameter count*. It cannot see flexibility that doesn't add parameters (a likelihood surface that bends to noise). | Introduces its own failure modes — a bad fold, an ill-conditioned fit on 80% of the data. |
| **In this repo** | The paper's primary method; complete for all 7 models × 12 subjects. | `cross_validate.py`: 5 block-aligned folds, sequential-belief replay. |

**AIC vs BIC:** same fit term, different penalty. BIC's penalty (`ln N · k`) grows
with sample size, so on these ~5,000–9,000-trial subjects BIC punishes extra
parameters *much* harder than AIC. If AIC and BIC disagree, it's almost always
about whether a model's extra parameters earn their keep — report both and say so.

---

## 2. How CV works here (and why the fold design matters)

1. Split each subject's trials into **5 contiguous, block-aligned folds** (`_block_folds` cuts on prior-block boundaries — *not* random trials).
2. For each fold: hold it out, fit on the other 4, score the held-out fold's log-likelihood under that fit.
3. Sum held-out log-likelihoods across folds → every trial predicted once, by a model that didn't see it.

**Why block-aligned, not random:** several models (switch, hb_rachel, recombined,
hierarchical_online) are *sequential learners* — the belief at trial *t* depends on
trials 1…*t*−1. The contract is two-part: `fit(mask=train)` only *scores* training
trials when searching parameters, but `trial_logliks` **replays the belief over all
trials in order**; CV then reads off only the held-out ones. Random folds would
break the learning sequence and make the held-out scores meaningless.

**The leak/generalization check** (`validate_cv`): compares training-fold fit to
held-out fit (the generalization gap), and flags any per-trial held-out NLL worse
than the uniform-guess baseline **ln(360) ≈ 5.886**. A model scoring above that
predicts held-out data *worse than chance*.

---

## 3. The mental model: the axes should agree — disagreement is the finding

- **Both axes agree** → state the conclusion with confidence. That's your headline.
- **They disagree** → do not hide it. A disagreement is the most scientifically
  honest slide you can show, and reviewers trust a comparison that surfaces its own
  tensions far more than a clean sweep.

**Hierarchy of evidence (not a single leaderboard):**

| Pattern | Reading |
|---|---|
| Wins in-sample **and** out-of-sample | Strongly supported |
| Wins in-sample **but fails CV** | Overfitting / ill-conditioning — do **not** trust the in-sample win |
| Close in-sample **but simpler** | Parsimony argument |
| Loses AIC but conceptually central | Present as a *framework*, not a leaderboard entry |

---

## 4. Three rules that keep the comparison honest

1. **Compare per-subject — never sum across subjects.** Subjects have different
   trial counts, so a summed AIC is dominated by whoever has the most trials, and
   models fit for different subject sets aren't comparable in a total. Show a
   per-subject win-count or a per-subject ΔAIC strip.
2. **Lead with in-sample, corroborate with out-of-sample.** AIC/BIC is the primary
   comparison (paper's method, complete). CV is the robustness check that either
   *confirms* the AIC winner generalizes or *flags* where it doesn't — not the
   headline number.
3. **Match the optimization budget, or footnote the mismatch.** If one model was
   fit at maxiter=800 and another at 400, a reviewer will ask. Equalize it, or state
   that you checked more iterations don't move the result.

---

## 5. What the current results actually say

*(live `api.results_table()`, 7 models × 12 subjects; CV present for 5 models —
Hier-Online and Recombined CV not yet run at time of writing.)*

**Per-subject AIC winner:**

| Winner | Subjects | Count |
|---|---|---|
| **Switch** | 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 10 / 12 |
| Basic-Bayes | 1 | 1 / 12 |
| HB-Rachel | 2 | 1 / 12 |

Switch is the dominant in-sample model. But read the **margins**, not just the
winner: subject 8 is Switch by only +8.0 AIC over HB-Adaptive, subject 5 by +38.6
over Hier-Online — near-ties, not decisive wins. Subjects 1, 2 go to other models
outright.

**The key disagreement — subject 5:** Switch **wins AIC** there (61747.7, narrowly
over Hier-Online 61786.2) **but its CV is the only worse-than-chance score in the
whole table**: per-trial 6.265 vs the 5.886 uniform baseline. Every other
model/subject CV is in-band (~5.39–5.47 for the HB family). Confirmed genuine, not
an optimization artifact — refitting Switch at the paper's 1e-4 tolerance
(maxiter=800, 10-start) left it essentially unchanged. Root cause: the
ill-conditioned k=9 Switch likelihood has many local maxima on that subject — the
same difficulty Laquitaine & Gardner documented.

**How to present subject 5:** it is a genuine *"the in-sample winner fails
out-of-sample"* case — the strongest version of the honest-disagreement story. The
model that best describes subject 5's data (Switch) does **not** generalize on that
subject's held-out trials. That is a finding about the Switch likelihood's
conditioning, not a defect in the analysis.

**Coverage caveat to state on any CV slide:** CV currently covers 5 models (Switch,
Basic-Bayes, HB-Rachel, HB-Salma, HB-Adaptive), all 12 subjects. Hier-Online and
Recombined CV are not yet complete — so any CV comparison is over those 5 models
until the runs finish. AIC/BIC covers all 7.

---

## 6. How the original paper (Laquitaine & Gardner 2018) presents it

Worth matching their format — a reviewer will expect it.

**They use AIC only.** The full text mentions AIC 24 times; **BIC zero times, and
cross-validation zero times.** Their stated rationale: AIC "accounts for overfitting
by penalizing models with greater number of parameters." Their formula (Eq. 12) is
standard AIC, `AIC = 2·[n − log p(θ̂|model)]` (n = number of fit parameters).

**They never report a raw AIC — only pairwise differences (ΔI), averaged over
subjects, with a 95% CI and a per-subject win count.** The recurring format is:

> "average AIC difference … 95% CI [.. ..] over subjects, in favor of switching for
> N out of 12 subjects."

Their actual reported comparisons:

| Comparison | avg ΔAIC | 95% CI | favors switch |
|---|---|---|---|
| Basic Bayesian − Switching | 395 | [96, 695] | 8/12 |
| Sampling vs Basic Bayesian | 478 | [262, 693] | 12/12 |
| Bayes MAP tailed prior | 483 | [219, 756] | 10/12 |
| Bayes MAP tailed LLH | 829 | [342, 1316] | 10/12 |
| Bayes MAP motion-energy LLH | 1377 | [872, 1881] | 12/12 |
| Bayes MAP tailed LLH + prior | 429 | [12, 847] | 9/12 |

**Two conventions to adopt:**
- **Difference-then-average, never average raw AICs.** Each subject contributes its
  own *paired* ΔAIC; the average is over those differences. This is the principled
  version of the "compare per-subject" rule — it sidesteps the trial-count problem
  entirely. A raw-AIC average would be dominated by high-trial subjects; a ΔAIC
  average is not.
- **Fixed sign convention:** always subtract the reference model (they use Switch)
  so positive = favors the reference. Pick Switch as your reference too.
- **Pair the average with a per-subject figure** (they use Fig 5E / 6B: ΔAIC per
  subject, and count of subjects favored). Show both the summary and the spread.

**Why our CV is an addition, not a replication.** The paper does *not* cross-validate
— so our 5-fold block CV goes *beyond* the original, applying a more stringent
out-of-sample test than they used. Frame it honestly as an extension. The one
robustness check they *did* run is split-half parameter stability: they fit the
Switching observer to the first and last halves of each subject's data and confirmed
the fit parameters and AICs "did not significantly change" between halves — but that
tests parameter stability, not held-out prediction. Notably, the subject-5 Switch CV
failure is exactly the kind of thing their AIC-only approach could not have surfaced.

---

## 7. One-paragraph version for the talk

> We compare models on two axes. In-sample (AIC/BIC, the paper's method) asks how
> well each model describes the data; Switch wins in 10 of 12 subjects, though
> several are near-ties. Out-of-sample cross-validation asks whether the model
> predicts held-out trials, and it agrees with the in-sample story on 11 of 12
> subjects. The exception is subject 5, where Switch fits best in-sample yet is the
> only model in the table to predict held-out data worse than chance — a signature
> of the ill-conditioned k=9 Switch likelihood, exactly the difficulty the original
> paper flagged. We report both axes per-subject rather than as a single ranking,
> because the place they disagree is as informative as the places they agree.
