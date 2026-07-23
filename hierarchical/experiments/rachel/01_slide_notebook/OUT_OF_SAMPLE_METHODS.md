# What out-of-sample method is right for this study?

A literature-grounded evaluation for the Posterior Motives model comparison
(Laquitaine & Gardner 2018 switching-observer paradigm, our 7-model registry).

The question is not "which model predicts best" in the abstract — it is a
**mechanistic adjudication**: does the observer *switch* between a prior and a
sensory strategy, or *graded-integrate*? That goal, plus four implementation
specifics, decides which out-of-sample test is defensible. All method claims below
are anchored to the methods literature (references at the end).

---

## The implementation specifics that constrain the choice

1. **Within-subject, thousands of trials, per subject.** Each subject has ~4,800–9,400
   trials and models are fit per subject. The parameter-to-data ratio is tiny
   (6–9 params vs thousands of trials).
2. **Mixed model roster.** Some models are **static with condition-specific parameters**
   (Switch, Basic-Bayes: a separate `k_prior` per prior width — 4 of their 9 params);
   others are **sequential learners** with no condition-specific prior parameters
   (HB-Adaptive, HB-Rachel, Hier-Online, Recombined learn the prior online).
3. **Sequential dependence.** The learning models' belief at trial *t* depends on the
   full feedback history; you cannot cleanly excise a subset of trials without
   disturbing the belief trajectory that predicts the rest.
4. **Prior width is the independent variable**, not a nuisance covariate — the
   experiment manipulates it to see how behavior changes.

---

## Method-by-method evaluation

### A. Qualitative signature prediction (posterior-predictive checks) — BEST SUPPORTED

Simulate estimate distributions from the fitted model and test whether it reproduces
**qualitative features the likelihood was not directly tuned to** — above all the
**bimodality** of estimate distributions, plus the bias and variability curves.

- **Why it fits this study:** it is exactly how the original paper adjudicated the
  mechanism (Switch predicts bimodality; Basic-Bayes cannot). It is
  **parameterization-neutral** — static and learning models are compared on the same
  simulated-behavior footing, so the mixed roster (specific #2) is not a problem.
- **Literature support:** posterior-predictive model checking (Gelman, Meng & Stern
  1996) is the standard tool for assessing whether a fitted model reproduces features
  of the data. Palminteri, Wyart & Koechlin (2017) argue specifically that
  **simulating the model and testing qualitative falsification** ("model simulation" /
  "falsification") is more diagnostic than fit indices for cognitive models, because a
  better fit does not guarantee the model reproduces the behavioral phenomenon of
  interest. Wilson & Collins (2019) make the same point in their "rules": inspect
  model-simulated behavior, not just fit statistics.
- **Verdict:** this is the primary out-of-sample evidence to lead with. It directly
  tests the mechanistic claim and every model can enter the comparison fairly.

### B. Trial-held-out cross-validation — SUPPORTED AS A ROBUSTNESS CHECK, with caveats

Hold out a fold of trials, fit on the rest, score held-out log-likelihood.

- **Why it partly fits:** it is **fair across the roster** (every model keeps all its
  parameters — no condition is deleted), so unlike option D it does not rig the
  comparison. It caught a real problem here (subject-5 Switch predicts held-out trials
  worse than chance — an ill-conditioned k=9 likelihood).
- **Caveats grounded in the specifics:**
  - With thousands of trials and few parameters (#1), CV and AIC largely agree by
    construction; the extra information over AIC is concentrated in pathological
    subjects. Arlot & Celisse (2010) note CV's value is greatest when the complexity
    penalty is uncertain — here it mostly is not.
  - For the learning models (#3) the train/test firewall is **leaky**: scoring a
    held-out fold replays the belief over all trials in order, so held-out trials still
    shape the belief state predicting later trials. The "out-of-sample" independence is
    weaker than the phrase implies — it tests "do parameters fit on these blocks
    predict those blocks," not prediction of truly unseen data.
  - Block-aligned folds are **not stratified on prior width** (worst on subjects 8 and
    4). Measured empirically, re-stratifying would shift each model's per-trial CV by
    ~0.06 nats **common-mode**, changing no ranking — so the imbalance is a caveat, not
    a confound.
- **Verdict:** keep it as a secondary robustness check. Present per-subject, never
  summed. Report the subject-5 finding honestly. Do not oversell it as *the*
  out-of-sample verdict.

### C. Parameter recovery / identifiability — SUPPORTED, PREREQUISITE not a headline

Simulate from the fitted model, refit, check parameters (and model identity) are
recovered.

- **Why it fits:** Wilson & Collins (2019) and Palminteri et al. (2017) treat parameter
  recovery and **model recovery** (confusion matrix: fit all models to data simulated
  from each) as a precondition for interpreting any fit or comparison. For the near-tie
  subjects and the ill-conditioned Switch likelihood, this tells you whether the fitted
  numbers mean anything before you argue about generalization.
- **Verdict:** worth running as due diligence — especially model recovery for the
  Switch-vs-graded pair — but it validates the comparison rather than being the
  out-of-sample result itself.

### D. Leave-one-condition-out (e.g. leave-one-prior-width-out) — NOT SUPPORTED HERE

Fit on 3 of 4 prior widths, predict the held-out width.

- **Why it fails on the specifics:** Switch and Basic-Bayes fit a **separate `k_prior`
  per width** (#2). Holding out a width means that width's prior parameter is **never
  estimated**, so prediction on the held-out width tests an unconstrained parameter,
  not generalization. It would produce a **rigged result favoring the learning models**
  (which share their mechanism across widths) purely as a parameterization artifact.
  It also holds out the **independent variable** (#4) — testing a capability the study
  never claims the observer has — and the sequential-dependence leak (#3) is worse when
  the split is by a variable braided through the trial sequence.
- **Verdict:** considered and rejected. Only a fair version exists *within* the
  learning family (which all share cross-width mechanisms), but then Switch/Basic-Bayes
  cannot enter, defeating the purpose. Do not use for the cross-model comparison.

---

## Recommendation for the presentation

1. **Lead the out-of-sample story with qualitative signature prediction (A)** —
   bimodality and bias/variability. It adjudicates the mechanism, it is what the paper
   leaned on, and it is fair to every model.
2. **Support with trial-held-out CV (B)** as a per-subject robustness check; report the
   subject-5 Switch generalization failure as a genuine finding, footnote the
   prior-width fold imbalance (measured ~0.06 nats common-mode, changes no ranking).
3. **Back both with parameter/model recovery (C)** as due diligence for the near-tie
   subjects and the ill-conditioned Switch likelihood.
4. **Do not use leave-one-prior-width-out (D)** — unfair across the mixed roster.

In one line: for a *mechanistic* question with condition-specific parameters and
sequential learners, **"does the model reproduce the behavioral signature it was not
tuned to" beats "does it predict held-out trials," and both beat "does it extrapolate
to a held-out condition."**

---

## References (verified via OpenAlex)

- Gelman, Meng & Stern (1996). Posterior predictive assessment of model fitness via
  realized discrepancies. *Statistica Sinica*.
- Busemeyer & Wang (2000). Model comparisons and model selections based on
  generalization criterion methodology. *Journal of Mathematical Psychology*.
  doi:10.1006/jmps.1999.1282
- Arlot & Celisse (2010). A survey of cross-validation procedures for model selection.
  *Statistics Surveys*. doi:10.1214/09-SS054
- Palminteri, Wyart & Koechlin (2017). The importance of falsification in computational
  cognitive modeling. *Trends in Cognitive Sciences*. doi:10.1016/j.tics.2017.03.011
- Yarkoni & Westfall (2017). Choosing prediction over explanation in psychology:
  lessons from machine learning. *Perspectives on Psychological Science*.
  doi:10.1177/1745691617693393
- Wilson & Collins (2019). Ten simple rules for the computational modeling of
  behavioral data. *eLife*. doi:10.7554/eLife.49547
- Guest & Martin (2021). How computational modeling can force theory building in
  psychological science. *Perspectives on Psychological Science*.
  doi:10.1177/1745691620970585
