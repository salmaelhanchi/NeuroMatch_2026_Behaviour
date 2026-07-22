# Critique and support for the two hypotheses, with alternatives

This note evaluates the two models you are proposing, grounds each in the original paper and in more recent literature, and suggests principled alternatives/additions. Throughout, it points to the exact place in `Model_explainer.md` and the notebook `Switching_Bayesian_Observer_starter.ipynb` that would have to change, so the modelling follows directly from the reading.

This version incorporates the two Neuromatch project decks you added in `Neuromatch Project PPTs/` — one that implements an online Bayesian prior-learning model and one that proposes a logistic-mixture switching model — since they are prior attempts at exactly these two hypotheses. (The original Airtable links were gated and unreadable, but these decks appear to be the same work.)

---

## Restating the two hypotheses precisely

**Hypothesis 1 — hierarchical/mixture reformulation of Switching.** On each trial a latent indicator decides whether the motion direction is treated as coming from a *uniform* (evidence-dominated) or a *peaked* (prior-dominated) distribution, and the mixing probability is set by the ratio of likelihood and prior strengths. This is exactly the reformulation the paper floats in its Discussion: *"a hierarchical Bayesian observer could be formulated in which perceptual judgments are determined by hierarchical beliefs, or hyper-priors (Lee and Mumford, 2003; Sato et al., 2007; Tenenbaum et al., 2011) that motion directions were drawn from either a uniform or a peaked distribution at each trial, with probabilities determined by the ratio of likelihood and prior strengths. This would effectively amount to a reformulation of our Switching observer in Bayesian terminology."*

**Hypothesis 2 — switching-posterior observer with online updating.** Keep the paper's *Switching posterior* observer (form the full posterior, then switch between the posterior mode and the prior mean), but drive the prior it uses from an *online* running estimate of the stimulus stream rather than a fixed per-block value.

---

## What the original paper already commits to (the baseline you are extending)

The Switching observer is defined at the readout stage, not the integration stage. Its switching probability is a fixed function of the two concentrations:

> p_prior = k_prior / (k_prior + k_e),  p_e = 1 − p_prior

and the per-trial estimate distribution is a mixture of a delta at the prior mean and the likelihood-based percept distribution, then a lapse mixture, then a convolution with motor noise (paper Eq. 6–7). Critically, **the Switching observer uses the same parameters as the Basic Bayesian observer** — it adds no free parameters, which is why its win on AIC (mean AICBasic−AICSwitching ≈ 395 over subjects) is meaningful rather than a flexibility artefact.

The paper also already built three things you should treat as your true baselines, because your hypotheses are variations on them:

The *Switching posterior* observer (forms the posterior, switches between its mode and the prior mean) — this is the starting point of Hypothesis 2. The *Prior Learning Model*, which let prior strength grow within a block as `k_prior·(1 − e^(−t/τ))` with a learning-rate parameter τ — this is the paper's *sequential* learning model, and the thing Hypothesis 2's *online* updating must be compared against. And a *Bayesian observer with long-tailed prior* (prior = mixture of von Mises and circular uniform), which is the closest existing relative of Hypothesis 1 and which **the Switching model still outfit**.

In `Model_explainer.md` these correspond to the "readout" step of the estimator (§ *The math, in the order the code computes it*) and to the estimator file `SLGirshickBayesLookupTable.m`; in the notebook they live in `girshick_lookup` (§2) and `trial_loglike` (§4).

---

## Evidence and prior attempts from the two Neuromatch decks

The two decks in `Neuromatch Project PPTs/` are direct precedents — one per hypothesis — and their empirical results sharpen both the support and the critique below.

**Deck A — "Learning to Predict Without Knowing" (The Bayes Heuristics).** A single-prior variant (mean 225°, std 80° — i.e. the *weakest, most uncertain* prior; 226-trial run units at 6/12/24% coherence; 6 subjects, 24–66 runs). They built exactly Hypothesis 2's top-down piece: an **online Bayesian prior-learning model** (updating the prior belief over trials from feedback), paired with a bottom-up Random-Forest classifier asking whether learning is even detectable in raw behaviour. Headline result: a **learned von Mises prior beat a uniform prior on RMSE for every subject** (e.g. 147→77, 118→47, 138→61), so online prior learning demonstrably helps. But they also report that only a subset of subjects showed clear learning trajectories and that fits kept substantial residual noise. That caveat is informative rather than damning: they used the *hardest* condition (std 80, where the prior is weak and the online learning signal is smallest), so low leverage and noisy curves are expected — which is precisely the identifiability worry raised under Hypothesis 2, and an argument for testing online updating in the *narrower*-prior blocks and around block transitions instead.

**Deck B — "Neuromatch Final Presentation" (The Wiggly Caterpillars).** Taking the switching model as given, they asked what drives the per-trial prior-vs-sensory choice and fit a **mixed-effects logistic regression**: `chose_sensory ~ prior_std + coherence + prior_std×coherence + human_estimated_error + cumulative_proportion_of_choosing_prior + (1|subject)`. Two findings bear directly on your hypotheses. First, the **prior_std effect was non-monotonic** (relative to std 80, subjects used *more* sensory at std 10 but *more* prior at std 20/40) — which **directly contradicts the Switching observer's monotonic mixing law** `p_prior = k_prior/(k_prior+k_e)` and is the single strongest in-house motivation for Hypothesis 1: the mixing weight is not a concentration ratio and needs to be free/inferred. Second, they found **hysteresis** — subjects were less likely to switch to sensory after having relied on the prior (cumulative prior reliance was predictive) — so the choice carries an **online state**, tying H1's mixing weight to H2's online updating. Their own proposed future model is essentially Hypothesis 1 made data-driven: a logistic function of the influential variables sets the probability of relying on prior vs sensory, a latent decision is drawn from it, and the estimate is drawn from one of two learnable distributions — exactly the "infer the mixing weight, then select/sample" architecture recommended below, with the weight parameterised by covariates rather than by the ratio.

**The synthesis both decks point to.** Combined, they motivate a single model that both hypotheses are special cases of: a **two-component mixture** (prior-distribution vs likelihood-distribution) whose **per-trial mixing probability is an online logistic function of prior_std, coherence and their interaction, recent estimation error, and cumulative prior reliance**, with the prior itself **learned online**. Deck B supplies the mixing-weight structure (H1); Deck A supplies the online prior-learning machinery (H2); the concentration-ratio Switching observer is the constrained baseline both are tested against. In the notebook this is one function — the mixture readout of §2 with `w` replaced by a fitted logistic of trial covariates — wrapped by the §8 online-prior scaffold.

---

## Hypothesis 1 — support, critique, and how to build it

### Support

The idea is the paper's own suggested "Bayesian terminology" for its winning model, and it inherits that model's empirical backing: the same bimodal estimate distributions, the same prior-attraction bias, and the same no-extra-parameter parsimony. Structurally it is a textbook *mixture / causal-inference* generative model — a latent binary cause per trial with a probability-weighted combination of two sub-models — which is one of the best-validated schemes in perception. The multisensory literature has repeatedly shown observers compute exactly this kind of probability-weighted average over discrete causal hypotheses ([Körding et al., 2007](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943); [Shams & Beierholm, 2022, *Neurosci. Biobehav. Rev.*](https://www.sciencedirect.com/science/article/pii/S0149763422001087)), and the hierarchical-inference framing has a plausible cortical substrate in feedback/belief-propagation accounts ([Lee & Mumford, 2003](https://doi.org/10.1364/JOSAA.20.001434); Tenenbaum et al., 2011). Recent work extends causal inference to exactly your setting — a single stream with a prior whose relevance must be judged trial-by-trial in a possibly volatile world ([Heald et al. / "How relevant is the prior? Bayesian causal inference for dynamic perception in volatile environments," *eLife* 2024–25 reviewed preprint](https://elifesciences.org/reviewed-preprints/105385)).

### Critique

The sharp point is **identifiability**. If you fix the mixing weight to the paper's ratio `k_prior/(k_prior+k_e)`, then Hypothesis 1 is *mathematically the same model* as the Switching observer written differently — it will fit identically and yield no new prediction. The reformulation only earns its keep if it *changes* something the Switching observer cannot express. Two ways it can:

1. **Infer the mixing weight instead of hard-coding it.** In a proper generative model the trial-by-trial responsibility of the "peaked" cause is the *posterior* over the latent indicator given the measurement, p(z=peaked | m) ∝ p(m | peaked)·π. This depends on where the measurement falls (near vs far from the prior mode), so it is *stimulus-adaptive* in a way the fixed ratio is not. That is a genuinely different, and more normative, model — and it is testable because it predicts the switch rate should vary with the displayed direction, not just with the coherence/prior condition.

2. **Choose a combination rule.** Once you have p(z|m) you can (a) *model-average* (probability-weighted mean — smooth, unimodal, would *lose* the bimodality that motivates the whole project), (b) *model-select* (report the more probable cause — winner-take-all), or (c) *probability-match* (sample the cause with probability p(z|m)). The paper's Switching observer is essentially option (c)/(b); model-averaging (a) is the Basic Bayesian-like limit. This taxonomy (averaging vs selection vs matching) is the crux of the causal-inference literature and is where your model comparison should live.

A second critique is empirical redundancy: the paper *already tested* the nearest mixture model (von-Mises-plus-uniform prior) and Switching beat it. So Hypothesis 1 must be the trial-latent version above, not merely a heavier-tailed prior, or you will likely reproduce a result the authors already have. Encouragingly, Deck B's non-monotonic prior_std result shows the fixed-ratio mixing law is empirically wrong in this dataset, so a free/inferred weight is not just cleaner but *necessary* — this is the version to build, and the logistic-covariate parameterisation the deck proposes is a concrete, data-grounded way to do it.

A third, and important, alternative-cause critique: **the bimodality may not be a switch at all.** [Chetverikov & Jehee (2023, *Nature Communications*)](https://doi.org/10.1038/s41467-023-43251-w) decode *bimodal* motion-direction distributions directly from human visual cortex and find matching bimodality in behaviour, which they explain by a **bimodal sensory likelihood** (motion inferred jointly from velocity and the orientation "streak"), i.e. multiplicative Bayesian integration of two cues, no switching required. This is the strongest current competitor to your account and should be included as a control model (a bimodal-likelihood Basic Bayesian observer): if it fits your data as well as the switch, the switching interpretation is undercut.

### What to change (H1)

The mixture happens at the *combination/readout* step. In `Model_explainer.md`, that is the "BAYESIAN INTEGRATION → READOUT" block described under *The math, in the order the code computes it*; in the notebook it is the body of `girshick_lookup` (§2). Concretely: build a new `mixture_lookup(motdir, k_llh, mode_prior, k_prior, combine=...)` that (i) computes the likelihood-driven estimate distribution `P_e(e|d)` exactly as now, (ii) computes the prior component as a delta (or narrow von Mises) at `mode_prior`, and (iii) returns `w·P_prior + (1−w)·P_e`. Set `w = k_prior/(k_prior+k_llh)` to reproduce Switching as a sanity check; then replace `w` by the per-measurement posterior `p(z|m)` and marginalise over `m` (same marginalisation loop already in §2) to get the causal-inference version. Everything downstream — lapse, motor convolution, likelihood — is unchanged, so `trial_loglike` (§4) wraps it without modification.

---

## Hypothesis 2 — support, critique, and how to build it

### First, pin down "online" vs "sequential"

This distinction is doing all the work, so define it explicitly, because both readings appear in the literature:

- **Sequential / normative:** maintain an exact (or particle/Kalman) posterior over the prior's parameters and update it with each new observation using Bayes' rule — full memory of the relevant sufficient statistics, optimal given the assumed generative model. The paper's Prior Learning Model (exponential τ) is a coarse, offline-fit stand-in for this.
- **Online / algorithmic:** update a running point estimate one trial at a time with a bounded-memory rule — a delta rule / leaky integrator with a learning rate α — a single-pass approximation. This is the "online" you are contrasting against "sequential."

Stating which one you mean matters because they make different predictions at exactly the moments your experiment probes: the four block transitions where the prior width changes.

### Support

Deck A is your strongest, most direct support: an online Bayesian prior-learning model already fits these subjects and beats a uniform-prior baseline for everyone. Beyond that, the premise that people update priors online is well established. [Norton, Acerbi, Ma & Landy (2019, *J. Neurophysiol.*)](https://www.cns.nyu.edu/malab/static/files/publications/2019%20Norton%20Acerbi%20Ma%20Landy.pdf) show human observers track changes in prior probability trial-by-trial and compare delta-rule against Bayesian updating directly — a close methodological template for what you want. The delta-rule/adaptive-learning-rate tradition ([Nassar et al., 2010]; [Wilson, Nassar & Gold, 2013, "a mixture of delta-rules"] — cited in the paper itself) and the volatility-aware Bayesian account ([Behrens et al., 2007]) provide the online rules to test. And the paper's own data are *consistent* with fast online learning: prior strength reached ~95% within ~45 trials (τ ≈ 14.9), and first-100 vs last-100-trial bias slopes correlated r = 0.84 — evidence that some within-block updating is really happening.

### Critique

That same fast, stable learning is the main threat to the hypothesis. If the prior is essentially learned within ~45 of ~8,000 trials, then for the overwhelming majority of trials an online model is *indistinguishable* from the fixed-prior Switching posterior — the online machinery only bites in the brief transients after each block change. So the hypothesis has **low statistical leverage** unless you specifically model and weight the post-transition trials, and unless the design has enough block transitions per subject to estimate the learning rate. Concretely, budget your model comparison around transition-locked trials, not the whole block.

Two further cautions. **Compounded stochasticity / identifiability:** you are stacking a discrete switch on top of a noisy online learner, so switch-rate parameters and learning-rate/learning-noise parameters can trade off; fit them jointly on simulated data first and check parameter recovery before trusting fits to subjects. **Choice of "online" rule is a hypothesis, not a detail:** a fixed-α delta rule, an adaptive-α rule, and a changepoint model make *different* transient predictions, and the interesting science is which one humans use — so plan to compare them, not just to adopt one.

### A stronger version worth adopting

Because your prior changes in discrete blocks, the normatively-matched online model is **Bayesian online changepoint detection** ([Adams & MacKay, 2007](https://lips.cs.princeton.edu/pdfs/adams2007changepoint.pdf)): it maintains a distribution over "time since the last change" and updates the prior online, automatically resetting learning at block boundaries — precisely the structure of your experiment. Its reduced/approximate cousins (delta-rule mixtures, [Wilson et al., 2013]) and the hierarchical/volatility formulations ([Behrens et al., 2007]; the Hierarchical Gaussian Filter, [Mathys et al., 2011](https://www.frontiersin.org/articles/10.3389/fnhum.2011.00039/full)) give you a ladder of online models from cheap-and-approximate to normative-and-hierarchical, all of which slot into the same estimator.

### What to change (H2)

The learning lives *outside* the estimator; the estimator itself is untouched. In `Model_explainer.md` this is the boundary between the per-trial loop in `SLgetLoglBayesianModel.m` and the condition-wise lookup table — i.e. instead of selecting one of four fixed `k_prior` by block, you feed a *time-varying* `(mode_t, k_prior_t)` per trial. The notebook already scaffolds this in §8 (`online_prior_stats`, a delta rule over the circular resultant vector): replace its update rule with your chosen online rule (fixed-α, adaptive-α, or changepoint), and at trial *t* call `girshick_lookup(..., mode_prior=mode_t, k_prior=k_t, readout="MAP")` — MAP being the "posterior mode" readout the Switching-posterior needs — then, as in the Switching model, mix that with a delta at the prior mean using `p_prior = k_t/(k_t+k_e)`. Fit the learning rate by maximum likelihood exactly like any other parameter (notebook §7). Because the lookup table now depends on the evolving `k_t`, drop the per-condition caching for the learning trials (noted in §9) or cache on a discretised grid of `k_t`.

---

## Cross-cutting alternative: the sampling / resource-rational bridge

Both hypotheses are, at heart, claims that people *approximate* Bayesian integration cheaply. The most economical formalisation of that is **posterior sampling with very few samples**: a one-sample readout of the posterior *is* a stochastic switch between the likelihood peak and the prior peak, which links your Switching/mixture models to the Sampling observer the paper tested and to the sampling readout already implemented in the notebook (§3). The resource-rational and sampling literature ([Vul, Goodman, Griffiths & Tenenbaum, 2014, "One and Done"]; [Sanborn & Chater, 2016]; [recent resource-rational perceptual work, 2021–23](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003661)) argues suboptimal-looking behaviour like switching is the optimal use of limited compute. Practically, this suggests a fourth model to run alongside H1 and H2: a **finite-sample posterior readout** (draw *n* samples, report their mode/mean), with *n* a free parameter — as *n*→∞ it becomes the Basic Bayesian observer, and at *n*=1 it approximates the switch, so it nests your competitors and turns "how many samples" into an empirical question. This is a small change to the readout block in `girshick_lookup` (§2) — sample from `post[:,m]` instead of taking its argmax/mean.

---

## Suggested model-comparison plan

Fit and compare, per subject, on the same estimate-distribution likelihood the paper uses (notebook §4, `trial_loglike`; AIC/BIC and cross-validated log-likelihood): the Basic Bayesian and Switching observers as anchors; H1 in both its fixed-ratio form (identity check against Switching) and its inferred-responsibility causal-inference form with averaging/selection/matching readouts; the **logistic-covariate mixture** from Deck B (mixing weight = fitted logistic of prior_std, coherence, their interaction, recent error, and cumulative prior reliance) as the data-driven H1; a bimodal-likelihood Basic Bayesian observer as the Chetverikov–Jehee control; H2's Switching-posterior with each online rule (fixed-α, adaptive-α, changepoint), evaluated with extra weight on transition-locked trials and compared against both the paper's exponential-τ sequential model and Deck A's online-VM-vs-uniform contrast; and the finite-sample readout as the resource-rational nest. Validate every new model by parameter recovery on simulated data before interpreting fits — the switch-plus-learning models especially. Two empirical targets any winning model must reproduce, both from Deck B: the **non-monotonic** dependence of sensory reliance on prior_std, and the **history/hysteresis** effect of past prior reliance.

---

## References

*Neuromatch project decks (in `Neuromatch Project PPTs/`):*
The Bayes Heuristics pod. *Learning to Predict Without Knowing: Implicit Priors Under Ambiguous Sensory History* (`LearningtoPredictwithoutKnowing-2.pdf`). — online Bayesian prior-learning model + Random-Forest learning-detection.
The Wiggly Caterpillars pod. *Human Motion Task / Neuromatch Final Presentation* (`NeuromatchFinalPresentation1.pdf`). — mixed-effects logistic regression of prior-vs-sensory choice; proposed logistic-mixture generative model.

*Literature:*
Adams, R.P. & MacKay, D.J.C. (2007). *Bayesian Online Changepoint Detection.* arXiv:0710.3742. https://lips.cs.princeton.edu/pdfs/adams2007changepoint.pdf
Behrens, T.E.J. et al. (2007). Learning the value of information in an uncertain world. *Nat. Neurosci.* 10, 1214–1221.
Chetverikov, A. & Jehee, J.F.M. (2023). Motion direction is represented as a bimodal probability distribution in the human visual cortex. *Nat. Commun.* 14, 7634. https://doi.org/10.1038/s41467-023-43251-w
Körding, K.P. et al. (2007). Causal inference in multisensory perception. *PLoS ONE* 2, e943. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943
Laquitaine, S. & Gardner, J.L. (2018). A Switching Observer for Human Perceptual Estimation. *Neuron* 97, 462–474. https://www.cell.com/neuron/fulltext/S0896-6273(17)31134-0
Lee, T.S. & Mumford, D. (2003). Hierarchical Bayesian inference in the visual cortex. *J. Opt. Soc. Am. A* 20, 1434–1448. https://doi.org/10.1364/JOSAA.20.001434
Mathys, C. et al. (2011). A Bayesian foundation for individual learning under uncertainty (Hierarchical Gaussian Filter). *Front. Hum. Neurosci.* 5, 39. https://www.frontiersin.org/articles/10.3389/fnhum.2011.00039/full
Nassar, M.R. et al. (2010). An approximately Bayesian delta-rule model of learning rate. *J. Neurosci.* 30, 12366–12378.
Norton, E.H., Acerbi, L., Ma, W.J. & Landy, M.S. (2019). Human online adaptation to changes in prior probability. *PLoS Comput. Biol.* / *J. Neurophysiol.* https://www.cns.nyu.edu/malab/static/files/publications/2019%20Norton%20Acerbi%20Ma%20Landy.pdf
Shams, L. & colleagues (2022). Bayesian causal inference: A unifying neuroscience theory. *Neurosci. Biobehav. Rev.* 137, 104619. https://www.sciencedirect.com/science/article/pii/S0149763422001087
Tenenbaum, J.B. et al. (2011). How to grow a mind: statistics, structure, and abstraction. *Science* 331, 1279–1285.
Vul, E., Goodman, N., Griffiths, T.L. & Tenenbaum, J.B. (2014). One and done? Optimal decisions from very few samples. *Cogn. Sci.* 38, 599–637.
Wilson, R.C., Nassar, M.R. & Gold, J.I. (2013). A mixture of delta-rules approximation to Bayesian inference in change-point problems. *PLoS Comput. Biol.* 9, e1003150.
"How relevant is the prior? Bayesian causal inference for dynamic perception in volatile environments" (2024–25 reviewed preprint, *eLife*). https://elifesciences.org/reviewed-preprints/105385
