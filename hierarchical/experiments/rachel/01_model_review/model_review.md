# Code & model review — `hierarchical/` switching-observer variants

Reviewer: Switching Observer specialist. Scope: the four observer models in
`hierarchical/`, their shared foundations, verification scripts, and the
subject-1/subject-3 fit results. Read against Laquitaine & Gardner (2018).

---

## 1. What is in the folder

| file | model | new mechanism vs. paper | # params |
|---|---|---|---|
| `switching_observer.py` | Static Switching observer (the paper's winner) | none — faithful port | 9 |
| `online_learner.py` | Hierarchical **Online** Switching observer | prior strength `k` is a *latent learned online* from feedback (belief `b_t(k)`, forgetting `λ`) | 6 |
| `asymptote_transient.py` | **Asymptote + Transient** Switching observer | per-block asymptote `k_asym` **+** within-block exponential transient `k_eff(t)` toward it (asymmetric `τ`) | 11 |
| `hb_integration.py` | Hierarchical Bayesian **integration** observer (the abstract's model) | **no switch**: mixed hyper-prior `α·VM(225,κ)+(1−α)/360`, MAP read-out, online learning of `κ` | 7 |

Foundations (`circular.py`, `bayes_lookup.py`, `online_helpers.py`) are faithful
ports of Laquitaine's MATLAB (`vmPdfs.m`, `SLGirshickBayesLookupTable.m`,
`SLcircConv.m`), on the 1° grid `θ∈1..360`, prior mean fixed at 225°.

---

## 2. How they are designed — the architecture is sound

**The design spine is a reduction hierarchy.** Every extension provably contains
the paper's static observer as a limiting case, verified numerically:

- `online_learner`: belief pinned to a fixed point mass ⇒ static observer
  (matches `switching_observer.py` to 1e-17).
- `asymptote_transient`: `carryover=False` ⇒ `k_eff ≡ k_asym` ⇒ exact static
  (ΔNLL = 0), and `τ→0` ⇒ static except trial 1 of each block.
- `hb_integration`: `α=1` reproduces `girshick_map_lookup` exactly (max|Δ|≈1e-17)
  — the mixture genuinely nests the paper's single-von-Mises machinery.

This is the right way to build model extensions. Because each new model *nests*
the baseline, a win by the extension is interpretable ("the added mechanism
helps"), and a **collapse back to the baseline is itself an informative negative
result** (and it happened — see §5). This is stronger practice than comparing
non-nested models, which is what most model-comparison papers do.

**The tractability argument is correct and non-obvious.** All three learning
models make the belief path *deterministic given the observed feedback sequence
and parameters* — because feedback is the exogenous displayed direction, nothing
on the belief path is sampled. Therefore the trial likelihood is an exact sum of
per-trial log-probs with **no particle filter**. This is the key move that makes
maximum-likelihood fitting feasible, and it is valid.

**Engineering is careful:** precompute/caching mirrors the MATLAB structure
(read-outs built once per parameter set, then belief enters as cheap scalar
expectations); numerically-stable von Mises (`i0e`, `exp(k cosθ − k)`); the
Murray–Morgenstern closed form rescues underflowed posteriors for strong priors.
Verification suites exist and pass (e.g. HB integration 12/12), including a
batch-vs-recursive belief cross-check that shares no code with the online path.

---

## 3. Implementation critique — concrete concerns

**C1. `hb_integration` read-out-then-average is not textbook Bayesian
integration (design fork F2).** The model computes the MAP read-out *per κ* and
then averages the read-outs over the belief, `Σ_κ b_t(κ)·readout(κ)`. The
textbook hierarchical-Bayesian observer would marginalize κ into a *single*
posterior and read out *once*. Because MAP is nonlinear, `E_κ[MAP(κ)] ≠
MAP(E_κ[posterior])`. The authors chose F2 deliberately, to match
`online_learner`'s H4 convention so any fit difference is switch-vs-integration
rather than a marginalization mismatch — a defensible call, and they document it.
But it has a real consequence: **averaging point-estimates across the belief is
itself a mixture-over-κ**, so part of the model's emergent bimodality comes from
the marginalization convention, not purely from the mixed prior. The "integration"
label should be qualified.

**C2. The integration model's bimodality is *temporal*, not within-trial — and
this is the crux the comparison hasn't fully exploited.** The posterior-predictive
check shows that *per trial* the integration model is **unimodal**; the two peaks
appear only when trials are **pooled** over a block in which the believed prior SD
swings ~8°→100°. Early (weak belief) → mass at the stimulus; late (sharp belief)
→ mass at the prior. The switch model, by contrast, produces a **within-trial**
bimodal response distribution on every qualifying trial. These are *different,
falsifiable predictions*. The sharp test: take a **stable late-block window**
(belief converged) and ask whether the data are still bimodal there. If yes, a
within-trial switch is needed and the integration account fails; if the data
bimodality only survives pooling across the learning transient, integration wins.
This discriminator is stated but not yet run — it is the single most valuable
next analysis.

**C3. The winning integration fit rejects the abstract's own mechanism.** On
subject 1 the fit drove `α→1.0`, switching *off* the uniform component. By the
reduction (T1) that is the plain single-von-Mises Girshick integration observer —
**not** the mixed hyper-prior the abstract proposes. So the headline "integration
beats switching (ΔAIC≈134 fair)" is really "*online-learning Bayesian
integration* beats switching," with the mixture doing none of the work. The team
is honest about this, but the abstract's mechanistic claim needs rewriting: on
current evidence the uniform-mixture ingredient is not what carries the win.

**C4. The pure `online_learner` is structurally mismatched to a multi-block
design.** It carries **one** belief over κ that must chase a target which jumps
across 4 different block widths every ~200 trials. With small `λ` it lags badly
at each change; the fits bear this out (subject 3 `λ≈0.003` ≈ pure accumulation →
the belief converges to a session-average κ and barely moves per block). That is
exactly why it has *fewer* parameters than static (6 vs 9) yet often fits *worse*,
and why `asymptote_transient` (per-block asymptotes + transient) beats it. Plainly
stated: **the single-latent online learner conflates "learning the prior" with
"tracking block switches,"** and cannot represent 4 distinct block priors well. It
is best read as a reduction anchor and stepping-stone, not a serious contender.

**C5. `k_like[0.24]` is unidentified on real data.** Across fits it runs to
7190 / 566 / ∞. Once `k_e ≫ k_motor` the motor noise dominates the read-out and
the likelihood is flat in `k_e` — correctly diagnosed as a ridge, but it recurs
in *every* model and undercuts the parameter-recovery claim for the
high-coherence sensory parameter specifically (recovery was shown on *simulated*
data where the ridge is milder). Fix: bound `k_e`, or reparametrize as an
effective precision capped by `k_motor`, and report `k_e[0.24]` as
"≥ (motor-limited)" rather than a point value.

**C6. Cross-validation is not a strict out-of-sample forecast of learning.** The
`cv` mode preserves trial order but the belief still sees held-out feedback
(feedback = the exogenous stimulus direction, not the response). So it scores
predictive fit of *responses* with order intact, not a causal forecast of the
*learning*. For discriminating the learning mechanism this is weak. It also
currently scores integration only — the other three models need CV before any CV
verdict.

**C7. κ-grid resolution asymmetry.** `online_learner` uses 40 κ points;
`hb_integration` uses ~15 (for speed, because its read-out *location* depends on
κ). Since the emergent pooled bimodality is built from per-κ read-outs, 15 points
may under-resolve its shape. A 15-vs-30 convergence check on AIC and the PPC
would close this.

**C8. Statistical caution the docs already flag, restated for the team.** Only
subjects 1 and 3 are fit — a 1–1 split. Subject-1 leads (~134 AIC) are
meaningful; the subject-3 switch-family ordering (~14 AIC spread across
static/online/AT) is a **tie**, not a ranking. Nothing is settled until the
all-12 batch runs, each model with equal multi-start effort (`fair_refit.py`, not
single-start). The `k_e` ridge and single-fit AIC noise (~tens of AIC) mean
current numbers are directional.

---

## 4. Strengths worth preserving

- Spec-first discipline (`generative-model.md` as a written oracle checked in
  later phases) — rare and valuable.
- Nesting/reduction verified to machine precision for all three extensions.
- Exact deterministic likelihood (no particle filter) — correct and fast.
- Honest documentation of failure modes (subject-5 stuck fit, `k_e` ridge,
  α-boundary win, single-fit AIC noise, CV caveat). The `hb_integration.md`
  "Honest status" section is a model of how to report partial results.
- The fairness correction (`fair_refit.py`, equal multi-start for every model)
  was applied *after* an initial unfair comparison flattered integration — and it
  shrank integration's subject-1 lead from ~314 to ~134. Catching and fixing that
  is exactly the right instinct.

---

## 5. Do they teach us anything new vs. the paper's variants?

**The paper's variants were all static.** Laquitaine & Gardner compared observers
that differ in *how they combine or select* prior and evidence within a trial
(Bayesian integration vs. switching; with/without a cardinal prior), with the
prior strength a **fixed fitted parameter per block**. The paper noted subjects
"quickly learn the priors" but did **not** model the learning — it fit the
asymptotic prior and assumed learning was complete. Its central result:
integration predicts unimodal estimates, the data are bimodal, so a **within-trial
switch** wins.

These four models add a **temporal / hierarchical axis the paper never modeled**,
and that is where the new content is:

1. **A real dynamical signature the static model cannot produce.** The empirical
   switch-probability curve (`switch_probability_curve.png`) shows prior-chosen
   fraction is high right after a block change and decays over ~15–20 trials
   (0.40→0.22, p<1e-4). The paper's constant-per-block switch weight *cannot*
   represent this. `asymptote_transient` captures it directly and tracks the
   empirical curve about twice as well as the pure online learner (SSE 0.094→0.051).
   **This is the strongest genuinely-new finding** — a within-block learning
   transient in how strongly subjects rely on the prior, which the original
   analysis missed.

2. **A legitimate reopening of the paper's central conclusion.** The HB
   integration model raises a sharp alternative: the bimodality that motivated the
   switch might emerge from **temporal mixing of a drifting Bayesian belief**
   rather than a within-trial selection mechanism. On subject 1 the
   integration+learning model *fairly* beats every switch model. That is a real,
   non-trivial challenge to "bimodality requires a switch."

3. **A more parsimonious account of prior acquisition.** Replacing 4 fitted
   per-block prior strengths with a single learning process (online `λ`, or
   AT's `τ`) reframes the priors as *emergent* rather than *fitted* — a stronger
   scientific claim when it holds.

**But the challenge is partial and, so far, self-undermining in two ways:**

- The integration win comes by **shedding the abstract's mixture** (α→1, C3), so
  what actually beats the switch is Bayesian integration *plus online learning*,
  not the proposed mixed hyper-prior.
- The integration model's bimodality is **pooled/temporal, not within-trial**
  (C2), which is a *different* empirical claim than the paper's — and it has not
  yet been tested against a stable late-block window, the analysis that would
  actually adjudicate switch-vs-integration.
- The best per-subject model so far (`asymptote_transient`, subject 1) is itself
  a **switching** model with learning added — i.e. current evidence, taken at face
  value, says the paper's switch is *right* and merely needs a learning transient
  bolted on, not that it should be replaced by integration.

**Net verdict.** Yes, they teach us something new: a within-block *learning
transient* in prior reliance that the static model cannot produce, and a
credible, sharply-testable alternative hypothesis (temporal mixing vs. within-trial
switch) for the origin of the bimodality. What they have **not** yet done is
overturn the switching account — on 2 subjects the result is 1–1, the integration
win discards its own headline mechanism, and the strongest model remains a switch.
The decisive experiment (C2: is the data bimodal within a converged late-block
window?) is designed but unrun.

---

## 6. Recommended next steps (in priority order)

1. **Run C2** — the within-trial-vs-temporal bimodality test on stable late-block
   windows. This adjudicates the paper's core claim and is cheap (no new fitting).
2. **All-12-subject batch**, every model multi-started fairly (`fair_refit.py`),
   per-subject jobs to avoid the results-file race. Report AIC **and** BIC.
3. **Decide α** — profile α-free vs α-fixed=1, since the winning fit sits on the
   boundary. State cleanly whether the uniform mixture ever helps.
4. **Fix `k_e[0.24]` identifiability** (bound or reparametrize) before trusting
   any high-coherence sensory estimate.
5. **Extend CV to all four models** and note the belief-sees-feedback caveat in
   any CV claim.
6. **κ-grid convergence check** (15 vs 30) for the integration model.
7. Reframe the abstract around the *learning transient* (the robust new finding)
   and the *temporal-vs-within-trial* question (the sharp new hypothesis), rather
   than the mixed hyper-prior mechanism, which the data have not supported.
