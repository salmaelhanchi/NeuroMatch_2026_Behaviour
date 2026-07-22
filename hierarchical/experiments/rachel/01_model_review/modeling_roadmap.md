# A first-time modeler's roadmap to the abstract's aims

For: someone building their first model. This walks you from "what is a model,
really" through the exact order of operations that gets you to the abstract's
goals — and flags the one thing about your aims that needs adjusting.

---

## Part 0 — The four ideas you need before touching code

Everything in this project is built from four concepts. Get these and the rest
is bookkeeping.

1. **A generative model is a recipe for making fake data.** It says: "given the
   stimulus on this trial and these parameter values, here is the *probability of
   every possible response the subject could give*." For the switching observer,
   that recipe is: draw a noisy sense of the direction, either trust it or fall
   back on the prior mean (a coin-flip weighted by reliability), occasionally
   lapse to a random guess, then blur it with motor noise. The output is a
   360-long probability vector over possible reported directions.

2. **The likelihood is how well the recipe explains the *real* data.** Take the
   subject's actual response on a trial, look up its probability under the model's
   360-long vector. Multiply that across all trials (or, because the numbers get
   tiny, *add the logs*). A model+parameters that assigns high probability to what
   the subject actually did has a high likelihood. We minimize the **negative log
   likelihood (NLL)** — same thing, flipped so "smaller = better."

3. **Fitting = searching parameter space for the lowest NLL.** You hand an
   optimizer (Nelder–Mead, CMA-ES) the NLL function; it tries parameter
   combinations until it can't lower the NLL further. The result is the
   "best-fit parameters" for that subject. Your code already does this.

4. **Model comparison must penalize complexity.** A model with more knobs will
   always fit at least as well, so raw NLL is unfair. **AIC = 2·(#params) + 2·NLL**
   and **BIC** (heavier penalty) add a complexity tax. Lower is better. Two models
   within a few AIC of each other are *tied*, not ranked — treat small gaps as
   noise.

If you can explain those four to a teammate, you understand what every script in
the folder is doing.

---

## Part 1 — Does the abstract's goal make sense? Yes, but reframe it

**Short answer: the *question* is excellent and worth doing; the *specific model*
the abstract names was already tested by the original authors and lost. Your real
contribution is one axis further out — and it's a good one.**

You said you were inspired by the paper's conclusion. You were reading the right
paragraph — but read it all the way to the end. In the Discussion (p11–12) the
authors do three things that bear directly on your abstract:

1. **They propose your model, by name.** They sketch "a hierarchical Bayesian
   observer ... in which perceptual judgments are determined by hierarchical
   beliefs, or hyper-priors ... that motion directions were drawn from either a
   uniform or a peaked distribution at each trial, with probabilities determined
   by the ratio of likelihood and prior strengths." That *is* the abstract's
   mixed hyper-prior (peaked von Mises + uniform).

2. **They say it would just be the switch in disguise.** That formulation, they
   write, "would effectively amount to a reformulation of our Switching observer
   in Bayesian terminology." This matches what the code already found: on subject 1
   the integration model won only by turning the mixture *off* (α→1), and its
   mixture weight plays the role of the switch probability.

3. **They already tried the static mixture and it lost.** "We have similarly
   explored ... mixtures of uniform and von Mises distributions, and found that
   while these could produce bimodal estimate distributions, the Switching model
   provided better fits."

So if the abstract's aim is read as "test whether a *static* mixed hyper-prior can
replace the switch," the honest answer is: **the original paper did that and
reported it doesn't.** You'd be reproducing a known negative result.

**But the authors left one door open, and your abstract walks through it — you
just need to say so explicitly.** They fit *static* priors; they assumed learning
was already complete and never modeled it. The sentence they *didn't* get to is:
what if the prior's strength is **learned trial-by-trial**, and the bimodality
comes from *mixing over a belief that is still moving*? That is the one thing none
of the paper's models did, and it is exactly what `online_learner`, `hb_integration`'s
κ-learning, and `asymptote_transient` add. The paper even hands you the caveat as
an invitation: "We cannot preclude ... some distributional form ... would allow a
Bayesian model with multiplicative integration to better explain the data."

**Reframed aim (this is defensible and new):**
> Laquitaine & Gardner fit static priors and showed a within-trial *switch* beats
> a static mixed hyper-prior. We ask whether the switch is still needed once the
> observer *learns* its prior online: does trial-by-trial hierarchical Bayesian
> inference reproduce the bimodality — and the newly-measured within-block
> *transient* in prior reliance — without an explicit selection mechanism?

That keeps everything you've built, honors the inspiration, and stakes out
territory the paper explicitly did not cover.

---

## Part 2 — The order of operations (do NOT fit real data first)

The single most common beginner mistake is to fit real data and start
interpreting parameters before checking the model can be trusted. Do these **in
order**. Each step is a gate: don't proceed until it passes.

1. **Simulate.** Pick parameter values by hand, generate fake responses from the
   model. Plot them. Do they look like the phenomenon (bimodal in the right
   regime)? This is `*_verify.py` / `*_simulate.py`. → Builds intuition and
   catches gross bugs. *Status: done, passing.*

2. **Parameter recovery.** Simulate data from *known* parameters, then fit and
   check you get those parameters back. If you can't recover a parameter from data
   you generated yourself, any value you fit to a real subject is meaningless.
   This is where you'll find `k_e[0.24]` is unreliable (it isn't recoverable — a
   real limitation to report, not hide). → *Status: mostly done; note the ridge.*

3. **Model recovery (the step people skip — don't).** Simulate data from model A,
   fit *both* A and B, confirm AIC picks A. Then swap. If your two models are so
   similar that you can't tell them apart even when you *know* the true generator,
   then any "A beats B on real data" claim is unsupportable. Given the paper's
   warning that integration and switching are near-reformulations of each other,
   **this is the load-bearing analysis of your whole project.** → *Status: partly
   built (Phase 7); make sure it covers switch-vs-integration, not just
   learning-vs-no-learning.*

4. **Fit real data — all 12 subjects, every model, equal effort.** Only now.
   Multi-start each fit (fitting is a hill-climb that can get stuck; start it from
   several places and keep the best). Save NLL, AIC, BIC, parameters, convergence
   flags. → *Status: only subjects 1 & 3 so far; this is the main compute job
   left.*

5. **Model comparison + posterior-predictive checks.** Tabulate AIC/BIC per
   subject. Then the check that actually convinces people: simulate from each
   fitted model and overlay the predicted response distribution on the subject's
   real histogram. A model can win on AIC and still visibly miss the data shape —
   the eye test matters. → *Status: started for integration; extend to all.*

6. **The one decisive experiment (cheap, high-value, not yet run).** The switch
   and the learning-integration model make *different* predictions about *when*
   the bimodality appears:
   - **Switch:** every qualifying trial is bimodal — two peaks *within a trial*.
   - **Learning integration:** each trial is *unimodal*; the two peaks only appear
     when you *pool* early-block trials (belief weak → mass at stimulus) with
     late-block trials (belief sharp → mass at prior).
   So: **look only at stable late-block trials, after the belief has converged. Is
   the data still bimodal there?** If yes → a within-trial switch is needed, the
   integration story fails. If the bimodality only survives pooling across the
   learning transient → integration wins. This single analysis adjudicates your
   central question and needs *no new fitting*. Run it early.

---

## Part 3 — How this maps onto your team's tasks

Your `project_tasks.md` already assigns leads. Here's how the tasks slot into the
gated order above, so nobody fits real data before the model is trustworthy:

| your task | lead | maps to step | note |
|---|---|---|---|
| HB implementation | Salma | 1 | built; the read-out convention (read-out-then-average) is a real choice — see model_review C1/C2 |
| Data prep + fitting | Romi | 4 | the all-12 batch; use the multi-started fitter, not single-start |
| Model validation | Anirban | 2 + 3 | **make sure model recovery covers switch-vs-integration** — the key check |
| Switching implementation | Anirban | (baseline) | done; it's the comparison anchor |
| Model comparison | Rachel | 5 + 6 | **own the late-block bimodality test (step 6)** — it's your headline result |
| Behavioral validation | Romi | cross-check | does measured estimation error track the model's learning curve? strengthens the story |
| Integration + presentation | Valeria | — | frame around the *learning transient* + *temporal-vs-within-trial* question |

---

## Part 4 — Three habits that keep a first model honest

1. **Never trust a fitted parameter you haven't recovered.** (Step 2.) If recovery
   fails for a parameter, say so and stop interpreting it. `k_e[0.24]` is your
   worked example.
2. **A model that fits better is not automatically "true."** It might just have
   more knobs, or be a relabeling of the other model (the paper's own warning).
   That's why steps 3 and 6 exist — they test *distinguishability* and *mechanism*,
   not just fit.
3. **Report the negative and the tie.** Your subjects-1-and-3 result is 1–1, and
   the subject-3 switch-family gaps are a tie. Saying so plainly is what makes the
   subject-1 win credible. The original authors modeled this honesty — their
   "we cannot preclude" sentence is why their paper is trusted.

---

## What I'd do next, concretely

If you want to move the science forward this week, in priority order:

1. **Run the late-block bimodality test (step 6).** No new fitting; directly tests
   the abstract's core question. I can run this now.
2. **Verify model recovery distinguishes switch from integration (step 3).** If it
   can't, that's the most important thing to know before fitting 12 subjects.
3. **Launch the all-12 fair fit batch (step 4).** The compute job that everything
   else waits on.
4. **Rewrite the abstract** around the learning transient and the
   temporal-vs-within-trial question (Part 1), citing the paper's own open door.

Tell me which and I'll start.
