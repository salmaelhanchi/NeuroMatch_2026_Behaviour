# HB - Rachel (hb_integration) — how to explain it to the team

## One sentence
Our observer does proper Bayesian estimation on every trial — combining what it
sees with what it expects — but it *learns how confident to be in its
expectation* as it goes, tightening or loosening its sense of the prior trial by
trial from the feedback.

## 30-second version
The paper's switching observer flips a coin each trial: use the prior OR use the
evidence. Ours never flips — it always INTEGRATES the two into one posterior
(standard Bayes). What makes it hierarchical is that the *width of the prior*
isn't fixed: it's a belief the observer updates. Early in a block it isn't sure
how tight the prior is; as feedback arrives it learns whether directions are
tightly clustered (narrow prior) or spread out (wide prior), and leans on the
prior more or less accordingly. The thing being learned is CONFIDENCE IN THE
PRIOR, learned online.

## The mechanics in five steps (maps to the M1–M5 equations doc)
1. PRIOR: a bump at 225° plus a small flat floor — α·(von Mises) + (1−α)/360.
   α is a fixed mixing weight; the bump's WIDTH is what gets learned.
2. EVIDENCE: a von Mises likelihood centered on the true direction, width set by
   coherence (3 fitted reliabilities, one per coherence level).
3. COMBINE (Bayes): likelihood × prior, read out the peak (MAP) — but done
   across a whole grid of possible prior widths (κ), then averaged, weighted by
   how much the observer currently believes each width.
4. LEARNED LATENT: that belief over widths, b_t(κ). Each trial: partially forget
   (drift toward the starting belief), then Bayes-update from the feedback
   direction. Over a block the belief concentrates on the true width.
5. MOTOR + LAPSE: added last, so the output is a realistic response distribution.

## Two things that make it interesting (lead with these)
- BIMODALITY IS EMERGENT, NOT BUILT IN. When the stimulus is far from 225°, the
  posterior naturally develops two peaks — one near the evidence, one pulled
  toward the prior — without ever telling the model to switch. The switching
  observer has to IMPOSE bimodality by committing to one source; ours produces
  it as a side effect of honest integration. This is the conceptual payoff of
  the abstract.
- THE LEARNED LATENT ACTUALLY WORKS. On the real data, the belief's implied
  prior width tracks each block's true width across all 12 subjects (80° blocks
  → ~22–61°, down to 10° blocks → ~10–12°). The "learning confidence in the
  prior" story shows up in the fits, not just the design.

## The honest caveat to have ready
α is a FIXED constant, not the thing being learned — the model learns confidence
by updating the prior's WIDTH (κ), not by adjusting α. Easy to muddle in a talk,
so be crisp: "we learn how tight the prior is, trial by trial." (There is a
separate free-α question in the background, but for describing what the model
DOES, width-learning is the headline.)

## Framing against the other three models, if asked
- vs SWITCH (the paper): they commit per trial with a fixed prior width; we
  integrate and learn the width.
- vs HB - Salma: same "learn the prior's width" idea, but she integrates the
  belief into ONE prior BEFORE reading out, while we average read-outs AFTER.
  Agree near the prior, diverge in the far / bimodal regime.
- vs HB - Rommana: she learns the mixture WEIGHT (how much to rely on the prior)
  rather than the prior's WIDTH — a different latent entirely, mechanically
  closer to the Switch.

## Seven fitted parameters (if someone asks for the parameter list)
3 k_like (sensory reliability per coherence) + α (fixed prior mixture weight)
+ k_motor (motor noise) + p_random (lapse) + λ (forgetting rate of the belief).
The prior concentration κ is NOT fitted — it is the learned latent.
