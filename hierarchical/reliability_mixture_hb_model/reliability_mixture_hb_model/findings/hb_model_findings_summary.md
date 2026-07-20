# HB model fitting: two findings from the 12-subject fit

Written after fitting the reliability-mixture HB model across all 12 subjects (see
`HB_model_handoff.md` and `kaggle_hb_multisubject_fit.ipynb`). Two things came out
of this fitting exercise worth documenting before moving to the final AIC
comparison against the paper's Switching observer (item 5): a structural issue in
how one parameter behaves, and a genuine correlation between the model and
independent behavioral data.

---

## Finding 1: `k_prior` is meaningless for some subjects in wide-prior blocks

### The problem, in one sentence

For several subjects, the model's estimate of "how tightly this subject clusters
their guesses around the fixed anchor point (225 degrees) in the widest (80-degree)
blocks" isn't actually being measured by the data. It just lands wherever the
optimizer happens to push it, sometimes a normal-looking number, sometimes as
extreme as we let it go.

### How we noticed

Each subject's model has a parameter called `k_prior` per block-width condition
(80, 40, 20, 10 degrees), a "concentration" number. Higher means the subject's
guesses cluster tightly around one point; lower means they're spread out. We kept
raising the maximum value we allowed this number to reach: 150, then 1000, then
5000. For most subjects, it settled comfortably below whatever ceiling we set,
a clear sign of a real, stable value. But for several subjects, it just kept
climbing to match whatever new ceiling we gave it, every single time.

That's the tell: **a number genuinely being measured by data settles somewhere.
A number that isn't measured just keeps climbing to match whatever limit you
allow.**

### Why this happens

The model also tracks a second thing: how much a subject is actually leaning on
that fixed anchor, moment to moment, called `prior_reliance`. This updates every
trial based on whether recent behavior "agreed" with the anchor.

The problem: the same clustering number described above is *also* used to decide
how strict that agreement check is. That creates a feedback loop:

1. In wide (80-degree) blocks, real behavior naturally varies a lot from trial to
   trial, that's expected, not a problem with the subject.
2. If the clustering number happens to drift even moderately large, the agreement
   check becomes strict enough to read this normal variation as "total
   disagreement."
3. That drags `prior_reliance` toward zero for this condition.
4. Once `prior_reliance` is near zero, the anchor barely factors into the model's
   predictions for that condition anymore, so the exact clustering number stops
   mattering to how well the model fits.
5. With nothing left to hold it in place, the number drifts further, often all
   the way to whatever ceiling exists.

It's a loop with no natural stopping point: a large value causes low reliance, and
low reliance removes any reason for the value to stay small.

### The evidence

We checked, for every subject, the average `prior_reliance` specifically during
80-degree blocks. The subjects whose clustering number exploded are exactly the
subjects whose reliance collapsed to near zero there (2 to 8%), while subjects
whose clustering number stayed reasonable had noticeably higher reliance (over
40% in one case). **8 of 12 subjects show this problem, always in the 80-degree
condition**, and 2 of those in the 40-degree condition too. Only 4 subjects (2, 4,
8, 12) are clean across every condition.

One nuance: a clean-looking, non-extreme value doesn't automatically mean it's
trustworthy. A couple of subjects' clustering numbers look unremarkable but still
pair with very low reliance, meaning they're just as unmeasured, they simply
happened to land somewhere ordinary rather than somewhere extreme.

### What this means going forward

- **Safe to use:** each subject's overall learning rate (`alpha`) and overall fit
  quality. Neither is caught in this loop.
- **Not safe to interpret:** any clustering number paired with very low reliance
  (roughly below 15%) for that subject and condition. Don't read meaning into it,
  and don't use it to judge whether subjects "correctly" track how wide a block's
  true variability is, that comparison isn't measurable for these cases.

### Does this mean the model itself is flawed?

Partly, but narrowly. The model's central idea, that subjects should trust a
fixed anchor more when it's reliable and less when it isn't, holds up clearly
and consistently across nearly all 12 subjects. This finding is evidence *for*
that mechanism, not against it.

The actual flaw is specific: one number was accidentally asked to do two
different jobs (describe clustering, and judge agreement), and that overlap
created the feedback loop. It's a fixable design detail, not a failure of the
underlying idea, and it was already the team's leading suspicion before all 12
subjects were fit. This work confirms and precisely explains that suspicion.

**For the upcoming AIC comparison**, this doesn't unfairly help the hierarchical
model. AIC penalizes every parameter the model uses, including the ones caught in
this loop, whether or not they're actually earning their keep. If the hierarchical
model still wins despite carrying that unearned penalty, that's a more convincing
result, not a less convincing one.

---

## Finding 2: `alpha` correlates with real behavior, specifically when the prior should matter most

### Background: two quick stats concepts used below

- **Correlation (r):** a number from -1 to 1 describing how strongly two
  measurements move together in a straight line. 1 means perfectly together, -1
  means perfectly opposite, 0 means no relationship at all.
- **p-value:** roughly, how likely it would be to see a correlation at least this
  strong purely by chance, if there were actually no real relationship. Smaller
  means more confidence it isn't a fluke. The common cutoff of 0.05 is just a
  convention, not proof.

### The question

`alpha` is the model's fitted learning rate, how fast a subject builds trust in
the fixed anchor. Is that number just an artifact of the fitting process, or does
it actually track something real in behavior?

To check, we used an independent, model-free behavioral measure called
`pull_increase`: take each subject's response error, specifically the part of it
that points toward the anchor rather than away from it, average that over the
first 15 trials of a block versus the last 100+, and take late minus early.
Positive means leaning on the anchor more by the end of a block; negative means
less.

### The result

| Comparison | r | p |
|---|---|---|
| `alpha` vs. `pull_increase`, averaged across all difficulty levels | 0.42 | 0.170 |
| `alpha` vs. `pull_increase` at the **weakest sensory evidence** (6% coherence) | **0.67** | **0.017** |
| `alpha` vs. `pull_increase` at medium evidence (12%) | 0.15 | 0.651 |
| `alpha` vs. `pull_increase` at the **strongest sensory evidence** (24%) | -0.005 | 0.988 |

The correlation only shows up when sensory evidence is weakest. That's not an
arbitrary result, it makes theoretical sense: when the incoming evidence is poor,
the anchor is doing most of the work, so that's exactly the condition where a
subject's fitted learning rate should be visible in their actual behavior. When
evidence is strong, it dominates regardless of how much a subject trusts the
anchor, so no relationship is expected there, and none was found.

**Robustness checks**, since only 12 subjects were fit and a single unusual data
point can swing a small correlation: removing the most extreme subject still
gives r = 0.60 (p = 0.051), same direction and similar strength. A version of this
test that only looks at relative ranking rather than exact values (Spearman
correlation, less sensitive to any one outlier) comes out slightly stronger:
r = 0.69, p = 0.013. So this isn't being driven by one unusual subject.

### Caveats to hold onto

- **Only 12 subjects, and 4 comparisons were tested.** A single p = 0.017 result
  out of four tests would not survive a strict correction for testing multiple
  comparisons at once (that correction, applied here, would require p < 0.0125).
  Treat this as a promising, theoretically sensible signal, not a settled result.
- **The overall group trend in `pull_increase` is mostly negative** at medium and
  high evidence levels, on average, subjects lean on the anchor *less* by the end
  of a block at those levels, not more. This finding is about *individual
  differences* at the weakest-evidence level specifically, not a claim that
  everyone learns to trust the anchor more over time in general.

---

## Bottom line before item 5

Both findings are compatible with the model's core hypothesis being sound. The
κ-identifiability issue affects specific numbers within specific subject and
condition combinations, not the parameters (`alpha`, overall fit) that the AIC
comparison actually depends on. The `alpha`-`pull_increase` correlation adds
independent, if modest and appropriately caveated, evidence that `alpha` reflects
something real rather than being an arbitrary product of the fitting process.
