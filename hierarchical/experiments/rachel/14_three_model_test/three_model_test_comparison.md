# Three-model test battery: bimodality + parameter recovery

The same two diagnostic tests, run on all three hierarchical observers on
**simulated data** (so the truth is known). Models:

- **Rachel** — `hb_integration.py`, integrate-then-**average** read-outs; learns κ; explicit α floor; **linear** forgetting (λ).
- **Salma** — branch `hierarchical_confidence`, integrate-then-read-out (one posterior); learns κ; **no α**; **geometric** forgetting (ρ).
- **Recombined** — `hb_integrate_before.py`, integrate-then-read-out (Salma) + α floor + linear forgetting (Rachel).

A methodological note up front: the tests are not perfectly symmetric, because
the models don't share the same learning parameters. Rachel and Recombined both
have (α, λ), so Test 2 is the same α–λ recovery for them. Salma has no α and
forgets via ρ, so her Test 2 is a 1-D ρ recovery. Test 1 is identical for all
three (Salma on her native 72-bin grid, the others on 360-bin).

---

## Test 1 — Does bimodality emerge from integration alone?

Criterion: with the belief settled on a WIDE prior (SD80-like feedback) and low
coherence (0.06), probe stimuli at increasing distance from the prior mean
(225°). "Bimodal" = substantial mass (>0.12) near BOTH the stimulus and 225°.

| distance from prior | Rachel | Salma | Recombined |
|---:|:--:|:--:|:--:|
| 0° (at prior) | unimodal ✓ | unimodal ✓ | unimodal ✓ |
| 40° | bimodal (0.34/0.49) | bimodal (0.46/0.73) | bimodal (0.19/0.62) |
| 80° | bimodal (0.16/0.38) | bimodal (0.29/0.23) | bimodal (0.12/0.45) |
| 120° | **bimodal (0.14/0.29)** | **unimodal (0.30/0.09)** | **bimodal (0.20/0.27)** |

(values = mass near stimulus / mass near prior)

**All three are correctly unimodal at the prior mean and bimodal at moderate
distances — the core emergent-bimodality result holds for every model.**

The performance difference is at the FAR extreme (120°), and it traces directly
to the α floor:

- **Salma loses bimodality at 120°** (prior-mode mass collapses to 0.09). With
  no α floor, her prior is a pure von Mises; 120° out it is essentially zero, so
  the likelihood wins outright and the second (prior) mode vanishes. Her model
  says: far enough from the prior, the observer just reports the stimulus.
- **Rachel and Recombined keep both modes alive at 120°** because their explicit
  (1−α)/360 floor guarantees residual prior mass everywhere on the circle. The
  prior mode never fully dies.

Neither behaviour is "wrong" — it's a genuine empirical prediction that differs
between the models, and the data can adjudicate it: *do humans still show a
prior-mode cluster for stimuli very far from 225°?* If yes, the α-floor models
are favoured; if the far-stimulus responses collapse onto the stimulus, Salma's
no-floor form is favoured. This is a clean, testable divergence for the paper.

---

## Test 2 — Are the learning parameters recoverable?

Simulate ~2400 trials (4 blocks SD80/40/20/10, 3 interleaved coherences) from
each model at a known truth, then map the NLL over its learning parameter(s).

| Model | learning params | truth | fit | verdict |
|---|---|---|---|---|
| **Recombined** | α, λ | (0.60, 0.15) | (0.60, 0.15) | **exact**; only 1 grid cell within 2 NLL — tightest basin |
| **Rachel** | α, λ | (0.60, 0.15) | (0.55, 0.10) | near (1 step off each); 5 cells within 2 NLL — looser basin |
| **Salma** | ρ | 0.90 | 0.90 | **exact**; clean 1-D curvature (ΔNLL≈26 over ±2 steps) |

Reading the results:

- **Recombined recovers best.** Exact truth, tightest basin, and the strongest
  λ=0 penalty (ΔNLL≈447 — i.e. "no forgetting" is decisively rejected). The
  integrate-before combination makes α and λ act on the percept through
  *different* routes (α shapes the single prior's floor; λ sets how fast the
  width belief moves), so they separate cleanly.
- **Rachel recovers slightly worse.** Fit lands one grid step low on both α and
  λ, and the basin is looser (5 near-tie cells vs 1). Its average-**after**
  combination lets α and λ trade off a little — a smaller α (less prior weight)
  and a smaller λ (belief sharpens more, stronger effective prior) partly cancel,
  producing a mild diagonal softness. Still identifiable (single joint minimum,
  λ=0 rejected at ΔNLL≈228), just less crisply.
- **Salma recovers ρ exactly** — unsurprising, because a single learning
  parameter cannot trade off against itself. This is the flip side of having no
  α: fewer learning parameters means no identifiability worry, at the cost of
  the flexibility the α floor buys (see Test 1).

---

## Combined read: what each result says about the models

| | Test 1 (bimodality) | Test 2 (recovery) | net |
|---|---|---|---|
| **Rachel** | robust incl. far band (α floor) | identifiable but loosest; mild α–λ tradeoff | solid; the α–λ softness is the one wrinkle |
| **Salma** | strongest at moderate distance, **fails far band** (no floor) | ρ exactly recovered (only 1 learning param) | cleanest fitting, but loses far-band bimodality |
| **Recombined** | robust incl. far band (α floor) | **best recovery**, tightest basin, no tradeoff | Salma's clean posterior + Rachel's floor = best of both on these two tests |

**Why the Recombined model comes out ahead on these two tests specifically:** it
inherits the α floor (so it keeps far-band bimodality, unlike Salma) AND the
integrate-before combination (so α and λ separate cleanly, unlike Rachel). The
two design choices that each cost one of the parent models a test are exactly
the two the recombination keeps.

## Important caveats
- These are recovery/behaviour tests on SIMULATED data — they show each model is
  self-consistent and identifiable, NOT which fits humans best. That requires the
  all-12-subject fit + AIC comparison (not yet run).
- Test 1's far-band divergence (α-floor vs not) is the scientifically interesting
  prediction; it is the thing to check against real far-from-prior trials.
- Rachel's α–λ softness is mild and grid-limited here (0.05 steps); a finer grid
  or a proper optimiser with a recovery repeat would quantify it precisely.
- Salma's test is 1-D by construction, so "recovers exactly" is a weaker claim
  than the 2-D basin results — it cannot exhibit a tradeoff because there is only
  one learning parameter to move.
