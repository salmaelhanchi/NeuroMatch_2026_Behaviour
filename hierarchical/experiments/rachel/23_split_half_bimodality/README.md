# Split-half bimodality — parameter stability (paper) + out-of-sample prediction (extension)

*Date: 2026-07-22*

*Tags: type:validation | claim:bimodality | status:result | presentation:tier2*


Makes the bimodality signature a genuine **out-of-sample** test, closing the gap
flagged in review: reproducing bimodality in the *fitted* estimate distribution
is in-sample (the fit objective already sees the whole distribution shape), so it
shows *architecture* (Basic-Bayes cannot express two clusters at any
parameterization) but not *generalization*.

## Relation to the paper (read this first)

This is **not** the analysis Laquitaine & Gardner ran, despite the shared phrase
"split-half." The paper (Neuron 97, p.466) fit the Switching observer to the
**first and last halves** (a *temporal* split) and compared the two fits' AICs
and prior-strength parameters *to each other* to show **parameter stationarity**
— that learning had plateaued and the fit does not drift across the session
(AICs n.s., Wilcoxon Z=32, p=0.62, n=12). That is an **in-sample** check: each
half is fit and scored on itself. The words "cross-validation", "held-out", and
"split" never appear in the paper.

This experiment does something different and stronger: a **cross-prediction**
(fit on one half, *predict the held-out half's* bimodality signature) with an
**interleaved** (even/odd) split chosen to remove the time confound. It is a
genuine **out-of-sample** test that goes *beyond* the paper. The paper's own
stationarity check is reproduced separately in
`paper_stationarity.py` (companion, below).

## What this does

For each subject:

1. Split trials into two **interleaved halves** (even/odd index) so each half
   spans the whole session — all four prior-width blocks, matched exposure.
2. **Fit** the model's parameters on one half (single Nelder–Mead start, the
   cross-validation convention in `registry._starts_for`).
3. **Evaluate** the bimodality signature on the **held-out** half: does the
   fitted-parameter prediction match what the subject actually did on trials the
   fit never saw?
4. Repeat swapping the halves (A→B and B→A); report both.

The signature reuses the canonical metrics from
`observers/comparison/shape_analysis.py` verbatim:

- **far-band prior-cluster mass** — response mass within ±18° of the 225° prior
  when the stimulus is ≥90° away, low coherence (≤0.12), widest prior (80).
- **valley-depth** — `1 − min_between_modes / min(peak heights)`, per single far
  stimulus direction, averaged. ~1 = two clean separated peaks; ~0 = one hump.
  This is the metric that mass alone misses (a broad hump has the same mass).

## Models

`switch` and `basic_bayes` — the two paper models. The prediction: on held-out
trials the **Switch** tracks the observed valley depth (its fitted parameters
generalize the two-peak structure), while **Basic-Bayes** sits near zero valley
regardless of fit — a single-posterior readout is structurally unimodal, so it
cannot generalize a signature it cannot express.

## Companion — the paper's actual stationarity check

`paper_stationarity.py` reproduces Laquitaine & Gardner's real analysis: fit the
Switching observer to the **first** vs **last** (temporal) half of each subject,
then Wilcoxon signed-rank test whether the AIC and the prior-strength parameters
differ between halves. Non-significant = the fit is stationary (learning has
plateaued). Our Z/p are reported beside the paper's (AIC p=0.62; k_prior p=0.90 /
0.30 / 0.23 for the 80° / 40° / 20° priors). This is the in-sample provenance
anchor; the cross-prediction test above is the out-of-sample extension.

## Files

- `split_half_bimodality.py` — cross-prediction fit+score harness (out-of-sample;
  serial — the sandbox blocks process pools and the fits are CPU/GIL-bound).
- `plot_split_half.py` — figure + summary table from the results JSON.
- `split_half_bimodality.png` — held-out predicted vs observed signature.
- `split_half_summary.csv` — per-subject, both directions, both metrics.
- `paper_stationarity.py` — the paper's first/last-half parameter-stability check.
- `stationarity_summary.json` — our Wilcoxon Z/p beside the paper's.
- `results/` — per-subject JSON + run logs.

## Reproduce

```
cd hierarchical
# out-of-sample cross-prediction (this experiment's contribution):
PYTHONPATH=. python experiments/rachel/23_split_half_bimodality/split_half_bimodality.py
PYTHONPATH=. python experiments/rachel/23_split_half_bimodality/plot_split_half.py
# the paper's own stationarity check (provenance anchor):
PYTHONPATH=. python experiments/rachel/23_split_half_bimodality/paper_stationarity.py
```
