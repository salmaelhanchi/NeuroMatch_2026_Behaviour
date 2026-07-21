# Model comparison: Recombined vs Switch (12 subjects)

**Models.** *Switch* — the paper's switching observer, selects prior OR likelihood each trial (k=9).
*Recombined* — a hierarchical-Bayesian observer that learns prior confidence online and combines prior with likelihood by **integrating before read-out** (convolution preserves two modes), k=7. Recombined is fit-only (excluded from cross-validation for runtime); comparison is on maximum-likelihood fit (NLL/AIC/BIC) and distribution shape / bimodality.

## Headline

| Metric | Recombined (k=7) | Switch (k=9) | Winner |
|---|---|---|---|
| Σ NLL (12 subj) | 410006 | 403213 | **Switch** (Δ 6793) |
| Σ AIC | 820180 | 806642 | **Switch** (Δ 13538) |
| Σ BIC | 820753 | 807379 | **Switch** (Δ 13375) |
| AIC wins (per subj) | 0/12 | **12/12** | Switch |
| BIC wins (per subj) | 0/12 | **12/12** | Switch |
| Mean far-band TV | 0.787 | 0.757 | Switch (marginal) |
| Truly bimodal subjects | 6/12 | **9/12** | Switch |

## Interpretation

**Switch wins decisively on fit** — lower NLL/AIC/BIC on every one of the 12 subjects, despite carrying *more* parameters (k=9 vs 7). The AIC gap averages ~1128 per subject, far beyond the ~4-point threshold for a meaningful difference. The extra flexibility Switch spends on per-condition k_prior/k_like reliabilities buys real explanatory power the integrate-before convolution does not recover.

**Bimodality — the reason recombined exists — does not overtake Switch either.** We expected integrate-before to reproduce the two-peak far-band distributions better than a selecting model. It does produce genuine bimodality (2 true local maxima at prior + stimulus) in **6/12** subjects — but Switch does so in **9/12**, and Switch's far-band shape (TV to observed) is marginally *closer* too (0.757 vs 0.787). The subjects where recombined fails to make two peaks (5, 6, 7, 8, 10, 12) are those with lower motor precision, where its single integrated posterior stays unimodal.

**Why.** Both models can be bimodal, but by different routes. Switch is bimodal *across trials* — it commits to one source per trial, so pooling piles mass at both the prior and the stimulus. Recombined is bimodal *within a trial* only when the convolution's two lobes survive the read-out; when the learned prior is confident and the likelihood weak, the stimulus lobe collapses and the distribution goes unimodal. Trial-level selection is a more robust generator of the observed two-peak structure than pre-read-out integration.

## Bottom line for the abstract

On this head-to-head, **the paper's Switch model remains the stronger account** — better fit and more reliable bimodality. Recombined's value is conceptual: it shows two-peak behaviour *can* arise from a learning Bayesian observer with no explicit switch (claim (i) is achievable in principle), but it does not yet match the switching heuristic quantitatively. Closing that gap would require either a wider learned-prior regime or a mixture read-out — a direction, not a finished result.

## Per-subject detail

| Subj | NLL R | NLL S | ΔAIC (R−S) | far-TV R | far-TV S | R bimodal | S bimodal |
|---|---|---|---|---|---|---|---|
| 1 | 39535 | 38529 | +2007 | 0.70 | 0.62 | ✓ | ✓ |
| 2 | 36810 | 36254 | +1107 | 0.87 | 0.86 | ✓ | ✓ |
| 3 | 43055 | 41739 | +2628 | 0.70 | 0.60 | ✓ | ✓ |
| 4 | 23430 | 23076 | +703 | 0.90 | 0.89 | ✓ | ✓ |
| 5 | 31913 | 30981 | +1859 | 0.84 | 0.76 | — | — |
| 6 | 38211 | 37585 | +1247 | 0.75 | 0.68 | — | — |
| 7 | 27318 | 27092 | +447 | 0.74 | 0.76 | — | ✓ |
| 8 | 31084 | 30950 | +263 | 0.83 | 0.84 | — | ✓ |
| 9 | 42749 | 42238 | +1018 | 0.71 | 0.70 | ✓ | ✓ |
| 10 | 31183 | 30890 | +583 | 0.84 | 0.83 | — | — |
| 11 | 31677 | 31297 | +755 | 0.75 | 0.75 | ✓ | ✓ |
| 12 | 33043 | 32581 | +919 | 0.82 | 0.80 | — | ✓ |

*ΔAIC > 0 means Switch is preferred. "bimodal" = the model's far-band marginal has two true local maxima (prominence ≥ 5% of peak), one near the prior (225°) and one near the stimulus.*
