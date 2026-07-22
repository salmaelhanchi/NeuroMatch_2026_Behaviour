# Model comparison

Fitted on 12 subject(s): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]. Δ values are relative to **Switch**. All models are on the 360° grid, so NLLs are directly comparable (no grid artifact).

| Model | k | ΣNLL | ΔAIC | ΔBIC | CV-NLL | CV/trial (mean±SD) | ΔFarband-NLL | Wins (AIC) | Wins (CV) |
|---|---|---|---|---|---|---|---|---|---|
| Switch | 9 | 403212.8 | +0.0 | +0.0 | nan | nan±nan | +0.0 | 10/12 | 5/12 |
| Basic-Bayes | 9 | 404748.7 | +3071.8 | +3071.8 | 407361.0 | 4.9304±0.2926 | +755.2 | 1/12 | 3/12 |
| HB-Adaptive | 6 | 404376.8 | +2256.1 | +2010.4 | 406727.1 | 4.9227±0.3025 | +721.7 | 1/12 | 4/12 |

*Negative Δ favours that model over the reference. Lead with CV-NLL (overfitting-proof); AIC/BIC corroborate.*

**ΔFarband-NLL** is the NLL gap vs the reference restricted to the bimodality regime (stimulus ≥60° from the 225° prior, low coherence, wide prior — ~16% of trials). The aggregate ΣNLL is dominated by the easy near-prior trials; this column tests the abstract's shape claim on the trials where shape actually varies. A model can win ΣNLL while losing here.
