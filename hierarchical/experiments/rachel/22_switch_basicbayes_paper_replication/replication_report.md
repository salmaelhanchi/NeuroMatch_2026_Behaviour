# Do our Switch and Basic-Bayes models replicate the paper?

**Yes — both reproduce Laquitaine & Gardner (2018) to machine precision (max |Δ| ≈ 2×10⁻¹⁷).**

## What "replicate" means here, and how it was tested

There was no prior check of this kind. The existing `verify_switching.py` /
`verify_basic_bayesian.py` suites test **internal identities** (reduction limits,
Eq. 6 weights, valid probability vectors, NLL sensitivity) — necessary, but they
only confirm the code is self-consistent, not that it matches the *paper's* model.

MATLAB/Octave is not available in this environment, so the reference code could
not be executed directly. Instead I made an **independent second implementation**:
I re-ported the paper's own MATLAB source — shipped in the repo at
`reference/laquitaine_gardner_matlab/` — straight into NumPy, importing nothing
from `observers/`. Two independent implementations agreeing is evidence of
reproducing the paper, not of the code agreeing with itself.

Reference files ported: `SLGirshickBayesLookupTable.m` (Girshick posterior + MAP
readout), `SLgetLoglBayesianModel.m` (lapse mixture + motor convolution + NLL),
`vmPdfs.m` (von Mises density), `SLcircConv.m` (circular convolution).

## Results (72 conditions: 6 directions × 3 coherences × 4 priors)

| Component | Production vs independent reference | Verdict |
|---|---|---|
| Girshick MAP lookup engine (shared by both models) | max \|Δ\| = **0.0** (bit-exact) | ✅ |
| Eq. 6 switching weights (reliability ratio) | max \|Δ\| = 1.1×10⁻¹⁶ | ✅ |
| **Switch** full estimate distribution | max \|Δ\| = **2.1×10⁻¹⁷** | ✅ |
| **Basic-Bayes** full estimate distribution | max \|Δ\| = **2.1×10⁻¹⁷** | ✅ |
| AIC/BIC formula | `2k+2·NLL` / `k·ln n+2·NLL` = paper's `2(k−ΣlogL)` | ✅ |
| Parameter count k=9 | 3 k_like + 4 k_prior + p_random + k_motor, cardinal fixed 0 | ✅ ("withoutCardinal") |

## The one documented difference: lapse convention

Under the raw paper convention the match is 3×10⁻⁵, not machine-zero. The single
cause is how the guess (lapse) mixture is normalized:

- **Paper** (`SLgetLoglBayesianModel.m`, lines 353–354): `(1 − p)·P + p·U`
- **Production** (Switch, Basic-Bayes, and the HB family): `(P + p·U)/(1 + p)`

These are the **same one-parameter family** of distributions, related by a
monotone reparameterization of the effective lapse rate (λ = p for the paper,
λ = p/(1+p) for production). Because `p_random` is a *fitted* free parameter, the
maximum-likelihood distribution, NLL, AIC and BIC are all identical — only the
reported numeric value of `p_random` differs (by ≤ 0.05 at the fitted values).
Swapping in the production convention makes the match exact (2×10⁻¹⁷).

Production uses `(P + p·U)/(1+p)` deliberately: it is the convention the whole HB
family shares, so Switch, Basic-Bayes and the hierarchical models are combined on
identical footing in the comparison.

## Reproduce

```
cd hierarchical
PYTHONPATH=. python -m observers.verification.verify_switch_basicbayes_vs_paper
# [PASS] Switch      == paper reference ... (max|Δ|=2.08e-17 over 72 conditions)
# [PASS] Basic-Bayes == paper reference ... (max|Δ|=2.08e-17 over 72 conditions)
```

**Bottom line for the slide/abstract:** the two paper models in our comparison
are faithful reimplementations of Laquitaine & Gardner's published code — the
in-sample and cross-validation numbers rest on verified implementations, so the
"Switch wins in-sample, hierarchical matches out-of-sample" claim is not an
artifact of a mis-ported baseline.
