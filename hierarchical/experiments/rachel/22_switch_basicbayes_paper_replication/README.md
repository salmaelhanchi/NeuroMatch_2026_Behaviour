# Switch & Basic-Bayes — paper replication check

*Date: 2026-07-22*

*Tags: type:validation | claim:switching-vs-integration | status:result | presentation:backup*

Do our `switch` and `basic_bayes` observers reproduce Laquitaine & Gardner
(2018)? **Yes — to machine precision (max |Δ| ≈ 2×10⁻¹⁷ over 72 conditions).**

The other `observers/verification/verify_*.py` suites test *internal* identities
(reduction limits, Eq. 6 weights, valid probability vectors) — they show the code
is self-consistent, not that it matches the *paper*. This experiment closes that
gap by cross-checking against the paper's own MATLAB reference.

## Method

MATLAB/Octave is not available, so the reference code (shipped in the repo at
`reference/laquitaine_gardner_matlab/`) was **re-ported independently into NumPy**,
importing nothing from `observers/`. Two independent implementations agreeing is
evidence of reproducing the paper rather than of the code agreeing with itself.
Reference files ported: `SLGirshickBayesLookupTable.m`, `SLgetLoglBayesianModel.m`,
`vmPdfs.m`, `SLcircConv.m`.

## Result

| Component | max \|Δ\| (production convention) |
|---|---|
| Girshick MAP lookup engine (shared by both models) | **0.0** (bit-exact) |
| Switch full estimate distribution | **2.1×10⁻¹⁷** |
| Basic-Bayes full estimate distribution | **2.1×10⁻¹⁷** |

The **only** deviation is a lapse-mixture reparameterization: the paper writes the
guess mixture as `(1−p)·P + p·U`; our code uses the algebraically-equivalent
`(P + p·U)/(1+p)` so Switch, Basic-Bayes and the HB family all share one
convention. Same one-parameter family (effective lapse λ = p vs p/(1+p)) → same
MLE distribution, NLL, AIC, BIC; only the reported `p_random` differs. Under the
raw paper convention the match is 3×10⁻⁵; under the production convention it is
exact. AIC/BIC formula and the k=9 parameter count ("withoutCardinal") also match.

## Files

- `verify_switch_basicbayes_vs_paper.py` — the independent reference re-port + the
  2-check suite. Mirror of `observers/verification/verify_switch_basicbayes_vs_paper.py`.
- `run_replication.py` — driver: regenerates the CSV and figure below.
- `replication_maxdiff.csv` — per-(component, lapse-convention) max |Δ|.
- `replication.png` — overlay of our distributions vs the reference (Switch bimodal,
  Basic-Bayes unimodal) + agreement bars vs the 1e-10 tolerance.

## Reproduce

```
cd hierarchical
PYTHONPATH=. python -m observers.verification.verify_switch_basicbayes_vs_paper
PYTHONPATH=. python experiments/rachel/22_switch_basicbayes_paper_replication/run_replication.py
```
