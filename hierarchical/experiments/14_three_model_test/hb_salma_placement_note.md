# HB - Salma added to observers/models

## What was added
Two things under `hierarchical/observers/models/`:

1. **`salma_hierarchical/`** — a vendored, **byte-identical** copy of the branch
   model (`independent_transcript_paper_model/src/hierarchical_confidence/`):
   `model.py`, `circular.py`, `readout.py`, `data.py`, plus an `__init__.py`.
   Checksums verified identical to the branch source at vendor time. Kept
   unmodified on purpose (see below).

2. **`hb_salma.py`** — a house-API adapter class `HBSalmaObserver` that wraps the
   vendored code and exposes the same interface as the other observers:
   `filter(directions, coherences, feedback=...)`, `estimate_distribution(...)`,
   `negative_log_likelihood(...)`, `aic`, `bic`, `N_PARAMS`.

## Why vendored verbatim + adapter, not a re-port
Her model uses a different contract from the house observers: it is
**subject-batch** (one ordered pass over a whole participant's trials) on a
**72-bin (5°)** grid with a tie-aware MAP read-out; the house observers are
**per-trial-sequence** on a **360-bin (1°)** grid. Re-deriving her mechanism in
the house style on a 360-bin grid would silently diverge from the HB - Salma
numbers already reported in the comparison docs. Vendoring her real code keeps
those numbers reproducible; the adapter only translates the calling convention.

## Faithfulness check (passed)
Driving the adapter and her original package on subject 9 with identical
parameters:
- native 72-bin NLL: **31180.0314 vs 31180.0314** (identical to 4 dp)
- predicted PMFs: **max abs diff 0.00e+00** (bit-identical)
- 6 fitted parameters (rho, 3 sensory kappas, motor kappa, lapse); no alpha.

## Grid caveat baked into the adapter
NLL/AIC on 72-bin vs 360-bin grids are NOT comparable. The adapter exposes:
- `negative_log_likelihood(..., grid="native")` — her own 72-bin score
  (reproduces the published HB - Salma numbers);
- `negative_log_likelihood(..., grid="deg360")` — her PMFs up-sampled to 360
  bins, directly comparable to the other observers' NLL.
Use `grid="deg360"` for any cross-model AIC.

## Placement in the tree
```
observers/models/
  hb_integration.py            HB - Rachel  (abstract model)
  hb_integrate_before.py       Recombined   (Rachel × Salma)
  hb_salma.py                  HB - Salma   ADAPTER  <-- new
  salma_hierarchical/          HB - Salma   vendored source  <-- new
    __init__.py, model.py, circular.py, readout.py, data.py
  switching_observer.py        paper's switch
  basic_bayesian.py            integration baseline
  anirban_variants/            the 4 Anirban models
```

## Usage
```python
from observers.models.hb_salma import HBSalmaObserver
obs = HBSalmaObserver(rho=0.9, sensory_kappas=(1.5, 3.0, 8.0),
                      motor_kappa=40.0, lapse=0.02)
out = obs.filter(directions, coherences, feedback=directions)   # like the others
nll_fair = obs.negative_log_likelihood(estimates, directions, coherences,
                                       feedback=directions, grid="deg360")
```

## Notes for the team
- The tree is a git repo; `hb_salma.py` and `salma_hierarchical/` are UNTRACKED
  (not committed), consistent with how the other new models were left — commit
  from your own checkout.
- To update HB - Salma if the branch changes: re-copy the 4 files from the branch
  and re-verify checksums; do not hand-edit the vendored copy.
- A fitting helper analogous to `hb_integration_fit.py` was NOT added — her own
  `fit.py` (Powell + multistart) lives in the vendored branch package. If the
  team wants a house-style fitter for her, that can be added; flagged, not built.
