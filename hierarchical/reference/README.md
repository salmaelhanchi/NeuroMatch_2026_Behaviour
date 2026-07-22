# Reference implementations

Third-party ground-truth code that our models are validated against. This is **not** our
code — it is the original authors' implementation, kept here for reference and comparison.

## `laquitaine_gardner_matlab/`

The MATLAB code from Laquitaine & Gardner (2018), *A Switching Observer for Human Perceptual
Estimation* (Neuron 97(2), 462–474), extracted from the authors' `projInference` repository
(the `gh-pages` branch). Contents:

- `codes/` — the model-fitting code, including `SLfitBayesianModel.m` and the `SL*` function
  library (`codes/assets/`, e.g. `SLGirshickBayesLookupTable.m`, `SLcircConv.m`) that implements
  the paper's Bayesian observer.
- `task/` — the psychophysics task code (`main.m`, `taskDotDir.m`, …).
- `analyses/` — eye-position / Hotelling analyses.
- `notebooks/`, `params.json`, `README.md` — the authors' own notebooks and config.

The rendered project website (fonts, task-demo videos, HTML/JS) that came bundled in the
`gh-pages` branch has been removed — it is available at the original GitHub repository if the
rendered site is ever needed. The paper's raw data is not stored here; our working copy of the
experiment-1 dataset lives in `../data/data01_direction4priors.csv`.
