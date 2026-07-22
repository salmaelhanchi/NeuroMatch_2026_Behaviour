# Experiments — Romi

Each folder is one self-contained piece of work, holding **all of its own output** — figures, tables, reports, notebooks, and any experiment-specific derived fit output (in a local `results/` subfolder).

## Organizing principle

- **Experiment-specific** output lives inside its experiment folder.
- **Shared, reusable resources stay at the top level** — read them, don't copy them:
  - `../../observers/` — the live pipeline code (registry, fitters, analysis, figure/table builders)
  - `../../results/fits/comparison/<model>/subject<N>.json` — the shared fitted-parameter database
  - `../../data/` — the dataset
  - `../../docs/` — reference material (Laquitaine & Gardner 2018 paper, the abstract)

## Tags

Each experiment's README carries a `Tags:` line with four facets. See [`experiments/rachel/README.md`](../rachel/README.md) for the full vocabulary (`type:`, `claim:`, `status:`, `presentation:`).

## Index

- **[`01_slide_notebook/`](01_slide_notebook/)** — HB-Adaptive learned prior-width recovery per block, per-subject learning rate λ, learned confidence α, and prior learning across coherences (ANOVA)
