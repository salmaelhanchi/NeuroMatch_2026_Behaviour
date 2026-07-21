"""
observers.comparison
=====================

Rerunnable, model-agnostic pipeline comparing perceptual-observer models on the
Laquitaine & Gardner motion-direction data. Every stage (batch fit,
cross-validation, distribution shape, recovery, figure, table) iterates the
MODEL_REGISTRY in ``registry.py``, so adding a model is a one-entry change here
and never a pipeline edit.

Stages:
  registry.py        the model registry (the extensibility spine)
  fit_batch.py       all registered models x subjects -> one JSON per cell
  cross_validate.py  block-fold held-out per-trial NLL
  shape_analysis.py  predicted vs observed histograms, TV-by-band, bimodality
  recovery.py        parameter recovery + model-recovery confusion matrix
  make_figure.py     multi-panel results figure (Panels A-E)
  make_table.py      model-comparison table (models as rows)
  run_all.py         orchestrator that reruns the whole pipeline
"""
