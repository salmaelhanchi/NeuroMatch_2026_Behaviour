"""
salma_hierarchical_helpers
==========================

**Vendored, verbatim** copy of the branch model's hierarchical-confidence
observer (branch `new_implementaion_with_hiearchial-_fiting`, package
`independent_transcript_paper_model/src/hierarchical_confidence/`), labelled
**HB - Salma** in the model-comparison documents.

The four modules here — `model.py`, `circular.py`, `readout.py`, `data.py` —
are byte-identical copies of the branch source (verified by checksum at
vendor time). They are kept unmodified on purpose: the model-comparison
numbers reported for HB - Salma were computed from this exact code, so any
re-derivation in the house API would risk silently diverging from them. To
update, re-copy from the branch and re-verify checksums — do not hand-edit.

What the model does (see `docs/branch_hierarchical_equations.md`, B1-B5):
  * prior basis = pure von Mises per kappa on a 72-bin angle grid, with an
    explicit kappa=0 (uniform) grid point instead of a separate mixture floor;
  * hidden-confidence trajectory over kappa with GEOMETRIC forgetting
    (log H_before = rho * log H_after), renormalised in log space;
  * integrate the confidence into ONE effective prior before read-out;
  * tie-aware MAP read-out, FFT motor convolution, lapse.
  * 6 fitted parameters: rho, 3 sensory kappas, motor kappa, lapse (NO alpha).

For house-API use (``filter`` / ``estimate_distribution`` /
``negative_log_likelihood`` like the other observers), import the adapter
``HBSalmaObserver`` from ``observers.models.hb_salma`` instead of driving these
classes directly.
"""
from .model import (
    GridSpec,
    ModelParameters,
    HierarchicalObserver,
    PreparedSubject,
    prepare_subject,
    PRIOR_MEAN_DEGREES,
)

__all__ = [
    "GridSpec",
    "ModelParameters",
    "HierarchicalObserver",
    "PreparedSubject",
    "prepare_subject",
    "PRIOR_MEAN_DEGREES",
]
