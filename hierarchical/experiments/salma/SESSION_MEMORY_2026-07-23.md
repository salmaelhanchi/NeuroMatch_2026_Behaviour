# Session Memory - 2026-07-23

Scope: this memory covers only today's Codex session about pulling and mapping the current remote `model-verification` branch, especially `hierarchical/experiments`.

What was checked:
- The current GitHub branch `model-verification` was queried directly through GitHub API because the local branch view was stale.
- The remote folder `hierarchical/experiments` exists and contains teammate folders: `anirban`, `rachel`, `romi`, `salma`, and `valeria`.
- The experiments folder is mostly reports, notebooks, figures, and per-person exploratory records. The shared model logic lives in `hierarchical/observers/`.

Current model map:
- The model comparison pipeline is controlled by `hierarchical/observers/comparison/registry.py`.
- The default comparison models are `hb_adaptive` and `switch`.
- The current main abstract model is `HBAdaptiveConfidenceObserver` in `hierarchical/observers/models/hb_adaptive_confidence.py`.
- `hb_adaptive` learns a joint hidden belief over prior width `kappa` and prior confidence `alpha` from feedback.
- The older `hb_integration` model is now named `hb_rachel`; it learns kappa while alpha is fitted/fixed.
- Salma's earlier model corresponds most closely to `hb_salma`, which uses geometric forgetting and a 72-bin/native-grid implementation scored on 360 degrees in the registry.

Files created for Salma's slide notebook folder:
- `hb_adaptive_learning_diagnostics.ipynb`: small notebook to rebuild and inspect the diagnostic PDF and summary tables.
- `build_hb_adaptive_learning_pdf.py`: script that replays fitted `hb_adaptive` observers over all subjects and creates the PDF/summaries.
- `figures/hb_adaptive_all_participants_learning_diagnostics.pdf`: 16-page PDF.
- `results/hb_adaptive_learning_anova_fixed_subject.csv`: ANOVA-style subject-fixed quantification.
- `results/hb_adaptive_learning_by_subject_condition_phase.csv`: learned alpha/SD summaries by subject, coherence, prior_std, and block phase.
- `results/hb_adaptive_block_alignment_summary.csv`: session/run block alignment diagnostic.
- `results/hb_adaptive_fit_summary.csv`: compact fit status table.

Interpretation from generated summaries:
- Pages 2-3 of the PDF answer the prior-learning-by-coherence/prior-width question.
- Page 1 reports the ANOVA-style quantification.
- Page 4 checks possible block shift/alignment.
- The analysis uses already fitted `hb_adaptive` parameters, then performs post-fit diagnostics; it does not refit a separate regression model with coherence and prior_std as learning parameters.
- The hidden learning update depends on feedback directions. Coherence changes sensory likelihood and response predictions, so coherence-stratified hidden-learning plots show where coherence trials fall on the same learned trajectory.
