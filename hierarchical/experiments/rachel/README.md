# Experiments

Each folder is one self-contained piece of work, holding **all of its own output** — figures, tables, reports, notebooks, and any experiment-specific derived fit output (in a local `results/` subfolder).

## Organizing principle

- **Experiment-specific** output lives inside its experiment folder (including derived fits like cross-validation, shape, and recovery output under `<experiment>/results/`).
- **Shared, reusable resources stay at the top level:**
  - `../../observers/` — the live pipeline code (registry, fitters, analysis, figure/table builders)
  - `../../results/fits/comparison/<model>/subject<N>.json` — the shared fitted-parameter database (per-subject JSONs, one folder per model), read by several experiments; expensive to regenerate, so kept in one place
  - `../../results/logs/` — pipeline run logs (execution history)
  - `../../docs/` — reference material (Laquitaine & Gardner 2018 paper, the submitted abstract, the Anirban scaffold)
  - `../../data/` — the dataset; `../../notebooks/` — shared onboarding/tutorial notebooks

## Tags

Each experiment's README carries a `Tags:` line with four facets:

- **`type:`** — kind of work: `build` (implement/modify a model) · `comparison` (head-to-head model evaluation) · `analysis` (probe a phenomenon) · `validation` (correctness / recovery / held-out) · `reproduction` (reproduce another implementation) · `docs` (reference/explanatory writeup)
- **`claim:`** — which abstract claim it bears on: `bimodality` (reproduces uni/bimodal distributions, claim i) · `prior-learning` (recovers block-specific prior widths, claim ii) · `switching-vs-integration` (the core reframing) · `none` (infrastructure)
- **`status:`** — `result` (a finding for the paper/talk) · `supporting` (corroborates a result) · `process` (how we got there) · `superseded` (kept for history)
- **`presentation:`** — Friday-talk priority: `tier1` (headline slide) · `tier2` (strong support) · `backup` (appendix/Q&A) · `skip`

Slice the folder by grepping the tag, e.g. `grep -rl "presentation:tier1" */README.md`.

## Presentation material (Neuromatch, Fri)

Maps onto the abstract's two claims. Lead with the held-out result; use bimodality as the visual payoff.

- **Tier 1** — [`19_holdout_replication`](19_holdout_replication/) (headline: learning-Bayes observers tie Switch out-of-sample, non-learning baseline does not) · [`15_bimodality_analysis`](15_bimodality_analysis/) (reproduces two-cluster distributions; caveat: under-weights stimulus cluster)
- **Tier 2** — [`17_intersubject`](17_intersubject/) (graded, not discrete: flat learned α, precision-driven differences) · [`18_recombined_vs_switch`](18_recombined_vs_switch/) (bimodality from the combination rule)
- **Backup / Q&A** — [`11_all12_batch_fit`](11_all12_batch_fit/), [`12_cross_validation`](12_cross_validation/) (fit/AIC/CV tables; note in-sample AIC favors Switch — which is why the held-out result leads), [`06_model_step_comparison`](06_model_step_comparison/), [`07_block_phase_reproduction`](07_block_phase_reproduction/)

## Index

- **[`01_model_review/`](01_model_review/)** — Model review, roadmap, switch-probability curve
- **[`02_abstract_drafts/`](02_abstract_drafts/)** — Abstract drafts (md/docx), revised abstract, team explainer
- **[`03_model_equations/`](03_model_equations/)** — Formal equations per observer (md + pdf) + generative-model / hb_integration notes
- **[`04_four_new_models/`](04_four_new_models/)** — Build report, manifest, Anirban-vision comparison for the 4 variants
- **[`05_model_definition_comparison/`](05_model_definition_comparison/)** — Definition-level diff: root notebooks vs package models
- **[`06_model_step_comparison/`](06_model_step_comparison/)** — Step-by-step model comparison (2- and 3-model, md + pdf)
- **[`07_block_phase_reproduction/`](07_block_phase_reproduction/)** — Per-block/phase hierarchy plots + TV-by-band / learned-prior-SD tables
- **[`08_validation_notebooks/`](08_validation_notebooks/)** — Tutorial + solutions + paper-match notebooks, builder, guides
- **[`09_derived_alpha/`](09_derived_alpha/)** — Derived-alpha variant: results, fit, figure, code
- **[`10_adaptive_volatility/`](10_adaptive_volatility/)** — Adaptive-volatility (AT) variant: results, fit, figure, code, note
- **[`11_all12_batch_fit/`](11_all12_batch_fit/)** — First full 12-subject batch fit: results, JSON, CSVs, figure, code
- **[`12_cross_validation/`](12_cross_validation/)** — Block-fold CV: held-out NLL results, figure, code, per-subject fits in results/
- **[`13_combined_recombined/`](13_combined_recombined/)** — Recombined integrate-before model: design note + recovery figure
- **[`14_three_model_test/`](14_three_model_test/)** — Three-model test + subject-9 figures + subject 1&3 fits in results/
- **[`15_bimodality_analysis/`](15_bimodality_analysis/)** — Main bimodality investigation: conditioned panels, comparison table/figure, recovery, shape+bimodality fits in results/
- **[`16_early_vs_late/`](16_early_vs_late/)** — Early-vs-late bimodality test (no transition found)
- **[`17_intersubject/`](17_intersubject/)** — Inter-subject mechanism: flat learned alpha, precision-driven differences
- **[`18_recombined_vs_switch/`](18_recombined_vs_switch/)** — Recombined vs Switch head-to-head: figure + report
- **[`19_holdout_replication/`](19_holdout_replication/)** — Held-out replication of Salma's protocol: harness, 12 JSONs, figure, report
- **[`20_switch_paper_fitting/`](20_switch_paper_fitting/)** — Switch fit to the paper's exact protocol (NM 1e-4 + CMA-ES): optimizer-independence confirmed, subject-5 CV failure diagnosed as a fold artifact — write-up, comparison CSV, 12 CMA-ES fits
- **[`21_hierarchical_online_diagnosis/`](21_hierarchical_online_diagnosis/)** — hierarchical_online's poor S1/S3 fits are genuine model misfit, not a fitting failure (180-point grid + forward-eval probe confirm the fit is optimal; still ~1040 nats worse than Basic-Bayes) — write-up
