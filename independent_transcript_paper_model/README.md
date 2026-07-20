# Independent Transcript-and-Paper Modeling Workspace

**Implementation:** `notebooks/03_gpu_long_multistart_fit.ipynb` provides resumable all-participant GPU multi-start fitting with four starts per participant.
**Results:** the GPU pilot passed computational gates but did not support bimodality; the new long-run workflow has passed smoke and resume tests only.
**Next:** execute the 8-16 hour fit, assess convergence and recovery, then compare the Basic Bayesian and Switching observers.

## Purpose

This folder is a clean modeling workspace for the motion-direction project. Its scientific design is reconstructed directly from the TA sessions and the original Switching Observer paper.

No existing implementation, notebook, code module, project guide, or previous result is an input to the design in this folder.

## Source authority

Use sources in this order:

1. `07_14.txt` and `0715.txt`: authority for what the project should change and how work should proceed.
2. *A Switching Observer for Human Perceptual Estimation*: authority for the experiment, original observer models, likelihood construction, fitting, and comparison.
3. `Hierarchical_explicit_stepwise.pdf` and `Model_explainer.pdf`: secondary explanations only.
4. `data01_direction4priors.csv`: raw execution input, not a source of theoretical assumptions.

When sources disagree, the TA correction wins over the older explanatory PDFs. In particular:

- the prior mean remains fixed at 225 degrees;
- the learned quantity is prior concentration/confidence;
- the final state of one block is carried into the next block;
- model validation and parameter recovery happen before scientific model comparison;
- models are fitted separately for each of the 12 subjects.

## Explicit exclusions

The documents here do not rely on:

- existing Python or notebook implementations;
- archived model results;
- earlier implementation plans or progress notes;
- fitted parameters produced before this workspace was created.

Those materials may later be compared against the independent result, but they cannot define it.

## Files

- `01_INDEPENDENT_TECHNICAL_MAP.md`: source-derived scientific and mathematical specification.
- `02_SOURCE_ONLY_EXECUTION_PLAN.md`: ordered analysis and modeling plan with validation gates.
- `03_STEP_BY_STEP_IMPLEMENTATION.md`: simplified technical implementation, organized as Input -> Process -> Output -> Check.

## Working rule

Every future assumption must be labeled as one of:

- **Paper-defined**: explicitly specified in the original paper.
- **TA-directed**: explicitly requested or corrected in the sessions.
- **Implementation choice**: needed to make the model executable but not fixed by the sources.
- **Empirical decision**: selected only after inspecting the raw data or validation results.

This labeling prevents convenient coding decisions from being mistaken for scientific claims.

