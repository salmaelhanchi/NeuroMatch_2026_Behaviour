"""Create the paper-supported participant-3 notebook from the validated template."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "01_participant_1_three_model_comparison.ipynb"
OUTPUT = ROOT / "notebooks" / "02_participant_3_bimodal_comparison.ipynb"

notebook = nbformat.read(TEMPLATE, as_version=4)
replacements = {
    "# Participant 1:": "# Participant 3:",
    "participant-1": "participant-3",
    "participant 1": "participant 3",
    '"participant_id": 1': '"participant_id": 3',
    'RUN_LABEL = "smoke" if SMOKE_TEST else "participant_1"': (
        'RUN_LABEL = "smoke_participant_3" if SMOKE_TEST else "participant_3"'
    ),
    "subject_id=1": "subject_id=3",
    "8,562": "9,412",
    "from standardized_hb.plotting import comparison_figure, state_figure": (
        "from standardized_hb.plotting import (\n"
        "    paper_bimodality_figure,\n"
        "    score_and_response_figure,\n"
        "    state_figure,\n"
        ")"
    ),
}

replacement_counts = {source: 0 for source in replacements}
for cell in notebook.cells:
    source = cell.source
    for old, new in replacements.items():
        replacement_counts[old] += source.count(old)
        source = source.replace(old, new)
    cell.source = source
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

missing = [source for source, count in replacement_counts.items() if count == 0]
if missing:
    raise RuntimeError(f"Participant notebook template changed; missing replacements: {missing}")

notebook.cells[0].source += (
    "\n\nParticipant 3 was selected from the paper rather than from a fitted result "
    "in this codebase. Figure 5F displays `sub03` as a clear subject-level "
    "bimodality example, and Figure 5E reports a 670-point AIC advantage for "
    "the Switching observer over the Basic Bayesian observer."
)


def find_cell(text: str, *, cell_type: str | None = None):
    matches = [
        cell
        for cell in notebook.cells
        if text in cell.source and (cell_type is None or cell.cell_type == cell_type)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {text!r}, found {len(matches)}.")
    return matches[0]


run_markdown = find_cell("## Run configuration", cell_type="markdown")
run_markdown.source = """## Run configuration

The normal interactive run fits all 9,412 participant-3 trials with two starts
per model and 60 objective evaluations per start. That matches the runtime of
the first standardized notebook and is intended for model exploration. Override
the budget with `HB_MAX_EVALUATIONS_PER_START=250` when you want a slower,
more exhaustive convergence check; `HB_SMOKE_TEST=1` retains the full trial
sequence but uses five evaluations for a pipeline check.

The integrated-prior likelihood is substantially more expensive than the other
two. A run is scientifically interpretable only when the convergence table
below reports success; otherwise treat the figures as diagnostics."""

run_code = find_cell("FIT_CONFIG = FitConfig(", cell_type="code")
run_code.source = run_code.source.replace(
    "FIT_CONFIG = FitConfig(\n"
    "    n_starts=1 if SMOKE_TEST else 2,\n"
    "    max_evaluations_per_start=5 if SMOKE_TEST else 60,\n"
    "    seed=20260721,\n"
    ")",
    "NORMAL_MAX_EVALUATIONS_PER_START = int(\n"
    "    os.environ.get(\"HB_MAX_EVALUATIONS_PER_START\", \"60\")\n"
    ")\n\n"
    "FIT_CONFIG = FitConfig(\n"
    "    n_starts=1 if SMOKE_TEST else 2,\n"
    "    max_evaluations_per_start=5 if SMOKE_TEST else NORMAL_MAX_EVALUATIONS_PER_START,\n"
    "    seed=20260721,\n"
    ")",
)
if "NORMAL_MAX_EVALUATIONS_PER_START" not in run_code.source:
    raise RuntimeError("Could not update the participant-3 optimizer budget.")

fit_markdown = find_cell("## Fit all three models", cell_type="markdown")
fit_markdown.source = """## Fit all three models

Each optimizer sees the same trials and transformed parameter bounds. AIC and
BIC use the same response binning and lapse likelihood, but they are treated as
valid model-comparison statistics only after all three optimizers converge."""

score_code = find_cell("score_table = pd.DataFrame(", cell_type="code")
score_code.source = """comparison_valid = all(result.success for result in fit_results.values())
score_table = pd.DataFrame(
    [
        {
            "model": name,
            "n_parameters": observers[name].parameter_count,
            "nll": result.nll,
            "aic": result.aic,
            "bic": result.bic,
            "delta_aic": result.aic - min(item.aic for item in fit_results.values()),
            "evaluations": result.evaluations,
            "elapsed_seconds": result.elapsed_seconds,
            "optimizer_success": result.success,
            "comparison_valid": comparison_valid,
        }
        for name, result in fit_results.items()
    ]
).sort_values("aic").reset_index(drop=True)
display(score_table)

parameter_table = pd.DataFrame(
    {name: result.parameters for name, result in fit_results.items()}
).T
display(parameter_table)

convergence_rows = []
for name, result in fit_results.items():
    observer = observers[name]
    fitted = np.asarray(result.raw_parameters)
    initial = observer.default_raw_parameters()
    bounds = np.asarray(observer.raw_bounds, dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    tolerance = 1e-3 * widths
    unchanged = np.abs(fitted - initial) <= tolerance
    at_bound = (
        (fitted - bounds[:, 0] <= tolerance)
        | (bounds[:, 1] - fitted <= tolerance)
    )
    convergence_rows.append(
        {
            "model": name,
            "success": result.success,
            "message": result.message,
            "unchanged_raw_parameters": ", ".join(
                np.asarray(observer.raw_parameter_names)[unchanged]
            ) or "none",
            "bound_hits": ", ".join(
                np.asarray(observer.raw_parameter_names)[at_bound]
            ) or "none",
        }
    )
convergence_table = pd.DataFrame(convergence_rows)
display(convergence_table)

if not comparison_valid:
    print(
        "WARNING: At least one optimizer did not converge. AIC/BIC and fitted "
        "trajectories below are provisional diagnostics, not a valid final model comparison."
    )"""

distribution_markdown = find_cell(
    "## Observed and predicted response distributions",
    cell_type="markdown",
)
distribution_markdown.source = """## Score overview and paper-matched bimodality diagnostic

The first figure shows provisional score differences and the marginal response
distribution. The second figure reproduces the participant-3 conditions used
in Figure 5F of the paper: 6% coherence, 80-degree prior width, motion
directions 75, 115, 125, 135, and 145 degrees, and 15-degree response bins.

`prior_std` is read here only to select trials for this descriptive figure. It
is loaded after fitting, is never added to `subject`, and is never passed to an
observer, prediction, likelihood, or update equation."""

distribution_code = find_cell(
    "figure = comparison_figure(subject, GRID, predictions, score_table)",
    cell_type="code",
)
distribution_code.source = """overview_figure = score_and_response_figure(
    subject,
    GRID,
    predictions,
    score_table,
)
overview_figure.savefig(
    OUTPUT_DIR / "response_and_score_comparison.png",
    dpi=180,
    bbox_inches="tight",
)
plt.show()

PAPER_COHERENCE = 0.06
PAPER_PRIOR_STD = 80.0
PAPER_DIRECTIONS = (75.0, 115.0, 125.0, 135.0, 145.0)
PLOTTING_COLUMNS = [
    "subject_id",
    "session_id",
    "run_id",
    "trial_index",
    "motion_direction",
    "motion_coherence",
    "prior_std",
]
header = pd.read_csv(DATA_CSV, nrows=0)
missing_plot_columns = sorted(set(PLOTTING_COLUMNS).difference(header.columns))
if missing_plot_columns:
    raise ValueError(f"Paper diagnostic requires plotting columns: {missing_plot_columns}")

plotting_rows = pd.read_csv(DATA_CSV, usecols=PLOTTING_COLUMNS)
plotting_rows = (
    plotting_rows.loc[plotting_rows["subject_id"] == subject.subject_id]
    .sort_values(["subject_id", "session_id", "run_id", "trial_index"], kind="mergesort")
    .reset_index(drop=True)
)
if len(plotting_rows) != subject.n_trials:
    raise ValueError("Plotting metadata does not align with the prepared participant trials.")
np.testing.assert_allclose(plotting_rows["motion_direction"], subject.directions)
np.testing.assert_array_equal(plotting_rows["session_id"], subject.session_ids)
np.testing.assert_array_equal(plotting_rows["run_id"], subject.run_ids)
np.testing.assert_array_equal(plotting_rows["trial_index"], subject.trial_indices)

paper_base_selector = (
    np.isclose(plotting_rows["motion_coherence"].to_numpy(dtype=float), PAPER_COHERENCE)
    & np.isclose(plotting_rows["prior_std"].to_numpy(dtype=float), PAPER_PRIOR_STD)
)
paper_selectors = {
    direction: paper_base_selector
    & np.isclose(subject.directions, direction)
    for direction in PAPER_DIRECTIONS
}
diagnostic_counts = pd.DataFrame(
    {
        "direction": PAPER_DIRECTIONS,
        "trials": [int(selector.sum()) for selector in paper_selectors.values()],
        "valid_responses": [
            int(np.count_nonzero(selector & subject.response_valid))
            for selector in paper_selectors.values()
        ],
    }
)
display(diagnostic_counts)
if (diagnostic_counts["valid_responses"] == 0).any():
    raise ValueError("At least one paper-matched direction has no valid responses.")

paper_figure = paper_bimodality_figure(
    subject,
    GRID,
    predictions,
    paper_selectors,
    coherence=PAPER_COHERENCE,
    prior_std=PAPER_PRIOR_STD,
    bin_width_degrees=15.0,
)
paper_figure.savefig(
    OUTPUT_DIR / "paper_matched_bimodality_diagnostic.png",
    dpi=180,
    bbox_inches="tight",
)
plt.show()"""

interpretation_markdown = find_cell("## Interpretation gate", cell_type="markdown")
interpretation_markdown.source = """## Interpretation gate

- Do not interpret AIC/BIC ordering unless `comparison_valid` is true for every
  row and the convergence table has been inspected.
- Parameters reported as unchanged or at a bound require scrutiny even after a
  nominally successful optimizer exit.
- Assess bimodality in the five paper-matched panels, not in a pooled or
  automatically selected 60-degree condition.
- The 80-degree `prior_std` label is descriptive plotting metadata only; all
  model states and updates continue to infer prior confidence from trial
  history without receiving that label.
- A full scientific comparison should add sequence-preserving held-out scoring
  and parameter recovery under this same standardized contract.
- Smoke-mode scores are pipeline checks and must not be reported as final fits."""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, OUTPUT)
print(OUTPUT)
