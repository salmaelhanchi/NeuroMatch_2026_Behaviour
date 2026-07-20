"""Build the block-phase and hidden-kappa posterior-predictive notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "04_all_participant_model_data_comparison.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# History-sensitive hierarchical model check: all participants

## Purpose

This notebook asks whether the hierarchical observer changes its predictions as it learns prior reliability within a block. It uses only the completed long-run artifacts:

- original responses and experimental conditions from the CSV;
- `block_phase` derived from each trial's position within its block;
- saved trial-level hierarchical predictions;
- saved inferred hidden-kappa trajectories.

Each participant receives one page. The top panel shows the inferred expected hidden kappa through time. The nine lower panels compare observed responses with hierarchical predictions for early, middle, and late block trials. No model fitting is performed.
"""
    ),
    md(
        r"""
## Step 1 - Locate inputs and declare the plotted conditions

**Input:** `outputs/gpu_long_multistart`, the raw CSV recorded in the long-run configuration, and the current project source.

**Process:** use the same conditions as the switching example: 6% coherence, prior SD 80 degrees, and directions 65, 145, and 225 degrees. Symmetric directions are folded around the fixed 225-degree prior mean.

**Output:** paths and plotting constants only.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import i0


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "hierarchical_confidence").exists():
            return candidate
    raise FileNotFoundError("Could not find independent_transcript_paper_model.")


ROOT = find_project_root(Path.cwd().resolve())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_confidence import load_and_prepare_data
from hierarchical_confidence.circular import angle_to_bin, circular_difference_degrees


FIT_OUTPUT_DIR = Path(
    os.getenv("HIERARCHICAL_LONG_OUTPUT_DIR", ROOT / "outputs" / "gpu_long_multistart")
).resolve()
PLOT_OUTPUT_DIR = Path(
    os.getenv("HIERARCHICAL_COMPARISON_OUTPUT_DIR", ROOT / "outputs" / "model_data_comparison")
).resolve()

with (FIT_OUTPUT_DIR / "long_run_config.json").open(encoding="utf-8") as handle:
    fit_config = json.load(handle)

DATA_PATH = Path(os.getenv("HIERARCHICAL_DATA_PATH", fit_config["data_path"])).resolve()
COHERENCE = 0.06
PRIOR_SD = 80.0
TARGET_DIRECTIONS = (65.0, 145.0, 225.0)
BLOCK_PHASES = ("early", "middle", "late")
PRIOR_MEAN = 225.0
SMOOTHING_KAPPA = 12.0

PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Original data: {DATA_PATH}")
print(f"Saved hierarchical fit: {FIT_OUTPUT_DIR}")
print(f"History-sensitive figures: {PLOT_OUTPUT_DIR}")
"""
    ),
    md(
        r"""
## Step 2 - Load and validate trials, predictions, and hidden states

**Depends on:** Step 1 paths.

**Process:** reproduce the chronological trial ordering used during fitting. For every participant, verify that both the prediction matrix and hidden-state table contain exactly one row per original trial. Verify that all 72-bin prediction rows sum to one and that hidden-state trial positions are consecutive.

**Output:** validated data structures for all participants. Failed optimizer fits remain included but are labelled on their pages.
"""
    ),
    code(
        r"""
data, data_audit = load_and_prepare_data(DATA_PATH)
fits = pd.read_csv(FIT_OUTPUT_DIR / "best_multistart_fit.csv")
diagnostics = pd.read_csv(FIT_OUTPUT_DIR / "multistart_optimization_diagnostics.csv")
fit_status = fits[["subject_id", "success", "nll", "status"]].merge(
    diagnostics[["subject_id", "selected_parameter_near_bound"]],
    on="subject_id",
    how="left",
    validate="one_to_one",
)

participant_ids = sorted(data["subject_id"].unique().tolist())
predictions_by_subject = {}
hidden_by_subject = {}
theta_degrees = None

for subject_id in participant_ids:
    rows = data.loc[data["subject_id"] == subject_id]

    prediction_path = FIT_OUTPUT_DIR / f"participant_{subject_id}_gpu_trial_predictions.npz"
    with np.load(prediction_path) as saved:
        response_pmf = saved["response_pmf"].astype(float)
        subject_theta = saved["theta_degrees"].astype(float)

    hidden = pd.read_csv(FIT_OUTPUT_DIR / f"participant_{subject_id}_hidden_state.csv")

    if response_pmf.shape != (len(rows), len(subject_theta)):
        raise ValueError(f"Participant {subject_id}: prediction shape does not match trials.")
    if not np.allclose(response_pmf.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError(f"Participant {subject_id}: saved response PMFs do not normalize.")
    if len(hidden) != len(rows):
        raise ValueError(f"Participant {subject_id}: hidden-state rows do not match trials.")
    if not np.array_equal(
        hidden["subject_trial_position"].to_numpy(dtype=int), np.arange(len(rows))
    ):
        raise ValueError(f"Participant {subject_id}: hidden-state trial positions are not consecutive.")
    if not np.array_equal(
        hidden["block_id"].to_numpy(dtype=str), rows["block_id"].to_numpy(dtype=str)
    ):
        raise ValueError(f"Participant {subject_id}: hidden-state block order does not match data.")

    if theta_degrees is None:
        theta_degrees = subject_theta
    elif not np.array_equal(theta_degrees, subject_theta):
        raise ValueError("Participants do not share the same response grid.")

    predictions_by_subject[subject_id] = response_pmf
    hidden_by_subject[subject_id] = hidden

print(
    f"Validated {len(participant_ids)} participants and "
    f"{sum(len(value) for value in hidden_by_subject.values()):,} aligned trials."
)
display(fit_status)
"""
    ),
    md(
        r"""
## Step 3 - Build phase-specific observed and predicted distributions

For each block, `block_phase` divides trial position into thirds:

- early: first third;
- middle: second third;
- late: final third.

The 65-degree condition is pooled with reflected 25-degree trials; 145 degrees is pooled with reflected 305-degree trials. Both observed responses and the corresponding predicted PMFs are reflected around 225 degrees before pooling. The 225-degree condition is included only once.

Observed and predicted curves receive the same von Mises smoothing (`kappa=12`) for visualization. Numerical summaries include both unsmoothed and visual smoothed total-variation distance.
"""
    ),
    code(
        r"""
N_ANGLES = len(theta_degrees)
RELATIVE_BIN_CENTERS = circular_difference_degrees(theta_degrees, PRIOR_MEAN)
REFLECTED_BIN_INDEX = angle_to_bin((2.0 * PRIOR_MEAN - theta_degrees) % 360.0, N_ANGLES)
PLOT_GRID = np.linspace(-180.0, 180.0, 361)


def von_mises_kernel(evaluation_degrees, source_degrees, kappa):
    difference_radians = np.deg2rad(
        evaluation_degrees[:, None] - np.asarray(source_degrees)[None, :]
    )
    density_per_radian = np.exp(kappa * np.cos(difference_radians)) / (
        2.0 * np.pi * i0(kappa)
    )
    return density_per_radian * np.deg2rad(1.0)


PLOT_KERNEL_FROM_BINS = von_mises_kernel(
    PLOT_GRID, RELATIVE_BIN_CENTERS, SMOOTHING_KAPPA
)


def select_folded_phase(subject_rows, subject_predictions, target_direction, phase):
    mirror_direction = float((2.0 * PRIOR_MEAN - target_direction) % 360.0)
    directions = subject_rows["motion_direction"].to_numpy(dtype=float)
    target_mask = np.isclose(directions, target_direction)
    if np.isclose(target_direction, mirror_direction):
        mirror_mask = np.zeros(len(subject_rows), dtype=bool)
    else:
        mirror_mask = np.isclose(directions, mirror_direction)

    selected_mask = (
        subject_rows["response_valid"].to_numpy(dtype=bool)
        & np.isclose(subject_rows["motion_coherence"].to_numpy(dtype=float), COHERENCE)
        & np.isclose(subject_rows["prior_std"].to_numpy(dtype=float), PRIOR_SD)
        & (subject_rows["block_phase"].astype(str).to_numpy() == phase)
        & (target_mask | mirror_mask)
    )
    selected_rows = subject_rows.loc[selected_mask].copy()
    selected_predictions = subject_predictions[selected_mask].copy()
    selected_is_mirror = mirror_mask[selected_mask]

    if selected_is_mirror.any():
        selected_rows.loc[selected_is_mirror, "response_angle"] = (
            2.0 * PRIOR_MEAN
            - selected_rows.loc[selected_is_mirror, "response_angle"].to_numpy(dtype=float)
        ) % 360.0
        selected_predictions[selected_is_mirror] = selected_predictions[
            selected_is_mirror
        ][:, REFLECTED_BIN_INDEX]

    return selected_rows, selected_predictions


def compare_phase_condition(subject_id, target_direction, phase):
    subject_rows = data.loc[data["subject_id"] == subject_id].copy()
    selected_rows, selected_predictions = select_folded_phase(
        subject_rows,
        predictions_by_subject[subject_id],
        target_direction,
        phase,
    )
    if selected_rows.empty:
        raise ValueError(
            f"Participant {subject_id}: no valid {phase} trials at direction {target_direction}."
        )

    response_angles = selected_rows["response_angle"].to_numpy(dtype=float)
    response_relative = circular_difference_degrees(response_angles, PRIOR_MEAN)
    response_bins = angle_to_bin(response_angles, N_ANGLES)
    empirical_pmf = np.bincount(response_bins, minlength=N_ANGLES).astype(float)
    empirical_pmf /= empirical_pmf.sum()
    mean_model_pmf = selected_predictions.mean(axis=0)

    observed_curve = von_mises_kernel(
        PLOT_GRID, response_relative, SMOOTHING_KAPPA
    ).mean(axis=1)
    model_curve = PLOT_KERNEL_FROM_BINS @ mean_model_pmf
    trial_probability = selected_predictions[
        np.arange(len(selected_rows)), response_bins
    ]

    summary = {
        "subject_id": int(subject_id),
        "block_phase": phase,
        "target_direction": float(target_direction),
        "motion_coherence": COHERENCE,
        "prior_std": PRIOR_SD,
        "n_valid_trials": int(len(selected_rows)),
        "total_variation": float(0.5 * np.abs(empirical_pmf - mean_model_pmf).sum()),
        "smoothed_total_variation_visual": float(
            0.5 * np.abs(observed_curve - model_curve).sum()
        ),
        "mean_trial_nll": float(-np.log(np.maximum(trial_probability, 1e-300)).mean()),
    }
    return observed_curve, model_curve, summary


curves = {}
summary_records = []
for subject_id in participant_ids:
    for phase in BLOCK_PHASES:
        for target_direction in TARGET_DIRECTIONS:
            observed_curve, model_curve, summary = compare_phase_condition(
                subject_id, target_direction, phase
            )
            curves[(subject_id, phase, target_direction)] = (
                observed_curve,
                model_curve,
            )
            summary_records.append(summary)

phase_summary = pd.DataFrame(summary_records).merge(
    fit_status[["subject_id", "success", "selected_parameter_near_bound"]],
    on="subject_id",
    how="left",
    validate="many_to_one",
)
phase_summary.to_csv(
    PLOT_OUTPUT_DIR / "phase_condition_model_data_summary.csv", index=False
)

subject_1_totals = (
    phase_summary.loc[phase_summary["subject_id"] == 1]
    .groupby("target_direction")["n_valid_trials"]
    .sum()
    .to_dict()
)
if subject_1_totals != {65.0: 22, 145.0: 96, 225.0: 96}:
    raise AssertionError(f"Unexpected folded participant 1 totals: {subject_1_totals}")

display(phase_summary)
"""
    ),
    md(
        r"""
## Step 4 - Plot hidden learning and phase-specific predictions

**Top panel:** raw expected hidden kappa in light orange, a 15-trial rolling mean in dark orange, true block prior SD on the right axis, and block boundaries in gray.

**Lower grid:** rows are early, middle, and late block phases; columns are far from prior, near prior, and at the prior mean. Maroon is the observed response density, orange is the hierarchical prediction, blue marks the prior mean, and gray dashed marks the target stimulus.

**Output:** one PNG per participant and a 12-page PDF. A `NON-CONVERGED FIT` label means the page is descriptive and should not be used for parameter interpretation.
"""
    ),
    code(
        r"""
DATA_COLOR = "#64130f"
MODEL_COLOR = "#e58b2a"
MODEL_RAW_COLOR = "#efbd7c"
PRIOR_COLOR = "#3478b8"
STIMULUS_COLOR = "#666666"
PRIOR_SD_COLOR = "#16827b"


def fit_label(subject_id):
    row = fit_status.loc[fit_status["subject_id"] == subject_id].iloc[0]
    if not bool(row["success"]):
        return "NON-CONVERGED FIT"
    if bool(row["selected_parameter_near_bound"]):
        return "converged; parameter near bound"
    return "converged"


def participant_history_figure(subject_id):
    fig = plt.figure(figsize=(15.5, 11.5))
    grid_spec = fig.add_gridspec(
        4,
        3,
        height_ratios=(1.25, 1.0, 1.0, 1.0),
        hspace=0.55,
        wspace=0.18,
    )

    trajectory_axis = fig.add_subplot(grid_spec[0, :])
    hidden = hidden_by_subject[subject_id]
    trial_position = hidden["subject_trial_position"].to_numpy(dtype=int)
    expected_kappa = hidden["expected_hidden_kappa"].to_numpy(dtype=float)
    smoothed_kappa = (
        pd.Series(expected_kappa).rolling(15, center=True, min_periods=1).mean().to_numpy()
    )
    trajectory_axis.plot(
        trial_position,
        expected_kappa,
        color=MODEL_RAW_COLOR,
        linewidth=0.45,
        alpha=0.65,
        label="expected hidden kappa (trial)",
    )
    trajectory_axis.plot(
        trial_position,
        smoothed_kappa,
        color=MODEL_COLOR,
        linewidth=1.25,
        label="expected hidden kappa (15-trial mean)",
    )

    block_change = hidden["block_id"].ne(hidden["block_id"].shift()).to_numpy()
    for boundary in trial_position[block_change][1:]:
        trajectory_axis.axvline(boundary, color="#d4d4d4", linewidth=0.45, zorder=0)

    prior_axis = trajectory_axis.twinx()
    prior_axis.step(
        trial_position,
        hidden["prior_std_diagnostic_only"].to_numpy(dtype=float),
        where="post",
        color=PRIOR_SD_COLOR,
        linewidth=0.9,
        alpha=0.8,
        label="true prior SD (diagnostic only)",
    )
    prior_axis.set_ylabel("true prior SD (degrees)", color=PRIOR_SD_COLOR)
    prior_axis.tick_params(axis="y", colors=PRIOR_SD_COLOR)
    prior_axis.set_ylim(0, 90)

    trajectory_axis.set_xlim(0, max(trial_position))
    trajectory_axis.set_ylabel("inferred expected kappa")
    trajectory_axis.set_xlabel("chronological trial position")
    trajectory_axis.set_title("Inferred hidden-prior reliability across trials")
    trajectory_axis.spines["top"].set_visible(False)
    prior_axis.spines["top"].set_visible(False)
    trajectory_axis.grid(axis="y", color="#e2e2e2", linewidth=0.6)
    handles_1, labels_1 = trajectory_axis.get_legend_handles_labels()
    handles_2, labels_2 = prior_axis.get_legend_handles_labels()
    trajectory_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    response_axes = []
    column_names = ("far from prior", "near prior", "at prior mean")
    for row_index, phase in enumerate(BLOCK_PHASES, start=1):
        for column_index, (target_direction, column_name) in enumerate(
            zip(TARGET_DIRECTIONS, column_names)
        ):
            axis = fig.add_subplot(grid_spec[row_index, column_index])
            response_axes.append(axis)
            observed_curve, model_curve = curves[(subject_id, phase, target_direction)]
            row = phase_summary.loc[
                (phase_summary["subject_id"] == subject_id)
                & (phase_summary["block_phase"] == phase)
                & np.isclose(phase_summary["target_direction"], target_direction)
            ].iloc[0]

            axis.fill_between(
                PLOT_GRID,
                observed_curve,
                color=DATA_COLOR,
                label="observed data",
            )
            axis.plot(
                PLOT_GRID,
                model_curve,
                color=MODEL_COLOR,
                linewidth=2.0,
                label="hierarchical model",
            )
            axis.axvline(0.0, color=PRIOR_COLOR, linewidth=1.7, label="prior mean")
            stimulus_relative = float(
                circular_difference_degrees(target_direction, PRIOR_MEAN)
            )
            axis.axvline(
                stimulus_relative,
                color=STIMULUS_COLOR,
                linewidth=1.0,
                linestyle="--",
                label="stimulus",
            )
            axis.set_title(
                f"{phase} | {column_name}: dir {target_direction:g} "
                f"(n={int(row['n_valid_trials'])})",
                fontsize=9,
            )
            axis.set_xlim(-180, 180)
            axis.set_xticks([-150, -100, -50, 0, 50, 100, 150])
            axis.grid(axis="y", color="#e2e2e2", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
            if column_index == 0:
                axis.set_ylabel("probability per degree")
            if row_index == 3:
                axis.set_xlabel("estimate - prior mean (degrees)")

    handles, labels = response_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"Participant {subject_id}: hierarchical learning and responses by block phase\n"
        f"6% coherence, prior SD 80 degrees | {fit_label(subject_id)}",
        fontsize=14,
        y=0.995,
    )
    fig.subplots_adjust(top=0.93, bottom=0.075, left=0.065, right=0.94)
    return fig


pdf_path = PLOT_OUTPUT_DIR / "all_participants_hierarchy_by_block_phase.pdf"
with PdfPages(pdf_path) as pdf:
    for subject_id in participant_ids:
        figure = participant_history_figure(subject_id)
        figure.savefig(
            PLOT_OUTPUT_DIR / f"participant_{subject_id}_hierarchy_by_block_phase.png",
            dpi=160,
            bbox_inches="tight",
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

phase_participant_summary = (
    phase_summary.groupby(
        ["subject_id", "block_phase", "success", "selected_parameter_near_bound"],
        as_index=False,
        observed=True,
    )
    .agg(
        plotted_trials=("n_valid_trials", "sum"),
        mean_smoothed_total_variation_visual=(
            "smoothed_total_variation_visual",
            "mean",
        ),
        mean_trial_nll=("mean_trial_nll", "mean"),
    )
)
phase_participant_summary.to_csv(
    PLOT_OUTPUT_DIR / "phase_participant_summary.csv", index=False
)

comparison_config = {
    "source_fit_output": str(FIT_OUTPUT_DIR),
    "data_path": str(DATA_PATH),
    "participants": [int(value) for value in participant_ids],
    "coherence": COHERENCE,
    "prior_sd": PRIOR_SD,
    "target_directions": list(TARGET_DIRECTIONS),
    "block_phases": list(BLOCK_PHASES),
    "fold_symmetric_directions": True,
    "smoothing_kappa_visual_only": SMOOTHING_KAPPA,
    "models_plotted": ["hierarchical"],
    "refit_performed": False,
}
with (PLOT_OUTPUT_DIR / "phase_comparison_config.json").open("w", encoding="utf-8") as handle:
    json.dump(comparison_config, handle, indent=2)

display(phase_participant_summary)
print(f"Saved 12 participant PNGs and 12-page PDF: {pdf_path}")
print("No model fitting was performed.")
"""
    ),
    md(
        r"""
## Reading the result

The trajectory panel shows whether inferred prior reliability follows changes in the stimulus-generating distribution. The lower panels show whether that changing hidden state produces visible early-to-late changes in predicted responses.

Small early/middle/late sample sizes, especially in the far-from-prior condition, make individual maroon curves noisy. The plots are descriptive posterior-predictive checks. They do not compare the hierarchical observer with a static Bayesian or switching observer.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
