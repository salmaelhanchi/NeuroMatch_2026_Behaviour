"""Build the readable main notebook from versionable cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_30_minute_hierarchical_fit.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# First 30-minute hierarchical prior-confidence fit

## Research question

Can a participant learn a hidden belief about **prior concentration** across trials and blocks, while the prior mean remains fixed at 225 degrees, and can that belief explain the observed response distributions?

This notebook is the main entry point for the first exploratory run. It fits two contrasting participants with a deliberately coarse numerical grid and a strict time budget. These results are for implementation checking and first scientific inspection, not final model comparison.

### Source labels used throughout

- **Paper-defined:** circular sensory likelihood, posterior readout, motor noise, lapse, and participant-specific fitting.
- **TA-directed:** learn prior confidence rather than prior mean; carry the hidden state between blocks; inspect distributions and recovery before comparison.
- **Implementation choice:** 72 angle bins, 16 fixed kappa support points, a uniform initial hidden state, six-decimal tie-aware MAP readout, and one Powell optimizer start.
- **Empirical decision:** choose two contrasting pilot participants using a transparent score calculated from observed responses.

The 16 kappa values are fixed support points for the hidden distribution. They are **not 16 fitted parameters**.
"""
    ),
    md(
        r"""
## Whole-project position of this run

```text
raw trials
  -> validate chronology and responses
  -> derive circular variables
  -> choose two contrasting pilot participants
  -> create fixed direction and kappa grids
  -> fit one chronological model per participant
  -> inspect optimization, response distributions, and hidden-state trajectories
  -> decide whether the implementation is ready for simulation and recovery
  -> only later fit all participants and compare models
```

Every section below states its dependency and output. A later section does not silently redefine an earlier variable.
"""
    ),
    md(
        r"""
## Step 0 - Runtime configuration

**Depends on:** no previous result.

**Process:** set paths and the numerical budget. The environment variable `HIERARCHICAL_SMOKE_TEST=1` activates a very small verification run; without it, this notebook uses the planned 30-minute settings.

**Produces:** one configuration shared by every later step.

**Why it matters:** grid resolution controls numerical approximation, while the evaluation and time limits control how long the exploratory optimization can run.
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


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "hierarchical_confidence").exists():
            return candidate
    raise FileNotFoundError("Could not find the independent_transcript_paper_model project root.")


ROOT = find_project_root(Path.cwd().resolve())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_confidence import (
    GridSpec,
    HierarchicalObserver,
    fit_subject,
    load_and_prepare_data,
    pilot_selection_table,
    prepare_subject,
)
from hierarchical_confidence.circular import angle_to_bin, wrap_degrees_signed

SMOKE_TEST = os.getenv("HIERARCHICAL_SMOKE_TEST", "0") == "1"
DATA_PATH = Path(
    os.getenv(
        "HIERARCHICAL_DATA_PATH",
        str(Path.home() / "Downloads" / "data01_direction4priors.csv"),
    )
).expanduser()

RUN_NAME = "smoke_test" if SMOKE_TEST else "pilot_30min"
OUTPUT_DIR = ROOT / "outputs" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_CONFIG = {
    "smoke_test": SMOKE_TEST,
    "n_angles": 24 if SMOKE_TEST else 72,
    "n_positive_kappa": 5 if SMOKE_TEST else 15,
    "total_kappa_support_points": 6 if SMOKE_TEST else 16,
    "kappa_min": 0.05,
    "kappa_max": 50.0,
    "pilot_participants": 2,
    "max_evaluations_per_participant": 2 if SMOKE_TEST else 100,
    "time_budget_minutes_per_participant": 0.25 if SMOKE_TEST else 12.0,
    "optimizer": "Powell, one bounded start",
    "posterior_readout": "MAP",
    "fixed_prior_mean_degrees": 225.0,
}

display(pd.DataFrame([RUN_CONFIG]).T.rename(columns={0: "value"}))
print(f"Data: {DATA_PATH}")
print(f"Outputs: {OUTPUT_DIR}")
"""
    ),
    md(
        r"""
## Step 1 - Load, order, and validate trials

**Depends on:** `DATA_PATH` from Step 0.

**Process:** require the experiment columns; verify one fixed prior mean at 225 degrees; verify unique participant/session/run/trial keys; and sort by participant, session, run, then trial index.

**Produces:** `data`, one chronological table, and `data_audit`, a compact record of the checks.

**Why it matters:** the hidden confidence state depends on earlier feedback. Sorting by experimental condition or dropping missing responses would change that state and therefore change the scientific model.
"""
    ),
    code(
        r"""
data, data_audit = load_and_prepare_data(DATA_PATH)
display(pd.DataFrame([data_audit]).T.rename(columns={0: "value"}))

with (OUTPUT_DIR / "data_audit.json").open("w", encoding="utf-8") as handle:
    json.dump(data_audit, handle, indent=2)

assert data_audit["subjects"] == 12
assert data_audit["fixed_prior_mean"] == 225.0
assert data["response_valid"].sum() + (~data["response_valid"]).sum() == len(data)
print("Data-contract checks passed.")
"""
    ),
    md(
        r"""
## Step 2 - Understand the derived variables

**Depends on:** the validated chronological table from Step 1.

**Process:** convert response coordinates to an angle and calculate signed circular differences.

**Produces:** variables used for fitting or diagnostics:

| Derived variable | Origin | How it helps answer the question |
|---|---|---|
| `block_id` | participant + session + run | Locates boundaries while ensuring the hidden state is not reset there. |
| `response_valid` | both response coordinates are finite | Missing responses are excluded from likelihood scoring but retained for feedback learning. |
| `response_angle` | `atan2(estimate_y, estimate_x) mod 360` | Converts the recorded response to the same circular space as stimuli and predictions. |
| `stimulus_from_prior` | shortest signed difference: stimulus - 225 | Measures conflict between sensory evidence and the fixed prior mean. |
| `response_from_prior` | shortest signed difference: response - 225 | Shows attraction toward or away from the fixed prior. |
| `response_error` | shortest signed difference: response - stimulus | Shows attraction toward sensory evidence and estimation error. |
| `block_phase` | relative trial position within a block | Supports early/middle/late adaptation diagnostics; it is not a model input. |

The raw `prior_std` condition is retained for later diagnostic comparison. The hierarchical model never reads it when initializing or updating hidden confidence.
"""
    ),
    code(
        r"""
derived_columns = [
    "subject_id", "session_id", "run_id", "trial_index", "block_id",
    "motion_direction", "motion_coherence", "prior_std", "prior_mean",
    "response_valid", "response_angle", "stimulus_from_prior",
    "response_from_prior", "response_error", "block_phase",
]
display(data[derived_columns].head(8))

boundary_demo = pd.DataFrame(
    {
        "quantity": ["difference(1, 359)", "difference(359, 1)"],
        "ordinary_subtraction": [-358, 358],
        "circular_difference": [2, -2],
    }
)
display(boundary_demo)
"""
    ),
    md(
        r"""
## Step 3 - Select two contrasting pilot participants

**Depends on:** the circular variables from Step 2.

**Process:** on lowest-coherence trials at least 60 degrees from the prior, calculate response mass within 20 degrees of the prior and within 20 degrees of the stimulus. Their geometric mean is the `dual_attraction_score`.

**Produces:** the participant with the highest score and the participant with the lowest score.

**Why it matters:** the first implementation should be exercised on both a strong dual-attraction pattern and a weaker pattern. This score only chooses pilots. It is not supplied to the model, is not a formal bimodality statistic, and does not determine the result.
"""
    ),
    code(
        r"""
selection = pilot_selection_table(data)
strong_subject = int(selection.iloc[0]["subject_id"])
weak_subject = int(selection.iloc[-1]["subject_id"])
pilot_subjects = [strong_subject, weak_subject]
selection["pilot_role"] = "not selected"
selection.loc[selection["subject_id"] == strong_subject, "pilot_role"] = "strong dual attraction"
selection.loc[selection["subject_id"] == weak_subject, "pilot_role"] = "weak dual attraction"
selection.to_csv(OUTPUT_DIR / "pilot_selection.csv", index=False)
display(selection)
print(f"Pilot participants: {pilot_subjects}")

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#d1495b" if sid == strong_subject else "#397367" if sid == weak_subject else "#aab2bd"
          for sid in selection["subject_id"]]
ax.bar(selection["subject_id"].astype(str), selection["dual_attraction_score"], color=colors)
ax.set(xlabel="Participant", ylabel="Dual-attraction selection score", title="Transparent pilot selection")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pilot_selection.png", dpi=160)
plt.show()
"""
    ),
    md(
        r"""
## Step 4 - Define the numerical representation

**Depends on:** only the runtime configuration from Step 0.

**Process:** represent direction using 72 centers separated by 5 degrees and hidden concentration using `0` plus 15 logarithmically spaced positive values from 0.05 to 50.

**Produces:** `grid`, the fixed numerical support used by both pilot fits.

**Why it matters:** `kappa = 0` represents an exactly uniform prior. Log spacing gives more detail near zero, where small concentration changes matter strongly. The participant learns probability mass across these candidates; the candidates themselves are not fitted.
"""
    ),
    code(
        r"""
grid = GridSpec(
    n_angles=RUN_CONFIG["n_angles"],
    n_positive_kappa=RUN_CONFIG["n_positive_kappa"],
    kappa_min=RUN_CONFIG["kappa_min"],
    kappa_max=RUN_CONFIG["kappa_max"],
)
print("Direction grid (degrees):")
print(grid.theta_degrees)
print("\nFixed kappa support:")
print(np.round(grid.kappa_values, 4))
assert len(grid.theta_degrees) == RUN_CONFIG["n_angles"]
assert len(grid.kappa_values) == RUN_CONFIG["total_kappa_support_points"]
assert grid.kappa_values[0] == 0.0
"""
    ),
    md(
        r"""
## Step 5 - Specify the trial calculation

**Depends on:** the chronological sequence from Step 1 and numerical grid from Step 4.

For each participant, the model fits six participant-level quantities:

1. memory `rho`;
2. one sensory concentration for each of the three coherence levels;
3. motor concentration;
4. lapse probability.

On trial $t$, the calculation is ordered as follows:

1. Discount the previous hidden state: $H_t^-(\kappa) \propto H_t(\kappa)^\rho$.
2. Marginalize concentration into one effective prior:
   $P_t(\theta)=\sum_\kappa H_t^-(\kappa)\,VM(\theta;225,\kappa)$.
3. For every possible internal measurement, multiply its sensory likelihood by that one effective prior.
4. Normalize each posterior, round it to six decimals, and apply one shared tie-aware MAP readout; every tied maximum receives equal mass.
5. Integrate over possible internal measurements.
6. Add motor noise and lapse, then score the observed response if present.
7. After prediction, update hidden concentration using the revealed true direction.

The final state of one block enters the first trial of the next block. `prior_std` is not used in these calculations.

**Produces:** one participant-level negative log-likelihood (NLL). Lower NLL means the model assigned more probability to the observed responses.
"""
    ),
    code(
        r"""
prepared = {subject_id: prepare_subject(data, subject_id, grid) for subject_id in pilot_subjects}
observers = {
    subject_id: HierarchicalObserver(prepared[subject_id], grid, batch_size=128)
    for subject_id in pilot_subjects
}

pre_fit_summary = pd.DataFrame(
    [
        {
            "subject_id": subject_id,
            "trials_used_for_state_updates": subject.n_trials,
            "valid_responses_scored": int(subject.response_valid.sum()),
            "missing_responses_not_scored": int((~subject.response_valid).sum()),
            "blocks_carried_across": len(np.unique(subject.block_ids)),
        }
        for subject_id, subject in prepared.items()
    ]
)
display(pre_fit_summary)
"""
    ),
    md(
        r"""
## Step 6 - Fit within the exploratory budget

**Depends on:** model objects from Step 5.

**Process:** run one bounded Powell optimization per participant. Positive concentrations are optimized on a log scale; `rho` and lapse are optimized on a logit scale. Each fit stops after 100 evaluations or 12 minutes, whichever comes first.

**Produces:** best parameters found, NLL, evaluation history, elapsed time, and optimizer status for each participant.

**Why it matters:** this first run asks whether the implementation can improve predictions and produce interpretable diagnostics. Reaching the time or evaluation budget does not mean numerical convergence; multiple starts and recovery come later.
"""
    ),
    code(
        r"""
fit_results = {}
for subject_id in pilot_subjects:
    print(f"Fitting participant {subject_id} ...")
    result = fit_subject(
        observers[subject_id],
        max_evaluations=RUN_CONFIG["max_evaluations_per_participant"],
        time_budget_seconds=60.0 * RUN_CONFIG["time_budget_minutes_per_participant"],
    )
    fit_results[subject_id] = result
    result.history.to_csv(OUTPUT_DIR / f"participant_{subject_id}_optimization_history.csv", index=False)
    print(
        f"  NLL {result.initial_nll:.2f} -> {result.nll:.2f}; "
        f"{result.evaluations} evaluations in {result.elapsed_seconds / 60.0:.2f} minutes"
    )
    print(f"  Status: {result.status}")

fit_summary = pd.DataFrame(
    [result.summary_record(prepared[subject_id].coherence_values)
     for subject_id, result in fit_results.items()]
)
fit_summary.to_csv(OUTPUT_DIR / "fit_summary.csv", index=False)
with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as handle:
    json.dump(RUN_CONFIG, handle, indent=2)
display(fit_summary)
"""
    ),
    md(
        r"""
## Step 7 - Inspect optimization rather than trusting one number

**Depends on:** the evaluation histories from Step 6.

**Produces:** NLL traces and explicit budget/convergence messages.

**Interpretation:** a falling trace shows that fitting improved on the declared starting parameters. A flat, erratic, or boundary-limited trace signals that the implementation or fitting budget needs revision before scientific interpretation.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, len(pilot_subjects), figsize=(12, 4), squeeze=False)
for ax, subject_id in zip(axes[0], pilot_subjects):
    history = fit_results[subject_id].history
    ax.plot(history["evaluation"], history["nll"], color="#2b6f77", linewidth=1.8)
    ax.scatter(history.iloc[-1]["evaluation"], history.iloc[-1]["nll"], color="#d1495b", s=24)
    ax.set(title=f"Participant {subject_id}", xlabel="Objective evaluation", ylabel="NLL")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Exploratory optimization traces")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "optimization_traces.png", dpi=160)
plt.show()
"""
    ),
    md(
        r"""
## Step 8 - Generate predictions from the fitted calculation

**Depends on:** fitted parameters from Step 6. This step must not run before fitting.

**Process:** rerun each full chronological sequence with the best parameters, now retaining every predicted response probability and every pre-feedback hidden state. Also record how often the normalized, six-decimal posterior has more than one MAP maximum.

**Produces:** normalized trial-level response distributions, tie-aware MAP diagnostics, and hidden-state summaries. `expected_hidden_kappa` is the probability-weighted average of the fixed kappa support; `modal_hidden_kappa` is the support point with most mass; entropy records remaining uncertainty.

**Why it matters:** the research question concerns response-distribution shape and learned confidence, not NLL alone.
"""
    ),
    code(
        r"""
predictions = {}
state_summaries = {}
for subject_id in pilot_subjects:
    parameters = fit_results[subject_id].parameters
    predictions[subject_id] = observers[subject_id].predict_response_pmfs(parameters)
    state_summaries[subject_id] = observers[subject_id].state_summary(parameters.rho)
    np.savez_compressed(
        OUTPUT_DIR / f"participant_{subject_id}_trial_predictions.npz",
        response_pmf=predictions[subject_id],
        theta_degrees=grid.theta_degrees,
    )
    state_summaries[subject_id].to_csv(
        OUTPUT_DIR / f"participant_{subject_id}_hidden_state.csv", index=False
    )
    assert np.allclose(predictions[subject_id].sum(axis=1), 1.0, atol=1e-8)
print("All saved trial-level response distributions sum to one.")

tie_diagnostics = pd.DataFrame([
    {
        "subject_id": subject_id,
        **observers[subject_id].tie_diagnostics(fit_results[subject_id].parameters),
    }
    for subject_id in pilot_subjects
])
tie_diagnostics.to_csv(OUTPUT_DIR / "tie_aware_map_diagnostics.csv", index=False)
display(tie_diagnostics)
print("Tie-aware MAP diagnostics use normalized posteriors rounded to six decimals.")"""
    ),
    md(
        r"""
## Step 9 - Compare observed and predicted distribution shapes

**Depends on:** full trial predictions from Step 8 and observed circular variables from Step 2.

**Process:** for display only, choose the low-coherence, high-conflict condition with the strongest observable dual-attraction score within each pilot participant. Plot response direction relative to the fixed prior mean.

**Produces:** one readable observed-versus-predicted diagnostic per pilot.

The diagnostic condition is selected after inspecting responses, so it is descriptive and cannot serve as confirmatory evidence. The model itself was fitted to every valid response from the participant, not only these plotted trials.
"""
    ),
    code(
        r"""
def choose_diagnostic_condition(subject_rows: pd.DataFrame) -> pd.Series:
    low = subject_rows["motion_coherence"].min()
    eligible = subject_rows.loc[
        subject_rows["response_valid"]
        & np.isclose(subject_rows["motion_coherence"], low)
        & (np.abs(subject_rows["stimulus_from_prior"]) >= 60.0)
    ].copy()
    eligible["near_prior"] = np.abs(eligible["response_from_prior"]) <= 20.0
    eligible["near_stimulus"] = np.abs(eligible["response_error"]) <= 20.0
    grouped = (
        eligible.groupby(["prior_std", "motion_direction", "motion_coherence"], as_index=False)
        .agg(trials=("response_valid", "size"), prior_mass=("near_prior", "mean"),
             stimulus_mass=("near_stimulus", "mean"))
    )
    grouped = grouped.loc[grouped["trials"] >= 8].copy()
    grouped["dual_attraction_score"] = np.sqrt(grouped["prior_mass"] * grouped["stimulus_mass"])
    return grouped.sort_values(["dual_attraction_score", "trials"], ascending=False).iloc[0]


diagnostic_records = []
fig, axes = plt.subplots(1, len(pilot_subjects), figsize=(13, 4.5), squeeze=False)
relative_theta = wrap_degrees_signed(grid.theta_degrees - 225.0)
plot_order = np.argsort(relative_theta)
bin_width = 360.0 / grid.n_angles

for ax, subject_id in zip(axes[0], pilot_subjects):
    rows = data.loc[data["subject_id"] == subject_id].copy().reset_index(drop=True)
    condition = choose_diagnostic_condition(rows)
    mask = (
        np.isclose(rows["prior_std"], condition["prior_std"])
        & np.isclose(rows["motion_direction"], condition["motion_direction"])
        & np.isclose(rows["motion_coherence"], condition["motion_coherence"])
        & rows["response_valid"]
    ).to_numpy()
    observed_bins = angle_to_bin(rows.loc[mask, "response_angle"].to_numpy(), grid.n_angles)
    observed = np.bincount(observed_bins, minlength=grid.n_angles).astype(float)
    observed /= observed.sum()
    predicted = predictions[subject_id][mask].mean(axis=0)
    stimulus_relative = float(wrap_degrees_signed(condition["motion_direction"] - 225.0))

    ax.bar(relative_theta[plot_order], observed[plot_order], width=0.82 * bin_width,
           color="#b8bec5", label="Observed")
    ax.plot(relative_theta[plot_order], predicted[plot_order], color="#d1495b",
            linewidth=2.2, label="Hierarchical prediction")
    ax.axvline(0.0, color="#2b6f77", linestyle="--", linewidth=1.3, label="Prior mean")
    ax.axvline(stimulus_relative, color="#111111", linestyle=":", linewidth=1.3, label="Stimulus")
    ax.set(
        title=f"Participant {subject_id}: SD {condition['prior_std']:g}, direction {condition['motion_direction']:g}",
        xlabel="Response relative to 225 degrees",
        ylabel="Probability",
        xlim=(-180, 180),
    )
    ax.spines[["top", "right"]].set_visible(False)
    diagnostic_records.append({"subject_id": subject_id, **condition.to_dict()})

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
fig.suptitle("Observed and predicted response distributions", y=1.04)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "observed_vs_predicted.png", dpi=160, bbox_inches="tight")
plt.show()

diagnostic_conditions = pd.DataFrame(diagnostic_records)
diagnostic_conditions.to_csv(OUTPUT_DIR / "diagnostic_conditions.csv", index=False)
display(diagnostic_conditions)
"""
    ),
    md(
        r"""
## Step 10 - Inspect learned confidence across blocks

**Depends on:** hidden-state summaries from Step 8.

**Process:** plot expected hidden kappa through chronological trials and compare it with the experiment's block-level `prior_std` label.

**Produces:** a trajectory diagnostic for whether concentration adapts and carries across boundaries.

`prior_std` is displayed only after fitting. A narrow experimental prior has smaller SD, while a stronger learned model prior has larger kappa, so the two axes move in opposite conceptual directions. Agreement is not guaranteed trial by trial because the model learns only from revealed directions and memory.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(len(pilot_subjects), 1, figsize=(13, 3.8 * len(pilot_subjects)), squeeze=False)
for ax, subject_id in zip(axes[:, 0], pilot_subjects):
    state = state_summaries[subject_id]
    x = state["subject_trial_position"]
    smooth = state["expected_hidden_kappa"].rolling(75, center=True, min_periods=1).mean()
    ax.plot(x, state["expected_hidden_kappa"], color="#9dc3c2", alpha=0.35, linewidth=0.7)
    ax.plot(x, smooth, color="#2b6f77", linewidth=1.8, label="Expected hidden kappa (75-trial mean)")
    boundaries = np.flatnonzero(state["block_id"].to_numpy()[1:] != state["block_id"].to_numpy()[:-1]) + 1
    for boundary in boundaries:
        ax.axvline(boundary, color="#d7dadd", linewidth=0.5, alpha=0.7)
    ax.set(title=f"Participant {subject_id}", xlabel="Chronological trial", ylabel="Expected hidden kappa")
    ax.spines[["top", "right"]].set_visible(False)
    second = ax.twinx()
    second.step(x, state["prior_std_diagnostic_only"], where="post", color="#d1495b", alpha=0.45,
                linewidth=0.8, label="Block prior SD (diagnostic only)")
    second.set_ylabel("Experiment prior SD")
    lines = ax.get_lines()[:2] + second.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", frameon=False)
fig.suptitle("Hidden confidence is continuous across block boundaries")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "hidden_kappa_trajectories.png", dpi=160)
plt.show()
"""
    ),
    md(
        r"""
## Step 11 - First-run decision gate

**Depends on:** all previous outputs.

This run is acceptable as an implementation pilot when:

- every predicted response distribution sums to one;
- every tied MAP maximum receives equal mass after six-decimal posterior rounding, and tie diagnostics are saved;
- the hidden state is carried through every block in sequence;
- NLL improves from the declared initial parameters;
- optimization status and limits are reported rather than hidden;
- observed and predicted distributions are inspected directly;
- fitted sensory concentration generally increases with coherence, or any violation is investigated;
- no claim of model superiority is made from these two exploratory fits.

The next scientific stage is controlled simulation and parameter recovery. Only after that should the model be fitted to all 12 participants with multiple starts and compared against Basic Bayesian and Switching observers.
"""
    ),
    code(
        r"""
decision_rows = []
for subject_id in pilot_subjects:
    result = fit_results[subject_id]
    sensory = result.parameters.sensory_kappas
    decision_rows.append(
        {
            "subject_id": subject_id,
            "prediction_pmfs_normalized": bool(np.allclose(predictions[subject_id].sum(axis=1), 1.0, atol=1e-8)),
            "nll_improved": bool(result.nll <= result.initial_nll),
            "sensory_kappa_increases_with_coherence": bool(np.all(np.diff(sensory) > 0.0)),
            "optimizer_reported_success": result.success,
            "optimizer_status": result.status,
        }
    )
decision_table = pd.DataFrame(decision_rows)
decision_table.to_csv(OUTPUT_DIR / "decision_gate.csv", index=False)
display(decision_table)
print(f"Run outputs are saved in: {OUTPUT_DIR}")
"""
    ),
    md(
        r"""
## Reproducible project commands

The notebook runs no hidden shell commands. All calculations above are ordinary Python cells. From PowerShell, use these documented commands:

```powershell
Set-Location "C:\Users\salma\Backup\Desktop\bayesian modeling\independent_transcript_paper_model"

# Check the helper calculations first
python -m pytest -q

# Open the notebook and choose Run All for the normal pilot
python -m jupyter lab

# Optional fast verification without spending the 30-minute budget
$env:HIERARCHICAL_SMOKE_TEST="1"
python -m jupyter nbconvert --to notebook --execute "notebooks\01_30_minute_hierarchical_fit.ipynb" --output "01_smoke_executed.ipynb" --output-dir "outputs\smoke_test" --ExecutePreprocessor.timeout=600
Remove-Item Env:HIERARCHICAL_SMOKE_TEST

# Optional noninteractive normal run
python -m jupyter nbconvert --to notebook --execute "notebooks\01_30_minute_hierarchical_fit.ipynb" --output "01_pilot_executed.ipynb" --output-dir "outputs\pilot_30min" --ExecutePreprocessor.timeout=2400
```

The same commands and their purpose are recorded in `COMMANDS.md` at the project root.
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
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
