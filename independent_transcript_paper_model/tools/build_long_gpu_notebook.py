"""Build the resumable all-participant CUDA multi-start notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_gpu_long_multistart_fit.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Long GPU multi-start hierarchical fit

## Research purpose

This notebook performs the longer participant-level fit that follows the two-participant pilot. It asks whether a hidden distribution over prior concentration can explain responses while the prior mean remains fixed at 225 degrees.

The default run fits **all 12 participants independently**, using **four reproducible starts per participant**. Each start receives at most **300 objective evaluations or 20 minutes**, so the declared maximum fitting budget is about 16 hours. Actual time can be shorter when the evaluation limit is reached first.

## What is different from the pilot

- Multiple dispersed starting values reduce dependence on one optimizer path.
- Every completed start is checkpointed immediately.
- An interrupted run can resume without repeating completed starts.
- The lowest GPU float32 NLL selects the winning start; CPU/GPU float64 calculations only validate that already-selected candidate.
- Final GPU predictions, hidden-state trajectories, tie-aware MAP diagnostics, and observed-versus-predicted shape checks are saved for every fitted participant.

This remains participant-level hierarchical latent-state modeling. It does not add a population-level distribution over participants.
"""
    ),
    md(
        r"""
## Step 0 - Imports and project location

**Input:** this independent project folder.

**Process:** locate `src/hierarchical_confidence` and import the tested data, model, fitting, GPU, and multi-start helpers.

**Output:** functions only. No data are loaded and no fit is started.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.signal import find_peaks
from scipy.stats import spearmanr


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "hierarchical_confidence").exists():
            return candidate
    raise FileNotFoundError("Could not find independent_transcript_paper_model.")


ROOT = find_project_root(Path.cwd().resolve())
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_confidence import (
    GridSpec,
    HierarchicalObserver,
    fit_subject,
    load_and_prepare_data,
    prepare_subject,
)
from hierarchical_confidence.circular import angle_to_bin
from hierarchical_confidence.fit import ParameterTransform
from hierarchical_confidence.gpu import (
    TROUBLESHOOTING_STEPS,
    TorchHierarchicalObserver,
    cuda_diagnostics,
)
from hierarchical_confidence.multistart import (
    make_multistart_schedule,
    parameters_from_record,
)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
"""
    ),
    md(
        r"""
## Step 1 - Read-only GPU detection

**Depends on:** the active Python environment and installed NVIDIA driver.

**Process:** query `nvidia-smi`, CUDA-enabled PyTorch, and a small allocation test. No driver or package is changed.

**Output:** `GPU_READY`. Fitting is blocked if this check fails.
"""
    ),
    code(
        r"""
hardware = cuda_diagnostics()
GPU_READY = bool(hardware["gpu_ready"])
display(pd.DataFrame([hardware]).T.rename(columns={0: "detected value"}))

if GPU_READY:
    print("GPU detection passed.")
else:
    print("GPU detection failed. No fitting will be attempted.")
    display(
        pd.DataFrame(
            {"troubleshooting step": TROUBLESHOOTING_STEPS},
            index=range(1, len(TROUBLESHOOTING_STEPS) + 1),
        )
    )
"""
    ),
    md(
        r"""
## Step 2 - Load chronological data and declare the run

**Depends on:** the raw CSV only.

**Process:** apply the existing validated cleaning and circular-variable derivations, then select all participant IDs unless `HIERARCHICAL_LONG_SUBJECTS` explicitly supplies a comma-separated subset.

**Output:** one immutable run configuration. Reusing an output name with incompatible settings raises an error instead of mixing results.

### Default long-run budget

- 12 participants;
- 4 starts per participant;
- 300 evaluations or 20 minutes per start;
- 72 direction bins and 16 fixed hidden-kappa support points;
- maximum fitting budget: `12 × 4 × 20 minutes = 16 hours`.

The environment variables shown in `GPU_COMMANDS.md` can change these settings without editing notebook cells.
"""
    ),
    code(
        r"""
SMOKE_TEST = os.getenv("HIERARCHICAL_LONG_SMOKE_TEST", "0") == "1"
RESUME = os.getenv("HIERARCHICAL_LONG_RESUME", "1") == "1"
DATA_PATH = Path(
    os.getenv(
        "HIERARCHICAL_DATA_PATH",
        str(Path.home() / "Downloads" / "data01_direction4priors.csv"),
    )
).expanduser()
default_run_name = "gpu_long_multistart_smoke" if SMOKE_TEST else "gpu_long_multistart"
RUN_NAME = os.getenv("HIERARCHICAL_LONG_RUN_NAME", default_run_name)
OUTPUT_DIR = ROOT / "outputs" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data, data_audit = load_and_prepare_data(DATA_PATH)
available_subjects = sorted(int(value) for value in data["subject_id"].unique())
requested_subjects = os.getenv("HIERARCHICAL_LONG_SUBJECTS", "").strip()
if requested_subjects:
    participant_ids = [int(value.strip()) for value in requested_subjects.split(",")]
    missing = sorted(set(participant_ids).difference(available_subjects))
    if missing:
        raise ValueError(f"Requested participant IDs are absent from the data: {missing}")
elif SMOKE_TEST:
    participant_ids = available_subjects[:2]
else:
    participant_ids = available_subjects

n_starts = 2 if SMOKE_TEST else int(os.getenv("HIERARCHICAL_LONG_N_STARTS", "4"))
max_evaluations = 2 if SMOKE_TEST else int(
    os.getenv("HIERARCHICAL_LONG_MAX_EVALUATIONS", "300")
)
minutes_per_start = 0.25 if SMOKE_TEST else float(
    os.getenv("HIERARCHICAL_LONG_MINUTES_PER_START", "20")
)
if len(participant_ids) != len(set(participant_ids)):
    raise ValueError("Participant IDs must be unique.")
if n_starts < 2:
    raise ValueError("A multi-start run requires at least two starts per participant.")
if max_evaluations < 1 or minutes_per_start <= 0.0:
    raise ValueError("Evaluation and time budgets must be positive.")

RUN_CONFIG = {
    "smoke_test": SMOKE_TEST,
    "resume_completed_starts": RESUME,
    "data_path": str(DATA_PATH.resolve()),
    "participant_ids": participant_ids,
    "n_participants": len(participant_ids),
    "n_starts_per_participant": n_starts,
    "base_seed": 20260720,
    "n_angles": 24 if SMOKE_TEST else 72,
    "n_positive_kappa": 5 if SMOKE_TEST else 15,
    "total_kappa_support_points": 6 if SMOKE_TEST else 16,
    "kappa_min": 0.05,
    "kappa_max": 50.0,
    "gpu_batch_size": 256 if SMOKE_TEST else 1024,
    "gpu_fit_dtype": "float32",
    "candidate_selection": "lowest GPU float32 NLL",
    "selected_candidate_validation": "CPU and GPU float64",
    "strict_relative_tolerance": 1e-8,
    "fit_dtype_relative_tolerance": 1e-4,
    "max_evaluations_per_start": max_evaluations,
    "time_budget_minutes_per_start": minutes_per_start,
    "maximum_declared_fit_hours": len(participant_ids) * n_starts * minutes_per_start / 60.0,
    "optimizer": "Powell with bounded parameters",
    "posterior_readout": "normalized six-decimal tie-aware MAP",
    "fixed_prior_mean_degrees": 225.0,
}

config_path = OUTPUT_DIR / "long_run_config.json"
immutable_keys = [key for key in RUN_CONFIG if key != "resume_completed_starts"]
if config_path.exists():
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (previous.get(key), RUN_CONFIG.get(key))
        for key in immutable_keys
        if previous.get(key) != RUN_CONFIG.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "This output name already contains a different run configuration. "
            f"Choose a new HIERARCHICAL_LONG_RUN_NAME. Differences: {mismatches}"
        )

write_json_atomic(config_path, RUN_CONFIG)
write_json_atomic(OUTPUT_DIR / "gpu_hardware.json", hardware)
write_json_atomic(OUTPUT_DIR / "data_audit.json", data_audit)
display(pd.DataFrame([RUN_CONFIG]).T.rename(columns={0: "value"}))
print(f"Outputs: {OUTPUT_DIR}")
print(f"Maximum declared fitting budget: {RUN_CONFIG['maximum_declared_fit_hours']:.2f} hours")
"""
    ),
    md(
        r"""
## Step 3 - Prepare participant sequences and observers

**Depends on:** Step 2 and successful GPU detection.

**Process:** preserve every participant's complete chronological trial sequence. The hidden state is not reset at block boundaries. CPU float64 observers are retained only for numerical equivalence and validation of the already-selected candidate. Optimization, candidate selection, and final response prediction use CUDA float32 observers.

**Output:** one prepared sequence and observer set per participant.
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
prepared = {
    subject_id: prepare_subject(data, subject_id, grid)
    for subject_id in participant_ids
}
cpu_observers = {
    subject_id: HierarchicalObserver(prepared[subject_id], grid, batch_size=128)
    for subject_id in participant_ids
}
gpu_validation_observers = {}
gpu_fit_observers = {}

if GPU_READY:
    gpu_validation_observers = {
        subject_id: TorchHierarchicalObserver(
            prepared[subject_id],
            grid,
            batch_size=RUN_CONFIG["gpu_batch_size"],
            dtype="float64",
        )
        for subject_id in participant_ids
    }
    gpu_fit_observers = {
        subject_id: TorchHierarchicalObserver(
            prepared[subject_id],
            grid,
            batch_size=RUN_CONFIG["gpu_batch_size"],
            dtype="float32",
        )
        for subject_id in participant_ids
    }

sequence_summary = pd.DataFrame(
    [
        {
            "subject_id": subject_id,
            "chronological_trials": subject.n_trials,
            "valid_responses_scored": int(subject.response_valid.sum()),
            "blocks_carried_across": int(pd.Series(subject.block_ids).nunique()),
        }
        for subject_id, subject in prepared.items()
    ]
)
display(sequence_summary)
"""
    ),
    md(
        r"""
## Step 4 - Mandatory numerical-equivalence gate

**Depends on:** Step 3.

**Process:** evaluate the declared default parameters on every selected participant using CPU float64, GPU float64, and GPU float32. GPU float64 must agree within `1e-8` relative error and fitting precision within `1e-4`.

**Output:** `GPU_EQUIVALENT`. Multi-start fitting cannot begin unless every participant passes both checks.
"""
    ),
    code(
        r"""
equivalence_records = []
GPU_EQUIVALENT = False
GPU_RUNTIME_ERROR = ""

if GPU_READY:
    for subject_id in participant_ids:
        try:
            parameters = ParameterTransform(
                prepared[subject_id].coherence_values
            ).default_parameters()
            cpu_nll = cpu_observers[subject_id].negative_log_likelihood(parameters)
            gpu64_nll = gpu_validation_observers[subject_id].negative_log_likelihood(parameters)
            gpu32_nll = gpu_fit_observers[subject_id].negative_log_likelihood(parameters)
        except Exception as error:
            GPU_RUNTIME_ERROR = (
                f"Equivalence calculation failed for participant {subject_id}: "
                f"{type(error).__name__}: {error}"
            )
            print(GPU_RUNTIME_ERROR)
            break
        strict_relative = abs(cpu_nll - gpu64_nll) / max(abs(cpu_nll), 1.0)
        fit_relative = abs(cpu_nll - gpu32_nll) / max(abs(cpu_nll), 1.0)
        equivalence_records.append(
            {
                "subject_id": subject_id,
                "cpu_float64_nll": cpu_nll,
                "gpu_float64_nll": gpu64_nll,
                "gpu_float32_nll": gpu32_nll,
                "float64_relative_difference": strict_relative,
                "float32_relative_difference": fit_relative,
                "float64_passed": strict_relative <= RUN_CONFIG["strict_relative_tolerance"],
                "float32_passed": fit_relative <= RUN_CONFIG["fit_dtype_relative_tolerance"],
            }
        )

equivalence = pd.DataFrame(equivalence_records)
if not equivalence.empty:
    write_csv_atomic(OUTPUT_DIR / "cpu_gpu_equivalence_all_participants.csv", equivalence)
    GPU_EQUIVALENT = bool(
        len(equivalence) == len(participant_ids)
        and equivalence[["float64_passed", "float32_passed"]].to_numpy().all()
    )
display(equivalence)
print("Equivalence gate:", "PASSED" if GPU_EQUIVALENT else "FAILED")
"""
    ),
    md(
        r"""
## Step 5 - Generate and save the multi-start schedule

**Depends on:** participant IDs and coherence levels, not participant responses.

**Process:** start 0 uses the declared pilot defaults. Remaining starts use a reproducible participant-specific Latin-hypercube design over broad transformed parameter ranges. Initial sensory kappas are ordered from low to high coherence, but the optimizer itself does not impose that ordering.

**Output:** `multistart_schedule.csv`, containing every initial value and seed before fitting begins.
"""
    ),
    code(
        r"""
coherence_values = np.sort(data["motion_coherence"].unique().astype(float))
schedule = make_multistart_schedule(
    participant_ids,
    coherence_values,
    n_starts=RUN_CONFIG["n_starts_per_participant"],
    base_seed=RUN_CONFIG["base_seed"],
)
write_csv_atomic(OUTPUT_DIR / "multistart_schedule.csv", schedule)
display(schedule.head(max(8, RUN_CONFIG["n_starts_per_participant"])))
print(f"Scheduled starts: {len(schedule)}")
"""
    ),
    md(
        r"""
## Step 6 - Run or resume checkpointed GPU optimization

**Depends on:** passed hardware and equivalence gates plus the saved start schedule.

**Process:** fit one participant/start at a time. Each objective is computed by the CUDA float32 observer. After a start finishes, its full evaluation history and fitted record are written atomically before the next start begins.

**Checkpoint structure:** `starts/subject_XX/start_YY_result.json` and `start_YY_history.csv`.

When resume mode is active, an existing completed result is skipped. If the notebook or computer stops, rerun the same command and output name; at most the currently active start is lost. `progress.json` can be inspected from another PowerShell window while no browser is open.
"""
    ),
    code(
        r"""
STARTS_DIR = OUTPUT_DIR / "starts"
STARTS_DIR.mkdir(parents=True, exist_ok=True)


def result_path(subject_id: int, start_id: int) -> Path:
    return STARTS_DIR / f"subject_{subject_id:02d}" / f"start_{start_id:02d}_result.json"


def history_path(subject_id: int, start_id: int) -> Path:
    return STARTS_DIR / f"subject_{subject_id:02d}" / f"start_{start_id:02d}_history.csv"


def completed_records() -> list[dict]:
    records = []
    for path in sorted(STARTS_DIR.glob("subject_*/start_*_result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("checkpoint_state") == "completed":
            records.append(record)
    return records


expected_starts = len(schedule)
run_started = perf_counter()
stop_after_error = False

if GPU_READY and GPU_EQUIVALENT:
    for _, start_row in schedule.iterrows():
        subject_id = int(start_row["subject_id"])
        start_id = int(start_row["start_id"])
        output_result = result_path(subject_id, start_id)
        output_history = history_path(subject_id, start_id)
        output_result.parent.mkdir(parents=True, exist_ok=True)

        if output_result.exists() and RESUME:
            saved = json.loads(output_result.read_text(encoding="utf-8"))
            if saved.get("checkpoint_state") == "completed":
                print(f"Skipping completed participant {subject_id}, start {start_id}.", flush=True)
                continue
        elif output_result.exists():
            raise RuntimeError(
                f"Checkpoint already exists at {output_result}. Enable resume or choose a new run name."
            )

        initial_parameters = parameters_from_record(
            start_row, coherence_values, prefix="initial_"
        )
        print(
            f"Fitting participant {subject_id}, start {start_id + 1}/"
            f"{RUN_CONFIG['n_starts_per_participant']} ...",
            flush=True,
        )
        try:
            result = fit_subject(
                gpu_fit_observers[subject_id],
                max_evaluations=RUN_CONFIG["max_evaluations_per_start"],
                time_budget_seconds=60.0 * RUN_CONFIG["time_budget_minutes_per_start"],
                initial_parameters=initial_parameters,
            )
        except Exception as error:
            GPU_RUNTIME_ERROR = (
                f"GPU fit failed for participant {subject_id}, start {start_id}: "
                f"{type(error).__name__}: {error}"
            )
            write_json_atomic(
                output_result.with_name(output_result.stem.replace("_result", "_failure") + ".json"),
                {
                    "checkpoint_state": "failed",
                    "subject_id": subject_id,
                    "start_id": start_id,
                    "error": GPU_RUNTIME_ERROR,
                    "recorded_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(GPU_RUNTIME_ERROR, flush=True)
            stop_after_error = True
            break

        history = result.history.copy()
        history.insert(0, "start_id", start_id)
        history.insert(0, "subject_id", subject_id)
        write_csv_atomic(output_history, history)

        record = {
            "checkpoint_state": "completed",
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "start_id": start_id,
            "start_source": start_row["start_source"],
            "subject_seed": int(start_row["subject_seed"]),
            **result.summary_record(prepared[subject_id].coherence_values),
        }
        for key, value in start_row.items():
            if str(key).startswith("initial_"):
                record[str(key)] = value
        write_json_atomic(output_result, record)

        completed = len(completed_records())
        write_json_atomic(
            OUTPUT_DIR / "progress.json",
            {
                "completed_starts": completed,
                "expected_starts": expected_starts,
                "percent_complete": 100.0 * completed / expected_starts,
                "last_completed_subject": subject_id,
                "last_completed_start": start_id,
                "elapsed_hours_this_execution": (perf_counter() - run_started) / 3600.0,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(
            f"  NLL {result.initial_nll:.2f} -> {result.nll:.2f}; "
            f"{result.evaluations} evaluations in {result.elapsed_seconds / 60.0:.2f} minutes; "
            f"checkpoint {completed}/{expected_starts}",
            flush=True,
        )
else:
    print("Optimization skipped because a required gate did not pass.")

all_start_summary = pd.DataFrame(completed_records())
if not all_start_summary.empty:
    all_start_summary = all_start_summary.sort_values(["subject_id", "start_id"]).reset_index(drop=True)
    write_csv_atomic(OUTPUT_DIR / "all_start_gpu_summary.csv", all_start_summary)
STARTS_COMPLETE = bool(len(all_start_summary) == expected_starts)
print(f"Completed starts: {len(all_start_summary)}/{expected_starts}")
display(all_start_summary)
"""
    ),
    md(
        r"""
## Step 7 - Select on GPU and validate the selected candidates

**Depends on:** all scheduled starts completing.

**Process:** within each participant, select the completed start with the lowest CUDA float32 NLL. Only that already-selected candidate is evaluated with CPU float64 and GPU float64 to verify numerical agreement. The validation calculations never choose the winning start.

**Output:** `best_multistart_fit.csv` and `selected_candidate_validation.csv`.
"""
    ),
    code(
        r"""
best_fits = pd.DataFrame()
selected_candidate_validation = pd.DataFrame()
SELECTION_VALIDATED = False

if STARTS_COMPLETE:
    best_fits = (
        all_start_summary.sort_values(["subject_id", "nll"])
        .groupby("subject_id", as_index=False, sort=True)
        .first()
    )
    validation_records = []
    for _, row in best_fits.iterrows():
        subject_id = int(row["subject_id"])
        parameters = parameters_from_record(row, coherence_values)
        cpu_nll = cpu_observers[subject_id].negative_log_likelihood(parameters)
        gpu64_nll = gpu_validation_observers[subject_id].negative_log_likelihood(parameters)
        gpu32_nll = float(row["nll"])
        float64_relative = abs(cpu_nll - gpu64_nll) / max(abs(cpu_nll), 1.0)
        fit_relative = abs(cpu_nll - gpu32_nll) / max(abs(cpu_nll), 1.0)
        validation_records.append(
            {
                "subject_id": subject_id,
                "selected_start_id": int(row["start_id"]),
                "selected_gpu_float32_nll": gpu32_nll,
                "cpu_float64_nll": cpu_nll,
                "gpu_float64_nll": gpu64_nll,
                "float64_relative_difference": float64_relative,
                "float32_relative_difference": fit_relative,
                "float64_passed": float64_relative
                <= RUN_CONFIG["strict_relative_tolerance"],
                "float32_passed": fit_relative
                <= RUN_CONFIG["fit_dtype_relative_tolerance"],
            }
        )
    selected_candidate_validation = pd.DataFrame(validation_records)
    SELECTION_VALIDATED = bool(
        selected_candidate_validation[["float64_passed", "float32_passed"]]
        .to_numpy()
        .all()
    )
    write_csv_atomic(OUTPUT_DIR / "best_multistart_fit.csv", best_fits)
    write_csv_atomic(
        OUTPUT_DIR / "selected_candidate_validation.csv",
        selected_candidate_validation,
    )
    display(best_fits)
    display(selected_candidate_validation)
    print("Selected-candidate validation:", "PASSED" if SELECTION_VALIDATED else "FAILED")
else:
    print("Candidate selection waits until every scheduled start has a checkpoint.")
"""
    ),
    md(
        r"""
## Step 8 - Diagnose optimization stability

**Depends on:** Step 7.

**Process:** compare the best and second-best starts, inspect whether the chosen start reported convergence, check normalized distance from parameter bounds, and verify increasing fitted sensory precision with coherence.

**Output:** `multistart_optimization_diagnostics.csv` and an NLL-by-start plot.

Multiple starts reduce local-start dependence; they do not create posterior uncertainty intervals. A small best-versus-second gap is reassuring only when the parameter estimates are also similar.
"""
    ),
    code(
        r"""
optimization_diagnostics = pd.DataFrame()
if not best_fits.empty:
    records = []
    for subject_id, group in all_start_summary.groupby("subject_id", sort=True):
        ordered = group.sort_values("nll").reset_index(drop=True)
        best = ordered.iloc[0]
        parameters = parameters_from_record(best, coherence_values)
        transform = ParameterTransform(coherence_values)
        raw = transform.encode(parameters)
        bounds = np.asarray(transform.bounds, dtype=float)
        normalized_margin = np.minimum(
            (raw - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]),
            (bounds[:, 1] - raw) / (bounds[:, 1] - bounds[:, 0]),
        )
        second_nll = float(ordered.iloc[1]["nll"])
        records.append(
            {
                "subject_id": int(subject_id),
                "selected_start_id": int(best["start_id"]),
                "best_gpu_float32_nll": float(best["nll"]),
                "second_best_gpu_float32_nll": second_nll,
                "second_minus_best_nll": second_nll - float(best["nll"]),
                "across_start_nll_range": float(
                    ordered["nll"].max() - ordered["nll"].min()
                ),
                "selected_optimizer_success": bool(best["success"]),
                "selected_evaluations": int(best["evaluations"]),
                "minimum_normalized_parameter_bound_margin": float(normalized_margin.min()),
                "selected_parameter_near_bound": bool(normalized_margin.min() < 0.01),
                "sensory_kappa_increases_with_coherence": bool(
                    np.all(np.diff(parameters.sensory_kappas) > 0.0)
                ),
            }
        )
    optimization_diagnostics = pd.DataFrame(records)
    write_csv_atomic(
        OUTPUT_DIR / "multistart_optimization_diagnostics.csv",
        optimization_diagnostics,
    )
    display(optimization_diagnostics)

    fig, ax = plt.subplots(figsize=(11, 5))
    for start_id, group in all_start_summary.groupby("start_id"):
        ax.scatter(
            group["subject_id"],
            group["nll"],
            label=f"Start {int(start_id)}",
            s=32,
            alpha=0.8,
        )
    ax.set(
        xlabel="Participant",
        ylabel="GPU float32 NLL",
        title="GPU multi-start candidate comparison",
        xticks=participant_ids,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multistart_nll_diagnostics.png", dpi=160)
    plt.show()
"""
    ),
    md(
        r"""
## Step 9 - Generate final GPU predictions and hidden states

**Depends on:** one validated selected candidate for every participant.

**Process:** rerun each full chronological sequence with the selected parameters, retain the trial-level response PMF, and compute the pre-feedback hidden-kappa trajectory. Tie-aware diagnostics are calculated from the same selected parameters.

**Output:** one compressed prediction file and hidden-state CSV per participant, plus normalization, likelihood, and tie checks.
"""
    ),
    code(
        r"""
prediction_checks = []
tie_records = []
hidden_relation_records = []

if len(best_fits) == len(participant_ids):
    for _, row in best_fits.iterrows():
        subject_id = int(row["subject_id"])
        parameters = parameters_from_record(row, coherence_values)
        prediction_path = OUTPUT_DIR / f"participant_{subject_id}_gpu_trial_predictions.npz"
        hidden_path = OUTPUT_DIR / f"participant_{subject_id}_hidden_state.csv"

        predictions = gpu_fit_observers[subject_id].predict_response_pmfs(parameters)
        if not np.allclose(predictions.sum(axis=1), 1.0, atol=2e-6):
            raise RuntimeError(f"Participant {subject_id} predictions do not normalize.")
        temporary_prediction = prediction_path.with_name(prediction_path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary_prediction,
            response_pmf=predictions,
            theta_degrees=grid.theta_degrees,
        )
        temporary_prediction.replace(prediction_path)

        hidden = gpu_fit_observers[subject_id].state_summary(parameters.rho)
        write_csv_atomic(hidden_path, hidden)

        subject = prepared[subject_id]
        valid_indices = np.flatnonzero(subject.response_valid)
        probabilities = predictions[valid_indices, subject.response_bins[valid_indices]]
        prediction_nll = -np.log(np.maximum(probabilities, np.finfo(float).tiny)).sum()
        selected_gpu_nll = float(row["nll"])
        prediction_checks.append(
            {
                "subject_id": subject_id,
                "all_pmfs_normalized": True,
                "prediction_readback_nll": prediction_nll,
                "selected_gpu_fit_nll": selected_gpu_nll,
                "absolute_nll_difference": abs(prediction_nll - selected_gpu_nll),
                "relative_nll_difference": abs(prediction_nll - selected_gpu_nll)
                / max(abs(selected_gpu_nll), 1.0),
            }
        )
        tie_records.append(
            {"subject_id": subject_id, **cpu_observers[subject_id].tie_diagnostics(parameters)}
        )

        block_means = hidden.groupby(
            ["block_id", "prior_std_diagnostic_only"], as_index=False
        )["expected_hidden_kappa"].mean()
        hidden_relation_records.append(
            {
                "subject_id": subject_id,
                "trial_level_spearman_prior_sd_vs_expected_kappa": float(
                    spearmanr(
                        hidden["prior_std_diagnostic_only"],
                        hidden["expected_hidden_kappa"],
                    ).statistic
                ),
                "block_mean_spearman_prior_sd_vs_expected_kappa": float(
                    spearmanr(
                        block_means["prior_std_diagnostic_only"],
                        block_means["expected_hidden_kappa"],
                    ).statistic
                ),
            }
        )
        print(f"Saved selected predictions and hidden state for participant {subject_id}.")

prediction_checks = pd.DataFrame(prediction_checks)
tie_diagnostics = pd.DataFrame(tie_records)
hidden_relation = pd.DataFrame(hidden_relation_records)
if not prediction_checks.empty:
    write_csv_atomic(OUTPUT_DIR / "prediction_checks.csv", prediction_checks)
    write_csv_atomic(OUTPUT_DIR / "tie_aware_map_diagnostics.csv", tie_diagnostics)
    write_csv_atomic(OUTPUT_DIR / "hidden_kappa_relation_to_prior_sd.csv", hidden_relation)
display(prediction_checks)
display(tie_diagnostics)
display(hidden_relation)
"""
    ),
    md(
        r"""
## Step 10 - Compare observed and predicted distribution shapes

**Depends on:** saved final predictions.

**Process:** align responses to the true stimulus, then compare the observed histogram with the mean predicted PMF within each participant × prior-SD × coherence cell. Total-variation distance is zero for identical distributions and one for non-overlapping distributions.

The multimodality screen is explicitly exploratory. It counts trial PMFs with at least two peaks separated by 30 degrees and repeats the count at 5% and 10% peak-prominence thresholds.

**Output:** shape-check and multimodality CSV files. These evaluate in-sample fit; they are not held-out predictive accuracy.
"""
    ),
    code(
        r"""
shape_records = []
multimodality_records = []


def has_separated_peaks(peaks: np.ndarray, n_bins: int, minimum_bins: int) -> bool:
    for index, first in enumerate(peaks):
        for second in peaks[index + 1 :]:
            distance = abs(int(first) - int(second))
            if min(distance, n_bins - distance) >= minimum_bins:
                return True
    return False


if len(best_fits) == len(participant_ids):
    for subject_id in participant_ids:
        rows = data.loc[data["subject_id"] == subject_id].copy().reset_index(drop=True)
        with np.load(
            OUTPUT_DIR / f"participant_{subject_id}_gpu_trial_predictions.npz"
        ) as saved:
            predictions = saved["response_pmf"].astype(float)
        valid = rows["response_valid"].to_numpy(dtype=bool)
        stimulus_bins = angle_to_bin(rows["motion_direction"].to_numpy(), grid.n_angles)
        response_bins = np.full(len(rows), -1, dtype=int)
        response_bins[valid] = angle_to_bin(
            rows.loc[valid, "response_angle"].to_numpy(), grid.n_angles
        )
        aligned_predictions = np.stack(
            [np.roll(predictions[index], -stimulus_bins[index]) for index in range(len(rows))]
        )

        valid_rows = rows.loc[valid]
        for (prior_sd, coherence), indices in valid_rows.groupby(
            ["prior_std", "motion_coherence"]
        ).groups.items():
            indices = np.asarray(list(indices), dtype=int)
            observed_bins = (
                response_bins[indices] - stimulus_bins[indices]
            ) % grid.n_angles
            observed = np.bincount(observed_bins, minlength=grid.n_angles).astype(float)
            observed /= observed.sum()
            predicted = aligned_predictions[indices].mean(axis=0)
            shape_records.append(
                {
                    "subject_id": subject_id,
                    "prior_sd": float(prior_sd),
                    "coherence": float(coherence),
                    "trials": len(indices),
                    "total_variation_distance": float(
                        0.5 * np.abs(observed - predicted).sum()
                    ),
                }
            )

        minimum_bins = max(1, int(round(30.0 / (360.0 / grid.n_angles))))
        for prominence_ratio in (0.05, 0.10):
            multimodal = 0
            for probability in predictions:
                extended = np.tile(probability, 3)
                peaks, _ = find_peaks(
                    extended,
                    prominence=prominence_ratio * probability.max(),
                    distance=minimum_bins,
                )
                central = peaks[
                    (peaks >= grid.n_angles) & (peaks < 2 * grid.n_angles)
                ] - grid.n_angles
                if len(central) >= 2 and has_separated_peaks(
                    central, grid.n_angles, minimum_bins
                ):
                    multimodal += 1
            multimodality_records.append(
                {
                    "subject_id": subject_id,
                    "prominence_fraction_of_maximum": prominence_ratio,
                    "minimum_peak_separation_degrees": 30.0,
                    "candidate_multimodal_trials": multimodal,
                    "total_trials": len(predictions),
                    "candidate_multimodal_fraction": multimodal / len(predictions),
                }
            )

shape_checks = pd.DataFrame(shape_records)
multimodality_screen = pd.DataFrame(multimodality_records)
if not shape_checks.empty:
    write_csv_atomic(OUTPUT_DIR / "observed_predicted_shape_checks.csv", shape_checks)
    write_csv_atomic(OUTPUT_DIR / "predicted_multimodality_screen.csv", multimodality_screen)
display(shape_checks)
display(multimodality_screen)
"""
    ),
    md(
        r"""
## Step 11 - Final completion record and interpretation gate

**Depends on:** all preceding steps.

The run is computationally complete only when:

- every scheduled start has a checkpoint;
- every participant has one GPU-selected candidate that passes the numerical validation gate;
- all trial-level prediction PMFs normalize;
- prediction readback likelihood agrees with the selected likelihood;
- tie-aware and observed-versus-predicted diagnostics are saved.

Completion does not by itself establish that hidden kappa exists in human inference. Parameter recovery and formal comparison with the Basic Bayesian and Switching observers remain required before that claim.
"""
    ),
    code(
        r"""
prediction_passed = bool(
    not prediction_checks.empty
    and prediction_checks["all_pmfs_normalized"].all()
    and (prediction_checks["relative_nll_difference"] <= 1e-4).all()
)
status = {
    "driver_detected": bool(hardware["driver_detected"]),
    "cuda_available": bool(hardware["cuda_available"]),
    "allocation_test_passed": bool(hardware["allocation_test_passed"]),
    "initial_equivalence_passed": GPU_EQUIVALENT,
    "selected_candidate_validation_passed": SELECTION_VALIDATED,
    "completed_starts": len(all_start_summary),
    "expected_starts": expected_starts,
    "all_starts_complete": STARTS_COMPLETE,
    "participants_selected": len(best_fits),
    "participants_expected": len(participant_ids),
    "prediction_checks_passed": prediction_passed,
    "runtime_error": GPU_RUNTIME_ERROR,
    "run_complete": bool(
        GPU_READY
        and GPU_EQUIVALENT
        and STARTS_COMPLETE
        and SELECTION_VALIDATED
        and len(best_fits) == len(participant_ids)
        and prediction_passed
    ),
    "output_directory": str(OUTPUT_DIR),
    "finished_at": datetime.now().isoformat(timespec="seconds"),
}
write_json_atomic(OUTPUT_DIR / "long_run_final_status.json", status)
display(pd.DataFrame([status]).T.rename(columns={0: "value"}))

if status["run_complete"]:
    print("Long GPU multi-start run completed and passed its computational gates.")
else:
    print("Run is incomplete or a gate failed. Inspect the status and resume checkpoints.")
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
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
