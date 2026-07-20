"""Build the separate CUDA-enabled pilot notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_gpu_hierarchical_fit.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# GPU-accelerated hierarchical prior-confidence fit

This is a separate computational version of the validated CPU pilot. It answers the same research question and uses the same chronological model:

> Can a participant learn a hidden distribution over prior concentration while the prior mean remains fixed at 225 degrees?

Only the heavy numerical calculations change location. Hidden-state learning remains sequential and is computed by the CPU reference code. Batched posterior construction, latent-measurement integration, motor noise, and response scoring run on the NVIDIA GPU.

## Safety rules

1. This notebook never installs or modifies an NVIDIA driver.
2. It skips GPU fitting when the driver or CUDA-enabled PyTorch is unavailable.
3. It first compares CPU and GPU NLL values on the same complete participant sequences.
4. It refuses to fit if numerical equivalence fails.
5. The CPU and GPU notebooks write to different output folders.
"""
    ),
    md(
        r"""
## Step 0 - Imports and project location

**Depends on:** the independent project folder and its tested CPU helper code.

**Produces:** imports only. No model is run and no driver setting is changed.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


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
    GridSpec, HierarchicalObserver, fit_subject, load_and_prepare_data,
    pilot_selection_table, prepare_subject,
)
from hierarchical_confidence.circular import angle_to_bin, wrap_degrees_signed
from hierarchical_confidence.fit import ParameterTransform
from hierarchical_confidence.gpu import (
    TROUBLESHOOTING_STEPS, TorchHierarchicalObserver, cuda_diagnostics,
)
"""
    ),
    md(
        r"""
## Step 1 - Detect the driver and CUDA backend

**Depends on:** `nvidia-smi` and the current notebook's Python environment.

**Process:** perform read-only hardware queries and ask PyTorch whether CUDA is available.

**Produces:** `GPU_READY`. If false, later fitting cells are skipped and troubleshooting steps are displayed. The notebook does not attempt a repair.
"""
    ),
    code(
        r"""
hardware = cuda_diagnostics()
GPU_READY = bool(hardware["gpu_ready"])
display(pd.DataFrame([hardware]).T.rename(columns={0: "detected value"}))

if GPU_READY:
    print("GPU check passed. CUDA fitting may proceed after numerical equivalence checks.")
else:
    print("GPU check failed. No GPU fitting will be attempted.")
    display(pd.DataFrame({"troubleshooting step": TROUBLESHOOTING_STEPS}, index=range(1, len(TROUBLESHOOTING_STEPS) + 1)))
"""
    ),
    md(
        r"""
## Step 2 - Declare the GPU run configuration

**Depends on:** no fitted result.

Normal mode mirrors the CPU pilot: two participants, 72 angle bins, 16 fixed kappa support points, and at most 100 evaluations or 12 minutes per participant. CUDA uses larger batches because the posterior calculations benefit from parallel tensor work.

`float64` is used for the strict validation calculation. Faster `float32` is used for exploratory fitting only after it passes a second declared tolerance.
"""
    ),
    code(
        r"""
SMOKE_TEST = os.getenv("HIERARCHICAL_GPU_SMOKE_TEST", "0") == "1"
DATA_PATH = Path(os.getenv(
    "HIERARCHICAL_DATA_PATH",
    str(Path.home() / "Downloads" / "data01_direction4priors.csv"),
)).expanduser()
RUN_NAME = "gpu_smoke_test" if SMOKE_TEST else "gpu_pilot_30min"
OUTPUT_DIR = ROOT / "outputs" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_CONFIG = {
    "smoke_test": SMOKE_TEST,
    "n_angles": 24 if SMOKE_TEST else 72,
    "n_positive_kappa": 5 if SMOKE_TEST else 15,
    "total_kappa_support_points": 6 if SMOKE_TEST else 16,
    "gpu_batch_size": 256 if SMOKE_TEST else 1024,
    "gpu_fit_dtype": "float32",
    "strict_validation_dtype": "float64",
    "strict_relative_tolerance": 1e-8,
    "fit_dtype_relative_tolerance": 1e-4,
    "max_evaluations_per_participant": 2 if SMOKE_TEST else 100,
    "time_budget_minutes_per_participant": 0.25 if SMOKE_TEST else 12.0,
}
display(pd.DataFrame([RUN_CONFIG]).T.rename(columns={0: "value"}))
print(f"Outputs: {OUTPUT_DIR}")

with (OUTPUT_DIR / "gpu_hardware.json").open("w", encoding="utf-8") as handle:
    json.dump(hardware, handle, indent=2)
with (OUTPUT_DIR / "gpu_run_config.json").open("w", encoding="utf-8") as handle:
    json.dump(RUN_CONFIG, handle, indent=2)
"""
    ),
    md(
        r"""
## Step 3 - Load the same chronological data and pilots

**Depends on:** the raw CSV and CPU data contract.

**Process:** use exactly the same cleaning, circular variables, pilot-selection rule, and complete participant sequences as the CPU notebook.

**Produces:** participants 3 and 5 for the current dataset, unless the observable selection table changes. The selection score never enters model fitting.
"""
    ),
    code(
        r"""
data, data_audit = load_and_prepare_data(DATA_PATH)
selection = pilot_selection_table(data)
pilot_subjects = [int(selection.iloc[0]["subject_id"]), int(selection.iloc[-1]["subject_id"])]
grid = GridSpec(
    n_angles=RUN_CONFIG["n_angles"],
    n_positive_kappa=RUN_CONFIG["n_positive_kappa"],
    kappa_min=0.05,
    kappa_max=50.0,
)
prepared = {subject_id: prepare_subject(data, subject_id, grid) for subject_id in pilot_subjects}
print(f"Pilot participants: {pilot_subjects}")
display(pd.DataFrame([data_audit]).T.rename(columns={0: "value"}))
"""
    ),
    md(
        r"""
## Step 4 - Construct CPU reference and GPU observers

**Depends on:** `GPU_READY` and the prepared sequences.

The observers share the same direction grid, kappa support, hidden-state update, normalized six-decimal tie-aware MAP rule, motor-noise model, lapse model, and missing-response handling. They differ only in where large array operations are executed.
"""
    ),
    code(
        r"""
cpu_observers = {
    subject_id: HierarchicalObserver(prepared[subject_id], grid, batch_size=128)
    for subject_id in pilot_subjects
}
gpu_validation_observers = {}
gpu_fit_observers = {}

if GPU_READY:
    gpu_validation_observers = {
        subject_id: TorchHierarchicalObserver(
            prepared[subject_id], grid, batch_size=RUN_CONFIG["gpu_batch_size"], dtype="float64"
        )
        for subject_id in pilot_subjects
    }
    gpu_fit_observers = {
        subject_id: TorchHierarchicalObserver(
            prepared[subject_id], grid, batch_size=RUN_CONFIG["gpu_batch_size"], dtype="float32"
        )
        for subject_id in pilot_subjects
    }
    print("CPU reference, float64 validation, and float32 fitting observers are ready.")
else:
    print("GPU observer construction skipped.")
"""
    ),
    md(
        r"""
## Step 5 - Mandatory CPU/GPU numerical equivalence gate

**Depends on:** Step 4. This must pass before optimization.

For each full participant sequence, calculate the initial NLL three ways:

1. validated NumPy/SciPy CPU reference;
2. CUDA `float64`, required to agree within relative tolerance `1e-8`;
3. CUDA `float32`, required to agree within relative tolerance `1e-4`.

The looser `float32` tolerance acknowledges reduced numerical precision and the normalized, six-decimal tie-aware MAP operation. Failure skips fitting instead of silently accepting different mathematics.
"""
    ),
    code(
        r"""
equivalence_records = []
GPU_EQUIVALENT = False

if GPU_READY:
    for subject_id in pilot_subjects:
        transform = ParameterTransform(prepared[subject_id].coherence_values)
        parameters = transform.default_parameters()
        cpu_nll = cpu_observers[subject_id].negative_log_likelihood(parameters)
        gpu64_nll = gpu_validation_observers[subject_id].negative_log_likelihood(parameters)
        gpu32_nll = gpu_fit_observers[subject_id].negative_log_likelihood(parameters)
        strict_relative = abs(cpu_nll - gpu64_nll) / max(abs(cpu_nll), 1.0)
        fit_relative = abs(cpu_nll - gpu32_nll) / max(abs(cpu_nll), 1.0)
        equivalence_records.append({
            "subject_id": subject_id,
            "cpu_nll": cpu_nll,
            "gpu_float64_nll": gpu64_nll,
            "gpu_float32_nll": gpu32_nll,
            "float64_relative_difference": strict_relative,
            "float32_relative_difference": fit_relative,
            "float64_passed": strict_relative <= RUN_CONFIG["strict_relative_tolerance"],
            "float32_passed": fit_relative <= RUN_CONFIG["fit_dtype_relative_tolerance"],
        })
    equivalence = pd.DataFrame(equivalence_records)
    GPU_EQUIVALENT = bool(equivalence[["float64_passed", "float32_passed"]].to_numpy().all())
    equivalence.to_csv(OUTPUT_DIR / "cpu_gpu_equivalence.csv", index=False)
    display(equivalence)
    print("Equivalence gate:", "PASSED" if GPU_EQUIVALENT else "FAILED - fitting will be skipped")
else:
    equivalence = pd.DataFrame()
    print("Equivalence check skipped because GPU detection failed.")
"""
    ),
    md(
        r"""
## Step 6 - Benchmark one objective evaluation

**Depends on:** a passed equivalence gate.

**Process:** warm up CUDA once, then time one CPU and one GPU NLL evaluation on the strong dual-attraction pilot. Warm-up time is excluded because CUDA initializes kernels on first use.

**Produces:** an empirical speed ratio for this computer. A ratio above one means GPU is faster. At the coarse 72-angle grid, GPU overhead may limit the gain; larger later grids should benefit more.
"""
    ),
    code(
        r"""
benchmark = pd.DataFrame()
if GPU_READY and GPU_EQUIVALENT:
    subject_id = pilot_subjects[0]
    parameters = ParameterTransform(prepared[subject_id].coherence_values).default_parameters()
    _ = gpu_fit_observers[subject_id].negative_log_likelihood(parameters)  # CUDA warm-up
    gpu_fit_observers[subject_id].synchronize()

    started = perf_counter()
    _ = cpu_observers[subject_id].negative_log_likelihood(parameters)
    cpu_seconds = perf_counter() - started

    started = perf_counter()
    _ = gpu_fit_observers[subject_id].negative_log_likelihood(parameters)
    gpu_fit_observers[subject_id].synchronize()
    gpu_seconds = perf_counter() - started

    benchmark = pd.DataFrame([{
        "subject_id": subject_id,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "cpu_time_divided_by_gpu_time": cpu_seconds / gpu_seconds,
        **gpu_fit_observers[subject_id].memory_record(),
    }])
    benchmark.to_csv(OUTPUT_DIR / "gpu_benchmark.csv", index=False)
    display(benchmark)
else:
    print("Benchmark skipped.")
"""
    ),
    md(
        r"""
## Step 7 - Fit on CUDA within the exploratory budget

**Depends on:** both hardware detection and numerical equivalence.

**Process:** use the same bounded one-start Powell optimizer as the CPU pilot. SciPy proposes six participant-level parameters; each objective calculation is evaluated by the CUDA observer.

**Produces:** independent GPU fit records. These do not overwrite the CPU reference results.
"""
    ),
    code(
        r"""
fit_results = {}
if GPU_READY and GPU_EQUIVALENT:
    for subject_id in pilot_subjects:
        print(f"GPU fitting participant {subject_id} ...")
        result = fit_subject(
            gpu_fit_observers[subject_id],
            max_evaluations=RUN_CONFIG["max_evaluations_per_participant"],
            time_budget_seconds=60.0 * RUN_CONFIG["time_budget_minutes_per_participant"],
        )
        fit_results[subject_id] = result
        result.history.to_csv(
            OUTPUT_DIR / f"participant_{subject_id}_gpu_optimization_history.csv", index=False
        )
        print(
            f"  NLL {result.initial_nll:.2f} -> {result.nll:.2f}; "
            f"{result.evaluations} evaluations in {result.elapsed_seconds / 60.0:.2f} minutes"
        )
        print(f"  {result.status}")

    fit_summary = pd.DataFrame([
        result.summary_record(prepared[subject_id].coherence_values)
        for subject_id, result in fit_results.items()
    ])
    fit_summary["backend"] = "PyTorch CUDA float32"
    fit_summary.to_csv(OUTPUT_DIR / "gpu_fit_summary.csv", index=False)
    display(fit_summary)
else:
    fit_summary = pd.DataFrame()
    print("GPU fitting skipped because a required gate did not pass.")
"""
    ),
    md(
        r"""
## Step 8 - Save predictions and hidden-confidence trajectories

**Depends on:** completed GPU fits.

The fitted GPU observer generates full trial-level response distributions using equal mass across every tied MAP maximum after six-decimal posterior rounding. Hidden-state summaries come from the shared validated sequential calculation. Every response distribution must sum to one before it is saved.
"""
    ),
    code(
        r"""
predictions = {}
state_summaries = {}
if fit_results:
    for subject_id, result in fit_results.items():
        predictions[subject_id] = gpu_fit_observers[subject_id].predict_response_pmfs(result.parameters)
        state_summaries[subject_id] = gpu_fit_observers[subject_id].state_summary(result.parameters.rho)
        assert np.allclose(predictions[subject_id].sum(axis=1), 1.0, atol=2e-6)
        np.savez_compressed(
            OUTPUT_DIR / f"participant_{subject_id}_gpu_trial_predictions.npz",
            response_pmf=predictions[subject_id], theta_degrees=grid.theta_degrees,
        )
        state_summaries[subject_id].to_csv(
            OUTPUT_DIR / f"participant_{subject_id}_gpu_hidden_state.csv", index=False
        )
    print("GPU response predictions normalize and were saved.")
    tie_diagnostics = pd.DataFrame([
        {
            "subject_id": subject_id,
            **cpu_observers[subject_id].tie_diagnostics(fit_results[subject_id].parameters),
        }
        for subject_id in predictions
    ])
    tie_diagnostics.to_csv(OUTPUT_DIR / "gpu_tie_aware_map_diagnostics.csv", index=False)
    display(tie_diagnostics)
else:
    print("Prediction generation skipped because no GPU fit completed.")
"""
    ),
    md(
        r"""
## Step 9 - Inspect optimization and hidden confidence

**Depends on:** Step 8.

These plots are diagnostics, not evidence that the hierarchical model wins. The optimization trace shows whether the declared start improved. The hidden-kappa trajectory shows the learned concentration state carried continuously across blocks.
"""
    ),
    code(
        r"""
if fit_results:
    fig, axes = plt.subplots(2, len(pilot_subjects), figsize=(13, 7), squeeze=False)
    for column, subject_id in enumerate(pilot_subjects):
        history = fit_results[subject_id].history
        axes[0, column].plot(history["evaluation"], history["nll"], color="#2b6f77")
        axes[0, column].set(title=f"Participant {subject_id}: GPU optimization", xlabel="Evaluation", ylabel="NLL")

        state = state_summaries[subject_id]
        smooth = state["expected_hidden_kappa"].rolling(75, center=True, min_periods=1).mean()
        axes[1, column].plot(state["subject_trial_position"], smooth, color="#d1495b")
        boundaries = np.flatnonzero(state["block_id"].to_numpy()[1:] != state["block_id"].to_numpy()[:-1]) + 1
        for boundary in boundaries:
            axes[1, column].axvline(boundary, color="#d7dadd", linewidth=0.45)
        axes[1, column].set(title="Hidden confidence across blocks", xlabel="Chronological trial", ylabel="Expected hidden kappa")
        for ax in axes[:, column]:
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpu_fit_diagnostics.png", dpi=160)
    plt.show()
else:
    print("Diagnostic plots skipped.")
"""
    ),
    md(
        r"""
## Step 10 - Final status and troubleshooting

The GPU result is usable as an exploratory computation only when:

- `driver_detected`, `cuda_available`, and `gpu_ready` are true;
- both CPU/GPU equivalence checks pass;
- response distributions normalize;
- optimizer limits and status are reported;
- CPU and GPU results remain separate for comparison.

If detection fails, do not modify the model or bypass the equivalence gate. Follow the displayed steps, using NVIDIA and PyTorch official installation pages, then rerun smoke mode.
"""
    ),
    code(
        r"""
final_status = {
    "driver_detected": hardware["driver_detected"],
    "cuda_available": hardware["cuda_available"],
    "gpu_ready": GPU_READY,
    "cpu_gpu_equivalence_passed": GPU_EQUIVALENT,
    "participants_fitted": len(fit_results),
    "output_directory": str(OUTPUT_DIR),
}
display(pd.DataFrame([final_status]).T.rename(columns={0: "status"}))

if not GPU_READY:
    display(pd.DataFrame({"troubleshooting step": TROUBLESHOOTING_STEPS}, index=range(1, len(TROUBLESHOOTING_STEPS) + 1)))
elif not GPU_EQUIVALENT:
    print("CUDA works, but numerical equivalence failed. Inspect cpu_gpu_equivalence.csv; do not fit yet.")
else:
    print("GPU hardware and numerical-equivalence gates passed.")
"""
    ),
    md(
        r"""
## Commands

No driver installation command is executed by this notebook. Reusable commands, smoke mode, monitoring, and troubleshooting are documented in `GPU_COMMANDS.md`.

```powershell
Set-Location "C:\Users\salma\Backup\Desktop\bayesian modeling\independent_transcript_paper_model"
python -m pytest -q
python -m jupyter lab
```

For a short end-to-end verification:

```powershell
$env:HIERARCHICAL_GPU_SMOKE_TEST="1"
python -m jupyter nbconvert --to notebook --execute "notebooks\02_gpu_hierarchical_fit.ipynb" --output "02_gpu_smoke_executed.ipynb" --output-dir "outputs\gpu_smoke_test" --ExecutePreprocessor.timeout=900
Remove-Item Env:HIERARCHICAL_GPU_SMOKE_TEST
```
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
