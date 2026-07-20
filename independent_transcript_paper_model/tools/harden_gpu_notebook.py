"""Add graceful runtime-error handling to the generated GPU notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "02_gpu_hierarchical_fit.ipynb"
notebook = nbformat.read(PATH, as_version=4)


def replace_code_cell(marker: str, source: str) -> None:
    for cell in notebook.cells:
        if cell.cell_type == "code" and marker in cell.source:
            cell.source = source.strip()
            return
    raise ValueError(f"Could not find GPU notebook cell containing: {marker}")


replace_code_cell(
    "cpu_observers = {",
    r'''
cpu_observers = {
    subject_id: HierarchicalObserver(prepared[subject_id], grid, batch_size=128)
    for subject_id in pilot_subjects
}
gpu_validation_observers = {}
gpu_fit_observers = {}
GPU_RUNTIME_ERROR = ""

if GPU_READY:
    try:
        gpu_validation_observers = {
            subject_id: TorchHierarchicalObserver(
                prepared[subject_id], grid,
                batch_size=RUN_CONFIG["gpu_batch_size"], dtype="float64"
            )
            for subject_id in pilot_subjects
        }
        gpu_fit_observers = {
            subject_id: TorchHierarchicalObserver(
                prepared[subject_id], grid,
                batch_size=RUN_CONFIG["gpu_batch_size"], dtype="float32"
            )
            for subject_id in pilot_subjects
        }
        print("CPU reference, float64 validation, and float32 fitting observers are ready.")
    except Exception as error:
        GPU_RUNTIME_ERROR = f"GPU observer construction failed: {type(error).__name__}: {error}"
        GPU_READY = False
        print(GPU_RUNTIME_ERROR)
        print("GPU fitting is disabled; no repair was attempted.")
        display(pd.DataFrame({"troubleshooting step": TROUBLESHOOTING_STEPS},
                             index=range(1, len(TROUBLESHOOTING_STEPS) + 1)))
else:
    print("GPU observer construction skipped.")
''',
)

replace_code_cell(
    "equivalence_records = []",
    r'''
equivalence_records = []
GPU_EQUIVALENT = False

if GPU_READY:
    try:
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
        GPU_EQUIVALENT = bool(
            equivalence[["float64_passed", "float32_passed"]].to_numpy().all()
        )
        equivalence.to_csv(OUTPUT_DIR / "cpu_gpu_equivalence.csv", index=False)
        display(equivalence)
        print("Equivalence gate:",
              "PASSED" if GPU_EQUIVALENT else "FAILED - fitting will be skipped")
    except Exception as error:
        GPU_RUNTIME_ERROR = f"GPU equivalence calculation failed: {type(error).__name__}: {error}"
        GPU_READY = False
        GPU_EQUIVALENT = False
        equivalence = pd.DataFrame(equivalence_records)
        print(GPU_RUNTIME_ERROR)
        print("GPU fitting is disabled; no repair was attempted.")
        display(pd.DataFrame({"troubleshooting step": TROUBLESHOOTING_STEPS},
                             index=range(1, len(TROUBLESHOOTING_STEPS) + 1)))
else:
    equivalence = pd.DataFrame()
    print("Equivalence check skipped because GPU detection or allocation failed.")
''',
)

replace_code_cell(
    "fit_results = {}",
    r'''
fit_results = {}
if GPU_READY and GPU_EQUIVALENT:
    for subject_id in pilot_subjects:
        print(f"GPU fitting participant {subject_id} ...")
        try:
            result = fit_subject(
                gpu_fit_observers[subject_id],
                max_evaluations=RUN_CONFIG["max_evaluations_per_participant"],
                time_budget_seconds=60.0 * RUN_CONFIG["time_budget_minutes_per_participant"],
            )
        except Exception as error:
            GPU_RUNTIME_ERROR = f"GPU fit failed: {type(error).__name__}: {error}"
            GPU_READY = False
            print(GPU_RUNTIME_ERROR)
            print("Remaining GPU fits are skipped; no repair was attempted.")
            break
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
    if not fit_summary.empty:
        fit_summary["backend"] = "PyTorch CUDA float32"
        fit_summary.to_csv(OUTPUT_DIR / "gpu_fit_summary.csv", index=False)
        display(fit_summary)
else:
    fit_summary = pd.DataFrame()
    print("GPU fitting skipped because a required gate did not pass.")
''',
)

replace_code_cell(
    "final_status = {",
    r'''
final_status = {
    "driver_detected": hardware["driver_detected"],
    "cuda_available": hardware["cuda_available"],
    "allocation_test_passed": hardware.get("allocation_test_passed", False),
    "gpu_ready_at_finish": GPU_READY,
    "cpu_gpu_equivalence_passed": GPU_EQUIVALENT,
    "participants_fitted": len(fit_results),
    "runtime_error": GPU_RUNTIME_ERROR,
    "output_directory": str(OUTPUT_DIR),
}
display(pd.DataFrame([final_status]).T.rename(columns={0: "status"}))

with (OUTPUT_DIR / "gpu_final_status.json").open("w", encoding="utf-8") as handle:
    json.dump(final_status, handle, indent=2)

if not GPU_READY:
    display(pd.DataFrame({"troubleshooting step": TROUBLESHOOTING_STEPS},
                         index=range(1, len(TROUBLESHOOTING_STEPS) + 1)))
elif not GPU_EQUIVALENT:
    print("CUDA works, but numerical equivalence failed. Do not fit yet.")
else:
    print("GPU hardware and numerical-equivalence gates passed.")
''',
)

nbformat.write(notebook, PATH)
print(f"Hardened {PATH}")
