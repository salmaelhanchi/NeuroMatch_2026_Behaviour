"""Guard benchmark and prediction cells against transient Windows CUDA loss."""

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
    raise ValueError(f"Could not find cell containing {marker}")


replace_code_cell(
    "benchmark = pd.DataFrame()",
    r'''
benchmark = pd.DataFrame()
if GPU_READY and GPU_EQUIVALENT:
    try:
        subject_id = pilot_subjects[0]
        parameters = ParameterTransform(prepared[subject_id].coherence_values).default_parameters()
        _ = gpu_fit_observers[subject_id].negative_log_likelihood(parameters)
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
    except Exception as error:
        GPU_RUNTIME_ERROR = f"GPU benchmark failed: {type(error).__name__}: {error}"
        GPU_READY = False
        print(GPU_RUNTIME_ERROR)
        print("GPU fitting is disabled; no repair was attempted.")
else:
    print("Benchmark skipped.")
''',
)

replace_code_cell(
    "predictions = {}",
    r'''
predictions = {}
state_summaries = {}
if fit_results:
    for subject_id, result in fit_results.items():
        try:
            predictions[subject_id] = gpu_fit_observers[subject_id].predict_response_pmfs(
                result.parameters
            )
        except Exception as error:
            GPU_RUNTIME_ERROR = f"GPU prediction failed: {type(error).__name__}: {error}"
            GPU_READY = False
            print(GPU_RUNTIME_ERROR)
            print("Remaining GPU predictions are skipped; no repair was attempted.")
            break
        state_summaries[subject_id] = gpu_fit_observers[subject_id].state_summary(
            result.parameters.rho
        )
        assert np.allclose(predictions[subject_id].sum(axis=1), 1.0, atol=2e-6)
        np.savez_compressed(
            OUTPUT_DIR / f"participant_{subject_id}_gpu_trial_predictions.npz",
            response_pmf=predictions[subject_id], theta_degrees=grid.theta_degrees,
        )
        state_summaries[subject_id].to_csv(
            OUTPUT_DIR / f"participant_{subject_id}_gpu_hidden_state.csv", index=False
        )
    if predictions:
        print("Completed GPU response predictions normalize and were saved.")
        tie_diagnostics = pd.DataFrame([
            {
                "subject_id": subject_id,
                **cpu_observers[subject_id].tie_diagnostics(fit_results[subject_id].parameters),
            }
            for subject_id in predictions
        ])
        tie_diagnostics.to_csv(
            OUTPUT_DIR / "gpu_tie_aware_map_diagnostics.csv", index=False
        )
        display(tie_diagnostics)
        print("Tie-aware MAP diagnostics use normalized posteriors rounded to six decimals.")
else:
    print("Prediction generation skipped because no GPU fit completed.")
''',
)

# The plotting cell should only use participants with completed predictions.
replace_code_cell(
    "if fit_results:\n    fig, axes",
    r'''
plotted_subjects = [subject_id for subject_id in pilot_subjects if subject_id in predictions]
if plotted_subjects:
    fig, axes = plt.subplots(2, len(plotted_subjects), figsize=(6.5 * len(plotted_subjects), 7), squeeze=False)
    for column, subject_id in enumerate(plotted_subjects):
        history = fit_results[subject_id].history
        axes[0, column].plot(history["evaluation"], history["nll"], color="#2b6f77")
        axes[0, column].set(title=f"Participant {subject_id}: GPU optimization", xlabel="Evaluation", ylabel="NLL")

        state = state_summaries[subject_id]
        smooth = state["expected_hidden_kappa"].rolling(75, center=True, min_periods=1).mean()
        axes[1, column].plot(state["subject_trial_position"], smooth, color="#d1495b")
        boundaries = np.flatnonzero(
            state["block_id"].to_numpy()[1:] != state["block_id"].to_numpy()[:-1]
        ) + 1
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
''',
)

nbformat.write(notebook, PATH)
print(f"Added runtime guards to {PATH}")
