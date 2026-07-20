import numpy as np
import pytest

from hierarchical_confidence.gpu import (
    TorchHierarchicalObserver,
    compare_cpu_gpu_nll,
    cuda_diagnostics,
)
from hierarchical_confidence.model import (
    GridSpec,
    HierarchicalObserver,
    ModelParameters,
    PreparedSubject,
)


def _small_subject(grid: GridSpec) -> PreparedSubject:
    return PreparedSubject(
        subject_id=77,
        directions=np.array([225.0, 235.0, 45.0, 55.0, 225.0, 45.0]),
        coherence_indices=np.array([0, 1, 0, 1, 0, 1]),
        coherence_values=np.array([0.06, 0.24]),
        response_bins=np.array([15, 16, 3, 4, 15, 3]) % grid.n_angles,
        response_valid=np.array([True, True, True, True, False, True]),
        prior_std=np.array([10.0, 10.0, 80.0, 80.0, 20.0, 20.0]),
        block_ids=np.array(["a", "a", "b", "b", "c", "c"]),
        trial_indices=np.array([1, 2, 1, 2, 1, 2]),
    )


def test_cuda_diagnostics_has_non_destructive_status_fields() -> None:
    report = cuda_diagnostics()
    expected = {"driver_detected", "torch_installed", "cuda_available", "gpu_ready"}
    assert expected.issubset(report)


def test_gpu_matches_cpu_on_controlled_sequence() -> None:
    report = cuda_diagnostics()
    if not report["gpu_ready"]:
        pytest.skip("CUDA allocation is not available in this test environment.")
    grid = GridSpec(n_angles=24, n_positive_kappa=5)
    subject = _small_subject(grid)
    cpu = HierarchicalObserver(subject, grid, batch_size=3)
    parameters = ModelParameters(
        rho=0.91,
        sensory_kappas=np.array([2.0, 9.0]),
        motor_kappa=18.0,
        lapse=0.015,
    )
    try:
        gpu = TorchHierarchicalObserver(subject, grid, batch_size=3, dtype="float64")
        comparison = compare_cpu_gpu_nll(cpu, gpu, parameters, relative_tolerance=1e-8)
        cpu_predictions = cpu.predict_response_pmfs(parameters)
        gpu_predictions = gpu.predict_response_pmfs(parameters)
    except Exception as error:
        if "CUDA" in str(error) or "cuda" in str(error) or "Memory allocation" in str(error):
            pytest.skip(f"CUDA became unavailable during the controlled test: {error}")
        raise
    assert comparison.passed, comparison
    np.testing.assert_allclose(gpu_predictions, cpu_predictions, rtol=1e-8, atol=1e-10)
