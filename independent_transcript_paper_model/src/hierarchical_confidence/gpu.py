"""Optional PyTorch/CUDA acceleration for the validated observer calculation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from .model import GridSpec, HierarchicalObserver, ModelParameters, PreparedSubject
from .readout import MAP_ROUND_DECIMALS

try:
    import torch
except ImportError:  # The CPU project must remain usable without PyTorch.
    torch = None


TROUBLESHOOTING_STEPS = [
    "Run nvidia-smi in PowerShell. If it is unavailable, install or update the NVIDIA driver from NVIDIA's official site, then restart Windows.",
    "Run: python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"",
    "If PyTorch reports a +cpu build or CUDA is False, use a separate environment and install a CUDA-enabled wheel from pytorch.org/get-started/locally/.",
    "Confirm that Jupyter uses the same Python environment in which CUDA-enabled PyTorch is installed.",
    "Do not disable the CPU-versus-GPU equivalence gate to work around a CUDA error.",
]


def cuda_diagnostics() -> dict[str, object]:
    """Inspect driver and PyTorch access without installing or modifying anything."""

    report: dict[str, object] = {
        "driver_detected": False,
        "driver_query": "",
        "torch_installed": torch is not None,
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "device_memory_mib": None,
        "allocation_test_passed": False,
        "allocation_error": "",
    }
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        report["driver_detected"] = True
        report["driver_query"] = completed.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        report["driver_query"] = f"nvidia-smi failed: {error}"

    if torch is not None:
        report["torch_version"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            report["device_name"] = properties.name
            report["device_memory_mib"] = round(properties.total_memory / 1024**2)
            try:
                probe = torch.linspace(0.0, 360.0, 72, dtype=torch.float32, device="cuda:0")
                probe = torch.softmax(torch.cos(torch.deg2rad(probe)), dim=0)
                torch.cuda.synchronize()
                report["allocation_test_passed"] = bool(torch.isfinite(probe).all().item())
                del probe
            except (RuntimeError, torch.AcceleratorError) as error:
                report["allocation_error"] = str(error).splitlines()[0]
    report["gpu_ready"] = bool(
        report["driver_detected"]
        and report["cuda_available"]
        and report["allocation_test_passed"]
    )
    return report


@dataclass(frozen=True)
class EquivalenceResult:
    cpu_nll: float
    gpu_nll: float
    absolute_difference: float
    relative_difference: float
    tolerance: float
    passed: bool

    def as_record(self) -> dict[str, float | bool]:
        return {
            "cpu_nll": self.cpu_nll,
            "gpu_nll": self.gpu_nll,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "relative_tolerance": self.tolerance,
            "passed": self.passed,
        }


class TorchHierarchicalObserver(HierarchicalObserver):
    """Hybrid observer: sequential hidden state on CPU, large batches on CUDA."""

    def __init__(
        self,
        subject: PreparedSubject,
        grid: GridSpec,
        batch_size: int = 1024,
        dtype: str = "float32",
        device: str = "cuda:0",
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. See TROUBLESHOOTING_STEPS.")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA. See TROUBLESHOOTING_STEPS.")
        super().__init__(subject, grid, batch_size=batch_size)
        if dtype not in {"float32", "float64"}:
            raise ValueError("GPU dtype must be 'float32' or 'float64'.")
        self.device = torch.device(device)
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self.dtype_name = dtype
        self._theta_gpu = torch.as_tensor(self.theta, dtype=self.torch_dtype, device=self.device)
        self._theta_bins_gpu = torch.arange(grid.n_angles, dtype=torch.long, device=self.device)
        self._directions_gpu = torch.as_tensor(
            self.subject.directions, dtype=self.torch_dtype, device=self.device
        )
        self._response_bins_gpu = torch.as_tensor(
            self.subject.response_bins, dtype=torch.long, device=self.device
        )
        self._eligible_valid_gpu = []
        self._eligible_all_gpu = []
        for coherence_index in range(len(self.subject.coherence_values)):
            valid = np.flatnonzero(
                (self.subject.coherence_indices == coherence_index) & self.subject.response_valid
            )
            all_trials = np.flatnonzero(self.subject.coherence_indices == coherence_index)
            self._eligible_valid_gpu.append(torch.as_tensor(valid, dtype=torch.long, device=self.device))
            self._eligible_all_gpu.append(
                torch.as_tensor(all_trials, dtype=torch.long, device=self.device)
            )

    def _sensory_log_likelihood_gpu(self, kappa: float):
        difference = torch.deg2rad(self._theta_gpu[:, None] - self._theta_gpu[None, :])
        log_values = kappa * (torch.cos(difference) - 1.0)
        return log_values - torch.logsumexp(log_values, dim=1, keepdim=True)

    def _measurement_pmfs_gpu(self, directions, kappa: float):
        difference = torch.deg2rad(self._theta_gpu[None, :] - directions[:, None])
        log_values = kappa * (torch.cos(difference) - 1.0)
        return torch.softmax(log_values, dim=1)

    def _motor_kernel_gpu(self, kappa: float):
        difference = torch.deg2rad(self._theta_gpu)
        log_values = kappa * (torch.cos(difference) - 1.0)
        return torch.softmax(log_values, dim=0)

    def _tie_aware_map_support_gpu(self, posterior_scores):
        posterior = torch.softmax(posterior_scores, dim=-1)
        factor = float(10**MAP_ROUND_DECIMALS)
        rounded = torch.round(posterior * factor) / factor
        maxima = torch.max(rounded, dim=-1, keepdim=True).values
        tied = rounded == maxima
        tie_counts = tied.sum(dim=-1)
        if torch.any(tie_counts < 1).item():
            raise RuntimeError("Every posterior must have at least one MAP bin.")
        return tied, tie_counts

    def negative_log_likelihood(self, parameters: ModelParameters) -> float:
        parameters.validate(len(self.subject.coherence_values))
        priors, _ = self.effective_priors(parameters.rho)
        log_priors = torch.as_tensor(
            np.log(np.maximum(priors, np.finfo(float).tiny)),
            dtype=self.torch_dtype,
            device=self.device,
        )
        motor = self._motor_kernel_gpu(parameters.motor_kappa)
        total_log_likelihood = torch.zeros((), dtype=self.torch_dtype, device=self.device)

        with torch.inference_mode():
            for coherence_index, sensory_kappa in enumerate(parameters.sensory_kappas):
                eligible = self._eligible_valid_gpu[coherence_index]
                sensory_log_likelihood = self._sensory_log_likelihood_gpu(float(sensory_kappa))
                for start in range(0, len(eligible), self.batch_size):
                    indices = eligible[start : start + self.batch_size]
                    posterior_scores = sensory_log_likelihood[None, :, :] + log_priors[indices, None, :]
                    tied_map, tie_counts = self._tie_aware_map_support_gpu(posterior_scores)
                    measurement = self._measurement_pmfs_gpu(
                        self._directions_gpu[indices], float(sensory_kappa)
                    )
                    response_bin = self._response_bins_gpu[indices, None]
                    motor_by_percept = motor[
                        (response_bin - self._theta_bins_gpu[None, :])
                        % self.grid.n_angles
                    ]
                    motor_probability = (
                        torch.einsum("bmt,bt->bm", tied_map.to(self.torch_dtype), motor_by_percept)
                        / tie_counts
                    )
                    probability = torch.sum(measurement * motor_probability, dim=1)
                    probability = (
                        (1.0 - parameters.lapse) * probability
                        + parameters.lapse / self.grid.n_angles
                    )
                    total_log_likelihood += torch.log(
                        torch.clamp(probability, min=torch.finfo(self.torch_dtype).tiny)
                    ).sum()
        return float((-total_log_likelihood).item())

    def predict_response_pmfs(self, parameters: ModelParameters) -> np.ndarray:
        parameters.validate(len(self.subject.coherence_values))
        priors, _ = self.effective_priors(parameters.rho)
        log_priors = torch.as_tensor(
            np.log(np.maximum(priors, np.finfo(float).tiny)),
            dtype=self.torch_dtype,
            device=self.device,
        )
        motor = self._motor_kernel_gpu(parameters.motor_kappa)
        predictions = torch.empty(
            (self.subject.n_trials, self.grid.n_angles),
            dtype=self.torch_dtype,
            device=self.device,
        )

        with torch.inference_mode():
            for coherence_index, sensory_kappa in enumerate(parameters.sensory_kappas):
                eligible = self._eligible_all_gpu[coherence_index]
                sensory_log_likelihood = self._sensory_log_likelihood_gpu(float(sensory_kappa))
                for start in range(0, len(eligible), self.batch_size):
                    indices = eligible[start : start + self.batch_size]
                    posterior_scores = sensory_log_likelihood[None, :, :] + log_priors[indices, None, :]
                    tied_map, tie_counts = self._tie_aware_map_support_gpu(posterior_scores)
                    measurement = self._measurement_pmfs_gpu(
                        self._directions_gpu[indices], float(sensory_kappa)
                    )
                    percept = torch.einsum(
                        "bm,bmt->bt",
                        measurement / tie_counts,
                        tied_map.to(self.torch_dtype),
                    )
                    convolved = torch.fft.ifft(
                        torch.fft.fft(percept, dim=1) * torch.fft.fft(motor)[None, :], dim=1
                    ).real
                    convolved = torch.clamp(convolved, min=0.0)
                    convolved /= convolved.sum(dim=1, keepdim=True)
                    predictions[indices] = (
                        (1.0 - parameters.lapse) * convolved
                        + parameters.lapse / self.grid.n_angles
                    )
        return predictions.cpu().numpy()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def memory_record(self) -> dict[str, float]:
        return {
            "allocated_mib": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mib": torch.cuda.memory_reserved(self.device) / 1024**2,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }


def compare_cpu_gpu_nll(
    cpu_observer: HierarchicalObserver,
    gpu_observer: TorchHierarchicalObserver,
    parameters: ModelParameters,
    relative_tolerance: float = 1e-6,
) -> EquivalenceResult:
    """Gate GPU fitting on agreement with the validated CPU objective."""

    cpu_nll = cpu_observer.negative_log_likelihood(parameters)
    gpu_nll = gpu_observer.negative_log_likelihood(parameters)
    difference = abs(cpu_nll - gpu_nll)
    relative = difference / max(abs(cpu_nll), 1.0)
    return EquivalenceResult(
        cpu_nll=cpu_nll,
        gpu_nll=gpu_nll,
        absolute_difference=difference,
        relative_difference=relative,
        tolerance=relative_tolerance,
        passed=bool(relative <= relative_tolerance),
    )


