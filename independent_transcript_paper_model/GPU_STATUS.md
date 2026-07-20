# Current GPU Status

Checked on July 20, 2026 while the CPU pilot notebook remained open.

## What is detected

- NVIDIA driver: detected by `nvidia-smi`.
- Device: NVIDIA GeForce RTX 3050 Laptop GPU.
- Dedicated memory: 4096 MiB.
- Driver version: 610.47.
- PyTorch: 2.8.0+cu126.
- PyTorch CUDA build: 12.6.
- `torch.cuda.is_available()`: true during successful checks.
- Small CUDA allocation and tensor-operation probe: succeeded during notebook startup.

## What is not yet validated

The full CPU-versus-GPU likelihood equivalence calculation encountered an intermittent CUDA runtime error. Separate checks produced Windows CUDA errors including `out of memory`, `device busy or unavailable`, error 304, and `unknown error`, despite `nvidia-smi` reporting no dedicated GPU memory in use.

Therefore:

- the driver is visible;
- CUDA-enabled PyTorch is installed;
- stable model computation has not yet been demonstrated;
- no GPU participant fit was started;
- no repair or system modification was attempted;
- the notebook correctly disabled GPU fitting.

The recorded smoke result is in `outputs/gpu_smoke_test/gpu_final_status.json`.

## Recommended retry sequence

Perform these steps after the current CPU fit has finished and its outputs are saved.

1. Close unused Jupyter kernels and other GPU-heavy applications.
2. Restart Windows to clear stale Windows Display Driver Model and CUDA contexts.
3. Keep the laptop plugged in and use the Windows/NVIDIA high-performance GPU mode.
4. Run the read-only checks:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); x=torch.zeros(1, device='cuda'); print(x)"
```

5. If both checks succeed, run the short project smoke test:

```powershell
Set-Location "C:\Users\salma\Backup\Desktop\bayesian modeling\independent_transcript_paper_model"
$env:HIERARCHICAL_GPU_SMOKE_TEST="1"
python -m jupyter nbconvert --to notebook --execute "notebooks\02_gpu_hierarchical_fit.ipynb" --output "02_gpu_smoke_executed.ipynb" --output-dir "outputs\gpu_smoke_test" --ExecutePreprocessor.timeout=900
Remove-Item Env:HIERARCHICAL_GPU_SMOKE_TEST
```

6. Start the normal GPU pilot only if the notebook reports that both equivalence checks passed.

## If CUDA remains unstable

1. Confirm in Windows Graphics Settings or NVIDIA Control Panel that `python.exe` uses the RTX 3050 rather than integrated graphics.
2. Confirm that Jupyter uses the same Python executable that reports PyTorch 2.8.0+cu126.
3. Use NVIDIA's official driver page to check for the appropriate current RTX 3050 Laptop GPU driver. Do not change the driver during an active CPU fit.
4. If the driver is healthy but PyTorch remains unstable, create a separate Python environment using the CUDA wheel selected on PyTorch's official installation page. Preserve the working CPU environment.
5. Rerun smoke mode. Do not bypass the CPU/GPU equivalence gate.

The GPU notebook is designed to remain optional. The CPU reference implementation remains the scientific fallback.
