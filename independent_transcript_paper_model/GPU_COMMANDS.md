# GPU Notebook Commands

The GPU implementation is separate from the CPU reference notebook. It does not install, update, or repair NVIDIA drivers.

## Enter the workspace

```powershell
Set-Location "C:\Users\salma\Backup\Desktop\bayesian modeling\independent_transcript_paper_model"
```

## Read-only driver and backend checks

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

Expected on this computer:

```text
NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 610.47
PyTorch 2.8.0+cu126, CUDA build 12.6, CUDA available True
```

## Open the GPU notebook

```powershell
python -m jupyter lab
```

Open `notebooks/02_gpu_hierarchical_fit.ipynb` and choose **Run All**. The notebook checks hardware and CPU/GPU numerical agreement before fitting. It skips fitting if either check fails.

## Short GPU smoke test

```powershell
$env:HIERARCHICAL_GPU_SMOKE_TEST="1"
python -m jupyter nbconvert --to notebook --execute "notebooks\02_gpu_hierarchical_fit.ipynb" --output "02_gpu_smoke_executed.ipynb" --output-dir "outputs\gpu_smoke_test" --ExecutePreprocessor.timeout=900
Remove-Item Env:HIERARCHICAL_GPU_SMOKE_TEST
```

## Normal GPU pilot

Run this after the CPU reference fit finishes:

```powershell
python -m jupyter nbconvert --to notebook --execute "notebooks\02_gpu_hierarchical_fit.ipynb" --output "02_gpu_pilot_executed.ipynb" --output-dir "outputs\gpu_pilot_30min" --ExecutePreprocessor.timeout=2400
```

## Monitor GPU use without changing it

```powershell
nvidia-smi -l 2
```

Press `Ctrl+C` to stop monitoring. This does not stop the notebook.

## Troubleshooting only if detection fails

1. Run `nvidia-smi`. If Windows does not recognize it, obtain the appropriate RTX 3050 Laptop GPU driver from NVIDIA's official driver page and restart Windows.
2. Run the Python backend check above. A version ending in `+cpu` cannot use CUDA.
3. If CUDA is false, use PyTorch's official Windows/Pip/CUDA selector to create a separate environment. Do not replace the working CPU environment during an active fit.
4. Confirm that the notebook kernel points to the environment containing CUDA-enabled PyTorch.
5. Rerun the smoke test. Do not bypass the equivalence gate.

No automated repair is performed by the project.

## Long all-participant multi-start fit

The long notebook fits all 12 participants independently with four starts each. Each start is limited to 300 evaluations or 20 minutes, giving a maximum declared fitting budget of about 16 hours. It checkpoints after every start and resumes completed work automatically.

Build or rebuild the notebook after changing its readable source:

```powershell
python "tools\build_long_gpu_notebook.py"
```

Run a short end-to-end smoke test first:

```powershell
$env:HIERARCHICAL_LONG_SMOKE_TEST="1"
python -m jupyter nbconvert --to notebook --execute "notebooks\03_gpu_long_multistart_fit.ipynb" --output "03_gpu_long_smoke_executed.ipynb" --output-dir "outputs\gpu_long_multistart_smoke" --ExecutePreprocessor.timeout=1800
Remove-Item Env:HIERARCHICAL_LONG_SMOKE_TEST
```

Run the long notebook without opening a browser:

```powershell
python -m jupyter nbconvert --to notebook --execute "notebooks\03_gpu_long_multistart_fit.ipynb" --output "03_gpu_long_multistart_executed.ipynb" --output-dir "outputs\gpu_long_multistart" --ExecutePreprocessor.timeout=72000
```

The command is finished when nbconvert prints a `Writing ... 03_gpu_long_multistart_executed.ipynb` line and the `PS ...>` prompt returns.

Monitor checkpoint progress from a second PowerShell window:

```powershell
Get-Content "outputs\gpu_long_multistart\progress.json"
```

After interruption, rerun the same long command. Completed `subject/start` checkpoints are skipped. To start a genuinely separate run, use a new output name:

```powershell
$env:HIERARCHICAL_LONG_RUN_NAME="gpu_long_multistart_second_run"
python -m jupyter nbconvert --to notebook --execute "notebooks\03_gpu_long_multistart_fit.ipynb" --output "03_gpu_long_multistart_executed.ipynb" --output-dir "outputs\gpu_long_multistart_second_run" --ExecutePreprocessor.timeout=72000
Remove-Item Env:HIERARCHICAL_LONG_RUN_NAME
```

Optional configuration variables are `HIERARCHICAL_LONG_SUBJECTS`, `HIERARCHICAL_LONG_N_STARTS`, `HIERARCHICAL_LONG_MAX_EVALUATIONS`, and `HIERARCHICAL_LONG_MINUTES_PER_START`. Changing any of them requires a new `HIERARCHICAL_LONG_RUN_NAME`; the notebook refuses to mix incompatible checkpoints.
