# Project Commands

All commands below are run from PowerShell. They are the complete reusable command set for building, checking, opening, and executing the first pilot notebook.

## Enter the independent workspace

```powershell
Set-Location "C:\Users\salma\Backup\Desktop\bayesian modeling\independent_transcript_paper_model"
```

## Check source syntax

```powershell
python -m compileall -q "src"
```

This catches invalid Python syntax without fitting the data.

## Run focused tests

```powershell
python -m pytest -q
```

This checks circular arithmetic, chronological cleaning, missing-response handling, kappa support, hidden-state continuity, normalized predictions, and agreement between plotted predictions and the fitted likelihood.

## Rebuild the notebook after editing its cell source

```powershell
python "tools\build_notebook.py"
```

The readable cell source is retained so notebook structure can be reviewed without manually editing notebook JSON.

## Open the main notebook

```powershell
python -m jupyter lab
```

Open `notebooks/01_30_minute_hierarchical_fit.ipynb` and choose **Run All**. With no special environment variable, this starts the planned two-participant exploratory fit.

## Execute a short smoke test

```powershell
$env:HIERARCHICAL_SMOKE_TEST="1"
python -m jupyter nbconvert --to notebook --execute "notebooks\01_30_minute_hierarchical_fit.ipynb" --output "01_smoke_executed.ipynb" --output-dir "outputs\smoke_test" --ExecutePreprocessor.timeout=600
Remove-Item Env:HIERARCHICAL_SMOKE_TEST
```

This uses 24 angle bins, 6 total kappa support points, and 2 evaluations per participant. It verifies the complete notebook without spending the normal fitting budget.

## Execute the normal pilot without the interactive interface

```powershell
python -m jupyter nbconvert --to notebook --execute "notebooks\01_30_minute_hierarchical_fit.ipynb" --output "01_pilot_executed.ipynb" --output-dir "outputs\pilot_30min" --ExecutePreprocessor.timeout=2400
```

This uses 72 angle bins, 16 total kappa support points, up to 100 evaluations, and a 12-minute limit for each of two participants.

## Use a different raw CSV location

```powershell
$env:HIERARCHICAL_DATA_PATH="C:\absolute\path\to\data01_direction4priors.csv"
python -m jupyter lab
Remove-Item Env:HIERARCHICAL_DATA_PATH
```

The default location is `C:\Users\salma\Downloads\data01_direction4priors.csv`.

## Implementation verification record

During initial implementation, the source syntax check and focused tests were run before notebook execution. The notebook smoke-test command is used for the final end-to-end verification. No command reads from or imports the excluded `new_implementation` folder.
