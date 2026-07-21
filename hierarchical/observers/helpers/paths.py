"""Central filesystem paths for the project.

Every script reads data and reads/writes results through the constants here, so
the layout is defined in exactly one place. Paths are absolute (anchored to the
project root via this file's location), so a script finds the data and results
folders no matter which directory you launch it from.
"""
from __future__ import annotations

from pathlib import Path

# observers/helpers/paths.py  ->  parents[2] is the project root (hierarchical/)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
DATA_CSV = DATA_DIR / "data01_direction4priors.csv"

RESULTS_DIR = ROOT / "results"
FITS_DIR = RESULTS_DIR / "fits"          # fitted-parameter JSON files
FIGURES_DIR = RESULTS_DIR / "figures"    # generated .png figures

# The fitted-parameter files, shared across fitting/analysis scripts.
HUMAN_FITS = FITS_DIR / "human_fit_results.json"          # static + online switching
AT_FITS = FITS_DIR / "at_fit_results.json"                # asymptote + transient
HB_FITS = FITS_DIR / "hb_integration_results.json"        # hierarchical integration
FAIR_FITS = FITS_DIR / "fair_fit_results.json"            # fair 4-model comparison
BASIC_FITS = FITS_DIR / "basic_bayesian_results.json"     # basic Bayesian baseline
