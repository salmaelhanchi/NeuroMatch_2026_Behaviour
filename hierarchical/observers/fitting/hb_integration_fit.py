"""
hb_integration_fit.py — COMPATIBILITY SHIM.

Renamed to ``hb_rachel_fit.py`` to match the team's "HB-Rachel" naming.
This shim re-exports everything so existing imports keep working, e.g.
    from observers.fitting.hb_integration_fit import fit, cv, human, recover
    import observers.fitting.hb_integration_fit as hb_fit
New code should import from ``observers.fitting.hb_rachel_fit`` instead.
"""
from observers.fitting.hb_rachel_fit import *          # noqa: F401,F403
from observers.fitting.hb_rachel_fit import (            # explicit non-star names
    pack, unpack, nll_masked, fit, recover, human, cv,
    _trial_logliks, _simulate, _load_subject, _starts_for, _row,
)

if __name__ == "__main__":
    # Preserve CLI behaviour: `python -m observers.fitting.hb_integration_fit ...`
    import runpy
    runpy.run_module("observers.fitting.hb_rachel_fit", run_name="__main__")
