"""
verify_hb_integration.py — COMPATIBILITY SHIM.

Renamed to ``verify_hb_rachel.py`` to match the team's "HB-Rachel" naming.
Re-exports everything; existing imports and the module CLI keep working:
    python -m observers.verification.verify_hb_integration
New code should use ``observers.verification.verify_hb_rachel``.
"""
from observers.verification.verify_hb_rachel import *   # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("observers.verification.verify_hb_rachel", run_name="__main__")
