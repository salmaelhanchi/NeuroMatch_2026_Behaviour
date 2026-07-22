"""
hierarchical_online_backfill.py
===============================
Backfill the human-readable ``params`` block of hierarchical_online result
JSONs from the always-complete raw ``theta`` vector.

Why this exists: the shared batch driver's ``_observer_params`` extractor uses a
fixed attribute list written for the other models (``p_random``/``k_like``) and
does not know this model's ``pi``/``R0`` or its ``p_rand``/``k_llh`` naming, so
the saved ``params`` dict can be partial (only ``k_motor``/``alpha``). ``theta``
always holds the full 8-vector, so we expand it via the fitter's ``unpack()``.
Idempotent: a JSON whose ``params`` is already complete is left unchanged.

    python -m observers.fitting.hierarchical_online_backfill          # all subjects
    python -m observers.fitting.hierarchical_online_backfill 5 8      # specific subjects
"""
import glob, json, os, sys
import numpy as np
from observers.fitting import hierarchical_online_fit as F

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                          "results", "fits", "comparison", "hierarchical_online")
NEEDED = ("k_llh", "pi", "p_rand", "k_motor", "alpha", "R0")


def backfill_file(path) -> str:
    with open(path) as fh:
        d = json.load(fh)
    params = d.get("params") or {}
    if all(k in params for k in NEEDED):
        return "already complete"
    theta = d.get("theta")
    if theta is None:
        return "SKIP (no theta)"
    full = F.unpack(np.asarray(theta, float))
    # named, JSON-friendly (coherence keys as strings, like the other models)
    d["params"] = {
        "k_llh": {str(k): float(v) for k, v in full["k_llh"].items()},
        "pi": float(full["pi"]), "p_rand": float(full["p_rand"]),
        "k_motor": float(full["k_motor"]), "alpha": float(full["alpha"]),
        "R0": float(full["R0"]), "mode_init": float(full.get("mode_init", 225.0)),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, path)   # atomic
    return "backfilled"


def main(argv):
    if argv:
        paths = [os.path.join(RESULT_DIR, f"subject{int(s)}.json") for s in argv]
    else:
        paths = sorted(glob.glob(os.path.join(RESULT_DIR, "subject*.json")))
    for p in paths:
        if os.path.exists(p):
            print(f"  {os.path.basename(p)}: {backfill_file(p)}", flush=True)
        else:
            print(f"  {os.path.basename(p)}: MISSING", flush=True)


if __name__ == "__main__":
    main([a for a in sys.argv[1:]])
