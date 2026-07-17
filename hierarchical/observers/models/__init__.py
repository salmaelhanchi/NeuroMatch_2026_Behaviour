"""observers.models — observer model implementations.

The four extension models (Anirban's spec) live here. The four core/original
models were moved to the repo-root ``other models/models/`` folder; the line
below re-includes that folder on this package's import path so their original
module names (e.g. ``observers.models.hb_integration``) keep resolving.
"""
import os as _os

_extra = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir,
                  "other models", "models"))
if _os.path.isdir(_extra) and _extra not in __path__:
    __path__.append(_extra)
