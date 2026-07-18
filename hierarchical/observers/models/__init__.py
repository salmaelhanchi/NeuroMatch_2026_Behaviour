"""observers.models — observer model implementations.

This branch holds the two models the abstract compares: the Switching observer
(``switching_observer.py``) and the HB integration model (``hb_integration.py``).
Both import the online switching observer's estimate machinery, which lives in
the repo-root ``other models/models/`` folder; the line below re-includes that
folder on this package's import path so ``observers.models.online_switching_observer``
keeps resolving.
"""
import os as _os

_extra = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir,
                  "other models", "models"))
if _os.path.isdir(_extra) and _extra not in __path__:
    __path__.append(_extra)
