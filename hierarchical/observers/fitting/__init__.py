"""observers.fitting — model fitters.

Fitters for the four core/original models were moved to the repo-root
``other models/fitting/`` folder; the line below re-includes it on this
package's import path so names like ``observers.fitting.hb_integration_fit``
keep resolving.
"""
import os as _os

_extra = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir,
                  "other models", "fitting"))
if _os.path.isdir(_extra) and _extra not in __path__:
    __path__.append(_extra)
