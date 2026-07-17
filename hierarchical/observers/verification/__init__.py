"""observers.verification — model verification suites.

Verification for the four core/original models was moved to the repo-root
``other models/verification/`` folder; the line below re-includes it on this
package's import path so names like ``observers.verification.verify_switching``
keep resolving.
"""
import os as _os

_extra = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), _os.pardir, _os.pardir,
                  "other models", "verification"))
if _os.path.isdir(_extra) and _extra not in __path__:
    __path__.append(_extra)
