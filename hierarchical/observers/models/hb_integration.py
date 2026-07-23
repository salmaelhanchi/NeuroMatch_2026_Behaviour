"""
hb_integration.py  — COMPATIBILITY SHIM.

This model was renamed to match the team's naming: the implementation now lives
in ``hb_rachel.py`` (class ``HBRachelObserver``, display label "HB-Rachel").

This shim re-exports every public name so existing imports keep working, e.g.
    from observers.models.hb_integration import HBIntegrationObserver
    from observers.models.hb_integration import mixture_prior, _map_readout, PRIOR_MEAN
New code should import from ``observers.models.hb_rachel`` instead.
"""
from observers.models.hb_rachel import *          # noqa: F401,F403
from observers.models.hb_rachel import (           # explicit: underscore names + alias
    HBRachelObserver,
    HBIntegrationObserver,
    mixture_prior,
    mixture_map_lookup,
    _map_readout,
    _map_readout_col,
    _map_readout_cols,
    PRIOR_MEAN,
)
