"""observers.fitting — model fitters.

One per-model fitter module each (thin wrappers over the shared optimisation
machinery in ``online_recovery.py``): ``switching_observer_fit``,
``online_switching_observer_fit``, ``basic_bayesian_fit``, ``hb_rachel_fit``,
``hb_adaptive_confidence_fit``, ``hierarchical_online_fit``,
``reliability_mixture_fit``, plus ``fair_refit`` for the fair multi-start
comparison table.
"""
