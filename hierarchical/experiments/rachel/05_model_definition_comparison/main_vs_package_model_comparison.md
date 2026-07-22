# Where the main-branch model notebooks diverge from the hierarchical/ package

Compares the two "previous version" notebooks now on `main` against the two
package modules in `hierarchical/observers/models/`:

| role | main-branch version | package version |
|---|---|---|
| switching | `Switching_Bayesian_Observer_starter.ipynb` | `observers/models/switching_observer.py` |
| HB integration | `hb_verified_model_implementation.ipynb` | `observers/models/hb_integration.py` |

The first thing to know: **these are not two edits of the same file.** The main
notebooks are function-based (free functions + a small `ObserverParams`
dataclass, run top-to-bottom in a notebook); the package is class-based modules
imported as `observers.models.*`. A line-by-line text diff is meaningless, so
this compares them where it matters — equations, parameters, readout, learning,
and what each reduces to.

The reassuring headline: **they agree on the core Bayesian machinery** (same
von Mises primitives, same Girshick MAP readout, the *same* mixed prior form).
They diverge on three axes: packaging/verifiability, the online-learning
implementation, and a few parameter conventions.

---

## 1. Switching: starter notebook vs switching_observer.py

**Same core.** Both build the estimate the same way: `girshick_lookup` /
`girshick_map_lookup` — measurement pdf `P(m|dir)`, likelihood `P(dir|m)`, a von
Mises prior at 225, MAP readout. The package's `switching_observer.py` was
ported from exactly these primitives, so the estimate distribution is the same
computation.

**Divergences:**

| aspect | starter notebook | switching_observer.py |
|---|---|---|
| Form | free functions (`girshick_lookup`, `trial_loglike`) | `SwitchingObserver` dataclass with `estimate_distribution`, `negative_log_likelihood` |
| Readout options | MAP / BLS / Sampling all in the notebook | MAP only (the paper's winning "switching" readout); BLS/sampling are separate models |
| Extra readout | `sampling_lookup(..., weight_tail)` present | not in this class (sampling is out of scope for the switch) |
| Cardinal prior | not present | `k_cardinal` param (default 0 = paper's "withoutCardinal" winner) |
| Default k_prior | `{80:0.5, 40:1.0, 20:2.5, 10:6.0}` | `{80:0.5, 40:1.4, 20:2.7, 10:8.7}` |
| Default k_motor | 30.0 | 40.0 |
| Verification | none | `verify_switching` (5/5): evidence readout == von Mises, prior == delta, weights == Eq.6 ratio, valid distributions, NLL ordering |

The parameter-default differences are cosmetic — both are *fitted* per subject,
so the starting values don't affect the answer. The substantive difference is
that the package version is a single-readout switch with a verification suite
and a cardinal-prior hook; the notebook is an exploratory multi-readout
playground.

---

## 2. HB: hb_verified_model_implementation.ipynb vs hb_integration.py

This is the more interesting comparison, because the two **converged on the same
model idea independently** — and then diverged on how faithfully the online
learning is implemented.

**What they agree on (this is real agreement, not coincidence):**
- **Identical mixed prior.** Both write the prior as
  `mix_weight * peaked_vonMises + (1 - mix_weight) * uniform`
  (`build_prior_grid` in the notebook; `mixture_prior` in the package, where
  `alpha` == `mix_weight`). Same equation.
- **Both compute a "responsibility."** The notebook has
  `compute_responsibility(z_peaked, z_uniform, mix_weight)`; the package treats
  the estimate as a *derived* responsibility (which mixture component explains
  the measurement). Same concept.
- **Same readout + motor/lapse tail.** MAP readout, von Mises motor
  convolution, uniform lapse floor. Same pipeline.

**Where they diverge — three real differences:**

### (a) Online learning: scaffold vs wired-in Bayesian belief

This is the biggest divergence and the one that matters for the abstract.

- **Notebook:** online learning is a **scaffold, not wired into the
  likelihood.** It carries proxy columns (`within_block_learning_progress`,
  `prior_confidence_t_proxy`) built with a fixed exponential
  `1 - exp(-t/tau)` shape, and an `update_online_prior_state` delta-rule that
  updates a *scalar* kappa: `kappa <- kappa + lr*(target - kappa)`. The actual
  `negative_log_likelihood` reads the prior confidence per trial from a proxy
  column or the static `k_prior_by_std` — the learning is documented and
  simulated but not estimated through the response likelihood.
- **Package:** online learning is **the model.** `hb_integration.py` carries a
  full **belief distribution** `b_t(kappa)` over a kappa grid, updates it with a
  proper Bayesian step from feedback (`forget` + `bayes_correct`), and averages
  the readout over that belief. `lam` is a real fitted volatility/forgetting
  parameter. This is verified (`verify_hb_integration` T6): the belief converges
  to the true kappa and the recursive update equals the batch posterior.

In short: the notebook *describes and simulates* online prior learning as a
deterministic delta-rule on a point estimate; the package *implements and fits*
it as Bayesian belief updating over a distribution. For the abstract's core
claim ("trial-by-trial learning of prior confidence"), the package version is
the one that actually tests the hypothesis.

### (b) What's learned, and the mix_weight/alpha default

- Notebook `mix_weight` defaults to **1.0** — i.e. by default it's a pure
  peaked prior with no uniform component (the mixture is available but off).
- Package `alpha` is a fitted mixture weight (default 0.6), held fixed across
  blocks *by design*, while **kappa** is the learned latent. The package
  documents this fork explicitly (learn kappa, not alpha) so it stays
  point-for-point comparable to the online switching observer.

### (c) Parameter set / packaging

| aspect | notebook | hb_integration.py |
|---|---|---|
| Params | `ObserverParams`: k_llh_by_coherence, k_prior_by_std, mix_weight, k_motor, lapse_rate, readout | dataclass: k_like(3), alpha, k_motor, p_random, lam, + kappa belief on a 15-pt grid |
| N params (fitted) | ~ 3 k_llh + 4 k_prior + mix_weight + k_motor + lapse | 7 (3 k_like + alpha + k_motor + p_random + lam) |
| Readout | switchable MAP/BLS/SAMPLE at param level | MAP (with alpha=1 -> exact Girshick reduction) |
| Reduction test | none automated | verified: alpha=1 == Girshick to 1e-17 |
| Verification | `model_variable_verification.md` (variable-name check) | `verify_hb_integration` (12/12 numerical checks) |

---

## Summary

- **Switching:** same underlying Girshick computation; the package adds a
  single-readout class, a cardinal hook, and a verification suite. No conceptual
  divergence.
- **HB:** same mixed-prior model and responsibility concept — genuine agreement
  — but the notebook implements online learning as a **deterministic delta-rule
  scaffold that is not fit through the likelihood**, while the package
  implements it as **Bayesian belief updating over kappa that is fit and
  verified**. That is the divergence that matters for the abstract.
- **Cross-cutting:** the package versions are verifiable (5/5 and 12/12
  numerical reductions) and reduce exactly to known baselines; the notebooks are
  readable exploratory implementations with variable-name/structural checks but
  no automated numerical verification.

Neither is "wrong" — the notebooks are the readable scaffolds the team built
first; the package modules are the fit-and-verify versions. For the abstract's
learning claim, use the package HB model, since it is the one where the
trial-by-trial learning is actually estimated rather than assumed.
