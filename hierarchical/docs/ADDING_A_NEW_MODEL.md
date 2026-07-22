# Adding a new model to the comparison

How to take a new observer from a bare class to a fully fit, cross-validated, and
validated member of the comparison. This is the *operational* companion to
`fitting_cv_validation_procedure.md` (which explains **why** each step exists);
here we cover **what to do**, in order.

Everything routes through the `ModelSpec` registry in
`observers/comparison/registry.py`. The pipeline never imports a model directly —
it only sees the registry — so once your model has a spec, every stage (fit, CV,
shape, recovery, figure, table) picks it up for free.

---

## 0. What a new model must provide

Three files, mirroring every existing model (see the per-model table in
`fitting_cv_validation_procedure.md`):

| file | what it holds |
|---|---|
| `observers/models/<m>.py` | the observer class: `estimate_distribution(...)`, `negative_log_likelihood(...)`, and (if it learns) `filter(directions, coherences, feedback, record_belief=)` |
| `observers/fitting/<m>_fit.py` | `N_PARAMS`, `pack(...)`/`unpack(theta)`, `fit(data, maxiter, mask)`, a per-trial log-lik function, and `_simulate(obs, design, seed)` for recovery |
| `observers/verification/verify_<m>.py` | model-specific identity checks with a `run() -> (passed, total)` |

Plus **one** `ModelSpec` entry and **one** line in `_BUILDERS` in `registry.py`.

The fitter may be shared (e.g. `recombined` reuses `hb_rachel_fit.py`) or inline in
the registry (e.g. `hb_salma`), but the interface above is what the spec needs.

---

## 1. Write the model class

Minimum surface the registry calls:

- `estimate_distribution(coherence, direction, belief_or_state) -> pmf` on the
  **360 degree grid** (length-360, sums to 1). If your model is natively on a
  coarser grid (like HB-Salma's 72 bins), expose a `grid="deg360"` up-sampled
  read-out so its NLL is comparable to the others — NLLs on different grids are
  **not** comparable.
- `negative_log_likelihood(estimates, directions, coherences, ...) -> float`.
- If the model **learns** a latent trial-by-trial, add
  `filter(directions, coherences, feedback=None, record_belief=False)`. Follow the
  house causal ordering exactly: **read out the response for trial t from the
  current belief, THEN update the belief with `feedback[t]`.** `feedback` must
  default to `directions` (the true motion direction — the exogenous stimulus, not
  the subject's estimate). This keeps the point-fit NLL a clean one-step-ahead
  predictive likelihood (no leakage).

## 2. Write the fitter

- `N_PARAMS` — the free-parameter count `k` (drives AIC/BIC).
- `pack(...) -> theta` and `unpack(theta) -> observer` — round-trip exactly
  `N_PARAMS` numbers, in **transformed** (unconstrained) space so Nelder-Mead can
  roam (log for concentrations, logit for probabilities).
- `fit(data, maxiter, mask) -> FitResult` — Nelder-Mead, tolerances
  `xatol=fatol=1e-2` (the house default; the paper uses 1e-4 — tighten if your
  surface is well-conditioned). Respect `mask` (a boolean over trials) so CV can
  fit on the training subset while the belief still filters over the full ordered
  sequence.
- `_simulate(obs, design, seed) -> data dict` — generate synthetic responses from
  known params; **required** for parameter/model recovery (Step 4).

## 3. Register the model

Add a `_<m>_spec()` builder and one `_BUILDERS` line. `ModelSpec` fields:

```python
ModelSpec(
    name="<m>", label="<Display>", n_params=F.N_PARAMS,
    color="#RRGGBB",          # fixed plot colour, threaded across every panel
    grid_deg=360,             # comparability note
    learns=True,              # does it carry a latent across trials?
    _fit=F.fit,
    _trial_logliks=F.trial_logliks,
    _simulate=F._simulate,    # enables recovery
    _rebuild=F.unpack,        # rebuild observer from a saved fit dict
    _predict=my_predict_fn,   # (obs, data) -> (N,360) per-trial predicted dists
)
```

Then add to `_BUILDERS`:

```python
_BUILDERS = {
    # ...
    "<m>": _<m>_spec,
}
```

`ALL_MODELS` and the canonical display order derive from `_BUILDERS`, so the model
now appears everywhere automatically.

## 4. Verify BEFORE fitting (Step 0)

Write `verify_<m>.py` and wire it into `observers/api.py` (`verify_<m>()` wrapper +
one line in `verify_all()`). It must assert, at minimum:

1. **Reduction** — the read-out collapses to a known closed form in a limiting
   case (e.g. alpha=1 -> single von Mises; single-kappa belief -> the base read-out).
2. **Normalisation** — every predicted distribution is a proper pmf.
3. **NLL ordering** — `NLL(true) < NLL(bad)` on self-simulated data.
4. **A DISCRIMINATOR** — at least one check that would **fail for the nearest
   sibling model**. This is the check that earns the model its place. Examples in
   the repo: integrate-before vs integrate-after diverge only under a *spread*
   belief; HB-Adaptive's alpha *moves* with feedback while HB-Rachel's is fixed.

Run it:

```bash
python -m observers.verification.verify_<m>              # this model only
python -c "from observers import api; api.verify_all()"  # whole suite
```

Do not fit until this is green.

## 5. Fit, cross-validate, and validate (Steps 1-4)

Run one model through the whole pipeline. Serial:

```bash
python -m observers.comparison.run_all --models <m>
```

or parallel (recommended for a batch):

```bash
python -m observers.comparison.run_parallel --fit-models <m> --cv-models <m> --workers 6
```

Key flags: `--subjects 1 2 3` (default all 12), `--maxiter 400`, `--folds 5`,
`--rec-nsim` (recovery simulations per model — raise to 20-50 for a real confusion
matrix), `--force` (ignore resume-skip). Both drivers are **resumable**: a fit is
skipped if an existing file was produced at `maxiter >=` the requested value.

Then stamp a reproducible validation record:

```bash
python -m observers.comparison.validate_all --models switch basic_bayes hb_adaptive <m> --ref switch --folds 5
```

This re-runs the cheap checks live (Step 0 + CV config), reads the durable JSON
outputs of the expensive stages, and writes
`results/fits/VALIDATION_REPORT.md` + `.json` stamped with git SHA, timestamp, and
library versions. It exits non-zero on any FAIL, so it can gate CI.

## 5b. Regenerating a model whose fits went stale (one-touch)

When a model's committed fits were produced by an under-converged or otherwise
superseded code path and must be regenerated through the *current* code, use the
one-touch orchestrator instead of chaining commands by hand:

```bash
python -m observers.comparison.refit_model --model <m>
```

It runs the whole workflow in order and stops at the first failure:

0. **verify** — `api.verify_<m>()` must be green before any compute is spent;
1. **refit + CV** — forced (`--force`, so the stale files are actually overwritten)
   multi-start point fit + K-fold CV of `<m>` **only**;
2. **comparison** — rebuild shape / figure / table over the full comparison set
   (default `CORE5`), so the refit doesn't collapse the figure onto one model;
3. **validate** — stamp the comparison-wide `results/fits/VALIDATION_REPORT.{md,json}`
   AND a per-model record `results/fits/comparison/<m>/validation.{md,json}` (the
   latter carries this model's per-subject fit table + start_spread check).

It removes the two footguns of the hand-typed version: forgetting `--force` (which
silently no-ops, because resume-skip keeps the stale file) and letting
`run_parallel`'s shared stages rebuild the figure over `--fit-models` alone. Useful
flags: `--no-cv` (fit-only models like `hb_salma`/`recombined`), `--subjects`,
`--compare <set>`, `--skip-compare`/`--skip-validate` for a bare refit. After it
finishes, confirm `start_spread` is **non-zero** in the new fit files — that is the
evidence the multi-start path actually explored.

## 6. Where the outputs land

See `docs/RESULTS_LAYOUT.md` for the full tree. In one line: point fits ->
`results/fits/comparison/<m>/subject<N>.json`; CV ->
`results/fits/comparison_cv/<m>/subject<N>_cv.json`; shape/recovery/report are
model-pooled under their own `results/fits/comparison_*` folders.

---

## Checklist for a new model

- [ ] `models/<m>.py` — 360 degree pmf read-out; house causal ordering if it learns
- [ ] `fitting/<m>_fit.py` — `N_PARAMS`, pack/unpack, `fit`, `_simulate`
- [ ] `verify_<m>.py` — reduction + normalisation + NLL ordering + **a discriminator**
- [ ] `ModelSpec` + `_BUILDERS` line in `registry.py`
- [ ] `verify_<m>()` wrapper + line in `api.verify_all()`
- [ ] `api.verify_all()` green
- [ ] `run_all`/`run_parallel` complete for the model
- [ ] `validate_all` report shows the model with no FAILs
- [ ] **House invariant:** adding the model left every *other* model's fits
      bit-identical (shared-code touches must not perturb siblings)
