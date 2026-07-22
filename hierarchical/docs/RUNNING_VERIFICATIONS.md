# Running the model verifications

Verification is **Step 0** of the pipeline: model-specific identity checks with
known answers that must hold regardless of data. Nothing downstream (fits, CV,
figures) should be trusted until these are green — a model that fails a reduction
or normalisation check is mis-implemented, and its NLL is meaningless.

Each check is not "does it run" but "does the model do the specific thing it
claims, and is it distinct from its nearest sibling?" — so every suite carries at
least one **discriminator** that would fail for a sibling model.

---

## Run everything (the one command)

```python
from observers import api
api.verify_all()      # prints PASS/FAIL per check; returns {suite: (passed, total)}
```

This runs all **8 registry models** and prints a grand total (currently
**78 checks**). `verify_all()` returns a dict, so you can gate on it:

```python
out = api.verify_all()
assert all(p == t for p, t in out.values()), "a verification failed"
```

## Run one model

Either through the API:

```python
from observers import api
api.verify_switching()          # -> (passed, total)
api.verify_hb_adaptive()
api.verify_recombined()
api.verify_hb_salma()
api.verify_reliability_mixture()
```

or the module directly (prints and sets a non-zero exit code on failure, so it
works in a shell / CI):

```bash
python -m observers.verification.verify_switching
python -m observers.verification.verify_hb_adaptive
```

## The suites and what each asserts

| suite (`api.verify_*`) | model | script | checks | discriminator |
|---|---|---|---|---|
| `verify_switching` | switch | `verify_switching.py` | 5 | Eq.6 reliability weights; prior read-out is δ@225 |
| `verify_basic_bayesian` | basic_bayes | `verify_basic_bayesian.py` | 5 | k_prior→0 ⇒ estimate at stimulus; k_like→0 ⇒ at prior |
| `verify_online` | *(online switch)* | `verify_online_switching_observer.py` | 6 | λ=0 no-op / λ=1 full reset; E[k] rises with aligned feedback |
| `verify_hb_rachel` | hb_rachel | `verify_hb_rachel.py` | 12 | prior-reliance declines with distance (integration) vs flat (switch) |
| `verify_hb_adaptive` | hb_adaptive | `verify_hb_adaptive.py` | 16 | α is **learned** (rises/falls with feedback) vs Rachel's fixed α |
| `verify_recombined` | recombined | `verify_recombined.py` | 11 | integrate-before ≠ integrate-after under a **spread** belief |
| `verify_hb_salma` | hb_salma | `verify_hb_salma.py` | 17 | native-72 vs deg360 grid contract; sensory-κ pulls percept off prior |
| `verify_reliability_mixture` | reliability_mixture | `verify_reliability_mixture.py` | 6 | reliance weight rises/falls with feedback agreement |

Notes:
- The `verify_online_switching_observer` suite covers the online-switch model,
  which is **not** a live registry key but is kept verified.
- `verify_hb_integration` is a backward-compat alias for `verify_hb_rachel`.
- Every suite exposes the house contract `run() -> (passed, total)` and clears its
  own results on entry, so it is safe to call repeatedly in a notebook.

## Common check families (what a green suite guarantees)

1. **Reduction** — the read-out collapses to a known closed form in a limiting
   case (α=1 → von Mises; single-κ belief → base read-out).
2. **Normalisation** — every predicted distribution is a proper pmf (sums to 1,
   non-negative).
3. **Emergence** — bimodality appears from graded integration (far + low
   coherence) and disappears near the prior / at high coherence.
4. **NLL ordering** — `NLL(true) < NLL(bad)` on self-simulated data.
5. **Discriminator** — the check that separates the model from its nearest sibling.
6. **Cost** — one full-sequence NLL eval on a real subject, so fit budget is known.

## When a check fails

- **Reduction/normalisation FAIL** → the model is mis-implemented; fix the model,
  not the check. Do not fit.
- **Discriminator FAIL** → the model is behaviourally indistinguishable from its
  sibling (or the intended structural difference was lost in a refactor); it does
  not earn a separate registry entry until this passes.
- **NLL-ordering FAIL** → the likelihood or the simulator disagree; a fit will
  chase noise.
- A **cost** line is informational (not a pass/fail gate), but a surprising time
  there is a heads-up about batch feasibility.

## Verification vs the full validation record

`api.verify_all()` is the fast, live correctness gate (seconds). It is distinct
from `python -m observers.comparison.validate_all`, which produces the durable,
**stamped** validation *record* — it re-runs Step 0 live AND reads the on-disk
outputs of the expensive stages (fits/CV/shape/recovery), then writes
`results/fits/VALIDATION_REPORT.md` + `.json` with git SHA, timestamp, and library
versions, exiting non-zero on any FAIL. Use `verify_all()` while developing; use
`validate_all` to certify a comparison for the record. See
`fitting_cv_validation_procedure.md` for the six-point checklist it walks.
