"""Build two paired notebooks: a runnable tutorial (with interpretation
questions) and a solutions notebook (same steps + answers). Kept in one builder
so the two stay in sync."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HEADER_SETUP = '''\
import sys, os
# Point Python at the package. Run this notebook from the repo root
# (hierarchical/) OR set the path below to your checkout.
REPO = os.path.abspath(os.getcwd())
if "observers" not in os.listdir(REPO):
    # adjust if you launched the notebook from docs/ or notebooks/
    REPO = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)
print("repo root:", REPO)
print("observers present:", "observers" in os.listdir(REPO))'''

def q(md):      # interpretation question block
    return new_markdown_cell("> **Question.** " + md)

def ans(md):    # solutions-only answer block
    return new_markdown_cell("> **Answer.** " + md)

# ---------------------------------------------------------------------------
# Shared step content: (markdown_intro, code, question_text, answer_text)
# code is a string; None question => no question for that step.
# ---------------------------------------------------------------------------
STEPS = []

STEPS.append(("intro", """# Validating the two abstract models — Tutorial

This notebook walks through how we confirm the two models the abstract compares
work: **`switching_observer`** (the paper's model) and **`hb_integration`** (our
hierarchical Bayesian model).

"Does it work" has **three** separate meanings, each needing its own evidence:

1. **Correctness** — does the code compute what the equations say? *(verification suites)*
2. **Fittability** — can it be fit to a real subject and return sensible parameters? *(smoke fit)*
3. **Parameter recovery** — if we simulate data from known parameters and refit, do we get them back? *(the test that tells us a fitted number is trustworthy)*

Every step below runs end-to-end. After the important ones, a **Question** asks
you to interpret what you just saw — jot your answer, then check the solutions
notebook.

All of these checks call scripts that already live in the `observers/` package —
nothing here is hand-written analysis. You can run each from a terminal too
(the equivalent command is shown).""", None, None, None))

STEPS.append(("setup", "## Setup\n\nPoint Python at the package and confirm the data + models import.",
              HEADER_SETUP, None, None))

STEPS.append(("importcheck", "Confirm all the pieces import.",
'''from observers.models.switching_observer import SwitchingObserver
from observers.models.hb_integration import HBIntegrationObserver
import observers.verification.verify_switching as verify_switch
import observers.verification.verify_hb_integration as verify_hb
import observers.fitting.switching_observer_fit as switch_fit
import observers.fitting.hb_integration_fit as hb_fit
import observers.fitting.online_recovery as recovery
print("all imports OK")''', None, None))

# ---- LEVEL 1 ----
STEPS.append(("l1head", """## Level 1 — Correctness

Each model ships a **verification suite**. These are stronger than "does it run":
each check compares the model against an **oracle with a known answer** (a
reduction, a limit, or an exact identity). If a reduction is off, the code is
wrong no matter how well it later fits.

Terminal equivalent:
```
python -m observers.verification.verify_switching
python -m observers.verification.verify_hb_integration
```""", None, None, None))

STEPS.append(("l1switch", "### switching_observer",
'''verify_switch.run()''',
"The third check says `switch weights match Eq.6 reliability ratio P(evidence)=0.6667`. Where does 0.6667 come from, given the prior and sensory concentrations used in that test?",
"The switch's reliance on evidence is `p_e = k_e / (k_e + k_prior)` (paper Eq. 6). The test uses `k_e = 2*k_prior`, so `p_e = 2k/(2k+k) = 2/3 = 0.6667`. The check confirms the code computes the exact reliability ratio, not an approximation."))

STEPS.append(("l1hb", "### hb_integration",
'''verify_hb.run()''',
"Two checks matter most here. (a) The `alpha=1 == Girshick` reduction passes at max|Δ| ~ 1e-17. Why is that reduction the single most important correctness check for this model? (b) The **discriminator** check reports `integ CV=1.34 vs switch CV=0.04`. What does that prove, and why does the whole abstract depend on it?",
"(a) With `alpha=1` the mixed prior becomes a pure von Mises with no uniform floor, so the model must collapse to the standard Girshick Bayesian estimator. Matching it to machine precision proves the integration read-out is implemented correctly — everything else (bimodality, learning) is built on that core being right. (b) CV (coefficient of variation) measures how much each model's prior-reliance changes as the stimulus moves away from the prior. Integration's reliance falls off sharply (CV=1.34) because the uniform floor takes over for far stimuli; the switch's stays flat (CV=0.04). This proves the two models make **different, measurable predictions** — if they didn't, comparing them on real data would be meaningless. The abstract's entire question ('integration vs switch') only has an answer because they're distinguishable."))

# ---- LEVEL 2 ----
STEPS.append(("l2head", """## Level 2 — Fittability

Can each model be fit to a real subject and return sensible parameters? We fit
**subject 1** (n = 8562 trials). The switch fits in ~30 s; HB integration is
slower (the read-out grid is evaluated per kappa), so we **cap the iterations**
here to keep the tutorial responsive — a production fit uses more starts and
runs 3–5 min/subject.

Terminal equivalent:
```
python -m observers.fitting.switching_observer_fit human 1
python -m observers.fitting.hb_integration_fit human 1
```""", None, None, None))

STEPS.append(("l2switch", "### Fit the switch to subject 1",
'''d = switch_fit._load_subject(1)
obs, nll, aic, bic = switch_fit.fit(d, maxiter=400)
print(f"switch subj1  n={len(d['estimates'])}  NLL={nll:.1f}  AIC={aic:.1f}  BIC={bic:.1f}")
print("implied prior SDs (deg):", {k: round(v,1) for k,v in obs.prior_sd_degrees().items()})
print("fitted k_like:", {k: round(float(v),2) for k,v in obs.k_like.items()})''',
"The implied prior SDs come back around 96/78/36/8 deg. The experiment's four blocks had prior SDs of 80/40/20/10 deg. Is the model working? And why might `k_like[0.24]` come back as a huge number (~10^5)?",
"Yes — the recovered SDs track the four block widths in the right order and rough magnitude (wide blocks give wide fitted priors), which is exactly what a working model should do. The huge `k_like[0.24]` is an **identifiability ridge**, not a bug: at high coherence (24%) the sensory likelihood is nearly a delta, so its exact concentration barely affects the fit and the optimiser pushes it to the ceiling. It means 'very reliable' — don't over-interpret that single number."))

STEPS.append(("l2hb", "### Fit HB integration to subject 1 (capped)",
'''d = hb_fit._load_subject(1)
obs, nll, theta = hb_fit.fit(d, maxiter=60)   # capped smoke fit
import numpy as np
N=7; aic = 2*N + 2*nll
print(f"hb_integration subj1  NLL={nll:.1f}  AIC={aic:.1f}  (capped maxiter=60)")
print("fitted alpha (mixture weight):", round(float(obs.alpha),3))
print("fitted k_like:", {k: round(float(v),2) for k,v in obs.k_like.items()})''',
"What does the fitted `alpha` (~0.5) mean in words, and why is it the parameter you most want to be able to trust?",
"`alpha` is the weight on the peaked prior in the mixed hyper-prior: `p(theta) = alpha * VonMises(225, kappa) + (1-alpha)/360`. An alpha ~ 0.5 means the subject's prior structure applies about half the time and the other half they behave as if any direction is equally likely. It's the model's defining parameter — it *is* the integration story — so if it can't be recovered reliably, the model's central claim is untestable. That's exactly what Level 3 checks."))

# ---- LEVEL 3 ----
STEPS.append(("l3head", """## Level 3 — Parameter recovery (the decisive test)

Verification proves the code is correct; fitting proves it runs. **Recovery**
proves a fitted number is *trustworthy*: simulate data from known parameters,
refit, and check you get them back. If recovery fails, no parameter you report
from real data means anything.

Terminal equivalent:
```
python -c "from observers.fitting.online_recovery import static_parameter_recovery; static_parameter_recovery()"
python -m observers.fitting.hb_integration_fit recover
```""", None, None, None))

STEPS.append(("l3switch", "### Switch recovery",
'''ok_switch = recovery.static_parameter_recovery(seeds=(1,2,3))''',
"`k_like` recovers almost exactly, but `k_prior[10]` (the tightest prior, SD~10 deg) recovers least precisely. Why is the tight prior the hardest parameter to pin down?",
"A very tight prior (SD10) means the prior and the true stimulus are usually close together, so on most trials you can't tell whether the estimate came from the prior or the evidence — both point to nearly the same place. Few trials are *informative* about the prior's exact strength, so its estimate is noisy. This is expected identifiability structure, not a defect: sensory reliabilities (many informative trials) always recover better than prior reliabilities, and among priors the tightest is loosest."))

STEPS.append(("l3hb", "### HB integration recovery",
'''hb_fit.recover()''',
"The output ends with `alpha CLUSTERS near truth (identifiable)` and reports alpha and p_random separately. Why is checking alpha *against the lapse rate* the crucial part of this test?",
"In any mixture model, a lapse (random responding) can masquerade as 'sometimes ignore the prior' — both produce responses unrelated to the stimulus. If alpha and the lapse rate `p_random` trade off, you can't tell a low-alpha subject from a high-lapse one, and alpha becomes meaningless. The recovery test shows alpha lands at 0.602 ± 0.006 (truth 0.6) while p_random stays separately recovered at ~0.03 — they do **not** trade off. That's what licenses reporting a fitted alpha as a real, interpretable quantity."))

STEPS.append(("summary", """## Summary

| | switching_observer | hb_integration |
|---|---|---|
| Correctness (verification) | 5/5 PASS | 12/12 PASS |
| Fittability (real subject) | ✓ | ✓ |
| Parameter recovery | ✓ (k_like tight; k_prior ordered) | ✓ (all <~10%; alpha identifiable) |

Both models are **verified, fittable, and identifiable**. Every check above is a
script in `observers/` you can re-run.

**Next (Task 5):** fit both models to all 12 subjects under identical
preprocessing and compare NLL / AIC / BIC plus cross-validated held-out
likelihood — the comparison that actually decides between them.""", None, None, None))


def build(with_answers: bool):
    cells = []
    for name, md, code, question, answer in STEPS:
        if name == "intro" and with_answers:
            md = md.replace("— Tutorial", "— Solutions").replace(
                "jot your answer, then check the solutions notebook.",
                "each Question is followed by its Answer.")
        if md:
            cells.append(new_markdown_cell(md))
        if code:
            cells.append(new_code_cell(code))
        if question:
            cells.append(q(question))
            if with_answers and answer:
                cells.append(ans(answer))
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    return nb

nbf.write(build(False), "validate_models_tutorial.ipynb")
nbf.write(build(True),  "validate_models_solutions.ipynb")
print("wrote validate_models_tutorial.ipynb and validate_models_solutions.ipynb")
print("tutorial cells:", len(build(False).cells), " solutions cells:", len(build(True).cells))
