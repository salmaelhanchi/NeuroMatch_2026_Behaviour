# Posterior Motives — start here

Morning! Everything you need for your presentation section is ready. Two things
per person: **your notebook** and **one guide for your LLM**. That's it.

## What to do (5 minutes)

1. **Open your notebook in Google Colab** (your name below). In Colab:
   `File → Open notebook → GitHub`, search `salmaelhanchi/NeuroMatch_2026_Behaviour`,
   pick branch **`model-verification`**, and open your notebook. It lives in
   your own folder at
   `hierarchical/experiments/<you>/01_slide_notebook/` (your exact path is in
   the table below).
2. **Runtime → Run all.** The first cell clones the repo and sets everything up
   (~1 min), then the notebook runs top to bottom. **This is a starter
   notebook** — I've set it up around your slide's questions with the two
   models already wired to the fitted results, so it's a working scaffold to
   build from, not your finished figures. Run it, see what it produces, then
   shape it into your slide.
3. **To edit or extend it, work with your LLM.** Paste
   **[`docs/REPO_GUIDE_FOR_LLMS.md`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/docs/REPO_GUIDE_FOR_LLMS.md)**
   into your LLM assistant (Claude / ChatGPT /
   Colab's Gemini) as context, then ask it in plain English for the changes you
   want. That guide teaches the assistant the whole repo — the data, the
   models, and the one API you call — so it can write correct code for you.

## Your notebook

| you | notebook (click to open on GitHub, `model-verification` branch) | your slide |
|---|---|---|
| **Anirban** | [`anirban_bimodality.ipynb`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/experiments/anirban/01_slide_notebook/anirban_bimodality.ipynb) | Hierarchical model overview + does it reproduce **bimodal & unimodal** responses? |
| **Rachel** | [`rachel_model_comparison.ipynb`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/experiments/rachel/01_slide_notebook/rachel_model_comparison.ipynb) | Switch vs Hierarchical: in-sample (AIC/BIC) **and** out-of-sample (cross-validation) |
| **Salma** | [`salma_prior_learning.ipynb`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/experiments/salma/01_slide_notebook/salma_prior_learning.ipynb) | Prior learning across prior widths (10/20/40/80) + ANOVA |
| **Romi** | [`romi_learning_rate.ipynb`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/experiments/romi/01_slide_notebook/romi_learning_rate.ipynb) | Learned prior width per block + the learning-rate λ per subject |
| **Valeria** | [`valeria_kappa_comparison.ipynb`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/experiments/valeria/01_slide_notebook/valeria_kappa_comparison.ipynb) | Switch's fixed per-block prior vs the Hierarchical model's learned prior |

Each notebook starts with two models — **Switch** (the paper's model) and
**HB-Rachel** (our hierarchical Bayesian observer) — already wired to the real
fitted results (not placeholders). Treat it as your starting point: the
analysis for your slide's questions is scaffolded in, and you refine it from
there.

I picked **HB-Rachel to start because it's the simplest of our hierarchical
models** — easiest to explain on a slide. But it's not your only option: there
are **six fitted models in total** (HB-Adaptive, HB-Salma, Recombined, and a
Basic-Bayes baseline, alongside Switch and HB-Rachel). If you'd rather show a
different one, or compare across several, **your LLM can swap or add models for
you** — they're all reachable through the same API, so it's usually a one-line
change. The full list is in [`docs/REPO_GUIDE_FOR_LLMS.md`](https://github.com/salmaelhanchi/NeuroMatch_2026_Behaviour/blob/model-verification/hierarchical/docs/REPO_GUIDE_FOR_LLMS.md).

Two more models are **in the works** — **Hier-Online** (learns both the prior's
mean *and* width online) and **Reliability-Mixture** (Romi's model, a discrete
prior-vs-evidence mixture with a learned reliance weight). Their code is done
and checked, but they aren't fitted yet, so they're not in the notebooks — hold
off on using them until the fits are in.

## If something's missing

Ask your LLM (with the guide loaded) — it can even write the code and set up a
git branch for you to push. It won't push on your behalf; it'll hand you the
exact command. Or just ping Rachel/Salma.

## Two notes

- **Everything is on the `model-verification` branch**, not `main`. The Colab
  setup cell already targets it — you don't have to think about this.
- **Save anything you make** — figures, tables — under
  `experiments/<you>/` so it stays with your work and out of everyone else's.

Go get 'em. — Rachel
