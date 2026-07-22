# Model equations — the six observer models

All models operate on the circular direction space θ ∈ {1, …, 360}° with the
prior always centred at **μ = 225°**. Every model shares four building blocks;
they differ only in **how the prior strength enters and whether it is learned**.

## Shared foundations

**Von Mises pdf** (concentration κ, mean μ), evaluated in the numerically stable
scaled form:

$$
V(\theta;\mu,\kappa)=\frac{\exp\!\big(\kappa\cos(\theta-\mu)-\kappa\big)}{2\pi\,I_{0}^{e}(\kappa)},\qquad I_{0}^{e}(\kappa)=I_{0}(\kappa)e^{-\kappa}
$$

As κ → ∞ this collapses to a delta at μ; as κ → 0 it becomes uniform.

**Sensory read-out** $p_{\mathrm e}(\theta\mid\theta_{\text{true}},c)$ — the
Girshick MAP look-up with the prior switched off (κ_prior = 0): a bump at the
stimulus direction whose width shrinks as the coherence-dependent sensory
concentration $k_{e}\equiv k_{\text{like}}(c)$ grows. Three values, one per
coherence $c\in\{0.06,0.12,0.24\}$.

**Prior read-out** — the Girshick MAP look-up with the likelihood switched off
(k_like = 0): a spike at μ = 225°, written $\delta_{225}$.

**Motor noise** — every percept distribution is circularly convolved (⊛) with a
zero-mean von Mises of concentration $k_{\text{motor}}$:

$$
p(\text{est})=p(\text{percept})\;\circledast\;V(\cdot\,;0,k_{\text{motor}})
$$

**Lapse** — a fraction $p_{\text{rand}}$ of trials are uniform; it is folded in
by renormalising the mixture weights to sum to 1 (denominator $1+p_{\text{rand}}$
below).

---

## 1. Static Switching observer (the paper's winner) — 9 parameters

Params: $k_{\text{like}}(c)$ ×3, $k_{\text{prior}}(s)$ ×4 (one per prior width
$s\in\{10,20,40,80\}$°), $p_{\text{rand}}$, $k_{\text{motor}}$.

**Switch weights (Eq. 6)** — reliability-ratio competition between prior and
evidence, with the prior strength a *fixed fitted constant per block*:

$$
w_{\text{prior}}=\frac{k_{\text{prior}}}{k_{\text{prior}}+k_{e}},\qquad
w_{\mathrm e}=\frac{k_{e}}{k_{\text{prior}}+k_{e}}
$$

**Estimate distribution (Eq. 7)** — stochastic mixture of the two read-outs plus
lapse, convolved with motor noise:

$$
p(\text{est}\mid\theta_{\text{true}},c,s)=\Big[\tfrac{w_{\mathrm e}}{1+p_{\text{rand}}}\,p_{\mathrm e}+\tfrac{w_{\text{prior}}}{1+p_{\text{rand}}}\,\delta_{225}+\tfrac{p_{\text{rand}}}{1+p_{\text{rand}}}\,\mathcal U\Big]\circledast V(\cdot;0,k_{\text{motor}})
$$

$\mathcal U = 1/360$ is the uniform. The two peaks (at θ_true and 225°) give the
observed **bimodality**; averaging over trials, not within a trial.

---

## 2. Online Hierarchical Switching observer — 6 parameters

Params: $k_{\text{like}}(c)$ ×3, $p_{\text{rand}}$, $k_{\text{motor}}$, **λ**
(volatility). The four per-block $k_{\text{prior}}$ are *replaced* by a single
belief that is **learned online** from feedback.

The observer carries a belief $b_t(\kappa)$ over prior strength on the
log-spaced grid $\kappa\in[0.01,60]$ (40 points). Two-step recursive update per
trial:

**Predict / volatility (H1a)** — leak toward the initial belief $b_0$:

$$
b_t^{-}(\kappa)=(1-\lambda)\,b_t(\kappa)+\lambda\,b_0(\kappa)
$$

**Correct (H1b)** — multiply by the likelihood of the revealed feedback
direction $f_t$ under each candidate strength, renormalise:

$$
b_{t+1}(\kappa)\;\propto\; b_t^{-}(\kappa)\,V(f_t;225,\kappa)
$$

**Read-out (H2–H4)** — the switch weight is the reliability ratio **averaged
over the belief**:

$$
w_{\text{prior},t}=\mathbb E_{b_t}\!\Big[\frac{\kappa}{\kappa+k_{e}}\Big]=\sum_{\kappa}b_t(\kappa)\,\frac{\kappa}{\kappa+k_{e}},\qquad w_{\mathrm e,t}=1-w_{\text{prior},t}
$$

then the same Eq.-7 mixture as model 1 with $w_{\text{prior}}\!\to\!w_{\text{prior},t}$.

**Reduction:** a belief pinned to a fixed point mass (λ irrelevant, no update) → model 1 exactly.

---

## 3. Asymptote + Transient (AT) Switching observer — 11 parameters

Params: $k_{\text{like}}(c)$ ×3, per-block asymptotes $k_{\text{asym}}(s)$ ×4,
$p_{\text{rand}}$, $k_{\text{motor}}$, and two time-constants
$\tau_{\text{tighten}},\tau_{\text{loosen}}$. Keeps per-block prior *levels* but
adds a within-block exponential **transient** toward them.

**Effective prior concentration (Eq. AT1)** — for trial $t$ measured from the
block onset ($t_{\text{blk}}=0,1,2,\dots$):

$$
k_{\text{eff}}(t)=k_{\text{asym}}+\big(k_{\text{start}}-k_{\text{asym}}\big)\,e^{-t_{\text{blk}}/\tau}
$$

where $k_{\text{start}}$ is the effective strength **carried over** from the end
of the previous block, and the time-constant is direction-dependent:

$$
\tau=\begin{cases}\tau_{\text{tighten}} & k_{\text{asym}}>k_{\text{start}}\quad(\text{prior narrowing})\\[2pt]\tau_{\text{loosen}} & k_{\text{asym}}<k_{\text{start}}\quad(\text{prior widening})\end{cases}
$$

The read-out is the static switch (Eq. 6/7) with $k_{\text{prior}}\to k_{\text{eff}}(t)$.

**Reductions:** carryover off ⇒ $k_{\text{eff}}\equiv k_{\text{asym}}$ = model 1; $\tau\to 0$ ⇒ static except trial 1 of each block.

*Caveat:* the block clock $t_{\text{blk}}$ **resets at the true (unsignalled)
block boundary** — the observer is told where blocks change.

---

## 4. Adaptive-Volatility Switching observer — 6 parameters

Params: $k_{\text{like}}(c)$ ×3, $p_{\text{rand}}$, $k_{\text{motor}}$, and a
single hazard rate **h**. The boundary-agnostic successor to AT: it detects
block changes itself via reduced-Bayesian change-point inference (Nassar 2010;
Adams & MacKay 2007), replacing AT's {4 asymptotes, $k_{\text{start}}$, 2 τ}
with one hazard.

Per-trial belief update over $\kappa$, using the feedback likelihood column
$\ell_t(\kappa)=V(f_t;225,\kappa)$:

**Change-point probability** — how well feedback fits the current belief vs. a
fresh reset to the hyper-prior $b_0$:

$$
p_{\text{stay}}=\sum_\kappa b_t(\kappa)\,\ell_t(\kappa),\qquad
p_{\text{change}}=\sum_\kappa b_0(\kappa)\,\ell_t(\kappa)
$$

$$
\mathrm{CPP}_t=\frac{h\,p_{\text{change}}}{h\,p_{\text{change}}+(1-h)\,p_{\text{stay}}}
$$

**Adaptive forget** — leak toward $b_0$ in proportion to CPP (a data-driven
learning rate replacing the fixed λ):

$$
b_t^{-}(\kappa)=(1-\mathrm{CPP}_t)\,b_t(\kappa)+\mathrm{CPP}_t\,b_0(\kappa)
$$

**Correct** — Bayes update as before: $b_{t+1}(\kappa)\propto b_t^{-}(\kappa)\,\ell_t(\kappa)$.

Read-out identical to model 2 (belief-averaged switch weight). **Reduction:** $h\to 0$ ⇒ $\mathrm{CPP}\to 0$ ⇒ exactly the online learner (model 2) with λ = 0.

---

## 5. HB Integration observer, free-α — 7 parameters

Params: $k_{\text{like}}(c)$ ×3, $p_{\text{rand}}$, $k_{\text{motor}}$, λ, and
the **fitted mixture weight α**. The abstract's model: **no switch** — a mixed
hyper-prior and a single MAP read-out.

**Mixed hyper-prior** on the true direction:

$$
p(\theta\mid\kappa,\alpha)=\alpha\,V(\theta;225,\kappa)+\frac{1-\alpha}{360}
$$

a peaked von Mises unioned with a uniform floor. The posterior
$\propto$ likelihood × this mixed prior is **bimodal** (one lobe pulled toward
225°, one at the evidence). The estimate is the **MAP** of that posterior,
computed per κ and then averaged over the online belief $b_t(\kappa)$ (which is
learned exactly as in model 2, updating κ not α):

$$
p_t(\text{est})=\sum_{\kappa} b_t(\kappa)\,\big[\text{MAP-readout}(\kappa,\alpha)\circledast V(\cdot;0,k_{\text{motor}})\big]
$$

α is a **free constant**, held fixed across blocks. **Reduction:** α = 1 removes
the uniform floor and reproduces the plain Girshick single-von-Mises integration
observer exactly (verified to ~1e-17). *On subject 1 the fit drove α → 1,
shedding the mixture.*

---

## 6. HB Integration observer, derived-α — 6 parameters

Params: $k_{\text{like}}(c)$ ×3, $p_{\text{rand}}$, $k_{\text{motor}}$, λ.
**No free α.** The faithful implementation of the paper's Discussion remark that
the mixture probabilities are "determined by the ratio of likelihood and prior
strengths." α is *computed* per κ-grid point as the reliability ratio — the same
quantity as the switch weight (Eq. 6):

$$
\alpha_i=\frac{\kappa_i}{\kappa_i+k_{e}}
$$

so the mixed hyper-prior becomes $p(\theta\mid\kappa_i)=\alpha_i\,V(\theta;225,\kappa_i)+(1-\alpha_i)/360$, and everything else (MAP read-out, belief-averaging, online κ-learning) is as in model 5. Dropping the free α costs one parameter (7 → 6).

This is the model that makes the project's central point concrete: with α tied
to the reliability ratio, the "integration" observer is **formally a
reformulation of the switch in Bayesian terms** — as the paper anticipated.

---

## Summary table

| # | model | prior strength enters as | learned? | free params |
|---|---|---|---|---|
| 1 | Static Switching | fixed $k_{\text{prior}}(s)$ per block | no | 9 |
| 2 | Online Switching | belief $b_t(\kappa)$, fixed λ leak | yes | 6 |
| 3 | Asymptote+Transient | $k_{\text{eff}}(t)$ exp. transient to per-block asymptote | yes (block clock given) | 11 |
| 4 | Adaptive-Volatility | belief $b_t(\kappa)$, CPP-driven leak | yes (self-detects blocks) | 6 |
| 5 | HB Integration free-α | mixed hyper-prior, α fitted | κ yes, α no | 7 |
| 6 | HB Integration derived-α | mixed hyper-prior, α = κ/(κ+k_e) | yes | 6 |

Models 1–4 (switch family) select **between** prior and evidence via $w_{\text{prior}}$; models 5–6 (integration family) **multiply** them into one posterior and read out its MAP. All six share $V$, the sensory/prior read-outs, motor noise, and the lapse; the equations above are the exact mechanisms in `switching_observer.py`, `online_learner.py`, `online_helpers.py`, `asymptote_transient.py`, `adaptive_volatility_switching.py`, `hb_integration.py`, and `hb_integration_derived.py`.
