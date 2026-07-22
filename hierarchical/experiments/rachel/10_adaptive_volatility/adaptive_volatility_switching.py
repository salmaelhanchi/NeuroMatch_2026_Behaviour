"""
adaptive_volatility_switching.py
================================

**The boundary-agnostic successor to `asymptote_transient.py`** — the version of
the "learning transient" model that is most defensible for the Laquitaine &
Gardner (2018) experiment, in which *block changes were unsignaled*.

Why replace AT's mechanism
--------------------------
`asymptote_transient.py` models within-block learning as an exponential relaxation
`k_eff(t) = k_asym + (k_start - k_asym)·exp(-t/tau)` whose clock `t` **resets at
the true block boundary**. But subjects were never told when the prior changed —
they had to *infer* it from the stream of feedback directions. So AT is handed
information the observer did not have, which is the first objection a reviewer
will raise. AT also carries 11 parameters (4 free asymptotes + 2 tau + start +
switch machinery), and those asymptotes/tau are weakly identified (subject-1
tau_tighten pinned at ~0; fragile fits).

This model fixes both at once by fusing AT's *goals* (per-block prior levels +
a within-block transient) with the online learner's *machinery*, using a
change-point-driven adaptive learning rate (reduced Bayesian online change-point
detection; Nassar et al. 2010; Adams & MacKay 2007; Behrens et al. 2007):

    p_stay   = Σ_k b_{t-1}(k) · V(f_t; 225, k)     # feedback under current belief
    p_change = Σ_k b_0(k)     · V(f_t; 225, k)     # feedback under reset-to-hyperprior
    CPP_t    = h·p_change / (h·p_change + (1-h)·p_stay)     # change-point probability
    b_t^-    = (1 - CPP_t)·b_{t-1} + CPP_t·b_0              # forget in proportion to CPP
    b_t(k)   ∝ b_t^-(k) · V(f_t; 225, k)                    # Bayes correction

Interpretation
--------------
- **Surprising feedback** (poorly explained by the current belief, e.g. just
  after an unsignaled block change) drives `CPP` up → strong forgetting toward
  the hyper-prior → fast re-adaptation. That IS AT's transient, but *triggered by
  inferred surprise, not the true boundary*.
- **Consistent feedback** → `CPP ≈ 0` → near-pure accumulation → the belief
  settles at that block's true prior width. That IS AT's per-block asymptote, but
  *emergent* rather than 4 free parameters.
- The single dynamics parameter is the **hazard rate `h`** (prior probability of a
  change per trial), replacing AT's {4 asymptotes, k_start, 2 tau}.

Properties (verified in __main__ and the fit script)
----------------------------------------------------
- **Nests the online learner:** h → 0 ⇒ CPP → 0 ⇒ pure accumulation
  (matches `OnlineHierarchicalObserver(lam=0)` to ~1e-10).
- **Tracks unsignaled block widths:** on synthetic 80/40/20/10° blocks the belief
  converges near each true width with no boundary information.
- **Bimodality preserved** (far-from-prior + low coherence → stimulus peak +
  prior peak), so it remains a genuine *switching* observer.
- **Fits at least as well as AT with 6 params vs 11** (subjects 1 & 3): AIC
  77133 vs 77252 (subj 1), 83482 vs 83488 (subj 3).

Free parameters (6): 3 k_e (per coherence) + k_motor + p_random + hazard h.
The inherited `lam` field is unused (forgetting is set by CPP each trial).

Everything else — the reliability-ratio switch read-out (Eq. 6), the
deterministic-belief exact likelihood, motor noise, lapses, AIC/BIC — is
inherited unchanged from `OnlineHierarchicalObserver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from online_learner import OnlineHierarchicalObserver


@dataclass
class AdaptiveVolatilitySwitchingObserver(OnlineHierarchicalObserver):
    """Online switching observer with a change-point-driven adaptive learning
    rate. Boundary-agnostic replacement for the asymptote+transient model.

    Free parameters (6): 3 k_e + k_motor + p_random + hazard.
    """

    hazard: float = 0.05
    N_PARAMS = 6

    def update_belief(self, belief, feedback_dir):
        """Reduced-Bayesian change-point update. `_obs_table[k, θ-1]` is the
        von Mises feedback likelihood V(θ;225,k) per grid point (built by the
        parent's `_prepare`); `_belief0` is the hyper-prior over k."""
        like_col = self._obs_table[:, feedback_dir - 1]

        # how well does feedback fit the current belief vs. a fresh reset?
        p_stay = float(belief @ like_col)
        p_change = float(self._belief0 @ like_col)

        h = self.hazard
        denom = h * p_change + (1.0 - h) * p_stay + 1e-320
        cpp = (h * p_change) / denom                       # change-point probability

        # forget toward the hyper-prior in proportion to CPP (adaptive rate)
        pred = (1.0 - cpp) * belief + cpp * self._belief0
        s = pred.sum()
        pred = pred / s if s > 0 else self._belief0.copy()

        # Bayes correction with this trial's feedback
        post = pred * like_col
        ps = post.sum()
        return post / ps if ps > 0 else pred


if __name__ == "__main__":
    from online_learner import OnlineHierarchicalObserver as _Onl
    from online_simulate import make_synthetic_design
    from circular import von_mises_std

    kl = {0.06: 1.0, 0.12: 3.0, 0.24: 8.0}

    # 1) reduction to the online learner at h -> 0
    d = np.array([85, 225, 300, 150, 200, 260] * 6)
    c = np.array([0.06, 0.24, 0.12] * 12)
    a = AdaptiveVolatilitySwitchingObserver(k_like=kl, k_motor=30.0, p_random=0.02, hazard=1e-9)
    o = _Onl(k_like=kl, k_motor=30.0, p_random=0.02, lam=0.0)
    ao = a.filter(d, c, feedback=d, sample=False)
    oo = o.filter(d, c, feedback=d, sample=False)
    md = max(float(np.abs(ao["dists"][t] - oo["dists"][t]).max()) for t in range(len(d)))
    print(f"reduction h->0 vs online(lam=0): max|Δ|={md:.2e}")

    # 2) boundary-agnostic tracking of unsignaled block widths
    des = make_synthetic_design(trials_per_block=200, seed=1)
    g = AdaptiveVolatilitySwitchingObserver(k_like=kl, k_motor=40.0, p_random=0.02, hazard=0.08)
    dirs = des.motion_direction.values.astype(int)
    cohs = des.motion_coherence.values.astype(float)
    tr = g.filter(dirs, cohs, feedback=dirs, sample=False, record_belief=True)
    sd = tr["believed_sd"]
    print("believed SD at block ends (no boundary info):",
          [f"{sd[(i + 1) * 200 - 1]:.0f}" for i in range(4)],
          "vs true [80,40,20,10]")
