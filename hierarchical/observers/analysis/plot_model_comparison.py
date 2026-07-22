"""
plot_model_comparison.py
========================

One figure comparing the fitted observers on the axes that together define them:

  A  fit quality      — fair (all multi-started) ΔAIC per subject.
  B  temporal dynamics — effective prior width over trials: how each model SETS
                         the prior strength trial-by-trial (this is where the
                         three switch models differ from each other).
  C  read-out          — response distribution at a MATCHED prior strength: how
                         each model turns prior+evidence into a response (this is
                         where the switch family differs from integration).

Panels B and C use canonical illustrative parameters (labelled as such) to show
mechanism; panel A uses the real fair-comparison AICs.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from observers.helpers.circular import von_mises_std, deg2rad_signed
from observers.helpers.dataset import SD_TO_K
from observers.models.switching_observer import SwitchingObserver
from observers.models.online_switching_observer import OnlineHierarchicalObserver
from observers.models.hb_rachel import HBRachelObserver
from observers.helpers.paths import FIGURES_DIR

# validated categorical palette; basic=grey baseline, integration=blue
COL = {"basic": "#7a7a7a", "static": "#008300", "online": "#e87ba4",
       "integration": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

KLIKE = {0.06: 1.0, 0.12: 3.0, 0.24: 8.0}
KMOTOR, PRAND, LAM = 30.0, 0.02, 0.08

# baseline first, then the four learning/switch models (only models present in
# the loaded AICs are drawn, so this stays correct if `basic` is not yet fit)
ORDER = ["basic", "static", "online", "integration"]


def load_comparison_aics(models=None):
    """Per-subject AIC for every fitted model, read from the per-model fit
    folders (``results/fits/comparison/<model>/subject<N>.json``).

    This is the single source of truth for the comparison, so the figure never
    drifts from the committed fits. Returns ``{subject: {model_key: aic}}`` for
    whichever (model, subject) fits exist on disk — missing fits are simply
    absent, so the figure honestly reflects what has been fit. Pass ``models``
    to restrict to a subset of registry keys (default: all registered models).
    """
    from observers.comparison.fit_batch import _result_path
    from observers.comparison.registry import ALL_MODELS, ALL_SUBJECTS
    keys = models or ALL_MODELS
    aic = {}
    for key in keys:
        for sid in ALL_SUBJECTS:
            p = _result_path(key, sid)
            if not p.exists():
                continue
            d = json.load(open(p))
            if d.get("aic") is None:
                continue
            aic.setdefault(sid, {})[key] = d["aic"]
    return aic


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
def panel_A(ax, aic):
    from observers.comparison.registry import build_registry, ALL_MODELS
    subs = sorted(aic, key=int)
    # registry insertion order is the canonical display order; only models
    # present for EVERY shown subject get a (consistent) bar slot
    models = [m for m in ALL_MODELS if all(m in aic[s] for s in subs)]
    reg = build_registry(models)
    col = {m: getattr(reg[m], "color", None) or "#888888" for m in models}
    lab = {m: getattr(reg[m], "label", m) for m in models}
    x = np.arange(len(subs)); w = 0.8 / max(len(models), 1)
    ymax = 0.0
    for i, m in enumerate(models):
        d = [aic[s][m] - min(aic[s].values()) for s in subs]
        ymax = max(ymax, max(d))
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, d, w,
                      color=col[m], label=lab[m], zorder=3)
        # mark the winner per subject with a dot at its (zero-height) base
        for b, v in zip(bars, d):
            if v == 0:
                ax.plot(b.get_x() + b.get_width() / 2, 0, marker="v",
                        color=col[m], markersize=5, zorder=5, clip_on=False)
    _style(ax)
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subs], color=INK)
    ax.set_xlabel("subject", color=INK, fontsize=9)
    ax.set_ylabel("ΔAIC from best  (lower = better fit)", color=INK, fontsize=9)
    ax.set_title("A · Fair fit (ΔAIC)", color=INK, fontsize=10,
                 loc="left", fontweight="bold")
    ax.set_ylim(0, ymax * 1.24 if ymax else 950)
    ncol = 3 if len(models) > 4 else len(models)
    ax.legend(frameon=False, fontsize=7.6, ncol=ncol, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), labelcolor=INK, columnspacing=1.0,
              handlelength=1.2, handletextpad=0.4)


# ---------------------------------------------------------------------------
def _belief_sd_path(obs, feedback):
    """Believed prior SD per trial from the belief path only (no read-out), so
    it is cheap enough to Monte-Carlo average over feedback realisations."""
    obs._prepare(np.array([225]), np.array([0.12]))   # builds the belief table
    b = obs.initial_belief(); out = []
    for f in feedback:
        out.append(von_mises_std(float(np.sum(b * obs.k_grid))))
        b = obs.update_belief(b, int(f))
    return np.array(out)


def panel_B(ax):
    """Effective prior width (deg) over a common block sequence — the EXPECTED
    trajectory (averaged over feedback noise) so the mechanism is legible."""
    blocks = [80, 20, 40, 10]; per = 80
    sd_seq = np.repeat(blocks, per); n = sd_seq.size
    k_true = np.array([SD_TO_K[s] for s in sd_seq])
    mu = deg2rad_signed(np.array([225.0]))[0]

    # static: fits k_prior per block -> knows the width exactly (step function)
    ax.step(np.arange(n), sd_seq, where="post", color=COL["static"], lw=2.2,
            label="static (fixed per block)", zorder=4)

    # online + integration: single global belief -> average over 30 feedback draws
    on = OnlineHierarchicalObserver(k_like=KLIKE, k_motor=KMOTOR, p_random=PRAND, lam=LAM)
    ig = HBRachelObserver(k_like=KLIKE, alpha=0.6, k_motor=KMOTOR, p_random=PRAND, lam=LAM)
    on_paths, ig_paths = [], []
    for seed in range(80):
        rng = np.random.RandomState(seed)
        fb = np.mod(np.round(np.degrees(rng.vonmises(mu, k_true))), 360).astype(int)
        fb[fb == 0] = 360
        on_paths.append(_belief_sd_path(on, fb)); ig_paths.append(_belief_sd_path(ig, fb))
    ax.plot(np.arange(n), np.mean(on_paths, 0), color=COL["online"], lw=2,
            label="online (one global belief)", zorder=3)
    ax.plot(np.arange(n), np.mean(ig_paths, 0), color=COL["integration"], lw=2.4,
            label="integration (mixed-prior belief)", zorder=6)

    for bk in range(1, len(blocks)):
        ax.axvline(bk * per, color=MUTED, lw=0.8, ls=":", alpha=0.6)
    _style(ax)
    ax.set_xlabel("trial   (blocks: true prior SD 80→20→40→10)", color=INK, fontsize=9)
    ax.set_ylabel("effective prior SD (deg)", color=INK, fontsize=9)
    ax.set_title("B · Prior strength set over trials", color=INK, fontsize=10,
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK, loc="upper right")
    ax.set_ylim(0, 92); ax.set_xlim(0, n)


# ---------------------------------------------------------------------------
def panel_C(ax):
    """Response distribution at a MATCHED prior (SD~40), far-from-prior + low coh."""
    theta, coh, k40 = 85, 0.06, SD_TO_K[40]
    grid = np.arange(1, 361)

    # switch family: static/online share ONE read-out at a fixed prior k.
    sw = SwitchingObserver(k_like=KLIKE, k_prior={"40": k40}, p_random=PRAND, k_motor=KMOTOR)
    d_sw = sw.estimate_distribution(theta, coh, "40")
    ax.fill_between(grid, d_sw, color=MUTED, alpha=0.18, zorder=1)
    ax.plot(grid, d_sw, color=MUTED, lw=2, label="switch family (static/online)", zorder=3)

    # integration: point-mass belief at the same k -> its own read-out
    ig = HBRachelObserver(k_like=KLIKE, alpha=0.6, k_motor=KMOTOR, p_random=PRAND)
    ig._prepare(np.array([theta]), np.array([coh]))
    b = np.zeros(ig.k_grid.size); b[int(np.argmin(np.abs(ig.k_grid - k40)))] = 1.0
    d_ig = ig.estimate_distribution(coh, theta, b)
    ax.plot(grid, d_ig, color=COL["integration"], lw=2.4, label="integration", zorder=4)

    ax.axvline(theta, color=MUTED, ls="--", lw=1, alpha=0.7)
    ax.axvline(225, color=MUTED, ls="--", lw=1, alpha=0.7)
    ax.text(theta, ax.get_ylim()[1], " stimulus", fontsize=8, color=MUTED, va="top")
    ax.text(225, ax.get_ylim()[1], " prior (225)", fontsize=8, color=MUTED, va="top")
    _style(ax)
    ax.set_xlim(1, 360); ax.set_xticks([1, 85, 180, 225, 360])
    ax.set_xlabel("reported direction (deg)", color=INK, fontsize=9)
    ax.set_ylabel("P(response)", color=INK, fontsize=9)
    ax.set_title("C · Read-out at matched prior", color=INK, fontsize=10,
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")


def make_figure():
    """Build the three-panel comparison figure and return it (renders inline in
    a notebook). Panel A reads the fair AICs from results/fits/ via
    load_comparison_aics()."""
    fig = plt.figure(figsize=(15, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.25, 1], wspace=0.28)
    panel_A(fig.add_subplot(gs[0]), load_comparison_aics())
    panel_B(fig.add_subplot(gs[1]))
    panel_C(fig.add_subplot(gs[2]))
    fig.suptitle("Observers of prior-guided motion estimation — fit · dynamics · read-out",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.5, y=1.06)
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # headless CLI: render to file, no display needed
    fig = make_figure()
    out_png = FIGURES_DIR / "model_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
