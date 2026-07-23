"""
plot_split_half.py — figure + summary table for the split-half bimodality test.

Reads results/split_half_subject*.json (written by split_half_bimodality.py) and
produces:
  split_half_bimodality.png   — held-out predicted vs observed far-band signature
  split_half_summary.csv       — per-subject, both directions, both metrics

The scientific read: on the HELD-OUT half, does each model's fitted-parameter
prediction of the bimodality signature (far-band prior-cluster mass, valley
depth) match what the subject actually did? Switch should track observed valley
depth; Basic-Bayes should sit near zero valley regardless of fit.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COL = {"switch": "#30638e", "basic_bayes": "#a0a0a0", "observed": "#222222"}
LABEL = {"switch": "Switch", "basic_bayes": "Basic-Bayes"}


def _degenerate(params, valley_pred):
    """A split-half fit is degenerate when the optimizer diverged so far it
    corrupts the far-band signature: the learnt prior collapsed toward uniform
    (k_prior[80] < 0.01, i.e. no prior to mix in) or the prediction is NaN
    (k_like ran to inf). The high-coherence k_like flat direction (delta cap at
    k>300) is NOT degenerate — it leaves the low-coherence signature untouched."""
    kp = params.get("k_prior") or {}
    kp80 = float(kp.get("80", float("nan")))
    return (not np.isfinite(kp80)) or (kp80 < 0.01) or (valley_pred is None) or \
           (isinstance(valley_pred, float) and np.isnan(valley_pred))


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "split_half_subject*.json"))):
        d = json.load(open(f))
        sid = d["subject"]
        for dk, rec in d["directions"].items():
            o = rec.get("observed") or {}
            for m in d["models"]:
                r = rec.get(m) or {}
                rows.append(dict(
                    subject=sid, direction=dk, model=m,
                    far_mass_obs=o.get("far_mass_obs"), valley_obs=o.get("valley_obs"),
                    far_mass_pred=r.get("far_mass_pred"), valley_pred=r.get("valley_pred"),
                    degenerate=_degenerate(r.get("params") or {}, r.get("valley_pred")),
                    n_far=o.get("n_far")))
    return pd.DataFrame(rows)


def figure(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ok = df[~df.degenerate]           # well-conditioned fits only for the headline
    # collapse the two directions by averaging per subject (both held-out)
    g = (ok.groupby(["subject", "model"])
           .agg(valley_obs=("valley_obs", "mean"), valley_pred=("valley_pred", "mean"))
           .reset_index())
    # subjects with a degenerate fit for a given model (marked, not silently dropped)
    deg_subs = set(df[df.degenerate].subject.unique())

    # Panel A: held-out predicted valley-depth vs observed (identity = perfect)
    ax = axes[0]
    for m in ["basic_bayes", "switch"]:
        s = g[g.model == m]
        ax.scatter(s.valley_obs, s.valley_pred, s=70, color=COL[m], alpha=0.85,
                   edgecolor="white", linewidth=1.1, label=LABEL[m], zorder=3)
    lim = [0, 1]
    ax.plot(lim, lim, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    # correlation annotation per model (well-conditioned)
    txt = []
    for m in ["switch", "basic_bayes"]:
        s = g[g.model == m]
        r = np.corrcoef(s.valley_obs, s.valley_pred)[0, 1] if len(s) > 2 else float("nan")
        txt.append(f"{LABEL[m]}: r={r:.2f}")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5,
            color="#333")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("observed valley-depth (held-out half)")
    ax.set_ylabel("predicted valley-depth\n(params fit on OTHER half)")
    ax.set_title("A · Two-peak structure generalizes for Switch, not Basic-Bayes",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # Panel B: per-subject valley-depth, observed vs each model (paired)
    ax = axes[1]
    subs = sorted(df.subject.unique())
    x = np.arange(len(subs))
    obs = [ok[ok.subject == s].valley_obs.mean() for s in subs]
    ax.plot(x, obs, "o-", color=COL["observed"], lw=1.6, label="observed", zorder=4)
    for m in ["switch", "basic_bayes"]:
        y = [g[(g.subject == s) & (g.model == m)].valley_pred.mean()
             if len(g[(g.subject == s) & (g.model == m)]) else np.nan for s in subs]
        ax.plot(x, y, "o-", color=COL[m], lw=1.6, alpha=0.85, label=LABEL[m], zorder=3)
    # mark subjects with a degenerate fit
    for s in deg_subs:
        ax.axvspan(subs.index(s) - 0.35, subs.index(s) + 0.35, color="#f2d0d0",
                   alpha=0.5, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(subs, fontsize=8)
    ax.set_xlabel("subject (shaded = ill-conditioned split-half fit)")
    ax.set_ylabel("valley-depth (held-out)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("B · Held-out valley-depth per subject", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    df.to_csv(os.path.join(HERE, "split_half_summary.csv"), index=False)
    figure(df, os.path.join(HERE, "split_half_bimodality.png"))
    ndeg = int(df.degenerate.sum())
    deg_subs = sorted(df[df.degenerate].subject.unique())
    print(f"Fits: {len(df)} | degenerate (excluded from headline): {ndeg} "
          f"(subjects {deg_subs}: ill-conditioned single-start NM on halved data)\n")
    g = df[~df.degenerate].dropna(subset=["valley_pred", "valley_obs"])
    print("Mean held-out valley-depth (well-conditioned fits, subjects x directions):")
    print(f"  observed    : {g.valley_obs.mean():.3f}")
    for m in df.model.unique():
        s = g[g.model == m]
        r = np.corrcoef(s.valley_obs, s.valley_pred)[0, 1] if len(s) > 2 else float("nan")
        print(f"  {LABEL.get(m, m):11s}: pred {s.valley_pred.mean():.3f}  "
              f"(corr with observed r={r:.2f}, n={len(s)})")
    print(f"\nfar-band prior-cluster mass (held-out), mean:")
    for m in df.model.unique():
        s = g[g.model == m]
        print(f"  {LABEL.get(m, m):11s}: pred {s.far_mass_pred.mean():.3f}  "
              f"obs {s.far_mass_obs.mean():.3f}")


if __name__ == "__main__":
    main()
