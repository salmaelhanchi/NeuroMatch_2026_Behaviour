from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
HIERARCHICAL_ROOT = HERE.parents[1]
sys.path.insert(0, str(HIERARCHICAL_ROOT))

from observers.models.hb_adaptive_confidence import HBAdaptiveConfidenceObserver


DATA_CSV = HIERARCHICAL_ROOT / "data" / "data01_direction4priors.csv"
FIT_DIR = HIERARCHICAL_ROOT / "results" / "fits" / "comparison" / "hb_adaptive"
OUT_DIR = HERE / "results"
PDF_PATH = HERE / "hb_adaptive_all_participants_learning_diagnostics.pdf"

PRIOR_MEAN = 225.0
COHERENCES = [0.06, 0.12, 0.24]
PRIOR_STDS = [10, 20, 40, 80]
PHASES = ["early", "middle", "late"]
PRIOR_BAND_COLORS = {
    10: "#e8f4fb",
    20: "#e7f4e4",
    40: "#fff1cc",
    80: "#f8dddd",
}
PRIOR_POINT_COLORS = {
    10: "#4e79a7",
    20: "#59a14f",
    40: "#f28e2b",
    80: "#e15759",
}
PRIOR_LINE_COLOR = "#111111"


def circdiff(angle, origin=PRIOR_MEAN):
    return (np.asarray(angle, dtype=float) - origin + 180.0) % 360.0 - 180.0


def circular_hist(values, bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def weighted_circular_hist(x, weights, bins):
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(weights)
    if not ok.any():
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(x[ok], bins=bins, weights=weights[ok])
    total = counts.sum()
    return counts / total if total else counts


def response_angle_from_xy(df):
    angle = np.degrees(np.arctan2(df["estimate_y"].to_numpy(), df["estimate_x"].to_numpy()))
    angle = angle % 360.0
    angle[angle == 0] = 360.0
    return angle


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df.sort_values(["subject_id", "session_id", "run_id", "trial_index"]).reset_index(drop=True)
    df["response_angle"] = response_angle_from_xy(df)
    valid = np.isfinite(df["response_angle"].to_numpy())
    estimate_dir = np.full(len(df), -1, dtype=int)
    rounded = np.rint(df.loc[valid, "response_angle"].to_numpy()).astype(int)
    rounded[rounded == 0] = 360
    estimate_dir[valid] = np.clip(rounded, 1, 360)
    df["estimate_dir"] = estimate_dir
    df["response_valid"] = valid
    df["block_key"] = (
        df["subject_id"].astype(str)
        + "_s"
        + df["session_id"].astype(str)
        + "_r"
        + df["run_id"].astype(str)
    )
    block_pos = df.groupby("block_key").cumcount()
    block_n = df.groupby("block_key")["trial_index"].transform("size")
    frac = (block_pos + 0.5) / block_n
    df["block_trial_position"] = block_pos + 1
    df["block_n_trials"] = block_n
    df["block_phase"] = np.select(
        [frac < 1.0 / 3.0, frac < 2.0 / 3.0],
        ["early", "middle"],
        default="late",
    )
    df["stimulus_from_prior"] = circdiff(df["motion_direction"])
    df["response_from_prior"] = circdiff(df["response_angle"])
    return df


def load_observer(subject_id: int):
    record = json.loads((FIT_DIR / f"subject{subject_id}.json").read_text())
    params = record["params"]
    k_like = {float(k): float(v) for k, v in params["k_like"].items()}
    obs = HBAdaptiveConfidenceObserver(
        k_like=k_like,
        k_motor=float(params["k_motor"]),
        p_random=float(params["p_random"]),
        lam=float(params["lam"]),
    )
    return obs, record


def replay_subject(subject_df, subject_id: int):
    obs, fit_record = load_observer(subject_id)
    directions = subject_df["motion_direction"].to_numpy(dtype=int)
    coherences = subject_df["motion_coherence"].to_numpy(dtype=float)
    out = obs.filter(directions, coherences, feedback=directions, record_belief=True)
    dists = np.asarray(out["dists"], dtype=float)
    state = subject_df[
        [
            "subject_id",
            "session_id",
            "run_id",
            "trial_index",
            "motion_direction",
            "motion_coherence",
            "prior_std",
            "block_key",
            "block_trial_position",
            "block_n_trials",
            "block_phase",
        ]
    ].copy()
    state["believed_sd"] = out["believed_sd"]
    state["believed_alpha"] = out["believed_alpha"]
    state["nll"] = fit_record["nll"]
    state["aic"] = fit_record["aic"]
    state["bic"] = fit_record["bic"]
    state["fit_converged"] = bool(fit_record.get("convergence", {}).get("converged", False))
    state["start_spread"] = float(fit_record.get("start_spread", np.nan))
    return obs, fit_record, dists, state


def block_alignment_table(df):
    rows = []
    for (sid, sess, run), g in df.groupby(["subject_id", "session_id", "run_id"], sort=True):
        priors = sorted(g["prior_std"].dropna().astype(int).unique().tolist())
        cohs = sorted(g["motion_coherence"].dropna().astype(float).unique().tolist())
        dirs = sorted(g["motion_direction"].dropna().astype(int).unique().tolist())
        rows.append(
            {
                "subject_id": int(sid),
                "session_id": int(sess),
                "run_id": int(run),
                "n_trials": int(len(g)),
                "prior_std_values": ";".join(map(str, priors)),
                "n_prior_std_values": len(priors),
                "coherence_values": ";".join(f"{c:g}" for c in cohs),
                "n_direction_values": len(dirs),
                "first_trial_index": int(g["trial_index"].iloc[0]),
                "last_trial_index": int(g["trial_index"].iloc[-1]),
                "first_motion_direction": int(g["motion_direction"].iloc[0]),
                "last_motion_direction": int(g["motion_direction"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def build_learning_summary(all_state):
    grouped = all_state.groupby(
        ["subject_id", "motion_coherence", "prior_std", "block_phase"], observed=True
    )
    return grouped.agg(
        n_trials=("trial_index", "size"),
        mean_believed_sd=("believed_sd", "mean"),
        mean_believed_alpha=("believed_alpha", "mean"),
        median_believed_sd=("believed_sd", "median"),
        median_believed_alpha=("believed_alpha", "median"),
    ).reset_index()


def ols_sse(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), X.shape[1]


def one_hot(values, drop_first=True):
    values = np.asarray(values)
    levels = list(pd.unique(values))
    start = 1 if drop_first else 0
    cols = []
    labels = []
    for level in levels[start:]:
        cols.append((values == level).astype(float))
        labels.append(str(level))
    if not cols:
        return np.empty((len(values), 0)), labels
    return np.column_stack(cols), labels


def anova_fixed_subject(summary, value_col):
    # One row per subject x coherence x prior_std, averaged over block phase.
    df = (
        summary.groupby(["subject_id", "motion_coherence", "prior_std"], observed=True)[value_col]
        .mean()
        .reset_index()
        .dropna()
    )
    y = df[value_col].to_numpy(dtype=float)
    intercept = np.ones((len(df), 1))
    subj, _ = one_hot(df["subject_id"].to_numpy())
    coh, _ = one_hot(df["motion_coherence"].to_numpy())
    prior, _ = one_hot(df["prior_std"].to_numpy())
    interaction_cols = []
    for i in range(coh.shape[1]):
        for j in range(prior.shape[1]):
            interaction_cols.append(coh[:, i] * prior[:, j])
    interaction = np.column_stack(interaction_cols) if interaction_cols else np.empty((len(df), 0))

    base = np.column_stack([intercept, subj])
    full_add = np.column_stack([base, coh, prior])
    full_int = np.column_stack([full_add, interaction])
    sse_full, p_full = ols_sse(y, full_add)
    df_resid = len(y) - p_full
    rows = []
    for effect, reduced in [
        ("coherence", np.column_stack([base, prior])),
        ("prior_std", np.column_stack([base, coh])),
    ]:
        sse_red, p_red = ols_sse(y, reduced)
        df_effect = p_full - p_red
        ms_effect = (sse_red - sse_full) / max(df_effect, 1)
        ms_error = sse_full / max(df_resid, 1)
        f_value = ms_effect / ms_error if ms_error > 0 else np.nan
        p_value = 1.0 - scipy_f_cdf(f_value, df_effect, df_resid) if np.isfinite(f_value) else np.nan
        rows.append(
            {
                "measure": value_col,
                "model": "subject_fixed + coherence + prior_std",
                "effect": effect,
                "df_effect": int(df_effect),
                "df_error": int(df_resid),
                "F": float(f_value),
                "p": float(p_value),
            }
        )

    sse_int, p_int = ols_sse(y, full_int)
    df_int_resid = len(y) - p_int
    df_effect = p_int - p_full
    ms_effect = (sse_full - sse_int) / max(df_effect, 1)
    ms_error = sse_int / max(df_int_resid, 1)
    f_value = ms_effect / ms_error if ms_error > 0 else np.nan
    p_value = 1.0 - scipy_f_cdf(f_value, df_effect, df_int_resid) if np.isfinite(f_value) else np.nan
    rows.append(
        {
            "measure": value_col,
            "model": "subject_fixed + coherence * prior_std",
            "effect": "coherence:prior_std",
            "df_effect": int(df_effect),
            "df_error": int(df_int_resid),
            "F": float(f_value),
            "p": float(p_value),
        }
    )
    return pd.DataFrame(rows)


def scipy_f_cdf(x, dfn, dfd):
    from scipy.stats import f

    return float(f.cdf(x, dfn, dfd))


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def add_prior_context_bands(ax, trial, prior_std, *, alpha=0.28):
    trial = np.asarray(trial, dtype=float)
    prior_std = np.asarray(prior_std, dtype=float)
    if trial.size == 0:
        return
    breaks = np.r_[0, np.flatnonzero(prior_std[1:] != prior_std[:-1]) + 1, prior_std.size]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        value = int(prior_std[start])
        ax.axvspan(
            trial[start] - 0.5,
            trial[stop - 1] + 0.5,
            color=PRIOR_BAND_COLORS.get(value, "#f1f1f1"),
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )


def plot_title_page(pdf, fit_rows, block_alignment, anova):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("HB-Adaptive model: learning diagnostics and data comparison", fontsize=16, y=0.96)
    ax = fig.add_subplot(111)
    ax.axis("off")
    n_subjects = fit_rows["subject"].nunique()
    converged = int(fit_rows["converged"].sum())
    text = [
        "Remote branch: model-verification",
        "Model: HB-AdaptiveConfidenceObserver (registry key: hb_adaptive)",
        "Core parameters fitted per subject: k_like[0.06], k_like[0.12], k_like[0.24], k_motor, p_random, lam.",
        "Latent learning replayed from feedback: joint belief b_t(kappa, alpha).",
        "",
        f"Subjects plotted: {n_subjects}; converged fits: {converged}/{n_subjects}.",
        f"Block alignment check: {int((block_alignment['n_prior_std_values'] > 1).sum())} session/run blocks contain more than one prior_std.",
        "",
        "Important interpretation note:",
        "The hidden belief update uses feedback directions, not coherence. Coherence changes the sensory likelihood and predictions.",
        "Therefore coherence-stratified learning plots show where trials of each coherence fall on the same learned trajectory.",
        "",
        "ANOVA-style fixed-subject tests are computed on subject means by coherence x prior_std.",
    ]
    ax.text(0.02, 0.92, "\n".join(text), va="top", ha="left", fontsize=11, linespacing=1.35)

    table = anova.copy()
    table["F"] = table["F"].map(lambda v: f"{v:.2f}")
    table["p"] = table["p"].map(lambda v: f"{v:.3g}")
    table = table[["measure", "effect", "df_effect", "df_error", "F", "p"]]
    ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.08, 0.96, 0.34],
    )
    pdf.savefig(fig)
    plt.close(fig)


def participant_page(pdf, subject_df, state_df, dists, fit_record):
    sid = int(subject_df["subject_id"].iloc[0])
    bins = np.linspace(-180, 180, 73)
    centers = 0.5 * (bins[:-1] + bins[1:])
    angle_bins = np.arange(1, 361)
    x_from_prior = circdiff(angle_bins)

    fig = plt.figure(figsize=(12, 10.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.15, 0.85, 1, 1, 1], hspace=0.5, wspace=0.28)
    fig.suptitle(
        f"Subject {sid}: HB-Adaptive predictions vs responses by block phase",
        fontsize=14,
        y=0.985,
    )
    ax_sd = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, :], sharex=ax_sd)

    trial = np.arange(1, len(state_df) + 1)
    add_prior_context_bands(ax_sd, trial, state_df["prior_std"].to_numpy(), alpha=0.34)
    ax_sd.plot(trial, state_df["believed_sd"], color="#30638e", lw=0.65, alpha=0.45, label="learned prior SD", zorder=2)
    ax_sd.plot(
        trial,
        pd.Series(state_df["believed_sd"]).rolling(35, min_periods=1).mean(),
        color="#003d5b",
        lw=1.8,
        label="35-trial mean",
        zorder=3,
    )
    ax_sd.step(
        trial,
        state_df["prior_std"],
        where="post",
        color=PRIOR_LINE_COLOR,
        lw=2.25,
        alpha=0.95,
        label="true block prior SD",
        zorder=4,
    )
    ax_sd.set_ylim(0, 86)
    ax_sd.set_yticks(PRIOR_STDS)
    ax_sd.set_ylabel("SD (deg)")
    ax_sd.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    ax_sd.grid(alpha=0.2)

    ax_a.plot(trial, state_df["believed_alpha"], color="#d1495b", lw=0.8, alpha=0.75, label="learned confidence alpha")
    ax_a.plot(
        trial,
        pd.Series(state_df["believed_alpha"]).rolling(25, min_periods=1).mean(),
        color="#8b1e3f",
        lw=1.6,
        label="25-trial mean",
    )
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.set_ylabel("alpha")
    ax_a.set_xlabel("chronological trial")
    ax_a.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    ax_a.grid(alpha=0.2)

    condition_specs = [
        ("far from prior", [65, 25], -160),
        ("near prior", [145, 305], -80),
        ("at prior mean", [225], 0),
    ]
    row_titles = ["early block trials", "middle block trials", "late block trials"]
    for r, phase in enumerate(PHASES):
        for c, (label, dirs, stim_x) in enumerate(condition_specs):
            ax = fig.add_subplot(gs[2 + r, c])
            mask = (
                (subject_df["block_phase"].to_numpy() == phase)
                & np.isclose(subject_df["motion_coherence"].to_numpy(float), 0.06)
                & subject_df["prior_std"].eq(80).to_numpy()
                & subject_df["motion_direction"].isin(dirs).to_numpy()
                & subject_df["response_valid"].to_numpy()
            )
            idx = np.flatnonzero(mask)
            obs_vals = []
            pred = np.zeros(len(bins) - 1)
            for i in idx:
                stim_delta = circdiff(subject_df["motion_direction"].iloc[i])
                sign = -1.0 if stim_delta > 0 else 1.0
                obs_vals.append(sign * subject_df["response_from_prior"].iloc[i])
                pred += weighted_circular_hist(sign * x_from_prior, dists[i], bins)
            obs_hist = circular_hist(obs_vals, bins)
            if idx.size:
                pred = pred / idx.size
            ax.fill_between(centers, obs_hist, color="#7b1e1e", alpha=0.82, label="data")
            ax.plot(centers, pred, color="#f28e2b", lw=1.8, label="HB-Adaptive")
            ax.axvline(0, color="#4e79a7", lw=1.2)
            ax.axvline(stim_x, color="0.45", lw=1.0, ls="--")
            ax.set_xlim(-180, 180)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(f"{label}\ncoh=0.06, prior SD=80, n={idx.size}", fontsize=9)
            else:
                ax.set_title(f"n={idx.size}", fontsize=9)
            if c == 0:
                ax.set_ylabel(row_titles[r])
            if r == 2:
                ax.set_xlabel("estimate - prior mean (deg)")
            ax.grid(alpha=0.16)
            if r == 0 and c == 2:
                ax.legend(loc="upper right", fontsize=7, frameon=False)

    status = fit_record.get("convergence", {})
    footer = (
        f"NLL={fit_record['nll']:.1f}, AIC={fit_record['aic']:.1f}, "
        f"converged={status.get('converged')}, start spread={fit_record.get('start_spread', np.nan):.1f}"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def summary_learning_pages(pdf, summary):
    colors = {"early": "#4e79a7", "middle": "#f28e2b", "late": "#59a14f"}
    for value_col, ylabel, title in [
        ("mean_believed_alpha", "learned confidence E[alpha]", "Learned prior confidence by prior width and coherence"),
        ("mean_believed_sd", "learned prior SD (deg)", "Learned prior width by prior width and coherence"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        fig.suptitle(title, fontsize=14)
        for ax, coh in zip(axes, COHERENCES):
            for phase in PHASES:
                rows = []
                for ps in PRIOR_STDS:
                    vals = summary[
                        np.isclose(summary["motion_coherence"], coh)
                        & summary["prior_std"].eq(ps)
                        & summary["block_phase"].eq(phase)
                    ][value_col]
                    rows.append((ps, vals.mean(), sem(vals)))
                x = np.array([r[0] for r in rows])
                y = np.array([r[1] for r in rows], dtype=float)
                e = np.array([r[2] for r in rows], dtype=float)
                ax.errorbar(x, y, yerr=e, marker="o", lw=1.8, capsize=3, color=colors[phase], label=phase)
            ax.set_title(f"coherence {coh:g}")
            ax.set_xticks(PRIOR_STDS)
            ax.set_xlabel("true prior SD (deg)")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel(ylabel)
        if value_col == "mean_believed_alpha":
            for ax in axes:
                ax.set_ylim(-0.03, 1.03)
        else:
            for ax in axes:
                ax.set_ylim(0, 86)
                ax.set_yticks(PRIOR_STDS)
        axes[-1].legend(frameon=False, fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def block_alignment_page(pdf, block_alignment):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), height_ratios=[1.1, 1.0])
    fig.suptitle("Block alignment diagnostic", fontsize=14)

    ax = axes[0]
    aligned = block_alignment.copy()
    aligned["block_number"] = aligned.groupby("subject_id").cumcount() + 1
    ok = aligned["n_prior_std_values"].eq(1)
    for prior_std in PRIOR_STDS:
        rows = aligned[ok & aligned["prior_std_values"].eq(str(prior_std))]
        ax.scatter(
            rows["block_number"],
            rows["subject_id"],
            s=58,
            color=PRIOR_POINT_COLORS[prior_std],
            edgecolor="white",
            linewidth=0.45,
            label=f"{prior_std} deg",
        )
    bad = aligned[~ok]
    if len(bad):
        ax.scatter(
            bad["block_number"],
            bad["subject_id"],
            s=72,
            marker="x",
            color="#d1495b",
            linewidth=1.5,
            label="mixed prior_std",
        )
    ax.set_ylabel("subject")
    ax.set_xlabel("chronological run block within subject")
    ax.set_title("Each point is one run block; color is the true prior SD")
    ax.set_yticks(sorted(aligned["subject_id"].unique()))
    ax.legend(title="prior SD", ncol=5, loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1]
    counts = block_alignment.groupby("n_prior_std_values").size()
    ax.bar([str(k) for k in counts.index], counts.values, color="#4e79a7")
    ax.set_xlabel("number of unique prior_std values inside a session/run block")
    ax.set_ylabel("block count")
    ax.set_title(
        "If any block has >1 unique prior_std, block definitions may be shifted or too broad"
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = load_data()
    subjects = sorted(data["subject_id"].dropna().astype(int).unique())
    all_states = []
    subject_payloads = {}
    fit_rows = []

    for sid in subjects:
        sdf = data[data["subject_id"].eq(sid)].reset_index(drop=True)
        obs, fit_record, dists, state = replay_subject(sdf, sid)
        all_states.append(state)
        subject_payloads[sid] = (sdf, dists, fit_record)
        fit_rows.append(
            {
                "subject": sid,
                "nll": fit_record["nll"],
                "aic": fit_record["aic"],
                "bic": fit_record["bic"],
                "converged": bool(fit_record.get("convergence", {}).get("converged", False)),
                "start_spread": float(fit_record.get("start_spread", np.nan)),
            }
        )
        print(f"replayed subject {sid}: {len(sdf)} trials")

    all_state = pd.concat(all_states, ignore_index=True)
    merged = data.merge(
        all_state[
            [
                "subject_id",
                "session_id",
                "run_id",
                "trial_index",
                "believed_sd",
                "believed_alpha",
            ]
        ],
        on=["subject_id", "session_id", "run_id", "trial_index"],
        how="left",
    )
    learning_summary = build_learning_summary(all_state)
    block_alignment = block_alignment_table(data)
    anova = pd.concat(
        [
            anova_fixed_subject(learning_summary, "mean_believed_alpha"),
            anova_fixed_subject(learning_summary, "mean_believed_sd"),
        ],
        ignore_index=True,
    )
    fit_df = pd.DataFrame(fit_rows)

    all_state.to_csv(OUT_DIR / "hb_adaptive_trial_learning_state.csv", index=False)
    merged.to_csv(OUT_DIR / "hb_adaptive_data_with_learning_state.csv", index=False)
    learning_summary.to_csv(OUT_DIR / "hb_adaptive_learning_by_subject_condition_phase.csv", index=False)
    block_alignment.to_csv(OUT_DIR / "hb_adaptive_block_alignment_summary.csv", index=False)
    anova.to_csv(OUT_DIR / "hb_adaptive_learning_anova_fixed_subject.csv", index=False)
    fit_df.to_csv(OUT_DIR / "hb_adaptive_fit_summary.csv", index=False)

    with PdfPages(PDF_PATH) as pdf:
        plot_title_page(pdf, fit_df, block_alignment, anova)
        summary_learning_pages(pdf, learning_summary)
        block_alignment_page(pdf, block_alignment)
        for sid in subjects:
            sdf, dists, fit_record = subject_payloads[sid]
            state = all_state[all_state["subject_id"].eq(sid)].reset_index(drop=True)
            participant_page(pdf, sdf, state, dists, fit_record)

    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()



