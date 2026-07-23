from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
STATE_CSV = HERE / "results" / "hb_adaptive_trial_learning_state.csv"
PDF_PATH = HERE / "hb_adaptive_prior_width_vs_true_blocks.pdf"

PRIOR_STDS = [10, 20, 40, 80]
PRIOR_BAND_COLORS = {
    10: "#e8f4fb",
    20: "#e7f4e4",
    40: "#fff1cc",
    80: "#f8dddd",
}


def add_prior_bands(ax, trials: np.ndarray, prior_std: np.ndarray) -> None:
    breaks = np.r_[0, np.flatnonzero(prior_std[1:] != prior_std[:-1]) + 1, len(prior_std)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        value = int(prior_std[start])
        ax.axvspan(
            trials[start] - 0.5,
            trials[stop - 1] + 0.5,
            color=PRIOR_BAND_COLORS.get(value, "#f3f3f3"),
            alpha=0.34,
            linewidth=0,
            zorder=0,
        )


def plot_subject(ax, subject_state: pd.DataFrame) -> None:
    subject_state = subject_state.sort_values(
        ["session_id", "run_id", "trial_index"], kind="mergesort"
    ).reset_index(drop=True)
    sid = int(subject_state["subject_id"].iloc[0])
    trial = np.arange(1, len(subject_state) + 1)
    prior = subject_state["prior_std"].to_numpy(dtype=float)
    learned = subject_state["believed_sd"].to_numpy(dtype=float)
    rolling = pd.Series(learned).rolling(50, min_periods=1).mean().to_numpy()

    add_prior_bands(ax, trial, prior)
    ax.plot(
        trial,
        learned,
        color="#30638e",
        lw=0.55,
        alpha=0.35,
        label="learned prior SD, trial",
        zorder=2,
    )
    ax.plot(
        trial,
        rolling,
        color="#003d5b",
        lw=1.8,
        label="learned prior SD, 50-trial mean",
        zorder=3,
    )
    ax.step(
        trial,
        prior,
        where="post",
        color="#111111",
        lw=2.4,
        label="true block prior SD",
        zorder=4,
    )
    ax.set_title(f"Participant {sid}", loc="left", fontsize=10, pad=3)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 10, 20, 40, 80])
    ax.set_ylabel("prior SD (deg)")
    ax.grid(alpha=0.18)


def main() -> None:
    if not STATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing {STATE_CSV}. Run build_hb_adaptive_learning_pdf.py first."
        )
    state = pd.read_csv(STATE_CSV)
    subjects = sorted(state["subject_id"].dropna().astype(int).unique())
    with PdfPages(PDF_PATH) as pdf:
        for page_start in range(0, len(subjects), 4):
            page_subjects = subjects[page_start : page_start + 4]
            fig, axes = plt.subplots(
                len(page_subjects),
                1,
                figsize=(11, 8.5),
                sharex=False,
                squeeze=False,
            )
            fig.suptitle(
                "HB-Adaptive learned prior width compared with true block prior",
                fontsize=14,
                y=0.985,
            )
            for i, sid in enumerate(page_subjects):
                ax = axes[i, 0]
                plot_subject(ax, state[state["subject_id"].eq(sid)])
                if i == len(page_subjects) - 1:
                    ax.set_xlabel("chronological trial")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=3,
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
            )
            fig.tight_layout(rect=[0, 0.055, 1, 0.965])
            pdf.savefig(fig)
            plt.close(fig)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
