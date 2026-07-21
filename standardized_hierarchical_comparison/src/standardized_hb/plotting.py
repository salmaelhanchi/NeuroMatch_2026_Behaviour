"""Comparison plots using the same bins and trial selectors for every model."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import GridSpec, PredictionResult
from .circular import angle_to_bin, circular_difference_degrees, wrap_signed_degrees
from .data import PreparedSubject


MODEL_COLORS = {
    "readout_average": "#3f6b9d",
    "reliability_mixture": "#b2513f",
    "integrated_prior": "#27856f",
    "switching_observer": "#8c6d31",
}


def _relative_curve(pmf: np.ndarray, reference_degrees: float, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    reference_bin = int(angle_to_bin(reference_degrees, grid.n_angles))
    shifted = np.roll(np.asarray(pmf, dtype=float), -reference_bin)
    relative = wrap_signed_degrees(grid.theta_degrees)
    order = np.argsort(relative)
    return relative[order], shifted[order]


def _binned_relative_curve(
    pmf: np.ndarray,
    reference_degrees: float,
    grid: GridSpec,
    bin_width_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate grid probabilities into wider circular bins."""

    grid_step = 360.0 / grid.n_angles
    bins_per_group = int(round(bin_width_degrees / grid_step))
    if bins_per_group < 1 or not np.isclose(bins_per_group * grid_step, bin_width_degrees):
        raise ValueError("bin_width_degrees must be an integer multiple of the grid step.")
    if grid.n_angles % bins_per_group:
        raise ValueError("The requested bin width must divide the circular response grid.")

    relative, probability = _relative_curve(pmf, reference_degrees, grid)
    grouped_relative = relative.reshape(-1, bins_per_group).mean(axis=1)
    grouped_probability = probability.reshape(-1, bins_per_group).sum(axis=1)
    return grouped_relative, grouped_probability


def _observed_absolute_pmf(subject: PreparedSubject, selector: np.ndarray, grid: GridSpec) -> np.ndarray:
    valid = selector & subject.response_valid
    counts = np.bincount(subject.response_bins[valid], minlength=grid.n_angles).astype(float)
    return counts / counts.sum()


def diagnostic_selector(subject: PreparedSubject, prior_mean: float) -> tuple[np.ndarray, float, float]:
    coherence = float(subject.coherence_values.min())
    eligible = (
        (subject.coherence_indices == int(np.argmin(subject.coherence_values)))
        & (np.abs(circular_difference_degrees(subject.directions, prior_mean)) >= 60.0)
        & subject.response_valid
    )
    directions, counts = np.unique(subject.directions[eligible], return_counts=True)
    if directions.size == 0:
        raise ValueError("No low-coherence, far-from-prior diagnostic condition exists.")
    direction = float(directions[np.argmax(counts)])
    return (
        np.isclose(subject.directions, direction)
        & (subject.coherence_indices == int(np.argmin(subject.coherence_values))),
        direction,
        coherence,
    )


def comparison_figure(
    subject: PreparedSubject,
    grid: GridSpec,
    predictions: Mapping[str, PredictionResult],
    score_table: pd.DataFrame,
):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    ordered = score_table.sort_values("aic").reset_index(drop=True)
    delta = ordered["aic"] - ordered["aic"].min()
    axes[0].bar(
        ordered["model"],
        delta,
        color=[MODEL_COLORS[name] for name in ordered["model"]],
    )
    axes[0].set(
        title=f"Participant {subject.subject_id} model comparison",
        ylabel="Delta AIC",
    )
    axes[0].tick_params(axis="x", rotation=25)

    all_trials = np.ones(subject.n_trials, dtype=bool)
    observed = _observed_absolute_pmf(subject, all_trials, grid)
    x, y = _relative_curve(observed, grid.prior_mean_degrees, grid)
    axes[1].plot(x, y, color="#222222", linewidth=2.2, label="observed")
    for name, result in predictions.items():
        x, y = _relative_curve(result.response_pmfs.mean(axis=0), grid.prior_mean_degrees, grid)
        axes[1].plot(x, y, color=MODEL_COLORS[name], linewidth=1.8, label=name)
    axes[1].set(
        title="All responses",
        xlabel="Response relative to 225 degrees",
        ylabel="Probability mass",
    )
    axes[1].legend(frameon=False, fontsize=8)

    selector, direction, coherence = diagnostic_selector(subject, grid.prior_mean_degrees)
    observed = _observed_absolute_pmf(subject, selector, grid)
    x, y = _relative_curve(observed, grid.prior_mean_degrees, grid)
    axes[2].plot(x, y, color="#222222", linewidth=2.2, label="observed")
    for name, result in predictions.items():
        x, y = _relative_curve(result.response_pmfs[selector].mean(axis=0), grid.prior_mean_degrees, grid)
        axes[2].plot(x, y, color=MODEL_COLORS[name], linewidth=1.8, label=name)
    stimulus_relative = float(circular_difference_degrees(direction, grid.prior_mean_degrees))
    axes[2].axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[2].axvline(stimulus_relative, color="#777777", linestyle=":", linewidth=1.0)
    axes[2].set(
        title=f"Diagnostic: {coherence:g} coherence, direction {direction:g}",
        xlabel="Response relative to 225 degrees",
        ylabel="Probability mass",
    )
    return figure


def score_and_response_figure(
    subject: PreparedSubject,
    grid: GridSpec,
    predictions: Mapping[str, PredictionResult],
    score_table: pd.DataFrame,
):
    """Compare scores and marginal responses without choosing a diagnostic condition."""

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    ordered = score_table.sort_values("aic").reset_index(drop=True)
    delta = ordered["aic"] - ordered["aic"].min()
    axes[0].bar(
        ordered["model"],
        delta,
        color=[MODEL_COLORS[name] for name in ordered["model"]],
    )
    converged = (
        bool(ordered["optimizer_success"].all())
        if "optimizer_success" in ordered
        else True
    )
    comparison_label = "model comparison" if converged else "provisional scores: fit not converged"
    axes[0].set(
        title=f"Participant {subject.subject_id} {comparison_label}",
        ylabel="Delta AIC",
    )
    axes[0].tick_params(axis="x", rotation=25)

    all_trials = np.ones(subject.n_trials, dtype=bool)
    observed = _observed_absolute_pmf(subject, all_trials, grid)
    x, y = _relative_curve(observed, grid.prior_mean_degrees, grid)
    axes[1].plot(x, y, color="#222222", linewidth=2.2, label="observed")
    for name, result in predictions.items():
        x, y = _relative_curve(result.response_pmfs.mean(axis=0), grid.prior_mean_degrees, grid)
        axes[1].plot(x, y, color=MODEL_COLORS[name], linewidth=1.8, label=name)
    axes[1].set(
        title="All responses",
        xlabel=f"Response relative to {grid.prior_mean_degrees:g} degrees",
        ylabel="Probability mass",
    )
    axes[1].legend(frameon=False, fontsize=8)
    return figure


def paper_bimodality_figure(
    subject: PreparedSubject,
    grid: GridSpec,
    predictions: Mapping[str, PredictionResult],
    selectors: Mapping[float, np.ndarray],
    *,
    coherence: float = 0.06,
    prior_std: float = 80.0,
    bin_width_degrees: float = 15.0,
):
    """Plot paper-matched direction conditions without exposing metadata to models."""

    if not selectors:
        raise ValueError("At least one diagnostic direction selector is required.")
    directions = sorted(float(direction) for direction in selectors)
    figure, axes = plt.subplots(
        1,
        len(directions),
        figsize=(3.35 * len(directions), 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, direction in zip(axes, directions):
        selector = np.asarray(selectors[direction], dtype=bool)
        if selector.shape != (subject.n_trials,):
            raise ValueError(f"Selector for direction {direction:g} has the wrong shape.")
        valid_count = int(np.count_nonzero(selector & subject.response_valid))
        if valid_count == 0:
            raise ValueError(f"No valid responses exist for diagnostic direction {direction:g}.")

        observed = _observed_absolute_pmf(subject, selector, grid)
        x, y = _binned_relative_curve(
            observed,
            grid.prior_mean_degrees,
            grid,
            bin_width_degrees,
        )
        axis.plot(
            x,
            y,
            color="#222222",
            linewidth=2.2,
            drawstyle="steps-mid",
            label="observed",
        )
        for name, result in predictions.items():
            x, y = _binned_relative_curve(
                result.response_pmfs[selector].mean(axis=0),
                grid.prior_mean_degrees,
                grid,
                bin_width_degrees,
            )
            axis.plot(x, y, color=MODEL_COLORS[name], linewidth=1.8, label=name)

        stimulus_relative = float(circular_difference_degrees(direction, grid.prior_mean_degrees))
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        axis.axvline(stimulus_relative, color="#777777", linestyle=":", linewidth=1.0)
        axis.set(
            title=f"Direction {direction:g} degrees (n={valid_count})",
            xlabel="Response relative to prior (degrees)",
            xlim=(-180.0, 180.0),
        )
        axis.set_xticks([-150, -75, 0, 75, 150])

    axes[0].set_ylabel(f"Probability per {bin_width_degrees:g}-degree bin")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Participant {subject.subject_id}: paper-matched bimodality diagnostic "
        f"(coherence {coherence:g}, prior SD {prior_std:g} degrees)"
    )
    return figure


def state_figure(
    subject: PreparedSubject,
    predictions: Mapping[str, PredictionResult],
    smoothing_window: int = 75,
):
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    session_starts = np.flatnonzero(subject.reset_before)
    kernel = np.ones(smoothing_window, dtype=float) / smoothing_window
    for axis, (name, result) in zip(axes, predictions.items()):
        state = np.asarray(result.state, dtype=float)
        smooth = np.convolve(state, kernel, mode="same")
        axis.plot(state, color=MODEL_COLORS[name], alpha=0.22, linewidth=0.7)
        axis.plot(smooth, color=MODEL_COLORS[name], linewidth=1.8)
        for boundary in session_starts[1:]:
            axis.axvline(boundary, color="#b5b5b5", linewidth=0.6)
        axis.set(title=name, ylabel=result.state_label)
    axes[-1].set_xlabel(f"Chronological participant-{subject.subject_id} trial")
    return figure
