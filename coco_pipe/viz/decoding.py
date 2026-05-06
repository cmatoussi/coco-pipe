"""
Decoding Visualization
======================

Focused plotting helpers for decoding result tables.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _temporal_summary_frame(result_or_scores: Any) -> pd.DataFrame:
    if hasattr(result_or_scores, "get_temporal_score_summary"):
        return result_or_scores.get_temporal_score_summary()
    return pd.DataFrame(result_or_scores)


def _filter_temporal_summary(
    summary: pd.DataFrame,
    metric: Optional[str] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    frame = summary.copy()
    if metric is not None:
        frame = frame[frame["Metric"] == metric]
    if model is not None:
        frame = frame[frame["Model"] == model]
    return frame


def plot_temporal_score_curve(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot mean temporal decoding score curves.

    Parameters
    ----------
    result_or_scores : ExperimentResult or DataFrame-like
        Result object or output from ``get_temporal_score_summary()``.
    metric, model : str, optional
        Optional filters.
    title : str, optional
        Figure title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    """
    summary = _filter_temporal_summary(
        _temporal_summary_frame(result_or_scores), metric=metric, model=model
    )
    if summary.empty or "Time" not in summary:
        raise ValueError("No 1D temporal score rows available to plot.")

    curve_data = summary[summary["Time"].notna()].copy()
    if curve_data.empty:
        raise ValueError("No 1D temporal score rows available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for (model_name, metric_name), group in curve_data.groupby(["Model", "Metric"]):
        times = group["Time"].to_numpy()
        numeric_times = pd.api.types.is_numeric_dtype(group["Time"])
        x_vals = times if numeric_times else np.arange(len(times))
        label = f"{model_name} / {metric_name}"
        ax.plot(x_vals, group["Mean"], marker="o", linewidth=2, label=label)
        if "Std" in group:
            ax.fill_between(
                x_vals,
                group["Mean"] - group["Std"],
                group["Mean"] + group["Std"],
                alpha=0.15,
            )
        if not numeric_times:
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(value) for value in times], rotation=45)

    ax.set_title(title or "Temporal Decoding Score")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    return fig


def plot_temporal_generalization_matrix(
    result_or_scores: Any,
    metric: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot a train-time by test-time temporal generalization heatmap.

    Parameters
    ----------
    result_or_scores : ExperimentResult or DataFrame-like
        Result object or output from ``get_temporal_score_summary()``.
    metric, model : str, optional
        Optional filters. When omitted and multiple matrices are present, the
        first model/metric pair in the table is plotted.
    title : str, optional
        Figure title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    """
    summary = _filter_temporal_summary(
        _temporal_summary_frame(result_or_scores), metric=metric, model=model
    )
    required = {"TrainTime", "TestTime", "Mean"}
    if summary.empty or not required.issubset(summary.columns):
        raise ValueError("No temporal generalization matrix rows available to plot.")

    matrix_data = summary[
        summary["TrainTime"].notna() & summary["TestTime"].notna()
    ].copy()
    if matrix_data.empty:
        raise ValueError("No temporal generalization matrix rows available to plot.")

    first = matrix_data.iloc[0]
    matrix_data = matrix_data[
        (matrix_data["Model"] == first["Model"])
        & (matrix_data["Metric"] == first["Metric"])
    ]
    train_order = pd.unique(matrix_data["TrainTime"])
    test_order = pd.unique(matrix_data["TestTime"])
    matrix = matrix_data.pivot(index="TrainTime", columns="TestTime", values="Mean")
    matrix = matrix.reindex(index=train_order, columns=test_order)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(np.asarray(matrix), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(value) for value in matrix.columns], rotation=45)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(value) for value in matrix.index])
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Train Time")
    ax.set_title(title or f"{first['Model']} / {first['Metric']}")
    fig.colorbar(im, ax=ax, label="Score")
    return fig
