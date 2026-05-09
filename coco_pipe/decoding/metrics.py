"""
Decoding Metrics
================

Metric lookup for decoding experiments.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

MetricTask = Literal["classification", "regression"]
ResponseMethod = Literal["predict", "proba", "score", "proba_or_score"]
MetricFamily = Literal[
    "label",
    "score_probability",
    "threshold_sweep",
    "calibration",
    "confusion",
    "regression",
    "temporal",
]


@dataclass(frozen=True)
class MetricSpec:
    """Decoding metric metadata used for validation and estimator responses."""

    name: str
    task: MetricTask
    scorer: Callable
    response_method: ResponseMethod = "predict"
    family: MetricFamily = "label"
    greater_is_better: bool = True


def _specificity_score(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


METRIC_REGISTRY: dict[str, MetricSpec] = {
    # Classification from hard predictions
    "accuracy": MetricSpec("accuracy", "classification", accuracy_score),
    "balanced_accuracy": MetricSpec(
        "balanced_accuracy",
        "classification",
        balanced_accuracy_score,
        family="confusion",
    ),
    "f1": MetricSpec(
        "f1",
        "classification",
        lambda y, p: f1_score(y, p, average="weighted"),
        family="confusion",
    ),
    "f1_macro": MetricSpec(
        "f1_macro",
        "classification",
        lambda y, p: f1_score(y, p, average="macro"),
        family="confusion",
    ),
    "f1_micro": MetricSpec(
        "f1_micro",
        "classification",
        lambda y, p: f1_score(y, p, average="micro"),
        family="confusion",
    ),
    "precision": MetricSpec(
        "precision",
        "classification",
        lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
        family="confusion",
    ),
    "recall": MetricSpec(
        "recall",
        "classification",
        lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
        family="confusion",
    ),
    "sensitivity": MetricSpec(
        "sensitivity",
        "classification",
        lambda y, p: recall_score(y, p, pos_label=1, zero_division=0),
        family="confusion",
    ),
    "specificity": MetricSpec(
        "specificity",
        "classification",
        _specificity_score,
        family="confusion",
    ),
    "matthews_corrcoef": MetricSpec(
        "matthews_corrcoef",
        "classification",
        matthews_corrcoef,
        family="confusion",
    ),
    "cohen_kappa": MetricSpec(
        "cohen_kappa",
        "classification",
        cohen_kappa_score,
        family="confusion",
    ),
    # Classification from probabilities or scores
    "roc_auc": MetricSpec(
        "roc_auc",
        "classification",
        roc_auc_score,
        "proba_or_score",
        family="threshold_sweep",
    ),
    "roc_auc_ovr_weighted": MetricSpec(
        "roc_auc_ovr_weighted",
        "classification",
        lambda y, p, **kwargs: roc_auc_score(
            y, p, multi_class="ovr", average="weighted", **kwargs
        ),
        "proba",
        family="threshold_sweep",
    ),
    "average_precision": MetricSpec(
        "average_precision",
        "classification",
        average_precision_score,
        "proba_or_score",
        family="threshold_sweep",
    ),
    "pr_auc": MetricSpec(
        "pr_auc",
        "classification",
        average_precision_score,
        "proba_or_score",
        family="threshold_sweep",
    ),
    "log_loss": MetricSpec(
        "log_loss",
        "classification",
        log_loss,
        "proba",
        family="score_probability",
        greater_is_better=False,
    ),
    "brier_score": MetricSpec(
        "brier_score",
        "classification",
        brier_score_loss,
        "proba",
        family="calibration",
        greater_is_better=False,
    ),
    # Regression
    "r2": MetricSpec("r2", "regression", r2_score, family="regression"),
    "neg_mean_squared_error": MetricSpec(
        "neg_mean_squared_error",
        "regression",
        lambda y, p: -mean_squared_error(y, p),
        family="regression",
    ),
    "neg_mean_absolute_error": MetricSpec(
        "neg_mean_absolute_error",
        "regression",
        lambda y, p: -mean_absolute_error(y, p),
        family="regression",
    ),
    "explained_variance": MetricSpec(
        "explained_variance",
        "regression",
        explained_variance_score,
        family="regression",
    ),
    "neg_root_mean_squared_error": MetricSpec(
        "neg_root_mean_squared_error",
        "regression",
        lambda y, p: -float(np.sqrt(mean_squared_error(y, p))),
        family="regression",
    ),
}


def get_scorer(name: str) -> Callable:
    """
    Retrieve a decoding metric by name.

    Parameters
    ----------
    name : str
        Metric name, for example ``accuracy`` or ``neg_mean_squared_error``.

    Returns
    -------
    Callable
        Metric function with signature ``(y_true, y_pred) -> float``.
    """
    return get_metric_spec(name).scorer


def get_metric_spec(name: str) -> MetricSpec:
    """Return metric metadata for ``name``."""
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: "
            f"{sorted(list(METRIC_REGISTRY.keys()))}"
        )
    return METRIC_REGISTRY[name]


def get_metric_names(
    task: MetricTask | None = None,
    family: MetricFamily | None = None,
) -> list[str]:
    """Return known metric names, optionally filtered by task and family."""
    return sorted(
        name
        for name, spec in METRIC_REGISTRY.items()
        if (task is None or spec.task == task)
        and (family is None or spec.family == family)
    )


def get_metric_families(task: MetricTask | None = None) -> list[MetricFamily]:
    """Return known metric families, optionally filtered by task."""
    return sorted(
        {
            spec.family
            for spec in METRIC_REGISTRY.values()
            if task is None or spec.task == task
        }
    )
