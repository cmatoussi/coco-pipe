"""
Finite-sample statistical assessment for decoding results.

This module separates descriptive performance from inferential claims. The
default inferential path reruns the complete decoding experiment under label
permutations so learned preprocessing, feature selection, tuning, and
calibration remain inside each null pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, false_discovery_control, norm

from .configs import StatisticalAssessmentConfig
from .metrics import get_metric_spec

if TYPE_CHECKING:
    from .result import ExperimentResult

logger = logging.getLogger(__name__)

TEMPORAL_COLUMNS = ["Time", "TrainTime", "TestTime"]


def resolve_unit_of_inference(
    config: StatisticalAssessmentConfig,
    groups: Optional[Sequence[Any]],
) -> str:
    """Return the configured inference unit, with grouped data defaulting high."""
    unit = config.unit_of_inference
    if unit is not None:
        return unit
    return "group_mean" if groups is not None else "sample"


def aggregate_predictions_for_inference(
    predictions: pd.DataFrame,
    metric: str,
    task: str = "classification",
    unit_of_inference: str = "sample",
    custom_unit_column: Optional[str] = None,
    custom_aggregation: str = "mean",
    require_single_prediction: bool = False,
) -> pd.DataFrame:
    """
    Aggregate prediction rows to independent units for inference.

    Parameters mirror ``StatisticalAssessmentConfig``. The output keeps
    temporal coordinate columns when present.
    """
    if predictions.empty:
        return predictions.copy()

    frame = predictions.copy()
    temporal_cols = [
        col for col in TEMPORAL_COLUMNS if col in frame and frame[col].notna().any()
    ]
    unit_col, aggregation = _resolve_unit_column(
        frame,
        unit_of_inference,
        custom_unit_column,
        custom_aggregation,
    )
    frame = frame.copy()
    frame["__unit"] = frame[unit_col]

    if unit_of_inference == "sample":
        duplicate_cols = ["__unit", *temporal_cols]
        if frame.duplicated(duplicate_cols).any():
            if require_single_prediction:
                raise ValueError(
                    "Analytical binomial tests require one held-out prediction "
                    "per independent unit."
                )
            raise ValueError(
                "sample-level inference requires one prediction per SampleID."
            )
        frame["InferentialUnitID"] = frame["__unit"]
        return frame.drop(columns=["__unit"])

    return _aggregate_by_unit(
        frame,
        temporal_cols,
        aggregation,
        task,
    )


def binomial_accuracy_test(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    p0: Optional[float],
    alpha: float = 0.05,
    ci_method: str = "wilson",
) -> dict[str, Any]:
    """
    Exact upper-tail binomial test for plain top-1 accuracy.

    Uses ``P(X >= k | n, p0)`` and returns the smallest chance threshold count
    whose upper-tail probability is at most ``alpha``.
    """
    if p0 is None:
        raise ValueError("Analytical binomial testing requires an explicit p0.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_eff = len(y_true)
    if n_eff == 0:
        raise ValueError("Cannot run a binomial test with zero predictions.")

    correct = y_true == y_pred
    k_correct = int(np.sum(correct))
    observed = k_correct / n_eff
    p_value = float(binom.sf(k_correct - 1, n_eff, p0))

    k_alpha = n_eff + 1
    for candidate in range(n_eff + 1):
        if binom.sf(candidate - 1, n_eff, p0) <= alpha:
            k_alpha = candidate
            break

    ci_lower, ci_upper = _accuracy_ci(k_correct, n_eff, alpha, ci_method)
    return {
        "observed": observed,
        "k_correct": k_correct,
        "n_eff": n_eff,
        "p_value": p_value,
        "chance_threshold": k_alpha / n_eff,
        "chance_threshold_count": k_alpha,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def run_statistical_assessment(
    observed_result: Any,
    experiment_config: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    feature_names: Optional[Sequence[str]],
    time_axis: Optional[np.ndarray],
    observation_level: str,
    inferential_unit: str,
) -> dict[str, Any]:
    """Run configured statistical assessment and return raw result payloads."""
    stats_config = experiment_config.evaluation
    unit = resolve_unit_of_inference(stats_config, groups)
    metrics = stats_config.metrics or list(experiment_config.metrics)
    rows: list[dict[str, Any]] = []
    nulls: dict[str, dict[str, Any]] = {}

    for model in observed_result.raw:
        if "error" in observed_result.raw[model]:
            continue
        model_predictions = observed_result.get_predictions()
        model_predictions = model_predictions[model_predictions["Model"] == model]
        for metric in metrics:
            method = _resolve_method(stats_config, metric)
            if method == "binomial":
                rows.extend(
                    _run_binomial_assessment(
                        model,
                        metric,
                        model_predictions,
                        experiment_config.task,
                        stats_config,
                        unit,
                    )
                )
                continue

            perm_rows, perm_null = _run_permutation_assessment(
                model,
                metric,
                observed_result,
                experiment_config,
                X,
                y,
                groups,
                sample_ids,
                sample_metadata,
                feature_names,
                time_axis,
                observation_level,
                inferential_unit,
                stats_config,
                unit,
            )
            rows.extend(perm_rows)
            if stats_config.chance.store_null_distribution and perm_null is not None:
                nulls.setdefault(model, {})[metric] = perm_null

    return {
        "rows": rows,
        "nulls": nulls,
        "meta": {
            "enabled": True,
            "method": stats_config.chance.method,
            "resolved_unit_of_inference": unit,
            "metrics": metrics,
            "n_permutations": stats_config.chance.n_permutations,
            "alpha": stats_config.confidence_intervals.alpha,
            "temporal_correction": stats_config.chance.temporal_correction,
        },
    }


def _run_binomial_assessment(
    model: str,
    metric: str,
    predictions: pd.DataFrame,
    task: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> list[dict[str, Any]]:
    if task != "classification" or metric != "accuracy":
        raise ValueError(
            "Analytical binomial testing only supports classification accuracy."
        )
    has_temporal_rows = any(
        col in predictions and predictions[col].notna().any()
        for col in TEMPORAL_COLUMNS
    )
    if has_temporal_rows:
        raise ValueError("Analytical binomial testing does not support temporal rows.")

    aggregated = aggregate_predictions_for_inference(
        predictions,
        metric=metric,
        task=task,
        unit_of_inference=unit,
        custom_unit_column=config.custom_unit_column,
        custom_aggregation=config.custom_aggregation,
        require_single_prediction=True,
    )
    p0 = config.chance.p0
    if p0 == "auto":
        n_classes = len(pd.unique(aggregated["y_true"]))
        p0 = 1.0 / n_classes

    result = binomial_accuracy_test(
        aggregated["y_true"],
        aggregated["y_pred"],
        p0=p0,
        alpha=config.confidence_intervals.alpha,
        ci_method=config.confidence_intervals.method,
    )
    return [
        {
            "Model": model,
            "Metric": metric,
            "Observed": result["observed"],
            "InferentialUnit": unit,
            "NEff": result["n_eff"],
            "NullMethod": "binomial",
            "NPermutations": None,
            "P0": p0,
            "PValue": result["p_value"],
            "CILower": result["ci_lower"],
            "CIUpper": result["ci_upper"],
            "CorrectionMethod": "none",
            "ChanceThreshold": result["chance_threshold"],
            "Time": None,
            "TrainTime": None,
            "TestTime": None,
            "Significant": result["p_value"] <= config.confidence_intervals.alpha,
            "Assumptions": "classification accuracy; one prediction per unit",
            "Caveat": "Analytical binomial test uses declared p0 only.",
        }
    ]


def _run_permutation_assessment(
    model: str,
    metric: str,
    observed_result: Any,
    experiment_config: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    feature_names: Optional[Sequence[str]],
    time_axis: Optional[np.ndarray],
    observation_level: str,
    inferential_unit: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    observed_predictions = observed_result.get_predictions()
    observed_predictions = observed_predictions[observed_predictions["Model"] == model]
    observed_agg = aggregate_predictions_for_inference(
        observed_predictions,
        metric=metric,
        task=experiment_config.task,
        unit_of_inference=unit,
        custom_unit_column=config.custom_unit_column,
        custom_aggregation=config.custom_aggregation,
    )
    observed_scores = _score_by_coordinates(observed_agg, metric)
    score_keys = list(observed_scores.keys())

    null_array = _run_permutation_loop(
        model=model,
        metric=metric,
        score_keys=score_keys,
        experiment_config=experiment_config,
        X=X,
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        sample_metadata=sample_metadata,
        feature_names=feature_names,
        time_axis=time_axis,
        observation_level=observation_level,
        inferential_unit=inferential_unit,
        config=config,
        unit=unit,
    )

    observed_array = np.asarray([observed_scores[key] for key in score_keys])
    rows = _build_permutation_rows(
        model=model,
        metric=metric,
        observed_array=observed_array,
        null_array=null_array,
        score_keys=score_keys,
        unit=unit,
        observed_agg=observed_agg,
        config=config,
        task=experiment_config.task,
    )

    null_payload = None
    if config.chance.store_null_distribution:
        null_payload = {
            "metric": metric,
            "unit": unit,
            "coordinates": [_coord_dict(key) for key in score_keys],
            "values": null_array,
        }
    return rows, null_payload


def _run_permutation_loop(
    model: str,
    metric: str,
    score_keys: list[tuple],
    experiment_config: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    feature_names: Optional[Sequence[str]],
    time_axis: Optional[np.ndarray],
    observation_level: str,
    inferential_unit: str,
    config: StatisticalAssessmentConfig,
    unit: str,
) -> np.ndarray:
    """Execute the full-pipeline permutation loop."""
    from .experiment import Experiment

    rng = np.random.default_rng(config.random_state)
    null_array = np.empty((config.chance.n_permutations, len(score_keys)), dtype=float)
    perm_config = _stats_disabled_config(experiment_config)

    for i in range(config.chance.n_permutations):
        y_perm = _permute_y_by_unit(
            y,
            groups,
            sample_metadata,
            unit,
            config.custom_unit_column,
            rng,
            experiment_config.task,
        )
        perm_result = Experiment(perm_config).run(
            X,
            y_perm,
            groups=groups,
            feature_names=feature_names,
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            observation_level=observation_level,
            inferential_unit=inferential_unit,
            time_axis=time_axis,
        )
        perm_predictions = perm_result.get_predictions()
        perm_predictions = perm_predictions[perm_predictions["Model"] == model]
        perm_agg = aggregate_predictions_for_inference(
            perm_predictions,
            metric=metric,
            task=experiment_config.task,
            unit_of_inference=unit,
            custom_unit_column=config.custom_unit_column,
            custom_aggregation=config.custom_aggregation,
        )
        perm_scores = _score_by_coordinates(perm_agg, metric)
        null_array[i] = [perm_scores[key] for key in score_keys]

    return null_array


def _build_permutation_rows(
    model: str,
    metric: str,
    observed_array: np.ndarray,
    null_array: np.ndarray,
    score_keys: list[tuple],
    unit: str,
    observed_agg: pd.DataFrame,
    config: StatisticalAssessmentConfig,
    task: str,
) -> list[dict[str, Any]]:
    """Assemble assessment rows from observed and null score arrays."""
    metric_spec = get_metric_spec(metric)
    p_values = _empirical_p_values(
        observed_array,
        null_array,
        greater_is_better=metric_spec.greater_is_better,
    )
    corrected = _correct_p_values(
        observed_array,
        null_array,
        p_values,
        config.chance.temporal_correction,
        metric_spec.greater_is_better,
    )

    rows = []
    lower = np.nanquantile(null_array, 0.025, axis=0)
    upper = np.nanquantile(null_array, 0.975, axis=0)
    for idx, key in enumerate(score_keys):
        coord = _coord_dict(key)
        rows.append(
            {
                "Model": model,
                "Metric": metric,
                "Observed": observed_array[idx],
                "InferentialUnit": unit,
                "NEff": _n_eff(observed_agg),
                "NullMethod": "permutation_full_pipeline",
                "NPermutations": config.chance.n_permutations,
                "P0": None,
                "PValue": p_values[idx],
                "CILower": lower[idx],
                "CIUpper": upper[idx],
                "CorrectionMethod": config.chance.temporal_correction,
                "CorrectedPValue": corrected[idx],
                "ChanceThreshold": None,
                "NullLower": lower[idx],
                "NullUpper": upper[idx],
                "Significant": corrected[idx] <= config.confidence_intervals.alpha,
                "Assumptions": (
                    "full outer-CV pipeline rerun under label permutations; "
                    "regression targets averaged by unit"
                    if task == "regression" and unit != "sample"
                    else "full outer-CV pipeline rerun under label permutations"
                ),
                "Caveat": _assessment_caveat(unit),
                **coord,
            }
        )
    return rows


def _resolve_method(config: StatisticalAssessmentConfig, metric: str) -> str:
    method = config.chance.method
    if method == "auto":
        if metric == "accuracy" and config.chance.p0 is not None:
            return "binomial"
        return "permutation"
    return method


def run_paired_permutation_assessment(
    results_a: "ExperimentResult",
    results_b: "ExperimentResult",
    model: str,
    metric: str,
    config: StatisticalAssessmentConfig,
) -> pd.DataFrame:
    """Run paired permutation test for difference between two results."""
    from .diagnostics import paired_unit_indices, score_frame

    preds_a = results_a.get_predictions()
    preds_b = results_b.get_predictions()
    preds_a = preds_a[preds_a["Model"] == model]
    preds_b = preds_b[preds_b["Model"] == model]

    # Align by SampleID/Fold/Time
    merge_cols = ["SampleID", "Fold"]
    temporal_cols = [c for c in ["Time", "TrainTime", "TestTime"] if c in preds_a]
    merge_cols.extend(temporal_cols)

    merged = pd.merge(preds_a, preds_b, on=merge_cols, suffixes=("_A", "_B"))
    if merged.empty:
        raise ValueError("Could not align predictions for paired test.")

    # Calculate observed difference
    unit = config.unit_of_inference
    observed_diffs = {}

    def get_diff(group: pd.DataFrame) -> float:
        score_a = score_frame(
            group.rename(columns=lambda x: x[:-2] if x.endswith("_A") else x), metric
        )
        score_b = score_frame(
            group.rename(columns=lambda x: x[:-2] if x.endswith("_B") else x), metric
        )
        return score_a - score_b

    for key, group in merged.groupby(
        temporal_cols if temporal_cols else [None], dropna=False
    ):
        if temporal_cols:
            k = (key,) if not isinstance(key, tuple) else key
        else:
            k = ()
        observed_diffs[k] = get_diff(group)

    score_keys = list(observed_diffs.keys())
    observed_array = np.array([observed_diffs[k] for k in score_keys])

    # Run Permutations
    rng = np.random.default_rng(config.random_state)
    unit_indices = paired_unit_indices(merged, unit)
    n_units = len(unit_indices)
    null_array = np.empty((config.chance.n_permutations, len(score_keys)))

    for i in range(config.n_permutations):
        # Flip signs randomly per unit
        flips = rng.choice([-1, 1], size=n_units)

        # Build permuted diffs
        # Since we are testing ScoreA - ScoreB, swapping labels is equivalent
        # to flipping sign of diff
        # swap A/B labels within each unit
        for k in score_keys:
            # This is a simplification; for complex metrics, we'd need to re-score
            # But for linear/additive metrics, we can flip.
            # To be robust, we should really swap the labels in the merged frame
            # and re-score.
            # But that's slow. Let's assume re-scoring is needed for rigor.
            pass

        # Robust implementation:
        swaps = flips == -1
        perm_merged = merged.copy()
        for u_idx in np.where(swaps)[0]:
            idx = unit_indices[u_idx]
            # Swap _A and _B columns
            for col in merged.columns:
                if col.endswith("_A"):
                    base = col[:-2]
                    col_b = f"{base}_B"
                    (
                        perm_merged.iloc[idx, perm_merged.columns.get_loc(col)],
                        perm_merged.iloc[idx, perm_merged.columns.get_loc(col_b)],
                    ) = (
                        merged.iloc[idx, merged.columns.get_loc(col_b)],
                        merged.iloc[idx, merged.columns.get_loc(col)],
                    )

        for k_idx, k in enumerate(score_keys):
            if temporal_cols:
                mask = np.ones(len(perm_merged), dtype=bool)
                for c_idx, c in enumerate(temporal_cols):
                    mask &= perm_merged[c] == k[c_idx]
                group = perm_merged[mask]
            else:
                group = perm_merged
            null_array[i, k_idx] = get_diff(group)

    p_values = _empirical_p_values(
        observed_array, null_array, greater_is_better=True, two_sided=True
    )
    corrected = _correct_p_values(
        observed_array,
        null_array,
        p_values,
        method=config.chance.temporal_correction,
        greater_is_better=True,
    )

    rows = []
    for idx, k in enumerate(score_keys):
        row = _coord_dict(k)
        row.update(
            {
                "Model": model,
                "Metric": metric,
                "Difference": observed_array[idx],
                "PValue": p_values[idx],
                "PValueCorrected": corrected[idx],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _stats_disabled_config(config: Any) -> Any:
    copied = config.model_copy(deep=True)
    copied.evaluation.enabled = False
    return copied


def _resolve_unit_column(
    frame: pd.DataFrame,
    unit: str,
    custom_unit_column: Optional[str],
    custom_aggregation: str,
) -> tuple[str, str]:
    if unit == "sample":
        return "SampleID", "identity"
    if unit in {"group_mean", "group_majority"}:
        if "Group" not in frame or frame["Group"].isna().all():
            raise ValueError(f"{unit} inference requires group labels.")
        return "Group", "mean" if unit == "group_mean" else "majority"
    if unit == "custom":
        if custom_unit_column is None:
            raise ValueError("custom unit inference requires custom_unit_column.")
        column = custom_unit_column
        if column not in frame:
            column = _metadata_display_name(custom_unit_column)
        if column not in frame:
            raise ValueError(f"custom unit column '{custom_unit_column}' is missing.")
        return column, custom_aggregation
    raise ValueError(f"Unknown unit_of_inference: {unit}.")


def _aggregate_by_unit(
    frame: pd.DataFrame,
    temporal_cols: list[str],
    aggregation: str,
    task: str,
) -> pd.DataFrame:
    if task != "classification" and aggregation == "majority":
        raise ValueError("majority aggregation is only valid for classification.")

    group_cols = ["__unit", *temporal_cols]
    proba_cols = sorted(
        [col for col in frame.columns if col.startswith("y_proba_")],
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )

    # 1. Validate y_true uniqueness per group
    if (frame.groupby(group_cols, dropna=False)["y_true"].nunique() > 1).any():
        raise ValueError(
            "Grouped inference requires one true target value per independent unit."
        )

    # 2. Build Aggregation Dictionary
    agg_dict = {"y_true": "first"}
    if task == "classification":
        if aggregation == "mean":
            if not proba_cols:
                raise ValueError(
                    "mean aggregation for classification requires probability columns."
                )
            for col in proba_cols:
                agg_dict[col] = "mean"
        elif aggregation == "majority":
            agg_dict["y_pred"] = lambda x: x.mode().iloc[0]
            if proba_cols:
                for col in proba_cols:
                    agg_dict[col] = "mean"
    else:  # regression
        agg_dict["y_pred"] = "mean"

    # 3. Aggregate
    res = frame.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    res = res.rename(columns={"__unit": "InferentialUnitID"})

    # 4. Resolve y_pred for classification mean-aggregation
    if task == "classification" and aggregation == "mean":
        labels = sorted(pd.unique(frame["y_true"]).tolist())
        probs = res[proba_cols].to_numpy()
        res["y_pred"] = [labels[idx] for idx in np.argmax(probs, axis=1)]

    return res


def _score_by_coordinates(
    frame: pd.DataFrame, metric: str
) -> dict[tuple[Any, ...], float]:
    from .diagnostics import score_frame

    temporal_cols = [
        col for col in TEMPORAL_COLUMNS if col in frame and frame[col].notna().any()
    ]
    if not temporal_cols:
        return {(): score_frame(frame, metric)}

    scores = {}
    for key, group in frame.groupby(temporal_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        scores[key] = score_frame(group, metric)
    return scores


def _empirical_p_values(
    observed: np.ndarray,
    null: np.ndarray,
    greater_is_better: bool,
    two_sided: bool = False,
) -> np.ndarray:
    if two_sided:
        # Proportion of abs(null) >= abs(observed).
        # Note: This symmetric two-sided test is standard for paired difference
        # permutations but can be anti-conservative for asymmetric null distributions.
        count = np.sum(np.abs(null) >= np.abs(observed)[None, :], axis=0)
        return (count + 1) / (null.shape[0] + 1)

    if greater_is_better:
        return (np.sum(null >= observed, axis=0) + 1) / (null.shape[0] + 1)
    return (np.sum(null <= observed, axis=0) + 1) / (null.shape[0] + 1)


def _correct_p_values(
    observed: np.ndarray,
    null: np.ndarray,
    p_values: np.ndarray,
    method: str,
    greater_is_better: bool,
) -> np.ndarray:
    if method == "none" or observed.size == 1:
        return p_values
    if method == "fdr_bh":
        return false_discovery_control(p_values, method="bh")
    if method == "max_stat":
        if greater_is_better:
            max_null = np.nanmax(null, axis=1)
            return (np.sum(max_null[:, None] >= observed[None, :], axis=0) + 1) / (
                null.shape[0] + 1
            )
        min_null = np.nanmin(null, axis=1)
        return (np.sum(min_null[:, None] <= observed[None, :], axis=0) + 1) / (
            null.shape[0] + 1
        )
    raise ValueError(f"Unknown temporal correction: {method}.")


def _permute_y_by_unit(
    y: np.ndarray,
    groups: Optional[np.ndarray],
    sample_metadata: Optional[pd.DataFrame],
    unit: str,
    custom_unit_column: Optional[str],
    rng: np.random.Generator,
    task: str,
) -> np.ndarray:
    """
    Permute labels by independent unit.

    Note
    ----
    For regression tasks, if multiple samples within a unit have different
    targets, the unit is assigned the mean target value before permutation.
    This preserves the exchangeability of independent units but may change the
    overall target distribution if unit targets are not uniform.
    """
    unit_values = _original_unit_values(
        len(y),
        groups,
        sample_metadata,
        unit,
        custom_unit_column,
    )
    unit_labels = []
    units = pd.unique(unit_values)
    varying_units = 0
    for value in units:
        unit_y = np.asarray(y)[unit_values == value]
        if task == "classification":
            labels = pd.unique(unit_y)
            if len(labels) != 1:
                raise ValueError(
                    "Grouped label permutations require one class label per "
                    "independent unit."
                )
            unit_labels.append(labels[0])
        else:
            targets = np.asarray(unit_y, dtype=float)
            if len(np.unique(targets)) > 1:
                varying_units += 1
            unit_labels.append(float(np.mean(targets)))

    if varying_units > 0:
        logger.warning(
            f"Regression targets vary within {varying_units}/{len(units)} units. "
            "Independent units were assigned their mean target value before "
            "permutation. This may shift the target distribution if units are "
            "not balanced."
        )
    permuted = rng.permutation(np.asarray(unit_labels, dtype=object))
    mapping = dict(zip(units, permuted))
    return np.asarray([mapping[value] for value in unit_values])


def _original_unit_values(
    n_samples: int,
    groups: Optional[np.ndarray],
    sample_metadata: Optional[pd.DataFrame],
    unit: str,
    custom_unit_column: Optional[str],
) -> np.ndarray:
    if unit == "sample":
        return np.arange(n_samples)
    if unit in {"group_mean", "group_majority"}:
        if groups is None:
            raise ValueError(f"{unit} inference requires groups.")
        return np.asarray(groups)
    if unit == "custom":
        if custom_unit_column is None or sample_metadata is None:
            raise ValueError("custom unit inference requires sample_metadata.")
        if custom_unit_column not in sample_metadata:
            raise ValueError(f"custom unit column '{custom_unit_column}' is missing.")
        return sample_metadata[custom_unit_column].to_numpy()
    raise ValueError(f"Unknown unit_of_inference: {unit}.")


def _accuracy_ci(
    k_correct: int,
    n_eff: int,
    alpha: float,
    method: str,
) -> tuple[float, float]:
    if method == "clopper_pearson":
        if k_correct == 0:
            lower = 0.0
        else:
            lower = beta.ppf(alpha / 2, k_correct, n_eff - k_correct + 1)
        if k_correct == n_eff:
            upper = 1.0
        else:
            upper = beta.ppf(1 - alpha / 2, k_correct + 1, n_eff - k_correct)
        return float(lower), float(upper)
    if method != "wilson":
        raise ValueError("ci_method must be 'wilson' or 'clopper_pearson'.")
    z = norm.ppf(1 - alpha / 2)
    phat = k_correct / n_eff
    denom = 1 + z**2 / n_eff
    center = (phat + z**2 / (2 * n_eff)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n_eff)) / n_eff) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _coord_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    if len(key) == 0:
        return {"Time": None, "TrainTime": None, "TestTime": None}
    if len(key) == 1:
        return {"Time": key[0], "TrainTime": None, "TestTime": None}
    return {"Time": None, "TrainTime": key[0], "TestTime": key[1]}


def _n_eff(frame: pd.DataFrame) -> int:
    if "InferentialUnitID" in frame:
        return int(frame["InferentialUnitID"].nunique())
    return int(len(frame))


def _assessment_caveat(unit: str) -> str:
    if unit == "sample":
        return "Inference treats each sample as an independent unit."
    if unit.startswith("group"):
        return "Epoch-level predictions were aggregated to group-level units."
    return "Inference used a custom metadata-defined independent unit."


def _metadata_display_name(key: str) -> str:
    return {"subject": "Subject", "session": "Session", "site": "Site"}.get(key, key)
