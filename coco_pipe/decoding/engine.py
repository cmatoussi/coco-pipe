"""
Decoding Engine
===============
Functions for fitting, scoring, and metadata extraction.
"""

import logging
import time
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

from .constants import GROUP_CV_STRATEGIES
from .metrics import get_metric_spec
from .splitters import get_cv_splitter

logger = logging.getLogger(__name__)


def fit_and_score_fold(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    sample_ids: np.ndarray,
    sample_metadata: Optional[pd.DataFrame],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    metrics: list[str],
    feature_selection_config: Any,
    calibration_config: Any,
    feature_names: Optional[list[str]] = None,
    force_serial: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single Cross-Validation fold: Fit, Predict, and Score.
    Standalone function for parallel execution.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx] if groups is not None else None
    captured_warnings = []
    fit_time = np.nan
    predict_time = np.nan
    score_time = np.nan

    # 1. Fit
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        backend = (
            joblib.parallel_backend("sequential") if force_serial else nullcontext()
        )
        with backend:
            fit_estimator(
                estimator,
                X_train,
                y_train,
                groups_train,
                feature_selection_config=feature_selection_config,
                calibration_config=calibration_config,
            )
    fit_time = time.perf_counter() - fit_start
    captured_warnings.extend(warning_records_to_dict("fit", warning_records))

    # 2. Predict (Standard or Temporal)
    predict_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        y_pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - predict_start
    captured_warnings.extend(warning_records_to_dict("predict", warning_records))
    test_groups = groups[test_idx] if groups is not None else None
    fold_data = {
        "sample_index": test_idx,
        "sample_id": sample_ids[test_idx],
        "group": test_groups,
        "sample_metadata": metadata_slice(sample_metadata, test_idx),
        "y_true": y_test,
        "y_pred": y_pred,
    }

    # 3. Predict probabilities for prediction exports when available.
    if hasattr(estimator, "predict_proba"):
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                fold_data["y_proba"] = estimator.predict_proba(X_test)
            captured_warnings.extend(
                warning_records_to_dict("predict_proba", warning_records)
            )
        except Exception:
            pass
    if "y_proba" not in fold_data and hasattr(estimator, "decision_function"):
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                fold_data["y_score"] = estimator.decision_function(X_test)
            captured_warnings.extend(
                warning_records_to_dict("decision_function", warning_records)
            )
        except Exception:
            pass

    # 4. Extract Feature Importances
    imp = None
    try:
        imp = extract_feature_importances(estimator)
    except Exception:
        pass

    # 5. Compute Metrics
    scores = {}
    is_multiclass = type_of_target(y_test) == "multiclass"

    score_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        for metric_name in metrics:
            metric_spec = get_metric_spec(metric_name)
            scorer = metric_spec.scorer
            if metric_spec.response_method == "predict":
                y_est = y_pred
                is_proba = False
            else:
                y_est, is_proba = get_metric_response(
                    estimator,
                    X_test,
                    metric_name,
                    metric_spec.response_method,
                    is_multiclass,
                )

            try:
                val = compute_metric_safe(
                    scorer,
                    y_test,
                    y_est,
                    is_multiclass,
                    is_proba=is_proba,
                )
                scores[metric_name] = val
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' failed in CV fold: {e}")
                scores[metric_name] = np.nan
    captured_warnings.extend(warning_records_to_dict("score", warning_records))
    score_time = time.perf_counter() - score_start

    # 6. Extract Metadata (Best Params, Selected Features)
    meta = {}
    try:
        meta = extract_metadata(
            estimator,
            feature_selection_config=feature_selection_config,
            feature_names=feature_names,
        )
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")

    return {
        "test_idx": test_idx,
        "preds": fold_data,
        "scores": scores,
        "importance": imp,
        "metadata": meta,
        "split": split_record(
            train_idx,
            test_idx,
            sample_ids,
            groups,
            sample_metadata,
        ),
        "diagnostics": {
            "fit_time": fit_time,
            "predict_time": predict_time,
            "score_time": score_time,
            "total_time": fit_time + predict_time + score_time,
            "warnings": captured_warnings,
        },
    }


def fit_estimator(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: Optional[np.ndarray],
    feature_selection_config: Any,
    calibration_config: Any,
) -> None:
    """Fit estimators, routing groups only where configured CV needs them."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    calibrated = isinstance(estimator, CalibratedClassifierCV)
    fitted_estimator = estimator.estimator if calibrated else estimator
    search_cv = isinstance(fitted_estimator, (GridSearchCV, RandomizedSearchCV))

    # Determine if SFS needs groups
    route_groups = (
        groups_train is not None
        and feature_selection_config.enabled
        and feature_selection_config.method == "sfs"
        and feature_selection_config.cv.strategy in GROUP_CV_STRATEGIES
    )

    if (
        calibrated
        and groups_train is not None
        and calibration_config.cv.strategy in GROUP_CV_STRATEGIES
    ):
        estimator.cv = get_cv_splitter(
            calibration_config.cv,
            groups=groups_train,
        )

    pass_groups = groups_train is not None and (search_cv or route_groups)
    fit_kwargs = {"groups": groups_train} if pass_groups else {}

    if route_groups:
        with config_context(enable_metadata_routing=True):
            estimator.fit(X_train, y_train, **fit_kwargs)
    else:
        estimator.fit(X_train, y_train, **fit_kwargs)


def extract_feature_importances(estimator: BaseEstimator) -> Optional[np.ndarray]:
    """Extract feature importances or coefficients from a fitted estimator."""
    # 1. Unwrap fitted hyperparameter search objects.
    if hasattr(estimator, "best_estimator_"):
        return extract_feature_importances(estimator.best_estimator_)

    # 2. Unwrap Pipeline
    if isinstance(estimator, Pipeline):
        fs_step = estimator.named_steps.get("fs")
        clf_step = estimator.named_steps.get("clf")

        raw_imp = extract_feature_importances(clf_step)
        if raw_imp is None:
            return None

        if fs_step:
            support = fs_step.get_support()
            full_imp = np.zeros_like(support, dtype=float)
            full_imp[support] = raw_imp
            return full_imp

        return raw_imp

    # 3. Extract from Base Estimator
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    if hasattr(estimator, "coef_"):
        if estimator.coef_.ndim > 1:
            return np.mean(np.abs(estimator.coef_), axis=0)
        return np.abs(estimator.coef_)

    return None


def compute_metric_safe(scorer, y_true, y_est, is_multiclass, is_proba=False):
    """Compute metric handling standard and temporal (diagonal) shapes."""
    # 1. Detect temporal shapes
    # Standard: (samples,) or (samples, classes)
    # Sliding: (samples, times) or (samples, classes, times)
    # Generalizing: (samples, times, times) or (samples, classes, times, times)
    is_temporal = (
        (y_est.ndim == 2 and not is_proba and y_true.ndim == 1)
        or y_est.ndim == 3
        or (y_est.ndim == 4 and is_proba)
    )

    if not is_temporal:
        return _score_slice(scorer, y_true, y_est, is_multiclass, is_proba)

    # 2. Temporal Dispatch
    if y_est.ndim == 2:  # Sliding (labels)
        return np.array(
            [
                _score_slice(scorer, y_true, y_est[:, t], is_multiclass, False)
                for t in range(y_est.shape[1])
            ]
        )

    if y_est.ndim == 3:
        if not is_proba:  # Generalizing (labels)
            n_tr, n_te = y_est.shape[1], y_est.shape[2]
            flat = [
                _score_slice(scorer, y_true, y_est[:, tr, te], is_multiclass, False)
                for tr in range(n_tr)
                for te in range(n_te)
            ]
            return np.array(flat).reshape(n_tr, n_te)

        # Sliding (proba)
        n_times = y_est.shape[2]
        return np.array(
            [
                _score_slice(scorer, y_true, y_est[:, :, t], is_multiclass, True)
                for t in range(n_times)
            ]
        )

    if y_est.ndim == 4:  # Generalizing (proba)
        n_tr, n_te = y_est.shape[2], y_est.shape[3]
        flat = [
            _score_slice(scorer, y_true, y_est[:, :, tr, te], is_multiclass, True)
            for tr in range(n_tr)
            for te in range(n_te)
        ]
        return np.array(flat).reshape(n_tr, n_te)

    raise ValueError(f"Unsupported y_est shape: {y_est.shape}")


def _score_slice(scorer, y_true, y_est_slice, is_multiclass, is_proba):
    """Internal helper to score a single temporal slice."""
    if not is_proba:
        return float(scorer(y_true, y_est_slice))

    # Handle probability scaling for binary
    if not is_multiclass and y_est_slice.ndim == 2 and y_est_slice.shape[1] == 2:
        y_est_slice = y_est_slice[:, 1]

    kwargs = {"multi_class": "ovr"} if is_multiclass else {}
    return float(scorer(y_true, y_est_slice, **kwargs))


def warning_records_to_dict(
    stage: str, warning_records: Sequence[Any]
) -> list[Dict[str, Any]]:
    """Return serializable warning records captured in one fold stage."""
    return [
        {
            "stage": stage,
            "category": record.category.__name__,
            "message": str(record.message),
        }
        for record in warning_records
    ]


def split_record(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sample_ids: np.ndarray,
    groups: Optional[np.ndarray],
    sample_metadata: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Return sample context for one outer-CV split."""
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_sample_id": sample_ids[train_idx],
        "test_sample_id": sample_ids[test_idx],
        "train_group": groups[train_idx] if groups is not None else None,
        "test_group": groups[test_idx] if groups is not None else None,
        "train_metadata": metadata_slice(sample_metadata, train_idx),
        "test_metadata": metadata_slice(sample_metadata, test_idx),
    }


def metadata_slice(
    sample_metadata: Optional[pd.DataFrame],
    indices: np.ndarray,
) -> Optional[Dict[str, list[Any]]]:
    """Return serializable sample metadata rows for selected indices."""
    if sample_metadata is None:
        return None
    return sample_metadata.iloc[indices].to_dict(orient="list")


def extract_metadata(
    estimator: BaseEstimator,
    feature_selection_config: Any,
    feature_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Extract training metadata like best Hyperparameters and Selected Features."""
    meta = {}
    if hasattr(estimator, "best_params_"):
        meta["best_params"] = estimator.best_params_
        meta["best_score"] = estimator.best_score_
        meta["best_index"] = estimator.best_index_
        meta["search_results"] = compact_search_results(estimator)
        search_best = estimator.best_estimator_
    else:
        search_best = estimator

    if isinstance(search_best, Pipeline):
        fs_step = search_best.named_steps.get("fs")
        clf_step = search_best.named_steps.get("clf")
        if fs_step and hasattr(fs_step, "get_support"):
            mask = fs_step.get_support()
            indices = np.flatnonzero(mask)
            n_feat = len(mask)
            if feature_names is None or len(feature_names) != n_feat:
                actual_names = [f"feature_{idx}" for idx in range(n_feat)]
            else:
                actual_names = list(feature_names)

            meta["feature_selection_method"] = feature_selection_config.method
            meta["selected_features"] = mask
            meta["selected_feature_indices"] = indices
            meta["selected_feature_names"] = [actual_names[idx] for idx in indices]
            meta["feature_names"] = actual_names
            if feature_selection_config.method == "k_best":
                if hasattr(fs_step, "scores_"):
                    meta["feature_scores"] = fs_step.scores_
                if hasattr(fs_step, "pvalues_"):
                    meta["feature_pvalues"] = fs_step.pvalues_
            if hasattr(fs_step, "ranking_"):
                meta["selection_order"] = fs_step.ranking_
            elif hasattr(fs_step, "selection_order_"):
                meta["selection_order"] = fs_step.selection_order_
        if clf_step is not None and hasattr(clf_step, "get_artifact_metadata"):
            meta["artifacts"] = clf_step.get_artifact_metadata()
    elif hasattr(search_best, "get_artifact_metadata"):
        meta["artifacts"] = search_best.get_artifact_metadata()

    return meta


def compact_search_results(estimator: BaseEstimator) -> list[Dict[str, Any]]:
    """Return compact, serializable search diagnostics from cv_results_."""
    cv_results = getattr(estimator, "cv_results_", None)
    if not cv_results:
        return []

    params = cv_results.get("params", [])
    ranks = cv_results.get("rank_test_score")
    means = cv_results.get("mean_test_score")
    stds = cv_results.get("std_test_score")

    rows = []
    for idx, param_set in enumerate(params):
        row = {"candidate": idx, "params": dict(param_set)}
        if ranks is not None:
            row["rank_test_score"] = int(np.asarray(ranks)[idx])
        if means is not None:
            row["mean_test_score"] = float(np.asarray(means, dtype=float)[idx])
        if stds is not None:
            row["std_test_score"] = float(np.asarray(stds, dtype=float)[idx])
        rows.append(row)
    return rows


def get_metric_response(
    estimator: BaseEstimator,
    X_test: np.ndarray,
    metric_name: str,
    response_method: str,
    is_multiclass: bool,
) -> tuple[np.ndarray, bool]:
    """Return the estimator output required by a probability/ranking metric."""
    if response_method == "proba":
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(f"Metric '{metric_name}' requires predict_proba.")
        return estimator.predict_proba(X_test), True

    if response_method == "proba_or_score":
        if hasattr(estimator, "predict_proba"):
            try:
                return estimator.predict_proba(X_test), True
            except Exception:
                pass
        if hasattr(estimator, "decision_function") and not is_multiclass:
            return estimator.decision_function(X_test), False
        if hasattr(estimator, "decision_function") and is_multiclass:
            raise ValueError(
                f"Metric '{metric_name}' requires predict_proba for multiclass."
            )
        raise ValueError(
            f"Metric '{metric_name}' requires predict_proba or decision_function."
        )

    raise ValueError(
        f"Metric '{metric_name}' has unsupported response method '{response_method}'."
    )
