"""
Decoding Diagnostics & Tidy Data Helpers
========================================
Functions for expanding and tidying raw decoding results into DataFrames.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import get_metric_spec


def time_value(index: int, time_axis: Optional[Sequence[Any]]) -> Any:
    """Return time axis value for a given index."""
    if time_axis is None or index >= len(time_axis):
        return index
    return time_axis[index]


def score_rows(
    model: str,
    fold_idx: int,
    metric: str,
    score: Any,
    time_axis: Optional[Sequence[Any]] = None,
) -> list[Dict[str, Any]]:
    """Expand scalar or temporal fold scores into tidy rows."""
    score = np.asarray(score)
    rows = []

    if score.ndim == 0:
        return [
            {
                "Model": model,
                "Fold": fold_idx,
                "Metric": metric,
                "Value": float(score),
            }
        ]

    if score.ndim == 1:
        for t_idx, val in enumerate(score):
            rows.append(
                {
                    "Model": model,
                    "Fold": fold_idx,
                    "Metric": metric,
                    "Time": time_value(t_idx, time_axis),
                    "Value": val,
                }
            )
        return rows

    if score.ndim == 2:
        for t_tr in range(score.shape[0]):
            for t_te in range(score.shape[1]):
                rows.append(
                    {
                        "Model": model,
                        "Fold": fold_idx,
                        "Metric": metric,
                        "TrainTime": time_value(t_tr, time_axis),
                        "TestTime": time_value(t_te, time_axis),
                        "Value": score[t_tr, t_te],
                    }
                )
        return rows

    return [{"Model": model, "Fold": fold_idx, "Metric": metric, "Value": score}]


def prediction_rows(
    model: str,
    fold_idx: int,
    preds: Dict[str, Any],
    time_axis: Optional[Sequence[Any]] = None,
) -> list[Dict[str, Any]]:
    """Expand scalar or temporal predictions into tidy rows."""
    y_true = np.asarray(preds["y_true"])
    y_pred = np.asarray(preds["y_pred"])
    y_proba = np.asarray(preds["y_proba"]) if "y_proba" in preds else None
    y_score = np.asarray(preds["y_score"]) if "y_score" in preds else None
    n_samples = len(y_true)
    sample_index = np.asarray(preds.get("sample_index", np.arange(n_samples)))
    sample_id = np.asarray(preds.get("sample_id", sample_index))
    groups = optional_values(preds.get("group"), n_samples)
    metadata = preds.get("sample_metadata") or {}

    if y_pred.ndim == 2 and y_true.ndim == 1:
        return sliding_prediction_rows(
            model,
            fold_idx,
            y_true,
            y_pred,
            y_proba,
            sample_index,
            sample_id,
            groups,
            metadata,
            time_axis=time_axis,
        )

    if y_pred.ndim == 3 and y_true.ndim == 1:
        return generalizing_prediction_rows(
            model,
            fold_idx,
            y_true,
            y_pred,
            y_proba,
            sample_index,
            sample_id,
            groups,
            metadata,
            time_axis=time_axis,
        )

    return standard_prediction_rows(
        model,
        fold_idx,
        y_true,
        y_pred,
        y_proba,
        y_score,
        sample_index,
        sample_id,
        groups,
        metadata,
    )


def standard_prediction_rows(
    model: str,
    fold_idx: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    y_score: Optional[np.ndarray],
    sample_index: np.ndarray,
    sample_id: np.ndarray,
    groups: np.ndarray,
    metadata: Dict[str, Sequence[Any]],
) -> list[Dict[str, Any]]:
    """Columnar implementation of standard prediction expansion."""
    n_samples = len(y_true)
    data = {
        "Model": [model] * n_samples,
        "Fold": [fold_idx] * n_samples,
        "SampleIndex": sample_index,
        "SampleID": sample_id,
        "Group": groups,
        "y_true": [row_value(y_true, i) for i in range(n_samples)],
        "y_pred": [row_value(y_pred, i) for i in range(n_samples)],
    }
    for key, values in metadata.items():
        v_arr = np.asarray(values, dtype=object)
        data[metadata_display_name(key)] = v_arr[:n_samples]

    df = pd.DataFrame(data)

    if y_proba is not None:
        if y_proba.ndim == 1:
            df["y_proba"] = y_proba
        elif y_proba.ndim == 2:
            for c_idx in range(y_proba.shape[1]):
                df[f"y_proba_{c_idx}"] = y_proba[:, c_idx]

    if y_score is not None:
        if y_score.ndim == 1:
            df["y_score"] = y_score
        elif y_score.ndim == 2:
            for c_idx in range(y_score.shape[1]):
                df[f"y_score_{c_idx}"] = y_score[:, c_idx]

    return df.to_dict(orient="records")


def sliding_prediction_rows(
    model: str,
    fold_idx: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    sample_index: np.ndarray,
    sample_id: np.ndarray,
    groups: np.ndarray,
    metadata: Dict[str, Sequence[Any]],
    time_axis: Optional[Sequence[Any]] = None,
) -> list[Dict[str, Any]]:
    """Columnar implementation of sliding prediction expansion."""
    n_samples, n_times = y_pred.shape
    n_total = n_samples * n_times

    # 1. Build Full-Length Columns
    time_values = [time_value(t, time_axis) for t in range(n_times)]

    data = {
        "Model": [model] * n_total,
        "Fold": [fold_idx] * n_total,
        "SampleIndex": np.repeat(sample_index, n_times),
        "SampleID": np.repeat(sample_id, n_times),
        "Group": np.repeat(groups, n_times),
        "y_true": np.repeat([row_value(y_true, i) for i in range(n_samples)], n_times),
        "Time": np.tile(time_values, n_samples),
        "y_pred": y_pred.ravel(),
    }

    for key, values in metadata.items():
        v_arr = np.asarray(values, dtype=object)
        data[metadata_display_name(key)] = np.repeat(v_arr[:n_samples], n_times)

    # 2. Add probabilities
    if (
        y_proba is not None
        and y_proba.ndim == 3
        and y_proba.shape[0] == n_samples
        and y_proba.shape[2] == n_times
    ):
        for c_idx in range(y_proba.shape[1]):
            data[f"y_proba_{c_idx}"] = y_proba[:, c_idx, :].ravel()

    # 3. Final Frame
    return pd.DataFrame(data).to_dict(orient="records")


def generalizing_prediction_rows(
    model: str,
    fold_idx: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    sample_index: np.ndarray,
    sample_id: np.ndarray,
    groups: np.ndarray,
    metadata: Dict[str, Sequence[Any]],
    time_axis: Optional[Sequence[Any]] = None,
) -> list[Dict[str, Any]]:
    """Columnar implementation of generalizing prediction expansion."""
    n_samples, n_train, n_test = y_pred.shape
    n_exp = n_train * n_test
    n_total = n_samples * n_exp

    # 1. Build Full-Length Columns
    train_times = [time_value(t, time_axis) for t in range(n_train)]
    test_times = [time_value(t, time_axis) for t in range(n_test)]

    data = {
        "Model": [model] * n_total,
        "Fold": [fold_idx] * n_total,
        "SampleIndex": np.repeat(sample_index, n_exp),
        "SampleID": np.repeat(sample_id, n_exp),
        "Group": np.repeat(groups, n_exp),
        "y_true": np.repeat([row_value(y_true, i) for i in range(n_samples)], n_exp),
        "TrainTime": np.tile(np.repeat(train_times, n_test), n_samples),
        "TestTime": np.tile(np.tile(test_times, n_train), n_samples),
        "y_pred": y_pred.ravel(),
    }

    for key, values in metadata.items():
        v_arr = np.asarray(values, dtype=object)
        data[metadata_display_name(key)] = np.repeat(v_arr[:n_samples], n_exp)

    # 2. Add probabilities
    if (
        y_proba is not None
        and y_proba.ndim == 4
        and y_proba.shape[0] == n_samples
        and y_proba.shape[2] == n_train
        and y_proba.shape[3] == n_test
    ):
        for c_idx in range(y_proba.shape[1]):
            data[f"y_proba_{c_idx}"] = y_proba[:, c_idx, :, :].ravel()

    # 3. Final Frame
    return pd.DataFrame(data).to_dict(orient="records")


def prediction_base_row(
    model: str,
    fold_idx: int,
    row_idx: int,
    y_true: np.ndarray,
    sample_index: np.ndarray,
    sample_id: np.ndarray,
    groups: np.ndarray,
    metadata: Dict[str, Sequence[Any]],
) -> Dict[str, Any]:
    row = {
        "Model": model,
        "Fold": fold_idx,
        "SampleIndex": sample_index[row_idx],
        "SampleID": sample_id[row_idx],
        "Group": groups[row_idx],
        "y_true": row_value(y_true, row_idx),
    }
    add_metadata_columns(row, metadata, row_idx)
    return row


def row_value(values: np.ndarray, row_idx: int) -> Any:
    val = values[row_idx]
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def add_standard_proba(row: Dict[str, Any], y_proba: np.ndarray, row_idx: int):
    if y_proba.ndim == 1:
        row["y_proba"] = y_proba[row_idx]
    elif y_proba.ndim == 2:
        for c_idx in range(y_proba.shape[1]):
            row[f"y_proba_{c_idx}"] = y_proba[row_idx, c_idx]


def add_standard_score(row: Dict[str, Any], y_score: np.ndarray, row_idx: int):
    if y_score.ndim == 1:
        row["y_score"] = y_score[row_idx]
    elif y_score.ndim == 2:
        for c_idx in range(y_score.shape[1]):
            row[f"y_score_{c_idx}"] = y_score[row_idx, c_idx]


def add_metadata_columns(
    row: Dict[str, Any], metadata: Dict[str, Sequence[Any]], row_idx: int
) -> None:
    for key, values in metadata.items():
        v_arr = np.asarray(values, dtype=object)
        val = v_arr[row_idx] if row_idx < len(v_arr) else None
        row[metadata_display_name(key)] = val


def metadata_display_name(key: str) -> str:
    return {"subject": "Subject", "session": "Session", "site": "Site"}.get(key, key)


def optional_values(values: Optional[Any], length: int) -> np.ndarray:
    if values is None:
        return np.full(length, None, dtype=object)
    return np.asarray(values)


def proba_matrix(group: pd.DataFrame, n_classes: int) -> Optional[np.ndarray]:
    """Return a probability matrix from prediction rows when present."""
    cols = [f"y_proba_{idx}" for idx in range(n_classes)]
    if not all(c in group for c in cols):
        return None
    pb = group[cols]
    if pb.isna().any().any():
        return None
    return pb.to_numpy(dtype=float)


def feature_names_for_result(res: Dict[str, Any], n_features: int) -> list[str]:
    """Resolve feature names from result metadata or importances."""
    imp = res.get("importances")
    if imp:
        names = imp.get("feature_names")
        if names is not None and len(names) == n_features:
            return list(names)
    for m in res.get("metadata", []):
        names = m.get("feature_names")
        if names is not None and len(names) == n_features:
            return list(names)
    return [f"feature_{idx}" for idx in range(n_features)]


def unit_indices(group: pd.DataFrame, unit: str) -> list[np.ndarray]:
    """Return row-index arrays for bootstrap units."""
    if unit == "group" and "Group" in group and group["Group"].notna().any():
        unit_values = group["Group"].to_numpy()
    elif unit in {"sample", "epoch"}:
        unit_values = group["SampleID"].to_numpy()
    elif unit in {"subject", "session", "site"}:
        col = metadata_display_name(unit)
        if col in group and group[col].notna().any():
            unit_values = group[col].to_numpy()
        else:
            raise ValueError(f"unit='{unit}' requires a non-empty {col} column.")
    else:
        raise ValueError(
            "unit must be 'sample', 'epoch', 'group', 'subject', 'session', or 'site'."
        )

    return [np.flatnonzero(unit_values == v) for v in pd.unique(unit_values)]


def paired_unit_indices(merged: pd.DataFrame, unit: str) -> list[np.ndarray]:
    """Return row-index arrays for paired permutation units."""
    if unit == "group" and "Group_A" in merged and merged["Group_A"].notna().any():
        unit_values = merged["Group_A"].to_numpy()
    elif unit in {"sample", "epoch"}:
        unit_values = merged["SampleID"].to_numpy()
    elif unit in {"subject", "session", "site"}:
        col = f"{metadata_display_name(unit)}_A"
        if col in merged and merged[col].notna().any():
            unit_values = merged[col].to_numpy()
        else:
            raise ValueError(f"unit='{unit}' requires a non-empty {col} column.")
    else:
        raise ValueError(
            "unit must be 'sample', 'epoch', 'group', 'subject', 'session', or 'site'."
        )

    return [np.flatnonzero(unit_values == v) for v in pd.unique(unit_values)]


def resolve_pos_label(y_true: np.ndarray, pos_label: Optional[Any]) -> Any:
    """Resolve positive label for binary curve diagnostics."""
    if pos_label is not None:
        return pos_label
    # pd.unique does not sort; explicit sort ensures consistent label ordering
    labels = sorted(pd.unique(y_true).tolist())
    return labels[-1]


def score_frame(frame: pd.DataFrame, metric: str) -> float:
    """Score a tidy prediction frame using the specified metric."""
    metric_spec = get_metric_spec(metric)
    y_true = frame["y_true"].to_numpy()

    # 1. Label-based metrics
    if metric_spec.response_method == "predict":
        return float(metric_spec.scorer(y_true, frame["y_pred"].to_numpy()))

    # 2. Probability-based metrics
    proba_cols = sorted(
        [col for col in frame.columns if col.startswith("y_proba_")],
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    if metric_spec.response_method in {"proba", "proba_or_score"} and proba_cols:
        proba = frame[proba_cols].to_numpy(dtype=float)
        # pd.unique does not sort; explicit sort ensures consistent label ordering
        labels = sorted(pd.unique(y_true).tolist())
        if metric == "brier_score":
            if proba.shape[1] != 2:
                raise ValueError("brier_score supports binary classification only.")
            return float(metric_spec.scorer(y_true, proba[:, 1]))
        if metric in {"roc_auc", "average_precision", "pr_auc"} and proba.shape[1] == 2:
            return float(metric_spec.scorer(y_true, proba[:, 1]))
        if metric == "log_loss":
            return float(metric_spec.scorer(y_true, proba, labels=labels))
        # Default OVR for multiclass
        return float(metric_spec.scorer(y_true, proba, multi_class="ovr"))

    # 3. Decision-function based metrics
    if (
        metric_spec.response_method in {"score", "proba_or_score"}
        and "y_score" in frame
    ):
        return float(metric_spec.scorer(y_true, frame["y_score"].to_numpy(dtype=float)))

    raise ValueError(f"Metric '{metric}' cannot be scored from available predictions.")
