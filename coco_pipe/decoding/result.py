from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .constants import RESULT_SCHEMA_VERSION
from .diagnostics import (
    feature_names_for_result,
    paired_unit_indices,
    prediction_rows,
    proba_matrix,
    resolve_pos_label,
    score_rows,
    unit_indices,
)
from .metrics import get_metric_spec

logger = logging.getLogger(__name__)


class ExperimentResult:
    """
    Unified Container for Experiment Results.
    Provides Tidy Data views for easier analysis.
    """

    def __init__(
        self,
        raw_results: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        schema_version: str = RESULT_SCHEMA_VERSION,
    ):
        self.raw = raw_results
        self.config = config or {}
        self.meta = meta or {}
        self.schema_version = schema_version

    def to_payload(self) -> Dict[str, Any]:
        """Return the serializable decoding result payload."""
        return {
            "schema_version": self.schema_version,
            "config": self.config,
            "meta": self.meta,
            "results": self.raw,
        }

    def save_json(self, path: Union[str, Path, Any], indent: int = 2):
        """Save results to a JSON file (standard-compliant, cross-version safe)."""
        import json

        payload = self.to_payload()

        def _to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.int16)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {str(k): _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj

        with open(path, "w") as f:
            json.dump(_to_serializable(payload), f, indent=indent)

    @classmethod
    def load_json(cls, path: Union[str, Path, Any]) -> "ExperimentResult":
        """Load results from a JSON file."""
        import json

        with open(path, "r") as f:
            payload = json.load(f)
        return cls(
            raw_results=payload["results"],
            config=payload.get("config"),
            meta=payload.get("meta"),
            schema_version=payload.get("schema_version", RESULT_SCHEMA_VERSION),
        )

    def summary(self) -> pd.DataFrame:
        """Get a high-level summary of performance (Mean/Std across folds)."""
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
            row = {"Model": model}
            for metric, stats in res["metrics"].items():
                mean = np.asarray(stats["mean"])
                std = np.asarray(stats["std"])
                if mean.ndim == 0 and std.ndim == 0:
                    row[f"{metric}_mean"] = float(mean)
                    row[f"{metric}_std"] = float(std)
            if len(row) > 1:
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("Model")

    def get_detailed_scores(self) -> pd.DataFrame:
        """Get fold-level scores for all models in long format."""
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
            metrics_data = res["metrics"]
            n_folds = len(next(iter(metrics_data.values()))["folds"])
            for fold_idx in range(n_folds):
                for metric, stats in metrics_data.items():
                    rows.extend(
                        score_rows(
                            model,
                            fold_idx,
                            metric,
                            stats["folds"][fold_idx],
                            time_axis=self._time_axis(),
                        )
                    )
        return pd.DataFrame(rows)

    def get_temporal_score_summary(self) -> pd.DataFrame:
        """Get temporal metric means/stds across folds in long format."""
        rows = []
        columns = ["Model", "Metric", "Time", "TrainTime", "TestTime", "Mean", "Std"]

        for model, res in self.raw.items():
            if "error" in res:
                continue
            for metric, stats in res.get("metrics", {}).items():
                folds = [np.asarray(fold) for fold in stats.get("folds", [])]
                if not folds or folds[0].ndim == 0:
                    continue
                stack = np.stack(folds)
                mean = np.nanmean(stack, axis=0)
                std = np.nanstd(stack, axis=0)

                if mean.ndim == 1:
                    for t_idx, val in enumerate(mean):
                        rows.append(
                            {
                                "Model": model,
                                "Metric": metric,
                                "Time": self._time_value(t_idx),
                                "Mean": val,
                                "Std": std[t_idx],
                            }
                        )
                elif mean.ndim == 2:
                    for t_tr in range(mean.shape[0]):
                        for t_te in range(mean.shape[1]):
                            rows.append(
                                {
                                    "Model": model,
                                    "Metric": metric,
                                    "TrainTime": self._time_value(t_tr),
                                    "TestTime": self._time_value(t_te),
                                    "Mean": mean[t_tr, t_te],
                                    "Std": std[t_tr, t_te],
                                }
                            )
        return pd.DataFrame(rows, columns=columns)

    def get_predictions(self) -> pd.DataFrame:
        """Get concatenated predictions for all models."""
        rows = []
        time_axis = self._time_axis()
        for model, res in self.raw.items():
            if "error" in res:
                continue
            for fold_idx, preds in enumerate(res["predictions"]):
                rows.extend(
                    prediction_rows(model, fold_idx, preds, time_axis=time_axis)
                )
        return pd.DataFrame(rows)

    def get_splits(self) -> pd.DataFrame:
        """Get outer-CV train/test membership in long format."""
        from .diagnostics import metadata_display_name, optional_values

        frames = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
            for fold_idx, split in enumerate(res.get("splits", [])):
                for set_name, idx_key, id_key, group_key, meta_key in [
                    (
                        "train",
                        "train_idx",
                        "train_sample_id",
                        "train_group",
                        "train_metadata",
                    ),
                    (
                        "test",
                        "test_idx",
                        "test_sample_id",
                        "test_group",
                        "test_metadata",
                    ),
                ]:
                    indices = np.asarray(split[idx_key])
                    n = len(indices)
                    if n == 0:
                        continue

                    data = {
                        "Model": [model] * n,
                        "Fold": [fold_idx] * n,
                        "Set": [set_name] * n,
                        "SampleIndex": indices,
                        "SampleID": np.asarray(split[id_key]),
                        "Group": optional_values(split.get(group_key), n),
                    }
                    metadata = split.get(meta_key) or {}
                    for key, values in metadata.items():
                        v_arr = np.asarray(values, dtype=object)
                        data[metadata_display_name(key)] = v_arr[:n]

                    frames.append(pd.DataFrame(data))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def get_fit_diagnostics(self) -> pd.DataFrame:
        """Get fold-level timing and warning diagnostics."""
        rows = []
        columns = [
            "Model",
            "Fold",
            "FitTime",
            "PredictTime",
            "ScoreTime",
            "TotalTime",
            "Stage",
            "WarningCategory",
            "WarningMessage",
        ]
        for model, res in self.raw.items():
            if "error" in res:
                continue
            for fold_idx, diag in enumerate(res.get("diagnostics", [])):
                base = {
                    "Model": model,
                    "Fold": fold_idx,
                    "FitTime": diag.get("fit_time"),
                    "PredictTime": diag.get("predict_time"),
                    "ScoreTime": diag.get("score_time"),
                    "TotalTime": diag.get("total_time"),
                }
                warnings_ = diag.get("warnings") or []
                if not warnings_:
                    rows.append(
                        {
                            **base,
                            "Stage": None,
                            "WarningCategory": None,
                            "WarningMessage": None,
                        }
                    )
                    continue
                for w in warnings_:
                    rows.append(
                        {
                            **base,
                            "Stage": w.get("stage"),
                            "WarningCategory": w.get("category"),
                            "WarningMessage": w.get("message"),
                        }
                    )
        return pd.DataFrame(rows, columns=columns)

    def get_confusion_matrices(
        self,
        model: Optional[str] = None,
        labels: Optional[Sequence[Any]] = None,
        normalize: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get fold-level confusion matrices in long format."""
        from sklearn.metrics import confusion_matrix

        preds = self._standard_prediction_frame(model=model)
        cols = ["Model", "Fold", "TrueLabel", "PredictedLabel", "Value"]
        rows = []
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)
        if labels is None:
            labels = sorted(
                pd.unique(pd.concat([preds["y_true"], preds["y_pred"]])).tolist()
            )
        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            matrix = confusion_matrix(
                group["y_true"], group["y_pred"], labels=labels, normalize=normalize
            )
            for t_idx, t_label in enumerate(labels):
                for p_idx, p_label in enumerate(labels):
                    rows.append(
                        {
                            "Model": m_name,
                            "Fold": f_idx,
                            "TrueLabel": t_label,
                            "PredictedLabel": p_label,
                            "Value": matrix[t_idx, p_idx],
                        }
                    )
        return pd.DataFrame(rows, columns=cols)

    def get_confusion_counts(
        self, model: Optional[str] = None, labels: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        """Get unnormalized per-fold confusion counts."""
        return self.get_confusion_matrices(model=model, labels=labels, normalize=None)

    def get_pooled_confusion_matrix(
        self,
        model: Optional[str] = None,
        labels: Optional[Sequence[Any]] = None,
        normalize: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get pooled out-of-fold confusion matrices in long format."""
        from sklearn.metrics import confusion_matrix

        preds = self._standard_prediction_frame(model=model)
        cols = ["Model", "TrueLabel", "PredictedLabel", "Value"]
        rows = []
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)
        if labels is None:
            labels = sorted(
                pd.unique(pd.concat([preds["y_true"], preds["y_pred"]])).tolist()
            )
        for m_name, group in preds.groupby("Model"):
            matrix = confusion_matrix(
                group["y_true"], group["y_pred"], labels=labels, normalize=normalize
            )
            for t_idx, t_label in enumerate(labels):
                for p_idx, p_label in enumerate(labels):
                    rows.append(
                        {
                            "Model": m_name,
                            "TrueLabel": t_label,
                            "PredictedLabel": p_label,
                            "Value": matrix[t_idx, p_idx],
                        }
                    )
        return pd.DataFrame(rows, columns=cols)

    def get_roc_curve(
        self, model: Optional[str] = None, pos_label: Optional[Any] = None
    ) -> pd.DataFrame:
        """Get binary or one-vs-rest ROC curve coordinates."""
        from sklearn.metrics import roc_curve

        rows = []
        cols = ["Model", "Fold", "Class", "Threshold", "FPR", "TPR"]
        for m_name, f_idx, label, y_binary, y_score in self._curve_score_groups(
            model, pos_label=pos_label
        ):
            fpr, tpr, thresholds = roc_curve(y_binary, y_score, pos_label=True)
            for thresh, f_val, t_val in zip(thresholds, fpr, tpr):
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Class": label,
                        "Threshold": thresh,
                        "FPR": f_val,
                        "TPR": t_val,
                    }
                )
        return pd.DataFrame(rows, columns=cols)

    def get_pr_curve(
        self, model: Optional[str] = None, pos_label: Optional[Any] = None
    ) -> pd.DataFrame:
        """Get binary or one-vs-rest precision-recall curve coordinates."""
        from sklearn.metrics import precision_recall_curve

        rows = []
        cols = ["Model", "Fold", "Class", "Threshold", "Precision", "Recall"]
        for m_name, f_idx, label, y_binary, y_score in self._curve_score_groups(
            model, pos_label=pos_label
        ):
            precision, recall, thresholds = precision_recall_curve(
                y_binary, y_score, pos_label=True
            )
            threshold_values = np.append(thresholds, np.nan)
            for thresh, p_val, r_val in zip(threshold_values, precision, recall):
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Class": label,
                        "Threshold": thresh,
                        "Precision": p_val,
                        "Recall": r_val,
                    }
                )
        return pd.DataFrame(rows, columns=cols)

    def get_roc_auc_summary(self, model: Optional[str] = None) -> pd.DataFrame:
        """Get summary ROC-AUC metrics across models and folds."""
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import LabelBinarizer

        rows = []
        cols = ["Model", "Fold", "MacroROCAUC", "WeightedROCAUC"]
        preds = self._standard_prediction_frame(model=model)
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)

        proba_cols = sorted(
            [col for col in preds.columns if col.startswith("y_proba_")],
            key=lambda v: int(v.rsplit("_", 1)[-1]),
        )
        if not proba_cols:
            return pd.DataFrame(rows, columns=cols)

        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_true = group["y_true"].to_numpy()
            y_proba = group[proba_cols].to_numpy()

            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            if y_true_bin.shape[1] == 1:
                macro = roc_auc_score(y_true_bin, y_proba[:, -1])
                weighted = macro
            else:
                macro = roc_auc_score(
                    y_true_bin, y_proba, multi_class="ovr", average="macro"
                )
                weighted = roc_auc_score(
                    y_true_bin, y_proba, multi_class="ovr", average="weighted"
                )

            rows.append(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "MacroROCAUC": float(macro),
                    "WeightedROCAUC": float(weighted),
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def get_pr_auc_summary(self, model: Optional[str] = None) -> pd.DataFrame:
        """Get summary PR-AUC (Average Precision) metrics across models and folds."""
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import LabelBinarizer

        rows = []
        cols = ["Model", "Fold", "MacroPRAUC", "WeightedPRAUC"]
        preds = self._standard_prediction_frame(model=model)
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)

        proba_cols = sorted(
            [col for col in preds.columns if col.startswith("y_proba_")],
            key=lambda v: int(v.rsplit("_", 1)[-1]),
        )
        if not proba_cols:
            return pd.DataFrame(rows, columns=cols)

        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_true = group["y_true"].to_numpy()
            y_proba = group[proba_cols].to_numpy()

            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            if y_true_bin.shape[1] == 1:
                macro = average_precision_score(y_true_bin, y_proba[:, -1])
                weighted = macro
            else:
                macro = average_precision_score(y_true_bin, y_proba, average="macro")
                weighted = average_precision_score(
                    y_true_bin, y_proba, average="weighted"
                )

            rows.append(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "MacroPRAUC": float(macro),
                    "WeightedPRAUC": float(weighted),
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def get_calibration_curve(
        self,
        model: Optional[str] = None,
        n_bins: int = 5,
        pos_label: Optional[Any] = None,
        strategy: str = "uniform",
    ) -> pd.DataFrame:
        """Get binary reliability/calibration curve coordinates."""
        from sklearn.calibration import calibration_curve

        rows = []
        cols = [
            "Model",
            "Fold",
            "Class",
            "MeanPredictedProbability",
            "FractionPositive",
        ]
        for m_name, f_idx, label, y_binary, y_score in self._curve_score_groups(
            model, require_probability=True, pos_label=pos_label
        ):
            p_true, p_pred = calibration_curve(
                y_binary.astype(int), y_score, n_bins=n_bins, strategy=strategy
            )
            for pr, tr in zip(p_pred, p_true):
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Class": label,
                        "MeanPredictedProbability": pr,
                        "FractionPositive": tr,
                    }
                )
        return pd.DataFrame(rows, columns=cols)

    def get_probability_diagnostics(self, model: Optional[str] = None) -> pd.DataFrame:
        """Get fold-level log-loss and Brier summaries when probabilities exist."""
        from sklearn.metrics import brier_score_loss, log_loss

        rows = []
        cols = ["Model", "Fold", "Metric", "Class", "Value"]
        preds = self._standard_prediction_frame(model=model)
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)
        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_true = group["y_true"].to_numpy()
            labels = sorted(pd.unique(y_true).tolist())
            y_proba = proba_matrix(group, len(labels))
            if y_proba is None:
                continue
            try:
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Metric": "log_loss",
                        "Class": None,
                        "Value": log_loss(y_true, y_proba, labels=labels),
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    f"log_loss scoring skipped for model={m_name} fold={f_idx}: {e}"
                )
            brier_values = []
            for c_idx, label in enumerate(labels):
                y_binary = np.asarray(y_true) == label
                val = brier_score_loss(y_binary.astype(int), y_proba[:, c_idx])
                brier_values.append(val)
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Metric": "brier_score_ovr",
                        "Class": label,
                        "Value": val,
                    }
                )
            if brier_values:
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Metric": "brier_score_macro",
                        "Class": None,
                        "Value": float(np.mean(brier_values)),
                    }
                )
        return pd.DataFrame(rows, columns=cols)

    def get_statistical_assessment(
        self,
        lightweight: bool = False,
        metric: str = "accuracy",
        n_permutations: int = 1000,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get finite-sample statistical assessment rows in long form.

        Parameters
        ----------
        lightweight : bool
            If True, perform a post-hoc label permutation on out-of-fold predictions.
            This is fast but doesn't account for pipeline leakage.
            If False (default), return the full-pipeline assessment if it was run.
        metric : str
            Metric to use for lightweight assessment.
        n_permutations : int
            Number of permutations for lightweight assessment.
        random_state : int
            Seed for lightweight permutations.
        """
        cols = [
            "Model",
            "Metric",
            "Observed",
            "InferentialUnit",
            "NEff",
            "NullMethod",
            "NPermutations",
            "P0",
            "PValue",
            "CILower",
            "CIUpper",
            "CorrectionMethod",
            "CorrectedPValue",
            "ChanceThreshold",
            "Time",
            "TrainTime",
            "TestTime",
            "NullLower",
            "NullUpper",
            "Significant",
            "Assumptions",
            "Caveat",
        ]

        if not lightweight:
            rows = []
            for res in self.raw.values():
                if "error" in res:
                    continue
                rows.extend(res.get("statistical_assessment", []))
            return pd.DataFrame(rows, columns=cols)

        # Lightweight post-hoc permutation
        from .diagnostics import score_frame

        rng = np.random.default_rng(random_state)
        rows = []
        preds = self._standard_prediction_frame()
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)

        for m_name, group in preds.groupby("Model"):
            y_t = group["y_true"].to_numpy()
            obs = score_frame(group, metric)

            null = []
            for _ in range(n_permutations):
                # Shuffle labels but keep predictions fixed
                perm_group = group.copy()
                perm_group["y_true"] = rng.permutation(y_t)
                null.append(score_frame(perm_group, metric))
            null = np.array(null)

            spec = get_metric_spec(metric)
            if spec.greater_is_better:
                p_val = (np.sum(null >= obs) + 1) / (n_permutations + 1)
            else:
                p_val = (np.sum(null <= obs) + 1) / (n_permutations + 1)

            rows.append(
                {
                    "Model": m_name,
                    "Metric": metric,
                    "Observed": obs,
                    "InferentialUnit": "sample",
                    "NEff": len(y_t),
                    "NullMethod": "posthoc_label_permutation",
                    "NPermutations": n_permutations,
                    "P0": None,
                    "PValue": float(p_val),
                    "CILower": float(np.quantile(null, 0.025)),
                    "CIUpper": float(np.quantile(null, 0.975)),
                    "CorrectionMethod": "none",
                    "CorrectedPValue": float(p_val),
                    "ChanceThreshold": None,
                    "Time": None,
                    "TrainTime": None,
                    "TestTime": None,
                    "NullLower": float(np.quantile(null, 0.025)),
                    "NullUpper": float(np.quantile(null, 0.975)),
                    "Significant": p_val <= 0.05,
                    "Assumptions": "i.i.d. samples; post-hoc label shuffle",
                    "Caveat": "Does not account for pipeline/tuning leakage.",
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def get_statistical_nulls(self) -> Dict[str, Any]:
        """Return stored statistical null distributions, when configured."""
        nulls = {}
        for model, res in self.raw.items():
            if "error" in res:
                continue
            if "statistical_nulls" in res:
                nulls[model] = res["statistical_nulls"]
        return nulls

    def get_model_artifacts(self) -> pd.DataFrame:
        """Return fold-level model artifact metadata in long form."""
        rows = []
        cols = ["Model", "Fold", "ArtifactType", "Key", "Value"]
        for model, res in self.raw.items():
            if "error" in res:
                continue
            for f_idx, m in enumerate(res.get("metadata", [])):
                artifacts = m.get("artifacts", {})
                for a_type, payload in artifacts.items():
                    if isinstance(payload, dict):
                        for k, v in payload.items():
                            rows.append(
                                {
                                    "Model": model,
                                    "Fold": f_idx,
                                    "ArtifactType": a_type,
                                    "Key": k,
                                    "Value": v,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "Model": model,
                                "Fold": f_idx,
                                "ArtifactType": "model",
                                "Key": a_type,
                                "Value": payload,
                            }
                        )
        return pd.DataFrame(rows, columns=cols)

    def get_bootstrap_confidence_intervals(
        self,
        metric: str = "accuracy",
        model: Optional[str] = None,
        unit: Optional[str] = None,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Bootstrap metric confidence intervals over configured inference units."""
        from .diagnostics import score_frame

        u_type = self._resolve_inference_unit(unit)
        rng = np.random.default_rng(random_state)
        preds = self._standard_prediction_frame(model=model)
        cols = [
            "Model",
            "Metric",
            "Unit",
            "NUnits",
            "Estimate",
            "CILower",
            "CIUpper",
            "NBootstraps",
        ]
        rows = []
        if preds.empty:
            return pd.DataFrame(rows, columns=cols)
        alpha = (1.0 - ci) / 2.0
        for m_name, group in preds.groupby("Model"):
            u_indices = unit_indices(group, u_type)
            est = score_frame(group, metric)
            boot = []
            for _ in range(n_bootstraps):
                sampled = rng.integers(0, len(u_indices), size=len(u_indices))
                indices = np.concatenate([u_indices[idx] for idx in sampled])
                sample = group.iloc[indices]
                try:
                    boot.append(score_frame(sample, metric))
                except Exception:
                    # Metrics like ROC-AUC may fail if only one class is present
                    # in a bootstrap sample
                    boot.append(np.nan)

            boot = np.array(boot)
            rows.append(
                {
                    "Model": m_name,
                    "Metric": metric,
                    "Unit": u_type,
                    "NUnits": len(u_indices),
                    "Estimate": est,
                    "CILower": float(np.nanquantile(boot, alpha)),
                    "CIUpper": float(np.nanquantile(boot, 1.0 - alpha)),
                    "NBootstraps": n_bootstraps,
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def compare_models_paired(
        self,
        model_a: str,
        model_b: str,
        metric: str = "accuracy",
        unit: Optional[str] = None,
        n_permutations: int = 1000,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Paired model comparison using outer-fold predictions on shared samples."""
        from .diagnostics import score_frame

        u_type = self._resolve_inference_unit(unit)
        preds = self._standard_prediction_frame()
        a, b = preds[preds["Model"] == model_a], preds[preds["Model"] == model_b]

        # Merge to find shared samples
        # We need to preserve all necessary columns for scoring
        # (y_true, y_pred, y_proba_*, y_score)
        merge_cols = ["SampleID", "y_true", "Fold"]
        for col in ["Time", "TrainTime", "TestTime"]:
            if col in a and col in b:
                merge_cols.append(col)

        merged = a.merge(b, on=merge_cols, suffixes=("_A", "_B"))
        cols = [
            "ModelA",
            "ModelB",
            "Metric",
            "Unit",
            "NUnits",
            "ScoreA",
            "ScoreB",
            "Difference",
            "PValue",
            "NPermutations",
        ]
        if merged.empty:
            return pd.DataFrame([], columns=cols)

        s_a = score_frame(
            merged.rename(columns=lambda x: x[:-2] if x.endswith("_A") else x), metric
        )
        s_b = score_frame(
            merged.rename(columns=lambda x: x[:-2] if x.endswith("_B") else x), metric
        )
        obs = s_a - s_b

        rng = np.random.default_rng(random_state)
        u_indices = paired_unit_indices(merged, u_type)
        null = []

        # Extract prediction/proba columns to swap
        pred_cols_a = [c for c in merged.columns if c.endswith("_A") and c != "Group_A"]
        pred_cols_b = [c.replace("_A", "_B") for c in pred_cols_a]

        for _ in range(n_permutations):
            perm_merged = merged.copy()
            swaps = rng.random(len(u_indices)) < 0.5
            for swap, idxs in zip(swaps, u_indices):
                if swap:
                    # Swap all prediction-related columns for these units
                    for ca, cb in zip(pred_cols_a, pred_cols_b):
                        tmp = perm_merged.loc[merged.index[idxs], ca].copy()
                        perm_merged.loc[merged.index[idxs], ca] = perm_merged.loc[
                            merged.index[idxs], cb
                        ].values
                        perm_merged.loc[merged.index[idxs], cb] = tmp.values

            p_s_a = score_frame(
                perm_merged.rename(columns=lambda x: x[:-2] if x.endswith("_A") else x),
                metric,
            )
            p_s_b = score_frame(
                perm_merged.rename(columns=lambda x: x[:-2] if x.endswith("_B") else x),
                metric,
            )
            null.append(p_s_a - p_s_b)

        p_val = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_permutations + 1)
        return pd.DataFrame(
            [
                {
                    "ModelA": model_a,
                    "ModelB": model_b,
                    "Metric": metric,
                    "Unit": u_type,
                    "NUnits": len(u_indices),
                    "ScoreA": s_a,
                    "ScoreB": s_b,
                    "Difference": obs,
                    "PValue": float(p_val),
                    "NPermutations": n_permutations,
                }
            ],
            columns=cols,
        )

    def _standard_prediction_frame(self, model: Optional[str] = None) -> pd.DataFrame:
        """Return scalar prediction rows, excluding temporal-expanded rows."""
        preds = self.get_predictions()
        if preds.empty:
            return preds
        if model is not None:
            preds = preds[preds["Model"] == model]
        for col in ["Time", "TrainTime", "TestTime"]:
            if col in preds:
                preds = preds[preds[col].isna()]
        return preds

    def _curve_score_groups(
        self,
        model: Optional[str] = None,
        require_probability: bool = False,
        pos_label: Optional[Any] = None,
    ):
        """Yield binary or one-vs-rest score arrays for curve accessors."""
        preds = self._standard_prediction_frame(model=model)
        if preds.empty:
            return
        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_t = group["y_true"].to_numpy()
            labels = sorted(pd.unique(y_t).tolist())
            if len(labels) < 2:
                continue
            if len(labels) == 2:
                label = resolve_pos_label(y_t, pos_label)
                l_idx = labels.index(label)
                p_col = f"y_proba_{l_idx}"
                if p_col in group and group[p_col].notna().all():
                    y_s = group[p_col].to_numpy(dtype=float)
                elif (
                    not require_probability
                    and "y_score" in group
                    and group["y_score"].notna().all()
                ):
                    y_s = group["y_score"].to_numpy(dtype=float)
                    if l_idx == 0:
                        y_s = -y_s
                else:
                    continue
                yield m_name, f_idx, label, np.asarray(y_t) == label, y_s
                continue
            for c_idx, label in enumerate(labels):
                p_col, s_col = f"y_proba_{c_idx}", f"y_score_{c_idx}"
                if p_col in group and group[p_col].notna().all():
                    y_s = group[p_col].to_numpy(dtype=float)
                elif (
                    not require_probability
                    and s_col in group
                    and group[s_col].notna().all()
                ):
                    y_s = group[s_col].to_numpy(dtype=float)
                else:
                    continue
                yield m_name, f_idx, label, np.asarray(y_t) == label, y_s

    def get_feature_importances(self, fold_level: bool = False) -> pd.DataFrame:
        """Get feature importances in long format."""
        cols = (
            ["Model", "Fold", "Feature", "FeatureName", "Importance", "Rank"]
            if fold_level
            else ["Model", "Feature", "FeatureName", "Mean", "Std", "Rank"]
        )
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
            imp = res.get("importances")
            if not imp:
                continue
            if fold_level:
                raw = np.asarray(imp.get("raw", []), dtype=float)
                if raw.ndim != 2:
                    continue
                f_names = feature_names_for_result(res, raw.shape[1])
                for f_idx, f_vals in enumerate(raw):
                    for ft_idx, val in enumerate(f_vals):
                        rows.append(
                            {
                                "Model": model,
                                "Fold": f_idx,
                                "Feature": ft_idx,
                                "FeatureName": f_names[ft_idx],
                                "Importance": val,
                            }
                        )
            else:
                means, stds = (
                    np.asarray(imp.get("mean", []), dtype=float).ravel(),
                    np.asarray(imp.get("std", []), dtype=float).ravel(),
                )
                if len(means) == 0:
                    continue
                f_names = feature_names_for_result(res, len(means))
                if len(stds) != len(means):
                    stds = np.full(len(means), np.nan)
                for ft_idx, m in enumerate(means):
                    rows.append(
                        {
                            "Model": model,
                            "Feature": ft_idx,
                            "FeatureName": f_names[ft_idx],
                            "Mean": m,
                            "Std": stds[ft_idx],
                        }
                    )
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        if fold_level:
            df["Rank"] = (
                df.groupby(["Model", "Fold"])["Importance"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
        else:
            df["Rank"] = (
                df.groupby("Model")["Mean"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
        return df

    def _metadata_columns_from_splits(self) -> list[str]:
        from .diagnostics import metadata_display_name

        cols = []
        for res in self.raw.values():
            if "error" in res:
                continue
            for split in res.get("splits", []):
                for m_key in ("train_metadata", "test_metadata"):
                    for key in (split.get(m_key) or {}).keys():
                        col = metadata_display_name(key)
                        if col not in cols:
                            cols.append(col)
        return cols

    def _resolve_inference_unit(self, unit: Optional[str]) -> str:
        if unit is not None:
            return unit
        return self.meta.get("inferential_unit") or "sample"

    def _time_axis(self) -> Optional[list[Any]]:
        t_axis = self.meta.get("time_axis")
        return list(t_axis) if t_axis is not None else None

    def _time_value(self, index: int) -> Any:
        from .diagnostics import time_value as get_time_val

        return get_time_val(index, self._time_axis())

    def get_best_params(self) -> pd.DataFrame:
        """Get the best hyperparameters selected per fold."""
        rows = []
        for m_name, res in self.raw.items():
            if "error" in res:
                continue
            if "metadata" in res:
                for f_idx, meta in enumerate(res["metadata"]):
                    if "best_params" in meta:
                        for p_name, p_val in meta["best_params"].items():
                            rows.append(
                                {
                                    "Model": m_name,
                                    "Fold": f_idx,
                                    "Param": p_name,
                                    "Value": p_val,
                                }
                            )
        return pd.DataFrame(rows)

    def get_search_results(self) -> pd.DataFrame:
        """Get compact hyperparameter-search diagnostics in long form."""
        rows = []
        cols = [
            "Model",
            "Fold",
            "Candidate",
            "Rank",
            "MeanTestScore",
            "StdTestScore",
            "Params",
        ]
        for m_name, res in self.raw.items():
            if "error" in res:
                continue
            for f_idx, meta in enumerate(res.get("metadata", [])):
                for s_row in meta.get("search_results", []):
                    rows.append(
                        {
                            "Model": m_name,
                            "Fold": f_idx,
                            "Candidate": s_row.get("candidate"),
                            "Rank": s_row.get("rank_test_score"),
                            "MeanTestScore": s_row.get("mean_test_score"),
                            "StdTestScore": s_row.get("std_test_score"),
                            "Params": s_row.get("params"),
                        }
                    )
        return pd.DataFrame(rows, columns=cols)

    def get_selected_features(self) -> pd.DataFrame:
        """Get fold-level selected feature masks in long format."""
        rows = []
        cols = ["Model", "Fold", "Feature", "FeatureName", "Selected", "Order"]
        for m_name, res in self.raw.items():
            if "error" in res:
                continue
            for f_idx, meta in enumerate(res.get("metadata", [])):
                if "selected_features" not in meta:
                    continue
                mask = np.asarray(meta["selected_features"], dtype=bool)
                order = meta.get("selection_order")
                f_names = meta.get("feature_names")
                if f_names is None or len(f_names) != len(mask):
                    f_names = [f"feature_{idx}" for idx in range(len(mask))]

                for ft_idx, selected in enumerate(mask):
                    s_order = None
                    if order is not None:
                        # If order is a ranking array (like RFE.ranking_)
                        if isinstance(order, (np.ndarray, list)) and len(order) == len(
                            mask
                        ):
                            s_order = int(order[ft_idx])
                        # If order is a list of selected indices in order
                        elif isinstance(order, (list, np.ndarray)) and ft_idx in order:
                            s_order = list(order).index(ft_idx) + 1

                    rows.append(
                        {
                            "Model": m_name,
                            "Fold": f_idx,
                            "Feature": ft_idx,
                            "FeatureName": f_names[ft_idx],
                            "Selected": bool(selected),
                            "Order": s_order,
                        }
                    )
        return pd.DataFrame(rows, columns=cols)

    def get_feature_scores(self) -> pd.DataFrame:
        """Get fold-level feature-selection scores."""
        rows = []
        cols = [
            "Model",
            "Fold",
            "Feature",
            "FeatureName",
            "Selector",
            "Score",
            "PValue",
            "Selected",
        ]
        for m_name, res in self.raw.items():
            if "error" in res:
                continue
            for f_idx, meta in enumerate(res.get("metadata", [])):
                if "feature_scores" not in meta:
                    continue
                scores = np.asarray(meta["feature_scores"], dtype=float)
                pvals = meta.get("feature_pvalues")
                if pvals is not None:
                    pvals = np.asarray(pvals, dtype=float)
                f_names = meta.get("feature_names")
                if f_names is None or len(f_names) != len(scores):
                    f_names = [f"feature_{idx}" for idx in range(len(scores))]
                sel = meta.get("selected_features")
                if sel is not None:
                    sel = np.asarray(sel, dtype=bool)
                for ft_idx, sc in enumerate(scores):
                    rows.append(
                        {
                            "Model": m_name,
                            "Fold": f_idx,
                            "Feature": ft_idx,
                            "FeatureName": f_names[ft_idx],
                            "Selector": meta.get("feature_selection_method"),
                            "Score": sc,
                            "PValue": pvals[ft_idx]
                            if pvals is not None and len(pvals) == len(scores)
                            else np.nan,
                            "Selected": bool(sel[ft_idx])
                            if sel is not None and len(sel) == len(scores)
                            else np.nan,
                        }
                    )
        return pd.DataFrame(rows, columns=cols)

    def get_feature_stability(self) -> pd.DataFrame:
        """Analyze feature selection stability across folds."""
        rows = []
        for m_name, res in self.raw.items():
            if "error" in res:
                continue
            if "metadata" in res:
                masks = []
                f_names = None
                for meta in res["metadata"]:
                    if "selected_features" in meta:
                        masks.append(meta["selected_features"])
                        if f_names is None and "feature_names" in meta:
                            f_names = meta["feature_names"]
                if masks:
                    stack = np.vstack(masks)
                    stability = np.mean(stack, axis=0)
                    for ft_idx, freq in enumerate(stability):
                        row = {"Model": m_name, "Feature": ft_idx, "Frequency": freq}
                        if f_names is not None and len(f_names) == len(stability):
                            row["FeatureName"] = f_names[ft_idx]
                        rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_generalization_matrix(self, metric: str = None) -> pd.DataFrame:
        """Get Generalization Matrix (Train Time x Test Time) averaged across folds."""
        for model_name, res in self.raw.items():
            if "error" in res:
                continue
            metrics_data = res["metrics"]
            if metric is None:
                metric = list(metrics_data.keys())[0]
            if metric not in metrics_data:
                continue
            fold_scores = metrics_data[metric]["folds"]
            valid_matrices = [
                s for s in fold_scores if isinstance(s, np.ndarray) and s.ndim == 2
            ]
            if valid_matrices:
                stack = np.stack(valid_matrices)
                mean = np.nanmean(stack, axis=0)
                time_axis = self._time_axis()
                if time_axis is not None and len(time_axis) == mean.shape[0]:
                    labels = time_axis
                else:
                    labels = list(range(mean.shape[0]))
                return pd.DataFrame(mean, index=labels, columns=labels)
        return pd.DataFrame()
