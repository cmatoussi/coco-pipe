"""
Decoding Core
=============
This module is responsible for:
1. Orchestrating the Cross-Validation loop.
2. Managing Estimator lifecycles (instantiation, fitting, prediction).
3. Computing metrics dynamically based on task type.
4. Aggregating results for downstream analysis.
"""

import atexit
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any, Dict, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import (
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
    f_regression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

from ..report.provenance import get_environment_info
from .configs import ExperimentConfig
from .metrics import get_metric_names, get_metric_spec
from .registry import get_estimator_cls
from .splitters import get_cv_splitter

logger = logging.getLogger(__name__)

GROUP_CV_STRATEGIES = {
    "group_kfold",
    "stratified_group_kfold",
    "leave_p_out",
    "leave_one_group_out",
}

RESULT_SCHEMA_VERSION = "decoding_result_v1"


class Experiment:
    """
    Main executor for decoding experiments.

    Parameters
    ----------
    config : ExperimentConfig
        The complete configuration for the experiment.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.result_: Optional["ExperimentResult"] = None
        self._validate_config()

    def _validate_config(self):
        """
        Perform comprehensive runtime validation of the configuration.

        Logic
        -----
        1. **Tuning Consistency**: Warns if `tuning.enabled` but no `grids`
           are provided.
        2. **Task vs Metrics**: Checks if metrics match the task (e.g. no 'accuracy'
           for regression). Raises ValueError if incompatible.
        3. **Task vs CV**: Checks if CV strategy matches task (e.g. no 'stratified'
           for regression). Raises ValueError if incompatible.
        4. **Task vs Model**: Heuristic check for model type (e.g. no Regressor for
           Classification). Raises ValueError if incompatible.

        Raises
        ------
        ValueError
            If configuration contains incompatible settings.
        """
        task = self.config.task

        # 1. Tuning Consistency
        if self.config.tuning.enabled and not self.config.grids:
            logger.warning(
                "Hyperparameter tuning is enabled but no 'grids' are defined in the "
                "config."
            )
        if self.config.tuning.enabled and self.config.tuning.cv is None:
            raise ValueError(
                "Hyperparameter tuning requires an explicit inner CV config at "
                "tuning.cv. The outer config.cv is used only for evaluation."
            )

        fs_conf = self.config.feature_selection
        if fs_conf.enabled and fs_conf.method == "sfs":
            if fs_conf.cv is None:
                raise ValueError(
                    "Sequential feature selection requires an explicit inner CV "
                    "config at feature_selection.cv."
                )

        for metric in self.config.metrics:
            metric_spec = get_metric_spec(metric)
            if metric_spec.task != task:
                raise ValueError(
                    f"Metric '{metric}' is incompatible with task '{task}'. "
                    f"Available {task} metrics: {get_metric_names(task)}."
                )

        # 3. Task vs CV Strategy
        if task == "regression":
            if "stratified" in self.config.cv.strategy:
                raise ValueError(
                    f"CV strategy '{self.config.cv.strategy}' implies stratification, "
                    f"which is invalid for regression tasks."
                )

        # 4. Task vs Model Type
        # We infer type from the config class name or method string
        for name, model_cfg in self.config.models.items():
            method_name = model_cfg.method.lower()

            if task == "classification":
                if "regressor" in method_name or "regression" in method_name:
                    # Exception: LogisticRegression is a classifier
                    if "logistic" not in method_name:
                        raise ValueError(
                            f"Model '{name}' ({model_cfg.method}) appears to be a "
                            f"Regressor, but task is 'classification'."
                        )

            elif task == "regression":
                if (
                    "classifier" in method_name
                    or "svc" in method_name
                    or "logistic" in method_name
                ):
                    # SVR is valid, SVC is not (usually)
                    raise ValueError(
                        f"Model '{name}' ({model_cfg.method}) appears to be a "
                        f"Classifier, but task is 'regression'."
                    )

    def _prepare_estimator(self, model_name: str, model_config: Any) -> BaseEstimator:
        """
        Orchestrate the creation of the full Estimator Pipeline.

        Steps
        -----
        1. **Instantiation**: Calls `_instantiate_model` to get the base estimator
           (handling recursion).
        2. **Scaling**: If `use_scaler=True`, prepends a StandardScaler.
        3. **Feature Selection**: If enabled, prepends the FS step (Filter or Wrapper).
        4. **Pipeline Construction**: wraps steps in `sklearn.pipeline.Pipeline`.
           - Enables caching if FS + Tuning are both active.
        5. **Tuning Wrapper**: If tuning is enabled for this model, wraps the Pipeline
           in GridSearchCV/RandomizedSearchCV via `_wrap_with_tuning`.

        Parameters
        ----------
        model_name : str
            Friendly name from config (used for grid lookup).
        model_config : Any
            Pydantic configuration object for the model.

        Returns
        -------
        BaseEstimator
            Final ready-to-run estimator (Pipeline or SearchCV).
        """
        # 1. Instantiate the Core Estimator
        full_est = self._instantiate_model(model_name, model_config)

        # 2. Build Pipeline Steps
        steps = []

        # Scaling
        if self.config.use_scaler:
            steps.append(("scaler", StandardScaler()))

        # Feature Selection
        if self.config.feature_selection.enabled:
            fs_step = self._create_fs_step(full_est)
            if fs_step:
                steps.append(fs_step)

        # Final Estimator
        steps.append(("clf", full_est))

        # 3. Create Pipeline with Caching if needed
        if (
            self.config.feature_selection.enabled
            and self.config.tuning.enabled
            and self.config.grids
        ):
            cachedir = mkdtemp()
            atexit.register(lambda: rmtree(cachedir, ignore_errors=True))
            est = Pipeline(steps, memory=cachedir)
        else:
            est = Pipeline(steps)

        # 4. Wrap with Tuning if enabled
        if (
            self.config.tuning.enabled
            and self.config.grids
            and model_name in self.config.grids
        ):
            est = self._wrap_with_tuning(est, model_name)

        return est

    def _instantiate_model(self, name: str, config: Any) -> BaseEstimator:
        """
        Instantiate a raw estimator from its configuration object.

        Logic
        -----
        1. **Registry Lookup**: Resolves class from `config.method`.
        2. **Recursion**: If config implies a meta-estimator (has `base_estimator`),
           recursively calls `_prepare_estimator` for the child.
        3. **Parameter Injection**: passed config fields as kwargs to `__init__`.

        Returns
        -------
        BaseEstimator
            The instantiated model (e.g., LogisticRegression instance) without pipeline
            wrappers.
        """
        # 1. Get Class
        est_cls = get_estimator_cls(config.method)

        # 2. Extract Params
        params = config.model_dump(exclude={"method"})

        # 3. Recursive Preparation (for Sliding/Generalizing internal 'base_estimator')
        if "base_estimator" in params:
            base_conf = params["base_estimator"]
            params["base_estimator"] = self._prepare_estimator(
                f"{name}_base", base_conf
            )

        # 4. Instantiate strictly. Config schemas should match estimator signatures.
        try:
            return est_cls(**params)
        except TypeError as exc:
            raise ValueError(
                f"Failed to instantiate model '{name}' with estimator "
                f"'{est_cls.__name__}': {exc}"
            ) from exc

    def _create_fs_step(self, estimator: BaseEstimator) -> Optional[tuple]:
        """
        Create a Feature Selection step for the pipeline.

        Logic
        -----
        - **Filter (k_best)**: Fast. selected before training the classifier based on
          statistical test. No inner CV loop required.
        - **Wrapper (sfs)**: Slow but accurate. Wraps the estimator in a
          SequentialFeatureSelector. This runs an **Inner CV Loop**
          (size = config.feature_selection.cv) to validate feature subsets.

        If used inside Hyperparameter Tuning, this step is part of the Pipeline,
        ensuring features are re-selected for every fold and every parameter
        combination (Nested Simplification).

        Returns
        -------
        tuple or None
            ("fs", Transformer) step for sklearn Pipeline.
        """
        fs_conf = self.config.feature_selection

        if fs_conf.method == "k_best":
            score_func = (
                f_classif if self.config.task == "classification" else f_regression
            )
            return (
                "fs",
                SelectKBest(
                    score_func=score_func,
                    k=fs_conf.n_features if fs_conf.n_features is not None else "all",
                ),
            )

        elif fs_conf.method == "sfs":
            inner_cv = get_cv_splitter(fs_conf.cv, require_groups=False)
            return (
                "fs",
                SequentialFeatureSelector(
                    estimator=clone(estimator),
                    n_features_to_select=fs_conf.n_features,
                    direction=fs_conf.direction,
                    cv=inner_cv,
                    scoring=self._resolve_fs_scoring(),
                    n_jobs=self.config.n_jobs,
                ),
            )
        return None

    def _resolve_fs_scoring(self) -> str:
        """Resolve SFS scoring from the explicit precedence chain."""
        return (
            self.config.feature_selection.scoring
            or self.config.tuning.scoring
            or self.config.metrics[0]
        )

    def _wrap_with_tuning(self, estimator: BaseEstimator, name: str) -> BaseEstimator:
        """
        Wrap the estimator (or pipeline) in a Hyperparameter Search object.

        This implements **Nested Cross-Validation** (Middle Loop):
        1. **Input**: A Pipeline (Scaler + FS + Classifier).
        2. **Search**: Creates a GridSearchCV / RandomizedSearchCV.
        3. **Process**:
           - For each fold of the *tuning* CV (defined by config.cv):
             - Train the Pipeline (including FS!) on the tuning train set.
             - Evaluate on the tuning validation set.
           - Finds the best (Hyperparameters + Features) combination.
           - Refits on the entire training set provided by the Outer Loop.

        This ensures simultaneous optimization of Preprocessing (FS) and Modeling
        parameters.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        grid = self.config.grids[name]

        mapped_grid = {}
        for k, v in grid.items():
            if "__" in k:
                mapped_grid[k] = v
            else:
                mapped_grid[f"clf__{k}"] = v
        grid = mapped_grid

        valid_params = estimator.get_params(deep=True)
        invalid_keys = sorted(key for key in grid if key not in valid_params)
        if invalid_keys:
            raise ValueError(
                f"Invalid tuning grid key(s) for model '{name}': "
                f"{invalid_keys}. Keys must match estimator parameters after "
                "pipeline mapping."
            )

        # SearchCV receives the outer training-fold groups later in fit(...).
        cv_splitter = get_cv_splitter(self.config.tuning.cv, require_groups=False)

        search_kwargs = {
            "estimator": estimator,
            "param_grid"
            if self.config.tuning.search_type == "grid"
            else "param_distributions": grid,
            "cv": cv_splitter,
            "scoring": self.config.tuning.scoring or self.config.metrics[0],
            "n_jobs": self.config.tuning.n_jobs,
            "refit": True,
        }

        if self.config.tuning.search_type == "grid":
            return GridSearchCV(**search_kwargs)
        else:
            return RandomizedSearchCV(
                n_iter=self.config.tuning.n_iter,
                random_state=self.config.tuning.random_state,
                **search_kwargs,
            )

    def run(
        self,
        X: np.ndarray,
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        time_axis: Optional[Sequence[Any]] = None,
    ) -> "ExperimentResult":
        """
        Execute the full experiment pipeline.

        This is the main entry point. It orchestrates:
        1. **Data Validation**: Checks input shapes and types.
        2. **Model Loop**: Iterates through all configured models.
        3. **Preparation**: Instantiates models -> Builds Pipelines (Scaler/FS) ->
           Wraps in Tuning.
        4. **Validation**: Runs the Outer Cross-Validation loop (optionally
           parallelized).
        5. **Aggregation**: Collects scores, predictions, and importances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (2D) or Time-Series data (3D).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target labels or values.
        groups : array-like of shape (n_samples,), optional
            Group labels for splitting (e.g., subject-specific splits).
        feature_names : list of str, optional
            Explicit feature names aligned to columns in ``X``. When omitted,
            names are generated as ``feature_0``, ``feature_1``, ...
        sample_ids : sequence, optional
            Explicit sample IDs aligned to rows in ``X``. When omitted, sample
            row positions are used.
        time_axis : sequence, optional
            Explicit temporal coordinate axis aligned to ``X.shape[-1]`` for
            temporal 3D inputs.

        Returns
        -------
        ExperimentResult
            Object containing results with methods to export to Tidy DataFrames.
        """
        start_time = time.time()
        logger.info(f"Starting Experiment: Task={self.config.task}")

        # 1. Validate Inputs
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) == 0:
            raise ValueError("Input X is empty.")
        if len(y) != len(X):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, y has {len(y)}."
            )

        self._feature_names = self._resolve_feature_names(X, feature_names)
        sample_ids = self._resolve_sample_ids(len(X), sample_ids)
        self._sample_ids = sample_ids
        time_axis = self._resolve_time_axis(X, time_axis)
        self._time_axis = time_axis

        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(X):
                raise ValueError(
                    f"Length mismatch: groups has {len(groups)}, X has {len(X)}."
                )

        self._validate_groups_for_cv(groups)

        # 2. Check Task Consistency (Classification vs Regression)
        target_type = type_of_target(y)
        if self.config.task == "classification" and target_type == "continuous":
            raise ValueError(
                f"Task is 'classification' but target type is '{target_type}'. "
                "Please check your labels or switch task to 'regression'."
            )

        # 3. Main Loop over Configured Models
        for friendly_name, model_cfg in self.config.models.items():
            logger.info(f"Evaluating Model: {friendly_name} ({model_cfg.method})")

            try:
                # A. Prepare (Instantiate + Scale + FS + Tune Wrapper)
                estimator = self._prepare_estimator(friendly_name, model_cfg)

                # B. Execute Cross-Validation
                # Note: Parallelism is handled inside _cross_validate if
                # config.n_jobs > 1
                cv_results = self._cross_validate(estimator, X, y, groups, sample_ids)

                # C. Store Results
                self.results[friendly_name] = cv_results

            except Exception as e:
                logger.error(
                    f"Failed to evaluate model '{friendly_name}': {e}", exc_info=True
                )
                self.results[friendly_name] = {"error": str(e), "status": "failed"}

        total_time = time.time() - start_time
        logger.info(f"Experiment Completed in {total_time:.2f}s")

        self.result_ = ExperimentResult(
            self.results,
            config=self.config.model_dump(),
            meta=self._build_result_meta(X, time_axis),
            schema_version=RESULT_SCHEMA_VERSION,
        )
        return self.result_

    @staticmethod
    def _resolve_sample_ids(
        n_samples: int, sample_ids: Optional[Sequence[Any]] = None
    ) -> np.ndarray:
        """Return explicit sample IDs or generated row-position IDs."""
        if sample_ids is None:
            return np.arange(n_samples)

        sample_ids = np.asarray(sample_ids)
        if len(sample_ids) != n_samples:
            raise ValueError(
                "sample_ids must align with rows in X: "
                f"expected {n_samples}, got {len(sample_ids)}."
            )
        return sample_ids

    @staticmethod
    def _resolve_time_axis(
        X: np.ndarray, time_axis: Optional[Sequence[Any]] = None
    ) -> Optional[np.ndarray]:
        """Return explicit or generated temporal coordinates for 3D inputs."""
        if X.ndim != 3:
            return np.asarray(time_axis) if time_axis is not None else None

        if time_axis is None:
            return np.arange(X.shape[-1])

        time_axis = np.asarray(time_axis)
        if len(time_axis) != X.shape[-1]:
            raise ValueError(
                "time_axis must align with the temporal dimension of X: "
                f"expected {X.shape[-1]}, got {len(time_axis)}."
            )
        return time_axis

    def _build_result_meta(
        self, X: np.ndarray, time_axis: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Build reproducibility metadata for the in-memory result payload."""
        meta = get_environment_info()
        meta.update(
            {
                "tag": self.config.tag,
                "task": self.config.task,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
            }
        )
        if time_axis is not None:
            meta["time_axis"] = time_axis.tolist()
        return meta

    def _validate_groups_for_cv(self, groups: Optional[np.ndarray]) -> None:
        """Fail clearly when configured outer or tuning CV requires groups."""
        if groups is not None:
            return

        if self.config.cv.strategy in GROUP_CV_STRATEGIES:
            raise ValueError(
                f"CV strategy '{self.config.cv.strategy}' requires groups."
            )

        if (
            self.config.tuning.enabled
            and self.config.tuning.cv is not None
            and self.config.tuning.cv.strategy in GROUP_CV_STRATEGIES
        ):
            raise ValueError(
                f"Tuning CV strategy '{self.config.tuning.cv.strategy}' "
                "requires groups."
            )

        fs_conf = self.config.feature_selection
        if (
            fs_conf.enabled
            and fs_conf.method == "sfs"
            and fs_conf.cv is not None
            and fs_conf.cv.strategy in GROUP_CV_STRATEGIES
        ):
            raise ValueError(
                f"Feature selection CV strategy '{fs_conf.cv.strategy}' "
                "requires groups."
            )

    def save_results(self, path: Optional[Union[str, Path]] = None):
        """
        Serialize results, configuration, and metadata to disk.

        Parameters
        ----------
        path : str or Path, optional
            Path to save the results. If None, uses config.output_dir.
            If both are None, raises ValueError.
        """
        if path is None:
            path = self.config.output_dir
            if path is None:
                raise ValueError("No output path specified in config or arguments.")

        path = Path(path)

        # 1. Bundle the same payload shape returned by Experiment.run().
        if self.result_ is not None:
            payload = self.result_.to_payload()
        else:
            meta = get_environment_info()
            meta.update(
                {
                    "tag": self.config.tag,
                    "task": self.config.task,
                    "n_samples": None,
                    "n_features": None,
                }
            )
            payload = ExperimentResult(
                self.results,
                config=self.config.model_dump(),
                meta=meta,
                schema_version=RESULT_SCHEMA_VERSION,
            ).to_payload()

        # 2. Create Directory
        # If path is a directory (no extension), append filename
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{self.config.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            target = path / filename
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path

        # 3. Save
        logger.info(f"Saving results to {target}")
        joblib.dump(payload, target)
        return target

    @staticmethod
    def load_results(path: Union[str, Path]) -> "ExperimentResult":
        """
        Load a saved experiment payload and wrap it in ExperimentResult.

        Returns
        -------
        ExperimentResult
            The loaded results wrapper.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError("Saved decoding result payload must be a dictionary.")
        required = {"schema_version", "config", "meta", "results"}
        missing = required - set(payload)
        if missing:
            raise ValueError(
                "Saved decoding result payload is missing required keys: "
                f"{sorted(missing)}."
            )
        if payload["schema_version"] != RESULT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported decoding result schema version: "
                f"{payload['schema_version']}."
            )
        return ExperimentResult(
            payload["results"],
            config=payload["config"],
            meta=payload["meta"],
            schema_version=payload["schema_version"],
        )

    def _cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        sample_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Execute the Outer Cross-Validation Loop (Evaluation).

        This is the **Level 1 (Top Level)** Splits:
        - Splits the entire dataset into K folds (defined by config.cv).
        - For each fold:
          1. **Training Data**: 80% (if 5-fold). Passed to `estimator.fit()`.
             - If `estimator` is a GridSearch (Tuning Enabled), it will internally split
               this 80% again for validation (Level 2 Split).
          2. **Test Data**: 20%. Used strictly for final `estimator.predict()`
             evaluation.

        Parallelization
        ---------------
        If `config.n_jobs > 1`, these folds run in parallel processes to speed up
        execution.
        """
        cv = get_cv_splitter(self.config.cv, groups=groups, y=y)

        # Prepare CV iterator
        splits = list(cv.split(X, y, groups))

        # Parallel Execution
        # We use n_jobs from config.
        n_jobs_outer = self.config.n_jobs

        # OVERSUBSCRIPTION PROTECTION
        # If outer loop is parallel, force inner estimators to run sequentially.
        # Otherwise, we might spawn N_outer * N_inner threads, crashing the system.
        parallel_estimator = clone(estimator)
        if n_jobs_outer != 1:
            parallel_estimator = self._force_serial_execution(parallel_estimator)

        parallel = joblib.Parallel(n_jobs=n_jobs_outer, verbose=self.config.verbose)

        results = parallel(
            joblib.delayed(self._fit_and_score_fold)(
                clone(parallel_estimator),
                X,
                y,
                groups,
                sample_ids,
                train_idx,
                test_idx,
            )
            for train_idx, test_idx in splits
        )

        # Unpack Results
        fold_scores = defaultdict(list)
        fold_preds = []
        fold_indices = []
        fold_importances = []
        fold_metadata = []
        fold_splits = []

        for res in results:
            fold_indices.append(res["test_idx"])
            fold_preds.append(res["preds"])
            fold_importances.append(res["importance"])
            fold_metadata.append(res.get("metadata", {}))
            fold_splits.append(res["split"])

            for m, s in res["scores"].items():
                fold_scores[m].append(s)

        # Aggregate Metrics
        metrics_summary = {
            m: {"mean": np.nanmean(s), "std": np.nanstd(s), "folds": s}
            for m, s in fold_scores.items()
        }

        # Aggregate Importances
        valid_imps = [f for f in fold_importances if f is not None]
        aggregated_importances = None
        if valid_imps:
            try:
                # Check consistency
                first_shape = valid_imps[0].shape
                if all(imp.shape == first_shape for imp in valid_imps):
                    stack = np.vstack(valid_imps)
                    aggregated_importances = {
                        "mean": np.mean(stack, axis=0),
                        "std": np.std(stack, axis=0),
                        "raw": stack,
                        "feature_names": self._metadata_feature_names(stack.shape[1]),
                    }
            except Exception:
                pass

        return {
            "metrics": metrics_summary,
            "predictions": fold_preds,
            "indices": fold_indices,
            "importances": aggregated_importances,
            "metadata": fold_metadata,
            "splits": fold_splits,
        }

    def _fit_and_score_fold(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        sample_ids: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Execute a single Cross-Validation fold: Fit, Predict, and Score.

        Optimized for:
        - **Standard Estimators**: (N, F) input -> (N,) output.
        - **Sliding Estimators**: (N, F, T) input -> (N, T) output (Diagonal Decoding).

        Returns
        -------
        dict
            Contains 'test_idx', 'preds' (y_pred, y_true, y_proba),
            'scores' (dict of metric values), and 'importance'.
        """
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx] if groups is not None else None

        # 1. Fit
        self._fit_estimator(estimator, X_train, y_train, groups_train)

        # 2. Predict (Standard or Temporal)
        y_pred = estimator.predict(X_test)
        test_groups = groups[test_idx] if groups is not None else None
        fold_data = {
            "sample_index": test_idx,
            "sample_id": sample_ids[test_idx],
            "group": test_groups,
            "y_true": y_test,
            "y_pred": y_pred,
        }

        # 3. Predict probabilities for prediction exports when available.
        if hasattr(estimator, "predict_proba"):
            try:
                fold_data["y_proba"] = estimator.predict_proba(X_test)
            except Exception:
                pass  # Some estimators have the method but fail if not calibrated
                # or supported correctly

        # 4. Extract Feature Importances
        imp = None
        try:
            imp = self._extract_feature_importances(estimator)
        except Exception:
            pass

        # 5. Compute Metrics
        scores = {}
        is_multiclass = type_of_target(y_test) == "multiclass"

        for metric_name in self.config.metrics:
            metric_spec = get_metric_spec(metric_name)
            scorer = metric_spec.scorer
            if metric_spec.response_method == "predict":
                y_est = y_pred
                is_proba = False
            else:
                y_est, is_proba = self._get_metric_response(
                    estimator,
                    X_test,
                    metric_name,
                    metric_spec.response_method,
                    is_multiclass,
                )

            try:
                val = self._compute_metric_safe(
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

        # 6. Extract Metadata (Best Params, Selected Features)
        meta = {}
        try:
            meta = self._extract_metadata(estimator)
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return {
            "test_idx": test_idx,
            "preds": fold_data,
            "scores": scores,
            "importance": imp,
            "metadata": meta,
            "split": self._split_record(train_idx, test_idx, sample_ids, groups),
        }

    @staticmethod
    def _split_record(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        sample_ids: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Return sample context for one outer-CV split."""
        record = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_sample_id": sample_ids[train_idx],
            "test_sample_id": sample_ids[test_idx],
            "train_group": groups[train_idx] if groups is not None else None,
            "test_group": groups[test_idx] if groups is not None else None,
        }
        return record

    def _fit_estimator(
        self,
        estimator: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: Optional[np.ndarray],
    ) -> None:
        """Fit estimators, routing groups only where configured CV needs them."""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        search_cv = isinstance(estimator, (GridSearchCV, RandomizedSearchCV))
        route_groups = groups_train is not None and self._uses_group_sfs_cv()
        pass_groups = groups_train is not None and (search_cv or route_groups)
        fit_kwargs = {"groups": groups_train} if pass_groups else {}

        if route_groups:
            with config_context(enable_metadata_routing=True):
                estimator.fit(X_train, y_train, **fit_kwargs)
        else:
            estimator.fit(X_train, y_train, **fit_kwargs)

    def _uses_group_sfs_cv(self) -> bool:
        """Whether SFS needs groups routed through fit metadata."""
        fs_conf = self.config.feature_selection
        return (
            fs_conf.enabled
            and fs_conf.method == "sfs"
            and fs_conf.cv is not None
            and fs_conf.cv.strategy in GROUP_CV_STRATEGIES
        )

    @staticmethod
    def _resolve_feature_names(
        X: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Return explicit feature names or generated array-column names."""
        if X.ndim < 2:
            expected = 1
        else:
            expected = X.shape[1]

        if feature_names is not None:
            names = [str(name) for name in feature_names]
            if len(names) != expected:
                raise ValueError(
                    "feature_names must align with the feature dimension of X: "
                    f"expected {expected}, got {len(names)}."
                )
            return names

        if X.ndim < 2:
            return ["feature_0"]
        return [f"feature_{idx}" for idx in range(X.shape[1])]

    def _extract_metadata(self, estimator: BaseEstimator) -> Dict[str, Any]:
        """
        Extract training metadata like best Hyperparameters and Selected Features.
        """
        meta = {}

        # 1. Best Params (from GridSearchCV/RandomizedSearchCV)
        if hasattr(estimator, "best_params_"):
            meta["best_params"] = estimator.best_params_
            meta["best_score"] = estimator.best_score_
            meta["best_index"] = estimator.best_index_
            meta["search_results"] = self._compact_search_results(estimator)
            # Unwrap best estimator for feature selection
            search_best = estimator.best_estimator_
        else:
            search_best = estimator

        # 2. Selected Features (from Pipeline step 'fs')
        if isinstance(search_best, Pipeline):
            fs_step = search_best.named_steps.get("fs")
            if fs_step and hasattr(fs_step, "get_support"):
                mask = fs_step.get_support()
                indices = np.flatnonzero(mask)
                feature_names = self._metadata_feature_names(len(mask))
                selector_method = self.config.feature_selection.method

                meta["feature_selection_method"] = selector_method
                meta["selected_features"] = mask
                meta["selected_feature_indices"] = indices
                meta["selected_feature_names"] = [feature_names[idx] for idx in indices]
                meta["feature_names"] = feature_names
                if selector_method == "k_best":
                    if hasattr(fs_step, "scores_"):
                        meta["feature_scores"] = fs_step.scores_
                    if hasattr(fs_step, "pvalues_"):
                        meta["feature_pvalues"] = fs_step.pvalues_

        return meta

    @staticmethod
    def _get_metric_response(
        estimator: BaseEstimator,
        X_test: np.ndarray,
        metric_name: str,
        response_method: str,
        is_multiclass: bool,
    ) -> tuple[np.ndarray, bool]:
        """Return the estimator output required by a probability/ranking metric."""
        if response_method == "proba":
            if not hasattr(estimator, "predict_proba"):
                raise ValueError(
                    f"Metric '{metric_name}' requires predict_proba, but the "
                    "estimator does not provide it."
                )
            try:
                return estimator.predict_proba(X_test), True
            except Exception as exc:
                raise ValueError(
                    f"Metric '{metric_name}' requires predict_proba, but "
                    "predict_proba failed for this estimator."
                ) from exc

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
                    f"Metric '{metric_name}' requires predict_proba for "
                    "multiclass targets; decision_function fallback is only "
                    "supported for binary targets."
                )
            raise ValueError(
                f"Metric '{metric_name}' requires predict_proba or "
                "decision_function, but the estimator provides neither."
            )

        raise ValueError(
            f"Metric '{metric_name}' has unsupported response method "
            f"'{response_method}'."
        )

    @staticmethod
    def _compact_search_results(estimator: BaseEstimator) -> list[Dict[str, Any]]:
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
            row = {
                "candidate": idx,
                "params": dict(param_set),
            }
            if ranks is not None:
                row["rank_test_score"] = int(np.asarray(ranks)[idx])
            if means is not None:
                row["mean_test_score"] = float(np.asarray(means, dtype=float)[idx])
            if stds is not None:
                row["std_test_score"] = float(np.asarray(stds, dtype=float)[idx])
            rows.append(row)

        return rows

    def _metadata_feature_names(self, n_features: int) -> list[str]:
        """Return feature names aligned to a fitted feature-selection mask."""
        feature_names = getattr(self, "_feature_names", None)
        if feature_names is None or len(feature_names) != n_features:
            return [f"feature_{idx}" for idx in range(n_features)]
        return list(feature_names)

    @staticmethod
    def _compute_metric_safe(scorer, y_true, y_est, is_multiclass, is_proba=False):
        """
        Compute metric handling standard and temporal (diagonal) shapes.

        Shapes Handled
        --------------
        - **Standard**: y_est is (N,) or (N, C)
        - **Generalizing (Matrix)**:
          - y_pred: (N, T_train, T_test) -> Score each (T_train, T_test) pair.
          - y_proba: (N, C, T_train, T_test) -> Score each (T_train, T_test) pair.
        """
        # 1. Temporal / Sliding Case (Extra Dimension)
        # Check for (N, T) predictions or (N, C, T) probabilities
        is_temporal = (
            (y_est.ndim == 2 and not is_proba and y_true.ndim == 1)
            or y_est.ndim == 3
            or (y_est.ndim == 4 and is_proba)
        )

        if is_temporal:
            # Case A: Binary/Regression Predictions (N, T)
            if y_est.ndim == 2:
                # Iterate over time (dim 1)
                return np.array(
                    [scorer(y_true, y_est[:, t]) for t in range(y_est.shape[1])]
                )

            # Case B: Probabilities (N, C, T) or Generalizing (N, T_train, T_test)
            if y_est.ndim == 3:
                # Logic:
                # - If input is NOT proba, (N, T, T) implies Generalizing Predictions.
                # - If input IS proba, (N, C, T) implies Sliding Probabilities.

                if not is_proba:
                    # Generalizing Predictions (N, T_train, T_test)
                    n_train = y_est.shape[1]
                    n_test = y_est.shape[2]
                    matrix_scores = np.zeros((n_train, n_test))

                    for t_tr in range(n_train):
                        for t_te in range(n_test):
                            y_slice = y_est[:, t_tr, t_te]
                            matrix_scores[t_tr, t_te] = scorer(y_true, y_slice)
                    return matrix_scores

                # Sliding Probabilities (N, C, T)
                n_times = y_est.shape[2]
                scores = []
                for t in range(n_times):
                    slice_y = y_est[:, :, t]  # (N, C)

                    if not is_multiclass:
                        if slice_y.shape[1] == 2:
                            slice_y = slice_y[:, 1]

                    kwargs = {"multi_class": "ovr"} if is_multiclass else {}
                    scores.append(scorer(y_true, slice_y, **kwargs))
                return np.array(scores)

            # Case C: GenEst Probabilities (N, C, T_train, T_test) -> 4D
            if y_est.ndim == 4:
                n_train = y_est.shape[2]
                n_test = y_est.shape[3]
                matrix_scores = np.zeros((n_train, n_test))

                for t_tr in range(n_train):
                    for t_te in range(n_test):
                        slice_y = y_est[:, :, t_tr, t_te]  # (N, C)

                        if not is_multiclass:
                            if slice_y.shape[1] == 2:
                                slice_y = slice_y[:, 1]

                        kwargs = {"multi_class": "ovr"} if is_multiclass else {}
                        matrix_scores[t_tr, t_te] = scorer(y_true, slice_y, **kwargs)
                return matrix_scores

        # 2. Standard Case (N,) or (N, C)
        kwargs = {}
        if is_proba:
            if is_multiclass:
                kwargs = {"multi_class": "ovr"}
            elif y_est.ndim == 2 and y_est.shape[1] == 2:
                # Standard Binary Probabilities -> Take Positive Class
                y_est = y_est[:, 1]

        return scorer(y_true, y_est, **kwargs)

    def _force_serial_execution(self, estimator: BaseEstimator) -> BaseEstimator:
        """
        Recursively set n_jobs=1 for the estimator and its sub-components.
        Used when the outer loop is already parallelized to avoid oversubscription.
        """
        # 1. Get all parameters
        params = estimator.get_params()

        # 2. Identify keys ending in 'n_jobs'
        updates = {}
        for key, value in params.items():
            if key.endswith("n_jobs") and value is not None and value != 1:
                updates[key] = 1

        # 3. Apply updates
        if updates:
            estimator.set_params(**updates)

        return estimator

    @staticmethod
    def _extract_feature_importances(estimator: BaseEstimator) -> Optional[np.ndarray]:
        """
        Extract feature importances or coefficients from a fitted estimator.
        Handles Pipelines and Feature Selection.
        """
        # 1. Unwrap fitted hyperparameter search objects.
        if hasattr(estimator, "best_estimator_"):
            return Experiment._extract_feature_importances(estimator.best_estimator_)

        # 2. Unwrap Pipeline
        if isinstance(estimator, Pipeline):
            # Check for FS step
            fs_step = estimator.named_steps.get("fs")
            clf_step = estimator.named_steps.get("clf")

            # Get raw importances from classifier
            raw_imp = Experiment._extract_feature_importances(clf_step)
            if raw_imp is None:
                return None

            # Map back if FS was used
            if fs_step:
                support = fs_step.get_support()  # bool mask of selected features
                # We need to reconstruct the full importance array with zeros (or NaNs)
                # for unselected
                full_imp = np.zeros_like(support, dtype=float)
                full_imp[support] = raw_imp
                return full_imp

            return raw_imp

        # 3. Extract from Base Estimator
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        if hasattr(estimator, "coef_"):
            # Handle multi-class coefs (n_classes, n_features) -> take magnitude/mean?
            # For strict "importance", usually mean of abs(coefs) across classes
            if estimator.coef_.ndim > 1:
                return np.mean(np.abs(estimator.coef_), axis=0)
            return np.abs(estimator.coef_)

        return None


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

    def summary(self) -> pd.DataFrame:
        """
        Get a high-level summary of performance (Mean/Std across folds).

        Returns
        -------
        pd.DataFrame
            Index: Model Name
            Columns: Metric Mean/Std
        """
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
        """
        Get fold-level scores for all models in long format.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Metric, Value, Time, TrainTime, TestTime
        """
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue

            metrics_data = res["metrics"]
            # Assume all metrics have same number of folds
            n_folds = len(next(iter(metrics_data.values()))["folds"])

            for fold_idx in range(n_folds):
                for metric, stats in metrics_data.items():
                    rows.extend(
                        self._score_rows(
                            model, fold_idx, metric, stats["folds"][fold_idx]
                        )
                    )
        return pd.DataFrame(rows)

    def get_temporal_score_summary(self) -> pd.DataFrame:
        """
        Get temporal metric means/stds across folds in long format.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Metric, Time, TrainTime, TestTime, Mean, Std
        """
        rows = []
        columns = ["Model", "Metric", "Time", "TrainTime", "TestTime", "Mean", "Std"]

        for model, res in self.raw.items():
            if "error" in res:
                continue

            for metric, stats in res.get("metrics", {}).items():
                folds = [np.asarray(fold) for fold in stats.get("folds", [])]
                if not folds or folds[0].ndim == 0:
                    continue
                if any(fold.shape != folds[0].shape for fold in folds):
                    continue

                stack = np.stack(folds)
                mean = np.nanmean(stack, axis=0)
                std = np.nanstd(stack, axis=0)

                if mean.ndim == 1:
                    for time_idx, value in enumerate(mean):
                        rows.append(
                            {
                                "Model": model,
                                "Metric": metric,
                                "Time": self._time_value(time_idx),
                                "Mean": value,
                                "Std": std[time_idx],
                            }
                        )
                elif mean.ndim == 2:
                    for train_time in range(mean.shape[0]):
                        for test_time in range(mean.shape[1]):
                            rows.append(
                                {
                                    "Model": model,
                                    "Metric": metric,
                                    "TrainTime": self._time_value(train_time),
                                    "TestTime": self._time_value(test_time),
                                    "Mean": mean[train_time, test_time],
                                    "Std": std[train_time, test_time],
                                }
                            )

        return pd.DataFrame(rows, columns=columns)

    def get_predictions(self) -> pd.DataFrame:
        """
        Get concatenated predictions for all models.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, SampleIndex, SampleID, Group, y_true, y_pred,
            temporal coordinates, and probability columns when available.
        """
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue

            for fold_idx, preds in enumerate(res["predictions"]):
                rows.extend(self._prediction_rows(model, fold_idx, preds))

        return pd.DataFrame(rows)

    def get_splits(self) -> pd.DataFrame:
        """
        Get outer-CV train/test membership in long format.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Set, SampleIndex, SampleID, Group
        """
        rows = []
        columns = ["Model", "Fold", "Set", "SampleIndex", "SampleID", "Group"]

        for model, res in self.raw.items():
            if "error" in res:
                continue

            for fold_idx, split in enumerate(res.get("splits", [])):
                for set_name, idx_key, id_key, group_key in [
                    ("train", "train_idx", "train_sample_id", "train_group"),
                    ("test", "test_idx", "test_sample_id", "test_group"),
                ]:
                    indices = np.asarray(split[idx_key])
                    sample_ids = np.asarray(split[id_key])
                    groups = self._optional_values(split.get(group_key), len(indices))
                    for row_idx, sample_index in enumerate(indices):
                        rows.append(
                            {
                                "Model": model,
                                "Fold": fold_idx,
                                "Set": set_name,
                                "SampleIndex": sample_index,
                                "SampleID": sample_ids[row_idx],
                                "Group": groups[row_idx],
                            }
                        )

        return pd.DataFrame(rows, columns=columns)

    def get_feature_importances(self, fold_level: bool = False) -> pd.DataFrame:
        """
        Get feature importances in long format.

        Parameters
        ----------
        fold_level : bool
            If True, return one row per fold and feature. Otherwise return
            aggregate mean/std rows.
        """
        if fold_level:
            columns = ["Model", "Fold", "Feature", "FeatureName", "Importance"]
        else:
            columns = ["Model", "Feature", "FeatureName", "Mean", "Std"]

        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue
            importances = res.get("importances")
            if not importances:
                continue

            if fold_level:
                raw = np.asarray(importances.get("raw", []), dtype=float)
                if raw.ndim != 2:
                    continue
                feature_names = self._feature_names_for_result(res, raw.shape[1])
                for fold_idx, fold_values in enumerate(raw):
                    for feat_idx, value in enumerate(fold_values):
                        rows.append(
                            {
                                "Model": model,
                                "Fold": fold_idx,
                                "Feature": feat_idx,
                                "FeatureName": feature_names[feat_idx],
                                "Importance": value,
                            }
                        )
            else:
                means = np.asarray(importances.get("mean", []), dtype=float).ravel()
                stds = np.asarray(importances.get("std", []), dtype=float).ravel()
                if len(means) == 0:
                    continue
                feature_names = self._feature_names_for_result(res, len(means))
                if len(stds) != len(means):
                    stds = np.full(len(means), np.nan)
                for feat_idx, mean in enumerate(means):
                    rows.append(
                        {
                            "Model": model,
                            "Feature": feat_idx,
                            "FeatureName": feature_names[feat_idx],
                            "Mean": mean,
                            "Std": stds[feat_idx],
                        }
                    )

        return pd.DataFrame(rows, columns=columns)

    def _score_rows(
        self, model: str, fold_idx: int, metric: str, score: Any
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
            for time_idx, value in enumerate(score):
                rows.append(
                    {
                        "Model": model,
                        "Fold": fold_idx,
                        "Metric": metric,
                        "Time": self._time_value(time_idx),
                        "Value": value,
                    }
                )
            return rows

        if score.ndim == 2:
            for train_time in range(score.shape[0]):
                for test_time in range(score.shape[1]):
                    rows.append(
                        {
                            "Model": model,
                            "Fold": fold_idx,
                            "Metric": metric,
                            "TrainTime": self._time_value(train_time),
                            "TestTime": self._time_value(test_time),
                            "Value": score[train_time, test_time],
                        }
                    )
            return rows

        return [
            {
                "Model": model,
                "Fold": fold_idx,
                "Metric": metric,
                "Value": score,
            }
        ]

    def _prediction_rows(
        self, model: str, fold_idx: int, preds: Dict[str, Any]
    ) -> list[Dict[str, Any]]:
        """Expand scalar or temporal predictions into tidy rows."""
        y_true = np.asarray(preds["y_true"])
        y_pred = np.asarray(preds["y_pred"])
        y_proba = np.asarray(preds["y_proba"]) if "y_proba" in preds else None
        n_samples = len(y_true)
        sample_index = np.asarray(preds.get("sample_index", np.arange(n_samples)))
        sample_id = np.asarray(preds.get("sample_id", sample_index))
        groups = self._optional_values(preds.get("group"), n_samples)

        if y_pred.ndim == 2 and y_true.ndim == 1:
            return self._sliding_prediction_rows(
                model,
                fold_idx,
                y_true,
                y_pred,
                y_proba,
                sample_index,
                sample_id,
                groups,
            )

        if y_pred.ndim == 3 and y_true.ndim == 1:
            return self._generalizing_prediction_rows(
                model,
                fold_idx,
                y_true,
                y_pred,
                y_proba,
                sample_index,
                sample_id,
                groups,
            )

        rows = []
        for row_idx in range(n_samples):
            row = self._prediction_base_row(
                model, fold_idx, row_idx, y_true, sample_index, sample_id, groups
            )
            row["y_pred"] = self._row_value(y_pred, row_idx)
            if y_proba is not None:
                self._add_standard_proba(row, y_proba, row_idx)
            rows.append(row)
        return rows

    def _sliding_prediction_rows(
        self,
        model: str,
        fold_idx: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        sample_index: np.ndarray,
        sample_id: np.ndarray,
        groups: np.ndarray,
    ) -> list[Dict[str, Any]]:
        rows = []
        for row_idx in range(len(y_true)):
            for time_idx in range(y_pred.shape[1]):
                row = self._prediction_base_row(
                    model, fold_idx, row_idx, y_true, sample_index, sample_id, groups
                )
                row["Time"] = self._time_value(time_idx)
                row["y_pred"] = y_pred[row_idx, time_idx]
                if (
                    y_proba is not None
                    and y_proba.ndim == 3
                    and y_proba.shape[0] == len(y_true)
                    and y_proba.shape[2] == y_pred.shape[1]
                ):
                    for class_idx in range(y_proba.shape[1]):
                        row[f"y_proba_{class_idx}"] = y_proba[
                            row_idx, class_idx, time_idx
                        ]
                rows.append(row)
        return rows

    def _generalizing_prediction_rows(
        self,
        model: str,
        fold_idx: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        sample_index: np.ndarray,
        sample_id: np.ndarray,
        groups: np.ndarray,
    ) -> list[Dict[str, Any]]:
        rows = []
        for row_idx in range(len(y_true)):
            for train_time in range(y_pred.shape[1]):
                for test_time in range(y_pred.shape[2]):
                    row = self._prediction_base_row(
                        model,
                        fold_idx,
                        row_idx,
                        y_true,
                        sample_index,
                        sample_id,
                        groups,
                    )
                    row["TrainTime"] = self._time_value(train_time)
                    row["TestTime"] = self._time_value(test_time)
                    row["y_pred"] = y_pred[row_idx, train_time, test_time]
                    if (
                        y_proba is not None
                        and y_proba.ndim == 4
                        and y_proba.shape[0] == len(y_true)
                        and y_proba.shape[2] == y_pred.shape[1]
                        and y_proba.shape[3] == y_pred.shape[2]
                    ):
                        for class_idx in range(y_proba.shape[1]):
                            row[f"y_proba_{class_idx}"] = y_proba[
                                row_idx, class_idx, train_time, test_time
                            ]
                    rows.append(row)
        return rows

    @staticmethod
    def _prediction_base_row(
        model: str,
        fold_idx: int,
        row_idx: int,
        y_true: np.ndarray,
        sample_index: np.ndarray,
        sample_id: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "Model": model,
            "Fold": fold_idx,
            "SampleIndex": sample_index[row_idx],
            "SampleID": sample_id[row_idx],
            "Group": groups[row_idx],
            "y_true": ExperimentResult._row_value(y_true, row_idx),
        }

    @staticmethod
    def _row_value(values: np.ndarray, row_idx: int) -> Any:
        value = values[row_idx]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _add_standard_proba(row: Dict[str, Any], y_proba: np.ndarray, row_idx: int):
        if y_proba.ndim == 1:
            row["y_proba"] = y_proba[row_idx]
        elif y_proba.ndim == 2:
            for class_idx in range(y_proba.shape[1]):
                row[f"y_proba_{class_idx}"] = y_proba[row_idx, class_idx]

    @staticmethod
    def _optional_values(values: Optional[Any], length: int) -> np.ndarray:
        if values is None:
            return np.full(length, None, dtype=object)
        return np.asarray(values)

    def _time_axis(self) -> Optional[list[Any]]:
        time_axis = self.meta.get("time_axis")
        if time_axis is None:
            return None
        return list(time_axis)

    def _time_value(self, index: int) -> Any:
        time_axis = self._time_axis()
        if time_axis is None or index >= len(time_axis):
            return index
        return time_axis[index]

    @staticmethod
    def _feature_names_for_result(res: Dict[str, Any], n_features: int) -> list[str]:
        importances = res.get("importances")
        if importances:
            feature_names = importances.get("feature_names")
            if feature_names is not None and len(feature_names) == n_features:
                return list(feature_names)

        for meta in res.get("metadata", []):
            feature_names = meta.get("feature_names")
            if feature_names is not None and len(feature_names) == n_features:
                return list(feature_names)
        return [f"feature_{idx}" for idx in range(n_features)]

    def get_best_params(self) -> pd.DataFrame:
        """
        Get the best hyperparameters selected per fold (if Tuning was enabled).

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Param, Value
        """
        rows = []
        for model_name, res in self.raw.items():
            if "error" in res:
                continue

            # Check if metadata exists.
            if "metadata" in res:
                for fold_idx, meta in enumerate(res["metadata"]):
                    if "best_params" in meta:
                        for p_name, p_val in meta["best_params"].items():
                            rows.append(
                                {
                                    "Model": model_name,
                                    "Fold": fold_idx,
                                    "Param": p_name,
                                    "Value": p_val,
                                }
                            )

        return pd.DataFrame(rows)

    def get_search_results(self) -> pd.DataFrame:
        """
        Get compact hyperparameter-search diagnostics in long form.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Candidate, Rank, MeanTestScore, StdTestScore,
            Params
        """
        rows = []
        columns = [
            "Model",
            "Fold",
            "Candidate",
            "Rank",
            "MeanTestScore",
            "StdTestScore",
            "Params",
        ]

        for model_name, res in self.raw.items():
            if "error" in res:
                continue

            for fold_idx, meta in enumerate(res.get("metadata", [])):
                for search_row in meta.get("search_results", []):
                    rows.append(
                        {
                            "Model": model_name,
                            "Fold": fold_idx,
                            "Candidate": search_row.get("candidate"),
                            "Rank": search_row.get("rank_test_score"),
                            "MeanTestScore": search_row.get("mean_test_score"),
                            "StdTestScore": search_row.get("std_test_score"),
                            "Params": search_row.get("params"),
                        }
                    )

        return pd.DataFrame(rows, columns=columns)

    def get_selected_features(self) -> pd.DataFrame:
        """
        Get fold-level selected feature masks in long format.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Feature, FeatureName, Selected
        """
        rows = []
        columns = ["Model", "Fold", "Feature", "FeatureName", "Selected"]

        for model_name, res in self.raw.items():
            if "error" in res:
                continue

            for fold_idx, meta in enumerate(res.get("metadata", [])):
                if "selected_features" not in meta:
                    continue

                mask = np.asarray(meta["selected_features"], dtype=bool)
                feature_names = meta.get("feature_names")
                if feature_names is None or len(feature_names) != len(mask):
                    feature_names = [f"feature_{idx}" for idx in range(len(mask))]

                for feat_idx, selected in enumerate(mask):
                    rows.append(
                        {
                            "Model": model_name,
                            "Fold": fold_idx,
                            "Feature": feat_idx,
                            "FeatureName": feature_names[feat_idx],
                            "Selected": bool(selected),
                        }
                    )

        return pd.DataFrame(rows, columns=columns)

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get fold-level feature-selection scores when the selector exposes them.

        ``SelectKBest`` exposes univariate scores and, for the default
        ``f_classif`` / ``f_regression`` functions, p-values. SFS does not expose
        stable per-feature scores, so SFS folds do not appear in this table.

        Returns
        -------
        pd.DataFrame
            Columns: Model, Fold, Feature, FeatureName, Selector, Score,
            PValue, Selected
        """
        rows = []
        columns = [
            "Model",
            "Fold",
            "Feature",
            "FeatureName",
            "Selector",
            "Score",
            "PValue",
            "Selected",
        ]

        for model_name, res in self.raw.items():
            if "error" in res:
                continue

            for fold_idx, meta in enumerate(res.get("metadata", [])):
                if "feature_scores" not in meta:
                    continue

                scores = np.asarray(meta["feature_scores"], dtype=float)
                pvalues = meta.get("feature_pvalues")
                if pvalues is not None:
                    pvalues = np.asarray(pvalues, dtype=float)

                feature_names = meta.get("feature_names")
                if feature_names is None or len(feature_names) != len(scores):
                    feature_names = [f"feature_{idx}" for idx in range(len(scores))]

                selected = meta.get("selected_features")
                if selected is not None:
                    selected = np.asarray(selected, dtype=bool)

                for feat_idx, score in enumerate(scores):
                    rows.append(
                        {
                            "Model": model_name,
                            "Fold": fold_idx,
                            "Feature": feat_idx,
                            "FeatureName": feature_names[feat_idx],
                            "Selector": meta.get("feature_selection_method"),
                            "Score": score,
                            "PValue": (
                                pvalues[feat_idx]
                                if pvalues is not None and len(pvalues) == len(scores)
                                else np.nan
                            ),
                            "Selected": (
                                bool(selected[feat_idx])
                                if selected is not None and len(selected) == len(scores)
                                else np.nan
                            ),
                        }
                    )

        return pd.DataFrame(rows, columns=columns)

    def get_feature_stability(self) -> pd.DataFrame:
        """
        Analyze feature selection stability across folds.

        Returns
        -------
        pd.DataFrame
            Index: Feature Index/Name
            Columns: Selection Frequency (0.0 - 1.0)
        """
        rows = []
        for model_name, res in self.raw.items():
            if "error" in res:
                continue

            if "metadata" in res:
                # Collect masks
                masks = []
                feature_names = None
                for meta in res["metadata"]:
                    if "selected_features" in meta:
                        masks.append(meta["selected_features"])
                        if feature_names is None and "feature_names" in meta:
                            feature_names = meta["feature_names"]

                if masks:
                    # Stack: (n_folds, n_features)
                    stack = np.vstack(masks)
                    stability = np.mean(stack, axis=0)  # 0 to 1

                    for feat_idx, freq in enumerate(stability):
                        row = {
                            "Model": model_name,
                            "Feature": feat_idx,
                            "Frequency": freq,
                        }
                        if feature_names is not None and len(feature_names) == len(
                            stability
                        ):
                            row["FeatureName"] = feature_names[feat_idx]
                        rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def get_generalization_matrix(self, metric: str = None) -> pd.DataFrame:
        """
        Get Generalization Matrix (Train Time x Test Time) averaged across folds.

        Parameters
        ----------
        metric : str, optional
            The metric to retrieve (e.g., 'accuracy', 'roc_auc').
            Defaults to the first metric found in results.

        Returns
        -------
        pd.DataFrame
            Index: Train Time
            Columns: Test Time
            Values: Average Score
        """
        # 1. Collect all matrices for the metric
        for model_name, res in self.raw.items():
            if "error" in res:
                continue

            metrics_data = res["metrics"]
            if metric is None:
                metric = list(metrics_data.keys())[0]

            if metric not in metrics_data:
                continue

            fold_scores = metrics_data[metric]["folds"]
            # Check if scores are matrices (2D arrays)
            valid_matrices = [
                s for s in fold_scores if isinstance(s, np.ndarray) and s.ndim == 2
            ]

            if valid_matrices:
                # Stack and Mean -> (n_folds, n_train, n_test) -> (n_train, n_test)
                stack = np.stack(valid_matrices)
                mean_matrix = np.mean(stack, axis=0)
                train_axis = [
                    self._time_value(idx) for idx in range(mean_matrix.shape[0])
                ]
                test_axis = [
                    self._time_value(idx) for idx in range(mean_matrix.shape[1])
                ]
                return pd.DataFrame(mean_matrix, index=train_axis, columns=test_axis)

        return pd.DataFrame()
