"""
Decoding Experiment
===================
Main executor for decoding experiments.
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
from .capabilities import (
    canonical_estimator_name,
    get_selector_capabilities,
    resolve_estimator_capabilities,
    resolve_estimator_spec,
)
from .configs import ExperimentConfig
from .constants import GROUP_CV_STRATEGIES, RESULT_SCHEMA_VERSION
from .engine import fit_and_score_fold
from .metrics import get_metric_names, get_metric_spec
from .registry import get_estimator_cls
from .result import ExperimentResult
from .splitters import get_cv_splitter

logger = logging.getLogger(__name__)


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
        self.result_: Optional[ExperimentResult] = None
        self._model_specs: Dict[str, Any] = {}
        self._model_capabilities: Dict[str, Any] = {}
        self._propagate_random_state()
        self._validate_config()

    def _validate_config(self):
        """Perform comprehensive runtime validation of the configuration."""
        task = self.config.task
        if self.config.tuning.enabled and not self.config.grids:
            logger.warning(
                "Hyperparameter tuning is enabled but no 'grids' are defined "
                "in the config."
            )
        if self.config.calibration.enabled and task != "classification":
            raise ValueError(
                "Probability calibration is only available for classification."
            )

        self._validate_inner_cv_overrides()

        for metric in self.config.metrics:
            if get_metric_spec(metric).task != task:
                raise ValueError(
                    f"Metric '{metric}' is incompatible with task '{task}'. "
                    f"Available {task} metrics: {get_metric_names(task)}."
                )

        for metric in self._evaluation_metrics():
            if get_metric_spec(metric).task != task:
                raise ValueError(
                    f"Statistical assessment metric '{metric}' is incompatible "
                    f"with task '{task}'."
                )

        if task == "regression" and "stratified" in self.config.cv.strategy:
            raise ValueError(
                f"CV strategy '{self.config.cv.strategy}' is invalid "
                "for regression tasks."
            )

        for name, model_cfg in self.config.models.items():
            spec = resolve_estimator_spec(model_cfg)
            caps = spec.to_capabilities()
            self._model_specs[name] = spec
            self._model_capabilities[name] = caps
            if not caps.supports_task(task):
                raise ValueError(f"Model '{name}' does not support task '{task}'.")

    def _prepare_estimator(self, model_name: str, model_config: Any) -> BaseEstimator:
        """Orchestrate the creation of the full Estimator Pipeline."""
        self._validate_metric_capabilities(model_name, model_config)
        full_est = self._instantiate_model(model_name, model_config)
        steps = []
        allow_prep = self._allows_pipeline_preprocessing(model_config)

        if self.config.use_scaler and allow_prep:
            steps.append(("scaler", StandardScaler()))

        if self.config.feature_selection.enabled and allow_prep:
            fs_step = self._create_fs_step(full_est)
            if fs_step:
                steps.append(fs_step)
        elif self.config.feature_selection.enabled and not allow_prep:
            raise ValueError(
                "Feature selection is only valid for classical 2D tabular "
                "or embedding inputs."
            )

        steps.append(("clf", full_est))

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

        if (
            self.config.tuning.enabled
            and self.config.grids
            and model_name in self.config.grids
        ):
            est = self._wrap_with_tuning(est, model_name)

        if self.config.calibration.enabled:
            est = self._wrap_with_calibration(est)
        return est

    def _resolved_tuning_cv(self):
        return self.config.tuning.cv or self._outer_cv_copy()

    def _resolved_feature_selection_cv(self):
        fs_conf = self.config.feature_selection
        if fs_conf.cv is not None:
            return fs_conf.cv
        if self.config.tuning.enabled:
            return self._resolved_tuning_cv()
        return self._outer_cv_copy()

    def _resolved_calibration_cv(self):
        return self.config.calibration.cv or self._outer_cv_copy()

    def _outer_cv_copy(self):
        return self.config.cv.model_copy(deep=True)

    @staticmethod
    def _allows_pipeline_preprocessing(model_config: Any) -> bool:
        if getattr(model_config, "kind", None) != "classical":
            return False
        return getattr(model_config, "input_kind", "tabular") in {
            "tabular",
            "embeddings",
        }

    def _propagate_random_state(self):
        """Propagate the global random_state to all components if set."""
        global_seed = self.config.random_state
        if global_seed is None:
            return

        from numpy.random import SeedSequence

        ss = SeedSequence(global_seed)

        # Derive seeds for main blocks (stable order)
        # 0: cv, 1: tuning, 2: feature_selection, 3: evaluation, 4: models
        child_seeds = ss.spawn(5)

        self.config.cv.random_state = int(child_seeds[0].generate_state(1)[0])
        self.config.tuning.random_state = int(child_seeds[1].generate_state(1)[0])
        self.config.feature_selection.random_state = int(
            child_seeds[2].generate_state(1)[0]
        )
        self.config.evaluation.random_state = int(child_seeds[3].generate_state(1)[0])

        # Models
        model_names = sorted(self.config.models.keys())
        model_seeds = child_seeds[4].spawn(len(model_names))
        for name, seed in zip(model_names, model_seeds):
            cfg = self.config.models[name]
            derived_seed = int(seed.generate_state(1)[0])

            # Handle standard models with explicit fields
            if hasattr(cfg, "random_state"):
                cfg.random_state = derived_seed

            # Handle ClassicalModelConfig by injecting into params if supported
            if getattr(cfg, "kind", None) == "classical" and hasattr(cfg, "params"):
                spec = resolve_estimator_spec(cfg)
                if spec.supports_random_state:
                    cfg.params["random_state"] = derived_seed

            # Handle temporal wrappers
            if hasattr(cfg, "base") and hasattr(cfg.base, "random_state"):
                cfg.base.random_state = derived_seed
            if (
                hasattr(cfg, "base")
                and getattr(cfg.base, "kind", None) == "classical"
                and hasattr(cfg.base, "params")
            ):
                spec = resolve_estimator_spec(cfg.base)
                if spec.supports_random_state:
                    cfg.base.params["random_state"] = derived_seed

            # Handle neural wrappers
            if hasattr(cfg, "head") and hasattr(cfg.head, "random_state"):
                cfg.head.random_state = derived_seed
            if (
                hasattr(cfg, "head")
                and getattr(cfg.head, "kind", None) == "classical"
                and hasattr(cfg.head, "params")
            ):
                spec = resolve_estimator_spec(cfg.head)
                if spec.supports_random_state:
                    cfg.head.params["random_state"] = derived_seed

    def _validate_inner_cv_overrides(self) -> None:
        if self.config.cv.strategy not in GROUP_CV_STRATEGIES:
            return
        checks = []
        if self.config.tuning.enabled:
            checks.append(
                (
                    "tuning.cv",
                    self._resolved_tuning_cv(),
                    self.config.tuning.cv is not None,
                    self.config.tuning.allow_nongroup_inner_cv,
                )
            )
        fs_conf = self.config.feature_selection
        if fs_conf.enabled and fs_conf.method == "sfs":
            inherited = (
                fs_conf.cv is None
                and self.config.tuning.enabled
                and self.config.tuning.cv is not None
            )
            allowed = fs_conf.allow_nongroup_inner_cv or (
                inherited and self.config.tuning.allow_nongroup_inner_cv
            )
            checks.append(
                (
                    "feature_selection.cv",
                    self._resolved_feature_selection_cv(),
                    fs_conf.cv is not None or inherited,
                    allowed,
                )
            )
        if self.config.calibration.enabled:
            checks.append(
                (
                    "calibration.cv",
                    self._resolved_calibration_cv(),
                    self.config.calibration.cv is not None,
                    self.config.calibration.allow_nongroup_inner_cv,
                )
            )

        for name, cv_cfg, explicit, allowed in checks:
            if cv_cfg.strategy in GROUP_CV_STRATEGIES:
                continue
            if explicit and allowed:
                continue
            raise ValueError(
                f"Outer CV strategy is group-based, but {name} strategy "
                f"'{cv_cfg.strategy}' is not. Set "
                "allow_nongroup_inner_cv=True to acknowledge leakage."
            )

    def _wrap_with_calibration(self, estimator: BaseEstimator) -> BaseEstimator:
        from sklearn.calibration import CalibratedClassifierCV

        cv = get_cv_splitter(self._resolved_calibration_cv(), require_groups=False)
        return CalibratedClassifierCV(
            estimator=estimator,
            method=self.config.calibration.method,
            cv=cv,
            n_jobs=self.config.calibration.n_jobs,
        )

    def _validate_metric_capabilities(self, model_name: str, model_config: Any) -> None:
        caps = resolve_estimator_capabilities(model_config)
        for metric in self.config.metrics:
            spec = get_metric_spec(metric)
            if (
                spec.response_method == "proba"
                and not self.config.calibration.enabled
                and not caps.has_response("predict_proba")
            ):
                raise ValueError(
                    f"Metric '{metric}' requires predict_proba, but model "
                    f"'{model_name}' doesn't provide it."
                )

    def _instantiate_model(self, name: str, config: Any) -> BaseEstimator:
        kind = getattr(config, "kind", None)
        if kind == "classical":
            est_cls = get_estimator_cls(canonical_estimator_name(config.estimator))
            return est_cls(**config.params)
        if kind == "frozen_backbone":
            from .neural import FrozenBackboneDecoder

            return FrozenBackboneDecoder(config.backbone, config.head, self.config.task)
        if kind == "neural_finetune":
            from .neural import NeuralFineTuneEstimator

            return NeuralFineTuneEstimator(
                **config.model_dump(exclude={"kind"}), task=self.config.task
            )
        if kind == "foundation_embedding":
            from .embedding_extractors import build_embedding_extractor

            return build_embedding_extractor(config)
        if kind == "temporal":
            method = (
                "SlidingEstimator"
                if config.wrapper == "sliding"
                else "GeneralizingEstimator"
            )
            est_cls = get_estimator_cls(method)
            params = config.model_dump(exclude={"kind", "wrapper", "base"})
            params["base_estimator"] = self._prepare_estimator(
                f"{name}_base", config.base
            )
            return est_cls(**params)
        est_cls = get_estimator_cls(config.method)
        params = config.model_dump(exclude={"method"})
        if "base_estimator" in params:
            params["base_estimator"] = self._prepare_estimator(
                f"{name}_base", params["base_estimator"]
            )
        return est_cls(**params)

    def _create_fs_step(self, estimator: BaseEstimator) -> Optional[tuple]:
        fs_conf = self.config.feature_selection
        if fs_conf.method == "k_best":
            score_func = (
                f_classif if self.config.task == "classification" else f_regression
            )
            return (
                "fs",
                SelectKBest(score_func=score_func, k=fs_conf.n_features or "all"),
            )
        if fs_conf.method == "sfs":
            cv = get_cv_splitter(
                self._resolved_feature_selection_cv(), require_groups=False
            )
            return (
                "fs",
                SequentialFeatureSelector(
                    estimator=clone(estimator),
                    n_features_to_select=fs_conf.n_features,
                    direction=fs_conf.direction,
                    cv=cv,
                    scoring=self._resolve_fs_scoring(),
                    n_jobs=self.config.n_jobs,
                ),
            )
        return None

    def _resolve_fs_scoring(self) -> str:
        return (
            self.config.feature_selection.scoring
            or self.config.tuning.scoring
            or self.config.metrics[0]
        )

    def _wrap_with_tuning(self, estimator: BaseEstimator, name: str) -> BaseEstimator:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        grid = self.config.grids[name]
        mapped = {k if "__" in k else f"clf__{k}": v for k, v in grid.items()}
        valid = estimator.get_params(deep=True)
        invals = [k for k in mapped if k not in valid]
        if invals:
            raise ValueError(f"Invalid tuning keys for '{name}': {invals}")
        cv = get_cv_splitter(self._resolved_tuning_cv(), require_groups=False)
        kwargs = {
            "estimator": estimator,
            "cv": cv,
            "scoring": self.config.tuning.scoring or self.config.metrics[0],
            "n_jobs": self.config.tuning.n_jobs,
            "refit": True,
        }
        if self.config.tuning.search_type == "grid":
            return GridSearchCV(param_grid=mapped, **kwargs)
        return RandomizedSearchCV(
            param_distributions=mapped,
            n_iter=self.config.tuning.n_iter,
            random_state=self.config.tuning.random_state,
            **kwargs,
        )

    def run(
        self,
        X: np.ndarray,
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        sample_metadata: Optional[Union[pd.DataFrame, Dict[str, Sequence[Any]]]] = None,
        observation_level: str = "sample",
        inferential_unit: Optional[str] = None,
        time_axis: Optional[Sequence[Any]] = None,
    ) -> ExperimentResult:
        """Execute the full experiment pipeline."""
        start_time = time.time()
        logger.info(f"Starting Experiment: Task={self.config.task}")
        X, y = np.asarray(X), np.asarray(y)
        if len(X) == 0:
            raise ValueError("X is empty.")
        if len(y) != len(X):
            raise ValueError("Length mismatch between X and y.")

        self._feature_names = self._resolve_feature_names(X, feature_names)
        self._sample_ids = self._resolve_sample_ids(len(X), sample_ids)
        if observation_level not in {"sample", "epoch"}:
            raise ValueError("observation_level must be 'sample' or 'epoch'.")
        self._sample_metadata = self._resolve_sample_metadata(len(X), sample_metadata)
        self._sample_metadata, groups = self._resolve_group_metadata(
            len(X), self._sample_metadata, groups
        )
        self._observation_level, self._inferential_unit = (
            observation_level,
            self._resolve_inferential_unit(
                observation_level, inferential_unit, self._sample_metadata
            ),
        )
        self._time_axis = self._resolve_time_axis(X, time_axis)

        self._validate_input_capabilities(X)
        self._validate_groups_for_cv(groups)
        if self.config.task == "classification" and type_of_target(y) == "continuous":
            raise ValueError("Task is 'classification' but target is 'continuous'.")

        for name, cfg in self.config.models.items():
            logger.info(f"Evaluating Model: {name} ({self._model_label(cfg)})")
            try:
                est = self._prepare_estimator(name, cfg)
                self.results[name] = self._cross_validate(
                    est, X, y, groups, self._sample_ids, self._sample_metadata
                )
            except Exception as e:
                logger.error(f"Failed model '{name}': {e}", exc_info=True)
                self.results[name] = {"error": str(e), "status": "failed"}

        logger.info(f"Experiment Completed in {time.time() - start_time:.2f}s")
        res_obj = ExperimentResult(
            self.results,
            config=self.config.model_dump(),
            meta=self._build_result_meta(X, self._time_axis),
        )

        if self.config.evaluation.enabled:
            from .stats import run_statistical_assessment

            assessment = run_statistical_assessment(
                res_obj,
                self.config,
                X,
                y,
                groups,
                self._sample_ids,
                self._sample_metadata,
                self._feature_names,
                self._time_axis,
                self._observation_level,
                self._inferential_unit,
            )
            res_obj.meta["statistical_assessment"] = assessment["meta"]
            for m_name, m_res in res_obj.raw.items():
                if "error" in m_res:
                    continue
                m_res["statistical_assessment"] = [
                    r for r in assessment["rows"] if r.get("Model") == m_name
                ]
                if m_name in assessment["nulls"]:
                    m_res["statistical_nulls"] = assessment["nulls"][m_name]
        return res_obj

    def _cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        sample_ids: np.ndarray,
        sample_metadata: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        cv = get_cv_splitter(self.config.cv, groups=groups, y=y)
        splits = list(cv.split(X, y, groups))
        est = clone(estimator)
        force_serial = self.config.n_jobs != 1

        parallel = joblib.Parallel(
            n_jobs=self.config.n_jobs, verbose=self.config.verbose
        )
        results = parallel(
            joblib.delayed(fit_and_score_fold)(
                clone(est),
                X,
                y,
                groups,
                sample_ids,
                sample_metadata,
                train_idx,
                test_idx,
                metrics=self.config.metrics,
                feature_selection_config=self.config.feature_selection,
                calibration_config=self.config.calibration,
                feature_names=self._feature_names,
                force_serial=force_serial,
            )
            for train_idx, test_idx in splits
        )

        fold_scores = defaultdict(list)
        f_idx, f_preds, f_imps, f_meta, f_splits, f_diags = [], [], [], [], [], []
        for res in results:
            f_idx.append(res["test_idx"])
            f_preds.append(res["preds"])
            f_imps.append(res["importance"])
            f_meta.append(res.get("metadata", {}))
            f_splits.append(res["split"])
            f_diags.append(res.get("diagnostics", {}))
            for m, s in res["scores"].items():
                fold_scores[m].append(s)

        metrics = {
            m: {"mean": np.nanmean(s), "std": np.nanstd(s), "folds": s}
            for m, s in fold_scores.items()
        }
        valid_imps = [f for f in f_imps if f is not None]
        agg_imp = None
        if valid_imps and all(imp.shape == valid_imps[0].shape for imp in valid_imps):
            stack = np.vstack(valid_imps)
            agg_imp = {
                "mean": np.mean(stack, axis=0),
                "std": np.std(stack, axis=0),
                "raw": stack,
                "feature_names": self._metadata_feature_names(stack.shape[1]),
            }

        return {
            "metrics": metrics,
            "predictions": f_preds,
            "indices": f_idx,
            "importances": agg_imp,
            "metadata": f_meta,
            "splits": f_splits,
            "diagnostics": f_diags,
        }

    @staticmethod
    def _resolve_sample_ids(n: int, ids: Optional[Sequence[Any]]) -> np.ndarray:
        if ids is None:
            return np.arange(n)
        ids = np.asarray(ids)
        if len(ids) != n:
            raise ValueError(f"sample_ids length mismatch: {len(ids)} vs {n}")
        if len(pd.unique(ids)) != n:
            raise ValueError("sample_ids must be unique.")
        return ids

    @staticmethod
    def _resolve_sample_metadata(
        n: int, meta: Optional[Union[pd.DataFrame, Dict[str, Sequence[Any]]]]
    ) -> Optional[pd.DataFrame]:
        if meta is None:
            return None
        df = pd.DataFrame(meta).reset_index(drop=True)
        if len(df) != n:
            raise ValueError("sample_metadata length mismatch.")
        miss = sorted({"subject", "session"} - set(df.columns))
        if miss:
            raise ValueError(f"sample_metadata missing {miss}.")
        if "site" not in df.columns:
            df["site"] = None
        return df

    def _resolve_group_metadata(
        self, n: int, meta: Optional[pd.DataFrame], groups: Optional[np.ndarray]
    ) -> tuple:
        key = self.config.cv.group_key
        if groups is not None:
            gv = np.asarray(groups)
            if len(gv) != n:
                raise ValueError("groups length mismatch.")
            if key is not None:
                if meta is None:
                    meta = pd.DataFrame({key: gv})
                elif key not in meta:
                    meta[key] = gv
            return meta, gv
        if key is not None:
            if meta is None or key not in meta:
                raise ValueError(f"group_key '{key}' missing.")
            return meta, meta[key].to_numpy()
        return meta, None

    def _resolve_inferential_unit(
        self, level: str, unit: Optional[str], meta: Optional[pd.DataFrame]
    ) -> str:
        if unit is not None:
            return unit
        return "subject" if level == "epoch" and meta is not None else "sample"

    def _resolve_time_axis(
        self, X: np.ndarray, axis: Optional[Sequence[Any]]
    ) -> Optional[np.ndarray]:
        if X.ndim != 3:
            return np.asarray(axis) if axis is not None else None
        if axis is None:
            return np.arange(X.shape[-1])
        axis = np.asarray(axis)
        if len(axis) != X.shape[-1]:
            raise ValueError("time_axis length mismatch.")
        return axis

    def _build_result_meta(
        self, X: np.ndarray, t_axis: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        meta = get_environment_info()
        meta.update(
            {
                "tag": self.config.tag,
                "task": self.config.task,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
                "observation_level": getattr(self, "_observation_level", "sample"),
                "inferential_unit": getattr(self, "_inferential_unit", "sample"),
                "run_manifest": {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "model_names": list(self.config.models),
                    "cv_strategy": self.config.cv.strategy,
                    "metrics": list(self.config.metrics),
                },
                "hardware_provenance": {"n_jobs": self.config.n_jobs},
                "capabilities": self._capability_payload(),
            }
        )
        if t_axis is not None:
            meta["time_axis"] = t_axis.tolist()
        return meta

    def _capability_payload(self) -> Dict[str, Any]:
        sels = {}
        if self.config.feature_selection.enabled:
            sels[self.config.feature_selection.method] = get_selector_capabilities(
                self.config.feature_selection.method
            ).to_dict()
        return {
            "models": {n: c.to_dict() for n, c in self._model_capabilities.items()},
            "estimator_specs": {n: s.to_dict() for n, s in self._model_specs.items()},
            "feature_selectors": sels,
            "metrics": {
                m: {
                    "task": get_metric_spec(m).task,
                    "response_method": get_metric_spec(m).response_method,
                    "family": get_metric_spec(m).family,
                }
                for m in self.config.metrics
            },
        }

    def _validate_input_capabilities(self, X: np.ndarray) -> None:
        rank = "3d_temporal" if X.ndim == 3 else "2d"
        for n, c in self._model_capabilities.items():
            if rank not in c.input_ranks:
                raise ValueError(f"Model '{n}' doesn't support rank '{rank}'.")
        if self.config.feature_selection.enabled:
            sel = get_selector_capabilities(self.config.feature_selection.method)
            if rank not in sel.input_ranks:
                raise ValueError(
                    f"FS method '{sel.method}' doesn't support rank '{rank}'."
                )

    def _validate_groups_for_cv(self, groups: Optional[np.ndarray]) -> None:
        if (
            self.config.cv.strategy in GROUP_CV_STRATEGIES
            and not self.config.cv.group_key
        ):
            raise ValueError(
                f"Strategy '{self.config.cv.strategy}' requires group_key."
            )
        if groups is not None:
            return
        if self.config.cv.strategy in GROUP_CV_STRATEGIES:
            raise ValueError("Outer CV requires groups.")
        if (
            self.config.tuning.enabled
            and self._resolved_tuning_cv().strategy in GROUP_CV_STRATEGIES
        ):
            raise ValueError("Tuning CV requires groups.")

    def save_results(self, path: Optional[Union[str, Path]] = None):
        if path is None:
            path = self.config.output_dir
            if path is None:
                raise ValueError("No output path specified.")
        path = Path(path)

        if self.result_ is not None:
            res_obj = self.result_
        else:
            res_obj = ExperimentResult(
                self.results,
                config=self.config.model_dump(),
                meta=get_environment_info(),
                schema_version=RESULT_SCHEMA_VERSION,
            )

        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            target = (
                path
                / f"{self.config.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path

        logger.info(f"Saving results to {target}")
        if target.suffix == ".json":
            res_obj.save_json(target)
        else:
            joblib.dump(res_obj.to_payload(), target)
        return target

    @staticmethod
    def load_results(path: Union[str, Path]) -> ExperimentResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        if path.suffix == ".json":
            return ExperimentResult.load_json(path)

        payload = joblib.load(path)
        return ExperimentResult(
            payload["results"],
            config=payload.get("config"),
            meta=payload.get("meta"),
            schema_version=payload.get("schema_version", RESULT_SCHEMA_VERSION),
        )

    def _metadata_feature_names(self, n: int) -> list[str]:
        names = getattr(self, "_feature_names", None)
        return (
            list(names)
            if names is not None and len(names) == n
            else [f"feature_{idx}" for idx in range(n)]
        )

    def _resolve_feature_names(
        self, X: np.ndarray, names: Optional[Sequence[str]]
    ) -> list[str]:
        exp = 1 if X.ndim < 2 else X.shape[1]
        if names is not None:
            if len(names) != exp:
                raise ValueError(
                    f"feature_names length mismatch: {len(names)} vs {exp}"
                )
            return [str(n) for n in names]
        return (
            ["feature_0"]
            if X.ndim < 2
            else [f"feature_{idx}" for idx in range(X.shape[1])]
        )

    def _evaluation_metrics(self) -> list[str]:
        eval_cfg = self.config.evaluation
        ms = []
        if eval_cfg.metrics:
            ms.extend(eval_cfg.metrics)
        return sorted(set(ms))
