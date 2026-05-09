"""
Typed estimator and capability metadata for decoding.

Estimator specs are the single source of truth for lazy imports, lightweight
capability checks, fit-smoke policy, dependency extras, and default search
spaces. Detailed estimator parameter validation remains delegated to sklearn.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

TaskName = Literal["classification", "regression"]
InputRank = Literal["2d", "3d_temporal", "tokens"]
InputKind = Literal[
    "tabular",
    "temporal",
    "epoched",
    "embeddings",
    "tokens",
    "tabular_2d",
    "embedding_2d",
    "temporal_3d",
]
EstimatorFamily = Literal[
    "linear",
    "tree",
    "ensemble",
    "svm",
    "neighbors",
    "neural",
    "bayes",
    "dummy",
    "temporal",
    "foundation",
]
PredictionInterface = Literal["predict", "predict_proba", "decision_function"]
GroupedMetadata = Literal["none", "search_cv", "sfs_metadata_routing"]
FeatureSelectionSupport = Literal["univariate", "sfs", "disabled"]
CalibrationSupport = Literal["eligible", "already_probabilistic", "unsupported"]
ImportanceSupport = Literal[
    "coefficients",
    "feature_importances",
    "permutation",
    "saliency",
    "unavailable",
]
TemporalSupport = Literal["none", "sliding", "generalizing", "native"]
DependencyGroup = Literal[
    "core",
    "mne",
    "torch",
    "braindecode",
    "transformers",
    "peft",
    "quant",
]


@dataclass(frozen=True)
class EstimatorCapabilities:
    """Machine-readable capabilities for a decoding estimator."""

    method: str
    tasks: tuple[TaskName, ...]
    input_ranks: tuple[InputRank, ...] = ("2d",)
    prediction_interfaces: tuple[PredictionInterface, ...] = ("predict",)
    grouped_metadata: tuple[GroupedMetadata, ...] = ("none",)
    feature_selection: tuple[FeatureSelectionSupport, ...] = ("univariate", "sfs")
    calibration: CalibrationSupport = "eligible"
    importance: tuple[ImportanceSupport, ...] = ("unavailable",)
    temporal: TemporalSupport = "none"
    dependencies: tuple[DependencyGroup, ...] = ("core",)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly capability dictionary."""
        return asdict(self)

    def supports_task(self, task: str) -> bool:
        return task in self.tasks

    def has_response(self, response: str) -> bool:
        return response in self.prediction_interfaces


@dataclass(frozen=True)
class EstimatorSpec:
    """Typed registry entry for a decoding estimator."""

    name: str
    import_path: str
    family: EstimatorFamily
    task: tuple[TaskName, ...]
    input_kinds: tuple[InputKind, ...] = ("tabular_2d",)
    supports_groups: bool = False
    supports_proba: bool = False
    supports_decision_function: bool = False
    supports_calibration: bool = True
    supports_feature_names: bool = True
    dependency_extra: DependencyGroup = "core"
    fit_smoke_required: bool = True
    default_search_space: dict[str, list[Any]] = field(default_factory=dict)
    feature_selection: tuple[FeatureSelectionSupport, ...] = ("univariate", "sfs")
    importance: tuple[ImportanceSupport, ...] = ("unavailable",)
    temporal: TemporalSupport = "none"
    calibration: CalibrationSupport = "eligible"
    supports_random_state: bool = False

    @property
    def module_path(self) -> str:
        return self.import_path.split(":")[0]

    @property
    def class_name(self) -> str:
        if ":" in self.import_path:
            return self.import_path.split(":", 1)[1]
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly spec dictionary."""
        return asdict(self)

    def to_capabilities(self) -> EstimatorCapabilities:
        """Return lightweight capability metadata derived from the spec."""
        responses = ["predict"]
        if self.supports_proba:
            responses.append("predict_proba")
        if self.supports_decision_function:
            responses.append("decision_function")

        input_ranks = []
        for kind in self.input_kinds:
            if kind in {"temporal_3d", "epoched"}:
                rank = "3d_temporal"
            elif kind == "tokens":
                rank = "tokens"
            else:
                rank = "2d"
            if rank not in input_ranks:
                input_ranks.append(rank)

        return EstimatorCapabilities(
            method=self.name,
            tasks=self.task,
            input_ranks=tuple(input_ranks),
            prediction_interfaces=tuple(responses),
            grouped_metadata=("search_cv",) if self.supports_groups else ("none",),
            feature_selection=self.feature_selection,
            calibration=self.calibration,
            importance=self.importance,
            temporal=self.temporal,
            dependencies=(self.dependency_extra,),
        )


@dataclass(frozen=True)
class SelectorCapabilities:
    """Machine-readable capabilities for a decoding feature selector."""

    method: str
    input_ranks: tuple[InputRank, ...]
    support: tuple[FeatureSelectionSupport, ...]
    grouped_metadata: tuple[GroupedMetadata, ...] = ("none",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CLASSIFICATION = ("classification",)
_REGRESSION = ("regression",)
_BOTH_TASKS = ("classification", "regression")
_COEF = ("coefficients",)
_TREE_IMPORTANCE = ("feature_importances",)


def _spec(
    name: str,
    import_path: str,
    family: EstimatorFamily,
    task: tuple[TaskName, ...],
    *,
    supports_groups: bool = False,
    supports_proba: bool = False,
    supports_decision_function: bool = False,
    supports_calibration: bool = True,
    supports_feature_names: bool = True,
    dependency_extra: DependencyGroup = "core",
    fit_smoke_required: bool = True,
    default_search_space: dict[str, list[Any]] | None = None,
    input_kinds: tuple[InputKind, ...] = ("tabular_2d",),
    feature_selection: tuple[FeatureSelectionSupport, ...] = ("univariate", "sfs"),
    importance: tuple[ImportanceSupport, ...] = ("unavailable",),
    temporal: TemporalSupport = "none",
    calibration: CalibrationSupport = "eligible",
    supports_random_state: bool = False,
) -> EstimatorSpec:
    return EstimatorSpec(
        name=name,
        import_path=import_path,
        family=family,
        task=task,
        input_kinds=input_kinds,
        supports_groups=supports_groups,
        supports_proba=supports_proba,
        supports_decision_function=supports_decision_function,
        supports_calibration=supports_calibration,
        supports_feature_names=supports_feature_names,
        dependency_extra=dependency_extra,
        fit_smoke_required=fit_smoke_required,
        default_search_space=default_search_space or {},
        feature_selection=feature_selection,
        importance=importance,
        temporal=temporal,
        calibration=calibration,
        supports_random_state=supports_random_state,
    )


ESTIMATOR_SPECS: dict[str, EstimatorSpec] = {
    # Classifiers
    "LogisticRegression": _spec(
        "LogisticRegression",
        "sklearn.linear_model",
        "linear",
        _CLASSIFICATION,
        supports_proba=True,
        supports_decision_function=True,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "RandomForestClassifier": _spec(
        "RandomForestClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "SVC": _spec(
        "SVC",
        "sklearn.svm",
        "svm",
        _CLASSIFICATION,
        supports_proba=True,
        supports_decision_function=True,
        supports_random_state=True,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "LinearSVC": _spec(
        "LinearSVC",
        "sklearn.svm",
        "svm",
        _CLASSIFICATION,
        supports_decision_function=True,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "KNeighborsClassifier": _spec(
        "KNeighborsClassifier",
        "sklearn.neighbors",
        "neighbors",
        _CLASSIFICATION,
        supports_proba=True,
        supports_random_state=False,
        default_search_space={"n_neighbors": [3, 5, 7]},
    ),
    "GradientBoostingClassifier": _spec(
        "GradientBoostingClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "learning_rate": [0.03, 0.1],
        },
    ),
    "SGDClassifier": _spec(
        "SGDClassifier",
        "sklearn.linear_model",
        "linear",
        _CLASSIFICATION,
        supports_decision_function=True,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001, 0.01]},
    ),
    "MLPClassifier": _spec(
        "MLPClassifier",
        "sklearn.neural_network",
        "neural",
        _CLASSIFICATION,
        supports_proba=True,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001]},
    ),
    "GaussianNB": _spec(
        "GaussianNB",
        "sklearn.naive_bayes",
        "bayes",
        _CLASSIFICATION,
        supports_proba=True,
        calibration="already_probabilistic",
        supports_random_state=False,
        default_search_space={"var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "LinearDiscriminantAnalysis": _spec(
        "LinearDiscriminantAnalysis",
        "sklearn.discriminant_analysis",
        "linear",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_COEF,
    ),
    "AdaBoostClassifier": _spec(
        "AdaBoostClassifier",
        "sklearn.ensemble",
        "ensemble",
        _CLASSIFICATION,
        supports_proba=True,
        importance=_TREE_IMPORTANCE,
        default_search_space={
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0],
        },
    ),
    "DummyClassifier": _spec(
        "DummyClassifier",
        "sklearn.dummy",
        "dummy",
        _CLASSIFICATION,
        supports_proba=True,
        supports_calibration=False,
        calibration="unsupported",
        default_search_space={},
    ),
    # Regressors
    "LinearRegression": _spec(
        "LinearRegression",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        default_search_space={},
    ),
    "Ridge": _spec(
        "Ridge",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"alpha": [0.1, 1.0, 10.0]},
    ),
    "Lasso": _spec(
        "Lasso",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
    "ElasticNet": _spec(
        "ElasticNet",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={
            "alpha": [0.001, 0.01, 0.1],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
    ),
    "RandomForestRegressor": _spec(
        "RandomForestRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        supports_random_state=True,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "SVR": _spec(
        "SVR",
        "sklearn.svm",
        "svm",
        _REGRESSION,
        default_search_space={"C": [0.1, 1.0, 10.0]},
    ),
    "GradientBoostingRegressor": _spec(
        "GradientBoostingRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        default_search_space={
            "n_estimators": [100, 300],
            "learning_rate": [0.03, 0.1],
        },
    ),
    "SGDRegressor": _spec(
        "SGDRegressor",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001, 0.01]},
    ),
    "MLPRegressor": _spec(
        "MLPRegressor",
        "sklearn.neural_network",
        "neural",
        _REGRESSION,
        supports_random_state=True,
        default_search_space={"alpha": [0.0001, 0.001]},
    ),
    "DummyRegressor": _spec(
        "DummyRegressor",
        "sklearn.dummy",
        "dummy",
        _REGRESSION,
        supports_calibration=False,
        calibration="unsupported",
        default_search_space={},
    ),
    "DecisionTreeRegressor": _spec(
        "DecisionTreeRegressor",
        "sklearn.tree",
        "tree",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        default_search_space={"max_depth": [None, 5, 10]},
    ),
    "KNeighborsRegressor": _spec(
        "KNeighborsRegressor",
        "sklearn.neighbors",
        "neighbors",
        _REGRESSION,
        default_search_space={"n_neighbors": [3, 5, 7]},
    ),
    "ExtraTreesRegressor": _spec(
        "ExtraTreesRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        default_search_space={
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
        },
    ),
    "HistGradientBoostingRegressor": _spec(
        "HistGradientBoostingRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        default_search_space={
            "max_iter": [100, 300],
            "learning_rate": [0.03, 0.1],
        },
    ),
    "AdaBoostRegressor": _spec(
        "AdaBoostRegressor",
        "sklearn.ensemble",
        "ensemble",
        _REGRESSION,
        importance=_TREE_IMPORTANCE,
        default_search_space={
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0],
        },
    ),
    "BayesianRidge": _spec(
        "BayesianRidge",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        default_search_space={"alpha_1": [1e-7, 1e-6]},
    ),
    "ARDRegression": _spec(
        "ARDRegression",
        "sklearn.linear_model",
        "linear",
        _REGRESSION,
        importance=_COEF,
        default_search_space={"alpha_1": [1e-7, 1e-6]},
    ),
    # Temporal wrappers inherit task/response details from their base estimator.
    "SlidingEstimator": _spec(
        "SlidingEstimator",
        "mne.decoding",
        "temporal",
        _BOTH_TASKS,
        input_kinds=("temporal_3d",),
        dependency_extra="mne",
        fit_smoke_required=False,
        feature_selection=("disabled",),
        temporal="sliding",
        default_search_space={},
    ),
    "GeneralizingEstimator": _spec(
        "GeneralizingEstimator",
        "mne.decoding",
        "temporal",
        _BOTH_TASKS,
        input_kinds=("temporal_3d",),
        dependency_extra="mne",
        fit_smoke_required=False,
        feature_selection=("disabled",),
        temporal="generalizing",
        default_search_space={},
    ),
    "FoundationEmbeddingModel": _spec(
        "FoundationEmbeddingModel",
        "coco_pipe.decoding.embedding_extractors:DummyEmbeddingExtractor",
        "foundation",
        _BOTH_TASKS,
        input_kinds=("epoched", "embeddings", "tabular", "temporal", "tokens"),
        supports_calibration=False,
        dependency_extra="core",
        fit_smoke_required=False,
        feature_selection=("disabled",),
    ),
    "FrozenBackboneDecoder": _spec(
        "FrozenBackboneDecoder",
        "coco_pipe.decoding.neural:FrozenBackboneDecoder",
        "foundation",
        _BOTH_TASKS,
        input_kinds=("epoched", "embeddings", "tabular", "temporal", "tokens"),
        supports_proba=True,
        supports_decision_function=True,
        supports_calibration=False,
        dependency_extra="core",
        fit_smoke_required=False,
        feature_selection=("disabled",),
        importance=("permutation",),
    ),
    "NeuralFineTuneEstimator": _spec(
        "NeuralFineTuneEstimator",
        "coco_pipe.decoding.neural:NeuralFineTuneEstimator",
        "neural",
        _BOTH_TASKS,
        input_kinds=("epoched", "temporal", "tokens"),
        supports_proba=True,
        supports_decision_function=True,
        supports_calibration=False,
        dependency_extra="torch",
        fit_smoke_required=False,
        feature_selection=("disabled",),
        importance=("saliency", "permutation"),
    ),
}


ESTIMATOR_CAPABILITIES: dict[str, EstimatorCapabilities] = {
    name: spec.to_capabilities() for name, spec in ESTIMATOR_SPECS.items()
}


SELECTOR_CAPABILITIES: dict[str, SelectorCapabilities] = {
    "k_best": SelectorCapabilities(
        "k_best", input_ranks=("2d",), support=("univariate",)
    ),
    "sfs": SelectorCapabilities(
        "sfs",
        input_ranks=("2d",),
        support=("sfs",),
        grouped_metadata=("sfs_metadata_routing",),
    ),
}


def register_estimator_spec(spec: EstimatorSpec) -> EstimatorSpec:
    """Register or replace an estimator spec."""
    ESTIMATOR_SPECS[spec.name] = spec
    ESTIMATOR_CAPABILITIES[spec.name] = spec.to_capabilities()
    return spec


def get_estimator_spec(method: str) -> EstimatorSpec:
    """Return the typed estimator spec for ``method``."""
    if method not in ESTIMATOR_SPECS:
        raise ValueError(f"No decoding estimator spec registered for '{method}'.")
    return ESTIMATOR_SPECS[method]


def list_estimator_specs() -> dict[str, EstimatorSpec]:
    """Return typed specs for known decoding estimators."""
    return {name: ESTIMATOR_SPECS[name] for name in sorted(ESTIMATOR_SPECS)}


def get_estimator_capabilities(method: str) -> EstimatorCapabilities:
    """Return estimator capabilities derived from the typed spec registry."""
    return get_estimator_spec(method).to_capabilities()


def resolve_estimator_spec(config: Any) -> EstimatorSpec:
    """
    Return the estimator spec for a config, with simple config-aware tweaks.

    This intentionally handles only obvious response-interface cases such as
    ``SVC(probability=False)``. Detailed estimator behavior remains sklearn's job.
    """
    kind = getattr(config, "kind", None)
    if kind == "classical":
        spec = get_estimator_spec(canonical_estimator_name(config.estimator))
    elif kind == "foundation_embedding":
        spec = EstimatorSpec(
            name="FoundationEmbeddingModel",
            import_path=(
                "coco_pipe.decoding.embedding_extractors:DummyEmbeddingExtractor"
            ),
            family="foundation",
            task=("classification", "regression"),
            input_kinds=(config.input_kind,),
            supports_proba=False,
            supports_decision_function=False,
            supports_calibration=False,
            feature_selection=("disabled",),
            importance=("unavailable",),
            dependency_extra="core",
            fit_smoke_required=False,
        )
    elif kind == "frozen_backbone":
        head_spec = resolve_estimator_spec(config.head)
        spec = replace(
            head_spec,
            name="FrozenBackboneDecoder",
            import_path="coco_pipe.decoding.neural:FrozenBackboneDecoder",
            family="foundation",
            input_kinds=(config.backbone.input_kind,),
            feature_selection=(
                ("univariate", "sfs")
                if config.backbone.input_kind == "embeddings"
                else ("disabled",)
            ),
            importance=("permutation",),
        )
    elif kind == "neural_finetune":
        spec = EstimatorSpec(
            name="NeuralFineTuneEstimator",
            import_path="coco_pipe.decoding.neural:NeuralFineTuneEstimator",
            family="neural",
            task=("classification", "regression"),
            input_kinds=(config.input_kind,),
            supports_proba=True,
            supports_decision_function=True,
            supports_calibration=False,
            feature_selection=("disabled",),
            importance=("saliency", "permutation"),
            dependency_extra=(
                "peft" if config.train_mode in {"lora", "qlora"} else "torch"
            ),
            fit_smoke_required=False,
        )
    elif kind == "temporal":
        base_spec = resolve_estimator_spec(config.base)
        method = (
            "SlidingEstimator"
            if config.wrapper == "sliding"
            else "GeneralizingEstimator"
        )
        spec = replace(
            get_estimator_spec(method),
            task=base_spec.task,
            supports_proba=base_spec.supports_proba,
            supports_decision_function=base_spec.supports_decision_function,
            supports_calibration=base_spec.supports_calibration,
            supports_feature_names=False,
        )
    else:
        spec = get_estimator_spec(config.method)

    if config.method == "SVC" and not getattr(config, "probability", True):
        spec = replace(spec, supports_proba=False, supports_decision_function=True)

    if config.method == "SGDClassifier" and getattr(config, "loss", None) in {
        "log_loss",
        "modified_huber",
    }:
        spec = replace(spec, supports_proba=True, supports_decision_function=True)

    if config.method in {"SlidingEstimator", "GeneralizingEstimator"}:
        base_spec = resolve_estimator_spec(config.base_estimator)
        spec = replace(
            spec,
            task=base_spec.task,
            supports_proba=base_spec.supports_proba,
            supports_decision_function=base_spec.supports_decision_function,
            supports_calibration=base_spec.supports_calibration,
            supports_feature_names=False,
        )

    return spec


def canonical_estimator_name(name: str) -> str:
    aliases = {
        "logistic_regression": "LogisticRegression",
        "random_forest_classifier": "RandomForestClassifier",
        "linear_svc": "LinearSVC",
        "lda": "LinearDiscriminantAnalysis",
        "dummy_classifier": "DummyClassifier",
        "ridge": "Ridge",
        "random_forest_regressor": "RandomForestRegressor",
    }
    return aliases.get(name, name)


def resolve_estimator_capabilities(config: Any) -> EstimatorCapabilities:
    """Return config-aware capabilities derived from ``resolve_estimator_spec``."""
    return resolve_estimator_spec(config).to_capabilities()


def get_selector_capabilities(method: str) -> SelectorCapabilities:
    """Return feature-selector capabilities for ``method``."""
    if method not in SELECTOR_CAPABILITIES:
        raise ValueError(
            f"No decoding capabilities registered for selector '{method}'."
        )
    return SELECTOR_CAPABILITIES[method]
