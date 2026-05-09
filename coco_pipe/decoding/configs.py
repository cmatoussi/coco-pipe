"""
Decoding Configurations
=======================

Comprehensive Pydantic models for strict validation of Decoding/ML experiments.

Key Components:
- ModelConfigs: extensive hyperparameters for each estimator.
- ExperimentConfig: Top-level configuration for the entire analysis workflow.
"""

from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# --- Base Schemas ---


class BaseEstimatorConfig(BaseModel):
    """Base configuration for any estimator."""

    model_config = ConfigDict(extra="forbid")


# --- Mixins ---


class LinearMixin(BaseModel):
    """Common parameters for linear models."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None


class RegularizedLinearMixin(BaseModel):
    """Parameters for regularized linear models."""

    fit_intercept: bool = True
    copy_X: bool = True
    tol: float = 1e-3
    max_iter: Optional[int] = None
    positive: bool = False
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class TreeMixin(BaseModel):
    """Common parameters for Tree-based models."""

    n_estimators: int = Field(100, ge=1)
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, int, float, None] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    n_jobs: Optional[int] = None
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )
    verbose: int = 0
    warm_start: bool = False


class SupportVectorMixin(BaseModel):
    """Common parameters for Support Vector Machines."""

    C: float = Field(1.0, gt=0.0)
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf"
    degree: int = 3
    gamma: Union[str, float] = "scale"
    coef0: float = 0.0
    tol: float = 1e-3
    verbose: bool = False
    max_iter: int = -1
    shrinking: bool = True
    cache_size: float = 200


class SGDMixin(BaseModel):
    loss: str = "hinge"
    penalty: Optional[Literal["l2", "l1", "elasticnet"]] = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-3
    shuffle: bool = True
    verbose: int = 0
    epsilon: float = 0.1
    n_jobs: Optional[int] = None
    learning_rate: str = "optimal"
    eta0: float = 0.0
    power_t: float = 0.5
    early_stopping: bool = False


class MLPMixin(BaseModel):
    """Common parameters for Multi-layer Perceptron models."""

    hidden_layer_sizes: tuple = (100,)
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    solver: Literal["lbfgs", "sgd", "adam"] = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant"
    learning_rate_init: float = 0.001
    power_t: float = 0.5
    max_iter: int = 200
    shuffle: bool = True
    tol: float = 1e-4
    verbose: bool = False
    warm_start: bool = False
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    max_fun: int = 15000
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class GradientBoostingMixin(BaseModel):
    """Common parameters for Gradient Boosting models."""

    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: Literal["friedman_mse", "squared_error"] = "friedman_mse"
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    init: Optional[str] = None
    max_features: Union[str, int, float, None] = None
    alpha: float = 0.9
    verbose: int = 0
    max_leaf_nodes: Optional[int] = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: Optional[int] = None
    tol: float = 1e-4
    ccp_alpha: float = 0.0
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5
    warm_start: bool = False
    average: bool = False
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


# --- Classifiers ---


class LogisticRegressionConfig(BaseEstimatorConfig):
    method: Literal["LogisticRegression"] = "LogisticRegression"
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = Field(1.0, gt=0.0)
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Optional[Union[Dict, str]] = None
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs"
    max_iter: int = 100
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class RandomForestClassifierConfig(BaseEstimatorConfig, TreeMixin):
    method: Literal["RandomForestClassifier"] = "RandomForestClassifier"
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    bootstrap: bool = True
    oob_score: bool = False
    class_weight: Optional[Union[str, Dict, List]] = None
    max_samples: Optional[Union[int, float]] = None


class SVCConfig(BaseEstimatorConfig, SupportVectorMixin):
    method: Literal["SVC"] = "SVC"
    probability: bool = True  # Default to True for metrics requiring proba
    class_weight: Optional[Union[Dict, str]] = None
    decision_function_shape: Literal["ovo", "ovr"] = "ovr"
    break_ties: bool = False
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class LinearSVCConfig(BaseEstimatorConfig):
    method: Literal["LinearSVC"] = "LinearSVC"
    penalty: Literal["l1", "l2"] = "l2"
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge"
    dual: Union[bool, Literal["auto"]] = "auto"
    tol: float = 1e-4
    C: float = Field(1.0, gt=0.0)
    multi_class: Literal["ovr", "crammer_singer"] = "ovr"
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Optional[Union[Dict, str]] = None
    verbose: int = 0
    max_iter: int = 1000
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class KNeighborsClassifierConfig(BaseEstimatorConfig):
    method: Literal["KNeighborsClassifier"] = "KNeighborsClassifier"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Optional[Dict] = None
    n_jobs: Optional[int] = None


class GradientBoostingClassifierConfig(BaseEstimatorConfig, GradientBoostingMixin):
    method: Literal["GradientBoostingClassifier"] = "GradientBoostingClassifier"
    loss: Literal["log_loss", "exponential"] = "log_loss"


class SGDClassifierConfig(BaseEstimatorConfig, SGDMixin):
    method: Literal["SGDClassifier"] = "SGDClassifier"
    class_weight: Optional[Union[Dict, str]] = None


class MLPClassifierConfig(BaseEstimatorConfig, MLPMixin):
    method: Literal["MLPClassifier"] = "MLPClassifier"


class GaussianNBConfig(BaseEstimatorConfig):
    method: Literal["GaussianNB"] = "GaussianNB"
    priors: Optional[List[float]] = None
    var_smoothing: float = 1e-9


class LDAConfig(BaseEstimatorConfig):
    method: Literal["LinearDiscriminantAnalysis"] = "LinearDiscriminantAnalysis"
    solver: Literal["svd", "lsqr", "eigen"] = "svd"
    shrinkage: Optional[Union[str, float]] = None
    priors: Optional[List[float]] = None
    n_components: Optional[int] = None
    store_covariance: bool = False
    tol: float = 1e-4


class AdaBoostClassifierConfig(BaseEstimatorConfig):
    method: Literal["AdaBoostClassifier"] = "AdaBoostClassifier"
    n_estimators: int = 50
    learning_rate: float = 1.0
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class DummyClassifierConfig(BaseEstimatorConfig):
    method: Literal["DummyClassifier"] = "DummyClassifier"
    strategy: Literal["stratified", "most_frequent", "prior", "uniform"] = "prior"
    constant: Optional[Any] = None
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


# --- Deep Learning / Foundation Models ---


class LPFTConfig(BaseEstimatorConfig):
    """
    Configuration for Linear-Probe Fine-Tuning (LP-FT).
    Reference: Kumar et al. (2022).
    """

    method: Literal["LPFTClassifier"] = "LPFTClassifier"
    backbone_name: str = "gpt2"
    # LP Step
    lp_lr: float = 1e-3
    lp_epochs: int = 10
    # FT Step
    ft_lr: float = 1e-5
    ft_epochs: int = 5
    batch_size: int = 32
    max_length: int = 128
    device: str = "cpu"


class SkorchClassifierConfig(BaseEstimatorConfig):
    """Configuration for generic PyTorch wrappers via Skorch."""

    method: Literal["SkorchClassifier"] = "SkorchClassifier"
    module_name: str
    max_epochs: int = 10
    lr: float = 0.01
    batch_size: int = 64
    optimizer: str = "Adam"
    device: str = "cpu"


# --- Temporal Meta-Estimators (MNE) ---


class SlidingEstimatorConfig(BaseEstimatorConfig):
    """
    Configuration for MNE's SlidingEstimator.
    Fits a separate estimator for each time point.
    """

    method: Literal["SlidingEstimator"] = "SlidingEstimator"
    base_estimator: "EstimatorConfigType"
    scoring: Optional[Union[str, Callable]] = None
    n_jobs: Optional[int] = 1
    position: Optional[float] = 0
    allow_2d: bool = False
    verbose: Optional[Union[bool, str, int]] = None


class GeneralizingEstimatorConfig(BaseEstimatorConfig):
    """
    Configuration for MNE's GeneralizingEstimator.
    Fits an estimator on each time point and tests on all other time points.
    """

    method: Literal["GeneralizingEstimator"] = "GeneralizingEstimator"
    base_estimator: "EstimatorConfigType"
    scoring: Optional[Union[str, Callable]] = None
    n_jobs: Optional[int] = 1
    position: Optional[float] = 0
    allow_2d: bool = False
    verbose: Optional[Union[bool, str, int]] = None


# --- Regressors ---


class LinearRegressionConfig(BaseEstimatorConfig, LinearMixin):
    method: Literal["LinearRegression"] = "LinearRegression"
    positive: bool = False


class RidgeConfig(BaseEstimatorConfig, RegularizedLinearMixin):
    method: Literal["Ridge"] = "Ridge"
    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    solver: str = "auto"


class LassoConfig(BaseEstimatorConfig, RegularizedLinearMixin):
    method: Literal["Lasso"] = "Lasso"
    alpha: float = 1.0
    precompute: Union[bool, List] = False
    fit_intercept: bool = True
    copy_X: bool = True
    selection: Literal["cyclic", "random"] = "cyclic"
    warm_start: bool = False


class ElasticNetConfig(BaseEstimatorConfig, RegularizedLinearMixin):
    method: Literal["ElasticNet"] = "ElasticNet"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    precompute: Union[bool, List] = False
    fit_intercept: bool = True
    copy_X: bool = True
    selection: Literal["cyclic", "random"] = "cyclic"
    warm_start: bool = False


class RandomForestRegressorConfig(BaseEstimatorConfig, TreeMixin):
    method: Literal["RandomForestRegressor"] = "RandomForestRegressor"
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = (
        "squared_error"
    )
    bootstrap: bool = True
    oob_score: bool = False
    max_samples: Optional[Union[int, float]] = None


class SVRConfig(BaseEstimatorConfig, SupportVectorMixin):
    method: Literal["SVR"] = "SVR"
    epsilon: float = 0.1


class GradientBoostingRegressorConfig(BaseEstimatorConfig, GradientBoostingMixin):
    method: Literal["GradientBoostingRegressor"] = "GradientBoostingRegressor"
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = (
        "squared_error"
    )


class SGDRegressorConfig(BaseEstimatorConfig, SGDMixin):
    method: Literal["SGDRegressor"] = "SGDRegressor"
    loss: str = "squared_error"


class MLPRegressorConfig(BaseEstimatorConfig, MLPMixin):
    method: Literal["MLPRegressor"] = "MLPRegressor"


class DummyRegressorConfig(BaseEstimatorConfig):
    method: Literal["DummyRegressor"] = "DummyRegressor"
    strategy: Literal["mean", "median", "quantile", "constant"] = "mean"
    constant: Optional[Union[int, float, List]] = None
    quantile: Optional[float] = None


class DecisionTreeRegressorConfig(BaseEstimatorConfig):
    method: Literal["DecisionTreeRegressor"] = "DecisionTreeRegressor"
    criterion: Literal["squared_error", "friedman_mse", "absolute_error", "poisson"] = (
        "squared_error"
    )
    splitter: Literal["best", "random"] = "best"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, int, float, None] = None
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0


class KNeighborsRegressorConfig(BaseEstimatorConfig):
    method: Literal["KNeighborsRegressor"] = "KNeighborsRegressor"
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Optional[Dict] = None
    n_jobs: Optional[int] = None


class ExtraTreesRegressorConfig(BaseEstimatorConfig, TreeMixin):
    method: Literal["ExtraTreesRegressor"] = "ExtraTreesRegressor"
    bootstrap: bool = False
    oob_score: bool = False
    max_samples: Optional[Union[int, float]] = None


class HistGradientBoostingRegressorConfig(BaseEstimatorConfig):
    method: Literal["HistGradientBoostingRegressor"] = "HistGradientBoostingRegressor"
    loss: Literal["squared_error", "absolute_error", "poisson", "quantile"] = (
        "squared_error"
    )
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    max_depth: Optional[int] = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    categorical_features: Optional[Union[List[int], List[str], List[bool]]] = None
    monotonic_cst: Optional[Any] = None
    interaction_cst: Optional[Any] = None
    warm_start: bool = False
    early_stopping: Union[bool, Literal["auto"]] = "auto"
    scoring: Optional[str] = "loss"
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    verbose: int = 0
    random_state: Optional[int] = None


class AdaBoostRegressorConfig(BaseEstimatorConfig):
    method: Literal["AdaBoostRegressor"] = "AdaBoostRegressor"
    n_estimators: int = 50
    learning_rate: float = 1.0
    loss: Literal["linear", "square", "exponential"] = "linear"
    random_state: Optional[int] = Field(
        42, description="Random seed for reproducibility."
    )


class BayesianRidgeConfig(BaseEstimatorConfig):
    method: Literal["BayesianRidge"] = "BayesianRidge"
    max_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    alpha_init: Optional[float] = None
    lambda_init: Optional[float] = None
    compute_score: bool = False
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


class ARDRegressionConfig(BaseEstimatorConfig):
    method: Literal["ARDRegression"] = "ARDRegression"
    max_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    compute_score: bool = False
    threshold_lambda: float = 10000.0
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


# --- Recursive Unions Implementation ---


# 1. Define Atomic Unions (Terminal nodes)
AtomicEstimator = Union[
    LogisticRegressionConfig,
    RandomForestClassifierConfig,
    SVCConfig,
    LinearSVCConfig,
    KNeighborsClassifierConfig,
    GradientBoostingClassifierConfig,
    SGDClassifierConfig,
    MLPClassifierConfig,
    GaussianNBConfig,
    LDAConfig,
    AdaBoostClassifierConfig,
    DummyClassifierConfig,
    # Regressors
    LinearRegressionConfig,
    RidgeConfig,
    LassoConfig,
    ElasticNetConfig,
    RandomForestRegressorConfig,
    SVRConfig,
    GradientBoostingRegressorConfig,
    SGDRegressorConfig,
    MLPRegressorConfig,
    DummyRegressorConfig,
    DecisionTreeRegressorConfig,
    KNeighborsRegressorConfig,
    ExtraTreesRegressorConfig,
    HistGradientBoostingRegressorConfig,
    AdaBoostRegressorConfig,
    BayesianRidgeConfig,
    ARDRegressionConfig,
]

# 2. Define the Recursive Union using Annotated Discriminator
# This allows Pydantic to choose the correct class based on 'method' field
EstimatorConfigType = Annotated[
    Union[AtomicEstimator, SlidingEstimatorConfig, GeneralizingEstimatorConfig],
    Field(discriminator="method"),
]


# --- Experiment Config ---


class CVConfig(BaseModel):
    """Cross-validation settings."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal[
        "stratified",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
        "leave_p_out",
        "leave_one_group_out",
        "timeseries",
        "split",
    ] = "stratified"
    n_splits: int = Field(5, ge=2)
    shuffle: bool = True
    random_state: int = 42
    test_size: float = Field(
        0.2, gt=0.0, lt=1.0, description="Holdout size for strategy='split'."
    )
    stratify: bool = Field(
        False, description="Whether strategy='split' should stratify by y."
    )
    group_key: Optional[str] = Field(
        None, description="sample_metadata column used by grouped CV strategies."
    )


class TuningConfig(BaseModel):
    """
    Hyperparameter Tuning Configuration.
    Use this to define HOW to search (random vs grid).
    The WHAT (the grid itself) is passed in ExperimentConfig.grids.
    """

    enabled: bool = False
    search_type: Literal["grid", "random"] = "grid"
    n_iter: int = Field(10, description="Number of iterations for random search")
    scoring: Optional[str] = None  # Metric to optimize (defaults to first in list)
    n_jobs: int = -1
    random_state: Optional[int] = Field(
        42, description="Random seed used by RandomizedSearchCV."
    )
    cv: Optional[CVConfig] = Field(
        None,
        description=(
            "Inner CV used for model selection. Defaults to the outer CV family."
        ),
    )
    allow_nongroup_inner_cv: bool = Field(
        False,
        description=(
            "Allow a non-grouped tuning CV under grouped outer CV. This explicitly "
            "acknowledges the leakage/generalization trade-off."
        ),
    )


class FeatureSelectionConfig(BaseModel):
    """Feature selection settings."""

    enabled: bool = False
    method: Literal["k_best", "sfs"] = "sfs"
    n_features: Optional[int] = Field(None, description="Number of features to select.")
    direction: Literal["forward", "backward"] = "forward"
    cv: Optional[CVConfig] = Field(
        None,
        description=(
            "Inner CV used by SequentialFeatureSelector. Defaults to tuning.cv "
            "when available, otherwise the outer CV family."
        ),
    )
    scoring: Optional[str] = None
    allow_nongroup_inner_cv: bool = Field(
        False,
        description=(
            "Allow a non-grouped SFS CV under grouped outer CV. This explicitly "
            "acknowledges the leakage/generalization trade-off."
        ),
    )


class CalibrationConfig(BaseModel):
    """Probability calibration settings for classification estimators."""

    enabled: bool = False
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: Optional[CVConfig] = Field(
        None,
        description=(
            "Inner CV used by CalibratedClassifierCV. Defaults to the outer CV "
            "family. Calibration data stays disjoint from each base-estimator "
            "training fold."
        ),
    )
    n_jobs: Optional[int] = None
    allow_nongroup_inner_cv: bool = Field(
        False,
        description=(
            "Allow a non-grouped calibration CV under grouped outer CV. This "
            "explicitly acknowledges the leakage/generalization trade-off."
        ),
    )


class ConfidenceIntervalConfig(BaseModel):
    """Confidence interval settings."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    method: Literal["wilson", "clopper_pearson"] = "wilson"


class ChanceAssessmentConfig(BaseModel):
    """Null-hypothesis (chance level) assessment settings."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["permutation", "binomial", "auto"] = "permutation"
    n_permutations: int = Field(1000, ge=1)
    p0: Union[float, Literal["auto"], None] = Field(
        "auto",
        description="Chance level for binomial test (e.g., 0.5 for binary).",
    )
    temporal_correction: Literal["max_stat", "fdr_bh", "none"] = Field(
        "max_stat", description="Method to correct for multiple comparisons."
    )
    store_null_distribution: bool = False


class StatisticalAssessmentConfig(BaseModel):
    """
    Finite-sample statistical assessment settings.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    random_state: Optional[int] = 42
    metrics: Optional[List[str]] = Field(
        None, description="Subset of experiment metrics to run assessment for."
    )

    chance: ChanceAssessmentConfig = Field(default_factory=ChanceAssessmentConfig)
    confidence_intervals: ConfidenceIntervalConfig = Field(
        default_factory=ConfidenceIntervalConfig
    )

    unit_of_inference: Optional[
        Literal["sample", "group_mean", "group_majority", "custom"]
    ] = Field(
        None,
        description="Independent unit for label permutation or binomial counts.",
    )
    custom_unit_column: Optional[str] = None
    custom_aggregation: Literal["mean", "majority"] = "mean"


class ClassicalModelConfig(BaseEstimatorConfig):
    """Final public config for sklearn-backed classical estimators."""

    kind: Literal["classical"] = "classical"
    estimator: str
    params: Dict[str, Any] = Field(default_factory=dict)
    input_kind: Literal["tabular", "embeddings"] = "tabular"


class FoundationEmbeddingModelConfig(BaseEstimatorConfig):
    """Config for pretrained/frozen embedding extraction."""

    kind: Literal["foundation_embedding"] = "foundation_embedding"
    provider: Literal["dummy", "braindecode", "huggingface"] = "dummy"
    model_name: str = "dummy"
    input_kind: Literal["tabular", "temporal", "epoched", "embeddings", "tokens"] = (
        "epoched"
    )
    pooling: Literal["mean", "flatten", "last"] = "mean"
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    embedding_dim: Optional[int] = None


class LoRAConfig(BaseModel):
    """LoRA adapter settings."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    target_modules: Union[str, List[str]] = "all-linear"


class QuantizationConfig(BaseModel):
    """Quantization settings for QLoRA-style workflows."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"


class DeviceConfig(BaseModel):
    """Serializable device and precision policy."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class CheckpointConfig(BaseModel):
    """Serializable checkpoint policy."""

    model_config = ConfigDict(extra="forbid")

    save: Literal["none", "best", "last", "all"] = "best"
    monitor: str = "val_loss"
    output_dir: Optional[Path] = None


class TrainerConfig(BaseModel):
    """Minimal neural training settings."""

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = Field(10, ge=1)
    early_stopping_patience: Optional[int] = Field(None, ge=1)
    batch_size: int = Field(32, ge=1)
    validation_fraction: float = Field(0.2, ge=0.0, lt=1.0)


class TrainStageConfig(BaseModel):
    """Fine-tuning stage schedule entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    epochs: int = Field(..., ge=1)
    train_backbone: bool = False
    train_head: bool = True


class FrozenBackboneDecoderConfig(BaseEstimatorConfig):
    """Frozen embedding backbone plus explicit classical head."""

    kind: Literal["frozen_backbone"] = "frozen_backbone"
    backbone: FoundationEmbeddingModelConfig
    head: ClassicalModelConfig


class NeuralFineTuneConfig(BaseEstimatorConfig):
    """Trainable neural/foundation-model estimator config."""

    kind: Literal["neural_finetune"] = "neural_finetune"
    provider: Literal["dummy", "braindecode", "huggingface"] = "dummy"
    model_name: str = "dummy"
    input_kind: Literal["temporal", "epoched", "tokens"] = "epoched"
    train_mode: Literal["full", "frozen", "linear_probe", "lora", "qlora"] = "full"
    optimizer: Dict[str, Any] = Field(default_factory=lambda: {"name": "adamw"})
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    lora: Optional[LoRAConfig] = None
    quantization: Optional[QuantizationConfig] = None
    stages: List[TrainStageConfig] = Field(default_factory=list)


class TemporalDecoderConfig(BaseEstimatorConfig):
    """Final public config for MNE temporal meta-estimators."""

    kind: Literal["temporal"] = "temporal"
    wrapper: Literal["sliding", "generalizing"] = "sliding"
    base: ClassicalModelConfig
    scoring: Optional[Union[str, Callable]] = None
    n_jobs: Optional[int] = 1
    position: Optional[float] = 0
    allow_2d: bool = False
    verbose: Optional[Union[bool, str, int]] = None


ModelConfigType = Annotated[
    Union[
        ClassicalModelConfig,
        FoundationEmbeddingModelConfig,
        FrozenBackboneDecoderConfig,
        NeuralFineTuneConfig,
        TemporalDecoderConfig,
    ],
    Field(discriminator="kind"),
]


class ExperimentConfig(BaseModel):
    """
    Master configuration for a Decoding Experiment.
    """

    model_config = ConfigDict(extra="forbid")

    task: Literal["classification", "regression"] = "classification"
    output_dir: Optional[Path] = None
    tag: str = "experiment"
    random_state: Optional[int] = Field(
        None,
        description=(
            "Master random seed. If set, it is used to derive seeds for all "
            "components (CV, Tuning, Models, etc.) ensuring global reproducibility."
        ),
    )

    # Map of Friendly Name -> Polymorphic Config Object
    models: Dict[str, ModelConfigType]

    # Map of Friendly Name -> Parameter Grid (Search Space)
    grids: Optional[Dict[str, Dict[str, List[Any]]]] = None

    cv: CVConfig = Field(default_factory=CVConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    evaluation: StatisticalAssessmentConfig = Field(
        default_factory=StatisticalAssessmentConfig
    )

    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "roc_auc"],
        description="List of metrics to compute.",
    )

    use_scaler: bool = Field(
        True, description="Whether to scalar normalize features upstream."
    )
    n_jobs: int = -1
    verbose: bool = True
