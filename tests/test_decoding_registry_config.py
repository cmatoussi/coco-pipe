import pytest
from pydantic import ValidationError

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    AdaBoostClassifierConfig,
    AdaBoostRegressorConfig,
    ARDRegressionConfig,
    BayesianRidgeConfig,
    DecisionTreeRegressorConfig,
    DummyClassifierConfig,
    DummyRegressorConfig,
    ElasticNetConfig,
    ExtraTreesRegressorConfig,
    GaussianNBConfig,
    GradientBoostingClassifierConfig,
    GradientBoostingRegressorConfig,
    HistGradientBoostingRegressorConfig,
    KNeighborsClassifierConfig,
    KNeighborsRegressorConfig,
    LassoConfig,
    LDAConfig,
    LinearRegressionConfig,
    LogisticRegressionConfig,
    MLPClassifierConfig,
    MLPRegressorConfig,
    RandomForestClassifierConfig,
    RandomForestRegressorConfig,
    RidgeConfig,
    SGDClassifierConfig,
    SGDRegressorConfig,
    SVCConfig,
    SVRConfig,
)
from coco_pipe.decoding.registry import get_estimator_cls, register_estimator

ACTIVE_SKLEARN_CONFIGS = [
    LogisticRegressionConfig,
    RandomForestClassifierConfig,
    SVCConfig,
    KNeighborsClassifierConfig,
    GradientBoostingClassifierConfig,
    SGDClassifierConfig,
    MLPClassifierConfig,
    GaussianNBConfig,
    LDAConfig,
    AdaBoostClassifierConfig,
    DummyClassifierConfig,
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


def _experiment_for_instantiation():
    return Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": {"method": "LogisticRegression"}},
            metrics=["accuracy"],
            n_jobs=1,
            verbose=False,
        )
    )


def test_every_active_sklearn_config_method_resolves():
    for config_cls in ACTIVE_SKLEARN_CONFIGS:
        config = config_cls()

        assert get_estimator_cls(config.method) is not None


def test_every_active_sklearn_default_config_instantiates():
    experiment = _experiment_for_instantiation()

    for config_cls in ACTIVE_SKLEARN_CONFIGS:
        config = config_cls()

        estimator = experiment._instantiate_model(config.method, config)

        assert estimator is not None


def test_experiment_config_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            task="classification",
            models={"lr": {"method": "LogisticRegression"}},
            unexpected=True,
        )


def test_removed_deprecated_config_fields_are_rejected():
    with pytest.raises(ValidationError):
        LogisticRegressionConfig(multiclass="ovr")

    with pytest.raises(ValidationError):
        AdaBoostClassifierConfig(algorithm="SAMME")

    with pytest.raises(ValidationError):
        BayesianRidgeConfig(n_iter=10)

    with pytest.raises(ValidationError):
        ARDRegressionConfig(n_iter=10)


def test_modern_iteration_fields_are_exposed():
    bayes = BayesianRidgeConfig(max_iter=12)
    ard = ARDRegressionConfig(max_iter=13)

    assert bayes.model_dump()["max_iter"] == 12
    assert ard.model_dump()["max_iter"] == 13
    assert "n_iter" not in bayes.model_dump()
    assert "n_iter" not in ard.model_dump()


def test_sgd_penalty_accepts_none_not_null_string():
    assert SGDClassifierConfig(penalty=None).penalty is None
    with pytest.raises(ValidationError):
        SGDClassifierConfig(penalty="null")


def test_fm_and_skorch_placeholders_are_not_active_experiment_configs():
    with pytest.raises(ValidationError) as lpft_error:
        ExperimentConfig(
            task="classification",
            models={"fm": {"method": "LPFTClassifier"}},
        )
    assert "LPFTClassifier" in str(lpft_error.value)

    with pytest.raises(ValidationError) as skorch_error:
        ExperimentConfig(
            task="classification",
            models={"skorch": {"method": "SkorchClassifier", "module_name": "Net"}},
        )
    assert "SkorchClassifier" in str(skorch_error.value)


def test_invalid_constructor_params_are_not_silently_dropped():
    @register_estimator("StrictFakeEstimator")
    class StrictFakeEstimator:
        def __init__(self, known=1):
            self.known = known

    class FakeConfig:
        method = "StrictFakeEstimator"

        def model_dump(self, exclude=None):
            data = {"method": self.method, "unknown": 2}
            for key in exclude or set():
                data.pop(key, None)
            return data

    experiment = _experiment_for_instantiation()

    with pytest.raises(ValueError, match="Failed to instantiate model 'fake'"):
        experiment._instantiate_model("fake", FakeConfig())
