import numpy as np
import pytest

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.capabilities import (
    EstimatorSpec,
    get_estimator_capabilities,
    get_estimator_spec,
    get_selector_capabilities,
    list_estimator_specs,
    resolve_estimator_capabilities,
)
from coco_pipe.decoding.configs import (
    CVConfig,
    FeatureSelectionConfig,
    GeneralizingEstimatorConfig,
    LogisticRegressionConfig,
    RidgeConfig,
    SlidingEstimatorConfig,
    SVCConfig,
)
from coco_pipe.decoding.registry import get_estimator_cls, list_capabilities


def test_core_estimators_expose_capability_metadata():
    capabilities = list_capabilities()

    assert "LogisticRegression" in capabilities
    assert capabilities["LogisticRegression"].supports_task("classification")
    assert capabilities["Ridge"].supports_task("regression")
    assert "predict_proba" in capabilities["LogisticRegression"].prediction_interfaces
    assert "coefficients" in capabilities["Ridge"].importance


def test_estimator_specs_are_the_registry_source_of_truth():
    specs = list_estimator_specs()
    logreg = get_estimator_spec("LogisticRegression")

    assert isinstance(logreg, EstimatorSpec)
    assert specs["LogisticRegression"] == logreg
    assert logreg.family == "linear"
    assert logreg.task == ("classification",)
    assert logreg.input_kinds == ("tabular_2d",)
    assert logreg.supports_proba is True
    assert logreg.supports_decision_function is True
    assert logreg.dependency_extra == "core"
    assert logreg.fit_smoke_required is True
    assert logreg.default_search_space["C"] == [0.1, 1.0, 10.0]
    assert get_estimator_cls("LogisticRegression") is not None


def test_capabilities_are_derived_from_estimator_specs():
    specs = list_estimator_specs()
    capabilities = list_capabilities()

    for name, spec in specs.items():
        assert capabilities[name] == spec.to_capabilities()


def test_svc_probability_flag_updates_declared_response_interfaces():
    with_proba = resolve_estimator_capabilities(SVCConfig(probability=True))
    without_proba = resolve_estimator_capabilities(SVCConfig(probability=False))

    assert "predict_proba" in with_proba.prediction_interfaces
    assert "predict_proba" not in without_proba.prediction_interfaces
    assert "decision_function" in without_proba.prediction_interfaces


def test_probability_metric_mismatch_fails_before_nested_cv_for_that_model():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.array([0, 1] * 10)
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"svc": SVCConfig(probability=False, kernel="linear")},
            metrics=["log_loss"],
            cv=CVConfig(strategy="stratified", n_splits=2),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    assert result.raw["svc"]["status"] == "failed"
    assert "requires predict_proba" in result.raw["svc"]["error"]
    assert "capability" in result.raw["svc"]["error"]


def test_ranking_metric_accepts_decision_function_capability():
    X = np.random.default_rng(1).normal(size=(24, 3))
    y = np.array([0, 1] * 12)
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"svc": SVCConfig(probability=False, kernel="linear")},
            metrics=["roc_auc"],
            cv=CVConfig(strategy="stratified", n_splits=2),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    assert "error" not in result.raw["svc"]


def test_selector_capabilities_reject_temporal_input_rank():
    pytest.importorskip("mne")
    X = np.random.default_rng(2).normal(size=(12, 3, 4))
    y = np.array([0, 1] * 6)

    with pytest.raises(ValueError, match="Feature selection method 'k_best'"):
        Experiment(
            ExperimentConfig(
                task="classification",
                models={
                    "sliding": SlidingEstimatorConfig(
                        base_estimator=LogisticRegressionConfig(max_iter=100),
                        n_jobs=1,
                    )
                },
                metrics=["accuracy"],
                feature_selection=FeatureSelectionConfig(
                    enabled=True,
                    method="k_best",
                    n_features=2,
                ),
                cv=CVConfig(strategy="stratified", n_splits=2),
                n_jobs=1,
                verbose=False,
            )
        ).run(X, y)


def test_temporal_capabilities_reject_2d_input_rank():
    X = np.random.default_rng(3).normal(size=(12, 3))
    y = np.array([0, 1] * 6)

    with pytest.raises(ValueError, match="expects input rank"):
        Experiment(
            ExperimentConfig(
                task="classification",
                models={
                    "generalizing": GeneralizingEstimatorConfig(
                        base_estimator=LogisticRegressionConfig(max_iter=100),
                        n_jobs=1,
                    )
                },
                metrics=["accuracy"],
                cv=CVConfig(strategy="stratified", n_splits=2),
                n_jobs=1,
                verbose=False,
            )
        ).run(X, y)


def test_task_support_uses_capabilities_not_method_name_heuristics():
    with pytest.raises(ValueError, match="does not support task 'classification'"):
        Experiment(
            ExperimentConfig(
                task="classification",
                models={"ridge": RidgeConfig()},
                metrics=["accuracy"],
            )
        )

    caps = get_estimator_capabilities("Ridge")
    assert caps.tasks == ("regression",)


def test_capabilities_are_stored_in_result_provenance():
    X = np.random.default_rng(4).normal(size=(20, 3))
    y = np.array([0, 1] * 10)
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig(max_iter=100)},
            metrics=["accuracy"],
            cv=CVConfig(strategy="stratified", n_splits=2),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    caps = result.meta["capabilities"]
    assert caps["models"]["lr"]["method"] == "LogisticRegression"
    assert caps["estimator_specs"]["lr"]["family"] == "linear"
    assert caps["estimator_specs"]["lr"]["default_search_space"]["C"] == [
        0.1,
        1.0,
        10.0,
    ]
    assert caps["metrics"]["accuracy"]["response_method"] == "predict"
    assert caps["metrics"]["accuracy"]["family"] == "label"
    assert caps["models"]["lr"]["input_ranks"] == ("2d",)


def test_selector_capability_metadata_is_available():
    k_best = get_selector_capabilities("k_best")
    sfs = get_selector_capabilities("sfs")

    assert k_best.input_ranks == ("2d",)
    assert "univariate" in k_best.support
    assert "sfs_metadata_routing" in sfs.grouped_metadata
