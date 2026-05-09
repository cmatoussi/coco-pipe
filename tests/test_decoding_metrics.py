import numpy as np
import pytest

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig
from coco_pipe.decoding.metrics import (
    get_metric_families,
    get_metric_names,
    get_metric_spec,
    get_scorer,
)


def test_classification_scorers():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert get_scorer("accuracy")(y_true, y_pred) == pytest.approx(0.75)
    assert get_scorer("balanced_accuracy")(y_true, y_pred) == pytest.approx(0.75)
    assert get_scorer("f1")(y_true, y_pred) == pytest.approx(0.7333333333)
    assert get_scorer("f1_macro")(y_true, y_pred) == pytest.approx(0.7333333333)
    assert get_scorer("f1_micro")(y_true, y_pred) == pytest.approx(0.75)
    assert get_scorer("precision")(y_true, y_pred) == pytest.approx(0.8333333333)
    assert get_scorer("recall")(y_true, y_pred) == pytest.approx(0.75)


def test_binary_classification_specialized_scorers():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    y_proba = np.array([0.25, 0.75, 0.75, 0.25])

    assert get_scorer("sensitivity")(y_true, y_pred) == pytest.approx(0.5)
    assert get_scorer("specificity")(y_true, y_pred) == pytest.approx(0.5)
    assert get_scorer("average_precision")(y_true, y_score) == pytest.approx(
        0.8333333333
    )
    assert get_scorer("pr_auc")(y_true, y_score) == pytest.approx(0.8333333333)
    assert get_scorer("brier_score")(y_true, y_proba) == pytest.approx(0.0625)
    assert get_scorer("log_loss")(y_true, y_proba) == pytest.approx(0.287682072)


def test_roc_auc_scorer():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    assert get_scorer("roc_auc")(y_true, y_score) == pytest.approx(0.75)


def test_precision_zero_division_returns_zero():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.zeros_like(y_true)

    assert get_scorer("precision")(y_true, y_pred) == pytest.approx(0.25)


def test_regression_scorers():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    assert get_scorer("r2")(y_true, y_pred) == pytest.approx(0.948608137)
    assert get_scorer("neg_mean_squared_error")(y_true, y_pred) == pytest.approx(-0.375)
    assert get_scorer("neg_mean_absolute_error")(y_true, y_pred) == pytest.approx(-0.5)
    assert get_scorer("explained_variance")(y_true, y_pred) == pytest.approx(
        0.9571734475
    )


def test_metric_registry_exposes_task_metadata():
    assert get_metric_spec("roc_auc").task == "classification"
    assert get_metric_spec("roc_auc").response_method == "proba_or_score"
    assert get_metric_spec("roc_auc").family == "threshold_sweep"
    assert get_metric_spec("log_loss").response_method == "proba"
    assert get_metric_spec("log_loss").family == "score_probability"
    assert get_metric_spec("brier_score").family == "calibration"
    assert "accuracy" in get_metric_names("classification")
    assert "r2" in get_metric_names("regression")


def test_metric_registry_exposes_family_metadata_and_filters():
    assert "roc_auc" in get_metric_names(family="threshold_sweep")
    assert "average_precision" in get_metric_names(
        task="classification",
        family="threshold_sweep",
    )
    assert get_metric_names(task="regression", family="threshold_sweep") == []
    assert "confusion" in get_metric_families("classification")
    assert get_metric_families("regression") == ["regression"]


def test_metric_task_validation_uses_registry():
    with pytest.raises(ValueError, match="Available regression metrics"):
        Experiment(
            ExperimentConfig(
                task="regression",
                models={"ridge": {"method": "Ridge"}},
                metrics=["accuracy"],
                cv=CVConfig(strategy="kfold", n_splits=3),
                n_jobs=1,
                verbose=False,
            )
        )


def test_roc_auc_can_use_decision_function_fallback():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = np.tile([0, 1], 15)
    X[:, 0] += y * 1.5

    config = ExperimentConfig(
        task="classification",
        models={
            "svc": {
                "method": "SVC",
                "kernel": "linear",
                "probability": False,
            }
        },
        metrics=["roc_auc"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y)

    assert "error" not in result.raw["svc"]
    assert not np.isnan(result.raw["svc"]["metrics"]["roc_auc"]["folds"]).any()


def test_probability_metric_requires_predict_proba():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    y = np.tile([0, 1], 15)

    config = ExperimentConfig(
        task="classification",
        models={
            "svc": {
                "method": "SVC",
                "kernel": "linear",
                "probability": False,
            }
        },
        metrics=["log_loss"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y)

    assert result.raw["svc"]["status"] == "failed"
    assert "requires predict_proba" in result.raw["svc"]["error"]


def test_unknown_metric_raises_helpful_error():
    with pytest.raises(ValueError, match="Unknown metric 'not_a_metric'"):
        get_scorer("not_a_metric")
