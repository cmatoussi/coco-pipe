import numpy as np
from sklearn.datasets import make_classification, make_regression

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    CVConfig,
    DecisionTreeRegressorConfig,
    DummyClassifierConfig,
    DummyRegressorConfig,
    LogisticRegressionConfig,
    RidgeConfig,
)


def test_binary_classification_baseline_multiple_metrics_and_predictions():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig(max_iter=200)},
            metrics=["accuracy", "roc_auc", "average_precision"],
            cv=CVConfig(
                strategy="stratified", n_splits=3, shuffle=True, random_state=0
            ),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y, sample_ids=[f"s{idx}" for idx in range(len(y))])

    summary = result.summary()
    assert set(summary.columns) >= {
        "accuracy_mean",
        "roc_auc_mean",
        "average_precision_mean",
    }
    predictions = result.get_predictions()
    assert len(predictions) == len(y)
    assert {"SampleID", "y_true", "y_pred", "y_proba_0", "y_proba_1"}.issubset(
        predictions.columns
    )


def test_multiclass_classification_baseline_runs():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=1,
    )
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig(max_iter=250)},
            metrics=["accuracy", "f1_macro"],
            cv=CVConfig(
                strategy="stratified", n_splits=3, shuffle=True, random_state=1
            ),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    summary = result.summary()
    assert "accuracy_mean" in summary.columns
    assert "f1_macro_mean" in summary.columns
    assert len(result.get_predictions()) == len(y)


def test_regression_baseline_runs_and_exports_predictions():
    X, y = make_regression(
        n_samples=45,
        n_features=4,
        n_informative=3,
        noise=0.1,
        random_state=2,
    )
    result = Experiment(
        ExperimentConfig(
            task="regression",
            models={"ridge": RidgeConfig()},
            metrics=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
            cv=CVConfig(strategy="kfold", n_splits=3, shuffle=True, random_state=2),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    summary = result.summary()
    assert set(summary.columns) >= {
        "r2_mean",
        "neg_mean_squared_error_mean",
        "neg_mean_absolute_error_mean",
    }
    predictions = result.get_predictions()
    assert len(predictions) == len(y)
    assert {"y_true", "y_pred"}.issubset(predictions.columns)


def test_multiple_models_and_failed_model_are_reported_independently():
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2))])
    y = np.array([0] * 10 + [1] * 10)
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={
                "dummy": DummyClassifierConfig(strategy="most_frequent"),
                "bad": LogisticRegressionConfig(
                    penalty="l1",
                    solver="lbfgs",
                    max_iter=100,
                ),
            },
            metrics=["accuracy"],
            cv=CVConfig(
                strategy="stratified", n_splits=2, shuffle=True, random_state=0
            ),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    assert set(result.raw) == {"dummy", "bad"}
    assert "error" not in result.raw["dummy"]
    assert result.raw["bad"]["status"] == "failed"
    assert "lbfgs" in result.raw["bad"]["error"]
    assert "dummy" in result.summary().index


def test_named_feature_importances_from_tree_model():
    X, y = make_regression(
        n_samples=40,
        n_features=3,
        n_informative=2,
        random_state=3,
    )
    result = Experiment(
        ExperimentConfig(
            task="regression",
            models={"tree": DecisionTreeRegressorConfig(random_state=0)},
            metrics=["r2"],
            cv=CVConfig(strategy="kfold", n_splits=2, shuffle=True, random_state=3),
            use_scaler=False,
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y, feature_names=["alpha", "beta", "gamma"])

    importances = result.get_feature_importances()
    assert importances["FeatureName"].tolist() == ["alpha", "beta", "gamma"]
    assert importances["Mean"].notna().all()


def test_regression_failed_model_does_not_hide_successful_model():
    X, y = make_regression(n_samples=30, n_features=2, random_state=4)
    result = Experiment(
        ExperimentConfig(
            task="regression",
            models={
                "dummy": DummyRegressorConfig(strategy="mean"),
                "bad": RidgeConfig(solver="not_a_solver"),
            },
            metrics=["r2"],
            cv=CVConfig(strategy="kfold", n_splits=2, shuffle=True, random_state=4),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    assert "error" not in result.raw["dummy"]
    assert result.raw["bad"]["status"] == "failed"
    assert "not_a_solver" in result.raw["bad"]["error"]
