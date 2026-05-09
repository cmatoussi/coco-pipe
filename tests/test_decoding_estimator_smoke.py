import warnings

import numpy as np
import pytest

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.capabilities import list_estimator_specs, resolve_estimator_spec
from coco_pipe.decoding.configs import (
    AdaBoostClassifierConfig,
    AdaBoostRegressorConfig,
    ARDRegressionConfig,
    BayesianRidgeConfig,
    CVConfig,
    DecisionTreeRegressorConfig,
    DummyClassifierConfig,
    DummyRegressorConfig,
    ElasticNetConfig,
    ExtraTreesRegressorConfig,
    FeatureSelectionConfig,
    GaussianNBConfig,
    GradientBoostingClassifierConfig,
    GradientBoostingRegressorConfig,
    HistGradientBoostingRegressorConfig,
    KNeighborsClassifierConfig,
    KNeighborsRegressorConfig,
    LassoConfig,
    LDAConfig,
    LinearRegressionConfig,
    LinearSVCConfig,
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


def _classification_data():
    rng = np.random.default_rng(10)
    y = np.tile([0, 1], 16)
    X = rng.normal(size=(len(y), 6))
    X[:, 0] += y * 2.0
    X[:, 1] -= y * 1.0
    return X, y


def _regression_data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(28, 5))
    y = X[:, 0] * 1.5 - X[:, 1] * 0.5 + rng.normal(scale=0.05, size=X.shape[0])
    return X, y


CLASSIFIER_SMOKE_CONFIGS = {
    "DummyClassifier": DummyClassifierConfig(strategy="prior"),
    "LogisticRegression": LogisticRegressionConfig(solver="liblinear", max_iter=200),
    "LinearSVC": LinearSVCConfig(max_iter=500),
    "LinearDiscriminantAnalysis": LDAConfig(),
    "RandomForestClassifier": RandomForestClassifierConfig(
        n_estimators=5, max_depth=3, n_jobs=1
    ),
    "SVC": SVCConfig(kernel="linear", probability=True, max_iter=500),
    "KNeighborsClassifier": KNeighborsClassifierConfig(n_neighbors=3),
    "GradientBoostingClassifier": GradientBoostingClassifierConfig(n_estimators=5),
    "SGDClassifier": SGDClassifierConfig(max_iter=500),
    "MLPClassifier": MLPClassifierConfig(hidden_layer_sizes=(4,), max_iter=50),
    "GaussianNB": GaussianNBConfig(),
    "AdaBoostClassifier": AdaBoostClassifierConfig(n_estimators=5),
}


REGRESSOR_SMOKE_CONFIGS = {
    "DummyRegressor": DummyRegressorConfig(strategy="mean"),
    "Ridge": RidgeConfig(),
    "RandomForestRegressor": RandomForestRegressorConfig(
        n_estimators=5, max_depth=3, n_jobs=1
    ),
    "LinearRegression": LinearRegressionConfig(),
    "Lasso": LassoConfig(max_iter=500),
    "ElasticNet": ElasticNetConfig(max_iter=500),
    "SVR": SVRConfig(kernel="linear"),
    "GradientBoostingRegressor": GradientBoostingRegressorConfig(n_estimators=5),
    "SGDRegressor": SGDRegressorConfig(max_iter=500),
    "MLPRegressor": MLPRegressorConfig(hidden_layer_sizes=(4,), max_iter=50),
    "DecisionTreeRegressor": DecisionTreeRegressorConfig(max_depth=3),
    "KNeighborsRegressor": KNeighborsRegressorConfig(n_neighbors=3),
    "ExtraTreesRegressor": ExtraTreesRegressorConfig(
        n_estimators=5, max_depth=3, n_jobs=1
    ),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressorConfig(
        max_iter=5, min_samples_leaf=2
    ),
    "AdaBoostRegressor": AdaBoostRegressorConfig(n_estimators=5),
    "BayesianRidge": BayesianRidgeConfig(),
    "ARDRegression": ARDRegressionConfig(),
}


SMOKE_CONFIGS = {
    **CLASSIFIER_SMOKE_CONFIGS,
    **REGRESSOR_SMOKE_CONFIGS,
}


def _instantiation_experiment():
    return Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig(max_iter=100)},
            metrics=["accuracy"],
            n_jobs=1,
            verbose=False,
        )
    )


def test_every_fit_smoke_required_estimator_has_a_smoke_case():
    required = {
        name for name, spec in list_estimator_specs().items() if spec.fit_smoke_required
    }

    assert required <= set(SMOKE_CONFIGS)


@pytest.mark.parametrize("method", sorted(SMOKE_CONFIGS))
def test_registered_estimator_survives_fit_predict_and_declared_responses(method):
    config = SMOKE_CONFIGS[method]
    spec = resolve_estimator_spec(config)
    X, y = (
        _classification_data() if "classification" in spec.task else _regression_data()
    )
    X_test = X[:5]

    estimator = _instantiation_experiment()._instantiate_model(method, config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X, y)
        y_pred = estimator.predict(X_test)

        assert y_pred.shape[0] == X_test.shape[0]

        if spec.supports_proba:
            y_proba = estimator.predict_proba(X_test)
            assert y_proba.shape[0] == X_test.shape[0]

        if spec.supports_decision_function:
            y_score = estimator.decision_function(X_test)
            assert y_score.shape[0] == X_test.shape[0]


def test_select_k_best_pipeline_survives_outer_cv():
    X, y = _classification_data()
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig(solver="liblinear", max_iter=200)},
            metrics=["accuracy"],
            cv=CVConfig(strategy="stratified", n_splits=2),
            feature_selection=FeatureSelectionConfig(
                enabled=True,
                method="k_best",
                n_features=3,
            ),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    assert result.raw["lr"]["status"] == "success"
    assert len(result.get_predictions()) == len(y)


def test_sequential_feature_selector_pipeline_survives_outer_cv():
    X, y = _classification_data()
    result = Experiment(
        ExperimentConfig(
            task="classification",
            models={"lr": LogisticRegressionConfig(solver="liblinear", max_iter=200)},
            metrics=["accuracy"],
            cv=CVConfig(strategy="stratified", n_splits=2),
            feature_selection=FeatureSelectionConfig(
                enabled=True,
                method="sfs",
                n_features=3,
                cv=CVConfig(strategy="stratified", n_splits=2),
            ),
            n_jobs=1,
            verbose=False,
        )
    ).run(X, y)

    assert result.raw["lr"]["status"] == "success"
    assert len(result.get_predictions()) == len(y)
