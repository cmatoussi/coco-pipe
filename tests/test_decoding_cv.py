import numpy as np
import pytest
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, TuningConfig
from coco_pipe.decoding.splitters import SimpleSplit, get_cv_splitter


def test_stratified_and_kfold_splitters_construct_from_config():
    stratified = get_cv_splitter(CVConfig(strategy="stratified", n_splits=4))
    assert isinstance(stratified, StratifiedKFold)
    assert stratified.get_n_splits() == 4

    kfold = get_cv_splitter(CVConfig(strategy="kfold", n_splits=3, shuffle=False))
    assert isinstance(kfold, KFold)
    assert kfold.get_n_splits() == 3


@pytest.mark.parametrize(
    "strategy",
    [
        "group_kfold",
        "stratified_group_kfold",
        "leave_p_out",
        "leave_one_group_out",
    ],
)
def test_group_strategies_require_groups(strategy):
    with pytest.raises(ValueError, match="requires groups"):
        get_cv_splitter(CVConfig(strategy=strategy, n_splits=2))


def test_group_kfold_has_no_train_test_group_overlap():
    X = np.zeros((12, 2))
    y = np.array([0, 1] * 6)
    groups = np.repeat(np.arange(6), 2)

    splitter = get_cv_splitter(
        CVConfig(strategy="group_kfold", n_splits=3), groups=groups
    )
    assert isinstance(splitter.cv, GroupKFold)

    for train_idx, test_idx in splitter.split(X, y):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_stratified_group_kfold_has_no_train_test_group_overlap():
    X = np.zeros((24, 2))
    y = np.tile([0, 1, 0, 1], 6)
    groups = np.repeat(np.arange(6), 4)

    splitter = get_cv_splitter(
        CVConfig(strategy="stratified_group_kfold", n_splits=3),
        groups=groups,
    )
    assert isinstance(splitter.cv, StratifiedGroupKFold)

    for train_idx, test_idx in splitter.split(X, y):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_leave_one_group_out_has_no_train_test_group_overlap():
    X = np.zeros((12, 2))
    y = np.array([0, 1] * 6)
    groups = np.repeat(np.arange(6), 2)

    splitter = get_cv_splitter(
        CVConfig(strategy="leave_one_group_out", n_splits=2),
        groups=groups,
    )
    assert isinstance(splitter.cv, LeaveOneGroupOut)

    observed_test_groups = []
    for train_idx, test_idx in splitter.split(X, y):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert len(test_groups) == 1
        assert train_groups.isdisjoint(test_groups)
        observed_test_groups.extend(test_groups)

    assert set(observed_test_groups) == set(groups)


def test_timeseries_splitter_preserves_time_order():
    X = np.zeros((12, 2))
    y = np.arange(12)

    splitter = get_cv_splitter(CVConfig(strategy="timeseries", n_splits=3))
    assert isinstance(splitter, TimeSeriesSplit)

    for train_idx, test_idx in splitter.split(X, y):
        assert train_idx.max() < test_idx.min()


def test_holdout_split_uses_test_size():
    X = np.zeros((20, 2))
    y = np.array([0, 1] * 10)

    splitter = get_cv_splitter(
        CVConfig(strategy="split", n_splits=2, test_size=0.3, random_state=0)
    )
    assert isinstance(splitter, SimpleSplit)

    train_idx, test_idx = next(splitter.split(X, y))
    assert len(train_idx) == 14
    assert len(test_idx) == 6


def test_holdout_split_can_stratify_by_y():
    X = np.zeros((20, 2))
    y = np.array([0] * 10 + [1] * 10)

    splitter = get_cv_splitter(
        CVConfig(
            strategy="split",
            n_splits=2,
            test_size=0.4,
            stratify=True,
            random_state=0,
        ),
        y=y,
    )
    train_idx, test_idx = next(splitter.split(X, y))

    assert set(y[train_idx]) == {0, 1}
    assert set(y[test_idx]) == {0, 1}
    assert np.bincount(y[test_idx]).tolist() == [4, 4]


def test_grouped_outer_cv_experiment_respects_group_boundaries():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 4))
    y = np.tile([0, 1, 0, 1], 6)
    groups = np.repeat(np.arange(6), 4)

    config = ExperimentConfig(
        task="classification",
        models={
            "lr": {
                "method": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 200,
            }
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, groups=groups)
    assert "lr" in result.raw
    assert "error" not in result.raw["lr"]

    for test_idx in result.raw["lr"]["indices"]:
        test_idx = np.asarray(test_idx)
        for group in set(groups[test_idx]):
            assert set(np.flatnonzero(groups == group)).issubset(set(test_idx))


def test_tuning_defaults_to_outer_group_cv_family():
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": {
                "method": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 200,
            }
        },
        grids={"lr": {"C": [0.1, 1.0]}},
        tuning=TuningConfig(enabled=True, scoring="accuracy", n_jobs=1),
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    estimator = Experiment(config)._prepare_estimator("lr", config.models["lr"])

    assert isinstance(estimator.cv, GroupKFold)


def test_nongroup_tuning_cv_under_grouped_outer_requires_override():
    with pytest.raises(ValueError, match="allow_nongroup_inner_cv"):
        Experiment(
            ExperimentConfig(
                task="classification",
                models={
                    "lr": {
                        "method": "LogisticRegression",
                        "solver": "liblinear",
                        "max_iter": 200,
                    }
                },
                grids={"lr": {"C": [0.1, 1.0]}},
                tuning=TuningConfig(
                    enabled=True,
                    scoring="accuracy",
                    n_jobs=1,
                    cv=CVConfig(strategy="stratified", n_splits=2),
                ),
                metrics=["accuracy"],
                cv=CVConfig(strategy="group_kfold", n_splits=3),
                n_jobs=1,
                verbose=False,
            )
        )


def test_nongroup_tuning_cv_under_grouped_outer_allows_explicit_override():
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": {
                "method": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 200,
            }
        },
        grids={"lr": {"C": [0.1, 1.0]}},
        tuning=TuningConfig(
            enabled=True,
            scoring="accuracy",
            n_jobs=1,
            cv=CVConfig(strategy="stratified", n_splits=2),
            allow_nongroup_inner_cv=True,
        ),
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    estimator = Experiment(config)._prepare_estimator("lr", config.models["lr"])

    assert isinstance(estimator.cv, StratifiedKFold)


def test_grouped_tuning_receives_training_fold_groups():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 5))
    y = np.tile([0, 1, 0, 1], 8)
    groups = np.repeat(np.arange(8), 4)

    config = ExperimentConfig(
        task="classification",
        models={
            "lr": {
                "method": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 200,
            }
        },
        grids={"lr": {"C": [0.1, 1.0]}},
        tuning=TuningConfig(
            enabled=True,
            scoring="accuracy",
            n_jobs=1,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
        ),
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=4),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, groups=groups)

    assert "error" not in result.raw["lr"]
    best_params = result.get_best_params()
    assert not best_params.empty
    assert set(best_params["Param"]) == {"clf__C"}

    metadata = result.raw["lr"]["metadata"][0]
    assert "best_score" in metadata
    assert "best_index" in metadata
    assert metadata["search_results"]

    search_results = result.get_search_results()
    assert not search_results.empty
    assert set(search_results["Params"].iloc[0]) == {"clf__C"}
    assert result.raw["lr"]["importances"] is not None


def test_random_search_uses_tuning_random_state():
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": {
                "method": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 200,
            }
        },
        grids={"lr": {"C": [0.1, 1.0, 10.0]}},
        tuning=TuningConfig(
            enabled=True,
            search_type="random",
            n_iter=2,
            scoring="accuracy",
            n_jobs=1,
            random_state=7,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    estimator = Experiment(config)._prepare_estimator("lr", config.models["lr"])

    assert isinstance(estimator, RandomizedSearchCV)
    assert estimator.random_state == 7


def test_invalid_tuning_grid_key_fails_before_fit_with_clear_error():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(24, 4))
    y = np.tile([0, 1], 12)

    config = ExperimentConfig(
        task="classification",
        models={
            "lr": {
                "method": "LogisticRegression",
                "solver": "liblinear",
                "max_iter": 200,
            }
        },
        grids={"lr": {"not_a_parameter": [1, 2]}},
        tuning=TuningConfig(
            enabled=True,
            scoring="accuracy",
            n_jobs=1,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y)

    assert result.raw["lr"]["status"] == "failed"
    assert "Invalid tuning grid key" in result.raw["lr"]["error"]
