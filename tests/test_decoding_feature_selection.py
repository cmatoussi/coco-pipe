import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    CVConfig,
    FeatureSelectionConfig,
    TuningConfig,
)


def _classification_data(n_samples=24, n_features=4):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = np.tile([0, 1], n_samples // 2)
    X[:, 0] += y * 1.5
    X[:, 1] += y * 0.75
    return X, y


def _lr_model():
    return {
        "method": "LogisticRegression",
        "solver": "liblinear",
        "max_iter": 200,
    }


def test_k_best_default_all_handles_fewer_than_ten_features():
    X, y = _classification_data(n_features=4)

    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="k_best",
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y)

    assert "error" not in result.raw["lr"]
    selected = result.get_selected_features()
    assert not selected.empty
    assert set(selected["FeatureName"]) == {
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    }
    assert selected.groupby(["Model", "Fold"])["Selected"].sum().eq(4).all()


def test_k_best_explicit_records_indices_names_and_scores():
    X, y = _classification_data(n_features=4)
    feature_names = ["alpha", "beta", "theta", "delta"]

    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="k_best",
            n_features=2,
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, feature_names=feature_names)
    meta = result.raw["lr"]["metadata"][0]

    assert meta["feature_selection_method"] == "k_best"
    assert "selected_feature_indices" in meta
    assert len(meta["selected_feature_indices"]) == 2
    assert set(meta["selected_feature_names"]).issubset(set(feature_names))
    assert len(meta["feature_scores"]) == 4
    assert len(meta["feature_pvalues"]) == 4

    selected = result.get_selected_features()
    assert list(selected.columns) == [
        "Model",
        "Fold",
        "Feature",
        "FeatureName",
        "Selected",
    ]
    assert set(selected["FeatureName"]) == set(feature_names)

    scores = result.get_feature_scores()
    assert list(scores.columns) == [
        "Model",
        "Fold",
        "Feature",
        "FeatureName",
        "Selector",
        "Score",
        "PValue",
        "Selected",
    ]
    assert not scores.empty
    assert set(scores["FeatureName"]) == set(feature_names)
    assert set(scores["Selector"]) == {"k_best"}
    assert scores["Score"].notna().all()

    stability = result.get_feature_stability()
    assert "FeatureName" in stability.columns
    assert set(stability["FeatureName"]) == set(feature_names)


def test_feature_names_must_align_with_array_feature_dimension():
    X, y = _classification_data(n_features=4)

    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        n_jobs=1,
        verbose=False,
    )

    with pytest.raises(ValueError, match="feature_names must align"):
        Experiment(config).run(X, y, feature_names=["alpha", "beta"])


def test_sfs_requires_explicit_feature_selection_cv():
    with pytest.raises(ValueError, match="feature_selection.cv"):
        Experiment(
            ExperimentConfig(
                task="classification",
                models={"lr": _lr_model()},
                metrics=["accuracy"],
                cv=CVConfig(strategy="stratified", n_splits=3),
                feature_selection=FeatureSelectionConfig(
                    enabled=True,
                    method="sfs",
                    n_features=2,
                ),
                n_jobs=1,
                verbose=False,
            )
        )


def test_group_based_sfs_cv_uses_group_splitter():
    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )

    experiment = Experiment(config)
    estimator = experiment._prepare_estimator("lr", experiment.config.models["lr"])

    assert isinstance(estimator.named_steps["fs"].cv, GroupKFold)


def test_group_based_sfs_cv_requires_groups_at_run():
    X, y = _classification_data(n_samples=24, n_features=4)
    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )

    with pytest.raises(ValueError, match="requires groups"):
        Experiment(config).run(X, y)


def test_group_based_sfs_cv_runs_with_groups():
    X, y = _classification_data(n_samples=24, n_features=4)
    groups = np.repeat(np.arange(6), 4)

    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=2),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, groups=groups)

    assert "error" not in result.raw["lr"]
    selected = result.get_selected_features()
    assert not selected.empty
    assert selected.groupby(["Model", "Fold"])["Selected"].sum().eq(2).all()
    assert result.get_feature_scores().empty


def test_group_based_sfs_cv_has_no_inner_group_overlap(monkeypatch):
    original_split = GroupKFold.split
    observed_splits = []

    def recording_split(self, X, y=None, groups=None):
        for train_idx, test_idx in original_split(self, X, y, groups):
            if groups is not None:
                group_values = np.asarray(groups)
                observed_splits.append(
                    {
                        "n_samples": len(group_values),
                        "train_groups": set(group_values[train_idx]),
                        "test_groups": set(group_values[test_idx]),
                    }
                )
            yield train_idx, test_idx

    monkeypatch.setattr(GroupKFold, "split", recording_split)

    X, y = _classification_data(n_samples=32, n_features=4)
    groups = np.repeat(np.arange(8), 4)

    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="group_kfold", n_splits=2),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="group_kfold", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, groups=groups)

    assert "error" not in result.raw["lr"]
    assert observed_splits
    assert any(split["n_samples"] < len(groups) for split in observed_splits)
    assert all(
        split["train_groups"].isdisjoint(split["test_groups"])
        for split in observed_splits
    )


def test_sfs_uses_feature_selection_scoring_when_set():
    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            scoring="balanced_accuracy",
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )

    estimator = Experiment(config)._prepare_estimator("lr", config.models["lr"])

    assert estimator.named_steps["fs"].scoring == "balanced_accuracy"


def test_sfs_allows_stratified_holdout_feature_selection_cv():
    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(
                strategy="split",
                n_splits=2,
                test_size=0.25,
                stratify=True,
                random_state=0,
            ),
        ),
        n_jobs=1,
        verbose=False,
    )

    estimator = Experiment(config)._prepare_estimator("lr", config.models["lr"])

    assert estimator.named_steps["fs"].cv.stratify is True


def test_sfs_scoring_falls_back_to_tuning_then_first_metric():
    tuning_config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        grids={"lr": {"C": [0.1, 1.0]}},
        metrics=["f1_macro"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        tuning=TuningConfig(
            enabled=True,
            scoring="accuracy",
            n_jobs=1,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )
    tuning_estimator = Experiment(tuning_config)._prepare_estimator(
        "lr", tuning_config.models["lr"]
    )

    assert tuning_estimator.estimator.named_steps["fs"].scoring == "accuracy"

    metric_config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        metrics=["f1_macro"],
        cv=CVConfig(strategy="stratified", n_splits=3),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )
    metric_estimator = Experiment(metric_config)._prepare_estimator(
        "lr", metric_config.models["lr"]
    )

    assert metric_estimator.named_steps["fs"].scoring == "f1_macro"


def test_sfs_with_tuning_records_selected_feature_names_from_best_estimator():
    X, y = _classification_data(n_samples=30, n_features=4)
    feature_names = ["alpha", "beta", "theta", "delta"]

    config = ExperimentConfig(
        task="classification",
        models={"lr": _lr_model()},
        grids={"lr": {"C": [0.1, 1.0]}},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2),
        tuning=TuningConfig(
            enabled=True,
            scoring="accuracy",
            n_jobs=1,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            method="sfs",
            n_features=2,
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, feature_names=feature_names)

    assert "error" not in result.raw["lr"]
    assert result.raw["lr"]["metadata"][0]["feature_selection_method"] == "sfs"
    selected = result.get_selected_features()
    assert not selected.empty
    assert set(selected["FeatureName"]) == set(feature_names)
    assert selected.groupby(["Model", "Fold"])["Selected"].sum().eq(2).all()
    assert result.get_feature_scores().empty
