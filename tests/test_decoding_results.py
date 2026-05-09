import numpy as np

from coco_pipe.decoding.cache import make_feature_cache_key
from coco_pipe.decoding.configs import (
    CVConfig,
    ExperimentConfig,
    LogisticRegressionConfig,
)
from coco_pipe.decoding.core import RESULT_SCHEMA_VERSION, Experiment, ExperimentResult


def _classification_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.8],
            [-1.0, -0.6],
            [-0.8, -0.4],
            [0.8, 0.4],
            [1.0, 0.6],
            [1.5, 0.8],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _config(output_dir=None):
    return ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=200)},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=0),
        output_dir=output_dir,
        tag="result_schema_test",
        n_jobs=1,
        verbose=False,
    )


def test_run_result_payload_stores_config_provenance_sample_ids_and_groups():
    X, y = _classification_data()
    sample_ids = np.array([f"sample_{idx}" for idx in range(len(y))])
    groups = np.array(["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"])
    sample_metadata = {
        "subject": ["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"],
        "session": ["visit1"] * len(y),
    }

    result = Experiment(_config()).run(
        X,
        y,
        groups=groups,
        sample_ids=sample_ids,
        sample_metadata=sample_metadata,
        observation_level="epoch",
        feature_names=["left", "right"],
    )

    payload = result.to_payload()
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["config"]["tag"] == "result_schema_test"
    assert payload["meta"]["task"] == "classification"
    assert payload["meta"]["n_samples"] == len(y)
    assert payload["meta"]["n_features"] == X.shape[1]
    assert payload["meta"]["observation_level"] == "epoch"
    assert payload["meta"]["inferential_unit"] == "subject"
    assert payload["meta"]["sample_metadata_columns"] == [
        "subject",
        "session",
        "site",
    ]
    assert "versions" in payload["meta"]

    predictions = result.get_predictions()
    assert {
        "SampleIndex",
        "SampleID",
        "Group",
        "Subject",
        "Session",
        "Site",
    }.issubset(predictions.columns)
    assert set(predictions["SampleID"]) == set(sample_ids)
    assert set(predictions["Group"]) == set(groups)
    assert set(predictions["Subject"]) == {"s0", "s1", "s2", "s3"}
    assert set(predictions["Session"]) == {"visit1"}

    splits = result.get_splits()
    assert set(splits["Set"]) == {"train", "test"}
    assert set(splits["SampleID"]) == set(sample_ids)
    assert set(splits["Group"]) == set(groups)
    assert {"Subject", "Session", "Site"}.issubset(splits.columns)

    ci = result.get_bootstrap_confidence_intervals(
        n_bootstraps=10,
        random_state=0,
    )
    assert set(ci["Unit"]) == {"subject"}


def test_duplicate_sample_ids_are_rejected():
    X, y = _classification_data()

    try:
        Experiment(_config()).run(X, y, sample_ids=["s0"] * len(y))
    except ValueError as exc:
        assert "sample_ids must be unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate sample_ids to fail.")


def test_observation_level_is_explicitly_limited():
    X, y = _classification_data()

    try:
        Experiment(_config()).run(X, y, observation_level="trial")
    except ValueError as exc:
        assert "observation_level must be 'sample' or 'epoch'" in str(exc)
    else:
        raise AssertionError("Expected invalid observation_level to fail.")


def test_sample_metadata_requires_subject_and_session():
    X, y = _classification_data()

    try:
        Experiment(_config()).run(
            X,
            y,
            sample_metadata={"subject": [f"s{idx}" for idx in range(len(y))]},
        )
    except ValueError as exc:
        assert "sample_metadata must include subject and session" in str(exc)
    else:
        raise AssertionError("Expected incomplete sample_metadata to fail.")


def test_explicit_inferential_unit_overrides_epoch_default():
    X, y = _classification_data()
    metadata = {
        "subject": [f"s{idx // 2}" for idx in range(len(y))],
        "session": ["visit1"] * len(y),
    }

    result = Experiment(_config()).run(
        X,
        y,
        sample_metadata=metadata,
        observation_level="epoch",
        inferential_unit="epoch",
    )

    assert result.meta["inferential_unit"] == "epoch"
    ci = result.get_bootstrap_confidence_intervals(
        n_bootstraps=10,
        random_state=0,
    )
    assert set(ci["Unit"]) == {"epoch"}


def test_save_load_roundtrip_preserves_decoding_payload(tmp_path):
    X, y = _classification_data()
    exp = Experiment(_config(output_dir=tmp_path))
    result = exp.run(X, y, sample_ids=[f"s{idx}" for idx in range(len(y))])

    path = exp.save_results()
    loaded = Experiment.load_results(path)

    assert loaded.schema_version == result.schema_version
    assert loaded.config["tag"] == result.config["tag"]
    assert loaded.meta["n_samples"] == result.meta["n_samples"]
    assert loaded.raw.keys() == result.raw.keys()
    assert loaded.to_payload()["schema_version"] == RESULT_SCHEMA_VERSION


def test_get_predictions_expands_temporal_arrays():
    raw = {
        "sliding": {
            "metrics": {},
            "predictions": [
                {
                    "sample_index": np.array([0, 1]),
                    "sample_id": np.array(["s0", "s1"]),
                    "group": np.array(["g0", "g1"]),
                    "y_true": np.array([0, 1]),
                    "y_pred": np.array([[0, 1], [1, 1]]),
                    "y_proba": np.array(
                        [
                            [[0.8, 0.4], [0.2, 0.6]],
                            [[0.3, 0.2], [0.7, 0.8]],
                        ]
                    ),
                }
            ],
        },
        "generalizing": {
            "metrics": {},
            "predictions": [
                {
                    "sample_index": np.array([0, 1]),
                    "sample_id": np.array(["s0", "s1"]),
                    "group": None,
                    "y_true": np.array([0, 1]),
                    "y_pred": np.array(
                        [
                            [[0, 1], [1, 0]],
                            [[1, 1], [0, 1]],
                        ]
                    ),
                    "y_proba": np.ones((2, 2, 2, 2)) * 0.5,
                }
            ],
        },
    }

    predictions = ExperimentResult(raw).get_predictions()

    sliding = predictions[predictions["Model"] == "sliding"]
    assert len(sliding) == 4
    assert set(sliding["Time"]) == {0, 1}
    assert {"y_proba_0", "y_proba_1"}.issubset(sliding.columns)

    generalizing = predictions[predictions["Model"] == "generalizing"]
    assert len(generalizing) == 8
    assert set(generalizing["TrainTime"]) == {0, 1}
    assert set(generalizing["TestTime"]) == {0, 1}
    assert generalizing["Group"].isna().all()


def test_get_detailed_scores_expands_temporal_scores():
    result = ExperimentResult(
        {
            "model": {
                "metrics": {
                    "accuracy": {
                        "mean": 0.5,
                        "std": 0.1,
                        "folds": [
                            0.75,
                            np.array([0.1, 0.2]),
                            np.array([[0.1, 0.2], [0.3, 0.4]]),
                        ],
                    }
                }
            }
        }
    )

    scores = result.get_detailed_scores()

    assert len(scores[scores["Fold"] == 0]) == 1
    assert set(scores[scores["Fold"] == 1]["Time"]) == {0, 1}
    matrix_scores = scores[scores["Fold"] == 2]
    assert len(matrix_scores) == 4
    assert set(matrix_scores["TrainTime"]) == {0, 1}
    assert set(matrix_scores["TestTime"]) == {0, 1}


def test_get_feature_importances_returns_named_aggregate_and_fold_tables():
    result = ExperimentResult(
        {
            "rf": {
                "metrics": {},
                "importances": {
                    "mean": np.array([0.25, 0.75]),
                    "std": np.array([0.05, 0.10]),
                    "raw": np.array([[0.2, 0.8], [0.3, 0.7]]),
                },
                "metadata": [{"feature_names": ["alpha", "beta"]}],
            }
        }
    )

    aggregate = result.get_feature_importances()
    assert aggregate.columns.tolist() == [
        "Model",
        "Feature",
        "FeatureName",
        "Mean",
        "Std",
    ]
    assert aggregate["FeatureName"].tolist() == ["alpha", "beta"]
    assert aggregate["Mean"].tolist() == [0.25, 0.75]

    fold_level = result.get_feature_importances(fold_level=True)
    assert fold_level.columns.tolist() == [
        "Model",
        "Fold",
        "Feature",
        "FeatureName",
        "Importance",
    ]
    assert len(fold_level) == 4
    assert set(fold_level["Fold"]) == {0, 1}


def test_feature_cache_key_tracks_split_preprocessing_and_backbone_identity():
    base = make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-a",
    )

    assert base == make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-a",
    )
    assert base != make_feature_cache_key(
        train_sample_ids=["s0"],
        test_sample_ids=["s1", "s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-a",
    )
    assert base != make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-b",
        backbone_fingerprint="backbone-a",
    )
    assert base != make_feature_cache_key(
        train_sample_ids=["s0", "s1"],
        test_sample_ids=["s2"],
        preprocessing_fingerprint="prep-a",
        backbone_fingerprint="backbone-b",
    )
