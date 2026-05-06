import matplotlib.pyplot as plt
import numpy as np
import pytest

from coco_pipe.decoding.configs import (
    CVConfig,
    ExperimentConfig,
    GeneralizingEstimatorConfig,
    LogisticRegressionConfig,
    SlidingEstimatorConfig,
)
from coco_pipe.decoding.core import Experiment, ExperimentResult
from coco_pipe.report.core import Report
from coco_pipe.viz.decoding import (
    plot_temporal_generalization_matrix,
    plot_temporal_score_curve,
)


def _temporal_data(n_samples=12, n_channels=3, n_times=4):
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * (n_samples // 2))
    X = rng.normal(scale=0.1, size=(n_samples, n_channels, n_times))
    X[y == 1, 0, :] += 1.0
    X[y == 0, 0, :] -= 1.0
    return X, y


def _time_axis():
    return np.array([-0.1, 0.0, 0.1, 0.2])


def test_sliding_estimator_preserves_time_axis_in_scores_and_predictions():
    pytest.importorskip("mne")
    X, y = _temporal_data()
    times = _time_axis()
    config = ExperimentConfig(
        task="classification",
        models={
            "sliding": SlidingEstimatorConfig(
                base_estimator=LogisticRegressionConfig(max_iter=200),
                scoring="accuracy",
                n_jobs=1,
            )
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=0),
        use_scaler=True,
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, time_axis=times)

    predictions = result.get_predictions()
    assert set(predictions["Time"]) == set(times)

    scores = result.get_detailed_scores()
    assert set(scores["Time"].dropna()) == set(times)

    temporal_summary = result.get_temporal_score_summary()
    assert set(temporal_summary["Time"].dropna()) == set(times)
    assert result.summary().empty


def test_generalizing_estimator_produces_time_labeled_matrix_scores():
    pytest.importorskip("mne")
    X, y = _temporal_data()
    times = _time_axis()
    config = ExperimentConfig(
        task="classification",
        models={
            "generalizing": GeneralizingEstimatorConfig(
                base_estimator=LogisticRegressionConfig(max_iter=200),
                scoring="accuracy",
                n_jobs=1,
            )
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=0),
        use_scaler=True,
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y, time_axis=times)

    scores = result.get_detailed_scores()
    assert set(scores["TrainTime"].dropna()) == set(times)
    assert set(scores["TestTime"].dropna()) == set(times)

    matrix = result.get_generalization_matrix("accuracy")
    assert matrix.index.tolist() == times.tolist()
    assert matrix.columns.tolist() == times.tolist()


def test_4d_probability_metric_scoring_is_reached():
    from sklearn.metrics import roc_auc_score

    y_true = np.array([0, 1, 0, 1])
    y_proba = np.zeros((4, 2, 2, 2))
    y_proba[:, 1, :, :] = np.array([0.1, 0.8, 0.2, 0.9])[:, None, None]
    y_proba[:, 0, :, :] = 1.0 - y_proba[:, 1, :, :]

    scores = Experiment._compute_metric_safe(
        roc_auc_score,
        y_true,
        y_proba,
        is_multiclass=False,
        is_proba=True,
    )

    assert scores.shape == (2, 2)
    assert np.allclose(scores, 1.0)


def test_temporal_accessors_plots_and_report_use_time_axis():
    times = ["t0", "t1", "t2"]
    result = ExperimentResult(
        {
            "sliding": {
                "metrics": {
                    "accuracy": {
                        "mean": np.array([0.6, 0.7, 0.8]),
                        "std": np.array([0.01, 0.02, 0.03]),
                        "folds": [
                            np.array([0.5, 0.7, 0.9]),
                            np.array([0.7, 0.7, 0.7]),
                        ],
                    }
                },
                "predictions": [],
            },
            "generalizing": {
                "metrics": {
                    "accuracy": {
                        "mean": np.ones((3, 3)),
                        "std": np.zeros((3, 3)),
                        "folds": [np.ones((3, 3)), np.ones((3, 3))],
                    }
                },
                "predictions": [],
            },
        },
        meta={"time_axis": times},
    )

    temporal_summary = result.get_temporal_score_summary()
    assert set(temporal_summary["Time"].dropna()) == set(times)
    assert set(temporal_summary["TrainTime"].dropna()) == set(times)
    assert set(temporal_summary["TestTime"].dropna()) == set(times)

    fig_curve = plot_temporal_score_curve(result, model="sliding")
    assert isinstance(fig_curve, plt.Figure)
    plt.close(fig_curve)

    fig_matrix = plot_temporal_generalization_matrix(result, model="generalizing")
    assert isinstance(fig_matrix, plt.Figure)
    plt.close(fig_matrix)

    report = Report("Temporal")
    report.add_decoding_temporal(result)
    html = report.render()
    assert "Temporal Decoding" in html
    assert "Temporal Score Summary" in html
