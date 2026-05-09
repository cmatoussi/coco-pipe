import numpy as np
import pytest
from scipy.stats import binom
from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    ChanceAssessmentConfig,
    ClassicalModelConfig,
    CVConfig,
    StatisticalAssessmentConfig,
    TuningConfig,
)
from coco_pipe.decoding.core import ExperimentResult
from coco_pipe.decoding.stats import (
    aggregate_predictions_for_inference,
    binomial_accuracy_test,
)
from coco_pipe.report.core import Report
from coco_pipe.viz.decoding import plot_temporal_statistical_assessment


def _prediction_frame():
    return ExperimentResult(
        {
            "m": {
                "metrics": {},
                "predictions": [
                    {
                        "sample_index": np.arange(6),
                        "sample_id": np.array(["e0", "e1", "e2", "e3", "e4", "e5"]),
                        "group": np.array(["s0", "s0", "s1", "s1", "s2", "s2"]),
                        "sample_metadata": {
                            "subject": ["s0", "s0", "s1", "s1", "s2", "s2"],
                            "session": ["v1"] * 6,
                            "site": ["a", "a", "b", "b", "b", "b"],
                        },
                        "y_true": np.array([0, 0, 1, 1, 1, 1]),
                        "y_pred": np.array([0, 1, 1, 1, 0, 1]),
                        "y_proba": np.array(
                            [
                                [0.8, 0.2],
                                [0.4, 0.6],
                                [0.3, 0.7],
                                [0.2, 0.8],
                                [0.6, 0.4],
                                [0.4, 0.6],
                            ]
                        ),
                    }
                ],
            }
        }
    ).get_predictions()


def test_binomial_accuracy_test_returns_exact_tail_threshold_and_ci():
    result = binomial_accuracy_test(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        p0=0.5,
        alpha=0.05,
    )

    assert result["k_correct"] == 3
    assert result["n_eff"] == 4
    assert result["p_value"] == pytest.approx(binom.sf(2, 4, 0.5))
    assert 0 <= result["ci_lower"] <= result["observed"] <= result["ci_upper"] <= 1


def test_binomial_accuracy_test_requires_p0():
    with pytest.raises(ValueError, match="explicit p0"):
        binomial_accuracy_test([0, 1], [0, 1], p0=None)


def test_aggregation_sample_group_mean_group_majority_and_custom_units():
    predictions = _prediction_frame()

    sample = aggregate_predictions_for_inference(
        predictions,
        metric="accuracy",
        unit_of_inference="sample",
    )
    assert len(sample) == 6

    group_mean = aggregate_predictions_for_inference(
        predictions,
        metric="accuracy",
        unit_of_inference="group_mean",
    )
    assert group_mean["InferentialUnitID"].tolist() == ["s0", "s1", "s2"]
    assert "y_proba_0" in group_mean
    assert group_mean["y_pred"].tolist() == [0, 1, 1]

    group_majority = aggregate_predictions_for_inference(
        predictions,
        metric="accuracy",
        unit_of_inference="group_majority",
    )
    assert group_majority["y_pred"].tolist() == [0, 1, 0]

    custom = aggregate_predictions_for_inference(
        predictions,
        metric="accuracy",
        unit_of_inference="custom",
        custom_unit_column="subject",
        custom_aggregation="mean",
    )
    assert custom["InferentialUnitID"].tolist() == ["s0", "s1", "s2"]


def test_grouped_aggregation_rejects_inconsistent_true_labels():
    predictions = _prediction_frame()
    predictions.loc[predictions["Group"] == "s0", "y_true"] = [0, 1]

    with pytest.raises(ValueError, match="one true target"):
        aggregate_predictions_for_inference(
            predictions,
            metric="accuracy",
            unit_of_inference="group_mean",
        )


def test_binomial_assessment_rejects_non_accuracy_and_repeated_predictions():
    repeated = ExperimentResult(
        {
            "m": {
                "metrics": {},
                "predictions": [
                    {
                        "sample_index": np.array([0, 0]),
                        "sample_id": np.array(["s0", "s0"]),
                        "group": None,
                        "y_true": np.array([0, 0]),
                        "y_pred": np.array([0, 0]),
                    }
                ],
            }
        }
    )
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": ClassicalModelConfig(
                estimator="logistic_regression",
                params={"max_iter": 100},
            )
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2),
        evaluation=StatisticalAssessmentConfig(
            enabled=True,
            chance=ChanceAssessmentConfig(method="binomial", p0=0.5),
        ),
        n_jobs=1,
        verbose=False,
    )

    with pytest.raises(ValueError, match="one held-out prediction"):
        from coco_pipe.decoding.stats import run_statistical_assessment

        run_statistical_assessment(
            repeated,
            config,
            np.ones((2, 2)),
            np.array([0, 0]),
            None,
            np.array(["s0", "s0"]),
            None,
            ["a", "b"],
            None,
            "sample",
            "sample",
        )

    config.evaluation.metrics = ["balanced_accuracy"]
    with pytest.raises(ValueError, match="classification accuracy"):
        from coco_pipe.decoding.stats import run_statistical_assessment

        run_statistical_assessment(
            repeated,
            config,
            np.ones((2, 2)),
            np.array([0, 0]),
            None,
            np.array(["s0", "s0"]),
            None,
            ["a", "b"],
            None,
            "sample",
            "sample",
        )


def test_enabled_permutation_assessment_reruns_pipeline_and_stores_rows():
    X, y = make_classification(
        n_samples=24,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=3,
    )
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": ClassicalModelConfig(
                estimator="logistic_regression",
                params={"max_iter": 200, "solver": "liblinear"},
            )
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=3),
        evaluation=StatisticalAssessmentConfig(
            enabled=True,
            chance=ChanceAssessmentConfig(
                method="permutation",
                n_permutations=2,
                unit_of_inference="sample",
                store_null_distribution=True,
            ),
            random_state=4,
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y)

    assessment = result.get_statistical_assessment()
    assert not assessment.empty
    assert assessment.loc[0, "NullMethod"] == "permutation_full_pipeline"
    assert assessment.loc[0, "NPermutations"] == 2
    assert "statistical_assessment" in result.meta
    assert "lr" in result.get_statistical_nulls()


def test_permutation_assessment_works_with_tuning_path():
    X, y = make_classification(
        n_samples=24,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=5,
    )
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": ClassicalModelConfig(
                estimator="logistic_regression",
                params={"max_iter": 200, "solver": "liblinear"},
            )
        },
        grids={"lr": {"C": [0.1, 1.0]}},
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=5),
        tuning=TuningConfig(
            enabled=True,
            cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True),
            scoring="accuracy",
            n_jobs=1,
        ),
        evaluation=StatisticalAssessmentConfig(
            enabled=True,
            chance=ChanceAssessmentConfig(
                method="permutation",
                n_permutations=1,
                unit_of_inference="sample",
            ),
            random_state=6,
        ),
        n_jobs=1,
        verbose=False,
    )

    result = Experiment(config).run(X, y)

    assert not result.get_statistical_assessment().empty
    assert not result.get_best_params().empty


def test_temporal_statistical_assessment_accessor_plot_and_report():
    result = ExperimentResult(
        {
            "temporal": {
                "metrics": {},
                "predictions": [],
                "statistical_assessment": [
                    {
                        "Model": "temporal",
                        "Metric": "accuracy",
                        "Observed": 0.7,
                        "InferentialUnit": "sample",
                        "NEff": 10,
                        "NullMethod": "permutation_full_pipeline",
                        "NPermutations": 5,
                        "P0": None,
                        "PValue": 0.2,
                        "CILower": 0.4,
                        "CIUpper": 0.6,
                        "CorrectionMethod": "max_stat",
                        "CorrectedPValue": 0.3,
                        "ChanceThreshold": None,
                        "Time": 0,
                        "TrainTime": None,
                        "TestTime": None,
                        "NullLower": 0.35,
                        "NullUpper": 0.65,
                        "Significant": False,
                        "Assumptions": "full outer-CV pipeline",
                        "Caveat": "sample-level inference",
                    },
                    {
                        "Model": "temporal",
                        "Metric": "accuracy",
                        "Observed": 0.9,
                        "InferentialUnit": "sample",
                        "NEff": 10,
                        "NullMethod": "permutation_full_pipeline",
                        "NPermutations": 5,
                        "P0": None,
                        "PValue": 0.05,
                        "CILower": 0.4,
                        "CIUpper": 0.6,
                        "CorrectionMethod": "max_stat",
                        "CorrectedPValue": 0.05,
                        "ChanceThreshold": None,
                        "Time": 1,
                        "TrainTime": None,
                        "TestTime": None,
                        "NullLower": 0.35,
                        "NullUpper": 0.65,
                        "Significant": True,
                        "Assumptions": "full outer-CV pipeline",
                        "Caveat": "sample-level inference",
                    },
                ],
            }
        }
    )

    assessment = result.get_statistical_assessment()
    assert set(assessment["Time"]) == {0, 1}

    fig = plot_temporal_statistical_assessment(result)
    assert fig.axes

    report = Report("Stats")
    report.add_decoding_statistical_assessment(result)
    assert "Finite-Sample Statistical Assessment" in report.render()
