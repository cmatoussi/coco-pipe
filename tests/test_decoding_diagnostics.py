import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    CalibrationConfig,
    CVConfig,
    LinearSVCConfig,
    LogisticRegressionConfig,
)
from coco_pipe.decoding.core import ExperimentResult
from coco_pipe.report.core import Report
from coco_pipe.viz.decoding import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_fold_score_dispersion,
    plot_pr_curve,
    plot_roc_curve,
)


def _diagnostic_result():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=200)},
        metrics=["accuracy", "roc_auc", "brier_score"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=7),
        n_jobs=1,
        verbose=False,
    )
    return Experiment(config).run(X, y)


def test_fit_diagnostics_are_recorded_per_fold():
    result = _diagnostic_result()

    diagnostics = result.get_fit_diagnostics()

    assert len(diagnostics) == 2
    assert {"FitTime", "PredictTime", "ScoreTime", "TotalTime"}.issubset(
        diagnostics.columns
    )
    assert (diagnostics["FitTime"] >= 0).all()


def test_fit_diagnostics_expands_warning_records():
    result = ExperimentResult(
        {
            "model": {
                "metrics": {},
                "predictions": [],
                "diagnostics": [
                    {
                        "fit_time": 0.1,
                        "predict_time": 0.2,
                        "score_time": 0.3,
                        "total_time": 0.6,
                        "warnings": [
                            {
                                "stage": "fit",
                                "category": "ConvergenceWarning",
                                "message": "did not converge",
                            }
                        ],
                    }
                ],
            }
        }
    )

    diagnostics = result.get_fit_diagnostics()

    assert diagnostics.loc[0, "Stage"] == "fit"
    assert diagnostics.loc[0, "WarningCategory"] == "ConvergenceWarning"
    assert "did not converge" in diagnostics.loc[0, "WarningMessage"]


def test_confusion_roc_pr_and_calibration_accessors():
    result = _diagnostic_result()

    confusion = result.get_confusion_matrices()
    counts = result.get_confusion_counts()
    pooled = result.get_pooled_confusion_matrix()
    roc = result.get_roc_curve()
    pr = result.get_pr_curve()
    calibration = result.get_calibration_curve(n_bins=3)
    proba = result.get_probability_diagnostics()

    assert {"TrueLabel", "PredictedLabel", "Value"}.issubset(confusion.columns)
    assert {"TrueLabel", "PredictedLabel", "Value"}.issubset(counts.columns)
    assert {"TrueLabel", "PredictedLabel", "Value"}.issubset(pooled.columns)
    assert {"FPR", "TPR", "Threshold"}.issubset(roc.columns)
    assert {"Precision", "Recall", "Threshold"}.issubset(pr.columns)
    assert {
        "MeanPredictedProbability",
        "FractionPositive",
    }.issubset(calibration.columns)
    assert {"Metric", "Class", "Value"}.issubset(proba.columns)
    assert not confusion.empty
    assert not counts.empty
    assert not pooled.empty
    assert not roc.empty
    assert not pr.empty
    assert not calibration.empty
    assert {"log_loss", "brier_score_macro"}.issubset(set(proba["Metric"]))


def test_multiclass_curves_use_one_vs_rest_rows():
    raw = {
        "m": {
            "metrics": {},
            "predictions": [
                {
                    "sample_index": np.arange(6),
                    "sample_id": np.arange(6),
                    "group": None,
                    "y_true": np.array([0, 1, 2, 0, 1, 2]),
                    "y_pred": np.array([0, 1, 2, 1, 1, 0]),
                    "y_proba": np.array(
                        [
                            [0.8, 0.1, 0.1],
                            [0.1, 0.7, 0.2],
                            [0.1, 0.2, 0.7],
                            [0.3, 0.5, 0.2],
                            [0.2, 0.6, 0.2],
                            [0.4, 0.2, 0.4],
                        ]
                    ),
                }
            ],
        }
    }

    result = ExperimentResult(raw)

    assert set(result.get_roc_curve()["Class"]) == {0, 1, 2}
    assert set(result.get_pr_curve()["Class"]) == {0, 1, 2}
    assert set(result.get_calibration_curve(n_bins=2)["Class"]) == {0, 1, 2}
    assert not result.get_probability_diagnostics().empty


def test_permutation_bootstrap_and_paired_comparison_helpers():
    X, y = make_classification(
        n_samples=36,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=9,
    )
    groups = np.repeat(np.arange(12), 3)
    config = ExperimentConfig(
        task="classification",
        models={
            "lr": LogisticRegressionConfig(max_iter=200),
            "dummy": {"method": "DummyClassifier", "strategy": "prior"},
        },
        metrics=["accuracy"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=9),
        n_jobs=1,
        verbose=False,
    )
    result = Experiment(config).run(X, y, groups=groups)

    null = result.get_statistical_assessment(
        lightweight=True, n_permutations=20, random_state=1
    )
    ci = result.get_bootstrap_confidence_intervals(
        n_bootstraps=20,
        unit="group",
        random_state=1,
    )
    paired = result.compare_models_paired(
        "lr",
        "dummy",
        n_permutations=20,
        unit="group",
        random_state=1,
    )

    assert {"Observed", "NullLower", "NullUpper", "PValue"}.issubset(null.columns)
    assert {"Estimate", "CILower", "CIUpper", "NUnits"}.issubset(ci.columns)
    assert paired.loc[0, "ModelA"] == "lr"
    assert paired.loc[0, "ModelB"] == "dummy"
    assert 0 <= paired.loc[0, "PValue"] <= 1


def test_calibration_wraps_training_path_with_disjoint_inner_cv():
    config = ExperimentConfig(
        task="classification",
        models={"svm": LinearSVCConfig(max_iter=500)},
        metrics=["log_loss"],
        cv=CVConfig(strategy="stratified", n_splits=2),
        calibration=CalibrationConfig(
            enabled=True,
            method="sigmoid",
            cv=CVConfig(strategy="stratified", n_splits=2),
        ),
        n_jobs=1,
        verbose=False,
    )
    estimator = Experiment(config)._prepare_estimator("svm", config.models["svm"])

    assert estimator.__class__.__name__ == "CalibratedClassifierCV"
    assert estimator.method == "sigmoid"
    assert estimator.cv.__class__.__name__ == "StratifiedKFold"


def test_calibration_defaults_to_outer_group_cv_family():
    config = ExperimentConfig(
        task="classification",
        models={"svm": LinearSVCConfig(max_iter=500)},
        metrics=["log_loss"],
        cv=CVConfig(strategy="group_kfold", n_splits=2),
        calibration=CalibrationConfig(enabled=True, method="sigmoid"),
        n_jobs=1,
        verbose=False,
    )

    estimator = Experiment(config)._prepare_estimator("svm", config.models["svm"])

    assert estimator.__class__.__name__ == "CalibratedClassifierCV"
    assert estimator.cv.__class__.__name__ == "GroupKFold"


def test_nongroup_calibration_cv_under_grouped_outer_requires_override():
    with pytest.raises(ValueError, match="allow_nongroup_inner_cv"):
        Experiment(
            ExperimentConfig(
                task="classification",
                models={"svm": LinearSVCConfig(max_iter=500)},
                metrics=["log_loss"],
                cv=CVConfig(strategy="group_kfold", n_splits=2),
                calibration=CalibrationConfig(
                    enabled=True,
                    method="sigmoid",
                    cv=CVConfig(strategy="stratified", n_splits=2),
                ),
                n_jobs=1,
                verbose=False,
            )
        )


def test_diagnostic_plots_return_matplotlib_figures():
    result = _diagnostic_result()

    figures = [
        plot_confusion_matrix(result),
        plot_roc_curve(result),
        plot_pr_curve(result),
        plot_calibration_curve(result),
        plot_fold_score_dispersion(result),
    ]

    for fig in figures:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_decoding_diagnostics_report_section_renders():
    result = _diagnostic_result()
    report = Report("Diagnostics")

    report.add_decoding_diagnostics(result)
    html = report.render()

    assert "Decoding Diagnostics" in html
    assert "Inference Context" in html
    assert "Fold Scores" in html
    assert "Fit Diagnostics" in html
