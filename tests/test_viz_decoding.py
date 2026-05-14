import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig
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
        models={"lr": LogisticRegressionConfig(max_iter=200, kind="classical")},
        metrics=["accuracy", "roc_auc", "brier_score"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=7),
        n_jobs=1,
        verbose=False,
    )
    return Experiment(config).run(X, y)


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
