# Decoding

The decoding module runs classification and regression experiments through
explicit train/test splits. The outer CV in `config.cv` is always the evaluation
split. When hyperparameter tuning is enabled, `config.tuning.cv` is the required
inner model-selection split.

## Metrics

Supported classification metrics:

- `accuracy`
- `balanced_accuracy`
- `roc_auc`
- `average_precision`
- `pr_auc`
- `log_loss`
- `brier_score`
- `f1`
- `f1_macro`
- `f1_micro`
- `precision`
- `recall`
- `sensitivity`
- `specificity`

Supported regression metrics:

- `r2`
- `neg_mean_squared_error`
- `neg_mean_absolute_error`
- `explained_variance`

Metric/task validation is registry-based. Classification-only metrics cannot be
used for regression tasks, and regression-only metrics cannot be used for
classification tasks. Probability metrics such as `log_loss` and `brier_score`
require `predict_proba`. Ranking metrics such as `roc_auc` and
`average_precision` use `predict_proba` when available and fall back to
`decision_function` for binary classifiers.

## Cross-Validation

Supported `CVConfig.strategy` values:

- `stratified`
- `kfold`
- `group_kfold`
- `stratified_group_kfold`
- `leave_p_out`
- `leave_one_group_out`
- `timeseries`
- `split`

Group strategies require `groups` when running the experiment:

```python
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig

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
    cv=CVConfig(strategy="group_kfold", n_splits=5),
)

result = Experiment(config).run(X, y, groups=subject_ids)
```

`leave_one_group_out` uses scikit-learn `LeaveOneGroupOut` and therefore
requires `groups`.

When `groups` are supplied, decoding binds that group array to the splitter so
the same groups are used whenever `.split(...)` is called. This binding does
not turn non-group strategies such as `kfold` into group-safe strategies; use a
group strategy when train/test group isolation is required.

## Tuning CV

Tuning does not reuse the outer CV implicitly. If `tuning.enabled=True`, provide
`tuning.cv` explicitly.

```python
from coco_pipe.decoding.configs import CVConfig, TuningConfig

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
    metrics=["accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5),
    tuning=TuningConfig(
        enabled=True,
        scoring="accuracy",
        cv=CVConfig(strategy="group_kfold", n_splits=3),
        n_jobs=1,
    ),
)

result = Experiment(config).run(X, y, groups=subject_ids)
```

For grouped tuning, the outer training-fold groups are passed into
`GridSearchCV` or `RandomizedSearchCV`. Plain estimators and plain pipelines are
fit without groups.

Raw grid keys are mapped to the final classifier step, so `{"C": [...]}` becomes
`{"clf__C": [...]}`. Explicit pipeline keys such as `fs__n_features_to_select`
are left unchanged. Invalid keys fail before model fitting with a clear error.

For random search, set `tuning.random_state` for reproducibility:

```python
tuning=TuningConfig(
    enabled=True,
    search_type="random",
    n_iter=20,
    scoring="accuracy",
    random_state=42,
    cv=CVConfig(strategy="stratified", n_splits=3),
)
```

Tuned folds store compact search diagnostics, including best params, best score,
best index, candidate rank, mean validation score, and validation-score
standard deviation. Use `result.get_best_params()` and
`result.get_search_results()` to inspect them.

## Feature Selection

`feature_selection.method="k_best"` is a filter step based on `SelectKBest`.
It has no CV loop. If `n_features=None`, all features are kept, which makes the
default safe for datasets with fewer than ten features.

```python
from coco_pipe.decoding.configs import FeatureSelectionConfig

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
    cv=CVConfig(strategy="stratified", n_splits=5),
    feature_selection=FeatureSelectionConfig(
        enabled=True,
        method="k_best",
        n_features=20,
    ),
)
```

`feature_selection.method="sfs"` uses scikit-learn
`SequentialFeatureSelector`. SFS has its own required CV config at
`feature_selection.cv`; it does not reuse the outer evaluation CV and it does
not reuse `tuning.cv`.

```python
config = ExperimentConfig(
    task="classification",
    models={
        "lr": {
            "method": "LogisticRegression",
            "solver": "liblinear",
            "max_iter": 200,
        }
    },
    metrics=["balanced_accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5),
    feature_selection=FeatureSelectionConfig(
        enabled=True,
        method="sfs",
        n_features=10,
        scoring="balanced_accuracy",
        cv=CVConfig(strategy="stratified", n_splits=3),
    ),
)

result = Experiment(config).run(X, y, groups=subject_ids)
selected = result.get_selected_features()
stability = result.get_feature_stability()
```

Decoding is array-first. Pass feature names explicitly when names matter:

```python
experiment = Experiment(config)
result = experiment.run(
    X,
    y,
    groups=subject_ids,
    sample_ids=recording_ids,
    feature_names=["alpha", "beta", "theta", "delta"],
)
```

When `feature_names` is omitted, decoding generates names such as `feature_0`.
The names must align with the feature dimension of `X`. When `sample_ids` is
omitted, decoding uses row-position IDs.

For `k_best`, fitted fold metadata includes univariate feature scores and
p-values. Use `result.get_feature_scores()` to retrieve them in long form. SFS
does not expose stable per-feature scores in scikit-learn, so SFS folds do not
appear in `get_feature_scores()`.

SFS scoring is resolved in this order:

- `feature_selection.scoring`
- `tuning.scoring`
- the first entry in `metrics`

Group-aware SFS CV uses scikit-learn metadata routing. When
`feature_selection.cv` is `group_kfold`, `stratified_group_kfold`,
`leave_p_out`, or `leave_one_group_out`, decoding enables metadata routing
around the fit call and passes the outer training-fold groups into SFS. This
requires the package dependency `scikit-learn>1.6`.

SFS can use `feature_selection.cv=CVConfig(strategy="split", stratify=True)`.
The holdout splitter receives the fold-local `y` from SFS and uses it for
stratification.

SFS combined with hyperparameter tuning can be expensive because feature
subsets are evaluated inside tuning folds. The current implementation uses a
temporary sklearn pipeline cache for this combination.

## CV Loop Combinations

The decoding runner treats each CV layer as a separate decision:

- baseline: `config.cv`
- SFS only: `config.cv` plus `feature_selection.cv`
- tuning only: `config.cv` plus `tuning.cv`
- `k_best` plus tuning: `config.cv` plus `tuning.cv`
- SFS plus tuning: `config.cv` plus `tuning.cv` plus `feature_selection.cv`

## Result Schema

`Experiment.run(...)` returns an `ExperimentResult` with the current decoding
payload in memory:

```python
result = Experiment(config).run(
    X,
    y,
    groups=subject_ids,
    sample_ids=recording_ids,
    feature_names=feature_names,
)

payload = result.to_payload()
```

The payload contains:

- `schema_version`: currently `decoding_result_v1`
- `config`: the original experiment config
- `meta`: environment provenance plus tag, task, sample count, and feature count
- `results`: per-model folds, metrics, predictions, splits, importances, and
  metadata

Save/load uses that same payload shape:

```python
path = experiment.save_results()
loaded = Experiment.load_results(path)
```

Use the result accessors for tidy tables:

```python
predictions = result.get_predictions()
scores = result.get_detailed_scores()
splits = result.get_splits()
importances = result.get_feature_importances()
fold_importances = result.get_feature_importances(fold_level=True)
```

`get_predictions()` includes `SampleIndex`, `SampleID`, and `Group`.
Temporal predictions are expanded into long form with `Time` for sliding
outputs or `TrainTime` / `TestTime` for generalization outputs.

`get_detailed_scores()` also expands temporal metric arrays into long form.
Feature importances include `FeatureName` using explicit `feature_names` when
provided, otherwise generated feature names.

## Holdout Split

Use `strategy="split"` for a single train/test split. Configure the test size
with `test_size`. Classification holdout can stratify with `stratify=True`.

```python
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
    cv=CVConfig(
        strategy="split",
        n_splits=2,
        test_size=0.25,
        stratify=True,
        random_state=42,
    ),
)
```

`n_splits` remains part of `CVConfig` for schema consistency, but `split` always
produces one train/test split.

## Time Series Split

Use `strategy="timeseries"` for ordered train/test splits:

```python
config = ExperimentConfig(
    task="regression",
    models={"ridge": {"method": "Ridge"}},
    metrics=["r2"],
    cv=CVConfig(strategy="timeseries", n_splits=5),
)
```

The implementation delegates split feasibility to scikit-learn. Choose valid
split counts, group labels, and class distributions for your dataset.

## Temporal Decoding

Temporal decoding uses MNE meta-estimators for 3D arrays with layout
`(n_samples, n_features_or_channels, n_times)`.

```python
from coco_pipe.decoding.configs import (
    GeneralizingEstimatorConfig,
    LogisticRegressionConfig,
    SlidingEstimatorConfig,
)

sliding_config = ExperimentConfig(
    task="classification",
    models={
        "sliding": SlidingEstimatorConfig(
            base_estimator=LogisticRegressionConfig(max_iter=200),
            scoring="accuracy",
            n_jobs=1,
        )
    },
    metrics=["accuracy"],
    cv=CVConfig(strategy="stratified", n_splits=5),
)

result = Experiment(sliding_config).run(
    X_temporal,
    y,
    time_axis=epoch_times,
)
```

`time_axis` is optional. When supplied for 3D inputs, it must align with
`X.shape[-1]`. When omitted, decoding uses integer time positions. Temporal
score and prediction accessors preserve those labels:

```python
scores = result.get_detailed_scores()
temporal = result.get_temporal_score_summary()
predictions = result.get_predictions()
```

`SlidingEstimator` produces 1D temporal score curves. `GeneralizingEstimator`
produces train-time by test-time matrices:

```python
generalizing_config = ExperimentConfig(
    task="classification",
    models={
        "generalizing": GeneralizingEstimatorConfig(
            base_estimator=LogisticRegressionConfig(max_iter=200),
            scoring="accuracy",
            n_jobs=1,
        )
    },
    metrics=["accuracy"],
    cv=CVConfig(strategy="stratified", n_splits=5),
)

generalizing_result = Experiment(generalizing_config).run(
    X_temporal,
    y,
    time_axis=epoch_times,
)
matrix = generalizing_result.get_generalization_matrix("accuracy")
```

Temporal plotting helpers are available from `coco_pipe.viz`:

```python
from coco_pipe.viz import (
    plot_temporal_generalization_matrix,
    plot_temporal_score_curve,
)

fig_curve = plot_temporal_score_curve(result, metric="accuracy")
fig_matrix = plot_temporal_generalization_matrix(
    generalizing_result,
    metric="accuracy",
)
```

Reports can include a compact temporal section:

```python
report.add_decoding_temporal(result, metric="accuracy")
```
