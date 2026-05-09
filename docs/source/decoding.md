# Decoding

The decoding module runs classification and regression experiments through
explicit train/test splits. The outer CV in `config.cv` is always the evaluation
split. Learned preprocessing and model-selection steps are built inside the
fold-specific training path, including scaling, univariate feature selection,
SFS, calibration, and hyperparameter search.

Decoding does not currently expose a dimensionality-reduction transformer. When
one is added to this module, it should be inserted as a fold-local pipeline step
under the same rule.

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

Metrics are organized into capability-aware families:

- `label`: hard-label metrics such as `accuracy`
- `confusion`: confusion-derived metrics such as F1, precision, recall,
  sensitivity, specificity, and balanced accuracy
- `threshold_sweep`: ranking or threshold-sweep summaries such as `roc_auc`,
  `average_precision`, and `pr_auc`
- `score_probability`: probability-score metrics such as `log_loss`
- `calibration`: calibration-oriented metrics such as `brier_score`
- `regression`: regression metrics such as R2 and error metrics

Metric/task validation is registry-based. Classification-only metrics cannot be
used for regression tasks, and regression-only metrics cannot be used for
classification tasks. Probability metrics such as `log_loss` and `brier_score`
require `predict_proba`. Ranking metrics such as `roc_auc` and
`average_precision` use `predict_proba` when available and fall back to
`decision_function` for binary classifiers.

## Capability Contracts

Decoding uses a typed `EstimatorSpec` registry plus lightweight capability
metadata for estimators, metrics, and feature selectors. Estimator specs are the
single source of truth for constructor lookup, estimator family, task support,
input kind, prediction interface, temporal support, grouped metadata support,
feature-selection compatibility, calibration eligibility, dependency extras,
fit-smoke policy, default search spaces, and importance/interpretability
support.

The contract layer is intentionally small. It blocks clear unsupported
combinations before nested CV starts, for example:

- probability metrics such as `log_loss` with a model that does not declare
  `predict_proba`
- ranking metrics such as `roc_auc` with a model that declares neither
  `predict_proba` nor `decision_function`
- 2D feature selectors on 3D temporal inputs
- temporal wrappers used with non-temporal input arrays
- classifier configs used for regression, or regressor configs used for
  classification

It does not try to validate every sklearn parameter combination, class-balance
edge case, split feasibility issue, or scientific design choice. Those remain
the responsibility of sklearn and the user.

```python
from coco_pipe.decoding import (
    EstimatorSpec,
    get_capabilities,
    get_estimator_spec,
    list_capabilities,
    list_estimator_specs,
)

logreg_spec = get_estimator_spec("LogisticRegression")
logreg_caps = get_capabilities("LogisticRegression")
all_specs = list_estimator_specs()
all_caps = list_capabilities()
```

Capability metadata is also stored in `ExperimentResult.meta["capabilities"]`
for provenance and reporting, including both per-model capability metadata and
the resolved `EstimatorSpec` for each configured model. Search defaults can be
read from `EstimatorSpec.default_search_space`; explicit `TuningConfig` grids
remain the source of truth for actual model-selection runs.

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

Group strategies require `cv.group_key` and sample metadata. `groups=` is still
accepted as a compatibility alias that populates `sample_metadata[group_key]`.

```python
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig

config = ExperimentConfig(
    task="classification",
    models={
        "lr": ClassicalModelConfig(
            estimator="logistic_regression",
            params={"solver": "liblinear", "max_iter": 200},
        )
    },
    metrics=["accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5, group_key="subject"),
)

result = Experiment(config).run(
    X,
    y,
    sample_metadata={"subject": subject_ids, "session": session_ids},
)
```

`leave_one_group_out` uses scikit-learn `LeaveOneGroupOut` and therefore
requires `groups`.

When `groups` are supplied, decoding binds that group array to the splitter so
the same groups are used whenever `.split(...)` is called. This binding does
not turn non-group strategies such as `kfold` into group-safe strategies; use a
group strategy when train/test group isolation is required.

## Inner CV

When `tuning.enabled=True`, `tuning.cv` controls the inner model-selection
split. If omitted, decoding derives it from the outer CV family. When the outer
CV is group-based, the derived inner tuning CV is also group-based.

If the outer CV is group-based and you explicitly choose a non-grouped
`tuning.cv`, set `allow_nongroup_inner_cv=True` on `TuningConfig` to acknowledge
the leakage/generalization trade-off.

```python
from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig, TuningConfig

config = ExperimentConfig(
    task="classification",
    models={
        "lr": ClassicalModelConfig(
            estimator="logistic_regression",
            params={"solver": "liblinear", "max_iter": 200},
        )
    },
    grids={"lr": {"C": [0.1, 1.0, 10.0]}},
    metrics=["accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5, group_key="subject"),
    tuning=TuningConfig(
        enabled=True,
        scoring="accuracy",
        n_jobs=1,
    ),
)

result = Experiment(config).run(
    X,
    y,
    sample_metadata={"subject": subject_ids, "session": session_ids},
)
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
`SequentialFeatureSelector`. SFS is itself a CV-driven model-selection
procedure. If `feature_selection.cv` is omitted, SFS inherits `tuning.cv` when
tuning is enabled, otherwise it derives from the outer CV family. If the outer
CV is group-based, the default SFS CV is also group-based.

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
    sample_metadata={
        "subject": subject_ids,
        "session": session_ids,
        "site": site_ids,
    },
    feature_names=["alpha", "beta", "theta", "delta"],
)
```

When `feature_names` is omitted, decoding generates names such as `feature_0`.
The names must align with the feature dimension of `X`. When `sample_ids` is
omitted, decoding uses row-position IDs.

`sample_ids` must be unique at the independent-observation level. For EEG/MEEG
epoch decoding, pass one ID per epoch; for subject-level tables, pass one ID per
subject row.

For `k_best`, fitted fold metadata includes univariate feature scores and
p-values. Use `result.get_feature_scores()` to retrieve them in long form. SFS
does not expose stable per-feature scores in scikit-learn, so SFS folds do not
appear in `get_feature_scores()`.

SFS scoring is resolved in this order:

- `feature_selection.scoring`
- `tuning.scoring`
- the first entry in `metrics`

Group-aware SFS CV uses scikit-learn metadata routing. When the resolved
`feature_selection.cv` is `group_kfold`, `stratified_group_kfold`,
`leave_p_out`, or `leave_one_group_out`, decoding enables metadata routing
around the fit call and passes the outer training-fold groups into SFS. This
requires the package dependency `scikit-learn>1.6`.

If the outer CV is group-based and you explicitly choose a non-grouped
`feature_selection.cv`, set
`FeatureSelectionConfig(allow_nongroup_inner_cv=True)` to acknowledge the
trade-off.

SFS can use `feature_selection.cv=CVConfig(strategy="split", stratify=True)`.
The holdout splitter receives the fold-local `y` from SFS and uses it for
stratification.

SFS combined with hyperparameter tuning can be expensive because feature
subsets are evaluated inside tuning folds. The current implementation uses a
temporary sklearn pipeline cache for this combination.

## CV Loop Combinations

The decoding runner treats each CV layer as a separate decision:

- baseline: `config.cv`
- SFS only: `config.cv` plus resolved `feature_selection.cv`
- tuning only: `config.cv` plus resolved `tuning.cv`
- `k_best` plus tuning: `config.cv` plus resolved `tuning.cv`
- SFS plus tuning: `config.cv`, resolved `tuning.cv`, and resolved
  `feature_selection.cv`

## Result Schema

`Experiment.run(...)` returns an `ExperimentResult` with the current decoding
payload in memory:

```python
result = Experiment(config).run(
    X,
    y,
    groups=subject_ids,
    sample_ids=recording_ids,
    sample_metadata={
        "subject": subject_ids,
        "session": session_ids,
        "site": site_ids,
    },
    observation_level="epoch",
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
fit_diagnostics = result.get_fit_diagnostics()
confusion = result.get_confusion_matrices()
pooled_confusion = result.get_pooled_confusion_matrix()
roc_curve = result.get_roc_curve()
pr_curve = result.get_pr_curve()
calibration = result.get_calibration_curve()
probability_scores = result.get_probability_diagnostics()
null = result.get_statistical_assessment(lightweight=True, metric="accuracy")
ci = result.get_bootstrap_confidence_intervals(metric="accuracy", unit="group")
paired = result.compare_models_paired("model_a", "model_b", metric="accuracy")
stats = result.get_statistical_assessment()
importances = result.get_feature_importances()
fold_importances = result.get_feature_importances(fold_level=True)
```

`get_predictions()` includes `SampleIndex`, `SampleID`, and `Group`.
When `sample_metadata` is supplied, predictions and splits also include
`Subject`, `Session`, `Site`, and any additional metadata columns. The metadata
input must include `subject` and `session`; `site` is optional and is added as
an empty column when omitted.
Temporal predictions are expanded into long form with `Time` for sliding
outputs or `TrainTime` / `TestTime` for generalization outputs.

`get_detailed_scores()` also expands temporal metric arrays into long form.
Feature importances include `FeatureName` using explicit `feature_names` when
provided, otherwise generated feature names.

For epoch-level decoding, use `observation_level="epoch"`. When sample metadata
is available, result metadata defaults `inferential_unit` to `subject`, so
bootstrap confidence intervals and paired model comparisons use subjects by
default. Pass `inferential_unit="epoch"` to opt into epoch-level inference.

```python
ci = result.get_bootstrap_confidence_intervals(metric="accuracy")
paired = result.compare_models_paired("model_a", "model_b", metric="accuracy")
```

Future embedding and feature-extraction caches should include split identity
and upstream fingerprints. The decoding module exposes a small cache-key helper
for that contract:

```python
from coco_pipe.decoding import make_feature_cache_key

cache_key = make_feature_cache_key(
    train_sample_ids=train_ids,
    test_sample_ids=test_ids,
    preprocessing_fingerprint=preprocessing_hash,
    backbone_fingerprint=backbone_hash,
)
```

## Statistical Assessment

Finite-sample statistical assessment is opt-in and separate from descriptive
CV performance. Descriptive metrics such as accuracy, balanced accuracy, AUROC,
and temporal curves are always available from the standard result accessors.
Inferential claims require `StatisticalAssessmentConfig`.

```python
from coco_pipe.decoding.configs import (
    ChanceAssessmentConfig,
    ClassicalModelConfig,
    StatisticalAssessmentConfig,
)

config = ExperimentConfig(
    task="classification",
    models={
        "lr": ClassicalModelConfig(
            estimator="logistic_regression",
            params={"max_iter": 200},
        )
    },
    metrics=["accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5, group_key="subject"),
    evaluation=StatisticalAssessmentConfig(
        enabled=True,
        primary_metric="accuracy",
        chance=ChanceAssessmentConfig(
            method="permutation",
            n_permutations=1000,
            unit_of_inference="group_mean",
        ),
    ),
)

result = Experiment(config).run(
    X,
    y,
    sample_ids=epoch_ids,
    sample_metadata={
        "subject": subject_ids,
        "session": session_ids,
    },
    observation_level="epoch",
)
assessment = result.get_statistical_assessment()
```

When `evaluation.chance.unit_of_inference` is omitted, decoding uses
`group_mean` whenever grouped metadata are supplied and `sample` otherwise. For
EEG/MEEG epoch decoding this means epoch-level predictions can remain
descriptive while inferential metrics default to subject/group-level
aggregation. `group_mean` aggregates
probabilities before classification testing; `group_majority` aggregates hard
labels. `unit_of_inference="custom"` uses a named `sample_metadata` column.

The default method is full-pipeline permutation testing. Each permutation
reruns outer CV and all fold-local steps, including scaling, feature selection,
tuning, calibration, and learned preprocessing. This is slower than
fixed-prediction diagnostics, but it estimates the null for the full decoding
workflow.

Analytical binomial testing is intentionally narrow:

- task must be classification
- metric must be plain `accuracy`
- predictions must be non-temporal scalar rows
- each independent unit must contribute exactly one held-out prediction
- `p0` must be explicit

```python
evaluation=StatisticalAssessmentConfig(
    enabled=True,
    chance=ChanceAssessmentConfig(
        method="binomial",
        p0=0.5,
        ci_method="wilson",
    ),
)
```

Temporal statistical assessment stores one row per timepoint or
train/test-time coordinate. `temporal_correction="max_stat"` is the default
family-wise correction; `fdr_bh` is available for exploratory use. Cluster-based
temporal inference is not implemented yet.

Calling `result.get_statistical_assessment(lightweight=True)` provides a
lightweight diagnostic over fixed out-of-fold predictions. It does not refit
preprocessing, SFS, tuning, or calibration under the null, so it should not be
treated as the primary finite-sample inference path.

## Foundation-Model Workflows

Foundation-model workflows still enter through `Experiment.run(...)`. The
outer CV engine sees estimators; the configs decide whether fit means
embedding extraction, frozen-backbone decoding, full fine-tuning, LoRA, or
QLoRA.

```python
from coco_pipe.decoding.configs import (
    CheckpointConfig,
    DeviceConfig,
    FoundationEmbeddingModelConfig,
    FrozenBackboneDecoderConfig,
    LoRAConfig,
    NeuralFineTuneConfig,
    QuantizationConfig,
)

config = ExperimentConfig(
    task="classification",
    models={
        "labram_probe": FrozenBackboneDecoderConfig(
            backbone=FoundationEmbeddingModelConfig(
                provider="braindecode",
                model_name="labram-pretrained",
                input_kind="epoched",
                pooling="mean",
                cache_embeddings=True,
            ),
            head=ClassicalModelConfig(
                estimator="logistic_regression",
                params={"max_iter": 1000},
            ),
        )
    },
    metrics=["balanced_accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5, group_key="subject"),
)
```

Trainable neural estimators use one config family with `train_mode`:

```python
config = ExperimentConfig(
    task="classification",
    models={
        "reve_qlora": NeuralFineTuneConfig(
            provider="huggingface",
            model_name="brain-bzh/reve-base",
            input_kind="epoched",
            train_mode="qlora",
            lora=LoRAConfig(r=16, alpha=32, dropout=0.05),
            quantization=QuantizationConfig(enabled=True),
            device=DeviceConfig(device="auto", precision="bf16"),
            checkpoints=CheckpointConfig(save="best"),
        )
    },
    metrics=["balanced_accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5, group_key="subject"),
)
```

Neural and embedding runs expose artifact metadata through
`result.get_model_artifacts()`. First-wave QLoRA is restricted to Hugging Face
backbones with the `hf`, `peft`, and `quant` optional extras installed.

## Diagnostics

Shallow decoding diagnostics are exported from the same result schema as
standard scores and predictions:

- `get_fit_diagnostics()` returns fold fit/predict/score times and captured
  warnings such as convergence warnings
- `get_confusion_matrices()` returns fold-level confusion matrices in long form
- `get_pooled_confusion_matrix()` returns pooled out-of-fold confusion counts
- `get_roc_curve()` returns binary or one-vs-rest ROC curve coordinates when
  probability or decision scores are available
- `get_pr_curve()` returns binary or one-vs-rest precision-recall coordinates
- `get_calibration_curve()` returns binary or one-vs-rest reliability curve
  coordinates from probabilities
- `get_probability_diagnostics()` returns fold-level log-loss and Brier
  summaries when probabilities exist
- `get_statistical_assessment(lightweight=True)` returns lightweight
  label-permutation null summaries over fixed out-of-fold predictions
- `get_bootstrap_confidence_intervals()` returns bootstrap CIs over the result's
  default inferential unit, or over `sample`, `epoch`, `group`, `subject`,
  `session`, or `site` when `unit` is set explicitly
- `compare_models_paired()` compares two models on shared outer-fold
  predictions with a paired sign-swap permutation helper and the same
  inference-unit options

Diagnostic plots are available from `coco_pipe.viz`:

```python
from coco_pipe.viz import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_fold_score_dispersion,
    plot_pr_curve,
    plot_roc_curve,
)

fig_confusion = plot_confusion_matrix(result)
fig_roc = plot_roc_curve(result)
fig_pr = plot_pr_curve(result)
fig_calibration = plot_calibration_curve(result)
fig_scores = plot_fold_score_dispersion(result)
```

Reports can include a compact diagnostics section:

```python
report.add_decoding_diagnostics(result, metric="accuracy")
report.add_decoding_statistical_assessment(result, metric="accuracy")
```

Probability calibration is opt-in and happens inside the training path through
`sklearn.calibration.CalibratedClassifierCV`. Its resolved `calibration.cv`
defines disjoint inner calibration folds inside each outer-training fold. If
omitted, calibration CV derives from the outer CV family. Non-grouped
calibration CV under grouped outer CV requires
`CalibrationConfig(allow_nongroup_inner_cv=True)`.

```python
from coco_pipe.decoding.configs import CalibrationConfig

config = ExperimentConfig(
    task="classification",
    models={"svm": {"method": "LinearSVC"}},
    metrics=["log_loss"],
    calibration=CalibrationConfig(
        enabled=True,
        method="sigmoid",
    ),
)
```

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
    ClassicalModelConfig,
    TemporalDecoderConfig,
)

sliding_config = ExperimentConfig(
    task="classification",
    models={
        "sliding": TemporalDecoderConfig(
            wrapper="sliding",
            base=ClassicalModelConfig(
                estimator="logistic_regression",
                params={"max_iter": 200},
            ),
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
        "generalizing": TemporalDecoderConfig(
            wrapper="generalizing",
            base=ClassicalModelConfig(
                estimator="logistic_regression",
                params={"max_iter": 200},
            ),
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
