"""Lightweight public interfaces for decoding estimator families."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecoderEstimator(Protocol):
    """Sklearn-compatible estimator interface used by the outer CV engine."""

    def fit(self, X, y=None, **fit_params): ...

    def predict(self, X): ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params): ...


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Interface for pretrained or frozen embedding extractors."""

    def transform(self, X): ...

    def get_embedding_info(self) -> dict[str, Any]: ...


@runtime_checkable
class NeuralTrainable(Protocol):
    """Interface for trainable neural estimators with artifact metadata."""

    def get_training_history(self) -> list[dict[str, Any]]: ...

    def get_checkpoint_manifest(self) -> dict[str, Any]: ...

    def get_model_card_info(self) -> dict[str, Any]: ...

    def get_failure_diagnostics(self) -> dict[str, Any]: ...


@runtime_checkable
class StagedTrainable(Protocol):
    """Interface for staged training schedules."""

    def set_train_stage(self, stage: str): ...
