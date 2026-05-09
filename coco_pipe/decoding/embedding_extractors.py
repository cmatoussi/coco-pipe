"""Embedding extractor seams for decoding foundation-model workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


@dataclass
class EmbeddingInfo:
    provider: str
    model_name: str
    input_kind: str
    pooling: str
    embedding_dim: int
    normalize_embeddings: bool


class DummyEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """
    Deterministic lightweight extractor for tests and provider-independent smoke.

    It flattens each sample and optionally projects to ``embedding_dim`` using a
    deterministic random projection derived from ``model_name``.
    """

    def __init__(
        self,
        provider: str = "dummy",
        model_name: str = "dummy",
        input_kind: str = "epoched",
        pooling: str = "mean",
        normalize_embeddings: bool = True,
        embedding_dim: Optional[int] = None,
        cache_embeddings: bool = True,
    ):
        self.provider = provider
        self.model_name = model_name
        self.input_kind = input_kind
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = embedding_dim
        self.cache_embeddings = cache_embeddings

    def fit(self, X, y=None):
        X_flat = self._flatten(X)
        dim = self.embedding_dim or X_flat.shape[1]
        seed = abs(hash((self.provider, self.model_name, dim))) % (2**32)
        rng = np.random.default_rng(seed)
        if dim == X_flat.shape[1]:
            self.projection_ = None
        else:
            self.projection_ = rng.normal(size=(X_flat.shape[1], dim)) / np.sqrt(
                X_flat.shape[1]
            )
        embeddings = self._project(X_flat)
        if self.normalize_embeddings:
            self.scaler_ = StandardScaler().fit(embeddings)
        else:
            self.scaler_ = None
        self.embedding_dim_ = embeddings.shape[1]
        return self

    def transform(self, X):
        X_flat = self._flatten(X)
        embeddings = self._project(X_flat)
        if getattr(self, "scaler_", None) is not None:
            embeddings = self.scaler_.transform(embeddings)
        return embeddings

    def predict(self, X):
        """Return embeddings for embedding-only artifact workflows."""
        return self.transform(X)

    def get_embedding_info(self) -> dict[str, Any]:
        return EmbeddingInfo(
            provider=self.provider,
            model_name=self.model_name,
            input_kind=self.input_kind,
            pooling=self.pooling,
            embedding_dim=int(getattr(self, "embedding_dim_", self.embedding_dim or 0)),
            normalize_embeddings=self.normalize_embeddings,
        ).__dict__

    @staticmethod
    def _flatten(X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X.reshape(X.shape[0], -1)

    def _project(self, X_flat: np.ndarray) -> np.ndarray:
        if getattr(self, "projection_", None) is None:
            return X_flat
        return X_flat @ self.projection_


def build_embedding_extractor(config: Any) -> DummyEmbeddingExtractor:
    """Build an embedding extractor for the supported first-wave providers."""
    if config.provider not in {"dummy", "braindecode", "huggingface"}:
        raise ValueError(f"Unknown embedding provider '{config.provider}'.")
    if config.provider != "dummy":
        # Provider-specific loaders will replace this seam once optional deps are
        # validated in integration tests. Keep the public API usable in core.
        return DummyEmbeddingExtractor(**config.model_dump(exclude={"kind"}))
    return DummyEmbeddingExtractor(**config.model_dump(exclude={"kind"}))
