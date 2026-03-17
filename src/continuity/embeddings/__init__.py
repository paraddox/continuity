"""Embedding clients for Continuity retrieval pipelines."""

from .ollama import (
    OllamaEmbeddingBatch,
    OllamaEmbeddingClient,
    OllamaEmbeddingConfig,
    OllamaEmbeddingError,
)

__all__ = [
    "OllamaEmbeddingBatch",
    "OllamaEmbeddingClient",
    "OllamaEmbeddingConfig",
    "OllamaEmbeddingError",
]
