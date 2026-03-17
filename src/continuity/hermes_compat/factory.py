"""Backend factory for Hermes-compatible Continuity embedding."""

from __future__ import annotations

from continuity.embeddings.ollama import OllamaEmbeddingClient, OllamaEmbeddingConfig
from continuity.hermes_compat.config import (
    ContinuityVectorBackendKind,
    HermesMemoryBackendKind,
    HermesMemoryConfig,
)
from continuity.hermes_compat.manager import ContinuityHermesSessionManager
from continuity.index.zvec_index import InMemoryZvecBackend, ZvecBackend
from continuity.reasoning.base import ReasoningAdapter
from continuity.reasoning.codex_adapter import CodexAdapter, CodexAdapterConfig


def create_continuity_backend(
    config: HermesMemoryConfig | None = None,
    *,
    reasoning_adapter: ReasoningAdapter | None = None,
) -> tuple[ContinuityHermesSessionManager | None, HermesMemoryConfig]:
    resolved_config = config or HermesMemoryConfig.from_global_config()
    if resolved_config.backend is not HermesMemoryBackendKind.CONTINUITY:
        return None, resolved_config
    if not resolved_config.enabled:
        return None, resolved_config

    adapter = reasoning_adapter or CodexAdapter(
        config=CodexAdapterConfig(
            model=resolved_config.continuity_reasoning_model,
            reasoning_effort=resolved_config.continuity_reasoning_effort,  # type: ignore[arg-type]
        )
    )
    embedding_client = OllamaEmbeddingClient(
        OllamaEmbeddingConfig(
            model=resolved_config.continuity_embedding_model,
            base_url=resolved_config.continuity_embedding_base_url,
        )
    )
    if resolved_config.continuity_vector_backend is ContinuityVectorBackendKind.ZVEC:
        dimensions = resolved_config.continuity_embedding_dimensions
        if dimensions is None:
            raise ValueError(
                "continuity_embedding_dimensions must be configured when vectorBackend is zvec"
            )
        vector_backend = ZvecBackend(
            collection_path=str(resolved_config.continuity_collection_path),
            dimensions=dimensions,
        )
    else:
        vector_backend = InMemoryZvecBackend()

    manager = ContinuityHermesSessionManager(
        config=resolved_config,
        reasoning_adapter=adapter,
        embedding_client=embedding_client,
        vector_backend=vector_backend,
    )
    return manager, resolved_config
