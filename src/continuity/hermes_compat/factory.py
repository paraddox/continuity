"""Backend factory for Hermes-compatible Continuity embedding."""

from __future__ import annotations

import importlib

from continuity.embeddings.ollama import OllamaEmbeddingClient, OllamaEmbeddingConfig
from continuity.hermes_compat.config import (
    ContinuityVectorBackendKind,
    HermesMemoryBackendKind,
    HermesMemoryConfig,
)
from continuity.hermes_compat.manager import ContinuityHermesSessionManager
from continuity.index.zvec_index import InMemoryZvecBackend, ZvecBackend
from continuity.reasoning.base import ReasoningAdapter
from continuity.reasoning.codex_adapter import CodexAdapter, CodexAdapterConfig, ResponsesClient


class _HermesCodexResponsesClient:
    """Bridge Hermes's streaming-only Codex Responses backend to a create()->response contract."""

    def __init__(self, responses: object) -> None:
        self._responses = responses

    def create(self, **kwargs: object) -> object:
        stream_or_response = self._responses.create(stream=True, **kwargs)
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if event_type is None and isinstance(event, dict):
                    event_type = event.get("type")
                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue
                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Hermes Codex Responses stream did not emit a terminal response")


def _normalized_codex_model(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text or None


def _try_hermes_codex_runtime() -> tuple[ResponsesClient | None, str | None]:
    try:
        auth_module = importlib.import_module("hermes_cli.auth")
        config_module = importlib.import_module("hermes_cli.config")
        openai_module = importlib.import_module("openai")
    except Exception:
        return None, None

    try:
        creds = auth_module.resolve_codex_runtime_credentials()
    except Exception:
        return None, None

    api_key = str(creds.get("api_key", "") or "").strip()
    base_url = str(creds.get("base_url", "") or "").strip()
    if not api_key or not base_url:
        return None, None

    client = _HermesCodexResponsesClient(
        openai_module.OpenAI(api_key=api_key, base_url=base_url).responses
    )
    model = None
    try:
        config = config_module.load_config()
        if isinstance(config, dict):
            model_cfg = config.get("model")
            if isinstance(model_cfg, dict):
                provider = str(model_cfg.get("provider", "") or "").strip().lower()
                if provider == "openai-codex":
                    model = _normalized_codex_model(model_cfg.get("default") or model_cfg.get("name"))
    except Exception:
        model = None
    return client, model


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

    codex_client, hermes_model = _try_hermes_codex_runtime()
    adapter = reasoning_adapter or CodexAdapter(
        client=codex_client,
        config=CodexAdapterConfig(
            model=hermes_model or resolved_config.continuity_reasoning_model,
            reasoning_effort=resolved_config.continuity_reasoning_effort,  # type: ignore[arg-type]
        )
    )
    embedding_client = OllamaEmbeddingClient(
        config=OllamaEmbeddingConfig(
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
