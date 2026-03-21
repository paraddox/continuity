"""Backend factory for Hermes-compatible Continuity embedding."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

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
from continuity.reasoning.hermes_chat_adapter import HermesChatAdapter, HermesChatAdapterConfig


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


def _resolve_hermes_reasoning_adapter(
    config: HermesMemoryConfig,
) -> ReasoningAdapter | None:
    target = config.continuity_reasoning_target
    if not target.is_configured:
        return None

    hermes_config = _load_hermes_config()
    provider = target.provider
    model = target.model
    base_url: str | None = None
    api_key: str | None = None

    if target.target_name:
        named_target = _find_named_custom_provider(target.target_name, hermes_config)
        if named_target is not None:
            base_url = _clean_optional_text(named_target.get("base_url"))
            api_key = _clean_optional_text(named_target.get("api_key"))
            provider = provider or "custom"
            model = model or _clean_optional_text(named_target.get("model"))
        else:
            provider = provider or target.target_name

    resolved_provider = _clean_optional_text(provider)
    resolved_model = _clean_optional_text(model)
    resolved_effort = target.reasoning_effort or config.continuity_reasoning_effort

    if not resolved_provider:
        raise ValueError(
            "Continuity reasoning target requires either a named Hermes target or an explicit provider"
        )
    if not resolved_model:
        raise ValueError(
            "Continuity reasoning target requires a model, either explicitly or from the named Hermes target"
        )

    if resolved_provider == "openai-codex":
        codex_client, _ = _try_hermes_codex_runtime()
        return CodexAdapter(
            client=codex_client,
            config=CodexAdapterConfig(
                model=resolved_model,
                reasoning_effort=resolved_effort,  # type: ignore[arg-type]
            ),
        )

    auxiliary_module = importlib.import_module("agent.auxiliary_client")
    client, final_model = auxiliary_module.resolve_provider_client(
        resolved_provider,
        model=resolved_model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    if client is None or not final_model:
        raise RuntimeError(
            f"Unable to resolve Hermes reasoning target provider={resolved_provider!r} model={resolved_model!r}"
        )

    return HermesChatAdapter(
        client=client,
        config=HermesChatAdapterConfig(
            model=final_model,
            reasoning_effort=resolved_effort,
        ),
    )


def _clean_optional_text(value: object) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _load_hermes_config() -> Mapping[str, object]:
    config_module = importlib.import_module("hermes_cli.config")
    config = config_module.load_config()
    if isinstance(config, Mapping):
        return config
    return {}


def _normalized_target_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _find_named_custom_provider(
    target_name: str,
    config: Mapping[str, object],
) -> Mapping[str, object] | None:
    requested = _normalized_target_name(target_name.removeprefix("custom:"))
    providers = config.get("custom_providers")
    if not isinstance(providers, list):
        return None

    for entry in providers:
        if not isinstance(entry, Mapping):
            continue
        entry_name = _clean_optional_text(entry.get("name"))
        if not entry_name:
            continue
        if _normalized_target_name(entry_name) == requested:
            return entry
    return None


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

    adapter = reasoning_adapter or _resolve_hermes_reasoning_adapter(resolved_config)
    if adapter is None:
        codex_client, hermes_model = _try_hermes_codex_runtime()
        adapter = CodexAdapter(
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
