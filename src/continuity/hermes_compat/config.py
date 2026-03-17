"""Shared Hermes-facing config surface for Honcho and Continuity backends."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from continuity.config import ContinuityConfig, DEFAULT_CONFIG_PATH


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _optional_clean_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return _clean_text(cleaned, field_name=field_name)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _tuple_of_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple | list):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


class HermesMemoryBackendKind(StrEnum):
    HONCHO = "honcho"
    CONTINUITY = "continuity"


class ContinuityVectorBackendKind(StrEnum):
    IN_MEMORY = "inmemory"
    ZVEC = "zvec"


def _normalize_backend(value: object) -> HermesMemoryBackendKind:
    cleaned = str(value or HermesMemoryBackendKind.HONCHO.value).strip().lower()
    if cleaned == HermesMemoryBackendKind.CONTINUITY.value:
        return HermesMemoryBackendKind.CONTINUITY
    return HermesMemoryBackendKind.HONCHO


def _normalize_vector_backend(value: object) -> ContinuityVectorBackendKind:
    cleaned = str(value or ContinuityVectorBackendKind.IN_MEMORY.value).strip().lower()
    if cleaned == ContinuityVectorBackendKind.ZVEC.value:
        return ContinuityVectorBackendKind.ZVEC
    return ContinuityVectorBackendKind.IN_MEMORY


def _resolve_enabled(
    *,
    backend: HermesMemoryBackendKind,
    host_enabled: object,
    root_enabled: object,
    api_key: str | None,
) -> bool:
    if host_enabled is not None:
        return bool(host_enabled)
    if root_enabled is not None:
        return bool(root_enabled)
    if backend is HermesMemoryBackendKind.CONTINUITY:
        return True
    return bool(api_key)


@dataclass(frozen=True, slots=True)
class HermesMemoryConfig:
    backend: HermesMemoryBackendKind = HermesMemoryBackendKind.HONCHO
    enabled: bool = False
    api_key: str | None = None
    environment: str = "production"
    base_url: str | None = None
    linked_hosts: tuple[str, ...] = ()
    context_tokens: int | None = None
    dialectic_reasoning_level: str = "low"
    dialectic_max_chars: int = 600
    continuity_store_path: Path = Path.home() / ".hermes" / "continuity.db"
    continuity_vector_backend: ContinuityVectorBackendKind = ContinuityVectorBackendKind.IN_MEMORY
    continuity_collection_path: Path = Path.home() / ".hermes" / "continuity-zvec"
    continuity_embedding_model: str = "nomic-embed-text"
    continuity_embedding_base_url: str = "http://127.0.0.1:11434"
    continuity_embedding_dimensions: int | None = None
    continuity_reasoning_model: str = "gpt-5.4"
    continuity_reasoning_effort: str = "low"
    continuity_policy_name: str = "hermes_v1"
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", _clean_text(self.environment, field_name="environment"))
        object.__setattr__(
            self,
            "linked_hosts",
            tuple(dict.fromkeys(_clean_text(item, field_name="linked_hosts") for item in self.linked_hosts)),
        )
        object.__setattr__(
            self,
            "dialectic_reasoning_level",
            _clean_text(
                self.dialectic_reasoning_level,
                field_name="dialectic_reasoning_level",
            ),
        )
        object.__setattr__(
            self,
            "continuity_embedding_model",
            _clean_text(
                self.continuity_embedding_model,
                field_name="continuity_embedding_model",
            ),
        )
        object.__setattr__(
            self,
            "continuity_embedding_base_url",
            _clean_text(
                self.continuity_embedding_base_url,
                field_name="continuity_embedding_base_url",
            ),
        )
        object.__setattr__(
            self,
            "continuity_reasoning_model",
            _clean_text(
                self.continuity_reasoning_model,
                field_name="continuity_reasoning_model",
            ),
        )
        object.__setattr__(
            self,
            "continuity_reasoning_effort",
            _clean_text(
                self.continuity_reasoning_effort,
                field_name="continuity_reasoning_effort",
            ),
        )
        object.__setattr__(
            self,
            "continuity_policy_name",
            _clean_text(
                self.continuity_policy_name,
                field_name="continuity_policy_name",
            ),
        )
        object.__setattr__(self, "continuity_store_path", Path(self.continuity_store_path).expanduser())
        object.__setattr__(
            self,
            "continuity_collection_path",
            Path(self.continuity_collection_path).expanduser(),
        )
        if self.continuity_embedding_dimensions is not None and self.continuity_embedding_dimensions <= 0:
            raise ValueError("continuity_embedding_dimensions must be positive when provided")

    @property
    def host(self) -> str:
        return self.continuity.host

    @property
    def workspace_id(self) -> str:
        return self.continuity.workspace_id

    @property
    def ai_peer(self) -> str:
        return self.continuity.ai_peer

    @property
    def peer_name(self) -> str | None:
        return self.continuity.peer_name

    @property
    def memory_mode(self) -> str:
        return self.continuity.memory_mode

    @property
    def peer_memory_modes(self) -> dict[str, str]:
        return dict(self.continuity.peer_memory_modes)

    @property
    def write_frequency(self) -> str | int:
        return self.continuity.write_frequency

    @property
    def recall_mode(self) -> str:
        return self.continuity.recall_mode

    @property
    def session_strategy(self) -> str:
        return self.continuity.session_strategy

    @property
    def session_peer_prefix(self) -> bool:
        return self.continuity.session_peer_prefix

    @property
    def sessions(self) -> dict[str, str]:
        return dict(self.continuity.sessions)

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self.continuity.raw)

    def peer_memory_mode(self, peer_name: str) -> str:
        return self.continuity.peer_memory_mode(peer_name)

    def resolve_session_name(
        self,
        cwd: str | None = None,
        *,
        session_title: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        return self.continuity.resolve_session_name(
            cwd,
            session_title=session_title,
            session_id=session_id,
        )

    @classmethod
    def from_env(cls, *, host: str = "hermes") -> HermesMemoryConfig:
        backend = _normalize_backend(os.environ.get("HERMES_MEMORY_BACKEND"))
        api_key = _optional_clean_text(os.environ.get("HONCHO_API_KEY"), field_name="api_key")
        continuity = ContinuityConfig(
            host=host,
            workspace_id=host,
            ai_peer=host,
        )
        return cls(
            backend=backend,
            enabled=backend is HermesMemoryBackendKind.CONTINUITY or bool(api_key),
            api_key=api_key,
            environment=os.environ.get("HONCHO_ENVIRONMENT", "production"),
            base_url=_optional_clean_text(os.environ.get("HONCHO_URL"), field_name="base_url"),
            continuity_store_path=Path(
                os.environ.get("CONTINUITY_STORE_PATH", str(Path.home() / ".hermes" / "continuity.db"))
            ),
            continuity_collection_path=Path(
                os.environ.get(
                    "CONTINUITY_ZVEC_COLLECTION_PATH",
                    str(Path.home() / ".hermes" / "continuity-zvec"),
                )
            ),
            continuity_vector_backend=_normalize_vector_backend(
                os.environ.get("CONTINUITY_VECTOR_BACKEND")
            ),
            continuity_embedding_model=os.environ.get(
                "CONTINUITY_EMBEDDING_MODEL",
                "nomic-embed-text",
            ),
            continuity_embedding_base_url=os.environ.get(
                "CONTINUITY_OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ),
            continuity_embedding_dimensions=(
                None
                if os.environ.get("CONTINUITY_EMBEDDING_DIMENSIONS") is None
                else int(os.environ["CONTINUITY_EMBEDDING_DIMENSIONS"])
            ),
            continuity_reasoning_model=os.environ.get(
                "CONTINUITY_REASONING_MODEL",
                "gpt-5.4",
            ),
            continuity_reasoning_effort=os.environ.get(
                "CONTINUITY_REASONING_EFFORT",
                "low",
            ),
            continuity=continuity,
        )

    @classmethod
    def from_mapping(
        cls,
        raw_config: Mapping[str, object],
        *,
        host: str = "hermes",
    ) -> HermesMemoryConfig:
        raw = dict(raw_config)
        hosts = _mapping(raw.get("hosts"))
        host_block = _mapping(hosts.get(host))
        continuity_block = _mapping(raw.get("continuity"))
        host_continuity = _mapping(host_block.get("continuity"))

        backend = _normalize_backend(host_block.get("backend") or raw.get("backend"))
        api_key = _optional_clean_text(
            host_block.get("apiKey") or raw.get("apiKey") or os.environ.get("HONCHO_API_KEY"),
            field_name="api_key",
        )
        enabled = _resolve_enabled(
            backend=backend,
            host_enabled=host_block.get("enabled"),
            root_enabled=raw.get("enabled"),
            api_key=api_key,
        )
        continuity = ContinuityConfig.from_mapping(raw, host=host)

        return cls(
            backend=backend,
            enabled=enabled,
            api_key=api_key,
            environment=str(host_block.get("environment") or raw.get("environment") or "production"),
            base_url=_optional_clean_text(
                host_block.get("baseUrl") or raw.get("baseUrl") or os.environ.get("HONCHO_URL"),
                field_name="base_url",
            ),
            linked_hosts=_tuple_of_strings(host_block.get("linkedHosts")),
            context_tokens=(
                None
                if host_block.get("contextTokens") is None and raw.get("contextTokens") is None
                else int(host_block.get("contextTokens") or raw.get("contextTokens"))
            ),
            dialectic_reasoning_level=str(
                host_block.get("dialecticReasoningLevel")
                or raw.get("dialecticReasoningLevel")
                or "low"
            ),
            dialectic_max_chars=int(
                host_block.get("dialecticMaxChars")
                or raw.get("dialecticMaxChars")
                or 600
            ),
            continuity_store_path=Path(
                host_continuity.get("storePath")
                or continuity_block.get("storePath")
                or Path.home() / ".hermes" / "continuity.db"
            ),
            continuity_vector_backend=_normalize_vector_backend(
                host_continuity.get("vectorBackend")
                or continuity_block.get("vectorBackend")
            ),
            continuity_collection_path=Path(
                host_continuity.get("collectionPath")
                or continuity_block.get("collectionPath")
                or Path.home() / ".hermes" / "continuity-zvec"
            ),
            continuity_embedding_model=str(
                host_continuity.get("embeddingModel")
                or continuity_block.get("embeddingModel")
                or "nomic-embed-text"
            ),
            continuity_embedding_base_url=str(
                host_continuity.get("embeddingBaseUrl")
                or continuity_block.get("embeddingBaseUrl")
                or "http://127.0.0.1:11434"
            ),
            continuity_embedding_dimensions=(
                None
                if host_continuity.get("embeddingDimensions") is None
                and continuity_block.get("embeddingDimensions") is None
                else int(
                    host_continuity.get("embeddingDimensions")
                    or continuity_block.get("embeddingDimensions")
                )
            ),
            continuity_reasoning_model=str(
                host_continuity.get("reasoningModel")
                or continuity_block.get("reasoningModel")
                or "gpt-5.4"
            ),
            continuity_reasoning_effort=str(
                host_continuity.get("reasoningEffort")
                or continuity_block.get("reasoningEffort")
                or "low"
            ),
            continuity_policy_name=str(
                host_continuity.get("policyName")
                or continuity_block.get("policyName")
                or "hermes_v1"
            ),
            continuity=continuity,
        )

    @classmethod
    def from_global_config(
        cls,
        *,
        host: str = "hermes",
        config_path: Path | None = None,
    ) -> HermesMemoryConfig:
        path = config_path or DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls.from_env(host=host)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls.from_env(host=host)

        if not isinstance(raw, Mapping):
            return cls.from_env(host=host)
        config = cls.from_mapping(raw, host=host)
        backend_override = os.environ.get("HERMES_MEMORY_BACKEND")
        if backend_override is None:
            return config
        return cls(
            backend=_normalize_backend(backend_override),
            enabled=True,
            api_key=config.api_key,
            environment=config.environment,
            base_url=config.base_url,
            linked_hosts=config.linked_hosts,
            context_tokens=config.context_tokens,
            dialectic_reasoning_level=config.dialectic_reasoning_level,
            dialectic_max_chars=config.dialectic_max_chars,
            continuity_store_path=config.continuity_store_path,
            continuity_vector_backend=config.continuity_vector_backend,
            continuity_collection_path=config.continuity_collection_path,
            continuity_embedding_model=config.continuity_embedding_model,
            continuity_embedding_base_url=config.continuity_embedding_base_url,
            continuity_embedding_dimensions=config.continuity_embedding_dimensions,
            continuity_reasoning_model=config.continuity_reasoning_model,
            continuity_reasoning_effort=config.continuity_reasoning_effort,
            continuity_policy_name=config.continuity_policy_name,
            continuity=config.continuity,
        )
