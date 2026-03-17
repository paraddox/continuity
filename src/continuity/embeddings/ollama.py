"""Ollama embedding client for the local Continuity retrieval stack."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Protocol, runtime_checkable
from urllib import error, request


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _slugify(value: str) -> str:
    characters: list[str] = []
    previous_was_separator = False
    for char in value.lower():
        if char.isalnum():
            characters.append(char)
            previous_was_separator = False
            continue
        if not previous_was_separator:
            characters.append("_")
            previous_was_separator = True
    return "".join(characters).strip("_")


def _clean_dimensions(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError("dimensions must be positive")
    return value


def _clean_timeout(value: float) -> float:
    if value <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    return float(value)


def _clean_inputs(inputs: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(inputs, str):
        return (_clean_text(inputs, field_name="input"),)

    cleaned_values: list[str] = []
    for value in inputs:
        if not isinstance(value, str):
            raise ValueError("inputs must contain only strings")
        cleaned_values.append(_clean_text(value, field_name="input"))
    cleaned_inputs = tuple(cleaned_values)
    if not cleaned_inputs:
        raise ValueError("inputs must contain at least one item")
    return cleaned_inputs


def _clean_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OllamaEmbeddingError(f"{field_name} must be an integer when present")
    return value


@dataclass(frozen=True, slots=True)
class OllamaEmbeddingConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    dimensions: int | None = None
    truncate: bool = True
    keep_alive: str | None = None
    request_timeout_seconds: float = 30.0
    client_version: int = 1

    def __post_init__(self) -> None:
        base_url = _clean_text(self.base_url, field_name="base_url").rstrip("/")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", _clean_text(self.model, field_name="model"))
        object.__setattr__(self, "dimensions", _clean_dimensions(self.dimensions))
        object.__setattr__(self, "truncate", bool(self.truncate))
        object.__setattr__(self, "keep_alive", _clean_optional_text(self.keep_alive, field_name="keep_alive"))
        object.__setattr__(
            self,
            "request_timeout_seconds",
            _clean_timeout(self.request_timeout_seconds),
        )
        if self.client_version <= 0:
            raise ValueError("client_version must be positive")

    @property
    def fingerprint(self) -> str:
        dimension_segment = "native" if self.dimensions is None else f"dim_{self.dimensions}"
        return f"embedding:ollama_{_slugify(self.model)}_{dimension_segment}@{self.client_version}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/embed"

    def request_payload(self, inputs: Sequence[str]) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "input": list(inputs),
            "truncate": self.truncate,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload


@dataclass(frozen=True, slots=True)
class OllamaEmbeddingBatch:
    model: str
    embeddings: tuple[tuple[float, ...], ...]
    dimensions: int
    fingerprint: str
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _clean_text(self.model, field_name="model"))
        if not self.embeddings:
            raise ValueError("embeddings must contain at least one vector")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        for vector in self.embeddings:
            if len(vector) != self.dimensions:
                raise ValueError("every embedding must match the declared dimensions")
        object.__setattr__(self, "fingerprint", _clean_text(self.fingerprint, field_name="fingerprint"))
        object.__setattr__(self, "total_duration", _clean_optional_int(self.total_duration, field_name="total_duration"))
        object.__setattr__(self, "load_duration", _clean_optional_int(self.load_duration, field_name="load_duration"))
        object.__setattr__(
            self,
            "prompt_eval_count",
            _clean_optional_int(self.prompt_eval_count, field_name="prompt_eval_count"),
        )


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama embedding requests fail or return invalid data."""


@runtime_checkable
class OllamaTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> object: ...


class UrllibOllamaTransport:
    def post_json(
        self,
        *,
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> object:
        request_body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        http_request = request.Request(
            url,
            data=request_body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaEmbeddingError(
                f"Ollama embed request failed with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise OllamaEmbeddingError(f"Ollama embed request failed: {exc.reason}") from exc
        except OSError as exc:
            raise OllamaEmbeddingError(f"Ollama embed request failed: {exc}") from exc

        try:
            return json.loads(response_body)
        except JSONDecodeError as exc:
            raise OllamaEmbeddingError("Ollama embed response must be valid JSON") from exc


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        config: OllamaEmbeddingConfig | None = None,
        transport: OllamaTransport | None = None,
    ) -> None:
        self.config = config or OllamaEmbeddingConfig()
        self._transport = transport or UrllibOllamaTransport()

    def embed(self, inputs: str | Sequence[str]) -> OllamaEmbeddingBatch:
        cleaned_inputs = _clean_inputs(inputs)
        payload = self.config.request_payload(cleaned_inputs)
        raw_response = self._transport.post_json(
            url=self.config.endpoint,
            payload=payload,
            timeout_seconds=self.config.request_timeout_seconds,
        )
        return self._parse_batch(raw_response, expected_count=len(cleaned_inputs))

    def _parse_batch(self, raw_response: object, *, expected_count: int) -> OllamaEmbeddingBatch:
        if not isinstance(raw_response, Mapping):
            raise OllamaEmbeddingError("Ollama embed response must be a JSON object")

        model = raw_response.get("model")
        embeddings_payload = raw_response.get("embeddings")
        if not isinstance(model, str):
            raise OllamaEmbeddingError("Ollama embed response must include a string model")
        if not isinstance(embeddings_payload, list) or not embeddings_payload:
            raise OllamaEmbeddingError("Ollama embed response must include a non-empty embeddings list")

        embeddings = tuple(self._parse_vector(vector) for vector in embeddings_payload)
        if len(embeddings) != expected_count:
            raise OllamaEmbeddingError(
                f"Ollama embed response returned {len(embeddings)} embeddings for {expected_count} inputs"
            )

        dimensions = self.config.dimensions or len(embeddings[0])
        if dimensions <= 0:
            raise OllamaEmbeddingError("Ollama embed response vectors must not be empty")

        for vector in embeddings:
            if len(vector) != dimensions:
                raise OllamaEmbeddingError(f"expected embeddings with {dimensions} dimensions")

        return OllamaEmbeddingBatch(
            model=model,
            embeddings=embeddings,
            dimensions=dimensions,
            fingerprint=self.config.fingerprint,
            total_duration=raw_response.get("total_duration"),
            load_duration=raw_response.get("load_duration"),
            prompt_eval_count=raw_response.get("prompt_eval_count"),
        )

    def _parse_vector(self, raw_vector: object) -> tuple[float, ...]:
        if not isinstance(raw_vector, list) or not raw_vector:
            raise OllamaEmbeddingError("each embedding must be a non-empty list of numbers")

        vector: list[float] = []
        for raw_value in raw_vector:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise OllamaEmbeddingError("embedding values must be numeric")
            vector.append(float(raw_value))
        return tuple(vector)


__all__ = [
    "OllamaEmbeddingBatch",
    "OllamaEmbeddingClient",
    "OllamaEmbeddingConfig",
    "OllamaEmbeddingError",
]
