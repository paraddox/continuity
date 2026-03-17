"""Codex/OpenAI Responses adapter for Continuity reasoning operations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from continuity.policy import PolicyPack, hermes_v1_policy_pack

from .base import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningMessage,
    SessionSummaryRequest,
    StructuredGenerationRequest,
    TextResponse,
)


ReasoningEffort = Literal["none", "low", "medium", "high"]


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


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


@dataclass(frozen=True, slots=True)
class CodexAdapterConfig:
    model: str = "gpt-5.4"
    reasoning_effort: ReasoningEffort = "low"
    adapter_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _clean_text(self.model, field_name="model"))
        if self.adapter_version <= 0:
            raise ValueError("adapter_version must be positive")

    @property
    def strategy_id(self) -> str:
        return f"reasoning:codex_sdk_{_slugify(self.model)}_{self.reasoning_effort}"

    @property
    def fingerprint(self) -> str:
        return f"{self.strategy_id}@{self.adapter_version}"


@dataclass(frozen=True, slots=True)
class CodexStructuredSpec:
    name: str
    schema: dict[str, object]
    strict: bool = True
    payload_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, field_name="name"))
        if not self.schema:
            raise ValueError("schema must be non-empty")

    def as_text_format(self) -> dict[str, object]:
        return {
            "format": {
                "type": "json_schema",
                "name": self.name,
                "schema": self.schema,
                "strict": self.strict,
            }
        }


@dataclass(frozen=True, slots=True)
class CodexPromptPolicy:
    policy_stamp: str
    answer_query_instructions: str
    structured_generation_instructions: str
    session_summary_instructions: str
    claim_derivation_instructions: str
    generic_structured_spec: CodexStructuredSpec
    claim_derivation_spec: CodexStructuredSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(
            self,
            "answer_query_instructions",
            _clean_text(self.answer_query_instructions, field_name="answer_query_instructions"),
        )
        object.__setattr__(
            self,
            "structured_generation_instructions",
            _clean_text(
                self.structured_generation_instructions,
                field_name="structured_generation_instructions",
            ),
        )
        object.__setattr__(
            self,
            "session_summary_instructions",
            _clean_text(self.session_summary_instructions, field_name="session_summary_instructions"),
        )
        object.__setattr__(
            self,
            "claim_derivation_instructions",
            _clean_text(self.claim_derivation_instructions, field_name="claim_derivation_instructions"),
        )


def prompt_policy_for(policy_pack: PolicyPack) -> CodexPromptPolicy:
    policy_stamp = policy_pack.policy_stamp
    policy_slug = _slugify(policy_pack.policy_name)

    generic_structured_spec = CodexStructuredSpec(
        name=f"structured.{policy_slug}.v1",
        payload_key="payload",
        schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "additionalProperties": True,
                }
            },
            "required": ["payload"],
            "additionalProperties": False,
        },
    )
    claim_derivation_spec = CodexStructuredSpec(
        name=f"claims.{policy_slug}.v1",
        schema={
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string"},
                            "evidence_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["statement", "evidence_refs"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
    )

    return CodexPromptPolicy(
        policy_stamp=policy_stamp,
        answer_query_instructions=(
            f"You are Continuity's reasoning adapter for policy {policy_stamp}. "
            "Answer the user's memory question using only the supplied transcript and memory context. "
            "If the answer is unknown or withheld, say so directly."
        ),
        structured_generation_instructions=(
            f"You are Continuity's reasoning adapter for policy {policy_stamp}. "
            "Return only strict JSON matching the requested schema, with the task result in the top-level payload field."
        ),
        session_summary_instructions=(
            f"You are Continuity's reasoning adapter for policy {policy_stamp}. "
            "Write a concise session summary that preserves current-state facts and next-turn continuity."
        ),
        claim_derivation_instructions=(
            f"You are Continuity's reasoning adapter for policy {policy_stamp}. "
            "Derive candidate memory claims from the supplied observations. "
            "Every claim must include evidence_refs using observation labels such as observation:0."
        ),
        generic_structured_spec=generic_structured_spec,
        claim_derivation_spec=claim_derivation_spec,
    )


@runtime_checkable
class ResponsesClient(Protocol):
    def create(self, **kwargs: object) -> object: ...


class CodexAdapterError(RuntimeError):
    """Raised when the Codex adapter cannot produce a valid response."""


class CodexSDKUnavailableError(CodexAdapterError):
    """Raised when the OpenAI Python SDK is unavailable for live usage."""


def _default_responses_client() -> ResponsesClient:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise CodexSDKUnavailableError(
            "OpenAI Python SDK is required when no Responses client is injected"
        ) from exc

    return OpenAI().responses


class CodexAdapter:
    def __init__(
        self,
        *,
        client: ResponsesClient | None = None,
        config: CodexAdapterConfig | None = None,
        policy_pack: PolicyPack | None = None,
        prompt_policy: CodexPromptPolicy | None = None,
    ) -> None:
        self.config = config or CodexAdapterConfig()
        resolved_policy_pack = policy_pack or hermes_v1_policy_pack()
        self.prompt_policy = prompt_policy or prompt_policy_for(resolved_policy_pack)
        self._client = client or _default_responses_client()

    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        response = self._client.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            instructions=self.prompt_policy.answer_query_instructions,
            input=self._messages_with_tail(
                request.messages,
                ReasoningMessage(role="user", content=request.query),
            ),
        )
        return TextResponse(text=self._extract_output_text(response))

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput:
        response = self._client.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            instructions=self.prompt_policy.structured_generation_instructions,
            input=self._messages_with_tail(
                request.messages,
                ReasoningMessage(role="user", content=request.instructions),
            ),
            text=self.prompt_policy.generic_structured_spec.as_text_format(),
        )
        return RawStructuredOutput(
            payload=self._decode_structured_payload(
                response,
                self.prompt_policy.generic_structured_spec,
            )
        )

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse:
        response = self._client.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            instructions=self.prompt_policy.session_summary_instructions,
            input=self._messages_with_tail(
                request.messages,
                ReasoningMessage(
                    role="user",
                    content=f"Summarize session {request.session_key}.",
                ),
            ),
        )
        return TextResponse(text=self._extract_output_text(response))

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        response = self._client.create(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            instructions=self.prompt_policy.claim_derivation_instructions,
            input=self._messages_to_input(request.observations),
            text=self.prompt_policy.claim_derivation_spec.as_text_format(),
        )
        return RawStructuredOutput(
            payload=self._decode_structured_payload(
                response,
                self.prompt_policy.claim_derivation_spec,
            )
        )

    @staticmethod
    def _messages_to_input(messages: Sequence[ReasoningMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _messages_with_tail(
        self,
        messages: Sequence[ReasoningMessage],
        tail_message: ReasoningMessage,
    ) -> list[dict[str, str]]:
        combined = list(messages)
        combined.append(tail_message)
        return self._messages_to_input(combined)

    def _extract_output_text(self, response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        raise CodexAdapterError("Codex response did not include output_text")

    def _decode_structured_payload(
        self,
        response: object,
        spec: CodexStructuredSpec,
    ) -> object:
        output_text = self._extract_output_text(response)
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise CodexAdapterError(f"Codex structured response was not valid JSON for {spec.name}") from exc

        if not isinstance(payload, Mapping):
            raise CodexAdapterError(f"Codex structured response must decode to an object for {spec.name}")

        if spec.payload_key is None:
            return dict(payload)

        if spec.payload_key not in payload:
            raise CodexAdapterError(
                f"Codex structured response did not include required key {spec.payload_key!r} for {spec.name}"
            )

        return payload[spec.payload_key]
