"""Hermes-backed chat-completions adapter for Continuity reasoning operations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
from .codex_adapter import CodexAdapterError, CodexPromptPolicy, CodexStructuredSpec, prompt_policy_for


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


@dataclass(frozen=True, slots=True)
class HermesChatAdapterConfig:
    model: str
    reasoning_effort: str = "low"
    adapter_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _clean_text(self.model, field_name="model"))
        object.__setattr__(
            self,
            "reasoning_effort",
            _clean_text(self.reasoning_effort, field_name="reasoning_effort"),
        )
        if self.adapter_version <= 0:
            raise ValueError("adapter_version must be positive")


@runtime_checkable
class ChatCompletionsClient(Protocol):
    chat: Any


class HermesChatAdapter:
    """Continuity reasoning adapter backed by Hermes provider-resolved chat completions clients."""

    def __init__(
        self,
        *,
        client: ChatCompletionsClient,
        config: HermesChatAdapterConfig,
        policy_pack: PolicyPack | None = None,
        prompt_policy: CodexPromptPolicy | None = None,
    ) -> None:
        self._client = client
        self.config = config
        resolved_policy_pack = policy_pack or hermes_v1_policy_pack()
        self.prompt_policy = prompt_policy or prompt_policy_for(resolved_policy_pack)

    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=self._messages_with_system(
                request.messages,
                self.prompt_policy.answer_query_instructions,
                ReasoningMessage(role="user", content=request.query),
            ),
            temperature=0,
        )
        return TextResponse(text=self._extract_output_text(response))

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput:
        response = self._create_structured_response(
            system_instructions=self.prompt_policy.structured_generation_instructions,
            user_messages=self._messages_with_tail(
                request.messages,
                ReasoningMessage(role="user", content=request.instructions),
            ),
            spec=self.prompt_policy.generic_structured_spec,
        )
        return RawStructuredOutput(
            payload=self._decode_structured_payload(
                response,
                self.prompt_policy.generic_structured_spec,
            )
        )

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=self._messages_with_system(
                request.messages,
                self.prompt_policy.session_summary_instructions,
                ReasoningMessage(
                    role="user",
                    content=f"Summarize session {request.session_key}.",
                ),
            ),
            temperature=0,
        )
        return TextResponse(text=self._extract_output_text(response))

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        response = self._create_structured_response(
            system_instructions=self.prompt_policy.claim_derivation_instructions,
            user_messages=self._messages_to_input(request.observations),
            spec=self.prompt_policy.claim_derivation_spec,
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

    def _messages_with_system(
        self,
        messages: Sequence[ReasoningMessage],
        system_instructions: str,
        tail_message: ReasoningMessage,
    ) -> list[dict[str, str]]:
        combined: list[dict[str, str]] = [{"role": "system", "content": system_instructions}]
        combined.extend(self._messages_with_tail(messages, tail_message))
        return combined

    def _create_structured_response(
        self,
        *,
        system_instructions: str,
        user_messages: Sequence[dict[str, str]],
        spec: CodexStructuredSpec,
    ) -> object:
        request_kwargs = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system_instructions}, *list(user_messages)],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": spec.name,
                    "schema": spec.schema,
                    "strict": spec.strict,
                },
            },
        }
        try:
            return self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if not self._should_retry_without_schema(exc):
                raise
            fallback_messages = [
                {
                    "role": "system",
                    "content": (
                        f"{system_instructions}\n"
                        "Return only valid JSON matching this schema. "
                        f"Schema: {json.dumps(spec.schema, separators=(',', ':'), sort_keys=True)}"
                    ),
                },
                *list(user_messages),
            ]
            return self._client.chat.completions.create(
                model=self.config.model,
                messages=fallback_messages,
                temperature=0,
            )

    @staticmethod
    def _should_retry_without_schema(exc: Exception) -> bool:
        text = str(exc).casefold()
        return (
            "invalid schema for response_format" in text
            or "invalid_json_schema" in text
            or "response_format" in text
        )

    def _extract_output_text(self, response: object) -> str:
        try:
            message = response.choices[0].message
        except Exception as exc:
            raise CodexAdapterError("Chat-completions response did not include a message choice") from exc

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, Sequence) and not isinstance(content, str):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                elif isinstance(item, Mapping):
                    mapping_text = item.get("text")
                    if isinstance(mapping_text, str) and mapping_text.strip():
                        parts.append(mapping_text.strip())
            if parts:
                return "\n".join(parts)
        raise CodexAdapterError("Chat-completions response did not include output text")

    def _decode_structured_payload(
        self,
        response: object,
        spec: CodexStructuredSpec,
    ) -> object:
        output_text = self._extract_output_text(response)
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise CodexAdapterError(
                f"Chat-completions structured response was not valid JSON for {spec.name}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise CodexAdapterError(
                f"Chat-completions structured response must decode to an object for {spec.name}"
            )

        if spec.payload_key is None:
            return dict(payload)

        if spec.payload_key not in payload:
            raise CodexAdapterError(
                "Chat-completions structured response did not include required key "
                f"{spec.payload_key!r} for {spec.name}"
            )

        return payload[spec.payload_key]
