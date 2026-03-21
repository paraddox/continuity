#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.policy import hermes_v1_policy_pack
from continuity.reasoning import (
    AnswerQueryRequest,
    ClaimDerivationEnvelope,
    ClaimDerivationEnvelopeSchema,
    ClaimDerivationRequest,
    ClaimDerivationCandidate,
    RawStructuredOutput,
    ReasoningAdapter,
    ReasoningMessage,
    SessionSummaryRequest,
    StructuredGenerationRequest,
    StructuredOutputSchema,
    TextResponse,
    validate_structured_output,
)
from continuity.reasoning.codex_adapter import CodexAdapter
from continuity.reasoning.hermes_chat_adapter import HermesChatAdapter, HermesChatAdapterConfig


@dataclass(frozen=True)
class PreferencePayload:
    memory_type: str
    value: str


class PreferencePayloadSchema(StructuredOutputSchema[PreferencePayload]):
    name = "preference_payload.v1"

    def validate(self, payload: object) -> PreferencePayload:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a mapping")
        memory_type = payload.get("memory_type")
        value = payload.get("value")
        if not isinstance(memory_type, str) or not memory_type:
            raise ValueError("memory_type must be a non-empty string")
        if not isinstance(value, str) or not value:
            raise ValueError("value must be a non-empty string")
        return PreferencePayload(memory_type=memory_type, value=value)


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class RecordingResponsesClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)

    def create(self, **kwargs: object) -> FakeResponse:
        if not self._responses:
            raise AssertionError("no fake responses remaining")
        return self._responses.pop(0)


class RecordingChatCompletionsClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object) -> FakeResponse:
        if not self._responses:
            raise AssertionError("no fake responses remaining")
        output_text = self._responses.pop(0).output_text
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=output_text),
                )
            ]
        )


class FixtureBackedFakeReasoningAdapter:
    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        return TextResponse(text="Alice prefers black coffee.")

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput:
        return RawStructuredOutput(payload={"memory_type": "preference", "value": "black coffee"})

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse:
        return TextResponse(text=f"Summary for {request.session_key}: Alice prefers black coffee.")

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        return RawStructuredOutput(
            payload={
                "candidates": [
                    {
                        "claim_type": "preference",
                        "subject_ref": "observation:0.author",
                        "scope": "user",
                        "locus_key": "preference/favorite_drink",
                        "value": {"drink": "black coffee"},
                        "evidence_refs": ["observation:0"],
                    }
                ]
            }
        )


def sample_messages() -> tuple[ReasoningMessage, ...]:
    return (
        ReasoningMessage(role="user", content="Alice prefers black coffee."),
        ReasoningMessage(role="assistant", content="Stored."),
    )


def sample_query_request() -> AnswerQueryRequest:
    return AnswerQueryRequest(
        query="What does Alice prefer?",
        messages=sample_messages(),
    )


def sample_structured_request() -> StructuredGenerationRequest:
    return StructuredGenerationRequest(
        instructions="Extract the durable memory candidate.",
        messages=sample_messages(),
    )


def sample_summary_request() -> SessionSummaryRequest:
    return SessionSummaryRequest(
        session_key="telegram:123456",
        messages=sample_messages(),
    )


def sample_claim_request() -> ClaimDerivationRequest:
    return ClaimDerivationRequest(observations=sample_messages())


def expected_claim_envelope() -> ClaimDerivationEnvelope:
    return ClaimDerivationEnvelope(
        candidates=(
            ClaimDerivationCandidate(
                claim_type="preference",
                subject_ref="observation:0.author",
                scope="user",
                locus_key="preference/favorite_drink",
                value={"drink": "black coffee"},
                evidence_refs=("observation:0",),
            ),
        )
    )


class ReasoningAdapterContractMixin:
    def make_adapter(self) -> ReasoningAdapter:
        raise NotImplementedError

    def make_answer_adapter(self) -> ReasoningAdapter:
        return self.make_adapter()

    def make_structured_adapter(self) -> ReasoningAdapter:
        return self.make_adapter()

    def make_summary_adapter(self) -> ReasoningAdapter:
        return self.make_adapter()

    def make_claim_adapter(self) -> ReasoningAdapter:
        return self.make_adapter()

    def test_adapter_satisfies_protocol(self) -> None:
        self.assertIsInstance(self.make_adapter(), ReasoningAdapter)

    def test_answer_query_returns_shared_fixture_text(self) -> None:
        response = self.make_answer_adapter().answer_query(sample_query_request())
        self.assertEqual(response, TextResponse(text="Alice prefers black coffee."))

    def test_generate_structured_returns_schema_valid_fixture_payload(self) -> None:
        output = self.make_structured_adapter().generate_structured(sample_structured_request())
        validated = validate_structured_output(output, PreferencePayloadSchema())
        self.assertEqual(
            validated.payload,
            PreferencePayload(memory_type="preference", value="black coffee"),
        )

    def test_summarize_session_returns_shared_fixture_text(self) -> None:
        response = self.make_summary_adapter().summarize_session(sample_summary_request())
        self.assertEqual(
            response,
            TextResponse(text="Summary for telegram:123456: Alice prefers black coffee."),
        )

    def test_derive_claims_returns_schema_valid_claim_fixture(self) -> None:
        output = self.make_claim_adapter().derive_claims(sample_claim_request())
        validated = validate_structured_output(output, ClaimDerivationEnvelopeSchema())
        self.assertEqual(validated.payload, expected_claim_envelope())


class FakeAdapterContractTests(ReasoningAdapterContractMixin, unittest.TestCase):
    def make_adapter(self) -> ReasoningAdapter:
        return FixtureBackedFakeReasoningAdapter()


class CodexAdapterContractTests(ReasoningAdapterContractMixin, unittest.TestCase):
    def make_adapter(self) -> ReasoningAdapter:
        return self.make_answer_adapter()

    def _make_codex_adapter(self, output_text: str) -> ReasoningAdapter:
        client = RecordingResponsesClient([FakeResponse(output_text)])
        return CodexAdapter(client=client, policy_pack=hermes_v1_policy_pack())

    def make_answer_adapter(self) -> ReasoningAdapter:
        return self._make_codex_adapter("Alice prefers black coffee.")

    def make_structured_adapter(self) -> ReasoningAdapter:
        return self._make_codex_adapter(
            '{"payload":{"memory_type":"preference","value":"black coffee"}}'
        )

    def make_summary_adapter(self) -> ReasoningAdapter:
        return self._make_codex_adapter("Summary for telegram:123456: Alice prefers black coffee.")

    def make_claim_adapter(self) -> ReasoningAdapter:
        return self._make_codex_adapter(
            json.dumps(
                {
                    "candidates": [
                        {
                            "claim_type": "preference",
                            "subject_ref": "observation:0.author",
                            "scope": "user",
                            "locus_key": "preference/favorite_drink",
                            "value": {"drink": "black coffee"},
                            "evidence_refs": ["observation:0"],
                        }
                    ]
                }
            )
        )


class HermesChatAdapterContractTests(ReasoningAdapterContractMixin, unittest.TestCase):
    def make_adapter(self) -> ReasoningAdapter:
        return self.make_answer_adapter()

    def _make_chat_adapter(self, output_text: str) -> ReasoningAdapter:
        client = RecordingChatCompletionsClient([FakeResponse(output_text)])
        return HermesChatAdapter(
            client=client,
            config=HermesChatAdapterConfig(model="glm-5-turbo"),
            policy_pack=hermes_v1_policy_pack(),
        )

    def make_answer_adapter(self) -> ReasoningAdapter:
        return self._make_chat_adapter("Alice prefers black coffee.")

    def make_structured_adapter(self) -> ReasoningAdapter:
        return self._make_chat_adapter(
            '{"payload":{"memory_type":"preference","value":"black coffee"}}'
        )

    def make_summary_adapter(self) -> ReasoningAdapter:
        return self._make_chat_adapter("Summary for telegram:123456: Alice prefers black coffee.")

    def make_claim_adapter(self) -> ReasoningAdapter:
        return self._make_chat_adapter(
            json.dumps(
                {
                    "candidates": [
                        {
                            "claim_type": "preference",
                            "subject_ref": "observation:0.author",
                            "scope": "user",
                            "locus_key": "preference/favorite_drink",
                            "value": {"drink": "black coffee"},
                            "evidence_refs": ["observation:0"],
                        }
                    ]
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
