#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, fields
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.reasoning.base import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningAdapter,
    ReasoningMessage,
    SchemaValidationError,
    SessionSummaryRequest,
    StructuredGenerationRequest,
    StructuredOutputSchema,
    TextResponse,
    ValidatedStructuredOutput,
    validate_structured_output,
    publish_authoritative_mutation,
)


@dataclass(frozen=True)
class ClaimsEnvelope:
    claims: tuple[str, ...]


class ClaimsEnvelopeSchema(StructuredOutputSchema[ClaimsEnvelope]):
    name = "claims.v1"

    def validate(self, payload: object) -> ClaimsEnvelope:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a mapping")

        claims = payload.get("claims")
        if not isinstance(claims, list) or not all(isinstance(item, str) for item in claims):
            raise ValueError("claims must be a list of strings")

        return ClaimsEnvelope(claims=tuple(claims))


class FakeReasoningAdapter:
    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        return TextResponse(text=f"answer:{request.query}")

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput:
        return RawStructuredOutput(payload={"kind": "structured", "instructions": request.instructions})

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse:
        return TextResponse(text=f"summary:{request.session_key}")

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        claims = [message.content for message in request.observations if message.role == "user"]
        return RawStructuredOutput(payload={"claims": claims})


def sample_messages() -> tuple[ReasoningMessage, ...]:
    return (
        ReasoningMessage(role="user", content="Alice prefers black coffee"),
        ReasoningMessage(role="assistant", content="Stored."),
    )


def sample_query_request() -> AnswerQueryRequest:
    return AnswerQueryRequest(
        query="What does Alice prefer?",
        messages=sample_messages(),
    )


def sample_structured_request() -> StructuredGenerationRequest:
    return StructuredGenerationRequest(
        instructions="Extract normalized memory candidates.",
        messages=sample_messages(),
    )


def sample_summary_request() -> SessionSummaryRequest:
    return SessionSummaryRequest(
        session_key="telegram:123456",
        messages=sample_messages(),
    )


def sample_claim_request() -> ClaimDerivationRequest:
    return ClaimDerivationRequest(observations=sample_messages())


def _typecheck_contract() -> None:
    adapter: ReasoningAdapter = FakeReasoningAdapter()
    _query_response: TextResponse = adapter.answer_query(sample_query_request())
    _summary_response: TextResponse = adapter.summarize_session(sample_summary_request())
    _structured_output: RawStructuredOutput = adapter.generate_structured(sample_structured_request())
    claim_output: RawStructuredOutput = adapter.derive_claims(sample_claim_request())
    _validated_output: ValidatedStructuredOutput[ClaimsEnvelope] = validate_structured_output(
        claim_output,
        ClaimsEnvelopeSchema(),
    )


class ReasoningContractTests(unittest.TestCase):
    def test_reasoning_adapter_surface_is_minimal(self) -> None:
        public_methods = {
            name
            for name, value in ReasoningAdapter.__dict__.items()
            if callable(value) and not name.startswith("_")
        }

        self.assertEqual(
            public_methods,
            {"answer_query", "generate_structured", "summarize_session", "derive_claims"},
        )

    def test_contract_dtos_do_not_expose_provider_specific_fields(self) -> None:
        forbidden = {"provider", "model", "reasoning", "temperature", "response_format"}
        dto_fields = {
            name
            for dto_type in (
                AnswerQueryRequest,
                StructuredGenerationRequest,
                SessionSummaryRequest,
                ClaimDerivationRequest,
                RawStructuredOutput,
                TextResponse,
            )
            for name in (field.name for field in fields(dto_type))
        }

        self.assertTrue(forbidden.isdisjoint(dto_fields))

    def test_fake_adapter_satisfies_contract(self) -> None:
        adapter = FakeReasoningAdapter()

        self.assertIsInstance(adapter, ReasoningAdapter)
        self.assertEqual(adapter.answer_query(sample_query_request()).text, "answer:What does Alice prefer?")
        self.assertEqual(adapter.summarize_session(sample_summary_request()).text, "summary:telegram:123456")
        self.assertEqual(
            adapter.generate_structured(sample_structured_request()).payload,
            {"kind": "structured", "instructions": "Extract normalized memory candidates."},
        )

    def test_schema_validation_happens_before_authoritative_mutation(self) -> None:
        adapter = FakeReasoningAdapter()
        published: list[ClaimsEnvelope] = []

        validated = validate_structured_output(adapter.derive_claims(sample_claim_request()), ClaimsEnvelopeSchema())
        result = publish_authoritative_mutation(validated, lambda payload: published.append(payload) or len(payload.claims))

        self.assertEqual(result, 1)
        self.assertEqual(published, [ClaimsEnvelope(claims=("Alice prefers black coffee",))])

    def test_invalid_structured_output_cannot_publish(self) -> None:
        invalid_output = RawStructuredOutput(payload={"claims": "not-a-list"})
        published = False

        def _publish(_: ClaimsEnvelope) -> None:
            nonlocal published
            published = True

        with self.assertRaises(SchemaValidationError):
            validated = validate_structured_output(invalid_output, ClaimsEnvelopeSchema())
            publish_authoritative_mutation(validated, _publish)

        self.assertFalse(published)

    def test_authoritative_mutation_rejects_unvalidated_output(self) -> None:
        raw_output = RawStructuredOutput(payload={"claims": ["Alice prefers black coffee"]})

        with self.assertRaises(TypeError):
            publish_authoritative_mutation(raw_output, lambda payload: payload)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
