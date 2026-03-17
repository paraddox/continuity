#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.policy import hermes_v1_policy_pack
from continuity.reasoning import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningMessage,
    StructuredGenerationRequest,
    TextResponse,
)
from continuity.reasoning.codex_adapter import (
    CodexAdapter,
    CodexAdapterConfig,
    CodexAdapterError,
    CodexSDKUnavailableError,
    prompt_policy_for,
)


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class RecordingResponsesClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("no fake responses remaining")
        return self._responses.pop(0)


def sample_messages() -> tuple[ReasoningMessage, ...]:
    return (
        ReasoningMessage(role="user", content="Alice prefers black coffee."),
        ReasoningMessage(role="assistant", content="Stored."),
    )


class CodexAdapterConfigTests(unittest.TestCase):
    def test_default_config_centralizes_model_and_reasoning_identity(self) -> None:
        config = CodexAdapterConfig()

        self.assertEqual(config.model, "gpt-5.4")
        self.assertEqual(config.reasoning_effort, "low")
        self.assertEqual(config.strategy_id, "reasoning:codex_sdk_gpt_5_4_low")
        self.assertEqual(config.fingerprint, "reasoning:codex_sdk_gpt_5_4_low@1")

    def test_prompt_policy_tracks_policy_pack_identity(self) -> None:
        prompt_policy = prompt_policy_for(hermes_v1_policy_pack())

        self.assertEqual(prompt_policy.policy_stamp, "hermes_v1@1.0.0")
        self.assertEqual(prompt_policy.generic_structured_spec.name, "structured.hermes_v1.v1")
        self.assertEqual(prompt_policy.claim_derivation_spec.name, "claims.hermes_v1.v1")


class CodexAdapterTests(unittest.TestCase):
    def test_answer_query_uses_policy_prompt_and_centralized_sdk_config(self) -> None:
        client = RecordingResponsesClient([FakeResponse("Alice prefers black coffee.")])
        adapter = CodexAdapter(client=client, policy_pack=hermes_v1_policy_pack())

        response = adapter.answer_query(
            AnswerQueryRequest(
                query="What does Alice prefer?",
                messages=sample_messages(),
            )
        )

        self.assertEqual(response, TextResponse(text="Alice prefers black coffee."))

        call = client.calls[-1]
        self.assertEqual(call["model"], "gpt-5.4")
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertIn("hermes_v1@1.0.0", str(call["instructions"]))
        self.assertEqual(
            call["input"],
            [
                {"role": "user", "content": "Alice prefers black coffee."},
                {"role": "assistant", "content": "Stored."},
                {"role": "user", "content": "What does Alice prefer?"},
            ],
        )

    def test_generate_structured_uses_strict_json_schema_and_returns_payload_object(self) -> None:
        client = RecordingResponsesClient([FakeResponse('{"payload":{"memory_type":"preference","value":"black coffee"}}')])
        adapter = CodexAdapter(client=client, policy_pack=hermes_v1_policy_pack())

        response = adapter.generate_structured(
            StructuredGenerationRequest(
                instructions="Extract the durable memory candidate.",
                messages=sample_messages(),
            )
        )

        self.assertEqual(
            response,
            RawStructuredOutput(payload={"memory_type": "preference", "value": "black coffee"}),
        )

        call = client.calls[-1]
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(call["input"][-1], {"role": "user", "content": "Extract the durable memory candidate."})
        self.assertEqual(
            call["text"],
            {
                "format": {
                    "type": "json_schema",
                    "name": "structured.hermes_v1.v1",
                    "schema": prompt_policy_for(hermes_v1_policy_pack()).generic_structured_spec.schema,
                    "strict": True,
                }
            },
        )

    def test_derive_claims_uses_claim_schema_and_preserves_evidence_refs(self) -> None:
        client = RecordingResponsesClient(
            [
                FakeResponse(
                    '{"claims":[{"statement":"Alice prefers black coffee.","evidence_refs":["observation:0"]}]}'
                )
            ]
        )
        adapter = CodexAdapter(client=client, policy_pack=hermes_v1_policy_pack())

        response = adapter.derive_claims(ClaimDerivationRequest(observations=sample_messages()))

        self.assertEqual(
            response,
            RawStructuredOutput(
                payload={
                    "claims": [
                        {
                            "statement": "Alice prefers black coffee.",
                            "evidence_refs": ["observation:0"],
                        }
                    ]
                }
            ),
        )

        call = client.calls[-1]
        self.assertIn("evidence_refs", str(call["instructions"]))
        self.assertEqual(
            call["text"],
            {
                "format": {
                    "type": "json_schema",
                    "name": "claims.hermes_v1.v1",
                    "schema": prompt_policy_for(hermes_v1_policy_pack()).claim_derivation_spec.schema,
                    "strict": True,
                }
            },
        )

    def test_invalid_structured_json_raises_adapter_error(self) -> None:
        client = RecordingResponsesClient([FakeResponse("{not-json}")])
        adapter = CodexAdapter(client=client, policy_pack=hermes_v1_policy_pack())

        with self.assertRaises(CodexAdapterError):
            adapter.generate_structured(
                StructuredGenerationRequest(
                    instructions="Extract the durable memory candidate.",
                    messages=sample_messages(),
                )
            )

    def test_default_client_requires_openai_sdk_when_not_injected(self) -> None:
        with self.assertRaises(CodexSDKUnavailableError):
            CodexAdapter()

    def test_opt_in_live_round_trip_uses_real_openai_client(self) -> None:
        if os.environ.get("CONTINUITY_RUN_LIVE_OPENAI") != "1":
            self.skipTest("set CONTINUITY_RUN_LIVE_OPENAI=1 to run live adapter coverage")

        try:
            from openai import OpenAI
        except ImportError as exc:
            self.skipTest(f"openai package not installed: {exc}")

        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY is required for live adapter coverage")

        adapter = CodexAdapter(client=OpenAI().responses, policy_pack=hermes_v1_policy_pack())

        answer = adapter.answer_query(
            AnswerQueryRequest(
                query="Reply with exactly the word ready.",
            )
        )

        self.assertIsInstance(answer, TextResponse)
        self.assertTrue(answer.text.strip())


if __name__ == "__main__":
    unittest.main()
