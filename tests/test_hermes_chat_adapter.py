#!/usr/bin/env python3

from __future__ import annotations

import unittest
from types import SimpleNamespace

from continuity.policy import hermes_v1_policy_pack
from continuity.reasoning import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningMessage,
    StructuredGenerationRequest,
    TextResponse,
)
from continuity.reasoning.codex_adapter import CodexAdapterError, prompt_policy_for
from continuity.reasoning.hermes_chat_adapter import HermesChatAdapter, HermesChatAdapterConfig


def sample_messages() -> tuple[ReasoningMessage, ...]:
    return (
        ReasoningMessage(role="user", content="Alice prefers black coffee."),
        ReasoningMessage(role="assistant", content="Stored."),
    )


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _RecordingCompletions:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("no fake responses remaining")
        return self._responses.pop(0)


class _FailingOnceCompletions:
    def __init__(self, error: Exception, response: _FakeResponse) -> None:
        self._error = error
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self._error
        return self._response


def _client_with(completions: object) -> object:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class HermesChatAdapterTests(unittest.TestCase):
    def test_answer_query_uses_chat_completions_messages(self) -> None:
        completions = _RecordingCompletions([_FakeResponse("Alice prefers black coffee.")])
        adapter = HermesChatAdapter(
            client=_client_with(completions),
            config=HermesChatAdapterConfig(model="glm-5-turbo"),
            policy_pack=hermes_v1_policy_pack(),
        )

        response = adapter.answer_query(
            AnswerQueryRequest(
                query="What does Alice prefer?",
                messages=sample_messages(),
            )
        )

        self.assertEqual(response, TextResponse(text="Alice prefers black coffee."))
        call = completions.calls[-1]
        self.assertEqual(call["model"], "glm-5-turbo")
        self.assertEqual(
            call["messages"],
            [
                {
                    "role": "system",
                    "content": prompt_policy_for(hermes_v1_policy_pack()).answer_query_instructions,
                },
                {"role": "user", "content": "Alice prefers black coffee."},
                {"role": "assistant", "content": "Stored."},
                {"role": "user", "content": "What does Alice prefer?"},
            ],
        )

    def test_generate_structured_uses_json_schema_response_format(self) -> None:
        completions = _RecordingCompletions(
            [_FakeResponse('{"payload":{"memory_type":"preference","value":"black coffee"}}')]
        )
        adapter = HermesChatAdapter(
            client=_client_with(completions),
            config=HermesChatAdapterConfig(model="glm-5-turbo"),
            policy_pack=hermes_v1_policy_pack(),
        )

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
        call = completions.calls[-1]
        self.assertIn("response_format", call)
        self.assertEqual(
            call["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_hermes_v1_v1",
                    "schema": prompt_policy_for(hermes_v1_policy_pack()).generic_structured_spec.schema,
                    "strict": True,
                },
            },
        )

    def test_derive_claims_uses_claim_schema(self) -> None:
        completions = _RecordingCompletions(
            [
                _FakeResponse(
                    '{"candidates":[{"claim_type":"preference","subject_ref":"observation:0.author","scope":"user","locus_key":"preference/favorite_drink","value":{"drink":"black coffee"},"evidence_refs":["observation:0"]}]}'
                )
            ]
        )
        adapter = HermesChatAdapter(
            client=_client_with(completions),
            config=HermesChatAdapterConfig(model="glm-5-turbo"),
            policy_pack=hermes_v1_policy_pack(),
        )

        response = adapter.derive_claims(ClaimDerivationRequest(observations=sample_messages()))

        self.assertEqual(
            response,
            RawStructuredOutput(
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
            ),
        )
        call = completions.calls[-1]
        self.assertEqual(call["response_format"]["json_schema"]["name"], "claims_hermes_v1_v1")
        self.assertEqual(
            call["messages"],
            [
                {
                    "role": "system",
                    "content": prompt_policy_for(hermes_v1_policy_pack()).claim_derivation_instructions,
                },
                {
                    "role": "user",
                    "content": (
                        "Observations:\n"
                        "observation:0 role=user content=Alice prefers black coffee.\n"
                        "observation:1 role=assistant content=Stored."
                    ),
                },
            ],
        )

    def test_structured_generation_retries_without_schema_when_endpoint_rejects_response_format(self) -> None:
        completions = _FailingOnceCompletions(
            RuntimeError("Invalid schema for response_format 'structured_hermes_v1_v1'"),
            _FakeResponse('{"payload":{"memory_type":"preference","value":"black coffee"}}'),
        )
        adapter = HermesChatAdapter(
            client=_client_with(completions),
            config=HermesChatAdapterConfig(model="glm-5-turbo"),
            policy_pack=hermes_v1_policy_pack(),
        )

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
        self.assertEqual(len(completions.calls), 2)
        self.assertIn("response_format", completions.calls[0])
        self.assertNotIn("response_format", completions.calls[1])

    def test_invalid_structured_json_raises_adapter_error(self) -> None:
        completions = _RecordingCompletions([_FakeResponse("{not-json}")])
        adapter = HermesChatAdapter(
            client=_client_with(completions),
            config=HermesChatAdapterConfig(model="glm-5-turbo"),
            policy_pack=hermes_v1_policy_pack(),
        )

        with self.assertRaises(CodexAdapterError):
            adapter.generate_structured(
                StructuredGenerationRequest(
                    instructions="Extract the durable memory candidate.",
                    messages=sample_messages(),
                )
            )


if __name__ == "__main__":
    unittest.main()
