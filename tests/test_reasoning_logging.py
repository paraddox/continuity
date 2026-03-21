#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from continuity.reasoning.base import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    SessionSummaryRequest,
    StructuredGenerationRequest,
    TextResponse,
)
from continuity.reasoning.logging import (
    LoggingReasoningAdapter,
    ReasoningRuntimeMetadata,
    list_reasoning_events,
)
from continuity.store.schema import apply_migrations


class _FakeAdapter:
    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        return TextResponse(text="ready")

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput:
        return RawStructuredOutput(payload={"payload": "ok"})

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse:
        return TextResponse(text="summary")

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        return RawStructuredOutput(payload={"candidates": []})


class _FailingAdapter(_FakeAdapter):
    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        raise RuntimeError("boom")


class LoggingReasoningAdapterTests(unittest.TestCase):
    def test_logs_successful_reasoning_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "continuity.db"
            adapter = LoggingReasoningAdapter(
                delegate=_FakeAdapter(),
                store_path=store_path,
                metadata=ReasoningRuntimeMetadata(
                    adapter="HermesChatAdapter",
                    provider="zai",
                    model="glm-5-turbo",
                    target_name="GLM 5 Turbo",
                ),
            )
            self.addCleanup(adapter.close)

            response = adapter.answer_query(AnswerQueryRequest(query="ping"))

            self.assertEqual(response, TextResponse(text="ready"))
            connection = sqlite3.connect(store_path)
            connection.row_factory = sqlite3.Row
            events = list_reasoning_events(connection, limit=10)
            connection.close()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["provider"], "zai")
        self.assertEqual(events[0]["model"], "glm-5-turbo")
        self.assertTrue(events[0]["success"])

    def test_logs_failures_and_prunes_stale_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "continuity.db"
            connection = sqlite3.connect(store_path)
            apply_migrations(connection)
            stale = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            connection.execute(
                """
                INSERT INTO reasoning_events(
                    event_id,
                    recorded_at,
                    operation,
                    adapter,
                    provider,
                    model,
                    target_name,
                    base_url,
                    success,
                    latency_ms,
                    error_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "stale-event",
                    stale,
                    "answer_query",
                    "HermesChatAdapter",
                    "zai",
                    "glm-5-turbo",
                    "GLM 5 Turbo",
                    None,
                    0,
                    999,
                    "stale",
                ),
            )
            connection.commit()
            connection.close()

            adapter = LoggingReasoningAdapter(
                delegate=_FailingAdapter(),
                store_path=store_path,
                metadata=ReasoningRuntimeMetadata(
                    adapter="HermesChatAdapter",
                    provider="zai",
                    model="glm-5-turbo",
                    target_name="GLM 5 Turbo",
                ),
            )
            self.addCleanup(adapter.close)

            with self.assertRaisesRegex(RuntimeError, "boom"):
                adapter.answer_query(AnswerQueryRequest(query="ping"))

            connection = sqlite3.connect(store_path)
            connection.row_factory = sqlite3.Row
            events = list_reasoning_events(connection, limit=10)
            connection.close()

        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["success"])
        self.assertEqual(events[0]["error_text"], "boom")


if __name__ == "__main__":
    unittest.main()
