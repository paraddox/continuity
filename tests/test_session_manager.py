#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.config import ContinuityConfig
from continuity.ontology import MemoryPartition
from continuity.session_manager import SessionBufferKind, SessionManager
from continuity.store.claims import AdmissionDecision, AdmissionOutcome, CandidateMemory, ClaimScope
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository
from continuity.transactions import DurabilityWaterline


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def build_trace(
    *,
    candidate_id: str,
    outcome: AdmissionOutcome,
    recorded_at: datetime,
) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=outcome,
            recorded_at=recorded_at,
            rationale="kept out of durable memory",
        ),
        claim_type="ephemeral_context",
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type="ephemeral_context",
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.LOW,
            salience=AdmissionStrength.MEDIUM,
            rationale="useful only for the active session",
        ),
        thresholds=AdmissionThresholds(
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH,
            salience=AdmissionStrength.MEDIUM,
        ),
        budget=AdmissionWriteBudget(
            partition=MemoryPartition.SHARED_CONTEXT,
            window_key="session:telegram:123456",
            limit=4,
            used=4,
        ),
    )


class SessionManagerTests(unittest.TestCase):
    def build_manager(
        self,
        *,
        config: ContinuityConfig | None = None,
    ) -> tuple[sqlite3.Connection, SQLiteRepository, SessionManager]:
        connection = open_memory_database()
        repository = SQLiteRepository(connection)
        manager = SessionManager(
            repository,
            config=config
            or ContinuityConfig(
                host="hermes",
                workspace_id="internal-hermes",
                peer_name="hermes",
                session_peer_prefix=True,
            ),
        )
        self.addCleanup(connection.close)
        return connection, repository, manager

    def test_ensure_session_creates_host_scoped_session_record(self) -> None:
        _, repository, manager = self.build_manager(
            config=ContinuityConfig(
                host="hermes",
                workspace_id="internal-hermes",
                peer_name="hermes",
                recall_mode="context",
                write_frequency="turn",
                session_strategy="per-session",
                session_peer_prefix=True,
            )
        )

        session = manager.ensure_session(
            session_id="telegram:123456",
            cwd="/workspace/project",
            created_at=sample_time(),
        )

        self.assertEqual(session.session_id, "telegram:123456")
        self.assertEqual(session.session_name, "hermes-telegram:123456")
        self.assertEqual(session.host_namespace, "hermes")
        self.assertEqual(session.recall_mode, "context")
        self.assertEqual(session.write_frequency, "turn")
        self.assertEqual(repository.read_session("telegram:123456"), session)

    def test_save_turn_records_local_message_cache_and_async_waterline(self) -> None:
        _, repository, manager = self.build_manager(
            config=ContinuityConfig(host="hermes", write_frequency="async")
        )

        result = manager.save_turn(
            session_id="telegram:123456",
            message_id="message-1",
            role="user",
            author_subject_id="subject:user:self",
            content="remember that I prefer espresso",
            observed_at=sample_time(),
        )

        messages = repository.list_messages(session_id="telegram:123456")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].message_id, "message-1")
        self.assertEqual(result.requested_waterline, DurabilityWaterline.OBSERVATION_COMMITTED)
        self.assertEqual(result.execution.requested_waterline, DurabilityWaterline.OBSERVATION_COMMITTED)
        self.assertEqual(result.pending_turns, 0)

    def test_turn_mode_awaits_snapshot_publication_on_every_turn(self) -> None:
        _, _, manager = self.build_manager(
            config=ContinuityConfig(host="hermes", write_frequency="turn")
        )

        result = manager.save_turn(
            session_id="telegram:123456",
            message_id="message-1",
            role="assistant",
            author_subject_id="subject:assistant:hermes",
            content="espresso noted",
            observed_at=sample_time(),
        )

        self.assertEqual(result.requested_waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(result.pending_turns, 0)

    def test_session_mode_defers_snapshot_flush_until_explicit_session_flush(self) -> None:
        _, _, manager = self.build_manager(
            config=ContinuityConfig(host="hermes", write_frequency="session")
        )

        save_result = manager.save_turn(
            session_id="telegram:123456",
            message_id="message-1",
            role="user",
            author_subject_id="subject:user:self",
            content="defer this until session end",
            observed_at=sample_time(),
        )
        flush_result = manager.flush_session("telegram:123456")

        self.assertEqual(save_result.requested_waterline, DurabilityWaterline.OBSERVATION_COMMITTED)
        self.assertEqual(save_result.pending_turns, 1)
        self.assertEqual(flush_result.requested_waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(flush_result.pending_turns, 0)

    def test_integer_write_frequency_flushes_on_turn_threshold(self) -> None:
        _, _, manager = self.build_manager(
            config=ContinuityConfig(host="hermes", write_frequency=2)
        )

        first = manager.save_turn(
            session_id="telegram:123456",
            message_id="message-1",
            role="user",
            author_subject_id="subject:user:self",
            content="first buffered turn",
            observed_at=sample_time(),
        )
        second = manager.save_turn(
            session_id="telegram:123456",
            message_id="message-2",
            role="assistant",
            author_subject_id="subject:assistant:hermes",
            content="second buffered turn",
            observed_at=sample_time(1),
        )

        self.assertEqual(first.requested_waterline, DurabilityWaterline.OBSERVATION_COMMITTED)
        self.assertEqual(first.pending_turns, 1)
        self.assertEqual(second.requested_waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(second.pending_turns, 0)

    def test_session_manager_buffers_prompt_only_and_session_ephemeral_memory(self) -> None:
        _, _, manager = self.build_manager()

        prompt_only = manager.record_non_durable_memory(
            session_id="telegram:123456",
            candidate=CandidateMemory(
                candidate_id="candidate:prompt",
                claim_type="ephemeral_context",
                subject_id="subject:user:self",
                scope=ClaimScope.SESSION,
                value={"topic": "espresso"},
                source_observation_ids=("obs-1",),
            ),
            trace=build_trace(
                candidate_id="candidate:prompt",
                outcome=AdmissionOutcome.PROMPT_ONLY,
                recorded_at=sample_time(),
            ),
            updated_at=sample_time(),
        )
        ephemeral = manager.record_non_durable_memory(
            session_id="telegram:123456",
            candidate=CandidateMemory(
                candidate_id="candidate:session",
                claim_type="ephemeral_context",
                subject_id="subject:user:self",
                scope=ClaimScope.SESSION,
                value={"topic": "next turn only"},
                source_observation_ids=("obs-2",),
            ),
            trace=build_trace(
                candidate_id="candidate:session",
                outcome=AdmissionOutcome.SESSION_EPHEMERAL,
                recorded_at=sample_time(1),
            ),
            updated_at=sample_time(1),
        )

        prompt_only_buffer = manager.list_buffers(
            "telegram:123456",
            buffer_kind=SessionBufferKind.PROMPT_ONLY,
        )
        ephemeral_buffer = manager.list_buffers(
            "telegram:123456",
            buffer_kind=SessionBufferKind.SESSION_EPHEMERAL,
        )

        self.assertEqual(prompt_only.buffer_kind, SessionBufferKind.PROMPT_ONLY)
        self.assertEqual(ephemeral.buffer_kind, SessionBufferKind.SESSION_EPHEMERAL)
        self.assertEqual(prompt_only_buffer[0].payload["candidate_id"], "candidate:prompt")
        self.assertEqual(ephemeral_buffer[0].payload["candidate_id"], "candidate:session")

    def test_peer_specific_memory_mode_gating_is_explicit(self) -> None:
        _, _, manager = self.build_manager(
            config=ContinuityConfig.from_mapping(
                {
                    "memoryMode": {
                        "default": "honcho",
                        "search-only-peer": "off",
                    }
                }
            )
        )

        self.assertTrue(manager.memory_enabled_for_peer("primary-peer"))
        self.assertFalse(manager.memory_enabled_for_peer("search-only-peer"))

        with self.assertRaisesRegex(ValueError, "memory mode"):
            manager.save_turn(
                session_id="telegram:123456",
                message_id="message-1",
                role="user",
                author_subject_id="subject:user:self",
                content="should not persist",
                observed_at=sample_time(),
                peer_name="search-only-peer",
            )


if __name__ == "__main__":
    unittest.main()
