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

import continuity.resolution_queue as resolution_queue_module
from continuity.resolution_queue import (
    ResolutionAction,
    ResolutionPriority,
    ResolutionQueueItem,
    ResolutionRecord,
    ResolutionSource,
    ResolutionStatus,
    ResolutionSurface,
)
from continuity.store.claims import ClaimScope, SubjectKind
from continuity.store.schema import apply_migrations


ResolutionQueueRepository = getattr(
    resolution_queue_module,
    "ResolutionQueueRepository",
    None,
)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def seed_session(connection: sqlite3.Connection, *, session_id: str) -> None:
    connection.execute(
        """
        INSERT INTO sessions(
            session_id,
            host_namespace,
            session_name,
            recall_mode,
            write_frequency,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            "hermes",
            "Session",
            "balanced",
            "default",
            sample_time().isoformat(),
        ),
    )


def seed_subject(
    connection: sqlite3.Connection,
    *,
    subject_id: str,
    kind: SubjectKind,
    canonical_name: str,
) -> None:
    connection.execute(
        """
        INSERT INTO subjects(subject_id, kind, canonical_name, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            subject_id,
            kind.value,
            canonical_name,
            sample_time().isoformat(),
        ),
    )


def seed_candidate(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    subject_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO candidate_memories(
            candidate_id,
            claim_type,
            subject_id,
            scope,
            value_json,
            source_observation_ids_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            "preference.favorite_drink",
            subject_id,
            ClaimScope.USER.value,
            '{"value":"espresso"}',
            "[]",
            sample_time().isoformat(),
        ),
    )


class ResolutionQueueRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_items_and_returns_priority_ordered_queries(self) -> None:
        self.assertIsNotNone(ResolutionQueueRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_subject(
            connection,
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice",
        )
        seed_subject(
            connection,
            subject_id="subject:peer:hermes",
            kind=SubjectKind.PEER,
            canonical_name="Hermes",
        )
        seed_candidate(
            connection,
            candidate_id="candidate-1",
            subject_id="subject:user:alice",
        )
        repository = ResolutionQueueRepository(connection)

        highest_utility = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            rationale="direct confirmation is needed before durable promotion",
            created_at=sample_time(),
            utility_boost=4,
            surfaces=(ResolutionSurface.HOST_API, ResolutionSurface.PROMPT_QUEUE),
            candidate_id="candidate-1",
        )
        same_priority_lower_utility = ResolutionQueueItem(
            item_id="queue-2",
            source=ResolutionSource.NEEDS_FOLLOWUP,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="project/current_status",
            rationale="follow-up is still pending",
            created_at=sample_time(1),
            utility_boost=1,
            surfaces=(ResolutionSurface.HOST_API,),
        )
        lower_priority = ResolutionQueueItem(
            item_id="queue-3",
            source=ResolutionSource.STALE_ON_USE,
            priority=ResolutionPriority.NORMAL,
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            rationale="recent stale-on-use signal should stay visible",
            created_at=sample_time(2),
            utility_boost=9,
            surfaces=(ResolutionSurface.HOST_API,),
        )
        future_deferred = ResolutionQueueItem(
            item_id="queue-4",
            source=ResolutionSource.OPEN_QUESTION,
            priority=ResolutionPriority.IMMEDIATE,
            subject_id="subject:user:alice",
            locus_key="travel/future_plan",
            rationale="wait for more evidence before surfacing again",
            created_at=sample_time(3),
            status=ResolutionStatus.DEFERRED,
            deferred_until=sample_time(60),
            surfaces=(ResolutionSurface.HOST_API,),
        )
        prompt_only = ResolutionQueueItem(
            item_id="queue-5",
            source=ResolutionSource.CONFLICTED_LOCUS,
            priority=ResolutionPriority.IMMEDIATE,
            subject_id="subject:peer:hermes",
            locus_key="relationship/status",
            rationale="surface only in prompt assembly",
            created_at=sample_time(4),
            surfaces=(ResolutionSurface.PROMPT_QUEUE,),
        )

        repository.enqueue_item(highest_utility)
        repository.enqueue_item(same_priority_lower_utility)
        repository.enqueue_item(lower_priority)
        repository.enqueue_item(future_deferred)
        repository.enqueue_item(prompt_only)

        self.assertEqual(repository.read_item("queue-1"), highest_utility)
        self.assertIsNone(repository.read_item("missing"))
        self.assertEqual(
            tuple(
                item.item_id
                for item in repository.list_items(
                    subject_id="subject:user:alice",
                    surface=ResolutionSurface.HOST_API,
                    at_time=sample_time(10),
                )
            ),
            ("queue-1", "queue-2", "queue-3", "queue-4"),
        )
        self.assertEqual(
            tuple(
                item.item_id
                for item in repository.list_items(
                    subject_id="subject:user:alice",
                    surface=ResolutionSurface.HOST_API,
                    at_time=sample_time(10),
                    actionable_only=True,
                    limit=2,
                )
            ),
            ("queue-1", "queue-2"),
        )
        self.assertEqual(
            tuple(
                item.item_id
                for item in repository.list_items(
                    surface=ResolutionSurface.PROMPT_QUEUE,
                    at_time=sample_time(10),
                )
            ),
            ("queue-5", "queue-1"),
        )

    def test_repository_supports_defer_batch_escalate_and_resolve_operations(self) -> None:
        self.assertIsNotNone(ResolutionQueueRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_subject(
            connection,
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice",
        )
        repository = ResolutionQueueRepository(connection)

        queued_item = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.NEEDS_FOLLOWUP,
            priority=ResolutionPriority.NORMAL,
            subject_id="subject:user:alice",
            locus_key="project/current_status",
            rationale="pending follow-up remains open",
            created_at=sample_time(),
            surfaces=(ResolutionSurface.HOST_API,),
        )
        discard_item = ResolutionQueueItem(
            item_id="queue-2",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="preference/coffee",
            rationale="confirm before durable promotion",
            created_at=sample_time(1),
            surfaces=(ResolutionSurface.HOST_API,),
        )
        repository.enqueue_item(queued_item)
        repository.enqueue_item(discard_item)

        repository.defer_item("queue-1", until=sample_time(30))
        repository.assign_batch("queue-1", batch_key="batch-42")
        repository.escalate_priority(
            "queue-1",
            priority=ResolutionPriority.IMMEDIATE,
            utility_boost=5,
        )
        repository.record_resolution(
            ResolutionRecord(
                item_id="queue-1",
                action=ResolutionAction.KEEP_EPHEMERAL,
                rationale="the memory should remain session-scoped only",
                recorded_at=sample_time(31),
            )
        )
        repository.record_resolution(
            ResolutionRecord(
                item_id="queue-2",
                action=ResolutionAction.DISCARD,
                rationale="the candidate was noise",
                recorded_at=sample_time(32),
            )
        )

        resolved_item = repository.read_item("queue-1")
        dropped_item = repository.read_item("queue-2")

        self.assertIsNotNone(resolved_item)
        self.assertIsNotNone(dropped_item)
        self.assertEqual(resolved_item.priority, ResolutionPriority.IMMEDIATE)
        self.assertEqual(resolved_item.utility_boost, 5)
        self.assertEqual(resolved_item.batch_key, "batch-42")
        self.assertEqual(resolved_item.status, ResolutionStatus.RESOLVED)
        self.assertEqual(dropped_item.status, ResolutionStatus.DROPPED)
        self.assertEqual(
            repository.list_records(item_id="queue-1"),
            (
                ResolutionRecord(
                    item_id="queue-1",
                    action=ResolutionAction.KEEP_EPHEMERAL,
                    rationale="the memory should remain session-scoped only",
                    recorded_at=sample_time(31),
                ),
            ),
        )
        self.assertEqual(
            repository.list_records(item_id="queue-2"),
            (
                ResolutionRecord(
                    item_id="queue-2",
                    action=ResolutionAction.DISCARD,
                    rationale="the candidate was noise",
                    recorded_at=sample_time(32),
                ),
            ),
        )

    def test_repository_persists_session_and_provenance_links(self) -> None:
        self.assertIsNotNone(ResolutionQueueRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_session(connection, session_id="session-1")
        seed_subject(
            connection,
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice",
        )
        repository = ResolutionQueueRepository(connection)

        queued_item = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            session_id="session-1",
            locus_key="preference/favorite_drink",
            rationale="confirm the preference before durable promotion",
            created_at=sample_time(),
            surfaces=(ResolutionSurface.HOST_API,),
            claim_ids=("claim-1", "claim-1", "claim-2"),
            observation_ids=("obs-1", "obs-1"),
            outcome_ids=("outcome-1",),
        )

        repository.enqueue_item(queued_item)

        self.assertEqual(repository.read_item("queue-1"), queued_item)
        self.assertEqual(
            repository.list_items(
                session_id="session-1",
                surface=ResolutionSurface.HOST_API,
                at_time=sample_time(10),
            ),
            (queued_item,),
        )


if __name__ == "__main__":
    unittest.main()
