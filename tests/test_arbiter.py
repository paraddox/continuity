#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import continuity.arbiter as arbiter_module
from continuity.arbiter import ArbiterPublicationKind
from continuity.events import EventPayloadMode, SystemEventType
from continuity.store.schema import apply_migrations
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
)


MutationArbiter = getattr(arbiter_module, "MutationArbiter", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    apply_migrations(connection)
    return connection


class MutationArbiterTests(unittest.TestCase):
    def test_concurrent_publish_attempts_resolve_through_one_ordered_lane(self) -> None:
        self.assertIsNotNone(MutationArbiter)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        arbiter = MutationArbiter(connection)
        start_gate = threading.Barrier(5)

        def publish(index: int):
            start_gate.wait()
            return arbiter.publish(
                publication_kind=ArbiterPublicationKind.CLAIM_COMMIT,
                transaction_kind=TransactionKind.INGEST_TURN,
                phase=TransactionPhase.COMMIT_CLAIMS,
                object_ids=(f"claim:{index}",),
                published_at=sample_time(index),
                reached_waterline=DurabilityWaterline.CLAIM_COMMITTED,
                payload_mode=EventPayloadMode.MIXED,
                inline_payload=(f"claim:{index}",),
                reference_ids=(f"observation:{index}",),
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(publish, index) for index in range(4)]
            start_gate.wait()
            results = tuple(future.result() for future in futures)

        self.assertEqual(
            sorted(result.publication.lane_position for result in results),
            [1, 2, 3, 4],
        )
        self.assertEqual(
            sorted(result.event.journal_position for result in results),
            [1, 2, 3, 4],
        )
        self.assertEqual(
            tuple(publication.lane_position for publication in arbiter.list_publications()),
            (1, 2, 3, 4),
        )
        self.assertEqual(
            tuple(event.arbiter_lane_position for event in arbiter.journal.list_events()),
            (1, 2, 3, 4),
        )
        self.assertEqual(
            {event.event_type for event in arbiter.journal.list_events()},
            {SystemEventType.CLAIM_COMMITTED},
        )

    def test_snapshot_head_promotion_and_waterline_signal_publish_with_matching_event(self) -> None:
        self.assertIsNotNone(MutationArbiter)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        arbiter = MutationArbiter(connection)

        observation_record = arbiter.publish(
            publication_kind=ArbiterPublicationKind.OBSERVATION_COMMIT,
            transaction_kind=TransactionKind.INGEST_TURN,
            phase=TransactionPhase.COMMIT_OBSERVATIONS,
            object_ids=("observation:1",),
            published_at=sample_time(),
            reached_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
            payload_mode=EventPayloadMode.REFERENCE,
            reference_ids=("observation:1",),
        )
        snapshot_record = arbiter.publish(
            publication_kind=ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,
            transaction_kind=TransactionKind.PUBLISH_SNAPSHOT,
            phase=TransactionPhase.PUBLISH_SNAPSHOT,
            object_ids=("snapshot:candidate:1", "snapshot:active"),
            published_at=sample_time(1),
            snapshot_head_id="snapshot:active",
            reached_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
            payload_mode=EventPayloadMode.REFERENCE,
            reference_ids=("snapshot:active",),
        )

        self.assertEqual(observation_record.publication.lane_position, 1)
        self.assertEqual(snapshot_record.publication.lane_position, 2)
        self.assertEqual(snapshot_record.event.event_type, SystemEventType.SNAPSHOT_PUBLISHED)
        self.assertEqual(
            arbiter.read_publication(2),
            snapshot_record.publication,
        )
        self.assertEqual(
            arbiter.journal.reconstruct(object_id="snapshot:active"),
            (snapshot_record.event,),
        )


if __name__ == "__main__":
    unittest.main()
