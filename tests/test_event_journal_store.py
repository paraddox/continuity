#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import continuity.events as events_module
from continuity.arbiter import ArbiterPublication, ArbiterPublicationKind
from continuity.events import EventPayloadMode, SystemEvent, SystemEventType
from continuity.store.schema import apply_migrations
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
)


SystemEventJournal = getattr(events_module, "SystemEventJournal", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def seed_publication(connection: sqlite3.Connection, publication: ArbiterPublication) -> None:
    connection.execute(
        """
        INSERT INTO arbiter_publications(
            lane_position,
            publication_kind,
            transaction_kind,
            phase,
            object_ids_json,
            published_at,
            snapshot_head_id,
            reached_waterline
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            publication.lane_position,
            publication.publication_kind.value,
            publication.transaction_kind.value,
            publication.phase.value,
            str(list(publication.object_ids)).replace("'", '"'),
            publication.published_at.isoformat(),
            publication.snapshot_head_id,
            publication.reached_waterline.value if publication.reached_waterline else None,
        ),
    )
    connection.commit()


class SystemEventJournalTests(unittest.TestCase):
    def test_journal_round_trips_append_only_events_and_reconstruction_queries(self) -> None:
        self.assertIsNotNone(SystemEventJournal)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        journal = SystemEventJournal(connection)

        claim_publication = ArbiterPublication(
            lane_position=3,
            publication_kind=ArbiterPublicationKind.CLAIM_COMMIT,
            transaction_kind=TransactionKind.INGEST_TURN,
            phase=TransactionPhase.COMMIT_CLAIMS,
            object_ids=("claim:1", "observation:1"),
            published_at=sample_time(),
            reached_waterline=DurabilityWaterline.CLAIM_COMMITTED,
        )
        snapshot_publication = ArbiterPublication(
            lane_position=4,
            publication_kind=ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,
            transaction_kind=TransactionKind.PUBLISH_SNAPSHOT,
            phase=TransactionPhase.PUBLISH_SNAPSHOT,
            object_ids=("snapshot:candidate:1", "snapshot:active"),
            published_at=sample_time(1),
            snapshot_head_id="snapshot:active",
            reached_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )
        seed_publication(connection, claim_publication)
        seed_publication(connection, snapshot_publication)

        claim_event = SystemEvent.from_publication(
            journal_position=7,
            event_type=SystemEventType.CLAIM_COMMITTED,
            publication=claim_publication,
            payload_mode=EventPayloadMode.MIXED,
            recorded_at=sample_time(),
            inline_payload=("claim:1",),
            reference_ids=("observation:1",),
        )
        snapshot_event = SystemEvent.from_publication(
            journal_position=8,
            event_type=SystemEventType.SNAPSHOT_PUBLISHED,
            publication=snapshot_publication,
            payload_mode=EventPayloadMode.REFERENCE,
            recorded_at=sample_time(1),
            reference_ids=("snapshot:active",),
        )

        journal.append(claim_event)
        journal.append(snapshot_event)

        self.assertEqual(journal.read_event(7), claim_event)
        self.assertEqual(journal.next_position(), 9)
        self.assertEqual(
            journal.list_events(),
            (claim_event, snapshot_event),
        )
        self.assertEqual(
            journal.reconstruct(object_id="claim:1"),
            (claim_event,),
        )
        self.assertEqual(
            journal.reconstruct(from_position=8),
            (snapshot_event,),
        )

    def test_journal_rejects_mutating_existing_rows(self) -> None:
        self.assertIsNotNone(SystemEventJournal)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        journal = SystemEventJournal(connection)

        publication = ArbiterPublication(
            lane_position=2,
            publication_kind=ArbiterPublicationKind.OUTCOME_RECORDING,
            transaction_kind=TransactionKind.WRITE_CONCLUSION,
            phase=TransactionPhase.CAPTURE_REPLAY,
            object_ids=("outcome:1",),
            published_at=sample_time(),
        )
        seed_publication(connection, publication)

        event = SystemEvent.from_publication(
            journal_position=5,
            event_type=SystemEventType.OUTCOME_RECORDED,
            publication=publication,
            payload_mode=EventPayloadMode.INLINE,
            recorded_at=sample_time(),
            inline_payload=("user_confirmed",),
        )

        journal.append(event)

        with self.assertRaises(ValueError):
            journal.append(
                replace(
                    event,
                    inline_payload=("user_corrected",),
                )
            )


if __name__ == "__main__":
    unittest.main()
