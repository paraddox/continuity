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

import continuity.outcomes as outcomes_module
from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeTarget
from continuity.store.claims import SubjectKind
from continuity.store.schema import apply_migrations


OutcomeRepository = getattr(outcomes_module, "OutcomeRepository", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


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


class OutcomeRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_records_and_supports_target_queries(self) -> None:
        self.assertIsNotNone(OutcomeRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_subject(
            connection,
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice",
        )
        repository = OutcomeRepository(connection)

        first = OutcomeRecord(
            outcome_id="outcome-1",
            label=OutcomeLabel.PROMPT_INCLUDED,
            target=OutcomeTarget.CLAIM,
            target_id="claim-1",
            policy_stamp="hermes_v1@1.0.0",
            recorded_at=sample_time(),
            rationale="the claim was packed into the prompt",
            actor_subject_id="subject:user:alice",
            claim_ids=("claim-1",),
            observation_ids=("obs-1", "obs-1"),
        )
        second = OutcomeRecord(
            outcome_id="outcome-2",
            label=OutcomeLabel.USER_CONFIRMED,
            target=OutcomeTarget.RESOLUTION_QUEUE_ITEM,
            target_id="queue-1",
            policy_stamp="hermes_v1@1.0.0",
            recorded_at=sample_time(5),
            rationale="the queued follow-up was confirmed",
            actor_subject_id="subject:user:alice",
            capture_for_replay=False,
        )

        repository.record_outcome(first)
        repository.record_outcome(second)

        self.assertEqual(repository.read_record("outcome-1"), first)
        self.assertIsNone(repository.read_record("missing"))
        self.assertEqual(
            tuple(record.outcome_id for record in repository.list_records()),
            ("outcome-2", "outcome-1"),
        )
        self.assertEqual(
            repository.list_records(
                target=OutcomeTarget.CLAIM,
                target_id="claim-1",
            ),
            (first,),
        )
        self.assertEqual(
            repository.list_records(label=OutcomeLabel.USER_CONFIRMED),
            (second,),
        )
        self.assertEqual(
            repository.list_records(actor_subject_id="subject:user:alice", limit=1),
            (second,),
        )


if __name__ == "__main__":
    unittest.main()
