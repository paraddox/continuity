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
import continuity.utility as utility_module
from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeTarget
from continuity.policy import hermes_v1_policy_pack
from continuity.store.claims import SubjectKind
from continuity.store.schema import apply_migrations
from continuity.utility import (
    UtilitySignal,
    compile_utility_weight,
    utility_events_for_outcome,
)


OutcomeRepository = getattr(outcomes_module, "OutcomeRepository", None)
UtilityRepository = getattr(utility_module, "UtilityRepository", None)


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


def outcome_record(*, outcome_id: str, label: OutcomeLabel, recorded_at: datetime) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=outcome_id,
        label=label,
        target=OutcomeTarget.CLAIM,
        target_id="claim-1",
        policy_stamp="hermes_v1@1.0.0",
        recorded_at=recorded_at,
        rationale=f"{label.value} recorded for utility persistence",
        actor_subject_id="subject:user:alice",
        claim_ids=("claim-1",),
    )


class UtilityRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_events_and_compiled_weights(self) -> None:
        self.assertIsNotNone(OutcomeRepository)
        self.assertIsNotNone(UtilityRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_subject(
            connection,
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice",
        )
        outcome_repository = OutcomeRepository(connection)
        repository = UtilityRepository(connection)

        outcomes = (
            outcome_record(
                outcome_id="outcome-1",
                label=OutcomeLabel.PROMPT_INCLUDED,
                recorded_at=sample_time(),
            ),
            outcome_record(
                outcome_id="outcome-2",
                label=OutcomeLabel.ANSWER_CITED,
                recorded_at=sample_time(5),
            ),
        )
        for outcome in outcomes:
            outcome_repository.record_outcome(outcome)

        events = tuple(
            event
            for outcome in outcomes
            for event in utility_events_for_outcome(outcome)
        )
        repository.record_events(events)
        compiled = compile_utility_weight(
            target=OutcomeTarget.CLAIM,
            target_id="claim-1",
            policy=hermes_v1_policy_pack(),
            events=events,
        )
        repository.write_compiled_weight(compiled)

        self.assertEqual(
            tuple(event.signal for event in repository.list_events(target=OutcomeTarget.CLAIM)),
            (UtilitySignal.ANSWER_CITATION, UtilitySignal.PROMPT_INCLUSION),
        )
        self.assertEqual(
            repository.list_events(source_outcome_id="outcome-1"),
            (events[0],),
        )
        self.assertEqual(
            repository.read_compiled_weight(
                target=OutcomeTarget.CLAIM,
                target_id="claim-1",
                policy_stamp="hermes_v1@1.0.0",
            ),
            compiled,
        )
        self.assertEqual(
            repository.list_compiled_weights(target=OutcomeTarget.CLAIM),
            (compiled,),
        )


if __name__ == "__main__":
    unittest.main()
