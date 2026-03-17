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

import continuity.admission as admission_module
from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.ontology import MemoryPartition
from continuity.store.claims import AdmissionDecision, AdmissionOutcome, ClaimScope, SubjectKind
from continuity.store.schema import apply_migrations


AdmissionRepository = getattr(admission_module, "AdmissionRepository", None)


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


def seed_candidate(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    claim_type: str,
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
            claim_type,
            subject_id,
            ClaimScope.USER.value,
            '{"value":"espresso"}',
            "[]",
            sample_time().isoformat(),
        ),
    )


def build_trace(
    *,
    candidate_id: str,
    claim_type: str,
    outcome: AdmissionOutcome,
    budget_partition: MemoryPartition,
    budget_window_key: str,
    budget_limit: int,
    budget_used: int,
    recorded_at: datetime,
    evidence: AdmissionStrength = AdmissionStrength.HIGH,
    novelty: AdmissionStrength = AdmissionStrength.HIGH,
    stability: AdmissionStrength = AdmissionStrength.HIGH,
    salience: AdmissionStrength = AdmissionStrength.MEDIUM,
    rationale: str = "explicit stable preference",
    utility_signals: tuple[str, ...] = ("prompt_inclusion",),
) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=outcome,
            recorded_at=recorded_at,
            rationale=rationale,
        ),
        claim_type=claim_type,
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type=claim_type,
            evidence=evidence,
            novelty=novelty,
            stability=stability,
            salience=salience,
            rationale=rationale,
            utility_signals=utility_signals,
        ),
        thresholds=AdmissionThresholds(
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.LOW,
        ),
        budget=AdmissionWriteBudget(
            partition=budget_partition,
            window_key=budget_window_key,
            limit=budget_limit,
            used=budget_used,
        ),
    )


class AdmissionRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_decisions_and_exposes_outcome_queries(self) -> None:
        self.assertIsNotNone(AdmissionRepository)

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
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
        )
        seed_candidate(
            connection,
            candidate_id="candidate-2",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
        )
        seed_candidate(
            connection,
            candidate_id="candidate-3",
            claim_type="relationship.status",
            subject_id="subject:peer:hermes",
        )
        repository = AdmissionRepository(connection)

        durable_trace = build_trace(
            candidate_id="candidate-1",
            claim_type="preference.favorite_drink",
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            budget_partition=MemoryPartition.USER_MEMORY,
            budget_window_key="session:telegram:42",
            budget_limit=3,
            budget_used=0,
            recorded_at=sample_time(),
            utility_signals=("prompt_inclusion", "answer_citation"),
        )
        prompt_only_trace = build_trace(
            candidate_id="candidate-2",
            claim_type="preference.favorite_drink",
            outcome=AdmissionOutcome.PROMPT_ONLY,
            budget_partition=MemoryPartition.USER_MEMORY,
            budget_window_key="session:telegram:42",
            budget_limit=3,
            budget_used=1,
            recorded_at=sample_time(1),
            novelty=AdmissionStrength.LOW,
            stability=AdmissionStrength.LOW,
            rationale="useful for the current prompt but not stable enough",
            utility_signals=("prompt_inclusion",),
        )
        needs_confirmation_trace = build_trace(
            candidate_id="candidate-3",
            claim_type="relationship.status",
            outcome=AdmissionOutcome.NEEDS_CONFIRMATION,
            budget_partition=MemoryPartition.SHARED_CONTEXT,
            budget_window_key="peer:subject:peer:hermes",
            budget_limit=2,
            budget_used=0,
            recorded_at=sample_time(2),
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.LOW,
            salience=AdmissionStrength.HIGH,
            rationale="needs direct confirmation before publishing",
            utility_signals=("follow_up_priority",),
        )

        repository.record_decision(durable_trace)
        repository.record_decision(prompt_only_trace)
        repository.record_decision(needs_confirmation_trace)

        self.assertEqual(repository.read_decision("candidate-1"), durable_trace)
        self.assertIsNone(repository.read_decision("missing"))
        self.assertEqual(
            tuple(trace.decision.candidate_id for trace in repository.list_decisions()),
            ("candidate-3", "candidate-2", "candidate-1"),
        )
        self.assertEqual(
            tuple(
                trace.decision.candidate_id
                for trace in repository.list_decisions(subject_id="subject:user:alice")
            ),
            ("candidate-2", "candidate-1"),
        )
        self.assertEqual(
            tuple(
                trace.decision.candidate_id
                for trace in repository.list_decisions(outcome=AdmissionOutcome.PROMPT_ONLY)
            ),
            ("candidate-2",),
        )
        self.assertEqual(
            tuple(
                trace.decision.candidate_id
                for trace in repository.list_durable_promotions()
            ),
            ("candidate-1",),
        )
        self.assertEqual(
            tuple(
                trace.decision.candidate_id
                for trace in repository.list_durable_promotions(
                    partition=MemoryPartition.USER_MEMORY,
                    window_key="session:telegram:42",
                )
            ),
            ("candidate-1",),
        )
        self.assertEqual(
            repository.read_decision("candidate-2").shortfall_fields,
            ("novelty", "stability"),
        )

    def test_repository_tracks_budget_accounting_without_losing_audit_snapshots(self) -> None:
        self.assertIsNotNone(AdmissionRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_subject(
            connection,
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice",
        )
        seed_candidate(
            connection,
            candidate_id="candidate-1",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
        )
        seed_candidate(
            connection,
            candidate_id="candidate-2",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
        )
        seed_candidate(
            connection,
            candidate_id="candidate-3",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
        )
        repository = AdmissionRepository(connection)

        first_durable = build_trace(
            candidate_id="candidate-1",
            claim_type="preference.favorite_drink",
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            budget_partition=MemoryPartition.USER_MEMORY,
            budget_window_key="policy:hermes_v1@1.0.0",
            budget_limit=2,
            budget_used=0,
            recorded_at=sample_time(),
        )
        prompt_only = build_trace(
            candidate_id="candidate-2",
            claim_type="preference.favorite_drink",
            outcome=AdmissionOutcome.PROMPT_ONLY,
            budget_partition=MemoryPartition.USER_MEMORY,
            budget_window_key="policy:hermes_v1@1.0.0",
            budget_limit=2,
            budget_used=1,
            recorded_at=sample_time(1),
            novelty=AdmissionStrength.LOW,
            stability=AdmissionStrength.LOW,
            rationale="useful context but below durable thresholds",
        )
        second_durable = build_trace(
            candidate_id="candidate-3",
            claim_type="preference.favorite_drink",
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            budget_partition=MemoryPartition.USER_MEMORY,
            budget_window_key="policy:hermes_v1@1.0.0",
            budget_limit=2,
            budget_used=1,
            recorded_at=sample_time(2),
        )

        repository.record_decision(first_durable)
        self.assertEqual(
            repository.read_budget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="policy:hermes_v1@1.0.0",
            ),
            AdmissionWriteBudget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="policy:hermes_v1@1.0.0",
                limit=2,
                used=1,
            ),
        )

        repository.record_decision(prompt_only)
        self.assertEqual(
            repository.read_budget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="policy:hermes_v1@1.0.0",
            ),
            AdmissionWriteBudget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="policy:hermes_v1@1.0.0",
                limit=2,
                used=1,
            ),
        )

        repository.record_decision(second_durable)
        self.assertEqual(
            repository.read_budget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="policy:hermes_v1@1.0.0",
            ),
            AdmissionWriteBudget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="policy:hermes_v1@1.0.0",
                limit=2,
                used=2,
            ),
        )
        self.assertEqual(
            repository.read_decision("candidate-1").budget,
            first_durable.budget,
        )
        self.assertEqual(
            tuple(
                trace.decision.candidate_id
                for trace in repository.list_durable_promotions(
                    partition=MemoryPartition.USER_MEMORY,
                    window_key="policy:hermes_v1@1.0.0",
                )
            ),
            ("candidate-3", "candidate-1"),
        )


if __name__ == "__main__":
    unittest.main()
