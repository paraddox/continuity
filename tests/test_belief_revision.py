#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import continuity.store.belief_revision as belief_revision_module
except ModuleNotFoundError:
    belief_revision_module = None

import continuity.store.sqlite as sqlite_store_module
from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.epistemics import EpistemicStatus
from continuity.ontology import MemoryPartition
from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimRelation,
    ClaimRelationKind,
    ClaimScope,
    MemoryLocus,
    Observation,
    Subject,
    SubjectKind,
)
from continuity.store.schema import apply_migrations


BeliefRevisionEngine = getattr(belief_revision_module, "BeliefRevisionEngine", None)
BeliefStateRepository = getattr(belief_revision_module, "BeliefStateRepository", None)
SQLiteRepository = getattr(sqlite_store_module, "SQLiteRepository", None)
StoredDisclosurePolicy = getattr(sqlite_store_module, "StoredDisclosurePolicy", None)
SessionRecord = getattr(sqlite_store_module, "SessionRecord", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def build_admission_trace(
    *,
    candidate_id: str,
    claim_type: str,
    recorded_at: datetime,
    used: int = 0,
) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            recorded_at=recorded_at,
            rationale="explicit user statement",
        ),
        claim_type=claim_type,
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type=claim_type,
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH,
            salience=AdmissionStrength.MEDIUM,
            rationale="explicit user statement",
            utility_signals=("prompt_inclusion",),
        ),
        thresholds=AdmissionThresholds(
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.LOW,
        ),
        budget=AdmissionWriteBudget(
            partition=MemoryPartition.USER_MEMORY,
            window_key="session:hermes:test",
            limit=8,
            used=used,
        ),
    )


def ensure_subject(repository: SQLiteRepository, subject_id: str) -> None:
    repository.save_subject(
        Subject(
            subject_id=subject_id,
            kind=SubjectKind.USER,
            canonical_name=subject_id.rsplit(":", maxsplit=1)[-1].title(),
        ),
        created_at=sample_time(),
    )


def ensure_disclosure_policy(repository: SQLiteRepository) -> None:
    repository.save_disclosure_policy(
        StoredDisclosurePolicy(
            policy_id="assistant_internal",
            audience_principal="assistant_internal",
            channel="prompt",
            purpose="prompt",
            exposure_mode="direct",
            redaction_mode="none",
            capture_for_replay=True,
        )
    )


def ensure_session(repository: SQLiteRepository) -> None:
    if repository.read_session("session:hermes:test") is not None:
        return
    repository.save_session(
        SessionRecord(
            session_id="session:hermes:test",
            host_namespace="hermes",
            session_name="Belief Revision Tests",
            recall_mode="balanced",
            write_frequency="default",
            created_at=sample_time(),
        )
    )


def save_claim(
    repository: SQLiteRepository,
    *,
    subject_id: str,
    claim_id: str,
    claim_type: str,
    locus_key: str,
    aggregation_mode: AggregationMode,
    learned_at: datetime,
    value: dict[str, str],
    relations: tuple[ClaimRelation, ...] = (),
    valid_to: datetime | None = None,
) -> Claim:
    ensure_session(repository)
    budget = repository.admissions.read_budget(
        partition=MemoryPartition.USER_MEMORY,
        window_key="session:hermes:test",
    )
    candidate = CandidateMemory(
        candidate_id=f"candidate:{claim_id}",
        claim_type=claim_type,
        subject_id=subject_id,
        scope=ClaimScope.USER,
        value=value,
        source_observation_ids=(f"obs:{claim_id}",),
    )
    repository.save_candidate_memory(candidate, created_at=learned_at)

    trace = build_admission_trace(
        candidate_id=candidate.candidate_id,
        claim_type=claim_type,
        recorded_at=learned_at,
        used=0 if budget is None else budget.used,
    )
    repository.admissions.record_decision(trace)
    repository.save_observation(
        Observation(
            observation_id=f"obs:{claim_id}",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id=subject_id,
            content=json.dumps(value, sort_keys=True),
            observed_at=learned_at,
            metadata={"claim_id": claim_id},
        )
    )

    claim = Claim.from_candidate(
        claim_id=claim_id,
        candidate=candidate,
        admission=trace.decision,
        locus=MemoryLocus(
            subject_id=subject_id,
            locus_key=locus_key,
            scope=ClaimScope.USER,
            default_disclosure_policy="assistant_internal",
            conflict_set_key=locus_key,
            aggregation_mode=aggregation_mode,
        ),
        provenance=ClaimProvenance(
            observation_ids=(f"obs:{claim_id}",),
        ),
        disclosure_policy="assistant_internal",
        observed_at=learned_at,
        learned_at=learned_at,
        valid_from=learned_at,
        valid_to=valid_to,
        relations=relations,
    )
    repository.save_claim(claim)
    return claim


class BeliefRevisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(BeliefRevisionEngine)
        self.assertIsNotNone(BeliefStateRepository)
        self.assertIsNotNone(SQLiteRepository)
        self.assertIsNotNone(StoredDisclosurePolicy)
        self.assertIsNotNone(SessionRecord)

    def test_latest_wins_revision_persists_history_and_current_state(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        claims = SQLiteRepository(connection)
        beliefs = BeliefStateRepository(connection)
        engine = BeliefRevisionEngine(connection)

        ensure_disclosure_policy(claims)
        ensure_subject(claims, "subject:user:alice")

        first = save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-1",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(0),
            value={"drink": "espresso"},
        )
        engine.revise_subject(subject_id="subject:user:alice", as_of=sample_time(1))

        second = save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-2",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(5),
            value={"drink": "tea"},
        )
        revised = engine.revise_subject(subject_id="subject:user:alice", as_of=sample_time(10))

        current = beliefs.read_current_state(
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            policy_stamp="hermes_v1@1.0.0",
        )

        self.assertEqual(len(revised), 1)
        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(current.projection.active_claim_ids, (second.claim_id,))
        self.assertEqual(current.projection.historical_claim_ids, (second.claim_id, first.claim_id))
        self.assertEqual(current.projection.epistemic.status, EpistemicStatus.SUPPORTED)
        self.assertEqual(
            tuple(
                state.as_of
                for state in beliefs.list_states(
                    subject_id="subject:user:alice",
                    locus_key="preference/favorite_drink",
                    policy_stamp="hermes_v1@1.0.0",
                )
            ),
            (sample_time(10), sample_time(1)),
        )

    def test_corrections_and_conflicts_publish_epistemic_results(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        claims = SQLiteRepository(connection)
        beliefs = BeliefStateRepository(connection)
        engine = BeliefRevisionEngine(connection)

        ensure_disclosure_policy(claims)
        ensure_subject(claims, "subject:user:alice")

        save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-1",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(0),
            value={"drink": "espresso"},
        )
        corrected = save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-2",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(5),
            value={"drink": "tea"},
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.CORRECTS,
                    related_claim_id="claim-1",
                ),
            ),
        )
        save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-3",
            claim_type="relationship",
            locus_key="relationship/status",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(2),
            value={"status": "friend"},
        )
        save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-4",
            claim_type="relationship",
            locus_key="relationship/status",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(3),
            value={"status": "coworker"},
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.CONTRADICTS,
                    related_claim_id="claim-3",
                ),
            ),
        )

        engine.revise_subject(subject_id="subject:user:alice", as_of=sample_time(20))

        corrected_state = beliefs.read_current_state(
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            policy_stamp="hermes_v1@1.0.0",
        )
        conflicted_state = beliefs.read_current_state(
            subject_id="subject:user:alice",
            locus_key="relationship/status",
            policy_stamp="hermes_v1@1.0.0",
        )

        self.assertIsNotNone(corrected_state)
        self.assertIsNotNone(conflicted_state)
        assert corrected_state is not None
        assert conflicted_state is not None
        self.assertEqual(corrected_state.projection.active_claim_ids, (corrected.claim_id,))
        self.assertEqual(corrected_state.projection.epistemic.status, EpistemicStatus.SUPPORTED)
        self.assertEqual(conflicted_state.projection.active_claim_ids, ())
        self.assertEqual(conflicted_state.projection.epistemic.status, EpistemicStatus.CONFLICTED)

    def test_revise_all_keeps_subject_boundaries_isolated(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        claims = SQLiteRepository(connection)
        beliefs = BeliefStateRepository(connection)
        engine = BeliefRevisionEngine(connection)

        ensure_disclosure_policy(claims)
        ensure_subject(claims, "subject:user:alice")
        ensure_subject(claims, "subject:user:bob")

        save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-alice",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(0),
            value={"drink": "espresso"},
        )
        save_claim(
            claims,
            subject_id="subject:user:bob",
            claim_id="claim-bob",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(1),
            value={"drink": "oolong"},
        )

        revised = engine.revise_all(as_of=sample_time(5))

        self.assertEqual(len(revised), 2)
        alice_state = beliefs.read_current_state(
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            policy_stamp="hermes_v1@1.0.0",
        )
        bob_state = beliefs.read_current_state(
            subject_id="subject:user:bob",
            locus_key="preference/favorite_drink",
            policy_stamp="hermes_v1@1.0.0",
        )
        self.assertIsNotNone(alice_state)
        self.assertIsNotNone(bob_state)
        assert alice_state is not None
        assert bob_state is not None
        self.assertEqual(alice_state.projection.active_claim_ids, ("claim-alice",))
        self.assertEqual(bob_state.projection.active_claim_ids, ("claim-bob",))

    def test_decay_policy_is_explicit_by_claim_type(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        claims = SQLiteRepository(connection)
        beliefs = BeliefStateRepository(connection)
        engine = BeliefRevisionEngine(connection)

        ensure_disclosure_policy(claims)
        ensure_subject(claims, "subject:user:alice")

        save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-preference",
            claim_type="preference",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(0),
            value={"drink": "espresso"},
            valid_to=sample_time(2),
        )
        save_claim(
            claims,
            subject_id="subject:user:alice",
            claim_id="claim-relationship",
            claim_type="relationship",
            locus_key="relationship/status",
            aggregation_mode=AggregationMode.LATEST_WINS,
            learned_at=sample_time(0),
            value={"status": "friend"},
            valid_to=sample_time(2),
        )

        engine.revise_subject(subject_id="subject:user:alice", as_of=sample_time(20))

        stale_preference = beliefs.read_current_state(
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            policy_stamp="hermes_v1@1.0.0",
        )
        stale_relationship = beliefs.read_current_state(
            subject_id="subject:user:alice",
            locus_key="relationship/status",
            policy_stamp="hermes_v1@1.0.0",
        )

        self.assertIsNotNone(stale_preference)
        self.assertIsNotNone(stale_relationship)
        assert stale_preference is not None
        assert stale_relationship is not None
        self.assertEqual(stale_preference.projection.active_claim_ids, ("claim-preference",))
        self.assertEqual(stale_preference.projection.epistemic.status, EpistemicStatus.STALE)
        self.assertEqual(stale_relationship.projection.active_claim_ids, ())
        self.assertEqual(stale_relationship.projection.epistemic.status, EpistemicStatus.STALE)


if __name__ == "__main__":
    unittest.main()
