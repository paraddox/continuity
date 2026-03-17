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
    import continuity.store.sqlite as sqlite_store_module
except ModuleNotFoundError:
    sqlite_store_module = None

from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.forgetting import (
    ForgettingDecisionTrace,
    ForgettingMode,
    ForgettingOperation,
    ForgettingTarget,
    ForgettingTargetKind,
    forgetting_rule_for,
)
from continuity.ontology import MemoryPartition
from continuity.outcomes import OutcomeTarget
from continuity.resolution_queue import (
    ResolutionPriority,
    ResolutionQueueItem,
    ResolutionSource,
    ResolutionSurface,
)
from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimScope,
    MemoryLocus,
    Observation,
    Subject,
    SubjectAlias,
    SubjectKind,
)
from continuity.store.schema import apply_migrations
from continuity.utility import CompiledUtilityWeight, UtilitySignal


SQLiteRepository = getattr(sqlite_store_module, "SQLiteRepository", None)
SessionRecord = getattr(sqlite_store_module, "SessionRecord", None)
SessionMessageRecord = getattr(sqlite_store_module, "SessionMessageRecord", None)
StoredDisclosurePolicy = getattr(sqlite_store_module, "StoredDisclosurePolicy", None)


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
    outcome: AdmissionOutcome,
    budget_partition: MemoryPartition = MemoryPartition.USER_MEMORY,
    budget_window_key: str = "session:telegram:123",
    budget_limit: int = 3,
    budget_used: int = 0,
    recorded_at: datetime,
) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=outcome,
            recorded_at=recorded_at,
            rationale="stable explicit preference",
        ),
        claim_type=claim_type,
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type=claim_type,
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH,
            salience=AdmissionStrength.MEDIUM,
            rationale="stable explicit preference",
            utility_signals=("prompt_inclusion",),
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


class SQLiteRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(SQLiteRepository)
        self.assertIsNotNone(SessionRecord)
        self.assertIsNotNone(SessionMessageRecord)
        self.assertIsNotNone(StoredDisclosurePolicy)

    def test_repository_round_trips_core_subject_session_observation_and_claim_state(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = SQLiteRepository(connection)

        disclosure_policy = StoredDisclosurePolicy(
            policy_id="assistant_internal",
            audience_principal="assistant_internal",
            channel="prompt",
            purpose="prompt",
            exposure_mode="direct",
            redaction_mode="none",
            capture_for_replay=True,
        )
        repository.save_disclosure_policy(disclosure_policy)

        subject = Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
            aliases=(
                SubjectAlias(
                    alias=" Ally  Example ",
                    alias_type="nickname",
                    source_observation_ids=("obs-1",),
                ),
            ),
        )
        repository.save_subject(subject, created_at=sample_time())

        session = SessionRecord(
            session_id="telegram:123",
            host_namespace="hermes",
            session_name="Telegram Alice",
            recall_mode="balanced",
            write_frequency="default",
            created_at=sample_time(),
        )
        repository.save_session(session)

        message = SessionMessageRecord(
            message_id="message-1",
            session_id=session.session_id,
            role="user",
            author_subject_id=subject.subject_id,
            content="I still prefer espresso.",
            observed_at=sample_time(1),
            metadata={"source": "telegram"},
        )
        repository.save_message(message)

        observation = Observation(
            observation_id="obs-1",
            source_kind="session_message",
            session_id=session.session_id,
            author_subject_id=subject.subject_id,
            content=message.content,
            observed_at=sample_time(1),
            metadata={"message_id": message.message_id},
        )
        repository.save_observation(observation, message_id=message.message_id)

        candidate = CandidateMemory(
            candidate_id="candidate-1",
            claim_type="preference.favorite_drink",
            subject_id=subject.subject_id,
            scope=ClaimScope.USER,
            value={"drink": "espresso"},
            source_observation_ids=(observation.observation_id,),
        )
        repository.save_candidate_memory(candidate, created_at=sample_time(2))

        admission_trace = build_admission_trace(
            candidate_id=candidate.candidate_id,
            claim_type=candidate.claim_type,
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            recorded_at=sample_time(2),
        )
        repository.admissions.record_decision(admission_trace)

        locus = MemoryLocus(
            subject_id=subject.subject_id,
            locus_key="preference/favorite_drink",
            scope=ClaimScope.USER,
            default_disclosure_policy="assistant_internal",
            conflict_set_key="preference.favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
        )
        stored_locus = repository.save_memory_locus(locus)

        claim = Claim.from_candidate(
            claim_id="claim-1",
            candidate=candidate,
            admission=admission_trace.decision,
            locus=locus,
            provenance=ClaimProvenance(observation_ids=(observation.observation_id,)),
            disclosure_policy="assistant_internal",
            observed_at=observation.observed_at,
            learned_at=sample_time(3),
        )
        repository.save_claim(claim)

        self.assertEqual(repository.read_disclosure_policy("assistant_internal"), disclosure_policy)
        self.assertEqual(repository.read_subject(subject.subject_id), subject)
        self.assertEqual(repository.resolve_subject("ally example"), subject)
        self.assertEqual(repository.resolve_subject("Alice Example", kind=SubjectKind.USER), subject)
        self.assertEqual(repository.read_session(session.session_id), session)
        self.assertEqual(repository.list_messages(session_id=session.session_id), (message,))
        self.assertEqual(
            repository.list_observations(session_id=session.session_id),
            (observation,),
        )
        self.assertEqual(repository.read_candidate_memory(candidate.candidate_id), candidate)
        self.assertEqual(stored_locus.locus, locus)
        self.assertEqual(
            repository.read_memory_locus(subject.subject_id, locus.locus_key),
            stored_locus,
        )
        self.assertEqual(repository.read_claim(claim.claim_id), claim)
        self.assertEqual(
            repository.list_claims(subject_id=subject.subject_id),
            (claim,),
        )

    def test_repository_exposes_cross_cutting_lookup_helpers(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = SQLiteRepository(connection)

        disclosure_policy = StoredDisclosurePolicy(
            policy_id="assistant_internal",
            audience_principal="assistant_internal",
            channel="prompt",
            purpose="prompt",
            exposure_mode="direct",
            redaction_mode="none",
            capture_for_replay=True,
        )
        repository.save_disclosure_policy(disclosure_policy)

        subject = Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
        )
        repository.save_subject(subject, created_at=sample_time())

        candidate = CandidateMemory(
            candidate_id="candidate-1",
            claim_type="preference.favorite_drink",
            subject_id=subject.subject_id,
            scope=ClaimScope.USER,
            value={"drink": "espresso"},
            source_observation_ids=("obs-1",),
        )
        repository.save_candidate_memory(candidate, created_at=sample_time(1))

        admission_trace = build_admission_trace(
            candidate_id=candidate.candidate_id,
            claim_type=candidate.claim_type,
            outcome=AdmissionOutcome.NEEDS_CONFIRMATION,
            recorded_at=sample_time(2),
        )
        repository.admissions.record_decision(admission_trace)

        queue_item = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id=subject.subject_id,
            locus_key="preference/favorite_drink",
            rationale="confirm the durable preference before publishing",
            created_at=sample_time(3),
            utility_boost=2,
            surfaces=(ResolutionSurface.HOST_API,),
            candidate_id=candidate.candidate_id,
        )
        repository.resolution_queue.enqueue_item(queue_item)

        forgetting_target = ForgettingTarget(
            target_kind=ForgettingTargetKind.CLAIM,
            target_id="claim-1",
        )
        forgetting_trace = ForgettingDecisionTrace(
            operation=ForgettingOperation(
                operation_id="forget-1",
                target=forgetting_target,
                mode=ForgettingMode.SUPPRESS,
                requested_by=subject.subject_id,
                rationale="withdraw the claim from normal reads",
                policy_stamp="hermes_v1@1.0.0",
                recorded_at=sample_time(4),
            ),
            rule=forgetting_rule_for(ForgettingMode.SUPPRESS),
        )
        repository.forgetting.record_operation(forgetting_trace)

        expected_weight = CompiledUtilityWeight(
            target=OutcomeTarget.CLAIM,
            target_id="claim-1",
            policy_stamp="hermes_v1@1.0.0",
            weighted_score=6,
            signal_counts=((UtilitySignal.PROMPT_INCLUSION, 2),),
            source_event_ids=("outcome-1", "outcome-2"),
        )
        connection.execute(
            """
            INSERT INTO compiled_utility_weights(
                target,
                target_id,
                policy_stamp,
                weighted_score,
                signal_counts_json,
                source_event_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                expected_weight.target.value,
                expected_weight.target_id,
                expected_weight.policy_stamp,
                expected_weight.weighted_score,
                json.dumps({"prompt_inclusion": 2}),
                json.dumps(["outcome-1", "outcome-2"]),
            ),
        )
        connection.commit()

        self.assertEqual(repository.read_admission_trace(candidate.candidate_id), admission_trace)
        self.assertEqual(
            repository.list_resolution_items(
                subject_id=subject.subject_id,
                surface=ResolutionSurface.HOST_API,
                at_time=sample_time(10),
                actionable_only=True,
            ),
            (queue_item,),
        )
        self.assertEqual(
            repository.current_forgetting_record(forgetting_target),
            repository.forgetting.current_record_for_target(forgetting_target),
        )
        self.assertEqual(repository.read_disclosure_policy(disclosure_policy.policy_id), disclosure_policy)
        self.assertEqual(
            repository.read_compiled_utility_weight(
                target=expected_weight.target,
                target_id=expected_weight.target_id,
                policy_stamp=expected_weight.policy_stamp,
            ),
            expected_weight,
        )


if __name__ == "__main__":
    unittest.main()
