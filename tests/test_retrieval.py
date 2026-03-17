#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sqlite3
import sys
import unittest
from dataclasses import dataclass
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
from continuity.context_builder import (
    ContinuityContextBuilder,
    SubjectResolutionStatus,
)
from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.forgetting import (
    ForgettingDecisionTrace,
    ForgettingMode,
    ForgettingOperation,
    ForgettingTarget,
    ForgettingTargetKind,
    forgetting_rule_for,
)
from continuity.index.zvec_index import InMemoryZvecBackend
from continuity.ontology import MemoryPartition
from continuity.outcomes import OutcomeTarget
from continuity.resolution_queue import (
    ResolutionPriority,
    ResolutionQueueItem,
    ResolutionSource,
    ResolutionSurface,
)
from continuity.service import (
    ContinuityServiceFacade,
    ServiceOperation,
    ServiceRequest,
)
from continuity.epistemics import resolve_locus_belief
from continuity.store.belief_revision import BeliefStateRepository, StoredBeliefState
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
    SubjectAlias,
    SubjectKind,
)
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import (
    SQLiteRepository,
    SessionBufferRecord,
    SessionMessageRecord,
    SessionRecord,
    StoredDisclosurePolicy,
)
from continuity.utility import CompiledUtilityWeight
from continuity.views import ViewKind


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


@dataclass(frozen=True, slots=True)
class FakeEmbeddingBatch:
    model: str
    dimensions: int
    fingerprint: str
    embeddings: tuple[tuple[float, ...], ...]


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.model = "nomic-embed-text"
        self.fingerprint = "embedding:fake_nomic_embed_text@1"

    def embed(self, inputs: str | tuple[str, ...]) -> FakeEmbeddingBatch:
        texts = (inputs,) if isinstance(inputs, str) else tuple(inputs)
        return FakeEmbeddingBatch(
            model=self.model,
            dimensions=4,
            fingerprint=self.fingerprint,
            embeddings=tuple(self._vectorize(text) for text in texts),
        )

    def _vectorize(self, text: str) -> tuple[float, ...]:
        tokens = tuple(re.findall(r"[a-z]+", text.casefold()))
        return (
            float(sum(token in {"espresso", "coffee"} for token in tokens)),
            float(sum(token in {"tea", "green"} for token in tokens)),
            float(sum(token == "oolong" for token in tokens)),
            float(sum(token in {"profile", "belief", "timeline", "evidence"} for token in tokens)),
        )


def assistant_context(
    *,
    channel: DisclosureChannel,
    purpose: DisclosurePurpose,
    principal: DisclosurePrincipal = DisclosurePrincipal.CURRENT_USER,
) -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.ASSISTANT,
            viewer_subject_id="subject:assistant:hermes",
            active_user_id="subject:user:alice",
            active_peer_id="subject:peer:alice",
        ),
        audience_principal=principal,
        channel=channel,
        purpose=purpose,
        policy_stamp="hermes_v1@1.0.0",
    )


def peer_context(
    *,
    viewer_subject_id: str,
    active_peer_id: str,
    channel: DisclosureChannel,
    purpose: DisclosurePurpose,
) -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.PEER,
            viewer_subject_id=viewer_subject_id,
            active_peer_id=active_peer_id,
        ),
        audience_principal=DisclosurePrincipal.CURRENT_PEER,
        channel=channel,
        purpose=purpose,
        policy_stamp="hermes_v1@1.0.0",
    )


def build_admission_trace(
    *,
    candidate_id: str,
    claim_type: str,
    outcome: AdmissionOutcome,
    recorded_at: datetime,
    budget_window_key: str | None = None,
) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=outcome,
            recorded_at=recorded_at,
            rationale="retrieval test admission",
        ),
        claim_type=claim_type,
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type=claim_type,
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH,
            salience=AdmissionStrength.MEDIUM,
            rationale="retrieval test admission",
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
            window_key=budget_window_key or f"budget:{candidate_id}",
            limit=8,
            used=0,
        ),
    )


def save_weight(
    connection: sqlite3.Connection,
    *,
    target_id: str,
    weighted_score: int,
    signal_counts: dict[str, int],
) -> None:
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
            OutcomeTarget.CLAIM.value,
            target_id,
            "hermes_v1@1.0.0",
            weighted_score,
            json.dumps(signal_counts, sort_keys=True),
            json.dumps([f"outcome:{target_id}"]),
        ),
    )


def seed_default_policies(repository: SQLiteRepository) -> None:
    for policy_id, principal, channel, purpose in (
        ("assistant_internal", "assistant_internal", "prompt", "prompt"),
        ("current_user", "current_user", "search", "search"),
        ("current_peer", "current_peer", "search", "search"),
        ("shared_session", "shared_session", "prompt", "prompt"),
    ):
        repository.save_disclosure_policy(
            StoredDisclosurePolicy(
                policy_id=policy_id,
                audience_principal=principal,
                channel=channel,
                purpose=purpose,
                exposure_mode="direct",
                redaction_mode="none",
                capture_for_replay=True,
            )
        )


def seed_retrieval_state(connection: sqlite3.Connection) -> None:
    repository = SQLiteRepository(connection)
    beliefs = BeliefStateRepository(connection)
    seed_default_policies(repository)

    subjects = (
        Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
            aliases=(
                SubjectAlias(
                    alias="Ace Example",
                    alias_type="nickname",
                    source_observation_ids=("obs-alias",),
                ),
                SubjectAlias(
                    alias="Ally Example",
                    alias_type="nickname",
                    source_observation_ids=("obs-alias",),
                ),
            ),
        ),
        Subject(
            subject_id="subject:user:alicia",
            kind=SubjectKind.USER,
            canonical_name="Alicia Example",
            aliases=(
                SubjectAlias(
                    alias="Ally Example",
                    alias_type="nickname",
                    source_observation_ids=("obs-alias-2",),
                ),
            ),
        ),
        Subject(
            subject_id="subject:peer:alice",
            kind=SubjectKind.PEER,
            canonical_name="Peer Alice",
        ),
        Subject(
            subject_id="subject:peer:bob",
            kind=SubjectKind.PEER,
            canonical_name="Peer Bob",
        ),
    )
    for subject in subjects:
        repository.save_subject(subject, created_at=sample_time())

    repository.save_session(
        SessionRecord(
            session_id="session:hermes:test",
            host_namespace="hermes",
            session_name="Hermes Retrieval Test",
            recall_mode="hybrid",
            write_frequency="default",
            created_at=sample_time(),
        )
    )

    messages = (
        SessionMessageRecord(
            message_id="message-espresso",
            session_id="session:hermes:test",
            role="user",
            author_subject_id="subject:user:alice",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
        ),
        SessionMessageRecord(
            message_id="message-languages",
            session_id="session:hermes:test",
            role="user",
            author_subject_id="subject:user:alice",
            content="I write Python and Rust most days.",
            observed_at=sample_time(2),
        ),
        SessionMessageRecord(
            message_id="message-task-1",
            session_id="session:hermes:test",
            role="user",
            author_subject_id="subject:user:alice",
            content="The project is drafted.",
            observed_at=sample_time(3),
        ),
        SessionMessageRecord(
            message_id="message-task-2",
            session_id="session:hermes:test",
            role="user",
            author_subject_id="subject:user:alice",
            content="The project is now under review.",
            observed_at=sample_time(4),
        ),
        SessionMessageRecord(
            message_id="message-peer-tea",
            session_id="session:hermes:test",
            role="peer",
            author_subject_id="subject:peer:alice",
            content="I prefer green tea.",
            observed_at=sample_time(5),
        ),
        SessionMessageRecord(
            message_id="message-oolong",
            session_id="session:hermes:test",
            role="user",
            author_subject_id="subject:user:alice",
            content="I secretly switched to oolong.",
            observed_at=sample_time(6),
        ),
    )
    for message in messages:
        repository.save_message(message)

    observations = (
        Observation(
            observation_id="obs-espresso",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
            metadata={"message_id": "message-espresso"},
        ),
        Observation(
            observation_id="obs-languages",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content="I write Python and Rust most days.",
            observed_at=sample_time(2),
            metadata={"message_id": "message-languages"},
        ),
        Observation(
            observation_id="obs-task-1",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content="The project is drafted.",
            observed_at=sample_time(3),
            metadata={"message_id": "message-task-1"},
        ),
        Observation(
            observation_id="obs-task-2",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content="The project is now under review.",
            observed_at=sample_time(4),
            metadata={"message_id": "message-task-2"},
        ),
        Observation(
            observation_id="obs-peer-tea",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:peer:alice",
            content="I prefer green tea.",
            observed_at=sample_time(5),
            metadata={"message_id": "message-peer-tea"},
        ),
        Observation(
            observation_id="obs-oolong",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content="I secretly switched to oolong.",
            observed_at=sample_time(6),
            metadata={"message_id": "message-oolong"},
        ),
    )
    for observation in observations:
        repository.save_observation(observation, message_id=observation.metadata["message_id"])

    connection.execute(
        """
        INSERT INTO snapshots(snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction)
        VALUES (?, ?, ?, ?)
        """,
        ("snapshot-1", "hermes_v1@1.0.0", None, "publish_snapshot"),
    )

    def claim_from_candidate(
        *,
        claim_id: str,
        candidate_id: str,
        claim_type: str,
        subject_id: str,
        scope: ClaimScope,
        value: dict[str, object],
        observation_id: str,
        disclosure_policy: str,
        locus_key: str,
        aggregation_mode: AggregationMode,
        observed_at: datetime,
        learned_at: datetime,
        valid_to: datetime | None = None,
        relations: tuple[ClaimRelation, ...] = (),
    ) -> Claim:
        candidate = CandidateMemory(
            candidate_id=candidate_id,
            claim_type=claim_type,
            subject_id=subject_id,
            scope=scope,
            value=value,
            source_observation_ids=(observation_id,),
        )
        repository.save_candidate_memory(candidate, created_at=learned_at)
        admission_trace = build_admission_trace(
            candidate_id=candidate_id,
            claim_type=claim_type,
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            recorded_at=learned_at,
        )
        repository.admissions.record_decision(admission_trace)
        return Claim.from_candidate(
            claim_id=claim_id,
            candidate=candidate,
            admission=admission_trace.decision,
            locus=MemoryLocus(
                subject_id=subject_id,
                locus_key=locus_key,
                scope=scope,
                default_disclosure_policy=disclosure_policy,
                conflict_set_key=locus_key.replace("/", "."),
                aggregation_mode=aggregation_mode,
            ),
            provenance=ClaimProvenance(observation_ids=(observation_id,)),
            disclosure_policy=disclosure_policy,
            observed_at=observed_at,
            learned_at=learned_at,
            valid_to=valid_to,
            relations=relations,
        )

    claims = (
        claim_from_candidate(
            claim_id="claim-latte",
            candidate_id="candidate-latte",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"drink": "latte"},
            observation_id="obs-espresso",
            disclosure_policy="current_user",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            observed_at=sample_time(1),
            learned_at=sample_time(1),
            valid_to=sample_time(2),
        ),
        claim_from_candidate(
            claim_id="claim-espresso",
            candidate_id="candidate-espresso",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"drink": "espresso"},
            observation_id="obs-espresso",
            disclosure_policy="current_user",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            observed_at=sample_time(1),
            learned_at=sample_time(3),
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.SUPERSEDES,
                    related_claim_id="claim-latte",
                ),
            ),
        ),
        claim_from_candidate(
            claim_id="claim-python",
            candidate_id="candidate-python",
            claim_type="biography.languages",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"language": "Python"},
            observation_id="obs-languages",
            disclosure_policy="current_user",
            locus_key="biography/languages",
            aggregation_mode=AggregationMode.SET_UNION,
            observed_at=sample_time(2),
            learned_at=sample_time(2),
        ),
        claim_from_candidate(
            claim_id="claim-rust",
            candidate_id="candidate-rust",
            claim_type="biography.languages",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"language": "Rust"},
            observation_id="obs-languages",
            disclosure_policy="current_user",
            locus_key="biography/languages",
            aggregation_mode=AggregationMode.SET_UNION,
            observed_at=sample_time(2),
            learned_at=sample_time(3),
        ),
        claim_from_candidate(
            claim_id="claim-task-draft",
            candidate_id="candidate-task-draft",
            claim_type="task_state.project_status",
            subject_id="subject:user:alice",
            scope=ClaimScope.SHARED,
            value={"status": "drafted"},
            observation_id="obs-task-1",
            disclosure_policy="shared_session",
            locus_key="task_state/project_status",
            aggregation_mode=AggregationMode.TIMELINE,
            observed_at=sample_time(3),
            learned_at=sample_time(3),
        ),
        claim_from_candidate(
            claim_id="claim-task-review",
            candidate_id="candidate-task-review",
            claim_type="task_state.project_status",
            subject_id="subject:user:alice",
            scope=ClaimScope.SHARED,
            value={"status": "reviewing"},
            observation_id="obs-task-2",
            disclosure_policy="shared_session",
            locus_key="task_state/project_status",
            aggregation_mode=AggregationMode.TIMELINE,
            observed_at=sample_time(4),
            learned_at=sample_time(4),
        ),
        claim_from_candidate(
            claim_id="claim-peer-tea",
            candidate_id="candidate-peer-tea",
            claim_type="preference.favorite_drink",
            subject_id="subject:peer:alice",
            scope=ClaimScope.PEER,
            value={"drink": "green tea"},
            observation_id="obs-peer-tea",
            disclosure_policy="current_peer",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            observed_at=sample_time(5),
            learned_at=sample_time(5),
        ),
        claim_from_candidate(
            claim_id="claim-oolong",
            candidate_id="candidate-oolong",
            claim_type="preference.favorite_drink",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"drink": "oolong"},
            observation_id="obs-oolong",
            disclosure_policy="current_user",
            locus_key="preference/favorite_drink",
            aggregation_mode=AggregationMode.LATEST_WINS,
            observed_at=sample_time(6),
            learned_at=sample_time(6),
        ),
    )
    for claim in claims:
        repository.save_claim(claim)

    beliefs.record_state(
        StoredBeliefState(
            belief_id="belief-preference",
            policy_stamp="hermes_v1@1.0.0",
            projection=resolve_locus_belief(
                tuple(
                    claim
                    for claim in claims
                    if claim.subject_id == "subject:user:alice"
                    and claim.locus.locus_key == "preference/favorite_drink"
                ),
                as_of=sample_time(7),
            ),
            as_of=sample_time(7),
        )
    )
    beliefs.record_state(
        StoredBeliefState(
            belief_id="belief-languages",
            policy_stamp="hermes_v1@1.0.0",
            projection=resolve_locus_belief(
                tuple(
                    claim
                    for claim in claims
                    if claim.subject_id == "subject:user:alice"
                    and claim.locus.locus_key == "biography/languages"
                ),
                as_of=sample_time(7),
            ),
            as_of=sample_time(7),
        )
    )
    beliefs.record_state(
        StoredBeliefState(
            belief_id="belief-task-status",
            policy_stamp="hermes_v1@1.0.0",
            projection=resolve_locus_belief(
                tuple(
                    claim
                    for claim in claims
                    if claim.subject_id == "subject:user:alice"
                    and claim.locus.locus_key == "task_state/project_status"
                ),
                as_of=sample_time(7),
            ),
            as_of=sample_time(7),
        )
    )
    beliefs.record_state(
        StoredBeliefState(
            belief_id="belief-peer-tea",
            policy_stamp="hermes_v1@1.0.0",
            projection=resolve_locus_belief(
                tuple(
                    claim
                    for claim in claims
                    if claim.subject_id == "subject:peer:alice"
                    and claim.locus.locus_key == "preference/favorite_drink"
                ),
                as_of=sample_time(7),
            ),
            as_of=sample_time(7),
        )
    )

    prompt_candidate = CandidateMemory(
        candidate_id="candidate-prompt-only",
        claim_type="ephemeral_context",
        subject_id="subject:user:alice",
        scope=ClaimScope.SESSION,
        value={"note": "Alice is comparing espresso machines this week."},
        source_observation_ids=("obs-espresso",),
    )
    repository.save_candidate_memory(prompt_candidate, created_at=sample_time(7))
    repository.admissions.record_decision(
        build_admission_trace(
            candidate_id=prompt_candidate.candidate_id,
            claim_type=prompt_candidate.claim_type,
            outcome=AdmissionOutcome.PROMPT_ONLY,
            recorded_at=sample_time(7),
        )
    )

    repository.save_session_buffer(
        SessionBufferRecord(
            buffer_key="buffer:session:hermes:test:ephemeral",
            session_id="session:hermes:test",
            buffer_kind="session_ephemeral",
            payload={"text": "Last tool run timed out once before succeeding."},
            updated_at=sample_time(8),
        )
    )

    repository.resolution_queue.enqueue_item(
        ResolutionQueueItem(
            item_id="queue-clarify-milk",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="preference/milk",
            rationale="Ask whether Alice still prefers oat milk.",
            created_at=sample_time(9),
            utility_boost=4,
            surfaces=(ResolutionSurface.PROMPT_QUEUE, ResolutionSurface.HOST_API),
        )
    )

    repository.forgetting.record_operation(
        ForgettingDecisionTrace(
            operation=ForgettingOperation(
                operation_id="forget-oolong",
                target=ForgettingTarget(
                    target_kind=ForgettingTargetKind.CLAIM,
                    target_id="claim-oolong",
                ),
                mode=ForgettingMode.SUPPRESS,
                requested_by="subject:user:alice",
                rationale="withdraw secret preference from host-facing reads",
                policy_stamp="hermes_v1@1.0.0",
                recorded_at=sample_time(10),
            ),
            rule=forgetting_rule_for(ForgettingMode.SUPPRESS),
        )
    )

    save_weight(
        connection,
        target_id="claim-espresso",
        weighted_score=9,
        signal_counts={"answer_citation": 1, "prompt_inclusion": 1},
    )
    save_weight(
        connection,
        target_id="claim-python",
        weighted_score=4,
        signal_counts={"prompt_inclusion": 1},
    )
    connection.commit()


class ContinuityRetrievalTests(unittest.TestCase):
    def build_builder(self) -> tuple[ContinuityContextBuilder, sqlite3.Connection]:
        connection = open_memory_database()
        seed_retrieval_state(connection)
        builder = ContinuityContextBuilder(
            connection=connection,
            embedding_client=FakeEmbeddingClient(),
            vector_backend=InMemoryZvecBackend(),
            policy_name="hermes_v1",
            prompt_hard_token_budget=15,
            prompt_soft_token_budgets={"evidence": 3},
        )
        return builder, connection

    def test_subject_resolution_reports_unique_and_ambiguous_matches(self) -> None:
        builder, connection = self.build_builder()
        self.addCleanup(connection.close)

        unique = builder.resolve_subject_reference("Ace Example")
        ambiguous = builder.resolve_subject_reference("Ally Example")

        self.assertEqual(unique.status, SubjectResolutionStatus.RESOLVED)
        self.assertEqual(unique.subject_id, "subject:user:alice")
        self.assertEqual(unique.matched_by, "alias")

        self.assertEqual(ambiguous.status, SubjectResolutionStatus.AMBIGUOUS)
        self.assertEqual(
            ambiguous.candidate_subject_ids,
            ("subject:user:alice", "subject:user:alicia"),
        )

    def test_search_filters_forgotten_and_cross_peer_results_and_returns_explicit_view_hits(self) -> None:
        builder, connection = self.build_builder()
        self.addCleanup(connection.close)

        espresso_hits = builder.search(
            query_text="espresso",
            disclosure_context=assistant_context(
                channel=DisclosureChannel.SEARCH,
                purpose=DisclosurePurpose.SEARCH,
            ),
            target_snapshot_id="snapshot-1",
            limit=4,
        )
        oolong_hits = builder.search(
            query_text="oolong",
            disclosure_context=assistant_context(
                channel=DisclosureChannel.SEARCH,
                purpose=DisclosurePurpose.SEARCH,
            ),
            target_snapshot_id="snapshot-1",
            limit=4,
        )
        blocked_peer_hits = builder.search(
            query_text="green tea",
            disclosure_context=peer_context(
                viewer_subject_id="subject:peer:bob",
                active_peer_id="subject:peer:alice",
                channel=DisclosureChannel.SEARCH,
                purpose=DisclosurePurpose.SEARCH,
            ),
            target_snapshot_id="snapshot-1",
            limit=4,
        )

        self.assertTrue(espresso_hits)
        self.assertEqual(espresso_hits[0].view.compiled_view.kind, ViewKind.STATE)
        self.assertEqual(espresso_hits[0].view.payload["subject_id"], "subject:user:alice")
        self.assertFalse(oolong_hits)
        self.assertFalse(blocked_peer_hits)

    def test_view_assembly_covers_state_timeline_set_profile_and_evidence(self) -> None:
        builder, connection = self.build_builder()
        self.addCleanup(connection.close)
        context = assistant_context(
            channel=DisclosureChannel.PROFILE,
            purpose=DisclosurePurpose.PROFILE,
        )

        state_view = builder.build_state_view(
            subject_id="subject:user:alice",
            locus_key="preference/favorite_drink",
            disclosure_context=context,
            target_snapshot_id="snapshot-1",
        )
        set_view = builder.build_set_view(
            subject_id="subject:user:alice",
            locus_key="biography/languages",
            disclosure_context=context,
            target_snapshot_id="snapshot-1",
        )
        timeline_view = builder.build_timeline_view(
            subject_id="subject:user:alice",
            locus_key="task_state/project_status",
            disclosure_context=assistant_context(
                channel=DisclosureChannel.ANSWER,
                purpose=DisclosurePurpose.ANSWER,
                principal=DisclosurePrincipal.SHARED_SESSION,
            ),
            target_snapshot_id="snapshot-1",
        )
        profile_view = builder.build_profile_view(
            subject_id="subject:user:alice",
            disclosure_context=context,
            target_snapshot_id="snapshot-1",
        )
        evidence_view = builder.build_evidence_view(
            target_kind="claim",
            target_id="claim-espresso",
            disclosure_context=assistant_context(
                channel=DisclosureChannel.EVIDENCE,
                purpose=DisclosurePurpose.EVIDENCE,
            ),
            target_snapshot_id="snapshot-1",
        )

        self.assertEqual(state_view.compiled_view.kind, ViewKind.STATE)
        self.assertEqual(state_view.payload["active_values"], ("espresso",))
        self.assertEqual(state_view.compiled_view.claim_ids[0], "claim-espresso")

        self.assertEqual(set_view.compiled_view.kind, ViewKind.SET)
        self.assertEqual(set(set_view.payload["items"]), {"Python", "Rust"})

        self.assertEqual(timeline_view.compiled_view.kind, ViewKind.TIMELINE)
        self.assertEqual(
            tuple(entry["value"] for entry in timeline_view.payload["entries"]),
            ("drafted", "reviewing"),
        )

        self.assertEqual(profile_view.compiled_view.kind, ViewKind.PROFILE)
        self.assertIn("preference/favorite_drink", profile_view.payload["locus_keys"])
        self.assertIn("biography/languages", profile_view.payload["locus_keys"])

        self.assertEqual(evidence_view.compiled_view.kind, ViewKind.EVIDENCE)
        self.assertEqual(evidence_view.compiled_view.observation_ids, ("obs-espresso",))
        self.assertIn("I prefer espresso over tea.", evidence_view.payload["observations"][0]["content"])

    def test_prompt_view_uses_budgeted_fragments_and_exposes_auxiliary_prompt_memory(self) -> None:
        builder, connection = self.build_builder()
        self.addCleanup(connection.close)

        prompt_view = builder.build_prompt_view(
            session_id="session:hermes:test",
            disclosure_context=assistant_context(
                channel=DisclosureChannel.PROMPT,
                purpose=DisclosurePurpose.PROMPT,
            ),
            target_snapshot_id="snapshot-1",
            recall_mode="hybrid",
        )

        self.assertEqual(prompt_view.compiled_view.kind, ViewKind.PROMPT)
        self.assertEqual(
            prompt_view.payload["prompt_plan"]["included_fragment_ids"],
            (
                "state:subject:user:alice:preference/favorite_drink",
                "profile:subject:user:alice",
                "evidence:claim:claim-espresso",
            ),
        )
        self.assertEqual(
            prompt_view.payload["prompt_plan"]["excluded_fragments"],
            {
                "timeline:subject:user:alice:task_state/project_status": "hard_budget_exhausted",
            },
        )
        self.assertEqual(
            prompt_view.payload["prompt_plan"]["degradation_reasons"],
            {"evidence:claim:claim-espresso": "compressed evidence"},
        )
        self.assertEqual(
            prompt_view.payload["auxiliary_prompt_memory"],
            (
                {
                    "entry_kind": "prompt_only",
                    "entry_id": "candidate-prompt-only",
                    "text": "Alice is comparing espresso machines this week.",
                },
                {
                    "entry_kind": "session_ephemeral",
                    "entry_id": "buffer:session:hermes:test:ephemeral",
                    "text": "Last tool run timed out once before succeeding.",
                },
            ),
        )
        self.assertEqual(
            prompt_view.payload["follow_ups"],
            (
                {
                    "item_id": "queue-clarify-milk",
                    "subject_id": "subject:user:alice",
                    "locus_key": "preference/milk",
                    "rationale": "Ask whether Alice still prefers oat milk.",
                },
            ),
        )

    def test_service_executors_expose_transport_neutral_retrieval_payloads(self) -> None:
        builder, connection = self.build_builder()
        self.addCleanup(connection.close)
        facade = ContinuityServiceFacade(builder.service_executors())

        search_response = facade.execute(
            ServiceRequest(
                operation=ServiceOperation.SEARCH,
                request_id="request:search",
                payload={"query_text": "espresso", "limit": 2},
                disclosure_context=assistant_context(
                    channel=DisclosureChannel.SEARCH,
                    purpose=DisclosurePurpose.SEARCH,
                ),
                target_snapshot_id="snapshot-1",
            )
        )
        state_response = facade.execute(
            ServiceRequest(
                operation=ServiceOperation.GET_STATE_VIEW,
                request_id="request:state",
                payload={"view_key": "state:subject:user:alice:preference/favorite_drink"},
                disclosure_context=assistant_context(
                    channel=DisclosureChannel.PROFILE,
                    purpose=DisclosurePurpose.PROFILE,
                ),
                target_snapshot_id="snapshot-1",
            )
        )
        subject_response = facade.execute(
            ServiceRequest(
                operation=ServiceOperation.RESOLVE_SUBJECT,
                request_id="request:resolve-subject",
                payload={"reference_text": "Ace Example"},
            )
        )

        self.assertEqual(search_response.payload["results"][0]["view_kind"], "state")
        self.assertEqual(state_response.payload["view"]["view_kind"], "state")
        self.assertEqual(subject_response.payload["status"], "resolved")
        self.assertEqual(subject_response.payload["subject_id"], "subject:user:alice")


if __name__ == "__main__":
    unittest.main()
