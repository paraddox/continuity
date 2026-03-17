#!/usr/bin/env python3

from __future__ import annotations

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
from continuity.context_builder import ContinuityContextBuilder
from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.epistemics import resolve_locus_belief
from continuity.index.zvec_index import InMemoryZvecBackend
from continuity.reasoning.base import AnswerQueryRequest, ReasoningAdapter, TextResponse
from continuity.service import ContinuityServiceFacade, ServiceOperation, ServiceRequest
from continuity.ontology import MemoryPartition
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
    SubjectKind,
)
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository
from continuity.store.sqlite import SessionMessageRecord, SessionRecord, StoredDisclosurePolicy
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
        normalized = text.casefold()
        coffee_score = sum(
            normalized.count(token)
            for token in ("coffee", "espresso", "latte", "cappuccino", "drink")
        )
        tea_score = sum(normalized.count(token) for token in ("tea", "green"))
        history_score = sum(
            normalized.count(token)
            for token in ("history", "previous", "formerly", "used to")
        )
        status_score = sum(normalized.count(token) for token in ("stale", "current", "prefer"))
        return (
            float(coffee_score),
            float(tea_score),
            float(history_score),
            float(status_score),
        )


class FakeReasoningAdapter:
    def __init__(self) -> None:
        self.requests: list[AnswerQueryRequest] = []

    def answer_query(self, request: AnswerQueryRequest) -> TextResponse:
        self.requests.append(request)
        if "dana" in request.query.casefold():
            return TextResponse(text="Dana preferred cappuccino.")
        return TextResponse(
            text="Alice currently prefers espresso, though she previously preferred latte."
        )

    def generate_structured(self, request: object) -> object:
        raise AssertionError("not used in dialectic tests")

    def summarize_session(self, request: object) -> object:
        raise AssertionError("not used in dialectic tests")

    def derive_claims(self, request: object) -> object:
        raise AssertionError("not used in dialectic tests")


def assistant_answer_context() -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.ASSISTANT,
            viewer_subject_id="subject:assistant:hermes",
            active_user_id="subject:user:alice",
            active_peer_id="subject:peer:alice",
        ),
        audience_principal=DisclosurePrincipal.CURRENT_USER,
        channel=DisclosureChannel.ANSWER,
        purpose=DisclosurePurpose.ANSWER,
        policy_stamp="hermes_v1@1.0.0",
    )


def cross_peer_answer_context() -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.PEER,
            viewer_subject_id="subject:peer:bob",
            active_peer_id="subject:peer:alice",
        ),
        audience_principal=DisclosurePrincipal.CURRENT_PEER,
        channel=DisclosureChannel.ANSWER,
        purpose=DisclosurePurpose.ANSWER,
        policy_stamp="hermes_v1@1.0.0",
    )


def seed_dialectic_state(connection: sqlite3.Connection) -> None:
    repository = SQLiteRepository(connection)
    beliefs = BeliefStateRepository(connection)

    for policy_id, principal, channel, purpose in (
        ("assistant_internal", "assistant_internal", "prompt", "prompt"),
        ("current_user", "current_user", "answer", "answer"),
        ("current_peer", "current_peer", "answer", "answer"),
        ("shared_session", "shared_session", "answer", "answer"),
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

    for subject in (
        Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
        ),
        Subject(
            subject_id="subject:user:charlie",
            kind=SubjectKind.USER,
            canonical_name="Charlie Example",
        ),
        Subject(
            subject_id="subject:user:dana",
            kind=SubjectKind.USER,
            canonical_name="Dana Example",
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
    ):
        repository.save_subject(subject, created_at=sample_time())

    connection.execute(
        """
        INSERT INTO snapshots(snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction)
        VALUES (?, ?, ?, ?)
        """,
        ("snapshot-1", "hermes_v1@1.0.0", None, "publish_snapshot"),
    )
    repository.save_session(
        SessionRecord(
            session_id="session:hermes:test",
            host_namespace="hermes",
            session_name="Hermes Dialectic Test",
            recall_mode="hybrid",
            write_frequency="turn",
            created_at=sample_time(),
        )
    )

    def save_claim(
        *,
        claim_id: str,
        candidate_id: str,
        claim_type: str,
        subject_id: str,
        disclosure_policy: str,
        locus_key: str,
        aggregation_mode: str,
        value: dict[str, object],
        observation_id: str,
        observed_at: datetime,
        learned_at: datetime,
        valid_to: datetime | None = None,
        relations: tuple[ClaimRelation, ...] = (),
    ) -> Claim:
        candidate = CandidateMemory(
            candidate_id=candidate_id,
            claim_type=claim_type,
            subject_id=subject_id,
            scope=ClaimScope.USER if subject_id.startswith("subject:user:") else ClaimScope.PEER,
            value=value,
            source_observation_ids=(observation_id,),
        )
        repository.save_candidate_memory(candidate, created_at=learned_at)
        repository.admissions.record_decision(
            AdmissionDecisionTrace(
                decision=AdmissionDecision(
                    candidate_id=candidate_id,
                    outcome=AdmissionOutcome.DURABLE_CLAIM,
                    recorded_at=learned_at,
                    rationale="dialectic test admission",
                ),
                claim_type=claim_type,
                policy_stamp="hermes_v1@1.0.0",
                assessment=AdmissionAssessment(
                    claim_type=claim_type,
                    evidence=AdmissionStrength.HIGH,
                    novelty=AdmissionStrength.HIGH,
                    stability=AdmissionStrength.HIGH,
                    salience=AdmissionStrength.MEDIUM,
                    rationale="dialectic test admission",
                    utility_signals=("answer_citation",),
                ),
                thresholds=AdmissionThresholds(
                    evidence=AdmissionStrength.MEDIUM,
                    novelty=AdmissionStrength.MEDIUM,
                    stability=AdmissionStrength.MEDIUM,
                    salience=AdmissionStrength.LOW,
                ),
                budget=AdmissionWriteBudget(
                    partition=MemoryPartition.USER_MEMORY,
                    window_key=f"budget:{candidate_id}",
                    limit=8,
                    used=0,
                ),
            )
        )
        message_id = f"message:{observation_id}"
        repository.save_message(
            SessionMessageRecord(
                message_id=message_id,
                session_id="session:hermes:test",
                role="user",
                author_subject_id=subject_id,
                content=str(next(iter(value.values()))),
                observed_at=observed_at,
            )
        )
        repository.save_observation(
            Observation(
                observation_id=observation_id,
                source_kind="session_message",
                session_id="session:hermes:test",
                author_subject_id=subject_id,
                content=str(next(iter(value.values()))),
                observed_at=observed_at,
                metadata={"message_id": message_id},
            ),
            message_id=message_id,
        )
        claim = Claim.from_candidate(
            claim_id=claim_id,
            candidate=candidate,
            admission=AdmissionDecision(
                candidate_id=candidate_id,
                outcome=AdmissionOutcome.DURABLE_CLAIM,
                recorded_at=learned_at,
                rationale="dialectic test admission",
            ),
            locus=MemoryLocus(
                subject_id=subject_id,
                locus_key=locus_key,
                scope=candidate.scope,
                default_disclosure_policy=disclosure_policy,
                conflict_set_key=locus_key.replace("/", "."),
                aggregation_mode=AggregationMode(aggregation_mode),
            ),
            provenance=ClaimProvenance(observation_ids=(observation_id,)),
            disclosure_policy=disclosure_policy,
            observed_at=observed_at,
            learned_at=learned_at,
            valid_to=valid_to,
            relations=relations,
        )
        repository.save_claim(claim)
        return claim

    latte_claim = save_claim(
        claim_id="claim-latte",
        candidate_id="candidate-latte",
        claim_type="preference.favorite_drink",
        subject_id="subject:user:alice",
        disclosure_policy="current_user",
        locus_key="preference/favorite_drink",
        aggregation_mode="latest_wins",
        value={"drink": "latte"},
        observation_id="obs-latte",
        observed_at=sample_time(1),
        learned_at=sample_time(1),
        valid_to=sample_time(2),
    )
    espresso_claim = save_claim(
        claim_id="claim-espresso",
        candidate_id="candidate-espresso",
        claim_type="preference.favorite_drink",
        subject_id="subject:user:alice",
        disclosure_policy="current_user",
        locus_key="preference/favorite_drink",
        aggregation_mode="latest_wins",
        value={"drink": "espresso"},
        observation_id="obs-espresso",
        observed_at=sample_time(3),
        learned_at=sample_time(3),
        relations=(
            ClaimRelation(
                kind=ClaimRelationKind.SUPERSEDES,
                related_claim_id=latte_claim.claim_id,
            ),
        ),
    )
    peer_claim = save_claim(
        claim_id="claim-peer-green-tea",
        candidate_id="candidate-peer-green-tea",
        claim_type="preference.favorite_drink",
        subject_id="subject:peer:alice",
        disclosure_policy="current_peer",
        locus_key="preference/favorite_drink",
        aggregation_mode="latest_wins",
        value={"drink": "green tea"},
        observation_id="obs-peer-green-tea",
        observed_at=sample_time(4),
        learned_at=sample_time(4),
    )
    stale_claim = save_claim(
        claim_id="claim-dana-cappuccino",
        candidate_id="candidate-dana-cappuccino",
        claim_type="preference.favorite_drink",
        subject_id="subject:user:dana",
        disclosure_policy="current_user",
        locus_key="preference/favorite_drink",
        aggregation_mode="latest_wins",
        value={"drink": "cappuccino"},
        observation_id="obs-dana-cappuccino",
        observed_at=sample_time(1),
        learned_at=sample_time(1),
        valid_to=sample_time(2),
    )

    for belief_id, claims in (
        ("belief-alice-drink", (latte_claim, espresso_claim)),
        ("belief-peer-drink", (peer_claim,)),
        ("belief-dana-drink", (stale_claim,)),
    ):
        beliefs.record_state(
            StoredBeliefState(
                belief_id=belief_id,
                policy_stamp="hermes_v1@1.0.0",
                projection=resolve_locus_belief(claims, as_of=sample_time(6)),
                as_of=sample_time(6),
            )
        )

    connection.commit()


class DialecticAnswerViewTests(unittest.TestCase):
    def build_builder(self) -> tuple[ContinuityContextBuilder, FakeReasoningAdapter, sqlite3.Connection]:
        connection = open_memory_database()
        seed_dialectic_state(connection)
        adapter = FakeReasoningAdapter()
        builder = ContinuityContextBuilder(
            connection=connection,
            embedding_client=FakeEmbeddingClient(),
            vector_backend=InMemoryZvecBackend(),
            policy_name="hermes_v1",
            reasoning_adapter=adapter,
        )
        return builder, adapter, connection

    def test_answer_view_combines_current_belief_history_and_citations(self) -> None:
        builder, adapter, connection = self.build_builder()
        self.addCleanup(connection.close)

        view = builder.build_answer_view(
            question="What do you know about Alice's coffee preferences?",
            subject_id="subject:user:alice",
            disclosure_context=assistant_answer_context(),
            target_snapshot_id="snapshot-1",
        )

        self.assertEqual(view.compiled_view.kind, ViewKind.ANSWER)
        self.assertEqual(view.payload["answer_mode"], "assert")
        self.assertEqual(view.payload["epistemic_status"], "supported")
        self.assertEqual(
            view.payload["answer_text"],
            "Alice currently prefers espresso, though she previously preferred latte.",
        )
        self.assertEqual(
            tuple(item["summary"] for item in view.payload["current_beliefs"]),
            ("Alice Example currently prefers espresso.",),
        )
        self.assertEqual(
            tuple(item["summary"] for item in view.payload["historical_context"]),
            ("Earlier evidence said Alice Example prefers latte.",),
        )
        self.assertEqual(
            tuple(item["claim_id"] for item in view.payload["citations"]),
            ("claim-espresso", "claim-latte"),
        )
        self.assertEqual(
            tuple(item["observation_ids"] for item in view.payload["citations"]),
            (("obs-espresso",), ("obs-latte",)),
        )

        self.assertEqual(len(adapter.requests), 1)
        reasoning_request = adapter.requests[0]
        prompt_context = "\n".join(message.content for message in reasoning_request.messages)
        self.assertIn("Current belief: Alice Example currently prefers espresso.", prompt_context)
        self.assertIn("Superseded history: Earlier evidence said Alice Example prefers latte.", prompt_context)
        self.assertIn("Citations: claim-espresso, claim-latte", prompt_context)

    def test_answer_view_abstains_with_unknown_and_withheld_results(self) -> None:
        builder, adapter, connection = self.build_builder()
        self.addCleanup(connection.close)

        unknown_view = builder.build_answer_view(
            question="What do you know about Charlie's coffee preferences?",
            subject_id="subject:user:charlie",
            disclosure_context=assistant_answer_context(),
            target_snapshot_id="snapshot-1",
        )
        withheld_view = builder.build_answer_view(
            question="What does Peer Alice prefer to drink?",
            subject_id="subject:peer:alice",
            disclosure_context=cross_peer_answer_context(),
            target_snapshot_id="snapshot-1",
        )

        self.assertEqual(unknown_view.payload["answer_mode"], "abstain")
        self.assertEqual(unknown_view.payload["disclosure_result"], "unknown")
        self.assertIn("don't know", unknown_view.payload["answer_text"].casefold())

        self.assertEqual(withheld_view.payload["answer_mode"], "abstain")
        self.assertEqual(withheld_view.payload["disclosure_result"], "withheld")
        self.assertIn("withheld", withheld_view.payload["answer_text"].casefold())

        self.assertEqual(adapter.requests, [])

    def test_answer_view_qualifies_stale_answers(self) -> None:
        builder, adapter, connection = self.build_builder()
        self.addCleanup(connection.close)

        view = builder.build_answer_view(
            question="What coffee does Dana prefer?",
            subject_id="subject:user:dana",
            disclosure_context=assistant_answer_context(),
            target_snapshot_id="snapshot-1",
        )

        self.assertEqual(view.payload["answer_mode"], "qualify")
        self.assertEqual(view.payload["epistemic_status"], "stale")
        self.assertEqual(
            view.payload["answer_text"],
            "This may be outdated, but Dana preferred cappuccino.",
        )
        self.assertEqual(len(adapter.requests), 1)
        prompt_context = "\n".join(message.content for message in adapter.requests[0].messages)
        self.assertIn("Epistemic status: stale.", prompt_context)

    def test_service_executor_exposes_answer_view_payload(self) -> None:
        builder, adapter, connection = self.build_builder()
        self.addCleanup(connection.close)
        facade = ContinuityServiceFacade(builder.service_executors())

        response = facade.execute(
            ServiceRequest(
                operation=ServiceOperation.ANSWER_MEMORY_QUESTION,
                request_id="request:answer",
                payload={
                    "question": "What do you know about Alice's coffee preferences?",
                    "subject_id": "subject:user:alice",
                },
                disclosure_context=assistant_answer_context(),
                target_snapshot_id="snapshot-1",
            )
        )

        self.assertEqual(response.payload["view"]["view_kind"], "answer")
        self.assertEqual(response.payload["view"]["payload"]["answer_mode"], "assert")
        self.assertEqual(len(adapter.requests), 1)


if __name__ == "__main__":
    unittest.main()
