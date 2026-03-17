#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from continuity.index.zvec_index import InMemoryZvecBackend
from continuity.ontology import MemoryPartition
from continuity.resolution_queue import (
    ResolutionPriority,
    ResolutionQueueItem,
    ResolutionSource,
    ResolutionSurface,
)
from continuity.service import ContinuityServiceFacade, ServiceOperation, ServiceRequest
from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    CandidateMemory,
    ClaimScope,
)
from continuity.store.replay import ReplayRepository
from continuity.store.sqlite import SessionBufferRecord, SQLiteRepository
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
)
from tests.test_dialectic import (
    FakeEmbeddingClient,
    FakeReasoningAdapter,
    assistant_answer_context,
    open_memory_database,
    sample_time,
    seed_dialectic_state,
)


def prompt_context() -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.ASSISTANT,
            viewer_subject_id="subject:assistant:hermes",
            active_user_id="subject:user:alice",
            active_peer_id="subject:peer:alice",
        ),
        audience_principal=DisclosurePrincipal.CURRENT_USER,
        channel=DisclosureChannel.PROMPT,
        purpose=DisclosurePurpose.PROMPT,
        policy_stamp="hermes_v1@1.0.0",
    )


def seed_snapshot_boundary(connection: sqlite3.Connection, *, snapshot_id: str) -> None:
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
            17,
            "snapshot_head_promotion",
            "publish_snapshot",
            "publish_snapshot",
            f"[\"{snapshot_id}\"]",
            sample_time(10).isoformat(),
            "head:active",
            "snapshot_published",
        ),
    )
    connection.execute(
        """
        INSERT INTO system_events(
            journal_position,
            event_type,
            transaction_kind,
            arbiter_lane_position,
            payload_mode,
            recorded_at,
            object_ids_json,
            inline_payload_json,
            reference_ids_json,
            waterline
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            21,
            "snapshot_published",
            "publish_snapshot",
            17,
            "reference",
            sample_time(10).isoformat(),
            f"[\"{snapshot_id}\"]",
            "[]",
            f"[\"{snapshot_id}\"]",
            "snapshot_published",
        ),
    )
    connection.commit()


def build_prompt_only_trace(*, candidate_id: str) -> AdmissionDecisionTrace:
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate_id,
            outcome=AdmissionOutcome.PROMPT_ONLY,
            recorded_at=sample_time(7),
            rationale="keep near-term comparison context in prompt memory only",
        ),
        claim_type="ephemeral_context",
        policy_stamp="hermes_v1@1.0.0",
        assessment=AdmissionAssessment(
            claim_type="ephemeral_context",
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.MEDIUM,
            rationale="prompt-only context remains useful for the active session",
            utility_signals=("prompt_inclusion",),
        ),
        thresholds=AdmissionThresholds(
            evidence=AdmissionStrength.LOW,
            novelty=AdmissionStrength.LOW,
            stability=AdmissionStrength.LOW,
            salience=AdmissionStrength.LOW,
        ),
        budget=AdmissionWriteBudget(
            partition=MemoryPartition.USER_MEMORY,
            window_key="budget:prompt-only",
            limit=8,
            used=0,
        ),
    )


def seed_prompt_context(connection: sqlite3.Connection) -> None:
    repository = SQLiteRepository(connection)

    prompt_candidate = CandidateMemory(
        candidate_id="candidate-prompt-only",
        claim_type="ephemeral_context",
        subject_id="subject:user:alice",
        scope=ClaimScope.SESSION,
        value={"note": "Alice is comparing espresso machines this week."},
        source_observation_ids=("obs-espresso",),
    )
    repository.save_candidate_memory(prompt_candidate, created_at=sample_time(7))
    repository.admissions.record_decision(build_prompt_only_trace(candidate_id=prompt_candidate.candidate_id))

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
            candidate_id=prompt_candidate.candidate_id,
        )
    )


@dataclass(frozen=True, slots=True)
class BuilderHarness:
    builder: ContinuityContextBuilder
    facade: ContinuityServiceFacade
    replay: ReplayRepository
    adapter: FakeReasoningAdapter
    connection: sqlite3.Connection


def build_harness() -> BuilderHarness:
    connection = open_memory_database()
    seed_dialectic_state(connection)
    seed_snapshot_boundary(connection, snapshot_id="snapshot-1")
    seed_prompt_context(connection)

    adapter = FakeReasoningAdapter()
    builder = ContinuityContextBuilder(
        connection=connection,
        embedding_client=FakeEmbeddingClient(),
        vector_backend=InMemoryZvecBackend(),
        reasoning_adapter=adapter,
    )
    return BuilderHarness(
        builder=builder,
        facade=ContinuityServiceFacade(builder.service_executors()),
        replay=ReplayRepository(connection),
        adapter=adapter,
        connection=connection,
    )


class TurnArtifactCaptureTests(unittest.TestCase):
    def test_answer_view_capture_persists_retrieval_reasoning_and_inspection_payloads(self) -> None:
        harness = build_harness()
        self.addCleanup(harness.connection.close)

        response = harness.facade.execute(
            ServiceRequest(
                operation=ServiceOperation.ANSWER_MEMORY_QUESTION,
                request_id="request:answer-turn",
                payload={
                    "question": "What do you know about Alice's coffee preferences?",
                    "subject_id": "subject:user:alice",
                },
                disclosure_context=assistant_answer_context(),
                target_snapshot_id="snapshot-1",
            )
        )

        self.assertEqual(response.replay_artifact_ids, ("replay:request:answer-turn",))

        artifact = harness.replay.read_artifact("replay:request:answer-turn")
        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact.version, "turn_decision_v1")
        self.assertEqual(artifact.source_transaction, TransactionKind.PUBLISH_SNAPSHOT)
        self.assertEqual(artifact.source_waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(artifact.phase_boundary, TransactionPhase.PUBLISH_SNAPSHOT)
        payload = artifact.decision_payload
        self.assertEqual(
            artifact.source_object_ids,
            ("snapshot-1", payload["selection"]["view_key"]),
        )
        self.assertEqual(payload["surface"], "answer_view")
        self.assertEqual(payload["boundary"]["snapshot_id"], "snapshot-1")
        self.assertEqual(payload["boundary"]["journal_position"], 21)
        self.assertEqual(payload["boundary"]["arbiter_lane_position"], 17)
        self.assertEqual(payload["retrieval"]["query_text"], "What do you know about Alice's coffee preferences?")
        self.assertEqual(payload["retrieval"]["subject_id"], "subject:user:alice")
        self.assertEqual(
            tuple(candidate["view_key"] for candidate in payload["retrieval"]["candidates"]),
            (
                "state:subject:user:alice:preference/favorite_drink",
                "timeline:subject:user:alice:preference/favorite_drink",
                "evidence:observation:obs-espresso",
                "evidence:observation:obs-latte",
            ),
        )
        self.assertEqual(
            payload["selection"]["source_compiled_views"][0]["view_key"],
            "state:subject:user:alice:preference/favorite_drink",
        )
        self.assertEqual(
            tuple(payload["selection"]["selected_claim_ids"]),
            ("claim-espresso", "claim-latte"),
        )
        self.assertEqual(payload["reasoning"]["adapter"], "fake_reasoning_adapter")
        self.assertEqual(
            payload["reasoning"]["response_text"],
            "Alice currently prefers espresso, though she previously preferred latte.",
        )
        self.assertEqual(
            payload["utility_events"],
            (
                {"kind": "answer_citation", "claim_id": "claim-espresso"},
                {"kind": "answer_citation", "claim_id": "claim-latte"},
            ),
        )

        inspection = harness.facade.execute(
            ServiceRequest(
                operation=ServiceOperation.INSPECT_TURN_DECISION,
                request_id="request:inspect-turn",
                payload={"artifact_id": "replay:request:answer-turn"},
            )
        )
        self.assertEqual(inspection.payload["artifact"]["artifact_id"], "replay:request:answer-turn")
        self.assertEqual(inspection.payload["run"]["run_id"], "run:request:answer-turn")
        self.assertEqual(inspection.payload["bundle"]["bundle_id"], "bundle:request:answer-turn")

    def test_prompt_view_capture_persists_prompt_packing_admission_and_follow_up_decisions(self) -> None:
        harness = build_harness()
        self.addCleanup(harness.connection.close)

        response = harness.facade.execute(
            ServiceRequest(
                operation=ServiceOperation.GET_PROMPT_VIEW,
                request_id="request:prompt-turn",
                payload={"view_key": "prompt:session:hermes:test"},
                disclosure_context=prompt_context(),
                target_snapshot_id="snapshot-1",
            )
        )

        self.assertEqual(response.replay_artifact_ids, ("replay:request:prompt-turn",))

        artifact = harness.replay.read_artifact("replay:request:prompt-turn")
        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact.version, "turn_decision_v1")
        self.assertEqual(artifact.source_transaction, TransactionKind.PUBLISH_SNAPSHOT)
        self.assertEqual(artifact.source_waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(artifact.phase_boundary, TransactionPhase.PUBLISH_SNAPSHOT)

        payload = artifact.decision_payload
        self.assertEqual(payload["surface"], "prompt_view")
        self.assertEqual(payload["boundary"]["snapshot_id"], "snapshot-1")
        self.assertEqual(payload["selection"]["view_key"], "prompt:session:hermes:test")
        self.assertEqual(
            tuple(source_view["view_key"] for source_view in payload["selection"]["source_compiled_views"][:3]),
            (
                "state:subject:user:alice:preference/favorite_drink",
                "profile:subject:user:alice",
                "evidence:claim:claim-espresso",
            ),
        )
        self.assertEqual(
            payload["admission"]["decisions"],
            (
                {
                    "candidate_id": "candidate-prompt-only",
                    "claim_type": "ephemeral_context",
                    "outcome": "prompt_only",
                    "policy_stamp": "hermes_v1@1.0.0",
                    "rationale": "keep near-term comparison context in prompt memory only",
                    "utility_signals": ("prompt_inclusion",),
                },
            ),
        )
        self.assertEqual(
            payload["resolution_queue"]["surfaced_items"],
            (
                {
                    "item_id": "queue-clarify-milk",
                    "subject_id": "subject:user:alice",
                    "locus_key": "preference/milk",
                    "rationale": "Ask whether Alice still prefers oat milk.",
                },
            ),
        )
        self.assertEqual(
            tuple(fragment["fragment_id"] for fragment in payload["prompt_packing"]["included_fragments"][:3]),
            (
                "state:subject:user:alice:preference/favorite_drink",
                "profile:subject:user:alice",
                "evidence:claim:claim-espresso",
            ),
        )
        self.assertIn("candidate-prompt-only", tuple(payload["admission"]["candidate_ids"]))
        self.assertIn("queue-clarify-milk", tuple(event["item_id"] for event in payload["utility_events"] if event["kind"] == "resolution_follow_up"))


if __name__ == "__main__":
    unittest.main()
