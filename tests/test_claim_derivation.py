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

try:
    import continuity.reasoning.claim_derivation as claim_derivation_module
except ModuleNotFoundError:
    claim_derivation_module = None

from continuity.config import ContinuityConfig
from continuity.forgetting import (
    ForgettingDecisionTrace,
    ForgettingMode,
    ForgettingOperation,
    ForgettingSurface,
    ForgettingTarget,
    ForgettingTargetKind,
    ForgettingTombstone,
    forgetting_rule_for,
)
from continuity.index.zvec_index import InMemoryZvecBackend, ZvecIndex
from continuity.ontology import MemoryPartition
from continuity.reasoning.base import ClaimDerivationRequest, RawStructuredOutput
from continuity.session_manager import SessionBufferKind, SessionManager
from continuity.snapshots import (
    MemorySnapshot,
    SnapshotArtifactKind,
    SnapshotArtifactRef,
    SnapshotHead,
    SnapshotHeadState,
    SnapshotRepository,
)
from continuity.store.belief_revision import BeliefStateRepository
from continuity.store.claims import Observation, Subject, SubjectKind
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository, SessionRecord, StoredDisclosurePolicy
from continuity.transactions import TransactionKind


ClaimDerivationPipeline = getattr(claim_derivation_module, "ClaimDerivationPipeline", None)
SchemaValidatedClaimDerivationSchema = getattr(
    claim_derivation_module,
    "ClaimDerivationEnvelopeSchema",
    None,
)
fingerprint_candidate_content = getattr(
    claim_derivation_module,
    "fingerprint_candidate_content",
    None,
)


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
            fingerprint=self.fingerprint,
            embeddings=tuple((float(len(text.split())), 1.0, 0.0, 0.0) for text in texts),
        )


class FakeReasoningAdapter:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.requests: list[ClaimDerivationRequest] = []

    def answer_query(self, request: object) -> object:
        raise AssertionError("not used in claim derivation tests")

    def generate_structured(self, request: object) -> object:
        raise AssertionError("not used in claim derivation tests")

    def summarize_session(self, request: object) -> object:
        raise AssertionError("not used in claim derivation tests")

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput:
        self.requests.append(request)
        return RawStructuredOutput(payload=self.payload)


def seed_default_policies(repository: SQLiteRepository) -> None:
    for policy_id, principal, channel, purpose in (
        ("assistant_internal", "assistant_internal", "prompt", "prompt"),
        ("current_user", "current_user", "search", "search"),
        ("current_peer", "current_peer", "search", "search"),
        ("shared_session", "shared_session", "prompt", "prompt"),
        ("host_internal", "assistant_internal", "prompt", "prompt"),
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


def seed_active_snapshot(connection: sqlite3.Connection) -> None:
    snapshots = SnapshotRepository(connection)
    snapshots.save_snapshot(
        MemorySnapshot(
            snapshot_id="snapshot-active",
            policy_stamp="hermes_v1@1.0.0",
            parent_snapshot_id=None,
            created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
            artifact_refs=(
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.VECTOR_INDEX,
                    artifact_id="vector:index:bootstrap",
                ),
            ),
        )
    )
    snapshots.upsert_head(
        SnapshotHead(
            head_key="current",
            state=SnapshotHeadState.ACTIVE,
            snapshot_id="snapshot-active",
        )
    )


def seed_session_state(repository: SQLiteRepository) -> None:
    repository.save_session(
        SessionRecord(
            session_id="session:hermes:test",
            host_namespace="hermes",
            session_name="Hermes Claim Derivation Tests",
            recall_mode="hybrid",
            write_frequency="turn",
            created_at=sample_time(),
        )
    )
    repository.save_subject(
        Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
        ),
        created_at=sample_time(),
    )


def save_observation(
    repository: SQLiteRepository,
    *,
    observation_id: str,
    content: str,
    observed_at: datetime,
) -> None:
    repository.save_observation(
        Observation(
            observation_id=observation_id,
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content=content,
            observed_at=observed_at,
            metadata={"message_id": f"message:{observation_id}"},
        ),
    )


class ClaimDerivationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(ClaimDerivationPipeline)
        self.assertIsNotNone(SchemaValidatedClaimDerivationSchema)
        self.assertIsNotNone(fingerprint_candidate_content)

    def build_pipeline(
        self,
        *,
        adapter_payload: object,
    ) -> tuple[
        sqlite3.Connection,
        SQLiteRepository,
        SessionManager,
        FakeEmbeddingClient,
        InMemoryZvecBackend,
        FakeReasoningAdapter,
        object,
    ]:
        connection = open_memory_database()
        repository = SQLiteRepository(connection)
        seed_default_policies(repository)
        seed_session_state(repository)
        seed_active_snapshot(connection)

        manager = SessionManager(
            repository,
            config=ContinuityConfig(
                host="hermes",
                workspace_id="internal-hermes",
                peer_name="hermes",
                session_peer_prefix=True,
            ),
        )
        embedding_client = FakeEmbeddingClient()
        vector_backend = InMemoryZvecBackend()
        adapter = FakeReasoningAdapter(adapter_payload)
        pipeline = ClaimDerivationPipeline(
            connection=connection,
            adapter=adapter,
            embedding_client=embedding_client,
            vector_backend=vector_backend,
            session_manager=manager,
        )

        self.addCleanup(connection.close)
        return connection, repository, manager, embedding_client, vector_backend, adapter, pipeline

    def test_pipeline_publishes_durable_claims_buffers_non_durable_candidates_and_stages_snapshot(self) -> None:
        (
            connection,
            repository,
            manager,
            embedding_client,
            vector_backend,
            adapter,
            pipeline,
        ) = self.build_pipeline(
            adapter_payload={
                "candidates": [
                    {
                        "claim_type": "preference",
                        "subject_ref": "observation:0.author",
                        "scope": "user",
                        "locus_key": "preference/favorite_drink",
                        "value": {"drink": "espresso"},
                        "evidence_refs": ["observation:0"],
                    },
                    {
                        "claim_type": "ephemeral_context",
                        "subject_ref": "observation:1.author",
                        "scope": "session",
                        "locus_key": "session/next_turn_topic",
                        "value": {"note": "espresso follow-up"},
                        "evidence_refs": ["observation:1"],
                    },
                ]
            }
        )
        save_observation(
            repository,
            observation_id="obs-1",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
        )
        save_observation(
            repository,
            observation_id="obs-2",
            content="Please remind me about espresso next turn.",
            observed_at=sample_time(2),
        )

        result = pipeline.derive_from_observations(
            observation_ids=("obs-1", "obs-2"),
            session_id="session:hermes:test",
            source_transaction_kind=TransactionKind.INGEST_TURN,
            run_at=sample_time(3),
        )

        claims = repository.list_claims(subject_id="subject:user:alice")
        beliefs = BeliefStateRepository(connection)
        prompt_buffers = manager.list_buffers(
            "session:hermes:test",
            buffer_kind=SessionBufferKind.PROMPT_ONLY,
        )
        compiler_dependencies = {
            (dependency.upstream_node_id, dependency.downstream_node_id, dependency.role.value)
            for dependency in result.compiler_dependencies
        }

        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(len(result.claim_ids), 1)
        self.assertEqual({claim.claim_id for claim in claims}, set(result.claim_ids))
        self.assertEqual(claims[0].provenance.derivation_run_id, result.derivation_run_id)
        self.assertEqual(claims[0].locus.locus_key, "preference/favorite_drink")
        self.assertEqual(prompt_buffers[0].payload["candidate_id"], result.buffered_candidate_ids[0])
        self.assertEqual(
            beliefs.read_current_state(
                subject_id="subject:user:alice",
                locus_key="preference/favorite_drink",
                policy_stamp="hermes_v1@1.0.0",
            ).projection.active_claim_ids,
            result.claim_ids,
        )
        self.assertEqual(result.active_snapshot_id, "snapshot-active")
        self.assertNotEqual(result.staged_snapshot_id, result.active_snapshot_id)
        self.assertGreater(len(result.compiled_view_ids), 0)
        self.assertGreater(
            connection.execute("SELECT COUNT(*) FROM compiled_views").fetchone()[0],
            0,
        )
        self.assertIn(
            ("observation:obs-1", f"claim:{result.claim_ids[0]}", "provenance"),
            compiler_dependencies,
        )
        self.assertIn(
            ("policy:hermes_v1@1.0.0", f"claim:{result.claim_ids[0]}", "policy"),
            compiler_dependencies,
        )
        self.assertIn(
            ("adapter:reasoning:codex_sdk_gpt_5_4_low@1", f"claim:{result.claim_ids[0]}", "policy"),
            compiler_dependencies,
        )
        self.assertIsNotNone(
            repository.read_compiled_utility_weight(
                target=result.claim_utility_target,
                target_id=result.claim_ids[0],
                policy_stamp="hermes_v1@1.0.0",
            )
        )
        index = ZvecIndex(
            connection=connection,
            embedding_client=embedding_client,
            backend=vector_backend,
            policy_stamp="hermes_v1@1.0.0",
        )
        self.assertIn(
            f"vector:claim:{result.claim_ids[0]}",
            {record.record_id for record in index.list_records()},
        )

    def test_schema_invalid_output_does_not_publish_semantic_mutations(self) -> None:
        connection, repository, _, _, _, _, pipeline = self.build_pipeline(
            adapter_payload={
                "candidates": [
                    {
                        "claim_type": "preference",
                        "subject_ref": "observation:0.author",
                        "scope": "user",
                        "locus_key": "preference/favorite_drink",
                        "value": {"drink": "espresso"},
                        "evidence_refs": "observation:0",
                    }
                ]
            }
        )
        save_observation(
            repository,
            observation_id="obs-1",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
        )

        with self.assertRaises(ValueError):
            pipeline.derive_from_observations(
                observation_ids=("obs-1",),
                session_id="session:hermes:test",
                source_transaction_kind=TransactionKind.INGEST_TURN,
                run_at=sample_time(2),
            )

        self.assertEqual(repository.list_claims(), ())
        self.assertEqual(repository.list_candidate_memories(), ())
        self.assertEqual(BeliefStateRepository(connection).list_states(), ())
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM derivation_runs").fetchone()[0],
            0,
        )
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM compiled_views").fetchone()[0],
            0,
        )
        self.assertIsNotNone(repository.read_observation("obs-1"))

    def test_forgetting_tombstone_blocks_resurrection_from_derivation(self) -> None:
        connection, repository, _, _, _, _, pipeline = self.build_pipeline(
            adapter_payload={
                "candidates": [
                    {
                        "claim_type": "preference",
                        "subject_ref": "observation:0.author",
                        "scope": "user",
                        "locus_key": "preference/favorite_drink",
                        "value": {"drink": "espresso"},
                        "evidence_refs": ["observation:0"],
                    }
                ]
            }
        )
        save_observation(
            repository,
            observation_id="obs-1",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
        )
        repository.forgetting.record_operation(
            ForgettingDecisionTrace(
                operation=ForgettingOperation(
                    operation_id="forget-1",
                    target=ForgettingTarget(
                        target_kind=ForgettingTargetKind.CLAIM,
                        target_id="claim:expunged",
                    ),
                    mode=ForgettingMode.EXPUNGE,
                    requested_by="subject:user:alice",
                    rationale="memory withdrawn",
                    policy_stamp="hermes_v1@1.0.0",
                    recorded_at=sample_time(2),
                ),
                rule=forgetting_rule_for(ForgettingMode.EXPUNGE),
            ),
            tombstones=(
                ForgettingTombstone(
                    tombstone_id="tombstone-1",
                    operation_id="forget-1",
                    target=ForgettingTarget(
                        target_kind=ForgettingTargetKind.CLAIM,
                        target_id="claim:expunged",
                    ),
                    surface=ForgettingSurface.DERIVATION_PIPELINE,
                    content_fingerprint=fingerprint_candidate_content(
                        claim_type="preference",
                        subject_id="subject:user:alice",
                        locus_key="preference/favorite_drink",
                        value={"drink": "espresso"},
                    ),
                    recorded_at=sample_time(2),
                ),
            ),
        )

        result = pipeline.derive_from_observations(
            observation_ids=("obs-1",),
            session_id="session:hermes:test",
            source_transaction_kind=TransactionKind.INGEST_TURN,
            run_at=sample_time(3),
        )

        self.assertEqual(repository.list_claims(), ())
        self.assertEqual(result.claim_ids, ())
        self.assertEqual(result.buffered_candidate_ids, ())
        self.assertEqual(
            repository.read_admission_trace(result.decision_traces[0].decision.candidate_id).decision.outcome.value,
            "discard",
        )


if __name__ == "__main__":
    unittest.main()
