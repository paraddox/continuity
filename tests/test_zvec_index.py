#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


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
from continuity.compiler import CompiledArtifactKind, DependencyRole, DerivedArtifactKind, SourceInputKind
from continuity.epistemics import resolve_locus_belief
from continuity.ontology import MemoryPartition
from continuity.store.belief_revision import BeliefStateRepository, StoredBeliefState
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
    SubjectKind,
)
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import (
    SQLiteRepository,
    SessionMessageRecord,
    SessionRecord,
    StoredDisclosurePolicy,
)

import continuity.index.zvec_index as zvec_index_module
from continuity.index.zvec_index import IndexSourceKind, InMemoryZvecBackend, ZvecIndex


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
        if isinstance(inputs, str):
            texts = (inputs,)
        else:
            texts = tuple(inputs)
        return FakeEmbeddingBatch(
            model=self.model,
            dimensions=4,
            fingerprint=self.fingerprint,
            embeddings=tuple(self._vectorize(text) for text in texts),
        )

    def _vectorize(self, text: str) -> tuple[float, ...]:
        normalized = text.casefold()
        tokens = tuple(normalized.split())
        return (
            float(sum(token in {"espresso", "coffee"} for token in tokens)),
            float(sum(token == "tea" for token in tokens)),
            float(sum(token in {"belief", "state", "profile", "view"} for token in tokens)),
            float(len(tokens)),
        )


def build_admission_trace(*, candidate_id: str, claim_type: str, recorded_at: datetime) -> AdmissionDecisionTrace:
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
            used=0,
        ),
    )


def seed_memory_state(connection: sqlite3.Connection) -> None:
    repository = SQLiteRepository(connection)
    beliefs = BeliefStateRepository(connection)

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
    repository.save_subject(
        Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
        ),
        created_at=sample_time(),
    )
    repository.save_session(
        SessionRecord(
            session_id="session:hermes:test",
            host_namespace="hermes",
            session_name="Hermes Continuity Test",
            recall_mode="balanced",
            write_frequency="default",
            created_at=sample_time(),
        )
    )
    repository.save_message(
        SessionMessageRecord(
            message_id="message-1",
            session_id="session:hermes:test",
            role="user",
            author_subject_id="subject:user:alice",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
            metadata={"transport": "hermes"},
        )
    )
    repository.save_observation(
        Observation(
            observation_id="obs-1",
            source_kind="session_message",
            session_id="session:hermes:test",
            author_subject_id="subject:user:alice",
            content="I prefer espresso over tea.",
            observed_at=sample_time(1),
            metadata={"message_id": "message-1"},
        ),
        message_id="message-1",
    )

    candidate = CandidateMemory(
        candidate_id="candidate-1",
        claim_type="preference.favorite_drink",
        subject_id="subject:user:alice",
        scope=ClaimScope.USER,
        value={"drink": "espresso"},
        source_observation_ids=("obs-1",),
    )
    repository.save_candidate_memory(candidate, created_at=sample_time(2))
    admission_trace = build_admission_trace(
        candidate_id=candidate.candidate_id,
        claim_type=candidate.claim_type,
        recorded_at=sample_time(2),
    )
    repository.admissions.record_decision(admission_trace)

    locus = MemoryLocus(
        subject_id="subject:user:alice",
        locus_key="preference/favorite_drink",
        scope=ClaimScope.USER,
        default_disclosure_policy="assistant_internal",
        conflict_set_key="preference.favorite_drink",
        aggregation_mode=AggregationMode.LATEST_WINS,
    )
    claim = Claim.from_candidate(
        claim_id="claim-1",
        candidate=candidate,
        admission=admission_trace.decision,
        locus=locus,
        provenance=ClaimProvenance(observation_ids=("obs-1",)),
        disclosure_policy="assistant_internal",
        observed_at=sample_time(1),
        learned_at=sample_time(3),
        valid_from=sample_time(1),
    )
    repository.save_claim(claim)

    beliefs.record_state(
        StoredBeliefState(
            belief_id="belief-1",
            policy_stamp="hermes_v1@1.0.0",
            projection=resolve_locus_belief((claim,), as_of=sample_time(4)),
            as_of=sample_time(4),
        )
    )

    connection.execute(
        """
        INSERT INTO snapshots(snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction)
        VALUES (?, ?, ?, ?)
        """,
        ("snapshot-1", "hermes_v1@1.0.0", None, "publish_snapshot"),
    )
    connection.execute(
        """
        INSERT INTO compiled_views(
            compiled_view_id,
            kind,
            view_key,
            policy_stamp,
            snapshot_id,
            epistemic_status,
            payload_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "view-1",
            "state",
            "state:subject:user:alice:preference/favorite_drink",
            "hermes_v1@1.0.0",
            "snapshot-1",
            "supported",
            json.dumps({"summary": "Alice currently prefers espresso."}, sort_keys=True),
            sample_time(5).isoformat(),
        ),
    )
    connection.execute(
        """
        INSERT INTO compiled_view_claims(compiled_view_id, claim_id)
        VALUES (?, ?)
        """,
        ("view-1", "claim-1"),
    )
    connection.execute(
        """
        INSERT INTO compiled_view_observations(compiled_view_id, observation_id)
        VALUES (?, ?)
        """,
        ("view-1", "obs-1"),
    )
    connection.commit()


class ZvecIndexTests(unittest.TestCase):
    def test_zvec_document_ids_are_stable_and_backend_safe(self) -> None:
        self.assertTrue(
            hasattr(zvec_index_module, "_zvec_document_id"),
            "zvec backend should expose a stable document-id encoder for backend-safe IDs",
        )
        encoded = zvec_index_module._zvec_document_id("vector:claim:claim-1")

        self.assertRegex(encoded, r"^record_[0-9a-f]{56}$")
        self.assertLessEqual(len(encoded), 63)
        self.assertEqual(encoded, zvec_index_module._zvec_document_id("vector:claim:claim-1"))
        self.assertNotEqual(encoded, zvec_index_module._zvec_document_id("vector:claim:claim-2"))

    def test_rebuild_from_sqlite_indexes_all_supported_source_kinds(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_memory_state(connection)

        index = ZvecIndex(
            connection=connection,
            embedding_client=FakeEmbeddingClient(),
            backend=InMemoryZvecBackend(),
            policy_stamp="hermes_v1@1.0.0",
        )

        result = index.rebuild_from_sqlite()

        self.assertEqual(
            {record.source_kind for record in result.records},
            {
                IndexSourceKind.SESSION_MESSAGE,
                IndexSourceKind.OBSERVATION,
                IndexSourceKind.CLAIM,
                IndexSourceKind.BELIEF_STATE,
                IndexSourceKind.COMPILED_VIEW,
            },
        )
        self.assertEqual(
            {record.source_id for record in result.records},
            {"message-1", "obs-1", "claim-1", "belief-1", "view-1"},
        )
        self.assertEqual(
            {record.record_id for record in index.list_records()},
            {record.record_id for record in result.records},
        )
        self.assertIn(
            ("claim:claim-1", "vector:claim:claim-1", DependencyRole.INDEX.value),
            {
                (dependency.upstream_node_id, dependency.downstream_node_id, dependency.role.value)
                for dependency in result.compiler_dependencies
            },
        )
        self.assertIn(
            "vector:belief:belief-1",
            {node.node_id for node in result.compiler_nodes if node.kind is CompiledArtifactKind.VECTOR_INDEX_RECORD},
        )
        belief_record = next(record for record in result.records if record.source_kind is IndexSourceKind.BELIEF_STATE)
        self.assertEqual(belief_record.metadata["index_view"], "belief_state")
        self.assertEqual(belief_record.locus_key, "preference/favorite_drink")

    def test_search_resolves_hits_back_to_authoritative_records_and_nodes(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_memory_state(connection)

        index = ZvecIndex(
            connection=connection,
            embedding_client=FakeEmbeddingClient(),
            backend=InMemoryZvecBackend(),
            policy_stamp="hermes_v1@1.0.0",
        )
        index.rebuild_from_sqlite()

        hits = index.search("espresso belief state", topk=5)

        self.assertEqual(len(hits), 5)

        observation_hit = next(hit for hit in hits if hit.record.source_kind is IndexSourceKind.OBSERVATION)
        self.assertEqual(observation_hit.source_node.kind, SourceInputKind.OBSERVATION)
        self.assertEqual(observation_hit.record.source_id, "obs-1")
        self.assertEqual(observation_hit.source.observation_id, "obs-1")

        claim_hit = next(hit for hit in hits if hit.record.source_kind is IndexSourceKind.CLAIM)
        self.assertEqual(claim_hit.source_node.kind, DerivedArtifactKind.CLAIM)
        self.assertEqual(claim_hit.source.claim_id, "claim-1")

        belief_hit = next(hit for hit in hits if hit.record.source_kind is IndexSourceKind.BELIEF_STATE)
        self.assertEqual(belief_hit.source_node.kind, DerivedArtifactKind.LOCUS)
        self.assertEqual(
            belief_hit.source_node.node_id,
            "locus:subject:user:alice:preference/favorite_drink",
        )
        self.assertEqual(belief_hit.source.belief_id, "belief-1")

    def test_rebuild_replaces_stale_records_from_sqlite_truth(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_memory_state(connection)

        index = ZvecIndex(
            connection=connection,
            embedding_client=FakeEmbeddingClient(),
            backend=InMemoryZvecBackend(),
            policy_stamp="hermes_v1@1.0.0",
        )
        first_result = index.rebuild_from_sqlite()
        self.assertIn("vector:view:view-1", {record.record_id for record in first_result.records})

        connection.execute("DELETE FROM compiled_views WHERE compiled_view_id = ?", ("view-1",))
        connection.commit()

        second_result = index.rebuild_from_sqlite()

        self.assertIn("vector:view:view-1", second_result.deleted_record_ids)
        self.assertNotIn("vector:view:view-1", {record.record_id for record in index.list_records()})


if __name__ == "__main__":
    unittest.main()
