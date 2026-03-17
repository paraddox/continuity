#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import continuity.snapshots as snapshots_module
from continuity.arbiter import ArbiterPublicationKind
from continuity.events import SystemEventType
from continuity.snapshots import (
    MemorySnapshot,
    SnapshotArtifactKind,
    SnapshotArtifactRef,
    SnapshotHead,
    SnapshotHeadState,
    SnapshotRepository,
)
from continuity.store.schema import apply_migrations
from continuity.transactions import DurabilityWaterline, TransactionKind


SnapshotRuntime = getattr(snapshots_module, "SnapshotRuntime", None)
MaterializedSnapshot = getattr(snapshots_module, "MaterializedSnapshot", None)
PublishedSnapshot = getattr(snapshots_module, "PublishedSnapshot", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    apply_migrations(connection)
    return connection


def snapshot(
    snapshot_id: str,
    *,
    policy_stamp: str = "hermes_v1@1.0.0",
    parent_snapshot_id: str | None,
    created_by_transaction: TransactionKind,
    artifact_refs: tuple[SnapshotArtifactRef, ...],
) -> MemorySnapshot:
    return MemorySnapshot(
        snapshot_id=snapshot_id,
        policy_stamp=policy_stamp,
        parent_snapshot_id=parent_snapshot_id,
        created_by_transaction=created_by_transaction,
        artifact_refs=artifact_refs,
    )


class SnapshotRuntimeTests(unittest.TestCase):
    def seed_active_head(self, connection: sqlite3.Connection) -> SnapshotRepository:
        repository = SnapshotRepository(connection)
        repository.save_snapshot(
            snapshot(
                "snapshot:active",
                parent_snapshot_id=None,
                created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
                artifact_refs=(
                    SnapshotArtifactRef(
                        SnapshotArtifactKind.STATE_VIEW,
                        "state:user:alice:drink:v1",
                    ),
                    SnapshotArtifactRef(
                        SnapshotArtifactKind.PROFILE_VIEW,
                        "profile:user:alice:v1",
                    ),
                    SnapshotArtifactRef(
                        SnapshotArtifactKind.PROMPT_VIEW,
                        "prompt:session:1:v1",
                    ),
                ),
            )
        )
        repository.upsert_head(
            SnapshotHead(
                head_key="current",
                state=SnapshotHeadState.ACTIVE,
                snapshot_id="snapshot:active",
            )
        )
        return repository

    def test_runtime_materializes_candidate_snapshot_from_active_head_and_exposes_diff(self) -> None:
        self.assertIsNotNone(SnapshotRuntime)
        self.assertIsNotNone(MaterializedSnapshot)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = self.seed_active_head(connection)
        runtime = SnapshotRuntime(connection)

        materialized = runtime.materialize_candidate(
            candidate_snapshot_id="snapshot:candidate:1",
            policy_stamp="hermes_v1@1.0.0",
            created_by_transaction=TransactionKind.COMPILE_VIEWS,
            added_artifacts=(
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROFILE_VIEW,
                    "profile:user:alice:v2",
                ),
                SnapshotArtifactRef(
                    SnapshotArtifactKind.ANSWER_VIEW,
                    "answer:user:alice:memory:v1",
                ),
            ),
            removed_artifact_ids=("profile:user:alice:v1",),
        )

        self.assertEqual(materialized.base_snapshot.snapshot_id, "snapshot:active")
        self.assertEqual(materialized.candidate_snapshot.snapshot_id, "snapshot:candidate:1")
        self.assertEqual(
            materialized.candidate_snapshot.artifact_refs,
            (
                SnapshotArtifactRef(
                    SnapshotArtifactKind.STATE_VIEW,
                    "state:user:alice:drink:v1",
                ),
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROMPT_VIEW,
                    "prompt:session:1:v1",
                ),
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROFILE_VIEW,
                    "profile:user:alice:v2",
                ),
                SnapshotArtifactRef(
                    SnapshotArtifactKind.ANSWER_VIEW,
                    "answer:user:alice:memory:v1",
                ),
            ),
        )
        self.assertEqual(
            materialized.diff.added_artifacts,
            (
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROFILE_VIEW,
                    "profile:user:alice:v2",
                ),
                SnapshotArtifactRef(
                    SnapshotArtifactKind.ANSWER_VIEW,
                    "answer:user:alice:memory:v1",
                ),
            ),
        )
        self.assertEqual(
            materialized.diff.removed_artifacts,
            (
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROFILE_VIEW,
                    "profile:user:alice:v1",
                ),
            ),
        )
        self.assertEqual(
            repository.read_active_snapshot(head_key="current").snapshot_id,
            "snapshot:active",
        )
        self.assertEqual(
            repository.read_candidate_snapshot(head_key="current"),
            materialized.candidate_snapshot,
        )

    def test_runtime_supports_branchable_policy_snapshot_candidates_without_touching_current(self) -> None:
        self.assertIsNotNone(SnapshotRuntime)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = self.seed_active_head(connection)
        branch_runtime = SnapshotRuntime(connection, head_key="experiment:policy-v2")

        materialized = branch_runtime.materialize_candidate(
            candidate_snapshot_id="snapshot:experiment:1",
            base_snapshot_id="snapshot:active",
            policy_stamp="hermes_v2@0.1.0",
            created_by_transaction=TransactionKind.COMPILE_VIEWS,
            added_artifacts=(
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROMPT_VIEW,
                    "prompt:session:1:v2",
                ),
            ),
            removed_artifact_ids=("prompt:session:1:v1",),
        )

        self.assertEqual(materialized.base_snapshot.snapshot_id, "snapshot:active")
        self.assertEqual(materialized.candidate_head.head_key, "experiment:policy-v2")
        self.assertEqual(
            materialized.candidate_head.based_on_snapshot_id,
            "snapshot:active",
        )
        self.assertEqual(
            materialized.diff.removed_artifacts,
            (
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROMPT_VIEW,
                    "prompt:session:1:v1",
                ),
            ),
        )
        self.assertEqual(
            materialized.diff.added_artifacts,
            (
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROMPT_VIEW,
                    "prompt:session:1:v2",
                ),
            ),
        )
        self.assertEqual(
            repository.read_active_snapshot(head_key="current").snapshot_id,
            "snapshot:active",
        )
        self.assertEqual(
            repository.read_candidate_snapshot(head_key="experiment:policy-v2"),
            materialized.candidate_snapshot,
        )

    def test_runtime_promotes_candidate_atomically_and_serializes_competing_promotions(self) -> None:
        self.assertIsNotNone(SnapshotRuntime)
        self.assertIsNotNone(PublishedSnapshot)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = self.seed_active_head(connection)
        runtime = SnapshotRuntime(connection)
        materialized = runtime.materialize_candidate(
            candidate_snapshot_id="snapshot:candidate:1",
            policy_stamp="hermes_v1@1.0.0",
            created_by_transaction=TransactionKind.COMPILE_VIEWS,
            added_artifacts=(
                SnapshotArtifactRef(
                    SnapshotArtifactKind.PROFILE_VIEW,
                    "profile:user:alice:v2",
                ),
            ),
            removed_artifact_ids=("profile:user:alice:v1",),
        )

        start_gate = threading.Barrier(3)
        outcomes: list[object] = []

        def promote(index: int) -> None:
            start_gate.wait()
            try:
                outcomes.append(
                    runtime.promote_candidate(
                        promotion_id=f"promotion:{index}",
                        published_at=sample_time(index + 1),
                    )
                )
            except Exception as exc:  # pragma: no cover - asserted below
                outcomes.append(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(promote, index) for index in range(2)]
            start_gate.wait()
            for future in futures:
                future.result()

        successes = [outcome for outcome in outcomes if isinstance(outcome, PublishedSnapshot)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]

        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        published = successes[0]
        self.assertEqual(published.materialized.candidate_snapshot, materialized.candidate_snapshot)
        self.assertEqual(published.promotion.promoted_snapshot_id, "snapshot:candidate:1")
        self.assertEqual(published.publication.publication_kind, ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION)
        self.assertEqual(published.publication.reached_waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(published.event.event_type, SystemEventType.SNAPSHOT_PUBLISHED)
        self.assertEqual(
            repository.read_active_snapshot(head_key="current").snapshot_id,
            "snapshot:candidate:1",
        )
        self.assertIsNone(repository.read_candidate_snapshot(head_key="current"))
        self.assertEqual(
            tuple(record.promoted_snapshot_id for record in repository.list_promotions(head_key="current")),
            ("snapshot:candidate:1",),
        )
        self.assertEqual(
            tuple(
                publication.publication_kind
                for publication in runtime.arbiter.list_publications(
                    publication_kind=ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION
                )
            ),
            (ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,),
        )


if __name__ == "__main__":
    unittest.main()
