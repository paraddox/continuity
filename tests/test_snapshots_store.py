#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import continuity.snapshots as snapshots_module
from continuity.snapshots import (
    MemorySnapshot,
    SnapshotArtifactKind,
    SnapshotArtifactRef,
    SnapshotHead,
    SnapshotHeadState,
    SnapshotReadPin,
    SnapshotReadUse,
    diff_snapshots,
    promote_candidate_head,
)
from continuity.store.schema import apply_migrations
from continuity.transactions import TransactionKind


SnapshotRepository = getattr(snapshots_module, "SnapshotRepository", None)
SnapshotPromotionRecord = getattr(snapshots_module, "SnapshotPromotionRecord", None)


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def snapshot(
    snapshot_id: str,
    *,
    parent_snapshot_id: str | None,
    created_by_transaction: TransactionKind,
    artifact_refs: tuple[SnapshotArtifactRef, ...],
) -> MemorySnapshot:
    return MemorySnapshot(
        snapshot_id=snapshot_id,
        policy_stamp="hermes_v1@1.0.0",
        parent_snapshot_id=parent_snapshot_id,
        created_by_transaction=created_by_transaction,
        artifact_refs=artifact_refs,
    )


class SnapshotRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_snapshots_heads_pins_and_promotion_history(self) -> None:
        self.assertIsNotNone(SnapshotRepository)
        self.assertIsNotNone(SnapshotPromotionRecord)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = SnapshotRepository(connection)

        active_snapshot = snapshot(
            "snapshot-1",
            parent_snapshot_id=None,
            created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
            artifact_refs=(
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.STATE_VIEW,
                    artifact_id="state:user:alice:preference/coffee",
                ),
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.PROMPT_VIEW,
                    artifact_id="prompt:session:42",
                ),
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.VECTOR_INDEX,
                    artifact_id="zvec:segment:active",
                ),
            ),
        )
        candidate_snapshot = snapshot(
            "snapshot-2",
            parent_snapshot_id="snapshot-1",
            created_by_transaction=TransactionKind.COMPILE_VIEWS,
            artifact_refs=(
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.STATE_VIEW,
                    artifact_id="state:user:alice:preference/coffee",
                ),
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.PROFILE_VIEW,
                    artifact_id="profile:user:alice",
                ),
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.PROMPT_VIEW,
                    artifact_id="prompt:session:43",
                ),
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.VECTOR_INDEX,
                    artifact_id="zvec:segment:active",
                ),
            ),
        )
        active_head = SnapshotHead(
            head_key="current",
            state=SnapshotHeadState.ACTIVE,
            snapshot_id="snapshot-1",
        )
        candidate_head = SnapshotHead(
            head_key="current",
            state=SnapshotHeadState.CANDIDATE,
            snapshot_id="snapshot-2",
            based_on_snapshot_id="snapshot-1",
        )
        read_pin = SnapshotReadPin(
            snapshot_id="snapshot-1",
            read_use=SnapshotReadUse.RETRIEVAL,
            consumer_id="search:claim:user:alice",
        )
        promotion = promote_candidate_head(
            active_head=active_head,
            candidate_head=candidate_head,
        )

        repository.save_snapshot(active_snapshot)
        repository.save_snapshot(candidate_snapshot)
        repository.upsert_head(active_head)
        repository.upsert_head(candidate_head)
        repository.pin_read(read_pin)
        repository.record_promotion(
            promotion_id="promotion-1",
            head_key="current",
            promotion=promotion,
            recorded_at=sample_time(),
        )

        self.assertEqual(repository.list_snapshots(), (active_snapshot, candidate_snapshot))
        self.assertEqual(repository.read_snapshot("snapshot-2"), candidate_snapshot)
        self.assertEqual(repository.list_heads(), (active_head, candidate_head))
        self.assertEqual(repository.read_active_snapshot(head_key="current"), active_snapshot)
        self.assertEqual(repository.read_candidate_snapshot(head_key="current"), candidate_snapshot)
        self.assertEqual(repository.list_read_pins(snapshot_id="snapshot-1"), (read_pin,))
        self.assertEqual(
            repository.list_promotions(head_key="current"),
            (
                SnapshotPromotionRecord(
                    promotion_id="promotion-1",
                    head_key="current",
                    previous_active_snapshot_id="snapshot-1",
                    promoted_snapshot_id="snapshot-2",
                    recorded_at=sample_time(),
                ),
            ),
        )

        self.assertEqual(
            diff_snapshots(
                repository.read_active_snapshot(head_key="current"),
                repository.read_candidate_snapshot(head_key="current"),
            ).added_artifacts,
            (
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.PROFILE_VIEW,
                    artifact_id="profile:user:alice",
                ),
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.PROMPT_VIEW,
                    artifact_id="prompt:session:43",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
