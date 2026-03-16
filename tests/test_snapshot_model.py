#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    rollback_active_head,
)
from continuity.transactions import TransactionKind


class SnapshotContractTests(unittest.TestCase):
    def test_snapshot_artifact_kinds_and_read_uses_are_closed(self) -> None:
        self.assertEqual(
            {kind.value for kind in SnapshotArtifactKind},
            {
                "state_view",
                "timeline_view",
                "set_view",
                "profile_view",
                "prompt_view",
                "evidence_view",
                "answer_view",
                "vector_index",
            },
        )
        self.assertEqual(
            {use.value for use in SnapshotReadUse},
            {
                "retrieval",
                "prompt_assembly",
                "answer_query",
                "prefetch",
                "replay",
            },
        )

    def test_candidate_promotion_rollback_and_diff_are_explicit(self) -> None:
        active_snapshot = MemorySnapshot(
            snapshot_id="snapshot-1",
            policy_stamp="hermes_v1@1.0.0",
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
        candidate_snapshot = MemorySnapshot(
            snapshot_id="snapshot-2",
            policy_stamp="hermes_v1@1.0.0",
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

        diff = diff_snapshots(active_snapshot, candidate_snapshot)
        self.assertEqual(diff.from_snapshot_id, "snapshot-1")
        self.assertEqual(diff.to_snapshot_id, "snapshot-2")
        self.assertEqual(
            diff.added_artifacts,
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
        self.assertEqual(
            diff.removed_artifacts,
            (
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.PROMPT_VIEW,
                    artifact_id="prompt:session:42",
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

        promotion = promote_candidate_head(
            active_head=active_head,
            candidate_head=candidate_head,
        )
        self.assertEqual(promotion.previous_active_snapshot_id, "snapshot-1")
        self.assertEqual(promotion.promoted_snapshot_id, "snapshot-2")
        self.assertEqual(promotion.new_active_head.snapshot_id, "snapshot-2")
        self.assertEqual(promotion.new_active_head.state, SnapshotHeadState.ACTIVE)

        rollback = rollback_active_head(
            active_head=promotion.new_active_head,
            rollback_to_snapshot_id="snapshot-1",
            reason="candidate regression",
        )
        self.assertEqual(rollback.previous_active_snapshot_id, "snapshot-2")
        self.assertEqual(rollback.rollback_to_snapshot_id, "snapshot-1")
        self.assertEqual(rollback.reason, "candidate regression")
        self.assertEqual(rollback.new_active_head.snapshot_id, "snapshot-1")

    def test_reads_pin_one_snapshot_for_their_whole_operation(self) -> None:
        pin = SnapshotReadPin(
            snapshot_id="snapshot-2",
            read_use=SnapshotReadUse.RETRIEVAL,
            consumer_id="search:claim:user:alice",
        )

        self.assertEqual(pin.snapshot_id, "snapshot-2")
        self.assertEqual(pin.read_use, SnapshotReadUse.RETRIEVAL)
        self.assertEqual(pin.consumer_id, "search:claim:user:alice")


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_snapshot_consistency_layer(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("snapshot consistency layer", text)
        self.assertIn("candidate snapshot", text)
        self.assertIn("active head", text)
        self.assertIn("retrieval runs against one snapshot", text)
        self.assertIn("hosts read `current`", text)


if __name__ == "__main__":
    unittest.main()
