#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import continuity.tiers as tiers_module
from continuity.tiers import MemoryTier, RebuildUrgency, RetrievalBias, SnapshotResidency
from continuity.store.schema import apply_migrations


TierAssignment = getattr(tiers_module, "TierAssignment", None)
TierStateRepository = getattr(tiers_module, "TierStateRepository", None)
TierTransition = getattr(tiers_module, "TierTransition", None)
RetentionMetadata = getattr(tiers_module, "RetentionMetadata", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


class TierStateRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_assignments_and_retention_metadata_queries(self) -> None:
        self.assertIsNotNone(TierAssignment)
        self.assertIsNotNone(TierStateRepository)
        self.assertIsNotNone(RetentionMetadata)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = TierStateRepository(connection)

        hot_claim = TierAssignment(
            target_kind="claim",
            target_id="claim:user:alice:instruction/coffee",
            policy_stamp="hermes_v1@1.0.0",
            tier=MemoryTier.HOT,
            rationale="active instruction should stay prompt-adjacent",
            assigned_at=sample_time(),
        )
        warm_view = TierAssignment(
            target_kind="compiled_view",
            target_id="profile:user:alice",
            policy_stamp="hermes_v1@1.0.0",
            tier=MemoryTier.WARM,
            rationale="profile view remains durable in ordinary reads",
            assigned_at=sample_time(1),
        )
        frozen_replay = TierAssignment(
            target_kind="replay_artifact",
            target_id="replay:session:42",
            policy_stamp="hermes_v1@1.0.0",
            tier=MemoryTier.FROZEN,
            rationale="replay artifacts are archival by default",
            assigned_at=sample_time(2),
        )

        repository.upsert_assignment(hot_claim)
        repository.upsert_assignment(warm_view)
        repository.upsert_assignment(frozen_replay)

        self.assertEqual(
            repository.read_assignment(
                target_kind="claim",
                target_id="claim:user:alice:instruction/coffee",
                policy_stamp="hermes_v1@1.0.0",
            ),
            hot_claim,
        )
        self.assertIsNone(
            repository.read_assignment(
                target_kind="claim",
                target_id="missing",
                policy_stamp="hermes_v1@1.0.0",
            )
        )
        self.assertEqual(
            repository.list_assignments(policy_stamp="hermes_v1@1.0.0"),
            (hot_claim, warm_view, frozen_replay),
        )
        self.assertEqual(
            tuple(
                assignment.target_id
                for assignment in repository.list_assignments(
                    tiers=(MemoryTier.HOT, MemoryTier.WARM),
                )
            ),
            (
                "claim:user:alice:instruction/coffee",
                "profile:user:alice",
            ),
        )

        hot_metadata = repository.read_retention_metadata(
            target_kind="claim",
            target_id="claim:user:alice:instruction/coffee",
            policy_stamp="hermes_v1@1.0.0",
        )
        frozen_metadata = repository.read_retention_metadata(
            target_kind="replay_artifact",
            target_id="replay:session:42",
            policy_stamp="hermes_v1@1.0.0",
        )

        self.assertEqual(
            hot_metadata,
            RetentionMetadata(
                target_kind="claim",
                target_id="claim:user:alice:instruction/coffee",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.HOT,
                rationale="active instruction should stay prompt-adjacent",
                assigned_at=sample_time(),
                retrieval_bias=RetrievalBias.PRIMARY,
                rebuild_urgency=RebuildUrgency.IMMEDIATE,
                snapshot_residency=SnapshotResidency.ACTIVE,
                default_in_host_reads=True,
                expunge_guarded=False,
            ),
        )
        self.assertEqual(frozen_metadata.rebuild_urgency, RebuildUrgency.ARCHIVAL)
        self.assertEqual(frozen_metadata.snapshot_residency, SnapshotResidency.ARCHIVAL_ONLY)
        self.assertTrue(frozen_metadata.expunge_guarded)

    def test_repository_records_transitions_and_updates_current_assignment(self) -> None:
        self.assertIsNotNone(TierAssignment)
        self.assertIsNotNone(TierStateRepository)
        self.assertIsNotNone(TierTransition)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = TierStateRepository(connection)

        repository.upsert_assignment(
            TierAssignment(
                target_kind="claim",
                target_id="claim:user:alice:preference/coffee",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.WARM,
                rationale="stable preference begins warm",
                assigned_at=sample_time(),
            )
        )

        promoted = TierTransition(
            transition_id="transition-1",
            target_kind="claim",
            target_id="claim:user:alice:preference/coffee",
            policy_stamp="hermes_v1@1.0.0",
            from_tier=MemoryTier.WARM,
            to_tier=MemoryTier.HOT,
            rationale="recent utility spike should promote the active claim",
            transitioned_at=sample_time(5),
        )
        demoted = TierTransition(
            transition_id="transition-2",
            target_kind="claim",
            target_id="claim:user:alice:preference/coffee",
            policy_stamp="hermes_v1@1.0.0",
            from_tier=MemoryTier.HOT,
            to_tier=MemoryTier.COLD,
            rationale="the claim remains recallable but should leave default reads",
            transitioned_at=sample_time(30),
        )

        repository.record_transition(promoted)
        repository.record_transition(demoted)

        self.assertEqual(
            repository.read_assignment(
                target_kind="claim",
                target_id="claim:user:alice:preference/coffee",
                policy_stamp="hermes_v1@1.0.0",
            ),
            TierAssignment(
                target_kind="claim",
                target_id="claim:user:alice:preference/coffee",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.COLD,
                rationale="the claim remains recallable but should leave default reads",
                assigned_at=sample_time(30),
            ),
        )
        self.assertEqual(
            repository.list_transitions(
                target_kind="claim",
                target_id="claim:user:alice:preference/coffee",
                policy_stamp="hermes_v1@1.0.0",
            ),
            (promoted, demoted),
        )
        self.assertEqual(
            tuple(
                metadata.target_id
                for metadata in repository.list_retention_metadata(
                    tiers=(MemoryTier.COLD,),
                    target_kind="claim",
                )
            ),
            ("claim:user:alice:preference/coffee",),
        )


if __name__ == "__main__":
    unittest.main()
