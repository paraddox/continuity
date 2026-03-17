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

import continuity.forgetting as forgetting_module
from continuity.forgetting import (
    ArtifactResidency,
    ForgettingDecisionTrace,
    ForgettingMode,
    ForgettingOperation,
    ForgettingSurface,
    ForgettingTarget,
    ForgettingTargetKind,
    forgetting_rule_for,
)
from continuity.store.schema import apply_migrations


ForgettingRepository = getattr(forgetting_module, "ForgettingRepository", None)
ForgettingTombstone = getattr(forgetting_module, "ForgettingTombstone", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


class ForgettingRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_operations_effects_and_tombstones(self) -> None:
        self.assertIsNotNone(ForgettingRepository)
        self.assertIsNotNone(ForgettingTombstone)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = ForgettingRepository(connection)
        target = ForgettingTarget(
            target_kind=ForgettingTargetKind.CLAIM,
            target_id="claim-1",
        )
        trace = ForgettingDecisionTrace(
            operation=ForgettingOperation(
                operation_id="forget-1",
                target=target,
                mode=ForgettingMode.SUPPRESS,
                requested_by="subject:user:alice",
                rationale="hide this memory from ordinary reads",
                policy_stamp="hermes_v1@1.0.0",
                recorded_at=sample_time(),
            ),
            rule=forgetting_rule_for(ForgettingMode.SUPPRESS),
        )
        tombstones = (
            ForgettingTombstone(
                tombstone_id="tombstone-1",
                operation_id="forget-1",
                target=target,
                surface=ForgettingSurface.SNAPSHOT_STORE,
                content_fingerprint="sha256:snapshot",
                recorded_at=sample_time(),
            ),
            ForgettingTombstone(
                tombstone_id="tombstone-2",
                operation_id="forget-1",
                target=target,
                surface=ForgettingSurface.VECTOR_INDEX,
                content_fingerprint="sha256:vector",
                recorded_at=sample_time(1),
            ),
        )

        repository.record_operation(trace, tombstones=tombstones)

        stored = repository.read_record("forget-1")

        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.operation, trace.operation)
        self.assertTrue(stored.host_reads_withdrawn)
        self.assertEqual(
            stored.residency_for(ForgettingSurface.VECTOR_INDEX),
            ArtifactResidency.REMOVED,
        )
        self.assertEqual(
            stored.residency_for(ForgettingSurface.DERIVATION_PIPELINE),
            ArtifactResidency.HIDDEN_FROM_HOST,
        )
        self.assertFalse(stored.blocks_resurrection(ForgettingSurface.IMPORT_PIPELINE))
        self.assertEqual(
            {surface.value for surface in repository.surfaces_requiring_withdrawal(target)},
            {
                "vector_index",
                "snapshot_store",
                "prefetch_cache",
                "derivation_pipeline",
                "tombstone_ledger",
            },
        )
        self.assertEqual(repository.resurrection_guard_surfaces(target), ())
        self.assertEqual(
            tuple(record.operation.operation_id for record in repository.list_operations(target=target)),
            ("forget-1",),
        )
        self.assertEqual(
            tuple(
                tombstone.surface.value
                for tombstone in repository.list_tombstones(target=target)
            ),
            ("snapshot_store", "vector_index"),
        )

    def test_expunge_guards_prevent_resurrection_and_latest_operation_wins(self) -> None:
        self.assertIsNotNone(ForgettingRepository)
        self.assertIsNotNone(ForgettingTombstone)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = ForgettingRepository(connection)
        target = ForgettingTarget(
            target_kind=ForgettingTargetKind.IMPORTED_ARTIFACT,
            target_id="artifact-1",
        )

        repository.record_operation(
            ForgettingDecisionTrace(
                operation=ForgettingOperation(
                    operation_id="forget-1",
                    target=target,
                    mode=ForgettingMode.SUPPRESS,
                    requested_by="subject:user:alice",
                    rationale="withdraw imported content pending review",
                    policy_stamp="hermes_v1@1.0.0",
                    recorded_at=sample_time(),
                ),
                rule=forgetting_rule_for(ForgettingMode.SUPPRESS),
            ),
        )
        repository.record_operation(
            ForgettingDecisionTrace(
                operation=ForgettingOperation(
                    operation_id="forget-2",
                    target=target,
                    mode=ForgettingMode.EXPUNGE,
                    requested_by="subject:user:alice",
                    rationale="delete imported content permanently",
                    policy_stamp="hermes_v1@1.0.0",
                    recorded_at=sample_time(5),
                ),
                rule=forgetting_rule_for(ForgettingMode.EXPUNGE),
            ),
            tombstones=(
                ForgettingTombstone(
                    tombstone_id="tombstone-1",
                    operation_id="forget-2",
                    target=target,
                    surface=ForgettingSurface.IMPORT_PIPELINE,
                    content_fingerprint="sha256:artifact",
                    recorded_at=sample_time(5),
                ),
            ),
        )

        current = repository.current_record_for_target(target)

        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(current.operation.operation_id, "forget-2")
        self.assertEqual(current.operation.mode, ForgettingMode.EXPUNGE)
        self.assertTrue(repository.ordinary_read_blocked(target))
        self.assertEqual(
            {surface.value for surface in repository.resurrection_guard_surfaces(target)},
            {
                "vector_index",
                "snapshot_store",
                "prefetch_cache",
                "replay_artifacts",
                "archive_tier",
                "import_pipeline",
                "derivation_pipeline",
            },
        )
        self.assertTrue(
            {
                "claim_ledger",
                "observation_log",
                "snapshot_store",
                "archive_tier",
                "tombstone_ledger",
            }.issubset(
                {surface.value for surface in repository.surfaces_requiring_withdrawal(target)}
            )
        )
        self.assertEqual(
            tuple(record.operation.operation_id for record in repository.list_operations(target=target)),
            ("forget-2", "forget-1"),
        )
        self.assertEqual(
            tuple(tombstone.content_fingerprint for tombstone in repository.list_tombstones(target=target)),
            ("sha256:artifact",),
        )


if __name__ == "__main__":
    unittest.main()
