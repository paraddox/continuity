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

try:
    import continuity.prefetch as prefetch_module
except ModuleNotFoundError:
    prefetch_module = None

from continuity.compiler import (
    CompiledArtifactKind,
    CompilerNode,
    CompilerNodeCategory,
    CompilerStateRepository,
    DirtyCause,
    DirtyNode,
    DirtyReason,
)
from continuity.snapshots import (
    MemorySnapshot,
    SnapshotArtifactKind,
    SnapshotArtifactRef,
    SnapshotHead,
    SnapshotHeadState,
    SnapshotReadUse,
    SnapshotRepository,
)
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository, SessionRecord
from continuity.tiers import MemoryTier, TierAssignment, TierStateRepository
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
    TransactionRunner,
)


PrefetchRuntime = getattr(prefetch_module, "PrefetchRuntime", None) if prefetch_module else None
PrefetchRepository = getattr(prefetch_module, "PrefetchRepository", None) if prefetch_module else None
PrefetchStatus = getattr(prefetch_module, "PrefetchStatus", None) if prefetch_module else None


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def seed_session(
    connection: sqlite3.Connection,
    *,
    session_id: str = "session:1",
    recall_mode: str = "hybrid",
) -> None:
    SQLiteRepository(connection).save_session(
        SessionRecord(
            session_id=session_id,
            host_namespace="hermes",
            session_name="continuity-session",
            recall_mode=recall_mode,
            write_frequency="async",
            created_at=sample_time(),
        )
    )


def seed_active_snapshot(
    connection: sqlite3.Connection,
    *,
    snapshot_id: str = "snapshot:active",
    artifact_refs: tuple[SnapshotArtifactRef, ...],
) -> None:
    repository = SnapshotRepository(connection)
    repository.save_snapshot(
        MemorySnapshot(
            snapshot_id=snapshot_id,
            policy_stamp="hermes_v1@1.0.0",
            parent_snapshot_id=None,
            created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
            artifact_refs=artifact_refs,
        )
    )
    repository.upsert_head(
        SnapshotHead(
            head_key="current",
            state=SnapshotHeadState.ACTIVE,
            snapshot_id=snapshot_id,
        )
    )


def compiled_node(
    node_id: str,
    kind: CompiledArtifactKind,
    *,
    subject_id: str | None = None,
    locus_key: str | None = None,
) -> CompilerNode:
    return CompilerNode(
        node_id=node_id,
        category=CompilerNodeCategory.COMPILED_ARTIFACT,
        kind=kind,
        fingerprint=f"{node_id}@1",
        subject_id=subject_id,
        locus_key=locus_key,
    )


class PrefetchRuntimeTests(unittest.TestCase):
    def test_prefetch_transaction_warms_default_host_views_after_a_cold_first_turn(self) -> None:
        self.assertIsNotNone(PrefetchRuntime)
        self.assertIsNotNone(PrefetchRepository)
        self.assertIsNotNone(PrefetchStatus)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_session(connection)
        seed_active_snapshot(
            connection,
            artifact_refs=(
                SnapshotArtifactRef(SnapshotArtifactKind.STATE_VIEW, "view:state:session:1"),
                SnapshotArtifactRef(SnapshotArtifactKind.PROFILE_VIEW, "view:profile:user:self"),
                SnapshotArtifactRef(SnapshotArtifactKind.PROMPT_VIEW, "view:prompt:session:1"),
                SnapshotArtifactRef(SnapshotArtifactKind.VECTOR_INDEX, "vector:session:1"),
            ),
        )
        runtime = PrefetchRuntime(connection)

        self.assertIsNone(runtime.read_warm_cache(session_id="session:1"))

        runner = TransactionRunner(
            {
                TransactionPhase.PREFETCH: lambda context: runtime.warm_next_turn(
                    session_id=str(context.payload["session_id"]),
                    warmed_at=context.payload["warmed_at"],
                )
            }
        )

        execution = runner.run(
            TransactionKind.PREFETCH_NEXT_TURN,
            payload={
                "session_id": "session:1",
                "warmed_at": sample_time(5),
            },
        )

        warmed = execution.phase_execution_for(TransactionPhase.PREFETCH).output

        self.assertEqual(execution.reached_waterline, DurabilityWaterline.PREFETCH_WARMED)
        self.assertEqual(warmed.session_id, "session:1")
        self.assertEqual(warmed.snapshot_id, "snapshot:active")
        self.assertEqual(
            warmed.artifact_ids,
            (
                "view:state:session:1",
                "view:profile:user:self",
                "view:prompt:session:1",
            ),
        )
        self.assertEqual(runtime.read_warm_cache(session_id="session:1"), warmed)
        self.assertEqual(
            SnapshotRepository(connection).list_read_pins(
                snapshot_id="snapshot:active",
                read_use=SnapshotReadUse.PREFETCH,
            ),
            (warmed.read_pin,),
        )
        stored = PrefetchRepository(connection).read(session_id="session:1")
        self.assertEqual(stored, warmed)
        self.assertEqual(stored.status, PrefetchStatus.WARM)

    def test_prefetch_respects_tier_overrides_for_default_host_reads(self) -> None:
        self.assertIsNotNone(PrefetchRuntime)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_session(connection)
        seed_active_snapshot(
            connection,
            artifact_refs=(
                SnapshotArtifactRef(SnapshotArtifactKind.STATE_VIEW, "view:state:session:1"),
                SnapshotArtifactRef(SnapshotArtifactKind.PROMPT_VIEW, "view:prompt:session:1"),
            ),
        )
        TierStateRepository(connection).upsert_assignment(
            TierAssignment(
                target_kind="compiled_view",
                target_id="view:prompt:session:1",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.COLD,
                rationale="prompt view forced into recall-only tier",
                assigned_at=sample_time(1),
            )
        )

        warmed = PrefetchRuntime(connection).warm_next_turn(
            session_id="session:1",
            warmed_at=sample_time(5),
        )

        self.assertEqual(warmed.artifact_ids, ("view:state:session:1",))

    def test_prefetch_cache_is_pinned_to_its_snapshot_and_not_served_after_head_moves(self) -> None:
        self.assertIsNotNone(PrefetchRuntime)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_session(connection)
        seed_active_snapshot(
            connection,
            artifact_refs=(
                SnapshotArtifactRef(SnapshotArtifactKind.PROMPT_VIEW, "view:prompt:session:1"),
            ),
        )
        runtime = PrefetchRuntime(connection)
        warmed = runtime.warm_next_turn(
            session_id="session:1",
            warmed_at=sample_time(5),
        )

        snapshots = SnapshotRepository(connection)
        snapshots.save_snapshot(
            MemorySnapshot(
                snapshot_id="snapshot:next",
                policy_stamp="hermes_v1@1.0.0",
                parent_snapshot_id="snapshot:active",
                created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
                artifact_refs=(
                    SnapshotArtifactRef(SnapshotArtifactKind.PROMPT_VIEW, "view:prompt:session:2"),
                ),
            )
        )
        snapshots.upsert_head(
            SnapshotHead(
                head_key="current",
                state=SnapshotHeadState.ACTIVE,
                snapshot_id="snapshot:next",
            )
        )

        self.assertIsNone(runtime.read_warm_cache(session_id="session:1"))
        self.assertEqual(
            runtime.read_warm_cache(
                session_id="session:1",
                target_snapshot_id="snapshot:active",
            ),
            warmed,
        )

    def test_prefetch_invalidates_warm_cache_when_compiler_marks_pending_dirty_work(self) -> None:
        self.assertIsNotNone(PrefetchRuntime)
        self.assertIsNotNone(PrefetchRepository)
        self.assertIsNotNone(PrefetchStatus)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_session(connection)
        seed_active_snapshot(
            connection,
            artifact_refs=(
                SnapshotArtifactRef(SnapshotArtifactKind.PROMPT_VIEW, "view:prompt:session:1"),
            ),
        )
        runtime = PrefetchRuntime(connection)
        runtime.warm_next_turn(session_id="session:1", warmed_at=sample_time(5))

        compiler_state = CompilerStateRepository(connection)
        compiler_state.upsert_nodes(
            (
                compiled_node(
                    "view:prompt:session:1",
                    CompiledArtifactKind.PROMPT_VIEW,
                    subject_id="subject:user:self",
                ),
            )
        )
        compiler_state.enqueue_dirty_nodes(
            (
                DirtyNode(
                    node_id="view:prompt:session:1",
                    category=CompilerNodeCategory.COMPILED_ARTIFACT,
                    kind=CompiledArtifactKind.PROMPT_VIEW,
                    subject_id="subject:user:self",
                    locus_key=None,
                    causes=(
                        DirtyCause(
                            reason=DirtyReason.POLICY_UPGRADED,
                            changed_node_id="policy:hermes_v1",
                            changed_node_kind="policy_pack",
                            dependency_path=("policy:hermes_v1", "view:prompt:session:1"),
                        ),
                    ),
                ),
            ),
            queued_at=sample_time(6),
        )

        invalidated = runtime.invalidate_dirty_caches(invalidated_at=sample_time(7))

        self.assertEqual(len(invalidated), 1)
        self.assertIsNone(runtime.read_warm_cache(session_id="session:1"))
        stored = PrefetchRepository(connection).read(session_id="session:1")
        self.assertEqual(stored.status, PrefetchStatus.INVALIDATED)
        self.assertEqual(stored.invalidated_at, sample_time(7))


if __name__ == "__main__":
    unittest.main()
