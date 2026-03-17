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

import continuity.compiler as compiler_module
from continuity.compiler import (
    CompiledArtifactKind,
    CompilerDependency,
    CompilerNode,
    CompilerNodeCategory,
    CompilerStateRepository,
    DependencyRole,
    DerivedArtifactKind,
    DirtyQueueStatus,
    SourceInputKind,
)
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
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
)


IncrementalRebuildPlanner = getattr(compiler_module, "IncrementalRebuildPlanner", None)
RebuildArtifact = getattr(compiler_module, "RebuildArtifact", None)
StagedRebuild = getattr(compiler_module, "StagedRebuild", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def source_node(
    node_id: str,
    kind: SourceInputKind,
    fingerprint: str,
    *,
    subject_id: str | None = None,
    locus_key: str | None = None,
) -> CompilerNode:
    return CompilerNode(
        node_id=node_id,
        category=CompilerNodeCategory.SOURCE_INPUT,
        kind=kind,
        fingerprint=fingerprint,
        subject_id=subject_id,
        locus_key=locus_key,
    )


def derived_node(
    node_id: str,
    kind: DerivedArtifactKind,
    fingerprint: str,
    *,
    subject_id: str | None = None,
    locus_key: str | None = None,
) -> CompilerNode:
    return CompilerNode(
        node_id=node_id,
        category=CompilerNodeCategory.DERIVED_IR,
        kind=kind,
        fingerprint=fingerprint,
        subject_id=subject_id,
        locus_key=locus_key,
    )


def compiled_node(
    node_id: str,
    kind: CompiledArtifactKind,
    fingerprint: str,
    *,
    subject_id: str | None = None,
    locus_key: str | None = None,
) -> CompilerNode:
    return CompilerNode(
        node_id=node_id,
        category=CompilerNodeCategory.COMPILED_ARTIFACT,
        kind=kind,
        fingerprint=fingerprint,
        subject_id=subject_id,
        locus_key=locus_key,
    )


class IncrementalRebuildPlannerTests(unittest.TestCase):
    def test_planner_detects_changes_from_stored_state_and_replaces_pending_queue(self) -> None:
        self.assertIsNotNone(IncrementalRebuildPlanner)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = CompilerStateRepository(connection)
        planner = IncrementalRebuildPlanner(connection)

        previous = (
            source_node(
                "obs:alice:coffee",
                SourceInputKind.OBSERVATION,
                "obs@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            source_node(
                "obs:bob:tea",
                SourceInputKind.OBSERVATION,
                "obs@1",
                subject_id="subject:bob",
                locus_key="pref/tea",
            ),
            derived_node(
                "claim:alice:coffee",
                DerivedArtifactKind.CLAIM,
                "claim@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            derived_node(
                "claim:bob:tea",
                DerivedArtifactKind.CLAIM,
                "claim@1",
                subject_id="subject:bob",
                locus_key="pref/tea",
            ),
            compiled_node(
                "view:state:alice:coffee",
                CompiledArtifactKind.STATE_VIEW,
                "state@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            compiled_node(
                "view:state:bob:tea",
                CompiledArtifactKind.STATE_VIEW,
                "state@1",
                subject_id="subject:bob",
                locus_key="pref/tea",
            ),
            compiled_node(
                "view:profile:alice",
                CompiledArtifactKind.PROFILE_VIEW,
                "profile@1",
                subject_id="subject:alice",
            ),
            compiled_node(
                "view:profile:bob",
                CompiledArtifactKind.PROFILE_VIEW,
                "profile@1",
                subject_id="subject:bob",
            ),
        )
        dependencies = (
            CompilerDependency("obs:alice:coffee", "claim:alice:coffee", DependencyRole.CONTENT),
            CompilerDependency("obs:bob:tea", "claim:bob:tea", DependencyRole.CONTENT),
            CompilerDependency("claim:alice:coffee", "view:state:alice:coffee", DependencyRole.PROJECTION),
            CompilerDependency("claim:bob:tea", "view:state:bob:tea", DependencyRole.PROJECTION),
            CompilerDependency("view:state:alice:coffee", "view:profile:alice", DependencyRole.PROJECTION),
            CompilerDependency("view:state:bob:tea", "view:profile:bob", DependencyRole.PROJECTION),
        )
        current = (
            source_node(
                "obs:alice:coffee",
                SourceInputKind.OBSERVATION,
                "obs@2",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            *previous[1:],
        )

        repository.upsert_nodes(previous)
        repository.replace_dependencies(dependencies)

        plan = planner.plan_rebuild(
            nodes=current,
            dependencies=dependencies,
            queued_at=sample_time(),
        )

        self.assertEqual(
            [(change.node_id, change.reason.value) for change in plan.changes],
            [("obs:alice:coffee", "source_edited")],
        )
        self.assertEqual(
            plan.rebuild_order,
            ("claim:alice:coffee", "view:state:alice:coffee", "view:profile:alice"),
        )
        self.assertEqual(plan.affected_subject_ids, ("subject:alice",))
        self.assertEqual(plan.affected_locus_keys, ("pref/coffee",))
        self.assertEqual(repository.list_nodes(), current)
        self.assertEqual(repository.list_dirty_nodes(), plan.dirty_nodes)
        self.assertEqual(repository.read_rebuild_plan().rebuild_order, plan.rebuild_order)

        second_plan = planner.plan_rebuild(
            nodes=current,
            dependencies=dependencies,
            queued_at=sample_time(1),
        )

        self.assertEqual(second_plan.changes, ())
        self.assertEqual(second_plan.dirty_nodes, ())
        self.assertEqual(repository.list_dirty_nodes(), ())

    def test_planner_stages_candidate_snapshot_selectively_and_publishes_rebuild(self) -> None:
        self.assertIsNotNone(IncrementalRebuildPlanner)
        self.assertIsNotNone(RebuildArtifact)
        self.assertIsNotNone(StagedRebuild)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = CompilerStateRepository(connection)
        snapshots = SnapshotRepository(connection)
        planner = IncrementalRebuildPlanner(connection)

        previous = (
            source_node(
                "obs:alice:coffee",
                SourceInputKind.OBSERVATION,
                "obs@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            derived_node(
                "claim:alice:coffee",
                DerivedArtifactKind.CLAIM,
                "claim@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            compiled_node(
                "view:state:alice:coffee",
                CompiledArtifactKind.STATE_VIEW,
                "state@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            compiled_node(
                "view:profile:alice",
                CompiledArtifactKind.PROFILE_VIEW,
                "profile@1",
                subject_id="subject:alice",
            ),
            compiled_node(
                "vector:state:alice:coffee",
                CompiledArtifactKind.VECTOR_INDEX_RECORD,
                "vector@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
        )
        current = (
            source_node(
                "obs:alice:coffee",
                SourceInputKind.OBSERVATION,
                "obs@2",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            derived_node(
                "claim:alice:coffee",
                DerivedArtifactKind.CLAIM,
                "claim@1",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            compiled_node(
                "view:state:alice:coffee",
                CompiledArtifactKind.STATE_VIEW,
                "state@2",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
            compiled_node(
                "view:profile:alice",
                CompiledArtifactKind.PROFILE_VIEW,
                "profile@2",
                subject_id="subject:alice",
            ),
            compiled_node(
                "vector:state:alice:coffee",
                CompiledArtifactKind.VECTOR_INDEX_RECORD,
                "vector@2",
                subject_id="subject:alice",
                locus_key="pref/coffee",
            ),
        )
        dependencies = (
            CompilerDependency("obs:alice:coffee", "claim:alice:coffee", DependencyRole.CONTENT),
            CompilerDependency("claim:alice:coffee", "view:state:alice:coffee", DependencyRole.PROJECTION),
            CompilerDependency("view:state:alice:coffee", "view:profile:alice", DependencyRole.PROJECTION),
            CompilerDependency("view:state:alice:coffee", "vector:state:alice:coffee", DependencyRole.INDEX),
        )

        repository.upsert_nodes(previous)
        repository.replace_dependencies(dependencies)
        plan = planner.plan_rebuild(
            nodes=current,
            dependencies=dependencies,
            queued_at=sample_time(),
        )

        snapshots.save_snapshot(
            MemorySnapshot(
                snapshot_id="snapshot:active",
                policy_stamp="hermes_v1@1.0.0",
                parent_snapshot_id=None,
                created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
                artifact_refs=(
                    SnapshotArtifactRef(SnapshotArtifactKind.STATE_VIEW, "state:alice:coffee:v1"),
                    SnapshotArtifactRef(SnapshotArtifactKind.PROFILE_VIEW, "profile:alice:v1"),
                    SnapshotArtifactRef(SnapshotArtifactKind.PROFILE_VIEW, "profile:bob:v1"),
                    SnapshotArtifactRef(SnapshotArtifactKind.VECTOR_INDEX, "vector:alice:coffee:v1"),
                ),
            )
        )
        snapshots.upsert_head(
            SnapshotHead(
                head_key="current",
                state=SnapshotHeadState.ACTIVE,
                snapshot_id="snapshot:active",
            )
        )

        staged = planner.stage_rebuild(
            plan=plan,
            candidate_snapshot_id="snapshot:candidate:1",
            policy_stamp="hermes_v1@1.0.0",
            published_at=sample_time(1),
            rebuilt_artifacts=(
                RebuildArtifact(
                    source_node_id="view:state:alice:coffee",
                    artifact_ref=SnapshotArtifactRef(
                        SnapshotArtifactKind.STATE_VIEW,
                        "state:alice:coffee:v2",
                    ),
                    supersedes_artifact_ids=("state:alice:coffee:v1",),
                ),
                RebuildArtifact(
                    source_node_id="view:profile:alice",
                    artifact_ref=SnapshotArtifactRef(
                        SnapshotArtifactKind.PROFILE_VIEW,
                        "profile:alice:v2",
                    ),
                    supersedes_artifact_ids=("profile:alice:v1",),
                ),
                RebuildArtifact(
                    source_node_id="vector:state:alice:coffee",
                    artifact_ref=SnapshotArtifactRef(
                        SnapshotArtifactKind.VECTOR_INDEX,
                        "vector:alice:coffee:v2",
                    ),
                    supersedes_artifact_ids=("vector:alice:coffee:v1",),
                ),
            ),
        )

        self.assertEqual(staged.plan.rebuild_order, plan.rebuild_order)
        self.assertEqual(staged.active_snapshot_id, "snapshot:active")
        self.assertEqual(staged.candidate_snapshot.snapshot_id, "snapshot:candidate:1")
        self.assertEqual(
            staged.candidate_snapshot.artifact_refs,
            (
                SnapshotArtifactRef(SnapshotArtifactKind.PROFILE_VIEW, "profile:bob:v1"),
                SnapshotArtifactRef(SnapshotArtifactKind.STATE_VIEW, "state:alice:coffee:v2"),
                SnapshotArtifactRef(SnapshotArtifactKind.PROFILE_VIEW, "profile:alice:v2"),
                SnapshotArtifactRef(SnapshotArtifactKind.VECTOR_INDEX, "vector:alice:coffee:v2"),
            ),
        )
        self.assertEqual(
            snapshots.read_active_snapshot(head_key="current").snapshot_id,
            "snapshot:active",
        )
        self.assertEqual(
            snapshots.read_candidate_snapshot(head_key="current"),
            staged.candidate_snapshot,
        )
        self.assertEqual(staged.publication.transaction_kind, TransactionKind.COMPILE_VIEWS)
        self.assertEqual(staged.publication.phase, TransactionPhase.COMPILE_VIEWS)
        self.assertEqual(staged.publication.reached_waterline, DurabilityWaterline.VIEWS_COMPILED)
        self.assertEqual(staged.event.event_type, SystemEventType.VIEW_COMPILED)
        self.assertEqual(
            staged.event.reference_ids,
            (
                "snapshot:candidate:1",
                "state:alice:coffee:v2",
                "profile:alice:v2",
                "vector:alice:coffee:v2",
            ),
        )
        self.assertEqual(repository.list_dirty_nodes(), ())
        self.assertEqual(
            repository.list_dirty_nodes(status=DirtyQueueStatus.DONE),
            plan.dirty_nodes,
        )


if __name__ == "__main__":
    unittest.main()
