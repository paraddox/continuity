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

import continuity.compiler as compiler_module
from continuity.compiler import (
    CompiledArtifactKind,
    CompilerDependency,
    CompilerNode,
    CompilerNodeCategory,
    DependencyRole,
    DerivedArtifactKind,
    SourceInputKind,
    UtilityStateKind,
    detect_fingerprint_changes,
    plan_incremental_rebuild,
)
from continuity.store.schema import apply_migrations


CompilerStateRepository = getattr(compiler_module, "CompilerStateRepository", None)
DirtyQueueStatus = getattr(compiler_module, "DirtyQueueStatus", None)


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


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


def utility_node(
    node_id: str,
    fingerprint: str,
    *,
    subject_id: str | None = None,
    locus_key: str | None = None,
) -> CompilerNode:
    return CompilerNode(
        node_id=node_id,
        category=CompilerNodeCategory.UTILITY_STATE,
        kind=UtilityStateKind.COMPILED_WEIGHT,
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


class CompilerStateRepositoryTests(unittest.TestCase):
    def test_repository_round_trips_nodes_dependencies_and_dirty_queue(self) -> None:
        self.assertIsNotNone(CompilerStateRepository)
        self.assertIsNotNone(DirtyQueueStatus)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = CompilerStateRepository(connection)

        previous = (
            source_node("policy-hermes", SourceInputKind.POLICY_PACK, "policy@1"),
            utility_node("utility:alice", "utility@1", subject_id="subject:alice"),
            compiled_node("prompt:alice", CompiledArtifactKind.PROMPT_VIEW, "prompt@1", subject_id="subject:alice"),
            compiled_node("answer:alice", CompiledArtifactKind.ANSWER_VIEW, "answer@1", subject_id="subject:alice"),
        )
        current = (
            source_node("policy-hermes", SourceInputKind.POLICY_PACK, "policy@2"),
            utility_node("utility:alice", "utility@2", subject_id="subject:alice"),
            compiled_node("prompt:alice", CompiledArtifactKind.PROMPT_VIEW, "prompt@1", subject_id="subject:alice"),
            compiled_node("answer:alice", CompiledArtifactKind.ANSWER_VIEW, "answer@1", subject_id="subject:alice"),
        )
        dependencies = (
            CompilerDependency("policy-hermes", "prompt:alice", DependencyRole.POLICY),
            CompilerDependency("policy-hermes", "answer:alice", DependencyRole.POLICY),
            CompilerDependency("utility:alice", "prompt:alice", DependencyRole.UTILITY),
        )
        plan = plan_incremental_rebuild(
            nodes=current,
            dependencies=dependencies,
            changes=detect_fingerprint_changes(previous_nodes=previous, current_nodes=current),
        )

        repository.upsert_nodes(current)
        repository.replace_dependencies(dependencies)
        repository.enqueue_dirty_nodes(plan.dirty_nodes, queued_at=sample_time())

        self.assertEqual(repository.list_nodes(), current)
        self.assertEqual(repository.list_dependencies(), dependencies)
        self.assertEqual(
            repository.list_dependencies(role=DependencyRole.UTILITY),
            (CompilerDependency("utility:alice", "prompt:alice", DependencyRole.UTILITY),),
        )
        self.assertEqual(repository.list_dirty_nodes(), plan.dirty_nodes)
        self.assertEqual(repository.read_rebuild_plan().rebuild_order, plan.rebuild_order)

    def test_repository_queries_affected_subjects_loci_and_rebuild_order(self) -> None:
        self.assertIsNotNone(CompilerStateRepository)
        self.assertIsNotNone(DirtyQueueStatus)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = CompilerStateRepository(connection)

        previous = (
            source_node("obs:alice:coffee", SourceInputKind.OBSERVATION, "obs@1", subject_id="subject:alice", locus_key="pref/coffee"),
            source_node("obs:bob:tea", SourceInputKind.OBSERVATION, "obs@1", subject_id="subject:bob", locus_key="pref/tea"),
            derived_node("claim:alice:coffee", DerivedArtifactKind.CLAIM, "claim@1", subject_id="subject:alice", locus_key="pref/coffee"),
            derived_node("claim:bob:tea", DerivedArtifactKind.CLAIM, "claim@1", subject_id="subject:bob", locus_key="pref/tea"),
            compiled_node("state:alice:coffee", CompiledArtifactKind.STATE_VIEW, "state@1", subject_id="subject:alice", locus_key="pref/coffee"),
            compiled_node("state:bob:tea", CompiledArtifactKind.STATE_VIEW, "state@1", subject_id="subject:bob", locus_key="pref/tea"),
            compiled_node("profile:alice", CompiledArtifactKind.PROFILE_VIEW, "profile@1", subject_id="subject:alice"),
            compiled_node("profile:bob", CompiledArtifactKind.PROFILE_VIEW, "profile@1", subject_id="subject:bob"),
        )
        current = (
            source_node("obs:alice:coffee", SourceInputKind.OBSERVATION, "obs@2", subject_id="subject:alice", locus_key="pref/coffee"),
            source_node("obs:bob:tea", SourceInputKind.OBSERVATION, "obs@1", subject_id="subject:bob", locus_key="pref/tea"),
            derived_node("claim:alice:coffee", DerivedArtifactKind.CLAIM, "claim@1", subject_id="subject:alice", locus_key="pref/coffee"),
            derived_node("claim:bob:tea", DerivedArtifactKind.CLAIM, "claim@1", subject_id="subject:bob", locus_key="pref/tea"),
            compiled_node("state:alice:coffee", CompiledArtifactKind.STATE_VIEW, "state@1", subject_id="subject:alice", locus_key="pref/coffee"),
            compiled_node("state:bob:tea", CompiledArtifactKind.STATE_VIEW, "state@1", subject_id="subject:bob", locus_key="pref/tea"),
            compiled_node("profile:alice", CompiledArtifactKind.PROFILE_VIEW, "profile@1", subject_id="subject:alice"),
            compiled_node("profile:bob", CompiledArtifactKind.PROFILE_VIEW, "profile@1", subject_id="subject:bob"),
        )
        dependencies = (
            CompilerDependency("obs:alice:coffee", "claim:alice:coffee", DependencyRole.CONTENT),
            CompilerDependency("obs:bob:tea", "claim:bob:tea", DependencyRole.CONTENT),
            CompilerDependency("claim:alice:coffee", "state:alice:coffee", DependencyRole.PROJECTION),
            CompilerDependency("claim:bob:tea", "state:bob:tea", DependencyRole.PROJECTION),
            CompilerDependency("state:alice:coffee", "profile:alice", DependencyRole.PROJECTION),
            CompilerDependency("state:bob:tea", "profile:bob", DependencyRole.PROJECTION),
        )
        plan = plan_incremental_rebuild(
            nodes=current,
            dependencies=dependencies,
            changes=detect_fingerprint_changes(previous_nodes=previous, current_nodes=current),
        )

        repository.upsert_nodes(current)
        repository.replace_dependencies(dependencies)
        repository.enqueue_dirty_nodes(plan.dirty_nodes, queued_at=sample_time())

        self.assertEqual(repository.list_affected_subject_ids(), ("subject:alice",))
        self.assertEqual(
            repository.list_affected_locus_keys(subject_id="subject:alice"),
            ("pref/coffee",),
        )
        self.assertEqual(
            repository.read_rebuild_plan(subject_id="subject:alice").rebuild_order,
            ("claim:alice:coffee", "state:alice:coffee", "profile:alice"),
        )


if __name__ == "__main__":
    unittest.main()
