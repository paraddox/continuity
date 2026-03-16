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

from continuity.compiler import (
    CompiledArtifactKind,
    CompilerDependency,
    CompilerNode,
    CompilerNodeCategory,
    DependencyRole,
    DerivedArtifactKind,
    DirtyReason,
    SourceInputKind,
    UtilityStateKind,
    detect_fingerprint_changes,
    plan_incremental_rebuild,
)


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


class CompilerContractTests(unittest.TestCase):
    def test_compiler_node_categories_are_explicitly_separated(self) -> None:
        self.assertEqual(
            {category.value for category in CompilerNodeCategory},
            {"source_input", "derived_ir", "utility_state", "compiled_artifact"},
        )
        self.assertEqual(
            {kind.value for kind in CompiledArtifactKind},
            {
                "state_view",
                "timeline_view",
                "set_view",
                "profile_view",
                "prompt_view",
                "evidence_view",
                "answer_view",
                "vector_index_record",
            },
        )

        with self.assertRaises(ValueError):
            CompilerNode(
                node_id="claim-1",
                category=CompilerNodeCategory.COMPILED_ARTIFACT,
                kind=DerivedArtifactKind.CLAIM,
                fingerprint="claim@1",
            )

    def test_fingerprint_detection_classifies_source_claim_utility_and_policy_changes(self) -> None:
        previous = (
            source_node("obs-1", SourceInputKind.OBSERVATION, "obs@1", subject_id="subject:alice", locus_key="pref/coffee"),
            derived_node("claim-1", DerivedArtifactKind.CLAIM, "claim@1", subject_id="subject:alice", locus_key="pref/coffee"),
            utility_node("utility-1", "utility@1", subject_id="subject:alice", locus_key="pref/coffee"),
            source_node("policy-hermes", SourceInputKind.POLICY_PACK, "policy@1"),
        )
        current = (
            source_node("obs-1", SourceInputKind.OBSERVATION, "obs@2", subject_id="subject:alice", locus_key="pref/coffee"),
            derived_node("claim-1", DerivedArtifactKind.CLAIM, "claim@2", subject_id="subject:alice", locus_key="pref/coffee"),
            utility_node("utility-1", "utility@2", subject_id="subject:alice", locus_key="pref/coffee"),
            source_node("policy-hermes", SourceInputKind.POLICY_PACK, "policy@2"),
        )

        changes = detect_fingerprint_changes(previous_nodes=previous, current_nodes=current)

        self.assertEqual(
            [(change.node_id, change.reason.value) for change in changes],
            [
                ("claim-1", "claim_corrected"),
                ("obs-1", "source_edited"),
                ("policy-hermes", "policy_upgraded"),
                ("utility-1", "utility_input_changed"),
            ],
        )

    def test_rebuild_plan_is_subject_and_locus_scoped(self) -> None:
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

        self.assertEqual(
            plan.rebuild_order,
            ("claim:alice:coffee", "state:alice:coffee", "profile:alice"),
        )
        self.assertEqual(plan.affected_subject_ids, ("subject:alice",))
        self.assertEqual(plan.affected_locus_keys, ("pref/coffee",))

        profile_dirty = plan.dirty_node("profile:alice")
        self.assertEqual(
            profile_dirty.cause_paths,
            (("obs:alice:coffee", "claim:alice:coffee", "state:alice:coffee", "profile:alice"),),
        )
        self.assertEqual(profile_dirty.reasons, (DirtyReason.SOURCE_EDITED,))

    def test_rebuild_plan_keeps_policy_and_utility_edges_explicit(self) -> None:
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

        prompt_dirty = plan.dirty_node("prompt:alice")
        answer_dirty = plan.dirty_node("answer:alice")

        self.assertEqual(prompt_dirty.reasons, (DirtyReason.POLICY_UPGRADED, DirtyReason.UTILITY_INPUT_CHANGED))
        self.assertEqual(answer_dirty.reasons, (DirtyReason.POLICY_UPGRADED,))
        self.assertEqual(
            plan.rebuild_order,
            ("answer:alice", "prompt:alice"),
        )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_compiler_fingerprints_and_dirty_reasons(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("incremental memory compiler", text)
        self.assertIn("fingerprint", text)
        self.assertIn("dirty reasons", text)
        self.assertIn("utility_input_changed", text)
        self.assertIn("policy_upgraded", text)


if __name__ == "__main__":
    unittest.main()
