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

from continuity.ontology import MemoryClass, MemoryPartition, PromptRenderStyle
from continuity.policy import get_policy_pack, hermes_v1_policy_pack
from continuity.store.claims import AdmissionOutcome


class PolicyPackInvariantTests(unittest.TestCase):
    def test_hermes_v1_policy_pack_has_explicit_version_stamp(self) -> None:
        policy = hermes_v1_policy_pack()

        self.assertEqual(policy.policy_name, "hermes_v1")
        self.assertEqual(policy.version, "1.0.0")
        self.assertEqual(policy.policy_stamp, "hermes_v1@1.0.0")

    def test_policy_pack_surfaces_admission_rendering_and_retrieval_rules(self) -> None:
        policy = hermes_v1_policy_pack()

        self.assertEqual(
            policy.default_admission_outcome_for("preference"),
            AdmissionOutcome.DURABLE_CLAIM,
        )
        self.assertEqual(
            policy.default_admission_outcome_for("open_question"),
            AdmissionOutcome.NEEDS_CONFIRMATION,
        )
        self.assertEqual(
            policy.prompt_render_style_for("ephemeral_context"),
            PromptRenderStyle.SESSION_NOTE,
        )
        self.assertLess(
            policy.retrieval_rank_for("preference"),
            policy.retrieval_rank_for("ephemeral_context"),
        )

    def test_policy_pack_write_budgets_are_partition_specific(self) -> None:
        policy = hermes_v1_policy_pack()

        self.assertEqual(policy.write_budget_for_partition(MemoryPartition.USER_MEMORY), 8)
        self.assertEqual(policy.write_budget_for_partition(MemoryPartition.SHARED_CONTEXT), 6)
        self.assertEqual(policy.write_budget_for_partition(MemoryPartition.ASSISTANT_MEMORY), 2)
        self.assertEqual(policy.write_budget_for_partition(MemoryPartition.EPHEMERAL_STATE), 0)

    def test_policy_pack_replay_fingerprint_is_versioned_and_comparable(self) -> None:
        policy = hermes_v1_policy_pack()

        self.assertEqual(
            policy.replay_fingerprint(),
            ("hermes_v1@1.0.0", "hermes_core_v1"),
        )

    def test_policy_lookup_rejects_unknown_packs(self) -> None:
        with self.assertRaises(KeyError):
            get_policy_pack("unknown_policy")

    def test_policy_pack_only_exposes_enabled_ontology_types(self) -> None:
        policy = hermes_v1_policy_pack()

        self.assertEqual(policy.enabled_memory_classes(), tuple(MemoryClass))


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_versioned_policy_pack_contract(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("versioned policy packs", text)
        self.assertIn("hermes_v1", text)
        self.assertIn("policy version", text)
        self.assertIn("utility", text)
        self.assertIn("prompt rendering", text)


if __name__ == "__main__":
    unittest.main()
