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

from continuity.policy import hermes_v1_policy_pack
from continuity.store.claims import AdmissionOutcome
from continuity.tiers import (
    ArchivalArtifactKind,
    MemoryTier,
    RebuildUrgency,
    RetrievalBias,
    SnapshotResidency,
    hermes_v1_tier_policy,
    initial_tier_for_claim_type,
    tier_rule_for,
)
from continuity.views import ViewKind


class TierContractTests(unittest.TestCase):
    def test_v1_tiers_are_small_closed_and_policy_driven(self) -> None:
        self.assertEqual(
            {tier.value for tier in MemoryTier},
            {"hot", "warm", "cold", "frozen"},
        )

        hot = tier_rule_for(MemoryTier.HOT)
        self.assertEqual(hot.retrieval_bias, RetrievalBias.PRIMARY)
        self.assertEqual(hot.rebuild_urgency, RebuildUrgency.IMMEDIATE)
        self.assertEqual(hot.snapshot_residency, SnapshotResidency.ACTIVE)
        self.assertTrue(hot.default_in_host_reads)

        frozen = tier_rule_for(MemoryTier.FROZEN)
        self.assertEqual(frozen.retrieval_bias, RetrievalBias.AUDIT_ONLY)
        self.assertEqual(frozen.rebuild_urgency, RebuildUrgency.ARCHIVAL)
        self.assertEqual(frozen.snapshot_residency, SnapshotResidency.ARCHIVAL_ONLY)
        self.assertFalse(frozen.default_in_host_reads)

    def test_initial_claim_tiers_begin_only_after_durable_admission(self) -> None:
        policy = hermes_v1_policy_pack()

        self.assertEqual(
            initial_tier_for_claim_type(
                "instruction",
                admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                policy=policy,
            ),
            MemoryTier.HOT,
        )
        self.assertEqual(
            initial_tier_for_claim_type(
                "preference",
                admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                policy=policy,
            ),
            MemoryTier.WARM,
        )
        self.assertEqual(
            initial_tier_for_claim_type(
                "open_question",
                admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                policy=policy,
            ),
            MemoryTier.HOT,
        )

        with self.assertRaises(ValueError):
            initial_tier_for_claim_type(
                "open_question",
                admission_outcome=AdmissionOutcome.NEEDS_CONFIRMATION,
                policy=policy,
            )

        with self.assertRaises(ValueError):
            initial_tier_for_claim_type(
                "ephemeral_context",
                admission_outcome=AdmissionOutcome.PROMPT_ONLY,
                policy=policy,
            )

    def test_compiled_views_and_archival_artifacts_have_explicit_default_tiers(self) -> None:
        tier_policy = hermes_v1_tier_policy()

        self.assertEqual(tier_policy.view_tiers[ViewKind.PROMPT], (MemoryTier.HOT, MemoryTier.WARM))
        self.assertEqual(tier_policy.view_tiers[ViewKind.TIMELINE], (MemoryTier.WARM, MemoryTier.COLD))
        self.assertEqual(
            tier_policy.archival_tiers[ArchivalArtifactKind.REPLAY_RECORD],
            MemoryTier.FROZEN,
        )
        self.assertEqual(
            tier_policy.archival_tiers[ArchivalArtifactKind.SNAPSHOT_HISTORY],
            MemoryTier.FROZEN,
        )

        cold = tier_rule_for(MemoryTier.COLD)
        self.assertEqual(cold.snapshot_residency, SnapshotResidency.RECALLABLE)
        self.assertFalse(cold.default_in_host_reads)
        self.assertTrue(cold.expunge_guarded)


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_generational_tiering_layer(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("generational memory tiering layer", text)
        self.assertIn("hot", text)
        self.assertIn("warm", text)
        self.assertIn("cold", text)
        self.assertIn("frozen", text)
        self.assertIn("utility-driven promotion, demotion, and pruning bias", text)


if __name__ == "__main__":
    unittest.main()
