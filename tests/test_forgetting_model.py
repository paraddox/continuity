#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


class ForgettingContractTests(unittest.TestCase):
    def test_forgetting_modes_targets_and_surfaces_are_explicit(self) -> None:
        self.assertEqual(
            {mode.value for mode in ForgettingMode},
            {"supersede", "suppress", "seal", "expunge"},
        )
        self.assertEqual(
            {target.value for target in ForgettingTargetKind},
            {"claim", "locus", "subject", "session", "imported_artifact", "compiled_view"},
        )
        self.assertEqual(
            {surface.value for surface in ForgettingSurface},
            {
                "claim_ledger",
                "observation_log",
                "vector_index",
                "snapshot_store",
                "prefetch_cache",
                "replay_artifacts",
                "archive_tier",
                "import_pipeline",
                "derivation_pipeline",
                "tombstone_ledger",
            },
        )

    def test_suppress_withdraws_host_reads_but_keeps_auditable_content(self) -> None:
        rule = forgetting_rule_for(ForgettingMode.SUPPRESS)

        self.assertTrue(rule.host_reads_withdrawn)
        self.assertEqual(rule.residency_for(ForgettingSurface.CLAIM_LEDGER), ArtifactResidency.RETAIN_CONTENT)
        self.assertEqual(rule.residency_for(ForgettingSurface.OBSERVATION_LOG), ArtifactResidency.RETAIN_CONTENT)
        self.assertEqual(rule.residency_for(ForgettingSurface.VECTOR_INDEX), ArtifactResidency.REMOVED)
        self.assertEqual(rule.residency_for(ForgettingSurface.SNAPSHOT_STORE), ArtifactResidency.REMOVED)
        self.assertEqual(rule.residency_for(ForgettingSurface.REPLAY_ARTIFACTS), ArtifactResidency.RETAIN_CONTENT)

    def test_seal_removes_host_visible_payloads_and_keeps_only_minimal_traceability(self) -> None:
        rule = forgetting_rule_for(ForgettingMode.SEAL)

        self.assertTrue(rule.host_reads_withdrawn)
        self.assertEqual(
            rule.residency_for(ForgettingSurface.CLAIM_LEDGER),
            ArtifactResidency.ADMIN_METADATA_ONLY,
        )
        self.assertEqual(
            rule.residency_for(ForgettingSurface.OBSERVATION_LOG),
            ArtifactResidency.ADMIN_METADATA_ONLY,
        )
        self.assertEqual(rule.residency_for(ForgettingSurface.REPLAY_ARTIFACTS), ArtifactResidency.REMOVED)
        self.assertEqual(rule.residency_for(ForgettingSurface.TOMBSTONE_LEDGER), ArtifactResidency.ADMIN_METADATA_ONLY)

    def test_expunge_removes_recoverable_content_and_blocks_resurrection_paths(self) -> None:
        trace = ForgettingDecisionTrace(
            operation=ForgettingOperation(
                operation_id="forget-1",
                target=ForgettingTarget(
                    target_kind=ForgettingTargetKind.SUBJECT,
                    target_id="subject:user:alice",
                ),
                mode=ForgettingMode.EXPUNGE,
                requested_by="subject:user:alice",
                rationale="delete all recoverable traces",
                policy_stamp="hermes_v1@1.0.0",
                recorded_at=sample_time(),
            ),
            rule=forgetting_rule_for(ForgettingMode.EXPUNGE),
        )

        self.assertTrue(trace.rule.host_reads_withdrawn)
        self.assertEqual(trace.residency_for(ForgettingSurface.CLAIM_LEDGER), ArtifactResidency.REMOVED)
        self.assertEqual(trace.residency_for(ForgettingSurface.OBSERVATION_LOG), ArtifactResidency.REMOVED)
        self.assertEqual(trace.residency_for(ForgettingSurface.TOMBSTONE_LEDGER), ArtifactResidency.TOMBSTONE_ONLY)
        self.assertTrue(trace.blocks_resurrection(ForgettingSurface.IMPORT_PIPELINE))
        self.assertTrue(trace.blocks_resurrection(ForgettingSurface.REPLAY_ARTIFACTS))
        self.assertTrue(trace.blocks_resurrection(ForgettingSurface.DERIVATION_PIPELINE))

    def test_supersede_remains_a_revision_flow_not_an_erasure_flow(self) -> None:
        trace = ForgettingDecisionTrace(
            operation=ForgettingOperation(
                operation_id="forget-2",
                target=ForgettingTarget(
                    target_kind=ForgettingTargetKind.CLAIM,
                    target_id="claim-42",
                ),
                mode=ForgettingMode.SUPERSEDE,
                requested_by="subject:assistant:continuity",
                rationale="newer correction replaces prior claim",
                policy_stamp="hermes_v1@1.0.0",
                recorded_at=sample_time(),
            ),
            rule=forgetting_rule_for(ForgettingMode.SUPERSEDE),
        )

        self.assertFalse(trace.rule.host_reads_withdrawn)
        self.assertEqual(trace.residency_for(ForgettingSurface.CLAIM_LEDGER), ArtifactResidency.RETAIN_CONTENT)
        self.assertEqual(trace.residency_for(ForgettingSurface.SNAPSHOT_STORE), ArtifactResidency.RETAIN_CONTENT)
        self.assertFalse(trace.blocks_resurrection(ForgettingSurface.IMPORT_PIPELINE))


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_forgetting_modes_and_resurrection_guards(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("forgetting / retraction / erasure contract", text)
        self.assertIn("suppress", text)
        self.assertIn("seal", text)
        self.assertIn("expunge", text)
        self.assertIn("replay avoid resurrecting expunged content", text)
        self.assertIn("tombstones", text)


if __name__ == "__main__":
    unittest.main()
