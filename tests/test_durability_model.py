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

from continuity.forgetting import ForgettingMode
from continuity.transactions import (
    DurabilityWaterline,
    HostOperation,
    host_operation_contract_for,
    minimum_forgetting_waterline,
    transaction_contract_for,
)


class DurabilityContractTests(unittest.TestCase):
    def test_waterlines_are_closed_and_monotonic(self) -> None:
        self.assertEqual(
            {waterline.value for waterline in DurabilityWaterline},
            {
                "observation_committed",
                "claim_committed",
                "views_compiled",
                "snapshot_published",
                "prefetch_warmed",
            },
        )

        self.assertLess(
            DurabilityWaterline.OBSERVATION_COMMITTED.rank,
            DurabilityWaterline.CLAIM_COMMITTED.rank,
        )
        self.assertLess(
            DurabilityWaterline.CLAIM_COMMITTED.rank,
            DurabilityWaterline.VIEWS_COMPILED.rank,
        )
        self.assertLess(
            DurabilityWaterline.VIEWS_COMPILED.rank,
            DurabilityWaterline.SNAPSHOT_PUBLISHED.rank,
        )
        self.assertLess(
            DurabilityWaterline.SNAPSHOT_PUBLISHED.rank,
            DurabilityWaterline.PREFETCH_WARMED.rank,
        )

    def test_host_operations_map_to_explicit_minimum_waterlines(self) -> None:
        self.assertEqual(
            host_operation_contract_for(HostOperation.SAVE_TURN, write_frequency="async").minimum_waterline,
            DurabilityWaterline.OBSERVATION_COMMITTED,
        )
        self.assertEqual(
            host_operation_contract_for(HostOperation.SAVE_TURN, write_frequency="turn").minimum_waterline,
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )
        self.assertEqual(
            host_operation_contract_for(HostOperation.WRITE_CONCLUSION).minimum_waterline,
            DurabilityWaterline.VIEWS_COMPILED,
        )
        self.assertEqual(
            host_operation_contract_for(HostOperation.IMPORT_HISTORY).minimum_waterline,
            DurabilityWaterline.VIEWS_COMPILED,
        )
        self.assertEqual(
            host_operation_contract_for(HostOperation.PUBLISH_SNAPSHOT).minimum_waterline,
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )
        self.assertEqual(
            host_operation_contract_for(HostOperation.PREFETCH_NEXT_TURN).minimum_waterline,
            DurabilityWaterline.PREFETCH_WARMED,
        )
        self.assertEqual(
            host_operation_contract_for(HostOperation.READ_PROMPT).minimum_waterline,
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )

    def test_forgetting_withdrawal_modes_require_snapshot_publication(self) -> None:
        self.assertEqual(
            minimum_forgetting_waterline(ForgettingMode.SUPERSEDE),
            DurabilityWaterline.VIEWS_COMPILED,
        )
        self.assertEqual(
            minimum_forgetting_waterline(ForgettingMode.SUPPRESS),
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )
        self.assertEqual(
            minimum_forgetting_waterline(ForgettingMode.SEAL),
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )
        self.assertEqual(
            minimum_forgetting_waterline(ForgettingMode.EXPUNGE),
            DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )

    def test_transaction_contracts_only_claim_waterlines_they_can_reach(self) -> None:
        self.assertTrue(
            transaction_contract_for(HostOperation.WRITE_CONCLUSION.transaction_kind).supports_waterline(
                DurabilityWaterline.VIEWS_COMPILED
            )
        )
        self.assertTrue(
            transaction_contract_for(HostOperation.PREFETCH_NEXT_TURN.transaction_kind).supports_waterline(
                DurabilityWaterline.PREFETCH_WARMED
            )
        )
        self.assertFalse(
            transaction_contract_for(HostOperation.PREFETCH_NEXT_TURN.transaction_kind).supports_waterline(
                DurabilityWaterline.CLAIM_COMMITTED
            )
        )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_durability_contract_and_best_effort_prefetch(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("durability contract", text)
        self.assertIn("observation_committed", text)
        self.assertIn("claim_committed", text)
        self.assertIn("views_compiled", text)
        self.assertIn("snapshot_published", text)
        self.assertIn("prefetch_warmed", text)
        self.assertIn("writefrequency", text)
        self.assertIn("best-effort", text)


if __name__ == "__main__":
    unittest.main()
