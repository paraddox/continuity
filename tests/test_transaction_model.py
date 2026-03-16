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

from continuity.transactions import (
    PrefetchBehavior,
    TransactionKind,
    TransactionPhase,
    WriteFrequencySchedule,
    transaction_contract_for,
    transaction_contracts,
    write_frequency_policy_for,
)


class TransactionContractTests(unittest.TestCase):
    def test_transaction_kinds_are_closed_and_named(self) -> None:
        self.assertEqual(
            {kind.value for kind in TransactionKind},
            {
                "ingest_turn",
                "write_conclusion",
                "forget_memory",
                "import_history",
                "compile_views",
                "publish_snapshot",
                "prefetch_next_turn",
            },
        )

    def test_contract_registry_covers_each_transaction_and_ingest_order(self) -> None:
        contracts = transaction_contracts()

        self.assertEqual(set(contracts), set(TransactionKind))

        ingest = transaction_contract_for(TransactionKind.INGEST_TURN)
        self.assertEqual(
            ingest.phases,
            (
                TransactionPhase.NORMALIZE_OBSERVATIONS,
                TransactionPhase.COMMIT_OBSERVATIONS,
                TransactionPhase.RESOLVE_SUBJECTS,
                TransactionPhase.DERIVE_CANDIDATES,
                TransactionPhase.RUN_ADMISSION,
                TransactionPhase.RECORD_NON_DURABLE_CONTEXT,
                TransactionPhase.ASSIGN_LOCI,
                TransactionPhase.COMMIT_CLAIMS,
                TransactionPhase.REVISE_BELIEFS,
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
                TransactionPhase.PREFETCH,
            ),
        )
        self.assertEqual(ingest.prefetch_behavior, PrefetchBehavior.ENQUEUE)
        self.assertTrue(ingest.may_publish_snapshot)

    def test_forget_and_prefetch_transactions_keep_their_own_boundaries(self) -> None:
        forget = transaction_contract_for(TransactionKind.FORGET_MEMORY)
        self.assertEqual(
            forget.phases,
            (
                TransactionPhase.RESOLVE_FORGETTING,
                TransactionPhase.COMMIT_CLAIMS,
                TransactionPhase.REVISE_BELIEFS,
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
            ),
        )

        prefetch = transaction_contract_for(TransactionKind.PREFETCH_NEXT_TURN)
        self.assertEqual(prefetch.phases, (TransactionPhase.PREFETCH,))
        self.assertEqual(prefetch.prefetch_behavior, PrefetchBehavior.WARM_ONLY)
        self.assertFalse(prefetch.may_publish_snapshot)

    def test_write_frequency_maps_to_explicit_timing_policies(self) -> None:
        async_policy = write_frequency_policy_for("async")
        self.assertEqual(async_policy.schedule, WriteFrequencySchedule.PER_TURN)
        self.assertEqual(async_policy.batch_size, None)
        self.assertEqual(async_policy.trigger_transaction, TransactionKind.INGEST_TURN)
        self.assertEqual(async_policy.flush_on, "background")

        turn_policy = write_frequency_policy_for("turn")
        self.assertEqual(turn_policy.schedule, WriteFrequencySchedule.PER_TURN)
        self.assertEqual(turn_policy.flush_on, "same_turn")

        session_policy = write_frequency_policy_for("session")
        self.assertEqual(session_policy.schedule, WriteFrequencySchedule.SESSION_END)
        self.assertEqual(session_policy.flush_on, "session_end")

        batch_policy = write_frequency_policy_for(5)
        self.assertEqual(batch_policy.schedule, WriteFrequencySchedule.BATCHED_TURNS)
        self.assertEqual(batch_policy.batch_size, 5)
        self.assertEqual(batch_policy.flush_on, "turn_threshold")

        with self.assertRaises(ValueError):
            write_frequency_policy_for(0)


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_transaction_pipeline_and_phase_order(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("memory transaction pipeline", text)
        self.assertIn("ingest_turn", text)
        self.assertIn("write_conclusion", text)
        self.assertIn("forget_memory", text)
        self.assertIn("publish_snapshot", text)
        self.assertIn("prefetch_next_turn", text)
        self.assertIn("normalize observations", text)
        self.assertIn("capture replay", text)


if __name__ == "__main__":
    unittest.main()
