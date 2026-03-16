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

from continuity.arbiter import ArbiterPublication, ArbiterPublicationKind, OffLaneWork
from continuity.transactions import DurabilityWaterline, TransactionKind, TransactionPhase


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


class ArbiterContractTests(unittest.TestCase):
    def test_v1_arbiter_boundaries_are_explicit(self) -> None:
        self.assertEqual(
            {work.value for work in OffLaneWork},
            {
                "embedding_generation",
                "claim_derivation",
                "view_compilation",
                "prefetch_preparation",
            },
        )
        self.assertEqual(
            {kind.value for kind in ArbiterPublicationKind},
            {
                "observation_commit",
                "claim_commit",
                "belief_revision",
                "forgetting_publication",
                "view_publication",
                "work_status_transition",
                "snapshot_head_promotion",
                "durability_signal",
                "outcome_recording",
            },
        )

    def test_snapshot_publication_keeps_lane_order_snapshot_head_and_waterline_aligned(self) -> None:
        publication = ArbiterPublication(
            lane_position=17,
            publication_kind=ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,
            transaction_kind=TransactionKind.PUBLISH_SNAPSHOT,
            phase=TransactionPhase.PUBLISH_SNAPSHOT,
            object_ids=("snapshot:candidate:17", "snapshot:active"),
            published_at=sample_time(),
            snapshot_head_id="snapshot:active",
            reached_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )

        self.assertEqual(publication.replay_order_key, "arbiter:17")
        self.assertTrue(publication.requires_journal_entry)
        self.assertTrue(publication.is_snapshot_publication)

    def test_durability_signals_must_match_transaction_reachability(self) -> None:
        with self.assertRaises(ValueError):
            ArbiterPublication(
                lane_position=3,
                publication_kind=ArbiterPublicationKind.DURABILITY_SIGNAL,
                transaction_kind=TransactionKind.PREFETCH_NEXT_TURN,
                phase=TransactionPhase.PREFETCH,
                object_ids=("snapshot:active",),
                published_at=sample_time(),
                reached_waterline=DurabilityWaterline.CLAIM_COMMITTED,
            )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_mutation_arbiter_contract(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("mutation arbiter", text)
        self.assertIn("serialized commit lane", text)
        self.assertIn("off-lane computation", text)
        self.assertIn("snapshot-head promotion", text)
        self.assertIn("durability-waterline completion signaling", text)
        self.assertIn("arbiter order", text)


if __name__ == "__main__":
    unittest.main()
