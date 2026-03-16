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

from continuity.arbiter import ArbiterPublication, ArbiterPublicationKind
from continuity.events import EventPayloadMode, SystemEvent, SystemEventType
from continuity.transactions import DurabilityWaterline, TransactionKind, TransactionPhase


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


def snapshot_publication() -> ArbiterPublication:
    return ArbiterPublication(
        lane_position=5,
        publication_kind=ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,
        transaction_kind=TransactionKind.PUBLISH_SNAPSHOT,
        phase=TransactionPhase.PUBLISH_SNAPSHOT,
        object_ids=("snapshot:candidate:5", "snapshot:active"),
        published_at=sample_time(),
        snapshot_head_id="snapshot:active",
        reached_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
    )


class EventJournalContractTests(unittest.TestCase):
    def test_v1_event_types_and_payload_modes_are_closed(self) -> None:
        self.assertEqual(
            {event_type.value for event_type in SystemEventType},
            {
                "observation_ingested",
                "claim_committed",
                "belief_revised",
                "memory_forgotten",
                "view_compiled",
                "snapshot_published",
                "outcome_recorded",
            },
        )
        self.assertEqual(
            {mode.value for mode in EventPayloadMode},
            {"inline", "reference", "mixed"},
        )

    def test_journal_entries_link_append_only_order_back_to_arbiter_publication(self) -> None:
        event = SystemEvent.from_publication(
            journal_position=21,
            event_type=SystemEventType.SNAPSHOT_PUBLISHED,
            publication=snapshot_publication(),
            payload_mode=EventPayloadMode.REFERENCE,
            recorded_at=sample_time(),
            object_ids=("snapshot:active",),
            reference_ids=("snapshot:active", "replay:journal:21"),
        )

        self.assertEqual(event.arbiter_lane_position, 5)
        self.assertEqual(event.waterline, DurabilityWaterline.SNAPSHOT_PUBLISHED)
        self.assertEqual(event.append_only_key, (21, 5))
        self.assertTrue(event.has_reference_payload)
        self.assertFalse(event.has_inline_payload)

    def test_payload_modes_make_inline_vs_reference_storage_explicit(self) -> None:
        inline_event = SystemEvent(
            journal_position=8,
            event_type=SystemEventType.OUTCOME_RECORDED,
            transaction_kind=TransactionKind.WRITE_CONCLUSION,
            arbiter_lane_position=4,
            payload_mode=EventPayloadMode.INLINE,
            recorded_at=sample_time(),
            object_ids=("outcome:1",),
            inline_payload=("user_confirmed", "claim:1"),
        )
        mixed_event = SystemEvent(
            journal_position=9,
            event_type=SystemEventType.CLAIM_COMMITTED,
            transaction_kind=TransactionKind.INGEST_TURN,
            arbiter_lane_position=5,
            payload_mode=EventPayloadMode.MIXED,
            recorded_at=sample_time(),
            object_ids=("claim:1",),
            inline_payload=("claim:1",),
            reference_ids=("observation:1",),
            waterline=DurabilityWaterline.CLAIM_COMMITTED,
        )

        self.assertTrue(inline_event.has_inline_payload)
        self.assertFalse(inline_event.has_reference_payload)
        self.assertTrue(mixed_event.has_inline_payload)
        self.assertTrue(mixed_event.has_reference_payload)

        with self.assertRaises(ValueError):
            SystemEvent(
                journal_position=10,
                event_type=SystemEventType.VIEW_COMPILED,
                transaction_kind=TransactionKind.COMPILE_VIEWS,
                arbiter_lane_position=6,
                payload_mode=EventPayloadMode.INLINE,
                recorded_at=sample_time(),
                object_ids=("view:prompt:1",),
                reference_ids=("snapshot:active",),
            )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_system_event_journal_contract(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("system event journal", text)
        self.assertIn("append-only", text)
        self.assertIn("observation_ingested", text)
        self.assertIn("snapshot_published", text)
        self.assertIn("outcome_recorded", text)
        self.assertIn("payload or references", text)
        self.assertIn("journal order", text)


if __name__ == "__main__":
    unittest.main()
