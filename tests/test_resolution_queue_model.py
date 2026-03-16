#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.resolution_queue import (
    ResolutionAction,
    ResolutionEffect,
    ResolutionPriority,
    ResolutionQueueItem,
    ResolutionRecord,
    ResolutionSource,
    ResolutionStatus,
    ResolutionSurface,
)
from continuity.store.claims import AdmissionOutcome


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


class ResolutionQueueContractTests(unittest.TestCase):
    def test_resolution_sources_actions_and_surfaces_are_explicit(self) -> None:
        self.assertEqual(
            {source.value for source in ResolutionSource},
            {
                "needs_confirmation",
                "needs_followup",
                "open_question",
                "stale_on_use",
                "conflicted_locus",
            },
        )
        self.assertEqual(
            {action.value for action in ResolutionAction},
            {
                "confirm",
                "correct",
                "discard",
                "keep_ephemeral",
                "promote_to_durable_claim",
            },
        )
        self.assertEqual(
            {surface.value for surface in ResolutionSurface},
            {"prompt_queue", "host_api", "inspection"},
        )

    def test_policy_priority_remains_primary_before_utility_tiebreakers(self) -> None:
        now = sample_time()
        high_priority = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="preference/coffee",
            rationale="direct clarification needed",
            created_at=now,
            utility_boost=0,
        )
        low_priority_high_utility = ResolutionQueueItem(
            item_id="queue-2",
            source=ResolutionSource.STALE_ON_USE,
            priority=ResolutionPriority.NORMAL,
            subject_id="subject:user:alice",
            locus_key="preference/coffee",
            rationale="recent stale-on-use report",
            created_at=now,
            utility_boost=9,
        )
        same_priority_low_utility = ResolutionQueueItem(
            item_id="queue-3",
            source=ResolutionSource.NEEDS_FOLLOWUP,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="project/status",
            rationale="follow-up pending",
            created_at=sample_time(1),
            utility_boost=0,
        )
        same_priority_high_utility = ResolutionQueueItem(
            item_id="queue-4",
            source=ResolutionSource.NEEDS_FOLLOWUP,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="project/status",
            rationale="follow-up pending with high utility",
            created_at=sample_time(2),
            utility_boost=4,
        )

        self.assertLess(
            high_priority.priority_key(now),
            low_priority_high_utility.priority_key(now),
        )
        self.assertLess(
            same_priority_high_utility.priority_key(now),
            same_priority_low_utility.priority_key(now),
        )

    def test_deferred_items_wait_until_due(self) -> None:
        item = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.OPEN_QUESTION,
            priority=ResolutionPriority.NORMAL,
            subject_id="subject:user:alice",
            locus_key="question/travel",
            rationale="waiting for more evidence",
            created_at=sample_time(),
            status=ResolutionStatus.DEFERRED,
            deferred_until=sample_time(30),
        )

        self.assertFalse(item.is_actionable(sample_time(10)))
        self.assertTrue(item.is_actionable(sample_time(31)))

    def test_queue_items_surface_without_publishing_durable_claims(self) -> None:
        item = ResolutionQueueItem(
            item_id="queue-1",
            source=ResolutionSource.NEEDS_CONFIRMATION,
            priority=ResolutionPriority.HIGH,
            subject_id="subject:user:alice",
            locus_key="preference/coffee",
            rationale="confirm before durable promotion",
            created_at=sample_time(),
            surfaces=(ResolutionSurface.PROMPT_QUEUE, ResolutionSurface.HOST_API),
        )

        self.assertFalse(item.publishes_claim)
        self.assertTrue(item.surfaces_in_prompt)
        self.assertTrue(item.surfaces_via_host_api)

    def test_resolution_records_feed_admission_belief_outcomes_and_replay(self) -> None:
        promote = ResolutionRecord(
            item_id="queue-1",
            action=ResolutionAction.PROMOTE_TO_DURABLE_CLAIM,
            rationale="confirmed by user",
            recorded_at=sample_time(),
        )
        discard = ResolutionRecord(
            item_id="queue-2",
            action=ResolutionAction.DISCARD,
            rationale="candidate was noise",
            recorded_at=sample_time(),
        )

        self.assertEqual(promote.resulting_admission_outcome, AdmissionOutcome.DURABLE_CLAIM)
        self.assertEqual(
            promote.effects,
            frozenset(
                {
                    ResolutionEffect.ADMISSION,
                    ResolutionEffect.BELIEF_REVISION,
                    ResolutionEffect.OUTCOME_RECORDING,
                    ResolutionEffect.REPLAY_CAPTURE,
                }
            ),
        )
        self.assertEqual(discard.resulting_admission_outcome, AdmissionOutcome.DISCARD)
        self.assertEqual(
            discard.effects,
            frozenset(
                {
                    ResolutionEffect.ADMISSION,
                    ResolutionEffect.OUTCOME_RECORDING,
                    ResolutionEffect.REPLAY_CAPTURE,
                }
            ),
        )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_resolution_queue_contract(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("resolution queue", text)
        self.assertIn("needs_followup", text)
        self.assertIn("stale-on-use", text)
        self.assertIn("confirm", text)
        self.assertIn("keep ephemeral", text)
        self.assertIn("promote to durable claim", text)


if __name__ == "__main__":
    unittest.main()
