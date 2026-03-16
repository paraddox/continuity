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

from continuity.epistemics import EpistemicStatus
from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeTarget


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


class OutcomeContractTests(unittest.TestCase):
    def test_v1_outcome_labels_are_closed_and_explicit(self) -> None:
        self.assertEqual(
            {label.value for label in OutcomeLabel},
            {
                "prompt_included",
                "answer_cited",
                "user_confirmed",
                "user_corrected",
                "stale_on_use",
            },
        )

    def test_records_stay_attributable_and_distinct_from_epistemic_state(self) -> None:
        record = OutcomeRecord(
            outcome_id="outcome-1",
            label=OutcomeLabel.USER_CORRECTED,
            target=OutcomeTarget.CLAIM,
            target_id="claim-1",
            policy_stamp="hermes_v1@1.0.0",
            recorded_at=sample_time(),
            rationale="user corrected the remembered preference",
            actor_subject_id="subject:user:self",
            claim_ids=("claim-1", "claim-1"),
            observation_ids=("obs-1", "obs-1"),
        )

        self.assertEqual(record.claim_ids, ("claim-1",))
        self.assertEqual(record.observation_ids, ("obs-1",))
        self.assertTrue(record.capture_for_replay)
        self.assertNotIn(record.label.value, {status.value for status in EpistemicStatus})

        with self.assertRaises(ValueError):
            OutcomeRecord(
                outcome_id="outcome-2",
                label=OutcomeLabel.PROMPT_INCLUDED,
                target=OutcomeTarget.COMPILED_VIEW,
                target_id="view:prompt:1",
                policy_stamp="hermes_v1@1.0.0",
                recorded_at=sample_time(),
                rationale="prompt used the fragment",
            )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_outcome_ledger_and_separation_rules(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("outcome ledger", text)
        self.assertIn("prompt_included", text)
        self.assertIn("user_corrected", text)
        self.assertIn("stale_on_use", text)
        self.assertIn("distinct from epistemic status", text)
        self.assertIn("compiled utility weights", text)


if __name__ == "__main__":
    unittest.main()
