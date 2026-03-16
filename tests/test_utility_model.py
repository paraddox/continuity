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

from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeTarget
from continuity.policy import hermes_v1_policy_pack
from continuity.utility import (
    CompiledUtilityWeight,
    UtilitySignal,
    compile_utility_weight,
    utility_events_for_outcome,
)


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


def outcome_record(*, outcome_id: str, label: OutcomeLabel) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=outcome_id,
        label=label,
        target=OutcomeTarget.CLAIM,
        target_id="claim-1",
        policy_stamp="hermes_v1@1.0.0",
        recorded_at=sample_time(),
        rationale=f"{label.value} recorded for utility compilation",
        claim_ids=("claim-1",),
    )


class UtilityContractTests(unittest.TestCase):
    def test_v1_utility_signals_are_closed_and_supported_by_policy(self) -> None:
        self.assertEqual(
            {signal.value for signal in UtilitySignal},
            {
                "prompt_inclusion",
                "answer_citation",
                "user_corrected",
                "stale_on_use",
            },
        )

        policy = hermes_v1_policy_pack()
        for signal in UtilitySignal:
            with self.subTest(signal=signal):
                self.assertIn(signal.value, policy.utility_weights)

    def test_outcomes_compile_into_deterministic_weighted_scores(self) -> None:
        outcomes = (
            outcome_record(outcome_id="outcome-3", label=OutcomeLabel.ANSWER_CITED),
            outcome_record(outcome_id="outcome-1", label=OutcomeLabel.PROMPT_INCLUDED),
            outcome_record(outcome_id="outcome-4", label=OutcomeLabel.USER_CORRECTED),
            outcome_record(outcome_id="outcome-2", label=OutcomeLabel.PROMPT_INCLUDED),
        )
        events = tuple(
            event
            for outcome in outcomes
            for event in utility_events_for_outcome(outcome)
        )

        compiled = compile_utility_weight(
            target=OutcomeTarget.CLAIM,
            target_id="claim-1",
            policy=hermes_v1_policy_pack(),
            events=events,
        )

        self.assertIsInstance(compiled, CompiledUtilityWeight)
        self.assertEqual(compiled.weighted_score, 5)
        self.assertEqual(compiled.signal_count_for(UtilitySignal.PROMPT_INCLUSION), 2)
        self.assertEqual(compiled.signal_count_for(UtilitySignal.ANSWER_CITATION), 1)
        self.assertEqual(compiled.signal_count_for(UtilitySignal.USER_CORRECTED), 1)
        self.assertEqual(compiled.source_event_ids, ("outcome-1", "outcome-2", "outcome-3", "outcome-4"))

    def test_confirmation_remains_an_outcome_without_creating_utility_weight(self) -> None:
        events = utility_events_for_outcome(
            outcome_record(outcome_id="outcome-1", label=OutcomeLabel.USER_CONFIRMED)
        )

        self.assertEqual(events, ())


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_compiled_utility_weight_inputs(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("utility ledger", text)
        self.assertIn("prompt_inclusion", text)
        self.assertIn("answer_citation", text)
        self.assertIn("user_corrected", text)
        self.assertIn("stale_on_use", text)


if __name__ == "__main__":
    unittest.main()
