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

from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.ontology import MemoryPartition
from continuity.store.claims import AdmissionDecision, AdmissionOutcome


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


class AdmissionContractTests(unittest.TestCase):
    def test_admission_strengths_are_ordered_and_explicit(self) -> None:
        self.assertEqual(
            {strength.value for strength in AdmissionStrength},
            {"low", "medium", "high"},
        )
        self.assertLess(AdmissionStrength.LOW.score, AdmissionStrength.MEDIUM.score)
        self.assertLess(AdmissionStrength.MEDIUM.score, AdmissionStrength.HIGH.score)

    def test_threshold_shortfalls_are_explicit_for_admission_explanations(self) -> None:
        thresholds = AdmissionThresholds(
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.HIGH,
            salience=AdmissionStrength.LOW,
        )
        assessment = AdmissionAssessment(
            claim_type="preference",
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.LOW,
            stability=AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.MEDIUM,
            rationale="one-off mention with weak novelty",
            utility_signals=("prompt_inclusion",),
        )

        self.assertFalse(assessment.satisfies(thresholds))
        self.assertEqual(
            assessment.shortfall_fields(thresholds),
            ("novelty", "stability"),
        )

    def test_durable_admission_requires_thresholds_and_remaining_budget(self) -> None:
        trace = AdmissionDecisionTrace(
            decision=AdmissionDecision(
                candidate_id="candidate-1",
                outcome=AdmissionOutcome.DURABLE_CLAIM,
                recorded_at=sample_time(),
                rationale="explicit stable preference",
            ),
            claim_type="preference",
            policy_stamp="hermes_v1@1.0.0",
            assessment=AdmissionAssessment(
                claim_type="preference",
                evidence=AdmissionStrength.HIGH,
                novelty=AdmissionStrength.MEDIUM,
                stability=AdmissionStrength.HIGH,
                salience=AdmissionStrength.MEDIUM,
                rationale="repeated direct statement",
                utility_signals=("answer_citation",),
            ),
            thresholds=AdmissionThresholds(
                evidence=AdmissionStrength.MEDIUM,
                novelty=AdmissionStrength.MEDIUM,
                stability=AdmissionStrength.MEDIUM,
                salience=AdmissionStrength.LOW,
            ),
            budget=AdmissionWriteBudget(
                partition=MemoryPartition.USER_MEMORY,
                window_key="session:42",
                limit=8,
                used=3,
            ),
        )

        self.assertTrue(trace.publishes_claim)
        self.assertEqual(trace.shortfall_fields, ())
        self.assertEqual(trace.budget.remaining, 5)

        with self.assertRaises(ValueError):
            AdmissionDecisionTrace(
                decision=AdmissionDecision(
                    candidate_id="candidate-1",
                    outcome=AdmissionOutcome.DURABLE_CLAIM,
                    recorded_at=sample_time(),
                    rationale="budget exhausted",
                ),
                claim_type="preference",
                policy_stamp="hermes_v1@1.0.0",
                assessment=AdmissionAssessment(
                    claim_type="preference",
                    evidence=AdmissionStrength.HIGH,
                    novelty=AdmissionStrength.HIGH,
                    stability=AdmissionStrength.HIGH,
                    salience=AdmissionStrength.HIGH,
                    rationale="strong evidence",
                ),
                thresholds=AdmissionThresholds(
                    evidence=AdmissionStrength.MEDIUM,
                    novelty=AdmissionStrength.MEDIUM,
                    stability=AdmissionStrength.MEDIUM,
                    salience=AdmissionStrength.LOW,
                ),
                budget=AdmissionWriteBudget(
                    partition=MemoryPartition.USER_MEMORY,
                    window_key="session:42",
                    limit=8,
                    used=8,
                ),
            )

    def test_non_durable_outcomes_remain_usable_without_publishing_claims(self) -> None:
        outcomes = (
            (AdmissionOutcome.DISCARD, False, False),
            (AdmissionOutcome.SESSION_EPHEMERAL, True, False),
            (AdmissionOutcome.PROMPT_ONLY, True, False),
            (AdmissionOutcome.NEEDS_CONFIRMATION, True, True),
        )

        for outcome, retains_context, requires_queue in outcomes:
            with self.subTest(outcome=outcome):
                trace = AdmissionDecisionTrace(
                    decision=AdmissionDecision(
                        candidate_id="candidate-1",
                        outcome=outcome,
                        recorded_at=sample_time(),
                        rationale="kept out of durable memory",
                    ),
                    claim_type="open_question",
                    policy_stamp="hermes_v1@1.0.0",
                    assessment=AdmissionAssessment(
                        claim_type="open_question",
                        evidence=AdmissionStrength.LOW,
                        novelty=AdmissionStrength.MEDIUM,
                        stability=AdmissionStrength.LOW,
                        salience=AdmissionStrength.MEDIUM,
                        rationale="needs direct follow-up",
                    ),
                    thresholds=AdmissionThresholds(
                        evidence=AdmissionStrength.MEDIUM,
                        novelty=AdmissionStrength.MEDIUM,
                        stability=AdmissionStrength.MEDIUM,
                        salience=AdmissionStrength.LOW,
                    ),
                    budget=AdmissionWriteBudget(
                        partition=MemoryPartition.USER_MEMORY,
                        window_key="session:42",
                        limit=8,
                        used=8,
                    ),
                )

                self.assertFalse(trace.publishes_claim)
                self.assertEqual(trace.retains_candidate_context, retains_context)
                self.assertEqual(trace.requires_resolution_queue, requires_queue)


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_admission_thresholds_and_follow_up_paths(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("admission", text)
        self.assertIn("write budgets", text)
        self.assertIn("session_ephemeral", text)
        self.assertIn("prompt_only", text)
        self.assertIn("needs_confirmation", text)
        self.assertIn("resolution queue", text)


if __name__ == "__main__":
    unittest.main()
