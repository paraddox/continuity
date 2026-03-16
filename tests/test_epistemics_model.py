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

from continuity.epistemics import (
    AnswerMode,
    EpistemicAssessment,
    EpistemicStatus,
    EpistemicTarget,
    PromptExposure,
)


class EpistemicStatusTests(unittest.TestCase):
    def test_v1_epistemic_statuses_are_closed_and_explicit(self) -> None:
        self.assertEqual(
            {status.value for status in EpistemicStatus},
            {
                "supported",
                "unknown",
                "tentative",
                "conflicted",
                "stale",
                "needs_confirmation",
            },
        )

    def test_statuses_attach_to_claims_locus_resolutions_views_and_answers(self) -> None:
        self.assertEqual(
            {target.value for target in EpistemicTarget},
            {"claim", "locus_resolution", "compiled_view", "answer"},
        )

    def test_supported_state_can_assert_normally(self) -> None:
        assessment = EpistemicAssessment(
            status=EpistemicStatus.SUPPORTED,
            target=EpistemicTarget.ANSWER,
            rationale="fresh direct evidence",
        )

        self.assertEqual(assessment.answer_mode, AnswerMode.ASSERT)
        self.assertEqual(assessment.prompt_exposure, PromptExposure.INCLUDE)

    def test_unknown_and_conflicted_states_abstain(self) -> None:
        for status in (EpistemicStatus.UNKNOWN, EpistemicStatus.CONFLICTED):
            with self.subTest(status=status):
                assessment = EpistemicAssessment(
                    status=status,
                    target=EpistemicTarget.ANSWER,
                    rationale="insufficient evidence",
                )

                self.assertEqual(assessment.answer_mode, AnswerMode.ABSTAIN)
                self.assertEqual(assessment.prompt_exposure, PromptExposure.SUPPRESS)

    def test_tentative_and_stale_states_require_qualification(self) -> None:
        for status in (EpistemicStatus.TENTATIVE, EpistemicStatus.STALE):
            with self.subTest(status=status):
                assessment = EpistemicAssessment(
                    status=status,
                    target=EpistemicTarget.COMPILED_VIEW,
                    rationale="soft support only",
                )

                self.assertEqual(assessment.answer_mode, AnswerMode.QUALIFY)
                self.assertEqual(assessment.prompt_exposure, PromptExposure.QUALIFY)

    def test_needs_confirmation_requires_clarification(self) -> None:
        assessment = EpistemicAssessment(
            status=EpistemicStatus.NEEDS_CONFIRMATION,
            target=EpistemicTarget.ANSWER,
            rationale="policy requires user confirmation",
        )

        self.assertEqual(assessment.answer_mode, AnswerMode.ASK_CONFIRMATION)
        self.assertEqual(assessment.prompt_exposure, PromptExposure.SUPPRESS)


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_epistemic_and_abstention_rules(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("unknown", text)
        self.assertIn("tentative", text)
        self.assertIn("conflicted", text)
        self.assertIn("stale", text)
        self.assertIn("needs_confirmation", text)
        self.assertIn("abstain", text)
        self.assertIn("qualify", text)


if __name__ == "__main__":
    unittest.main()
