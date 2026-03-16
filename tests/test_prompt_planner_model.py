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

from continuity.epistemics import EpistemicStatus, PromptExposure
from continuity.prompt_planner import (
    PromptDisclosureAction,
    PromptFragmentCandidate,
    PromptPlan,
    PromptPlannerConfig,
    plan_prompt_view,
)
from continuity.views import ViewKind


class PromptPlannerContractTests(unittest.TestCase):
    def test_prompt_fragment_requires_claim_provenance_and_explicit_transforms(self) -> None:
        with self.assertRaises(ValueError):
            PromptFragmentCandidate(
                fragment_id="state-1",
                source_view=ViewKind.STATE,
                text="Alice prefers espresso.",
                token_estimate=6,
                priority_band=10,
                claim_ids=(),
            )

        with self.assertRaises(ValueError):
            PromptFragmentCandidate(
                fragment_id="profile-1",
                source_view=ViewKind.PROFILE,
                text="Alice profile card",
                token_estimate=8,
                priority_band=20,
                claim_ids=("claim-1",),
                disclosure_action=PromptDisclosureAction.REDACT,
            )

    def test_planner_packs_deterministically_under_budget_pressure(self) -> None:
        plan = plan_prompt_view(
            policy_stamp="hermes_v1@1.0.0",
            config=PromptPlannerConfig(
                hard_token_budget=15,
                soft_token_budgets={"evidence": 4},
            ),
            candidates=(
                PromptFragmentCandidate(
                    fragment_id="state-1",
                    source_view=ViewKind.STATE,
                    text="Alice prefers espresso.",
                    token_estimate=6,
                    priority_band=10,
                    utility_weight=4,
                    claim_ids=("claim-1",),
                    observation_ids=("obs-1",),
                ),
                PromptFragmentCandidate(
                    fragment_id="profile-1",
                    source_view=ViewKind.PROFILE,
                    text="Alice is a maintainer.",
                    token_estimate=5,
                    priority_band=20,
                    utility_weight=2,
                    claim_ids=("claim-2",),
                ),
                PromptFragmentCandidate(
                    fragment_id="evidence-1",
                    source_view=ViewKind.EVIDENCE,
                    text="Observation transcript excerpt",
                    token_estimate=5,
                    priority_band=30,
                    utility_weight=1,
                    claim_ids=("claim-1",),
                    observation_ids=("obs-1",),
                    soft_budget_group="evidence",
                    compressed_text="Evidence summary",
                    compressed_token_estimate=3,
                    degradation_reason="dropped low-priority evidence",
                ),
                PromptFragmentCandidate(
                    fragment_id="timeline-1",
                    source_view=ViewKind.TIMELINE,
                    text="Long timeline of edits",
                    token_estimate=8,
                    priority_band=40,
                    utility_weight=10,
                    claim_ids=("claim-3",),
                    compressed_text="Collapsed timeline",
                    compressed_token_estimate=4,
                    degradation_reason="collapsed timeline",
                ),
            ),
        )

        self.assertIsInstance(plan, PromptPlan)
        self.assertEqual(plan.fragment_ids_for_model, ("state-1", "profile-1", "evidence-1"))
        self.assertEqual(plan.token_estimate, 14)
        self.assertEqual(plan.degradation_reasons, ("dropped low-priority evidence",))
        self.assertEqual(
            {item.fragment_id: item.reason for item in plan.excluded_fragments},
            {"timeline-1": "hard_budget_exhausted"},
        )

    def test_planner_qualifies_stale_state_and_suppresses_unsafe_fragments(self) -> None:
        plan = plan_prompt_view(
            policy_stamp="hermes_v1@1.0.0",
            config=PromptPlannerConfig(hard_token_budget=20),
            candidates=(
                PromptFragmentCandidate(
                    fragment_id="stale-1",
                    source_view=ViewKind.STATE,
                    text="Old preference",
                    token_estimate=4,
                    priority_band=10,
                    claim_ids=("claim-1",),
                    epistemic_status=EpistemicStatus.STALE,
                ),
                PromptFragmentCandidate(
                    fragment_id="conflicted-1",
                    source_view=ViewKind.STATE,
                    text="Conflicted state",
                    token_estimate=4,
                    priority_band=20,
                    claim_ids=("claim-2",),
                    epistemic_status=EpistemicStatus.CONFLICTED,
                ),
                PromptFragmentCandidate(
                    fragment_id="withheld-1",
                    source_view=ViewKind.EVIDENCE,
                    text="Hidden evidence",
                    token_estimate=4,
                    priority_band=30,
                    claim_ids=("claim-3",),
                    disclosure_action=PromptDisclosureAction.WITHHOLD,
                    disclosure_reason="withheld_requires_consent",
                ),
            ),
        )

        included = {fragment.fragment_id: fragment for fragment in plan.included_fragments}
        self.assertEqual(included["stale-1"].epistemic_action, PromptExposure.QUALIFY)
        self.assertEqual(
            {item.fragment_id: item.reason for item in plan.excluded_fragments},
            {
                "conflicted-1": "suppressed_by_epistemic_status:conflicted",
                "withheld-1": "withheld_requires_consent",
            },
        )

    def test_redactions_and_actual_token_usage_are_exposed_when_available(self) -> None:
        plan = plan_prompt_view(
            policy_stamp="hermes_v1@1.0.0",
            config=PromptPlannerConfig(hard_token_budget=20),
            candidates=(
                PromptFragmentCandidate(
                    fragment_id="profile-1",
                    source_view=ViewKind.PROFILE,
                    text="Alice profile",
                    token_estimate=6,
                    actual_token_usage=5,
                    priority_band=10,
                    claim_ids=("claim-1",),
                    disclosure_action=PromptDisclosureAction.REDACT,
                    disclosure_reason="redacted_for_peer",
                ),
            ),
        )

        self.assertEqual(plan.actual_token_usage, 5)
        self.assertEqual(plan.included_fragments[0].disclosure_reason, "redacted_for_peer")


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_budgeted_prompt_planning(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("budgeted prompt planner", text)
        self.assertIn("hard token budget", text)
        self.assertIn("inclusion and exclusion reasons", text)
        self.assertIn("degradation ladder", text)
        self.assertIn("dropped low-priority evidence", text)
        self.assertIn("collapsed timeline", text)


if __name__ == "__main__":
    unittest.main()
