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

from continuity.epistemics import EpistemicStatus
from continuity.views import (
    CompiledView,
    ProvenanceSurface,
    SnapshotBinding,
    TierDefault,
    ViewKind,
    view_contract_for,
    view_contracts,
)


class ViewContractTests(unittest.TestCase):
    def test_view_kinds_are_named_and_closed(self) -> None:
        self.assertEqual(
            {kind.value for kind in ViewKind},
            {"state", "timeline", "set", "profile", "prompt", "evidence", "answer"},
        )

    def test_contract_registry_covers_each_view_kind(self) -> None:
        contracts = view_contracts()

        self.assertEqual(set(contracts), set(ViewKind))

        prompt_contract = view_contract_for(ViewKind.PROMPT)
        self.assertEqual(
            set(prompt_contract.dependency_view_kinds),
            {
                ViewKind.STATE,
                ViewKind.TIMELINE,
                ViewKind.SET,
                ViewKind.PROFILE,
                ViewKind.EVIDENCE,
            },
        )
        self.assertEqual(prompt_contract.snapshot_binding, SnapshotBinding.SESSION_SNAPSHOT)
        self.assertEqual(prompt_contract.provenance_surface, ProvenanceSurface.CLAIM_IDS)
        self.assertEqual(
            prompt_contract.tier_defaults,
            (TierDefault.HOT, TierDefault.WARM),
        )

        answer_contract = view_contract_for(ViewKind.ANSWER)
        self.assertEqual(
            set(answer_contract.dependency_view_kinds),
            {
                ViewKind.STATE,
                ViewKind.TIMELINE,
                ViewKind.SET,
                ViewKind.PROFILE,
                ViewKind.EVIDENCE,
            },
        )
        self.assertEqual(answer_contract.snapshot_binding, SnapshotBinding.QUERY_SNAPSHOT)

        evidence_contract = view_contract_for(ViewKind.EVIDENCE)
        self.assertEqual(
            evidence_contract.provenance_surface,
            ProvenanceSurface.CLAIMS_AND_OBSERVATIONS,
        )

    def test_compiled_views_require_claim_provenance(self) -> None:
        view = CompiledView(
            kind=ViewKind.STATE,
            view_key="state:subject:user:alice:preference/coffee",
            policy_stamp="hermes_v1@1.0.0",
            snapshot_id="snapshot-1",
            claim_ids=("claim-1", "claim-1", "claim-2"),
            observation_ids=("obs-1", "obs-1"),
            epistemic_status=EpistemicStatus.SUPPORTED,
        )

        self.assertEqual(view.claim_ids, ("claim-1", "claim-2"))
        self.assertEqual(view.observation_ids, ("obs-1",))

        with self.assertRaises(ValueError):
            CompiledView(
                kind=ViewKind.PROFILE,
                view_key="profile:subject:user:alice",
                policy_stamp="hermes_v1@1.0.0",
                snapshot_id="snapshot-1",
                claim_ids=(),
                epistemic_status=EpistemicStatus.SUPPORTED,
            )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_compiled_view_algebra(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("compiled view algebra", text)
        self.assertIn("state_view", text)
        self.assertIn("timeline_view", text)
        self.assertIn("set_view", text)
        self.assertIn("profile_view", text)
        self.assertIn("prompt_view", text)
        self.assertIn("evidence_view", text)
        self.assertIn("answer_view", text)


if __name__ == "__main__":
    unittest.main()
