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

from continuity.disclosure import (
    DisclosureAction,
    DisclosureChannel,
    DisclosureContext,
    DisclosurePolicy,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
    compose_disclosure_policies,
    disclosure_policy_for,
    evaluate_disclosure,
)


class DisclosureContractTests(unittest.TestCase):
    def test_v1_principals_channels_and_actions_are_explicit(self) -> None:
        self.assertEqual(
            {principal.value for principal in DisclosurePrincipal},
            {
                "assistant_internal",
                "current_user",
                "current_peer",
                "shared_session",
                "host_internal",
            },
        )
        self.assertEqual(
            {viewer.value for viewer in ViewerKind},
            {"assistant", "user", "peer", "host"},
        )
        self.assertEqual(
            {channel.value for channel in DisclosureChannel},
            {
                "prompt",
                "answer",
                "search",
                "profile",
                "evidence",
                "replay",
                "migration",
                "inspection",
            },
        )
        self.assertEqual(
            {action.value for action in DisclosureAction},
            {"allow", "summarize", "redact", "withhold", "needs_consent"},
        )

    def test_current_peer_policy_blocks_cross_peer_reads(self) -> None:
        policy = disclosure_policy_for("current_peer")

        allowed = evaluate_disclosure(
            policy,
            DisclosureContext(
                viewer=DisclosureViewer(
                    viewer_kind=ViewerKind.ASSISTANT,
                    active_peer_id="subject:peer:alice",
                ),
                audience_principal=DisclosurePrincipal.CURRENT_PEER,
                channel=DisclosureChannel.ANSWER,
                purpose=DisclosurePurpose.ANSWER,
                policy_stamp="hermes_v1@1.0.0",
            ),
        )
        blocked = evaluate_disclosure(
            policy,
            DisclosureContext(
                viewer=DisclosureViewer(
                    viewer_kind=ViewerKind.PEER,
                    viewer_subject_id="subject:peer:bob",
                    active_peer_id="subject:peer:alice",
                ),
                audience_principal=DisclosurePrincipal.CURRENT_PEER,
                channel=DisclosureChannel.ANSWER,
                purpose=DisclosurePurpose.ANSWER,
                policy_stamp="hermes_v1@1.0.0",
            ),
        )

        self.assertEqual(allowed.action, DisclosureAction.ALLOW)
        self.assertEqual(blocked.action, DisclosureAction.WITHHOLD)
        self.assertEqual(blocked.reason, "cross_peer_block")
        self.assertEqual(blocked.utility_signal, "withheld")

    def test_claim_locus_and_view_policies_compose_toward_the_most_restrictive_read(self) -> None:
        claim_policy = disclosure_policy_for("shared_session")
        locus_policy = DisclosurePolicy(
            policy_name="shared_session_profile_only",
            principal=DisclosurePrincipal.SHARED_SESSION,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT, ViewerKind.USER, ViewerKind.PEER}),
            allowed_channels=(DisclosureChannel.PROFILE, DisclosureChannel.ANSWER),
            allowed_purposes=(DisclosurePurpose.PROFILE, DisclosurePurpose.ANSWER),
            default_action=DisclosureAction.ALLOW,
        )
        view_override = DisclosurePolicy(
            policy_name="profile_projection_redaction",
            principal=DisclosurePrincipal.SHARED_SESSION,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT, ViewerKind.USER, ViewerKind.PEER}),
            allowed_channels=(DisclosureChannel.PROFILE,),
            allowed_purposes=(DisclosurePurpose.PROFILE,),
            default_action=DisclosureAction.REDACT,
            default_reason="redacted_for_profile_projection",
        )

        effective = compose_disclosure_policies(
            claim_policy,
            locus_policy,
            view_override,
            policy_name="profile_view_effective",
        )

        self.assertEqual(effective.principal, DisclosurePrincipal.SHARED_SESSION)
        self.assertEqual(effective.allowed_channels, (DisclosureChannel.PROFILE,))
        self.assertEqual(effective.allowed_purposes, (DisclosurePurpose.PROFILE,))
        self.assertEqual(effective.default_action, DisclosureAction.REDACT)
        self.assertEqual(effective.default_reason, "redacted_for_profile_projection")

    def test_consent_gated_reads_are_explicit_and_capture_withheld_utility(self) -> None:
        policy = DisclosurePolicy(
            policy_name="shared_session_evidence_consent",
            principal=DisclosurePrincipal.SHARED_SESSION,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT, ViewerKind.USER, ViewerKind.PEER}),
            allowed_channels=(DisclosureChannel.EVIDENCE,),
            allowed_purposes=(DisclosurePurpose.EVIDENCE,),
            default_action=DisclosureAction.ALLOW,
            consent_required_purposes=(DisclosurePurpose.EVIDENCE,),
        )

        decision = evaluate_disclosure(
            policy,
            DisclosureContext(
                viewer=DisclosureViewer(
                    viewer_kind=ViewerKind.ASSISTANT,
                    active_peer_id="subject:peer:alice",
                ),
                audience_principal=DisclosurePrincipal.SHARED_SESSION,
                channel=DisclosureChannel.EVIDENCE,
                purpose=DisclosurePurpose.EVIDENCE,
                policy_stamp="hermes_v1@1.0.0",
            ),
        )

        self.assertEqual(decision.action, DisclosureAction.NEEDS_CONSENT)
        self.assertEqual(decision.reason, "withheld_requires_consent")
        self.assertEqual(decision.utility_signal, "withheld")
        self.assertTrue(decision.captured_for_replay)


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_disclosure_principals_and_transforms(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("disclosure / audience layer", text)
        self.assertIn("assistant_internal", text)
        self.assertIn("current_peer", text)
        self.assertIn("shared_session", text)
        self.assertIn("needs_consent", text)
        self.assertIn("scope differs from audience", text)
        self.assertIn("claim-level defaults", text)
        self.assertIn("compiled-view overrides", text)


if __name__ == "__main__":
    unittest.main()
