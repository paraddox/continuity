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

from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.transactions import DurabilityWaterline, TransactionKind

try:
    from continuity.replay import (
        ReplayArtifact,
        ReplayComparison,
        ReplayInputBundle,
        ReplayMetric,
        ReplayMutationMode,
        ReplayRun,
        ReplayStep,
        ReplayStrategy,
    )
except ModuleNotFoundError:
    ReplayArtifact = None
    ReplayComparison = None
    ReplayInputBundle = None
    ReplayMetric = None
    ReplayMutationMode = None
    ReplayRun = None
    ReplayStep = None
    ReplayStrategy = None


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


def sample_context() -> DisclosureContext:
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind.ASSISTANT,
            viewer_subject_id="subject:assistant:hermes",
            active_user_id="subject:user:self",
            active_peer_id="subject:peer:openclaw",
        ),
        audience_principal=DisclosurePrincipal.ASSISTANT_INTERNAL,
        channel=DisclosureChannel.ANSWER,
        purpose=DisclosurePurpose.ANSWER,
        policy_stamp="hermes_v1@1.0.0",
    )


def sample_input_bundle(*, snapshot_id: str = "snapshot:active") -> ReplayInputBundle:
    assert ReplayInputBundle is not None

    return ReplayInputBundle(
        bundle_id="bundle-1",
        surface="answer_view",
        snapshot_id=snapshot_id,
        journal_position=21,
        arbiter_lane_position=17,
        disclosure_context=sample_context(),
        claim_ids=("claim-1", "claim-1", "claim-2"),
        observation_ids=("obs-1", "obs-1"),
        compiled_view_ids=("view:answer:1",),
        outcome_ids=("outcome-1",),
        reference_ids=("journal:21", "snapshot:active"),
        query_text="What changed for OpenClaw?",
    )


def sample_run(
    *,
    run_id: str,
    snapshot_id: str = "snapshot:active",
    policy_fingerprint: tuple[str, str] = ("hermes_v1@1.0.0", "hermes_core_v1"),
    reasoning_fingerprint: str = "reasoning:codex_sdk_gpt_5_4_low@1",
) -> ReplayRun:
    assert ReplayRun is not None
    assert ReplayStep is not None
    assert ReplayStrategy is not None
    assert ReplayMetric is not None

    return ReplayRun(
        run_id=run_id,
        input_bundle=sample_input_bundle(snapshot_id=snapshot_id),
        policy_fingerprint=policy_fingerprint,
        strategies=(
            ReplayStrategy(
                step=ReplayStep.RETRIEVAL,
                strategy_id="retrieval:hot_warm_ranker",
                fingerprint="retrieval:hot_warm_ranker@1",
            ),
            ReplayStrategy(
                step=ReplayStep.BELIEF,
                strategy_id="belief:locus_projection",
                fingerprint="belief:locus_projection@1",
            ),
            ReplayStrategy(
                step=ReplayStep.REASONING,
                strategy_id="reasoning:codex_sdk_gpt_5_4_low",
                fingerprint=reasoning_fingerprint,
            ),
        ),
        output_refs=("answer:1", "evidence:1"),
        metric_scores={
            ReplayMetric.CORRECTNESS: 5,
            ReplayMetric.FRESHNESS: 4,
            ReplayMetric.DISCLOSURE_SAFETY: 5,
            ReplayMetric.QUEUE_YIELD: 1,
            ReplayMetric.UTILITY_ALIGNMENT: 3,
        },
    )


class ReplayContractTests(unittest.TestCase):
    def test_replay_steps_metrics_and_mutation_mode_are_closed(self) -> None:
        self.assertIsNotNone(ReplayStep)
        self.assertIsNotNone(ReplayMetric)
        self.assertIsNotNone(ReplayMutationMode)

        self.assertEqual(
            {step.value for step in ReplayStep},
            {"retrieval", "belief", "reasoning"},
        )
        self.assertEqual(
            {metric.value for metric in ReplayMetric},
            {
                "correctness",
                "freshness",
                "disclosure_safety",
                "queue_yield",
                "utility_alignment",
            },
        )
        self.assertEqual(
            {mode.value for mode in ReplayMutationMode},
            {"read_only"},
        )

    def test_replay_input_bundle_pins_snapshot_request_context_and_stable_ids(self) -> None:
        self.assertIsNotNone(ReplayInputBundle)

        bundle = sample_input_bundle()

        self.assertEqual(bundle.claim_ids, ("claim-1", "claim-2"))
        self.assertEqual(bundle.observation_ids, ("obs-1",))
        self.assertEqual(bundle.compiled_view_ids, ("view:answer:1",))
        self.assertEqual(bundle.outcome_ids, ("outcome-1",))
        self.assertEqual(bundle.deterministic_key[:4], ("snapshot:active", 21, 17, "answer_view"))
        self.assertEqual(bundle.deterministic_key[4:7], ("assistant_internal", "answer", "answer"))

        with self.assertRaises(ValueError):
            ReplayInputBundle(
                bundle_id="bundle-2",
                surface="answer_view",
                snapshot_id="snapshot:active",
                journal_position=21,
                arbiter_lane_position=17,
                disclosure_context=sample_context(),
            )

    def test_replay_runs_require_explicit_policy_fingerprint_and_stage_strategies(self) -> None:
        self.assertIsNotNone(ReplayRun)
        self.assertIsNotNone(ReplayStep)
        self.assertIsNotNone(ReplayStrategy)

        run = sample_run(run_id="run-1")

        self.assertEqual(run.policy_fingerprint, ("hermes_v1@1.0.0", "hermes_core_v1"))
        self.assertEqual(
            run.strategy_for(ReplayStep.RETRIEVAL).strategy_id,
            "retrieval:hot_warm_ranker",
        )
        self.assertEqual(run.mutation_mode, ReplayMutationMode.READ_ONLY)

        with self.assertRaises(ValueError):
            ReplayRun(
                run_id="run-2",
                input_bundle=sample_input_bundle(),
                policy_fingerprint=("hermes_v1@1.0.0", "hermes_core_v1"),
                strategies=(
                    ReplayStrategy(
                        step=ReplayStep.RETRIEVAL,
                        strategy_id="retrieval:hot_warm_ranker",
                        fingerprint="retrieval:hot_warm_ranker@1",
                    ),
                    ReplayStrategy(
                        step=ReplayStep.BELIEF,
                        strategy_id="belief:locus_projection",
                        fingerprint="belief:locus_projection@1",
                    ),
                ),
                output_refs=("answer:1",),
                metric_scores={ReplayMetric.CORRECTNESS: 5},
            )

    def test_replay_artifacts_are_versioned_and_bound_to_transaction_waterlines(self) -> None:
        self.assertIsNotNone(ReplayArtifact)

        artifact = ReplayArtifact(
            artifact_id="replay:turn:1",
            version="replay_v1",
            source_transaction=TransactionKind.INGEST_TURN,
            source_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
            captured_at=sample_time(),
            baseline_run=sample_run(run_id="run-1"),
            source_object_ids=("snapshot:active", "answer:1"),
        )

        self.assertEqual(artifact.version, "replay_v1")
        self.assertEqual(artifact.policy_fingerprint, ("hermes_v1@1.0.0", "hermes_core_v1"))
        self.assertEqual(artifact.source_object_ids, ("snapshot:active", "answer:1"))

        with self.assertRaises(ValueError):
            ReplayArtifact(
                artifact_id="replay:turn:2",
                version="replay_v1",
                source_transaction=TransactionKind.PREFETCH_NEXT_TURN,
                source_waterline=DurabilityWaterline.CLAIM_COMMITTED,
                captured_at=sample_time(),
                baseline_run=sample_run(run_id="run-2"),
                source_object_ids=("snapshot:active",),
            )

    def test_counterfactual_comparisons_reuse_the_same_inputs_and_stay_read_only(self) -> None:
        self.assertIsNotNone(ReplayComparison)
        self.assertIsNotNone(ReplayStep)

        baseline = sample_run(run_id="run-1")
        candidate = sample_run(
            run_id="run-2",
            policy_fingerprint=("hermes_v1@1.1.0", "hermes_core_v1"),
            reasoning_fingerprint="reasoning:codex_sdk_gpt_5_4_low@2",
        )
        comparison = ReplayComparison(
            comparison_id="cmp-1",
            baseline_run=baseline,
            candidate_run=candidate,
            compared_steps=(
                ReplayStep.RETRIEVAL,
                ReplayStep.BELIEF,
                ReplayStep.REASONING,
            ),
            rationale="compare upgraded policy and reasoning strategy",
        )

        self.assertTrue(comparison.policy_changed)
        self.assertFalse(comparison.mutates_authoritative_state)
        self.assertEqual(comparison.changed_steps, frozenset({ReplayStep.REASONING}))

        with self.assertRaises(ValueError):
            ReplayComparison(
                comparison_id="cmp-2",
                baseline_run=baseline,
                candidate_run=sample_run(run_id="run-3", snapshot_id="snapshot:candidate:2"),
                compared_steps=(ReplayStep.REASONING,),
                rationale="snapshot mismatch should not compare",
            )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_replay_artifact_contract(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("replay artifact", text)
        self.assertIn("counterfactual replay", text)
        self.assertIn("deterministic replay inputs", text)
        self.assertIn("policy fingerprint", text)
        self.assertIn("read-only", text)
        self.assertIn("retrieval, belief, and reasoning", text)


if __name__ == "__main__":
    unittest.main()
