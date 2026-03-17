#!/usr/bin/env python3

from __future__ import annotations

import sys
import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from continuity.evals.replay_runner import (
    ReplayEvaluationExpectation,
    ReplayExecutionMode,
    ReplayFixtureCase,
    ReplayPlan,
    ReplayRunner,
    ReplayStageOverride,
)
from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeRepository, OutcomeTarget
from continuity.replay import ReplayMetric, ReplayStep
from continuity.service import ServiceOperation, ServiceRequest
from continuity.store.replay import ReplayRepository
from continuity.utility import CompiledUtilityWeight, UtilityRepository
from tests.test_dialectic import open_memory_database, sample_time
from tests.test_replay_store import (
    sample_artifact,
    sample_input_bundle,
    sample_run,
    seed_replay_prerequisites,
)
from tests.test_turn_artifacts import assistant_answer_context, build_harness


def seed_replay_fixture_state(connection: sqlite3.Connection) -> ReplayRepository:
    seed_replay_prerequisites(connection)
    replay_repository = ReplayRepository(connection)

    bundle = replace(
        sample_input_bundle(),
        outcome_ids=("outcome:stale-latte", "outcome:queue-confirmed"),
    )
    run = sample_run(run_id="run:fixture:baseline", bundle=bundle)
    artifact = sample_artifact(run=run)
    fixture_payload = {
        "surface": "answer_view",
        "retrieval": {
            "query_text": "What do you know about Alice's coffee preferences?",
            "limit": 2,
            "candidates": (
                {
                    "view_key": "timeline:subject:user:alice:preference/favorite_drink",
                    "view_kind": "timeline",
                    "score": 0.61,
                },
                {
                    "view_key": "evidence:observation:obs-latte",
                    "view_kind": "evidence",
                    "score": 0.58,
                },
            ),
        },
        "selection": {
            "selected_claim_ids": ("claim-latte",),
            "selected_beliefs": (
                {"claim_id": "claim-latte", "text": "Alice used to prefer latte."},
            ),
            "historical_beliefs": (),
            "source_compiled_views": (
                {
                    "view_key": "timeline:subject:user:alice:preference/favorite_drink",
                    "claim_ids": ("claim-latte",),
                },
            ),
        },
        "disclosure": {
            "result": "visible",
            "hidden_claim_ids": ("claim-private",),
        },
        "reasoning": {
            "adapter": "fixture_reasoner",
            "response_text": "Alice used to prefer latte.",
        },
        "resolution_queue": {
            "surfaced_items": (),
        },
        "utility_events": (
            {"kind": "answer_citation", "claim_id": "claim-latte"},
        ),
    }
    replay_repository.save_artifact(
        replace(
            artifact,
            artifact_id="replay:fixture:baseline",
            baseline_run=replace(
                run,
                run_id="run:fixture:baseline",
                output_refs=("answer:fixture:baseline",),
            ),
            decision_payload=fixture_payload,
        )
    )

    outcomes = OutcomeRepository(connection)
    outcomes.record_outcome(
        OutcomeRecord(
            outcome_id="outcome:stale-latte",
            label=OutcomeLabel.STALE_ON_USE,
            target=OutcomeTarget.CLAIM,
            target_id="claim-latte",
            policy_stamp="hermes_v1@1.0.0",
            recorded_at=sample_time(3),
            rationale="latte preference was corrected later",
            claim_ids=("claim-latte",),
        )
    )
    outcomes.record_outcome(
        OutcomeRecord(
            outcome_id="outcome:queue-confirmed",
            label=OutcomeLabel.USER_CONFIRMED,
            target=OutcomeTarget.RESOLUTION_QUEUE_ITEM,
            target_id="queue-check-oat",
            policy_stamp="hermes_v1@1.1.0",
            recorded_at=sample_time(4),
            rationale="the clarification queue item resolved useful state",
        )
    )

    utility = UtilityRepository(connection)
    utility.write_compiled_weight(
        CompiledUtilityWeight(
            target=OutcomeTarget.CLAIM,
            target_id="claim-espresso",
            policy_stamp="hermes_v1@1.1.0",
            weighted_score=8,
            signal_counts=(),
            source_event_ids=("outcome:utility-espresso",),
        )
    )

    return replay_repository


class ReplayRunnerTests(unittest.TestCase):
    def test_retrieval_only_replay_reuses_stored_turn_inputs_and_changes_one_step(self) -> None:
        harness = build_harness()
        self.addCleanup(harness.connection.close)
        harness.facade.execute(
            ServiceRequest(
                operation=ServiceOperation.ANSWER_MEMORY_QUESTION,
                request_id="request:answer-turn",
                payload={
                    "question": "What do you know about Alice's coffee preferences?",
                    "subject_id": "subject:user:alice",
                },
                disclosure_context=assistant_answer_context(),
                target_snapshot_id="snapshot-1",
            )
        )

        runner = ReplayRunner(replay_repository=harness.replay)
        evaluation = runner.evaluate_artifact(
            artifact_id="replay:request:answer-turn",
            plan=ReplayPlan(
                run_id="run:request:answer-turn:retrieval-alt",
                comparison_id="comparison:request:answer-turn:retrieval-alt",
                mode=ReplayExecutionMode.RETRIEVAL_ONLY,
                rationale="Prefer fresher state and timeline retrieval results first.",
                stage_overrides=(
                    ReplayStageOverride(
                        step=ReplayStep.RETRIEVAL,
                        strategy_id="retrieval:freshness_ranker",
                        fingerprint="retrieval:freshness_ranker@1",
                        payload={
                            "candidates": (
                                {
                                    "view_key": "state:subject:user:alice:preference/favorite_drink",
                                    "view_kind": "state",
                                    "score": 0.99,
                                },
                                {
                                    "view_key": "timeline:subject:user:alice:preference/favorite_drink",
                                    "view_kind": "timeline",
                                    "score": 0.94,
                                },
                            ),
                            "limit": 2,
                        },
                    ),
                ),
            ),
            expectation=ReplayEvaluationExpectation(
                expected_retrieval_view_keys=(
                    "state:subject:user:alice:preference/favorite_drink",
                    "timeline:subject:user:alice:preference/favorite_drink",
                ),
                expected_selected_claim_ids=("claim-espresso", "claim-latte"),
                expected_answer_substrings=("Alice currently prefers espresso",),
                expected_disclosure_result="visible",
                expected_utility_event_kinds=("answer_citation",),
            ),
        )

        self.assertEqual(
            evaluation.comparison_record.comparison.changed_steps,
            frozenset({ReplayStep.RETRIEVAL}),
        )
        self.assertEqual(
            evaluation.candidate_run.strategy_for(ReplayStep.RETRIEVAL).strategy_id,
            "retrieval:freshness_ranker",
        )
        self.assertEqual(
            evaluation.candidate_payload["retrieval"]["limit"],
            2,
        )
        self.assertEqual(
            tuple(metric for metric in evaluation.candidate_scores),
            tuple(ReplayMetric),
        )
        self.assertEqual(
            evaluation.candidate_payload["retrieval"]["candidates"][0]["view_key"],
            "state:subject:user:alice:preference/favorite_drink",
        )

    def test_belief_only_replay_improves_freshness_from_stale_outcomes_without_hydrating_claims(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        replay_repository = seed_replay_fixture_state(connection)

        runner = ReplayRunner(
            replay_repository=replay_repository,
            outcome_repository=OutcomeRepository(connection),
        )
        evaluation = runner.evaluate_artifact(
            artifact_id="replay:fixture:baseline",
            plan=ReplayPlan(
                run_id="run:fixture:belief-alt",
                comparison_id="comparison:fixture:belief-alt",
                mode=ReplayExecutionMode.BELIEF_ONLY,
                rationale="Promote the fresher espresso claim over the stale latte claim.",
                stage_overrides=(
                    ReplayStageOverride(
                        step=ReplayStep.BELIEF,
                        strategy_id="belief:freshness_first",
                        fingerprint="belief:freshness_first@1",
                        payload={
                            "selected_claim_ids": ("claim-espresso",),
                            "selected_beliefs": (
                                {
                                    "claim_id": "claim-espresso",
                                    "text": "Alice currently prefers espresso.",
                                },
                            ),
                            "historical_beliefs": (
                                {
                                    "claim_id": "claim-latte",
                                    "text": "Alice used to prefer latte.",
                                },
                            ),
                            "source_compiled_views": (
                                {
                                    "view_key": "state:subject:user:alice:preference/favorite_drink",
                                    "claim_ids": ("claim-espresso",),
                                },
                            ),
                        },
                    ),
                ),
            ),
            expectation=ReplayEvaluationExpectation(
                expected_selected_claim_ids=("claim-espresso",),
                fresh_claim_ids=("claim-espresso",),
            ),
        )

        self.assertEqual(
            evaluation.comparison_record.comparison.changed_steps,
            frozenset({ReplayStep.BELIEF}),
        )
        self.assertLess(
            evaluation.baseline_scores[ReplayMetric.FRESHNESS],
            evaluation.candidate_scores[ReplayMetric.FRESHNESS],
        )
        self.assertEqual(
            evaluation.candidate_payload["selection"]["selected_claim_ids"],
            ("claim-espresso",),
        )

    def test_end_to_end_fixture_harness_scores_queue_disclosure_and_utility_signals(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        replay_repository = seed_replay_fixture_state(connection)

        runner = ReplayRunner(
            replay_repository=replay_repository,
            outcome_repository=OutcomeRepository(connection),
            utility_repository=UtilityRepository(connection),
        )
        evaluations = runner.evaluate_cases(
            (
                ReplayFixtureCase(
                    fixture_id="belief-freshness",
                    artifact_id="replay:fixture:baseline",
                    plan=ReplayPlan(
                        run_id="run:fixture:belief-fixture",
                        comparison_id="comparison:fixture:belief-fixture",
                        mode=ReplayExecutionMode.BELIEF_ONLY,
                        rationale="Keep the fixture harness coverage for belief-only replays.",
                        stage_overrides=(
                            ReplayStageOverride(
                                step=ReplayStep.BELIEF,
                                strategy_id="belief:freshness_first",
                                fingerprint="belief:freshness_first@1",
                                payload={
                                    "selected_claim_ids": ("claim-espresso",),
                                    "selected_beliefs": (
                                        {
                                            "claim_id": "claim-espresso",
                                            "text": "Alice currently prefers espresso.",
                                        },
                                    ),
                                    "historical_beliefs": (),
                                    "source_compiled_views": (),
                                },
                            ),
                        ),
                    ),
                    expectation=ReplayEvaluationExpectation(
                        expected_selected_claim_ids=("claim-espresso",),
                        fresh_claim_ids=("claim-espresso",),
                    ),
                ),
                ReplayFixtureCase(
                    fixture_id="end-to-end-answer",
                    artifact_id="replay:fixture:baseline",
                    plan=ReplayPlan(
                        run_id="run:fixture:end-to-end",
                        comparison_id="comparison:fixture:end-to-end",
                        mode=ReplayExecutionMode.END_TO_END,
                        rationale="Upgrade retrieval, belief, and reasoning while preserving disclosure boundaries.",
                        policy_fingerprint=("hermes_v1@1.1.0", "hermes_core_v2"),
                        stage_overrides=(
                            ReplayStageOverride(
                                step=ReplayStep.RETRIEVAL,
                                strategy_id="retrieval:freshness_ranker",
                                fingerprint="retrieval:freshness_ranker@2",
                                payload={
                                    "query_text": "What do you know about Alice's coffee preferences?",
                                    "limit": 2,
                                    "candidates": (
                                        {
                                            "view_key": "state:subject:user:alice:preference/favorite_drink",
                                            "view_kind": "state",
                                            "score": 0.99,
                                        },
                                        {
                                            "view_key": "timeline:subject:user:alice:preference/favorite_drink",
                                            "view_kind": "timeline",
                                            "score": 0.96,
                                        },
                                    ),
                                },
                            ),
                            ReplayStageOverride(
                                step=ReplayStep.BELIEF,
                                strategy_id="belief:freshness_first",
                                fingerprint="belief:freshness_first@2",
                                payload={
                                    "selected_claim_ids": ("claim-espresso",),
                                    "selected_beliefs": (
                                        {
                                            "claim_id": "claim-espresso",
                                            "text": "Alice currently prefers espresso.",
                                        },
                                    ),
                                    "historical_beliefs": (
                                        {
                                            "claim_id": "claim-latte",
                                            "text": "Alice used to prefer latte.",
                                        },
                                    ),
                                    "source_compiled_views": (
                                        {
                                            "view_key": "state:subject:user:alice:preference/favorite_drink",
                                            "claim_ids": ("claim-espresso",),
                                        },
                                    ),
                                },
                            ),
                            ReplayStageOverride(
                                step=ReplayStep.REASONING,
                                strategy_id="reasoning:codex_sdk_gpt_5_4_low",
                                fingerprint="reasoning:codex_sdk_gpt_5_4_low@2",
                                payload={
                                    "adapter": "codex_sdk_gpt_5_4_low",
                                    "response_text": "Alice currently prefers espresso, and the old latte preference is stale.",
                                },
                            ),
                        ),
                        payload_overrides={
                            "resolution_queue": {
                                "surfaced_items": (
                                    {
                                        "item_id": "queue-check-oat",
                                        "subject_id": "subject:user:alice",
                                    },
                                ),
                            },
                            "utility_events": (
                                {"kind": "prompt_inclusion", "claim_id": "claim-espresso"},
                                {"kind": "answer_citation", "claim_id": "claim-espresso"},
                            ),
                        },
                    ),
                    expectation=ReplayEvaluationExpectation(
                        expected_retrieval_view_keys=(
                            "state:subject:user:alice:preference/favorite_drink",
                            "timeline:subject:user:alice:preference/favorite_drink",
                        ),
                        expected_selected_claim_ids=("claim-espresso",),
                        expected_answer_substrings=("Alice currently prefers espresso",),
                        expected_disclosure_result="visible",
                        forbidden_claim_ids=("claim-private",),
                        expected_queue_item_ids=("queue-check-oat",),
                        expected_utility_event_kinds=("prompt_inclusion", "answer_citation"),
                        fresh_claim_ids=("claim-espresso",),
                    ),
                ),
            )
        )

        self.assertEqual(tuple(evaluations), ("belief-freshness", "end-to-end-answer"))
        end_to_end = evaluations["end-to-end-answer"]
        self.assertEqual(
            end_to_end.comparison_record.comparison.changed_steps,
            frozenset({ReplayStep.RETRIEVAL, ReplayStep.BELIEF, ReplayStep.REASONING}),
        )
        self.assertEqual(end_to_end.comparison_record.comparison.policy_changed, True)
        self.assertEqual(
            end_to_end.candidate_scores[ReplayMetric.DISCLOSURE_SAFETY],
            5,
        )
        self.assertGreater(
            end_to_end.candidate_scores[ReplayMetric.QUEUE_YIELD],
            end_to_end.baseline_scores[ReplayMetric.QUEUE_YIELD],
        )
        self.assertGreater(
            end_to_end.candidate_scores[ReplayMetric.UTILITY_ALIGNMENT],
            end_to_end.baseline_scores[ReplayMetric.UTILITY_ALIGNMENT],
        )
        self.assertIsNotNone(
            replay_repository.read_comparison("comparison:fixture:end-to-end"),
        )


if __name__ == "__main__":
    unittest.main()
