#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import continuity.store.replay as replay_store_module
except ModuleNotFoundError:
    replay_store_module = None

from continuity.arbiter import ArbiterPublicationKind
from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
from continuity.events import EventPayloadMode, SystemEventType
from continuity.replay import (
    ReplayArtifact,
    ReplayComparison,
    ReplayMetric,
    ReplayRun,
    ReplayStep,
    ReplayStrategy,
)
from continuity.store.schema import apply_migrations
from continuity.transactions import DurabilityWaterline, TransactionKind, TransactionPhase


ReplayComparisonRecord = getattr(replay_store_module, "ReplayComparisonRecord", None)
ReplayRepository = getattr(replay_store_module, "ReplayRepository", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


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


def seed_replay_prerequisites(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO snapshots(snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction)
        VALUES (?, ?, ?, ?)
        """,
        (
            "snapshot:active",
            "hermes_v1@1.0.0",
            None,
            TransactionKind.PUBLISH_SNAPSHOT.value,
        ),
    )
    connection.execute(
        """
        INSERT INTO arbiter_publications(
            lane_position,
            publication_kind,
            transaction_kind,
            phase,
            object_ids_json,
            published_at,
            snapshot_head_id,
            reached_waterline
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            17,
            ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION.value,
            TransactionKind.PUBLISH_SNAPSHOT.value,
            TransactionPhase.PUBLISH_SNAPSHOT.value,
            json.dumps(["snapshot:active"]),
            sample_time().isoformat(),
            "head:active",
            DurabilityWaterline.SNAPSHOT_PUBLISHED.value,
        ),
    )
    connection.execute(
        """
        INSERT INTO system_events(
            journal_position,
            event_type,
            transaction_kind,
            arbiter_lane_position,
            payload_mode,
            recorded_at,
            object_ids_json,
            inline_payload_json,
            reference_ids_json,
            waterline
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            21,
            SystemEventType.SNAPSHOT_PUBLISHED.value,
            TransactionKind.PUBLISH_SNAPSHOT.value,
            17,
            EventPayloadMode.REFERENCE.value,
            sample_time().isoformat(),
            json.dumps(["snapshot:active"]),
            json.dumps([]),
            json.dumps(["snapshot:active"]),
            DurabilityWaterline.SNAPSHOT_PUBLISHED.value,
        ),
    )
    connection.commit()


def sample_input_bundle():
    from continuity.replay import ReplayInputBundle

    return ReplayInputBundle(
        bundle_id="bundle-1",
        surface="answer_view",
        snapshot_id="snapshot:active",
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
    bundle=None,
    policy_fingerprint: tuple[str, str] = ("hermes_v1@1.0.0", "hermes_core_v1"),
    reasoning_fingerprint: str = "reasoning:codex_sdk_gpt_5_4_low@1",
) -> ReplayRun:
    input_bundle = sample_input_bundle() if bundle is None else bundle
    return ReplayRun(
        run_id=run_id,
        input_bundle=input_bundle,
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


def sample_artifact(*, run: ReplayRun) -> ReplayArtifact:
    return ReplayArtifact(
        artifact_id="replay:turn:1",
        version="replay_v1",
        source_transaction=TransactionKind.INGEST_TURN,
        source_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
        captured_at=sample_time(1),
        baseline_run=run,
        source_object_ids=("snapshot:active", "answer:1"),
    )


def sample_comparison_record(*, baseline: ReplayRun, candidate: ReplayRun):
    assert ReplayComparisonRecord is not None

    return ReplayComparisonRecord(
        comparison=ReplayComparison(
            comparison_id="cmp-1",
            baseline_run=baseline,
            candidate_run=candidate,
            compared_steps=(
                ReplayStep.RETRIEVAL,
                ReplayStep.BELIEF,
                ReplayStep.REASONING,
            ),
            rationale="compare upgraded policy and reasoning strategy",
        ),
        compared_at=sample_time(2),
        metric_deltas={
            ReplayMetric.CORRECTNESS: 1,
            ReplayMetric.FRESHNESS: 0,
            ReplayMetric.DISCLOSURE_SAFETY: 0,
            ReplayMetric.QUEUE_YIELD: 0,
            ReplayMetric.UTILITY_ALIGNMENT: 1,
        },
        notes=("policy changed", "reasoning improved"),
    )


class ReplayRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(ReplayRepository)
        self.assertIsNotNone(ReplayComparisonRecord)

    def test_repository_round_trips_replay_state_and_query_filters(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_replay_prerequisites(connection)
        repository = ReplayRepository(connection)

        bundle = sample_input_bundle()
        baseline = sample_run(run_id="run-1", bundle=bundle)
        candidate = sample_run(
            run_id="run-2",
            bundle=bundle,
            policy_fingerprint=("hermes_v1@1.1.0", "hermes_core_v1"),
            reasoning_fingerprint="reasoning:codex_sdk_gpt_5_4_low@2",
        )
        artifact = sample_artifact(run=baseline)
        comparison_record = sample_comparison_record(baseline=baseline, candidate=candidate)

        repository.save_input_bundle(bundle)
        repository.save_run(baseline)
        repository.save_run(candidate)
        repository.save_artifact(artifact)
        repository.record_comparison(comparison_record)

        self.assertEqual(repository.read_input_bundle(bundle.bundle_id), bundle)
        self.assertEqual(
            repository.list_input_bundles(snapshot_id=bundle.snapshot_id),
            (bundle,),
        )
        self.assertEqual(repository.read_run(baseline.run_id), baseline)
        self.assertEqual(
            repository.list_runs(bundle_id=bundle.bundle_id),
            (baseline, candidate),
        )
        self.assertEqual(repository.read_artifact(artifact.artifact_id), artifact)
        self.assertEqual(
            repository.list_artifacts(
                source_transaction=artifact.source_transaction,
                journal_position=bundle.journal_position,
            ),
            (artifact,),
        )
        self.assertEqual(
            repository.read_comparison(comparison_record.comparison.comparison_id),
            comparison_record,
        )
        self.assertEqual(
            repository.list_comparisons(run_id=baseline.run_id),
            (comparison_record,),
        )

    def test_repository_rejects_mutating_captured_replay_rows(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_replay_prerequisites(connection)
        repository = ReplayRepository(connection)

        bundle = sample_input_bundle()
        baseline = sample_run(run_id="run-1", bundle=bundle)
        candidate = sample_run(
            run_id="run-2",
            bundle=bundle,
            policy_fingerprint=("hermes_v1@1.1.0", "hermes_core_v1"),
            reasoning_fingerprint="reasoning:codex_sdk_gpt_5_4_low@2",
        )
        artifact = sample_artifact(run=baseline)
        comparison_record = sample_comparison_record(baseline=baseline, candidate=candidate)

        repository.save_input_bundle(bundle)
        repository.save_run(baseline)
        repository.save_run(candidate)
        repository.save_artifact(artifact)
        repository.record_comparison(comparison_record)

        with self.assertRaises(ValueError):
            repository.save_input_bundle(replace(bundle, query_text="Use a different wording"))
        with self.assertRaises(ValueError):
            repository.save_run(
                replace(
                    baseline,
                    policy_fingerprint=("hermes_v1@2.0.0", "hermes_core_v2"),
                )
            )
        with self.assertRaises(ValueError):
            repository.save_artifact(
                replace(
                    artifact,
                    source_object_ids=("snapshot:active", "answer:2"),
                )
            )
        with self.assertRaises(ValueError):
            repository.record_comparison(
                replace(comparison_record, notes=("reasoning regressed",))
            )


if __name__ == "__main__":
    unittest.main()
