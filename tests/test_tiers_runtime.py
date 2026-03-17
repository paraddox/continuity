#!/usr/bin/env python3

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import continuity.tiers as tiers_module
from continuity.epistemics import EpistemicStatus
from continuity.outcomes import OutcomeTarget
from continuity.store.schema import apply_migrations
from continuity.tiers import MemoryTier
from continuity.transactions import TransactionKind
from continuity.views import ViewKind


TierAssignment = getattr(tiers_module, "TierAssignment", None)
TierRuntime = getattr(tiers_module, "TierRuntime", None)
TierStateRepository = getattr(tiers_module, "TierStateRepository", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def seed_snapshot(connection: sqlite3.Connection, *, snapshot_id: str) -> None:
    connection.execute(
        """
        INSERT INTO snapshots(snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction)
        VALUES (?, ?, ?, ?)
        """,
        (snapshot_id, "hermes_v1@1.0.0", None, TransactionKind.PUBLISH_SNAPSHOT.value),
    )


def seed_compiled_view(
    connection: sqlite3.Connection,
    *,
    compiled_view_id: str,
    kind: ViewKind,
    view_key: str,
    snapshot_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO compiled_views(
            compiled_view_id,
            kind,
            view_key,
            policy_stamp,
            snapshot_id,
            epistemic_status,
            payload_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            compiled_view_id,
            kind.value,
            view_key,
            "hermes_v1@1.0.0",
            snapshot_id,
            EpistemicStatus.SUPPORTED.value,
            json.dumps({}, sort_keys=True),
            sample_time().isoformat(),
        ),
    )


def save_weight(
    connection: sqlite3.Connection,
    *,
    target: OutcomeTarget,
    target_id: str,
    weighted_score: int,
    signal_counts: dict[str, int],
) -> None:
    connection.execute(
        """
        INSERT INTO compiled_utility_weights(
            target,
            target_id,
            policy_stamp,
            weighted_score,
            signal_counts_json,
            source_event_ids_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            target.value,
            target_id,
            "hermes_v1@1.0.0",
            weighted_score,
            json.dumps(signal_counts, sort_keys=True),
            json.dumps([], sort_keys=True),
        ),
    )


class TierRuntimeTests(unittest.TestCase):
    def test_runtime_rebalances_compiled_view_working_sets_and_exposes_default_reads(self) -> None:
        self.assertIsNotNone(TierRuntime)
        self.assertIsNotNone(TierAssignment)
        self.assertIsNotNone(TierStateRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)
        seed_snapshot(connection, snapshot_id="snapshot:active")
        seed_compiled_view(
            connection,
            compiled_view_id="view-hot",
            kind=ViewKind.STATE,
            view_key="state:subject:user:alice:preference/favorite_drink",
            snapshot_id="snapshot:active",
        )
        seed_compiled_view(
            connection,
            compiled_view_id="view-warm",
            kind=ViewKind.PROFILE,
            view_key="profile:subject:user:alice",
            snapshot_id="snapshot:active",
        )
        seed_compiled_view(
            connection,
            compiled_view_id="view-cold",
            kind=ViewKind.TIMELINE,
            view_key="timeline:subject:user:alice:task_state/project_status",
            snapshot_id="snapshot:active",
        )

        repository = TierStateRepository(connection)
        repository.upsert_assignment(
            TierAssignment(
                target_kind="compiled_view",
                target_id="view-hot",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.WARM,
                rationale="seeded warm",
                assigned_at=sample_time(),
            )
        )
        repository.upsert_assignment(
            TierAssignment(
                target_kind="compiled_view",
                target_id="view-warm",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.HOT,
                rationale="seeded hot",
                assigned_at=sample_time(1),
            )
        )
        repository.upsert_assignment(
            TierAssignment(
                target_kind="compiled_view",
                target_id="view-cold",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.WARM,
                rationale="seeded warm",
                assigned_at=sample_time(2),
            )
        )
        save_weight(
            connection,
            target=OutcomeTarget.COMPILED_VIEW,
            target_id="view-hot",
            weighted_score=9,
            signal_counts={"answer_citation": 1, "prompt_inclusion": 1},
        )
        save_weight(
            connection,
            target=OutcomeTarget.COMPILED_VIEW,
            target_id="view-warm",
            weighted_score=4,
            signal_counts={"prompt_inclusion": 1},
        )
        save_weight(
            connection,
            target=OutcomeTarget.COMPILED_VIEW,
            target_id="view-cold",
            weighted_score=-4,
            signal_counts={"stale_on_use": 1},
        )
        connection.commit()

        runtime = TierRuntime(connection)
        transitions = runtime.rebalance_working_set(
            target_kind="compiled_view",
            policy_stamp="hermes_v1@1.0.0",
            hot_limit=1,
            warm_limit=1,
            transitioned_at=sample_time(10),
        )

        self.assertEqual(
            tuple((transition.target_id, transition.from_tier, transition.to_tier) for transition in transitions),
            (
                ("view-hot", MemoryTier.WARM, MemoryTier.HOT),
                ("view-warm", MemoryTier.HOT, MemoryTier.WARM),
                ("view-cold", MemoryTier.WARM, MemoryTier.COLD),
            ),
        )
        self.assertIn("utility score 9", transitions[0].rationale)
        self.assertIn("hot working set", transitions[0].rationale)
        self.assertIn("warm working set", transitions[1].rationale)
        self.assertIn("recallable cold", transitions[2].rationale)
        self.assertEqual(
            tuple(
                metadata.target_id
                for metadata in runtime.default_read_metadata(
                    target_kind="compiled_view",
                    policy_stamp="hermes_v1@1.0.0",
                )
            ),
            ("view-hot", "view-warm"),
        )
        self.assertEqual(
            tuple(
                metadata.target_id
                for metadata in runtime.recall_metadata(
                    target_kind="compiled_view",
                    policy_stamp="hermes_v1@1.0.0",
                    include_archival=False,
                )
            ),
            ("view-cold",),
        )

    def test_runtime_archives_long_tail_replay_artifacts_into_frozen_tier(self) -> None:
        self.assertIsNotNone(TierRuntime)
        self.assertIsNotNone(TierAssignment)
        self.assertIsNotNone(TierStateRepository)

        connection = open_memory_database()
        self.addCleanup(connection.close)

        repository = TierStateRepository(connection)
        repository.upsert_assignment(
            TierAssignment(
                target_kind="replay_artifact",
                target_id="replay:older",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.COLD,
                rationale="seeded cold",
                assigned_at=sample_time(),
            )
        )
        repository.upsert_assignment(
            TierAssignment(
                target_kind="replay_artifact",
                target_id="replay:newer",
                policy_stamp="hermes_v1@1.0.0",
                tier=MemoryTier.COLD,
                rationale="seeded cold",
                assigned_at=sample_time(1),
            )
        )

        runtime = TierRuntime(connection)
        transitions = runtime.archive_long_tail(
            target_kind="replay_artifact",
            policy_stamp="hermes_v1@1.0.0",
            keep_latest=1,
            transitioned_at=sample_time(10),
        )

        self.assertEqual(
            tuple((transition.target_id, transition.from_tier, transition.to_tier) for transition in transitions),
            (("replay:older", MemoryTier.COLD, MemoryTier.FROZEN),),
        )
        self.assertIn("archival policy", transitions[0].rationale)
        self.assertEqual(
            tuple(
                metadata.target_id
                for metadata in runtime.default_read_metadata(
                    target_kind="replay_artifact",
                    policy_stamp="hermes_v1@1.0.0",
                )
            ),
            (),
        )
        self.assertEqual(
            tuple(
                (metadata.target_id, metadata.tier)
                for metadata in runtime.recall_metadata(
                    target_kind="replay_artifact",
                    policy_stamp="hermes_v1@1.0.0",
                    include_archival=True,
                )
            ),
            (
                ("replay:newer", MemoryTier.COLD),
                ("replay:older", MemoryTier.FROZEN),
            ),
        )


if __name__ == "__main__":
    unittest.main()
