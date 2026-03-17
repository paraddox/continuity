#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import continuity.migration as migration_module
except ModuleNotFoundError:
    migration_module = None

from continuity.forgetting import (
    ForgettingDecisionTrace,
    ForgettingMode,
    ForgettingOperation,
    ForgettingSurface,
    ForgettingTarget,
    ForgettingTargetKind,
    ForgettingTombstone,
    forgetting_rule_for,
)
from continuity.store.claims import AdmissionOutcome, SubjectKind
from continuity.store.schema import apply_migrations
from continuity.store.sqlite import SQLiteRepository


MigrationEntry = getattr(migration_module, "MigrationEntry", None)
artifact_identity_for_entry = getattr(migration_module, "artifact_identity_for_entry", None)
import_legacy_history = getattr(migration_module, "import_legacy_history", None)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def open_memory_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    apply_migrations(connection)
    return connection


def sample_entries() -> tuple[object, ...]:
    assert MigrationEntry is not None
    return (
        MigrationEntry(
            upload_name="prior_history.txt",
            content=(
                "<prior_conversation_history>\n"
                "<context>\n"
                "Imported before Honcho activation.\n"
                "</context>\n"
                "\n"
                "<transcript session_key=\"telegram:123456\" message_count=\"2\">\n"
                "[2026-01-01T00:00:00] user: Hello\n"
                "[2026-01-01T00:01:00] assistant: Hi!\n"
                "</transcript>\n"
                "</prior_conversation_history>\n"
            ),
            metadata={"source": "local_jsonl"},
        ),
        MigrationEntry(
            upload_name="consolidated_memory.md",
            content=(
                "<prior_memory_file>\n"
                "<context>\n"
                "Long-term notes.\n"
                "</context>\n"
                "\n"
                "User prefers concise technical answers.\n"
                "</prior_memory_file>\n"
            ),
            metadata={
                "source": "local_memory",
                "original_file": "MEMORY.md",
                "target_peer": "user",
            },
        ),
        MigrationEntry(
            upload_name="user_profile.md",
            content=(
                "<prior_memory_file>\n"
                "<context>\n"
                "Profile notes.\n"
                "</context>\n"
                "\n"
                "User lives in Bucharest.\n"
                "</prior_memory_file>\n"
            ),
            metadata={
                "source": "local_memory",
                "original_file": "USER.md",
                "target_peer": "user",
            },
        ),
        MigrationEntry(
            upload_name="agent_soul.md",
            content=(
                "<prior_memory_file>\n"
                "<context>\n"
                "Assistant identity.\n"
                "</context>\n"
                "\n"
                "Hermes should stay concise and direct.\n"
                "</prior_memory_file>\n"
            ),
            metadata={
                "source": "local_memory",
                "original_file": "SOUL.md",
                "target_peer": "ai",
            },
        ),
    )


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(MigrationEntry)
        self.assertIsNotNone(artifact_identity_for_entry)
        self.assertIsNotNone(import_legacy_history)

    def test_openclaw_import_seeds_subjects_tracks_artifacts_and_promotes_user_files(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = SQLiteRepository(connection)

        result = import_legacy_history(
            repository,
            session_id="telegram:123456",
            source_kind="openclaw",
            entries=sample_entries(),
            imported_at=sample_time(10),
        )

        session = repository.read_session("telegram:123456")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.host_namespace, "hermes")

        subjects = {subject.subject_id: subject for subject in repository.list_subjects()}
        self.assertEqual(subjects["subject:user:self"].kind, SubjectKind.USER)
        self.assertEqual(subjects["subject:assistant:hermes"].kind, SubjectKind.ASSISTANT)
        self.assertIn("user", subjects["subject:user:self"].normalized_names)
        self.assertIn("assistant", subjects["subject:assistant:hermes"].normalized_names)
        self.assertIn("ai", subjects["subject:assistant:hermes"].normalized_names)

        observations = repository.list_observations(session_id="telegram:123456")
        self.assertEqual(len(observations), 5)
        self.assertTrue(
            all("import_run_id" in observation.metadata for observation in observations),
        )

        claims = repository.list_claims(subject_id="subject:user:self")
        self.assertEqual(len(claims), 2)
        self.assertEqual(
            tuple(sorted(claim.locus.locus_key for claim in claims)),
            ("biography/consolidated_memory", "biography/user_profile"),
        )

        decisions = repository.admissions.list_decisions()
        self.assertEqual(len(decisions), 3)
        self.assertEqual(
            tuple(sorted(trace.claim_type for trace in decisions)),
            ("assistant_self_model", "biography", "biography"),
        )
        self.assertEqual(
            {trace.decision.outcome for trace in decisions},
            {AdmissionOutcome.DURABLE_CLAIM, AdmissionOutcome.SESSION_EPHEMERAL},
        )

        import_run_rows = connection.execute(
            "SELECT import_run_id, source_kind, source_ref FROM import_runs",
        ).fetchall()
        artifact_rows = connection.execute(
            """
            SELECT source_kind, source_ref, subject_id
            FROM migration_artifacts
            ORDER BY source_ref
            """,
        ).fetchall()

        self.assertEqual(
            [tuple(row) for row in import_run_rows],
            [(result.import_run_id, "openclaw", "telegram:123456")],
        )
        self.assertEqual(len(artifact_rows), 4)
        self.assertEqual(
            tuple(row[1] for row in artifact_rows),
            ("MEMORY.md", "SOUL.md", "USER.md", "prior_history.txt"),
        )
        self.assertEqual(result.blocked_artifact_ids, ())
        self.assertEqual(len(result.artifact_ids), 4)
        self.assertEqual(len(result.claim_ids), 2)

    def test_import_is_idempotent_for_the_same_entry_set(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = SQLiteRepository(connection)
        entries = sample_entries()

        first = import_legacy_history(
            repository,
            session_id="telegram:123456",
            source_kind="openclaw",
            entries=entries,
            imported_at=sample_time(10),
        )
        second = import_legacy_history(
            repository,
            session_id="telegram:123456",
            source_kind="openclaw",
            entries=entries,
            imported_at=sample_time(20),
        )

        self.assertEqual(second.import_run_id, first.import_run_id)
        self.assertEqual(second.artifact_ids, first.artifact_ids)
        self.assertEqual(second.claim_ids, first.claim_ids)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM import_runs").fetchone()[0], 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM migration_artifacts").fetchone()[0], 4)
        self.assertEqual(len(repository.list_observations(session_id="telegram:123456")), 5)
        self.assertEqual(len(repository.admissions.list_decisions()), 3)
        self.assertEqual(len(repository.list_claims(subject_id="subject:user:self")), 2)

    def test_expunge_tombstone_blocks_matching_artifact_reimport(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        repository = SQLiteRepository(connection)
        entries = sample_entries()[1:2]
        entry = entries[0]

        identity = artifact_identity_for_entry(
            session_id="telegram:123456",
            source_kind="openclaw",
            entry=entry,
        )
        target = ForgettingTarget(
            target_kind=ForgettingTargetKind.IMPORTED_ARTIFACT,
            target_id=identity.artifact_id,
        )
        repository.forgetting.record_operation(
            ForgettingDecisionTrace(
                operation=ForgettingOperation(
                    operation_id="forget-imported-memory",
                    target=target,
                    mode=ForgettingMode.EXPUNGE,
                    requested_by="subject:user:self",
                    rationale="Delete imported memory permanently.",
                    policy_stamp="hermes_v1@1.0.0",
                    recorded_at=sample_time(),
                ),
                rule=forgetting_rule_for(ForgettingMode.EXPUNGE),
            ),
            tombstones=(
                ForgettingTombstone(
                    tombstone_id="tombstone-imported-memory",
                    operation_id="forget-imported-memory",
                    target=target,
                    surface=ForgettingSurface.IMPORT_PIPELINE,
                    content_fingerprint=identity.content_fingerprint,
                    recorded_at=sample_time(),
                ),
            ),
        )

        result = import_legacy_history(
            repository,
            session_id="telegram:123456",
            source_kind="openclaw",
            entries=entries,
            imported_at=sample_time(10),
        )

        self.assertEqual(result.blocked_artifact_ids, (identity.artifact_id,))
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM migration_artifacts").fetchone()[0], 0)
        self.assertEqual(len(repository.list_observations()), 0)
        self.assertEqual(len(repository.admissions.list_decisions()), 0)
        self.assertEqual(len(repository.list_claims()), 0)


if __name__ == "__main__":
    unittest.main()
