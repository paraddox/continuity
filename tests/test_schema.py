#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.store.schema import (
    CURRENT_SCHEMA_VERSION,
    apply_migrations,
    current_schema_version,
)


def open_memory_database() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    return {row[0] for row in rows}


def foreign_keys_for(connection: sqlite3.Connection, table_name: str) -> set[tuple[str, str, str]]:
    rows = connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
    return {(row[3], row[2], row[4]) for row in rows}


class SchemaMigrationTests(unittest.TestCase):
    def test_apply_migrations_creates_full_memory_model_tables(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)

        applied_version = apply_migrations(connection)

        self.assertEqual(applied_version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(current_schema_version(connection), CURRENT_SCHEMA_VERSION)
        self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertTrue(
            {
                "schema_migrations",
                "runtime_config",
                "compatibility_state",
                "sessions",
                "session_messages",
                "session_ephemeral_buffers",
                "subjects",
                "subject_aliases",
                "subject_links",
                "subject_merges",
                "subject_splits",
                "import_runs",
                "migration_artifacts",
                "observations",
                "candidate_memories",
                "admission_write_budgets",
                "admission_decisions",
                "disclosure_policies",
                "memory_loci",
                "derivation_runs",
                "claims",
                "claim_sources",
                "claim_relations",
                "belief_states",
                "compiled_views",
                "compiled_view_claims",
                "compiled_view_observations",
                "vector_index_records",
                "arbiter_publications",
                "system_events",
                "compiler_nodes",
                "compiler_dependencies",
                "compiler_dirty_queue",
                "resolution_queue_items",
                "resolution_actions",
                "forgetting_operations",
                "forgetting_surface_effects",
                "forgetting_tombstones",
                "outcome_records",
                "utility_events",
                "compiled_utility_weights",
                "snapshots",
                "snapshot_artifacts",
                "snapshot_heads",
                "snapshot_read_pins",
                "snapshot_promotions",
                "replay_input_bundles",
                "replay_runs",
                "replay_run_strategies",
                "replay_artifacts",
                "tier_assignments",
                "tier_transitions",
                "prefetch_state",
            }.issubset(table_names(connection))
        )

    def test_core_foreign_keys_anchor_claims_views_and_snapshots(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        apply_migrations(connection)

        self.assertTrue(
            {
                ("subject_id", "subjects", "subject_id"),
                ("locus_id", "memory_loci", "locus_id"),
                ("candidate_id", "candidate_memories", "candidate_id"),
                ("admission_candidate_id", "admission_decisions", "candidate_id"),
                ("derivation_run_id", "derivation_runs", "derivation_run_id"),
            }.issubset(foreign_keys_for(connection, "claims"))
        )
        self.assertTrue(
            {
                ("snapshot_id", "snapshots", "snapshot_id"),
            }.issubset(foreign_keys_for(connection, "compiled_views"))
        )
        self.assertTrue(
            {
                ("parent_snapshot_id", "snapshots", "snapshot_id"),
            }.issubset(foreign_keys_for(connection, "snapshots"))
        )
        self.assertTrue(
            {
                ("snapshot_id", "snapshots", "snapshot_id"),
            }.issubset(foreign_keys_for(connection, "snapshot_artifacts"))
        )
        self.assertTrue(
            {
                ("snapshot_id", "snapshots", "snapshot_id"),
                ("based_on_snapshot_id", "snapshots", "snapshot_id"),
            }.issubset(foreign_keys_for(connection, "snapshot_heads"))
        )

    def test_schema_round_trip_is_idempotent(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        apply_migrations(connection)

        connection.execute(
            """
            INSERT INTO subjects(subject_id, kind, canonical_name, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "subject:user:alice",
                "user",
                "Alice Example",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.commit()

        reapplied_version = apply_migrations(connection)

        self.assertEqual(reapplied_version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(
            connection.execute("SELECT canonical_name FROM subjects WHERE subject_id = ?", ("subject:user:alice",)).fetchone()[0],
            "Alice Example",
        )


class SchemaRoundTripTests(unittest.TestCase):
    def test_representative_round_trip_spans_core_runtime_domains(self) -> None:
        connection = open_memory_database()
        self.addCleanup(connection.close)
        apply_migrations(connection)

        connection.execute(
            """
            INSERT INTO disclosure_policies(
                policy_id,
                audience_principal,
                channel,
                purpose,
                exposure_mode,
                redaction_mode,
                capture_for_replay,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "assistant_internal",
                "assistant",
                "prompt",
                "prompt",
                "direct",
                "none",
                1,
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO subjects(subject_id, kind, canonical_name, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "subject:user:alice",
                "user",
                "Alice Example",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO sessions(
                session_id,
                host_namespace,
                session_name,
                recall_mode,
                write_frequency,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "telegram:123456",
                "hermes",
                "telegram:123456",
                "auto",
                "async",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO session_messages(
                message_id,
                session_id,
                role,
                author_subject_id,
                content,
                observed_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg-1",
                "telegram:123456",
                "user",
                "subject:user:alice",
                "Alice prefers espresso.",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO observations(
                observation_id,
                source_kind,
                session_id,
                author_subject_id,
                message_id,
                content,
                observed_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "obs-1",
                "message",
                "telegram:123456",
                "subject:user:alice",
                "msg-1",
                "Alice prefers espresso.",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO candidate_memories(
                candidate_id,
                claim_type,
                subject_id,
                scope,
                value_json,
                source_observation_ids_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "candidate-1",
                "preference",
                "subject:user:alice",
                "user",
                '{"drink":"espresso"}',
                '["obs-1"]',
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO admission_write_budgets(partition, window_key, limit_value, used_value)
            VALUES (?, ?, ?, ?)
            """,
            (
                "user_memory",
                "2026-03-17",
                10,
                0,
            ),
        )
        connection.execute(
            """
            INSERT INTO admission_decisions(
                candidate_id,
                outcome,
                recorded_at,
                rationale,
                policy_stamp,
                claim_type,
                assessment_json,
                thresholds_json,
                budget_partition,
                budget_window_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "candidate-1",
                "durable_claim",
                "2026-03-17T12:00:00+00:00",
                "explicit user statement",
                "hermes_v1",
                "preference",
                '{"evidence":"high","novelty":"high","stability":"high","salience":"high"}',
                '{"evidence":"medium","novelty":"medium","stability":"medium","salience":"medium"}',
                "user_memory",
                "2026-03-17",
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_loci(
                locus_id,
                subject_id,
                locus_key,
                scope,
                default_disclosure_policy_id,
                conflict_set_key,
                aggregation_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "locus-1",
                "subject:user:alice",
                "preference/coffee",
                "user",
                "assistant_internal",
                "preference/coffee",
                "latest_wins",
            ),
        )
        connection.execute(
            """
            INSERT INTO derivation_runs(
                derivation_run_id,
                adapter_name,
                adapter_version,
                policy_stamp,
                schema_name,
                source_transaction_kind,
                created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "codex_sdk",
                "gpt-5.4-low",
                "hermes_v1",
                "claims.v1",
                "ingest_turn",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO claims(
                claim_id,
                claim_type,
                subject_id,
                locus_id,
                scope,
                disclosure_policy_id,
                value_json,
                candidate_id,
                admission_candidate_id,
                observed_at,
                learned_at,
                valid_from,
                valid_to,
                derivation_run_id,
                confidence_json,
                support_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "claim-1",
                "preference",
                "subject:user:alice",
                "locus-1",
                "user",
                "assistant_internal",
                '{"drink":"espresso"}',
                "candidate-1",
                "candidate-1",
                "2026-03-17T12:00:00+00:00",
                "2026-03-17T12:00:00+00:00",
                "2026-03-17T12:00:00+00:00",
                None,
                "run-1",
                "{}",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO claim_sources(claim_id, observation_id, source_rank)
            VALUES (?, ?, ?)
            """,
            ("claim-1", "obs-1", 0),
        )
        connection.execute(
            """
            INSERT INTO belief_states(
                belief_id,
                subject_id,
                locus_id,
                policy_stamp,
                epistemic_status,
                rationale,
                active_claim_ids_json,
                historical_claim_ids_json,
                as_of
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "belief-1",
                "subject:user:alice",
                "locus-1",
                "hermes_v1",
                "supported",
                "latest supported claim wins",
                '["claim-1"]',
                '["claim-1"]',
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO snapshots(snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction)
            VALUES (?, ?, ?, ?)
            """,
            ("snapshot-1", "hermes_v1", None, "publish_snapshot"),
        )
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
                "view-1",
                "state",
                "subject:user:alice/preference/coffee",
                "hermes_v1",
                "snapshot-1",
                "supported",
                '{"drink":"espresso"}',
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO compiled_view_claims(compiled_view_id, claim_id)
            VALUES (?, ?)
            """,
            ("view-1", "claim-1"),
        )
        connection.execute(
            """
            INSERT INTO compiled_view_observations(compiled_view_id, observation_id)
            VALUES (?, ?)
            """,
            ("view-1", "obs-1"),
        )
        connection.execute(
            """
            INSERT INTO vector_index_records(
                record_id,
                source_kind,
                source_id,
                subject_id,
                locus_key,
                policy_stamp,
                document_text,
                embedding_model,
                embedding_fingerprint,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "vec-1",
                "claim",
                "claim-1",
                "subject:user:alice",
                "preference/coffee",
                "hermes_v1",
                "Alice prefers espresso.",
                "nomic-embed-text",
                "embed@1",
                "{}",
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
                1,
                "snapshot_head_promotion",
                "publish_snapshot",
                "publish_snapshot",
                '["snapshot-1"]',
                "2026-03-17T12:00:00+00:00",
                "head:active",
                "snapshot_published",
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
                1,
                "snapshot_published",
                "publish_snapshot",
                1,
                "reference",
                "2026-03-17T12:00:00+00:00",
                '["snapshot-1"]',
                "[]",
                '["snapshot-1"]',
                "snapshot_published",
            ),
        )
        connection.execute(
            """
            INSERT INTO replay_input_bundles(
                bundle_id,
                surface,
                snapshot_id,
                journal_position,
                arbiter_lane_position,
                disclosure_context_json,
                claim_ids_json,
                observation_ids_json,
                compiled_view_ids_json,
                outcome_ids_json,
                reference_ids_json,
                query_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bundle-1",
                "prompt_view",
                "snapshot-1",
                1,
                1,
                '{"audience_principal":"assistant","channel":"prompt","purpose":"prompt","viewer":{"viewer_kind":"assistant","viewer_subject_id":"subject:assistant:continuity","active_user_id":"subject:user:alice","active_peer_id":"subject:user:alice"}}',
                '["claim-1"]',
                '["obs-1"]',
                '["view-1"]',
                "[]",
                '["snapshot-1"]',
                "What does Alice like to drink?",
            ),
        )
        connection.execute(
            """
            INSERT INTO replay_runs(
                run_id,
                bundle_id,
                policy_stamp,
                policy_fingerprint_id,
                mutation_mode,
                output_refs_json,
                metric_scores_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "replay-run-1",
                "bundle-1",
                "hermes_v1",
                "policy@1",
                "read_only",
                '["view-1"]',
                '{"correctness":1}',
            ),
        )
        connection.execute(
            """
            INSERT INTO replay_run_strategies(run_id, step, strategy_id, fingerprint)
            VALUES (?, ?, ?, ?)
            """,
            ("replay-run-1", "retrieval", "retrieval-default", "ret@1"),
        )
        connection.execute(
            """
            INSERT INTO replay_run_strategies(run_id, step, strategy_id, fingerprint)
            VALUES (?, ?, ?, ?)
            """,
            ("replay-run-1", "belief", "belief-default", "belief@1"),
        )
        connection.execute(
            """
            INSERT INTO replay_run_strategies(run_id, step, strategy_id, fingerprint)
            VALUES (?, ?, ?, ?)
            """,
            ("replay-run-1", "reasoning", "reasoning-default", "reason@1"),
        )
        connection.execute(
            """
            INSERT INTO replay_artifacts(
                artifact_id,
                version,
                source_transaction_kind,
                source_waterline,
                captured_at,
                baseline_run_id,
                source_object_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "replay-artifact-1",
                "replay_v1",
                "publish_snapshot",
                "snapshot_published",
                "2026-03-17T12:00:00+00:00",
                "replay-run-1",
                '["snapshot-1"]',
            ),
        )
        connection.execute(
            """
            INSERT INTO outcome_records(
                outcome_id,
                label,
                target,
                target_id,
                policy_stamp,
                recorded_at,
                rationale,
                actor_subject_id,
                claim_ids_json,
                observation_ids_json,
                capture_for_replay
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "outcome-1",
                "prompt_included",
                "claim",
                "claim-1",
                "hermes_v1",
                "2026-03-17T12:00:00+00:00",
                "included in prompt",
                "subject:user:alice",
                '["claim-1"]',
                '["obs-1"]',
                1,
            ),
        )
        connection.execute(
            """
            INSERT INTO utility_events(
                event_id,
                source_outcome_id,
                signal,
                target,
                target_id,
                policy_stamp,
                recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "utility-event-1",
                "outcome-1",
                "prompt_inclusion",
                "claim",
                "claim-1",
                "hermes_v1",
                "2026-03-17T12:00:00+00:00",
            ),
        )
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
                "claim",
                "claim-1",
                "hermes_v1",
                3,
                '{"prompt_inclusion":1}',
                '["utility-event-1"]',
            ),
        )
        connection.execute(
            """
            INSERT INTO forgetting_operations(
                operation_id,
                target_kind,
                target_id,
                mode,
                requested_by,
                rationale,
                policy_stamp,
                recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "forget-1",
                "claim",
                "claim-1",
                "suppress",
                "subject:user:alice",
                "user requested hiding",
                "hermes_v1",
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO forgetting_surface_effects(
                operation_id,
                surface,
                residency,
                blocks_resurrection
            )
            VALUES (?, ?, ?, ?)
            """,
            ("forget-1", "snapshot_store", "removed", 0),
        )
        connection.execute(
            """
            INSERT INTO forgetting_tombstones(
                tombstone_id,
                operation_id,
                target_kind,
                target_id,
                surface,
                content_fingerprint,
                recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tombstone-1",
                "forget-1",
                "claim",
                "claim-1",
                "snapshot_store",
                "sha256:abc",
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO compiler_nodes(node_id, category, kind, fingerprint, subject_id, locus_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "node-1",
                "derived_ir",
                "claim",
                "claim@1",
                "subject:user:alice",
                "preference/coffee",
            ),
        )
        connection.execute(
            """
            INSERT INTO compiler_nodes(node_id, category, kind, fingerprint, subject_id, locus_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "node-2",
                "compiled_artifact",
                "state_view",
                "view@1",
                "subject:user:alice",
                "preference/coffee",
            ),
        )
        connection.execute(
            """
            INSERT INTO compiler_dependencies(upstream_node_id, downstream_node_id, role)
            VALUES (?, ?, ?)
            """,
            ("node-1", "node-2", "projection"),
        )
        connection.execute(
            """
            INSERT INTO compiler_dirty_queue(
                node_id,
                reason,
                subject_id,
                locus_key,
                causes_json,
                status,
                queued_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "node-2",
                "claim_corrected",
                "subject:user:alice",
                "preference/coffee",
                '[{"changed_node_id":"node-1","dependency_path":["node-1","node-2"]}]',
                "pending",
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO resolution_queue_items(
                item_id,
                source,
                priority,
                subject_id,
                locus_key,
                rationale,
                created_at,
                utility_boost,
                status,
                surfaces_json,
                deferred_until,
                batch_key,
                candidate_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "resolution-1",
                "needs_confirmation",
                "high",
                "subject:user:alice",
                "preference/coffee",
                "confirm espresso preference",
                "2026-03-17T12:00:00+00:00",
                2,
                "open",
                '["host_api"]',
                None,
                None,
                "candidate-1",
            ),
        )
        connection.execute(
            """
            INSERT INTO resolution_actions(item_id, action, rationale, recorded_at, resulting_admission_outcome, effects_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "resolution-1",
                "confirm",
                "user confirmed preference",
                "2026-03-17T12:01:00+00:00",
                "durable_claim",
                '["admission","belief_revision","outcome_recording","replay_capture"]',
            ),
        )
        connection.execute(
            """
            INSERT INTO snapshot_artifacts(snapshot_id, artifact_kind, artifact_id)
            VALUES (?, ?, ?)
            """,
            ("snapshot-1", "state_view", "view-1"),
        )
        connection.execute(
            """
            INSERT INTO snapshot_heads(head_key, state, snapshot_id, based_on_snapshot_id)
            VALUES (?, ?, ?, ?)
            """,
            ("head:active", "active", "snapshot-1", None),
        )
        connection.execute(
            """
            INSERT INTO snapshot_read_pins(snapshot_id, read_use, consumer_id)
            VALUES (?, ?, ?)
            """,
            ("snapshot-1", "prompt_assembly", "hermes"),
        )
        connection.execute(
            """
            INSERT INTO snapshot_promotions(
                promotion_id,
                head_key,
                previous_active_snapshot_id,
                promoted_snapshot_id,
                recorded_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("promotion-1", "head:active", "snapshot-1", "snapshot-1", "2026-03-17T12:00:00+00:00"),
        )
        connection.execute(
            """
            INSERT INTO tier_assignments(
                target_kind,
                target_id,
                policy_stamp,
                tier,
                rationale,
                assigned_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "claim",
                "claim-1",
                "hermes_v1",
                "warm",
                "default durable preference tier",
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO tier_transitions(
                transition_id,
                target_kind,
                target_id,
                policy_stamp,
                from_tier,
                to_tier,
                rationale,
                transitioned_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tier-transition-1",
                "claim",
                "claim-1",
                "hermes_v1",
                "warm",
                "hot",
                "frequently reused",
                "2026-03-17T12:10:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO prefetch_state(
                prefetch_key,
                snapshot_id,
                session_id,
                status,
                artifact_ids_json,
                warmed_at,
                invalidated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "prefetch:telegram:123456",
                "snapshot-1",
                "telegram:123456",
                "warm",
                '["view-1"]',
                "2026-03-17T12:00:00+00:00",
                None,
            ),
        )
        connection.execute(
            """
            INSERT INTO import_runs(
                import_run_id,
                source_kind,
                source_ref,
                policy_stamp,
                started_at,
                finished_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "import-1",
                "hermes_local_files",
                "/tmp/hermes-memory.json",
                "hermes_v1",
                "2026-03-17T11:59:00+00:00",
                "2026-03-17T12:00:00+00:00",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO migration_artifacts(
                artifact_id,
                import_run_id,
                source_kind,
                source_ref,
                subject_id,
                imported_at,
                content_fingerprint,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "migration-1",
                "import-1",
                "profile_file",
                "/tmp/profile.md",
                "subject:user:alice",
                "2026-03-17T12:00:00+00:00",
                "sha256:def",
                "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO runtime_config(config_key, scope_kind, scope_id, value_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "write_frequency",
                "session",
                "telegram:123456",
                '"async"',
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO compatibility_state(compatibility_key, compatibility_value, updated_at)
            VALUES (?, ?, ?)
            """,
            (
                "host_api_contract",
                "transport-neutral-v1",
                "2026-03-17T12:00:00+00:00",
            ),
        )
        connection.commit()

        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
            1,
        )
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM compiled_views").fetchone()[0],
            1,
        )
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM replay_artifacts").fetchone()[0],
            1,
        )
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM tier_assignments").fetchone()[0],
            1,
        )


if __name__ == "__main__":
    unittest.main()
