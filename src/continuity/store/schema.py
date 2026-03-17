"""SQLite schema and migrations for the Continuity durable store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable

from continuity.admission import AdmissionStrength
from continuity.arbiter import ArbiterPublicationKind
from continuity.compiler import (
    CompiledArtifactKind,
    CompilerNodeCategory,
    DependencyRole,
    DerivedArtifactKind,
    DirtyReason,
    SourceInputKind,
    UtilityStateKind,
)
from continuity.epistemics import EpistemicStatus
from continuity.events import EventPayloadMode, SystemEventType
from continuity.forgetting import (
    ArtifactResidency,
    ForgettingMode,
    ForgettingSurface,
    ForgettingTargetKind,
)
from continuity.ontology import MemoryPartition
from continuity.outcomes import OutcomeLabel, OutcomeTarget
from continuity.replay import ReplayMutationMode, ReplayStep
from continuity.resolution_queue import (
    ResolutionAction,
    ResolutionPriority,
    ResolutionSource,
    ResolutionStatus,
)
from continuity.snapshots import SnapshotArtifactKind, SnapshotHeadState, SnapshotReadUse
from continuity.store.claims import (
    AdmissionOutcome,
    AggregationMode,
    ClaimRelationKind,
    ClaimScope,
    SubjectKind,
)
from continuity.tiers import MemoryTier
from continuity.transactions import DurabilityWaterline, TransactionKind, TransactionPhase
from continuity.utility import UtilitySignal
from continuity.views import ViewKind


CURRENT_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


def _enum_values(*enum_types: type[object]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for enum_type in enum_types:
        for member in enum_type:  # type: ignore[assignment]
            value = member.value  # type: ignore[attr-defined]
            if value not in seen:
                values.append(f"'{value}'")
                seen.add(value)
    return ", ".join(values)


def _enum_check(column_name: str, *enum_types: type[object]) -> str:
    return f"CHECK ({column_name} IN ({_enum_values(*enum_types)}))"


def _role_check(column_name: str) -> str:
    return f"CHECK ({column_name} IN ('user', 'assistant', 'system', 'tool', 'peer'))"


def _prefetch_status_check(column_name: str) -> str:
    return f"CHECK ({column_name} IN ('pending', 'warm', 'invalidated'))"


def _build_initial_schema_statements() -> tuple[str, ...]:
    return (
        f"""
        CREATE TABLE runtime_config (
            config_key TEXT NOT NULL,
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (config_key, scope_kind, scope_id)
        )
        """.strip(),
        """
        CREATE TABLE compatibility_state (
            compatibility_key TEXT PRIMARY KEY,
            compatibility_value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            host_namespace TEXT NOT NULL,
            session_name TEXT NOT NULL,
            recall_mode TEXT NOT NULL,
            write_frequency TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """.strip(),
        f"""
        CREATE TABLE session_messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL {_role_check("role")},
            author_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE RESTRICT,
            content TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """.strip(),
        """
        CREATE TABLE session_ephemeral_buffers (
            buffer_key TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            buffer_kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE subjects (
            subject_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL {_enum_check("kind", SubjectKind)},
            canonical_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """.strip(),
        """
        CREATE TABLE subject_aliases (
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            alias TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            alias_type TEXT NOT NULL,
            source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (subject_id, normalized_alias)
        )
        """.strip(),
        """
        CREATE TABLE subject_links (
            left_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            relation_kind TEXT NOT NULL,
            right_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (left_subject_id, relation_kind, right_subject_id)
        )
        """.strip(),
        """
        CREATE TABLE subject_merges (
            survivor_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            merged_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (survivor_subject_id, merged_subject_id)
        )
        """.strip(),
        """
        CREATE TABLE subject_splits (
            source_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            child_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (source_subject_id, child_subject_id)
        )
        """.strip(),
        """
        CREATE TABLE import_runs (
            import_run_id TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """.strip(),
        """
        CREATE TABLE migration_artifacts (
            artifact_id TEXT PRIMARY KEY,
            import_run_id TEXT NOT NULL REFERENCES import_runs(import_run_id) ON DELETE CASCADE,
            source_kind TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            subject_id TEXT REFERENCES subjects(subject_id) ON DELETE SET NULL,
            imported_at TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """.strip(),
        """
        CREATE TABLE observations (
            observation_id TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            author_subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE RESTRICT,
            message_id TEXT REFERENCES session_messages(message_id) ON DELETE SET NULL,
            content TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """.strip(),
        f"""
        CREATE TABLE candidate_memories (
            candidate_id TEXT PRIMARY KEY,
            claim_type TEXT NOT NULL,
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE RESTRICT,
            scope TEXT NOT NULL {_enum_check("scope", ClaimScope)},
            value_json TEXT NOT NULL,
            source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE admission_write_budgets (
            partition TEXT NOT NULL {_enum_check("partition", MemoryPartition)},
            window_key TEXT NOT NULL,
            limit_value INTEGER NOT NULL,
            used_value INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (partition, window_key),
            CHECK (limit_value >= 0),
            CHECK (used_value >= 0),
            CHECK (used_value <= limit_value)
        )
        """.strip(),
        f"""
        CREATE TABLE admission_decisions (
            candidate_id TEXT PRIMARY KEY REFERENCES candidate_memories(candidate_id) ON DELETE CASCADE,
            outcome TEXT NOT NULL {_enum_check("outcome", AdmissionOutcome)},
            recorded_at TEXT NOT NULL,
            rationale TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            claim_type TEXT NOT NULL,
            assessment_json TEXT NOT NULL DEFAULT '{{}}',
            thresholds_json TEXT NOT NULL DEFAULT '{{}}',
            budget_partition TEXT,
            budget_window_key TEXT,
            FOREIGN KEY (budget_partition, budget_window_key)
                REFERENCES admission_write_budgets(partition, window_key)
                ON DELETE SET NULL
        )
        """.strip(),
        """
        CREATE TABLE disclosure_policies (
            policy_id TEXT PRIMARY KEY,
            audience_principal TEXT NOT NULL,
            channel TEXT NOT NULL,
            purpose TEXT NOT NULL,
            exposure_mode TEXT NOT NULL,
            redaction_mode TEXT NOT NULL,
            capture_for_replay INTEGER NOT NULL DEFAULT 1 CHECK (capture_for_replay IN (0, 1)),
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """.strip(),
        f"""
        CREATE TABLE memory_loci (
            locus_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            locus_key TEXT NOT NULL,
            scope TEXT NOT NULL {_enum_check("scope", ClaimScope)},
            default_disclosure_policy_id TEXT NOT NULL REFERENCES disclosure_policies(policy_id) ON DELETE RESTRICT,
            conflict_set_key TEXT NOT NULL,
            aggregation_mode TEXT NOT NULL {_enum_check("aggregation_mode", AggregationMode)},
            UNIQUE (subject_id, locus_key)
        )
        """.strip(),
        f"""
        CREATE TABLE derivation_runs (
            derivation_run_id TEXT PRIMARY KEY,
            adapter_name TEXT NOT NULL,
            adapter_version TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            source_transaction_kind TEXT {_enum_check("source_transaction_kind", TransactionKind)},
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """.strip(),
        f"""
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            claim_type TEXT NOT NULL,
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE RESTRICT,
            locus_id TEXT NOT NULL REFERENCES memory_loci(locus_id) ON DELETE RESTRICT,
            scope TEXT NOT NULL {_enum_check("scope", ClaimScope)},
            disclosure_policy_id TEXT NOT NULL REFERENCES disclosure_policies(policy_id) ON DELETE RESTRICT,
            value_json TEXT NOT NULL,
            candidate_id TEXT NOT NULL REFERENCES candidate_memories(candidate_id) ON DELETE RESTRICT,
            admission_candidate_id TEXT NOT NULL REFERENCES admission_decisions(candidate_id) ON DELETE RESTRICT,
            observed_at TEXT NOT NULL,
            learned_at TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            derivation_run_id TEXT REFERENCES derivation_runs(derivation_run_id) ON DELETE SET NULL,
            confidence_json TEXT NOT NULL DEFAULT '{{}}',
            support_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """.strip(),
        """
        CREATE TABLE claim_sources (
            claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
            source_rank INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (claim_id, observation_id)
        )
        """.strip(),
        f"""
        CREATE TABLE claim_relations (
            claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            relation_kind TEXT NOT NULL {_enum_check("relation_kind", ClaimRelationKind)},
            related_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            PRIMARY KEY (claim_id, relation_kind, related_claim_id)
        )
        """.strip(),
        f"""
        CREATE TABLE snapshots (
            snapshot_id TEXT PRIMARY KEY,
            policy_stamp TEXT NOT NULL,
            parent_snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE SET NULL,
            created_by_transaction TEXT NOT NULL {_enum_check("created_by_transaction", TransactionKind)}
        )
        """.strip(),
        f"""
        CREATE TABLE belief_states (
            belief_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            locus_id TEXT NOT NULL REFERENCES memory_loci(locus_id) ON DELETE CASCADE,
            policy_stamp TEXT NOT NULL,
            epistemic_status TEXT NOT NULL {_enum_check("epistemic_status", EpistemicStatus)},
            rationale TEXT NOT NULL,
            active_claim_ids_json TEXT NOT NULL DEFAULT '[]',
            historical_claim_ids_json TEXT NOT NULL DEFAULT '[]',
            as_of TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE compiled_views (
            compiled_view_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL {_enum_check("kind", ViewKind)},
            view_key TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            epistemic_status TEXT NOT NULL {_enum_check("epistemic_status", EpistemicStatus)},
            payload_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL,
            UNIQUE (kind, view_key, snapshot_id)
        )
        """.strip(),
        """
        CREATE TABLE compiled_view_claims (
            compiled_view_id TEXT NOT NULL REFERENCES compiled_views(compiled_view_id) ON DELETE CASCADE,
            claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            PRIMARY KEY (compiled_view_id, claim_id)
        )
        """.strip(),
        """
        CREATE TABLE compiled_view_observations (
            compiled_view_id TEXT NOT NULL REFERENCES compiled_views(compiled_view_id) ON DELETE CASCADE,
            observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
            PRIMARY KEY (compiled_view_id, observation_id)
        )
        """.strip(),
        """
        CREATE TABLE vector_index_records (
            record_id TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            subject_id TEXT REFERENCES subjects(subject_id) ON DELETE SET NULL,
            locus_key TEXT,
            policy_stamp TEXT NOT NULL,
            document_text TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_fingerprint TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """.strip(),
        f"""
        CREATE TABLE arbiter_publications (
            lane_position INTEGER PRIMARY KEY,
            publication_kind TEXT NOT NULL {_enum_check("publication_kind", ArbiterPublicationKind)},
            transaction_kind TEXT NOT NULL {_enum_check("transaction_kind", TransactionKind)},
            phase TEXT NOT NULL {_enum_check("phase", TransactionPhase)},
            object_ids_json TEXT NOT NULL DEFAULT '[]',
            published_at TEXT NOT NULL,
            snapshot_head_id TEXT,
            reached_waterline TEXT {_enum_check("reached_waterline", DurabilityWaterline)}
        )
        """.strip(),
        f"""
        CREATE TABLE system_events (
            journal_position INTEGER PRIMARY KEY,
            event_type TEXT NOT NULL {_enum_check("event_type", SystemEventType)},
            transaction_kind TEXT NOT NULL {_enum_check("transaction_kind", TransactionKind)},
            arbiter_lane_position INTEGER NOT NULL REFERENCES arbiter_publications(lane_position) ON DELETE CASCADE,
            payload_mode TEXT NOT NULL {_enum_check("payload_mode", EventPayloadMode)},
            recorded_at TEXT NOT NULL,
            object_ids_json TEXT NOT NULL DEFAULT '[]',
            inline_payload_json TEXT NOT NULL DEFAULT '[]',
            reference_ids_json TEXT NOT NULL DEFAULT '[]',
            waterline TEXT {_enum_check("waterline", DurabilityWaterline)}
        )
        """.strip(),
        f"""
        CREATE TABLE compiler_nodes (
            node_id TEXT PRIMARY KEY,
            category TEXT NOT NULL {_enum_check("category", CompilerNodeCategory)},
            kind TEXT NOT NULL {_enum_check("kind", SourceInputKind, DerivedArtifactKind, UtilityStateKind, CompiledArtifactKind)},
            fingerprint TEXT NOT NULL,
            subject_id TEXT,
            locus_key TEXT
        )
        """.strip(),
        f"""
        CREATE TABLE compiler_dependencies (
            upstream_node_id TEXT NOT NULL REFERENCES compiler_nodes(node_id) ON DELETE CASCADE,
            downstream_node_id TEXT NOT NULL REFERENCES compiler_nodes(node_id) ON DELETE CASCADE,
            role TEXT NOT NULL {_enum_check("role", DependencyRole)},
            PRIMARY KEY (upstream_node_id, downstream_node_id, role)
        )
        """.strip(),
        f"""
        CREATE TABLE compiler_dirty_queue (
            queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES compiler_nodes(node_id) ON DELETE CASCADE,
            reason TEXT NOT NULL {_enum_check("reason", DirtyReason)},
            subject_id TEXT,
            locus_key TEXT,
            causes_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done')),
            queued_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE resolution_queue_items (
            item_id TEXT PRIMARY KEY,
            source TEXT NOT NULL {_enum_check("source", ResolutionSource)},
            priority TEXT NOT NULL {_enum_check("priority", ResolutionPriority)},
            subject_id TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
            locus_key TEXT NOT NULL,
            rationale TEXT NOT NULL,
            created_at TEXT NOT NULL,
            utility_boost INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL {_enum_check("status", ResolutionStatus)},
            surfaces_json TEXT NOT NULL DEFAULT '[]',
            deferred_until TEXT,
            batch_key TEXT,
            candidate_id TEXT REFERENCES candidate_memories(candidate_id) ON DELETE SET NULL
        )
        """.strip(),
        f"""
        CREATE TABLE resolution_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL REFERENCES resolution_queue_items(item_id) ON DELETE CASCADE,
            action TEXT NOT NULL {_enum_check("action", ResolutionAction)},
            rationale TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            resulting_admission_outcome TEXT NOT NULL {_enum_check("resulting_admission_outcome", AdmissionOutcome)},
            effects_json TEXT NOT NULL DEFAULT '[]'
        )
        """.strip(),
        f"""
        CREATE TABLE forgetting_operations (
            operation_id TEXT PRIMARY KEY,
            target_kind TEXT NOT NULL {_enum_check("target_kind", ForgettingTargetKind)},
            target_id TEXT NOT NULL,
            mode TEXT NOT NULL {_enum_check("mode", ForgettingMode)},
            requested_by TEXT NOT NULL,
            rationale TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE forgetting_surface_effects (
            operation_id TEXT NOT NULL REFERENCES forgetting_operations(operation_id) ON DELETE CASCADE,
            surface TEXT NOT NULL {_enum_check("surface", ForgettingSurface)},
            residency TEXT NOT NULL {_enum_check("residency", ArtifactResidency)},
            blocks_resurrection INTEGER NOT NULL DEFAULT 0 CHECK (blocks_resurrection IN (0, 1)),
            PRIMARY KEY (operation_id, surface)
        )
        """.strip(),
        f"""
        CREATE TABLE forgetting_tombstones (
            tombstone_id TEXT PRIMARY KEY,
            operation_id TEXT NOT NULL REFERENCES forgetting_operations(operation_id) ON DELETE CASCADE,
            target_kind TEXT NOT NULL {_enum_check("target_kind", ForgettingTargetKind)},
            target_id TEXT NOT NULL,
            surface TEXT NOT NULL {_enum_check("surface", ForgettingSurface)},
            content_fingerprint TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE outcome_records (
            outcome_id TEXT PRIMARY KEY,
            label TEXT NOT NULL {_enum_check("label", OutcomeLabel)},
            target TEXT NOT NULL {_enum_check("target", OutcomeTarget)},
            target_id TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            rationale TEXT NOT NULL,
            actor_subject_id TEXT REFERENCES subjects(subject_id) ON DELETE SET NULL,
            claim_ids_json TEXT NOT NULL DEFAULT '[]',
            observation_ids_json TEXT NOT NULL DEFAULT '[]',
            capture_for_replay INTEGER NOT NULL DEFAULT 1 CHECK (capture_for_replay IN (0, 1))
        )
        """.strip(),
        f"""
        CREATE TABLE utility_events (
            event_id TEXT PRIMARY KEY,
            source_outcome_id TEXT NOT NULL REFERENCES outcome_records(outcome_id) ON DELETE CASCADE,
            signal TEXT NOT NULL {_enum_check("signal", UtilitySignal)},
            target TEXT NOT NULL {_enum_check("target", OutcomeTarget)},
            target_id TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE compiled_utility_weights (
            target TEXT NOT NULL {_enum_check("target", OutcomeTarget)},
            target_id TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            weighted_score INTEGER NOT NULL,
            signal_counts_json TEXT NOT NULL DEFAULT '{{}}',
            source_event_ids_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (target, target_id, policy_stamp)
        )
        """.strip(),
        f"""
        CREATE TABLE snapshot_artifacts (
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            artifact_kind TEXT NOT NULL {_enum_check("artifact_kind", SnapshotArtifactKind)},
            artifact_id TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, artifact_kind, artifact_id)
        )
        """.strip(),
        f"""
        CREATE TABLE snapshot_heads (
            head_key TEXT NOT NULL,
            state TEXT NOT NULL {_enum_check("state", SnapshotHeadState)},
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            based_on_snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE SET NULL,
            PRIMARY KEY (head_key, state)
        )
        """.strip(),
        f"""
        CREATE TABLE snapshot_read_pins (
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            read_use TEXT NOT NULL {_enum_check("read_use", SnapshotReadUse)},
            consumer_id TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, read_use, consumer_id)
        )
        """.strip(),
        """
        CREATE TABLE snapshot_promotions (
            promotion_id TEXT PRIMARY KEY,
            head_key TEXT NOT NULL,
            previous_active_snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE SET NULL,
            promoted_snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            recorded_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE replay_input_bundles (
            bundle_id TEXT PRIMARY KEY,
            surface TEXT NOT NULL,
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            journal_position INTEGER NOT NULL REFERENCES system_events(journal_position) ON DELETE CASCADE,
            arbiter_lane_position INTEGER NOT NULL REFERENCES arbiter_publications(lane_position) ON DELETE CASCADE,
            disclosure_context_json TEXT NOT NULL,
            claim_ids_json TEXT NOT NULL DEFAULT '[]',
            observation_ids_json TEXT NOT NULL DEFAULT '[]',
            compiled_view_ids_json TEXT NOT NULL DEFAULT '[]',
            outcome_ids_json TEXT NOT NULL DEFAULT '[]',
            reference_ids_json TEXT NOT NULL DEFAULT '[]',
            query_text TEXT
        )
        """.strip(),
        f"""
        CREATE TABLE replay_runs (
            run_id TEXT PRIMARY KEY,
            bundle_id TEXT NOT NULL REFERENCES replay_input_bundles(bundle_id) ON DELETE CASCADE,
            policy_stamp TEXT NOT NULL,
            policy_fingerprint_id TEXT NOT NULL,
            mutation_mode TEXT NOT NULL {_enum_check("mutation_mode", ReplayMutationMode)},
            output_refs_json TEXT NOT NULL DEFAULT '[]',
            metric_scores_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """.strip(),
        f"""
        CREATE TABLE replay_run_strategies (
            run_id TEXT NOT NULL REFERENCES replay_runs(run_id) ON DELETE CASCADE,
            step TEXT NOT NULL {_enum_check("step", ReplayStep)},
            strategy_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            PRIMARY KEY (run_id, step)
        )
        """.strip(),
        f"""
        CREATE TABLE replay_artifacts (
            artifact_id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            source_transaction_kind TEXT NOT NULL {_enum_check("source_transaction_kind", TransactionKind)},
            source_waterline TEXT NOT NULL {_enum_check("source_waterline", DurabilityWaterline)},
            captured_at TEXT NOT NULL,
            baseline_run_id TEXT NOT NULL REFERENCES replay_runs(run_id) ON DELETE CASCADE,
            source_object_ids_json TEXT NOT NULL DEFAULT '[]'
        )
        """.strip(),
        f"""
        CREATE TABLE tier_assignments (
            target_kind TEXT NOT NULL,
            target_id TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            tier TEXT NOT NULL {_enum_check("tier", MemoryTier)},
            rationale TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            PRIMARY KEY (target_kind, target_id, policy_stamp)
        )
        """.strip(),
        f"""
        CREATE TABLE tier_transitions (
            transition_id TEXT PRIMARY KEY,
            target_kind TEXT NOT NULL,
            target_id TEXT NOT NULL,
            policy_stamp TEXT NOT NULL,
            from_tier TEXT NOT NULL {_enum_check("from_tier", MemoryTier)},
            to_tier TEXT NOT NULL {_enum_check("to_tier", MemoryTier)},
            rationale TEXT NOT NULL,
            transitioned_at TEXT NOT NULL
        )
        """.strip(),
        f"""
        CREATE TABLE prefetch_state (
            prefetch_key TEXT PRIMARY KEY,
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            status TEXT NOT NULL {_prefetch_status_check("status")},
            artifact_ids_json TEXT NOT NULL DEFAULT '[]',
            warmed_at TEXT,
            invalidated_at TEXT
        )
        """.strip(),
        "CREATE INDEX idx_subject_aliases_alias ON subject_aliases(normalized_alias)",
        "CREATE INDEX idx_observations_session_time ON observations(session_id, observed_at)",
        "CREATE INDEX idx_candidate_memories_subject ON candidate_memories(subject_id, claim_type)",
        "CREATE INDEX idx_memory_loci_subject_scope ON memory_loci(subject_id, scope)",
        "CREATE INDEX idx_claims_subject_locus ON claims(subject_id, locus_id, learned_at)",
        "CREATE INDEX idx_compiled_views_snapshot_kind ON compiled_views(snapshot_id, kind)",
        "CREATE INDEX idx_system_events_lane ON system_events(arbiter_lane_position)",
        "CREATE INDEX idx_replay_runs_bundle ON replay_runs(bundle_id)",
        "CREATE INDEX idx_resolution_queue_status_priority ON resolution_queue_items(status, priority, created_at)",
        "CREATE INDEX idx_tier_assignments_tier ON tier_assignments(tier)",
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial_full_memory_model",
        statements=_build_initial_schema_statements(),
    ),
    Migration(
        version=2,
        name="add_resolution_queue_session_and_provenance_links",
        statements=(
            """
            ALTER TABLE resolution_queue_items
            ADD COLUMN session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL
            """.strip(),
            """
            ALTER TABLE resolution_queue_items
            ADD COLUMN claim_ids_json TEXT NOT NULL DEFAULT '[]'
            """.strip(),
            """
            ALTER TABLE resolution_queue_items
            ADD COLUMN observation_ids_json TEXT NOT NULL DEFAULT '[]'
            """.strip(),
            """
            ALTER TABLE resolution_queue_items
            ADD COLUMN outcome_ids_json TEXT NOT NULL DEFAULT '[]'
            """.strip(),
            """
            CREATE INDEX idx_resolution_queue_session_priority
            ON resolution_queue_items(session_id, status, priority, created_at)
            """.strip(),
        ),
    ),
    Migration(
        version=3,
        name="add_replay_comparisons",
        statements=(
            f"""
            CREATE TABLE replay_comparisons (
                comparison_id TEXT PRIMARY KEY,
                baseline_run_id TEXT NOT NULL REFERENCES replay_runs(run_id) ON DELETE CASCADE,
                candidate_run_id TEXT NOT NULL REFERENCES replay_runs(run_id) ON DELETE CASCADE,
                mutation_mode TEXT NOT NULL {_enum_check("mutation_mode", ReplayMutationMode)},
                compared_steps_json TEXT NOT NULL DEFAULT '[]',
                rationale TEXT NOT NULL,
                metric_deltas_json TEXT NOT NULL DEFAULT '{{}}',
                notes_json TEXT NOT NULL DEFAULT '[]',
                compared_at TEXT NOT NULL
            )
            """.strip(),
            """
            CREATE INDEX idx_replay_comparisons_baseline_run
            ON replay_comparisons(baseline_run_id)
            """.strip(),
            """
            CREATE INDEX idx_replay_comparisons_candidate_run
            ON replay_comparisons(candidate_run_id)
            """.strip(),
        ),
    ),
)


def _ensure_foreign_keys(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")


def _ensure_migration_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def current_schema_version(connection: sqlite3.Connection) -> int:
    _ensure_foreign_keys(connection)
    _ensure_migration_table(connection)
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    applied_version = int(
        connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
    )
    return max(user_version, applied_version)


def apply_migrations(connection: sqlite3.Connection) -> int:
    _ensure_foreign_keys(connection)
    _ensure_migration_table(connection)
    applied_version = current_schema_version(connection)

    for migration in MIGRATIONS:
        if migration.version <= applied_version:
            continue
        with connection:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name) VALUES(?, ?)",
                (migration.version, migration.name),
            )
            connection.execute(f"PRAGMA user_version = {migration.version}")
        applied_version = migration.version

    return applied_version


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Migration",
    "MIGRATIONS",
    "apply_migrations",
    "current_schema_version",
]
