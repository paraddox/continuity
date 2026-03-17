"""SQLite-backed replay artifact repository for Continuity."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from continuity.disclosure import (
    DisclosureChannel,
    DisclosureContext,
    DisclosurePrincipal,
    DisclosurePurpose,
    DisclosureViewer,
    ViewerKind,
)
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
from continuity.transactions import DurabilityWaterline, TransactionKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    return _validate_timestamp(datetime.fromisoformat(value), field_name=field_name)


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _load_json_list(value: str) -> list[Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError("expected a JSON array payload")
    return loaded


def _load_json_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("expected a JSON object payload")
    return loaded


def _clean_notes(notes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_clean_text(note, field_name="notes") for note in notes))


def _serialize_disclosure_context(context: DisclosureContext) -> dict[str, Any]:
    return {
        "viewer": {
            "viewer_kind": context.viewer.viewer_kind.value,
            "viewer_subject_id": context.viewer.viewer_subject_id,
            "active_user_id": context.viewer.active_user_id,
            "active_peer_id": context.viewer.active_peer_id,
        },
        "audience_principal": context.audience_principal.value,
        "channel": context.channel.value,
        "purpose": context.purpose.value,
        "policy_stamp": context.policy_stamp,
    }


def _disclosure_context_from_payload(payload: dict[str, Any]) -> DisclosureContext:
    viewer_payload = payload["viewer"]
    if not isinstance(viewer_payload, dict):
        raise ValueError("disclosure_context_json viewer payload must be an object")
    return DisclosureContext(
        viewer=DisclosureViewer(
            viewer_kind=ViewerKind(viewer_payload["viewer_kind"]),
            viewer_subject_id=viewer_payload.get("viewer_subject_id"),
            active_user_id=viewer_payload.get("active_user_id"),
            active_peer_id=viewer_payload.get("active_peer_id"),
        ),
        audience_principal=DisclosurePrincipal(payload["audience_principal"]),
        channel=DisclosureChannel(payload["channel"]),
        purpose=DisclosurePurpose(payload["purpose"]),
        policy_stamp=payload["policy_stamp"],
    )


def _metric_payload(metric_scores: dict[ReplayMetric, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for metric, score in metric_scores.items():
        if not isinstance(metric, ReplayMetric):
            raise ValueError("metric score keys must be ReplayMetric values")
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError("metric score values must be integers")
        normalized[metric.value] = score
    return normalized


def _metric_scores_from_payload(payload: dict[str, Any]) -> dict[ReplayMetric, int]:
    return {
        ReplayMetric(metric): int(score)
        for metric, score in payload.items()
    }


@dataclass(frozen=True, slots=True)
class ReplayComparisonRecord:
    comparison: ReplayComparison
    compared_at: datetime
    metric_deltas: dict[ReplayMetric, int]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compared_at",
            _validate_timestamp(self.compared_at, field_name="compared_at"),
        )
        object.__setattr__(self, "notes", _clean_notes(self.notes))
        object.__setattr__(self, "metric_deltas", _metric_scores_from_payload(_metric_payload(self.metric_deltas)))


class ReplayRepository:
    """Persist replay bundles, runs, artifacts, and comparison records."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    def save_input_bundle(self, bundle: ReplayInputBundle) -> None:
        existing = self.read_input_bundle(bundle.bundle_id)
        if existing is not None:
            self._require_immutable(existing=existing, incoming=bundle, entity_name="replay input bundle")
            return

        with self._connection:
            self._connection.execute(
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
                    bundle.bundle_id,
                    bundle.surface,
                    bundle.snapshot_id,
                    bundle.journal_position,
                    bundle.arbiter_lane_position,
                    _dump_json(_serialize_disclosure_context(bundle.disclosure_context)),
                    _dump_json(list(bundle.claim_ids)),
                    _dump_json(list(bundle.observation_ids)),
                    _dump_json(list(bundle.compiled_view_ids)),
                    _dump_json(list(bundle.outcome_ids)),
                    _dump_json(list(bundle.reference_ids)),
                    bundle.query_text,
                ),
            )

    def read_input_bundle(self, bundle_id: str) -> ReplayInputBundle | None:
        row = self._connection.execute(
            """
            SELECT
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
            FROM replay_input_bundles
            WHERE bundle_id = ?
            """,
            (_clean_text(bundle_id, field_name="bundle_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._bundle_from_row(row)

    def list_input_bundles(
        self,
        *,
        snapshot_id: str | None = None,
        surface: str | None = None,
        journal_position: int | None = None,
        limit: int | None = None,
    ) -> tuple[ReplayInputBundle, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if snapshot_id is not None:
            clauses.append("snapshot_id = ?")
            parameters.append(_clean_text(snapshot_id, field_name="snapshot_id"))
        if surface is not None:
            clauses.append("surface = ?")
            parameters.append(_clean_text(surface, field_name="surface"))
        if journal_position is not None:
            clauses.append("journal_position = ?")
            parameters.append(journal_position)
        rows = self._select_rows(
            """
            SELECT
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
            FROM replay_input_bundles
            """,
            clauses=clauses,
            parameters=parameters,
            order_by="journal_position, arbiter_lane_position, bundle_id",
            limit=limit,
        )
        return tuple(self._bundle_from_row(row) for row in rows)

    def save_run(self, run: ReplayRun) -> None:
        self.save_input_bundle(run.input_bundle)

        existing = self.read_run(run.run_id)
        if existing is not None:
            self._require_immutable(existing=existing, incoming=run, entity_name="replay run")
            return

        with self._connection:
            self._connection.execute(
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
                    run.run_id,
                    run.input_bundle.bundle_id,
                    run.policy_fingerprint[0],
                    run.policy_fingerprint[1],
                    run.mutation_mode.value,
                    _dump_json(list(run.output_refs)),
                    _dump_json(_metric_payload(run.metric_scores)),
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO replay_run_strategies(run_id, step, strategy_id, fingerprint)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        run.run_id,
                        strategy.step.value,
                        strategy.strategy_id,
                        strategy.fingerprint,
                    )
                    for strategy in run.strategies
                ),
            )

    def read_run(self, run_id: str) -> ReplayRun | None:
        row = self._connection.execute(
            """
            SELECT
                run_id,
                bundle_id,
                policy_stamp,
                policy_fingerprint_id,
                mutation_mode,
                output_refs_json,
                metric_scores_json
            FROM replay_runs
            WHERE run_id = ?
            """,
            (_clean_text(run_id, field_name="run_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._run_from_row(row)

    def list_runs(
        self,
        *,
        bundle_id: str | None = None,
        policy_stamp: str | None = None,
        limit: int | None = None,
    ) -> tuple[ReplayRun, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if bundle_id is not None:
            clauses.append("bundle_id = ?")
            parameters.append(_clean_text(bundle_id, field_name="bundle_id"))
        if policy_stamp is not None:
            clauses.append("policy_stamp = ?")
            parameters.append(_clean_text(policy_stamp, field_name="policy_stamp"))
        rows = self._select_rows(
            """
            SELECT
                run_id,
                bundle_id,
                policy_stamp,
                policy_fingerprint_id,
                mutation_mode,
                output_refs_json,
                metric_scores_json
            FROM replay_runs
            """,
            clauses=clauses,
            parameters=parameters,
            order_by="run_id",
            limit=limit,
        )
        return tuple(self._run_from_row(row) for row in rows)

    def save_artifact(self, artifact: ReplayArtifact) -> None:
        self.save_run(artifact.baseline_run)

        existing = self.read_artifact(artifact.artifact_id)
        if existing is not None:
            self._require_immutable(existing=existing, incoming=artifact, entity_name="replay artifact")
            return

        with self._connection:
            self._connection.execute(
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
                    artifact.artifact_id,
                    artifact.version,
                    artifact.source_transaction.value,
                    artifact.source_waterline.value,
                    artifact.captured_at.isoformat(),
                    artifact.baseline_run.run_id,
                    _dump_json(list(artifact.source_object_ids)),
                ),
            )

    def read_artifact(self, artifact_id: str) -> ReplayArtifact | None:
        row = self._connection.execute(
            """
            SELECT
                artifact_id,
                version,
                source_transaction_kind,
                source_waterline,
                captured_at,
                baseline_run_id,
                source_object_ids_json
            FROM replay_artifacts
            WHERE artifact_id = ?
            """,
            (_clean_text(artifact_id, field_name="artifact_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._artifact_from_row(row)

    def list_artifacts(
        self,
        *,
        snapshot_id: str | None = None,
        source_transaction: TransactionKind | None = None,
        journal_position: int | None = None,
        limit: int | None = None,
    ) -> tuple[ReplayArtifact, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if snapshot_id is not None:
            clauses.append("bundles.snapshot_id = ?")
            parameters.append(_clean_text(snapshot_id, field_name="snapshot_id"))
        if source_transaction is not None:
            clauses.append("artifacts.source_transaction_kind = ?")
            parameters.append(source_transaction.value)
        if journal_position is not None:
            clauses.append("bundles.journal_position = ?")
            parameters.append(journal_position)
        rows = self._select_rows(
            """
            SELECT artifacts.artifact_id
            FROM replay_artifacts AS artifacts
            JOIN replay_runs AS runs ON runs.run_id = artifacts.baseline_run_id
            JOIN replay_input_bundles AS bundles ON bundles.bundle_id = runs.bundle_id
            """,
            clauses=clauses,
            parameters=parameters,
            order_by="artifacts.captured_at, artifacts.artifact_id",
            limit=limit,
        )
        return tuple(
            self._artifact_from_row(
                self._connection.execute(
                    """
                    SELECT
                        artifact_id,
                        version,
                        source_transaction_kind,
                        source_waterline,
                        captured_at,
                        baseline_run_id,
                        source_object_ids_json
                    FROM replay_artifacts
                    WHERE artifact_id = ?
                    """,
                    (row["artifact_id"],),
                ).fetchone()
            )
            for row in rows
        )

    def record_comparison(self, record: ReplayComparisonRecord) -> None:
        self.save_run(record.comparison.baseline_run)
        self.save_run(record.comparison.candidate_run)

        existing = self.read_comparison(record.comparison.comparison_id)
        if existing is not None:
            self._require_immutable(existing=existing, incoming=record, entity_name="replay comparison")
            return

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO replay_comparisons(
                    comparison_id,
                    baseline_run_id,
                    candidate_run_id,
                    mutation_mode,
                    compared_steps_json,
                    rationale,
                    metric_deltas_json,
                    notes_json,
                    compared_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.comparison.comparison_id,
                    record.comparison.baseline_run.run_id,
                    record.comparison.candidate_run.run_id,
                    record.comparison.mutation_mode.value,
                    _dump_json([step.value for step in record.comparison.compared_steps]),
                    record.comparison.rationale,
                    _dump_json(_metric_payload(record.metric_deltas)),
                    _dump_json(list(record.notes)),
                    record.compared_at.isoformat(),
                ),
            )

    def read_comparison(self, comparison_id: str) -> ReplayComparisonRecord | None:
        row = self._connection.execute(
            """
            SELECT
                comparison_id,
                baseline_run_id,
                candidate_run_id,
                mutation_mode,
                compared_steps_json,
                rationale,
                metric_deltas_json,
                notes_json,
                compared_at
            FROM replay_comparisons
            WHERE comparison_id = ?
            """,
            (_clean_text(comparison_id, field_name="comparison_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._comparison_record_from_row(row)

    def list_comparisons(
        self,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[ReplayComparisonRecord, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if run_id is not None:
            cleaned_run_id = _clean_text(run_id, field_name="run_id")
            clauses.append("(baseline_run_id = ? OR candidate_run_id = ?)")
            parameters.extend((cleaned_run_id, cleaned_run_id))
        rows = self._select_rows(
            """
            SELECT
                comparison_id,
                baseline_run_id,
                candidate_run_id,
                mutation_mode,
                compared_steps_json,
                rationale,
                metric_deltas_json,
                notes_json,
                compared_at
            FROM replay_comparisons
            """,
            clauses=clauses,
            parameters=parameters,
            order_by="compared_at, comparison_id",
            limit=limit,
        )
        return tuple(self._comparison_record_from_row(row) for row in rows)

    def _bundle_from_row(self, row: sqlite3.Row) -> ReplayInputBundle:
        disclosure_payload = _load_json_object(row["disclosure_context_json"])
        return ReplayInputBundle(
            bundle_id=row["bundle_id"],
            surface=row["surface"],
            snapshot_id=row["snapshot_id"],
            journal_position=row["journal_position"],
            arbiter_lane_position=row["arbiter_lane_position"],
            disclosure_context=_disclosure_context_from_payload(disclosure_payload),
            claim_ids=tuple(_load_json_list(row["claim_ids_json"])),
            observation_ids=tuple(_load_json_list(row["observation_ids_json"])),
            compiled_view_ids=tuple(_load_json_list(row["compiled_view_ids_json"])),
            outcome_ids=tuple(_load_json_list(row["outcome_ids_json"])),
            reference_ids=tuple(_load_json_list(row["reference_ids_json"])),
            query_text=row["query_text"],
        )

    def _run_from_row(self, row: sqlite3.Row) -> ReplayRun:
        bundle = self.read_input_bundle(row["bundle_id"])
        if bundle is None:
            raise LookupError(f"missing replay input bundle {row['bundle_id']}")
        strategy_rows = self._connection.execute(
            """
            SELECT step, strategy_id, fingerprint
            FROM replay_run_strategies
            WHERE run_id = ?
            """,
            (row["run_id"],),
        ).fetchall()
        strategies_by_step = {
            ReplayStep(strategy_row["step"]): ReplayStrategy(
                step=ReplayStep(strategy_row["step"]),
                strategy_id=strategy_row["strategy_id"],
                fingerprint=strategy_row["fingerprint"],
            )
            for strategy_row in strategy_rows
        }
        return ReplayRun(
            run_id=row["run_id"],
            input_bundle=bundle,
            policy_fingerprint=(row["policy_stamp"], row["policy_fingerprint_id"]),
            strategies=tuple(strategies_by_step[step] for step in ReplayStep),
            output_refs=tuple(_load_json_list(row["output_refs_json"])),
            metric_scores=_metric_scores_from_payload(_load_json_object(row["metric_scores_json"])),
            mutation_mode=ReplayMutationMode(row["mutation_mode"]),
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ReplayArtifact:
        baseline_run = self.read_run(row["baseline_run_id"])
        if baseline_run is None:
            raise LookupError(f"missing replay run {row['baseline_run_id']}")
        return ReplayArtifact(
            artifact_id=row["artifact_id"],
            version=row["version"],
            source_transaction=TransactionKind(row["source_transaction_kind"]),
            source_waterline=DurabilityWaterline(row["source_waterline"]),
            captured_at=_parse_timestamp(row["captured_at"], field_name="captured_at"),
            baseline_run=baseline_run,
            source_object_ids=tuple(_load_json_list(row["source_object_ids_json"])),
        )

    def _comparison_record_from_row(self, row: sqlite3.Row) -> ReplayComparisonRecord:
        baseline_run = self.read_run(row["baseline_run_id"])
        if baseline_run is None:
            raise LookupError(f"missing replay run {row['baseline_run_id']}")
        candidate_run = self.read_run(row["candidate_run_id"])
        if candidate_run is None:
            raise LookupError(f"missing replay run {row['candidate_run_id']}")
        compared_steps = tuple(
            ReplayStep(step)
            for step in _load_json_list(row["compared_steps_json"])
        )
        comparison = ReplayComparison(
            comparison_id=row["comparison_id"],
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            compared_steps=compared_steps,
            rationale=row["rationale"],
            mutation_mode=ReplayMutationMode(row["mutation_mode"]),
        )
        return ReplayComparisonRecord(
            comparison=comparison,
            compared_at=_parse_timestamp(row["compared_at"], field_name="compared_at"),
            metric_deltas=_metric_scores_from_payload(_load_json_object(row["metric_deltas_json"])),
            notes=tuple(_load_json_list(row["notes_json"])),
        )

    def _select_rows(
        self,
        base_query: str,
        *,
        clauses: list[str],
        parameters: list[Any],
        order_by: str,
        limit: int | None,
    ) -> tuple[sqlite3.Row, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = ""
        final_parameters = list(parameters)
        if limit is not None:
            limit_sql = " LIMIT ?"
            final_parameters.append(limit)
        rows = self._connection.execute(
            f"{base_query}{where} ORDER BY {order_by}{limit_sql}",
            tuple(final_parameters),
        ).fetchall()
        return tuple(rows)

    def _require_immutable(self, *, existing: object, incoming: object, entity_name: str) -> None:
        if existing != incoming:
            raise ValueError(f"{entity_name} is immutable once captured")


__all__ = [
    "ReplayComparisonRecord",
    "ReplayRepository",
]
