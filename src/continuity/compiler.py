"""Incremental compiler dependency and invalidation model for Continuity."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Iterable

from continuity.arbiter import ArbiterPublication, ArbiterPublicationKind, MutationArbiter
from continuity.events import EventPayloadMode, SystemEvent
from continuity.snapshots import (
    MemorySnapshot,
    SnapshotArtifactKind,
    SnapshotArtifactRef,
    SnapshotHead,
    SnapshotHeadState,
    SnapshotRepository,
)
from continuity.transactions import DurabilityWaterline, TransactionKind, TransactionPhase


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


class CompilerNodeCategory(StrEnum):
    SOURCE_INPUT = "source_input"
    DERIVED_IR = "derived_ir"
    UTILITY_STATE = "utility_state"
    COMPILED_ARTIFACT = "compiled_artifact"


class SourceInputKind(StrEnum):
    OBSERVATION = "observation"
    IMPORTED_ARTIFACT = "imported_artifact"
    ADMISSION_POLICY = "admission_policy"
    ADMISSION_DECISION = "admission_decision"
    SUBJECT_BINDING = "subject_binding"
    FORGETTING_OPERATION = "forgetting_operation"
    RESOLUTION_ACTION = "resolution_action"
    UTILITY_EVENT = "utility_event"
    POLICY_PACK = "policy_pack"
    REASONING_ADAPTER = "reasoning_adapter"
    PROMPT_RULE = "prompt_rule"


class DerivedArtifactKind(StrEnum):
    SUBJECT = "subject"
    CLAIM = "claim"
    LOCUS = "locus"
    CLAIM_RELATION = "claim_relation"


class UtilityStateKind(StrEnum):
    COMPILED_WEIGHT = "compiled_weight"


class CompiledArtifactKind(StrEnum):
    STATE_VIEW = "state_view"
    TIMELINE_VIEW = "timeline_view"
    SET_VIEW = "set_view"
    PROFILE_VIEW = "profile_view"
    PROMPT_VIEW = "prompt_view"
    EVIDENCE_VIEW = "evidence_view"
    ANSWER_VIEW = "answer_view"
    VECTOR_INDEX_RECORD = "vector_index_record"


class DependencyRole(StrEnum):
    CONTENT = "content"
    PROJECTION = "projection"
    MEMBERSHIP = "membership"
    POLICY = "policy"
    UTILITY = "utility"
    PROVENANCE = "provenance"
    INDEX = "index"


class DirtyReason(StrEnum):
    SOURCE_EDITED = "source_edited"
    ADMISSION_POLICY_CHANGED = "admission_policy_changed"
    CLAIM_CORRECTED = "claim_corrected"
    SUBJECT_IDENTITY_CHANGED = "subject_identity_changed"
    LOCUS_MEMBERSHIP_CHANGED = "locus_membership_changed"
    FORGETTING_CHANGED = "forgetting_changed"
    RESOLUTION_CHANGED = "resolution_changed"
    UTILITY_INPUT_CHANGED = "utility_input_changed"
    POLICY_UPGRADED = "policy_upgraded"
    ADAPTER_CHANGED = "adapter_changed"


_CATEGORY_KINDS: dict[CompilerNodeCategory, tuple[type[StrEnum], ...]] = {
    CompilerNodeCategory.SOURCE_INPUT: (SourceInputKind,),
    CompilerNodeCategory.DERIVED_IR: (DerivedArtifactKind,),
    CompilerNodeCategory.UTILITY_STATE: (UtilityStateKind,),
    CompilerNodeCategory.COMPILED_ARTIFACT: (CompiledArtifactKind,),
}
_CATEGORY_KIND_ENUMS: dict[CompilerNodeCategory, type[StrEnum]] = {
    CompilerNodeCategory.SOURCE_INPUT: SourceInputKind,
    CompilerNodeCategory.DERIVED_IR: DerivedArtifactKind,
    CompilerNodeCategory.UTILITY_STATE: UtilityStateKind,
    CompilerNodeCategory.COMPILED_ARTIFACT: CompiledArtifactKind,
}


def _node_sort_key(node: "CompilerNode") -> tuple[str, str, str, str, str]:
    return (
        node.category.value,
        node.subject_id or "",
        node.locus_key or "",
        node.kind.value,
        node.node_id,
    )


def _dirty_node_sort_key(dirty_node: "DirtyNode") -> tuple[str, str, str, str, str]:
    return (
        dirty_node.category.value,
        dirty_node.subject_id or "",
        dirty_node.locus_key or "",
        dirty_node.kind.value,
        dirty_node.node_id,
    )


def _dirty_reason_for(node: "CompilerNode") -> DirtyReason:
    if node.category is CompilerNodeCategory.SOURCE_INPUT:
        if node.kind in {SourceInputKind.OBSERVATION, SourceInputKind.IMPORTED_ARTIFACT}:  # type: ignore[arg-type]
            return DirtyReason.SOURCE_EDITED
        if node.kind in {SourceInputKind.ADMISSION_POLICY, SourceInputKind.ADMISSION_DECISION}:  # type: ignore[arg-type]
            return DirtyReason.ADMISSION_POLICY_CHANGED
        if node.kind is SourceInputKind.SUBJECT_BINDING:  # type: ignore[comparison-overlap]
            return DirtyReason.SUBJECT_IDENTITY_CHANGED
        if node.kind is SourceInputKind.FORGETTING_OPERATION:  # type: ignore[comparison-overlap]
            return DirtyReason.FORGETTING_CHANGED
        if node.kind is SourceInputKind.RESOLUTION_ACTION:  # type: ignore[comparison-overlap]
            return DirtyReason.RESOLUTION_CHANGED
        if node.kind is SourceInputKind.UTILITY_EVENT:  # type: ignore[comparison-overlap]
            return DirtyReason.UTILITY_INPUT_CHANGED
        if node.kind in {SourceInputKind.POLICY_PACK, SourceInputKind.PROMPT_RULE}:  # type: ignore[arg-type]
            return DirtyReason.POLICY_UPGRADED
        if node.kind is SourceInputKind.REASONING_ADAPTER:  # type: ignore[comparison-overlap]
            return DirtyReason.ADAPTER_CHANGED
    if node.category is CompilerNodeCategory.DERIVED_IR:
        if node.kind is DerivedArtifactKind.CLAIM:  # type: ignore[comparison-overlap]
            return DirtyReason.CLAIM_CORRECTED
        if node.kind is DerivedArtifactKind.SUBJECT:  # type: ignore[comparison-overlap]
            return DirtyReason.SUBJECT_IDENTITY_CHANGED
        if node.kind is DerivedArtifactKind.LOCUS:  # type: ignore[comparison-overlap]
            return DirtyReason.LOCUS_MEMBERSHIP_CHANGED
        return DirtyReason.CLAIM_CORRECTED
    if node.category is CompilerNodeCategory.UTILITY_STATE:
        return DirtyReason.UTILITY_INPUT_CHANGED
    if node.kind in {CompiledArtifactKind.PROMPT_VIEW, CompiledArtifactKind.ANSWER_VIEW}:  # type: ignore[arg-type]
        return DirtyReason.POLICY_UPGRADED
    return DirtyReason.CLAIM_CORRECTED


@dataclass(frozen=True, slots=True)
class CompilerNode:
    node_id: str
    category: CompilerNodeCategory
    kind: StrEnum
    fingerprint: str
    subject_id: str | None = None
    locus_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_text(self.node_id, field_name="node_id"))
        object.__setattr__(self, "fingerprint", _clean_text(self.fingerprint, field_name="fingerprint"))
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))

        allowed_kinds = _CATEGORY_KINDS[self.category]
        if not isinstance(self.kind, allowed_kinds):
            raise ValueError(f"{self.category.value} nodes must use the correct kind enum")


@dataclass(frozen=True, slots=True)
class CompilerDependency:
    upstream_node_id: str
    downstream_node_id: str
    role: DependencyRole

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "upstream_node_id",
            _clean_text(self.upstream_node_id, field_name="upstream_node_id"),
        )
        object.__setattr__(
            self,
            "downstream_node_id",
            _clean_text(self.downstream_node_id, field_name="downstream_node_id"),
        )
        if self.upstream_node_id == self.downstream_node_id:
            raise ValueError("compiler dependencies must connect distinct nodes")


@dataclass(frozen=True, slots=True)
class CompilerChange:
    node_id: str
    category: CompilerNodeCategory
    kind: StrEnum
    reason: DirtyReason
    previous_fingerprint: str | None
    current_fingerprint: str | None
    subject_id: str | None = None
    locus_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_text(self.node_id, field_name="node_id"))
        object.__setattr__(
            self,
            "previous_fingerprint",
            _clean_optional_text(self.previous_fingerprint, field_name="previous_fingerprint"),
        )
        object.__setattr__(
            self,
            "current_fingerprint",
            _clean_optional_text(self.current_fingerprint, field_name="current_fingerprint"),
        )
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))
        if self.previous_fingerprint == self.current_fingerprint:
            raise ValueError("compiler changes require a fingerprint delta")


@dataclass(frozen=True, slots=True)
class DirtyCause:
    reason: DirtyReason
    changed_node_id: str
    changed_node_kind: str
    dependency_path: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "changed_node_id",
            _clean_text(self.changed_node_id, field_name="changed_node_id"),
        )
        object.__setattr__(
            self,
            "changed_node_kind",
            _clean_text(self.changed_node_kind, field_name="changed_node_kind"),
        )
        object.__setattr__(
            self,
            "dependency_path",
            tuple(_clean_text(node_id, field_name="dependency_path") for node_id in self.dependency_path),
        )
        if not self.dependency_path:
            raise ValueError("dirty causes require a dependency path")


@dataclass(frozen=True, slots=True)
class DirtyNode:
    node_id: str
    category: CompilerNodeCategory
    kind: StrEnum
    subject_id: str | None
    locus_key: str | None
    causes: tuple[DirtyCause, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_text(self.node_id, field_name="node_id"))
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))
        sorted_causes = tuple(
            sorted(
                set(self.causes),
                key=lambda cause: (cause.reason.value, cause.changed_node_id, cause.dependency_path),
            )
        )
        object.__setattr__(self, "causes", sorted_causes)
        if not self.causes:
            raise ValueError("dirty nodes require at least one dirty cause")

    @property
    def reasons(self) -> tuple[DirtyReason, ...]:
        return tuple(dict.fromkeys(cause.reason for cause in self.causes))

    @property
    def cause_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(dict.fromkeys(cause.dependency_path for cause in self.causes))


class DirtyQueueStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    changes: tuple[CompilerChange, ...]
    dirty_nodes: tuple[DirtyNode, ...]
    rebuild_order: tuple[str, ...]
    affected_subject_ids: tuple[str, ...]
    affected_locus_keys: tuple[str, ...]

    def dirty_node(self, node_id: str) -> DirtyNode:
        cleaned_node_id = _clean_text(node_id, field_name="node_id")
        for dirty_node in self.dirty_nodes:
            if dirty_node.node_id == cleaned_node_id:
                return dirty_node
        raise KeyError(cleaned_node_id)


@dataclass(frozen=True, slots=True)
class RebuildArtifact:
    source_node_id: str
    artifact_ref: SnapshotArtifactRef
    supersedes_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_node_id",
            _clean_text(self.source_node_id, field_name="source_node_id"),
        )
        object.__setattr__(
            self,
            "supersedes_artifact_ids",
            tuple(
                dict.fromkeys(
                    _clean_text(artifact_id, field_name="supersedes_artifact_ids")
                    for artifact_id in self.supersedes_artifact_ids
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class StagedRebuild:
    plan: RebuildPlan
    active_snapshot_id: str
    candidate_snapshot: MemorySnapshot
    publication: ArbiterPublication
    event: SystemEvent

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "active_snapshot_id",
            _clean_text(self.active_snapshot_id, field_name="active_snapshot_id"),
        )
        if self.candidate_snapshot.parent_snapshot_id != self.active_snapshot_id:
            raise ValueError("candidate_snapshot must be based on the active snapshot")
        if self.publication.publication_kind is not ArbiterPublicationKind.VIEW_PUBLICATION:
            raise ValueError("staged rebuilds must publish compiled-view work")


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _kind_from_storage(*, category: CompilerNodeCategory, raw_kind: str) -> StrEnum:
    enum_type = _CATEGORY_KIND_ENUMS[category]
    return enum_type(raw_kind)


def _json_dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


class CompilerStateRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def replace_nodes(self, nodes: Iterable[CompilerNode]) -> None:
        rows = tuple(nodes)
        _index_nodes(rows)

        with self._connection:
            if rows:
                placeholders = ", ".join("?" for _ in rows)
                self._connection.execute(
                    f"DELETE FROM compiler_nodes WHERE node_id NOT IN ({placeholders})",
                    tuple(node.node_id for node in rows),
                )
            else:
                self._connection.execute("DELETE FROM compiler_nodes")

            self.upsert_nodes(rows)

    def upsert_nodes(self, nodes: Iterable[CompilerNode]) -> None:
        rows = tuple(nodes)
        if not rows:
            return

        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO compiler_nodes(node_id, category, kind, fingerprint, subject_id, locus_key)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    category = excluded.category,
                    kind = excluded.kind,
                    fingerprint = excluded.fingerprint,
                    subject_id = excluded.subject_id,
                    locus_key = excluded.locus_key
                """,
                (
                    (
                        node.node_id,
                        node.category.value,
                        node.kind.value,
                        node.fingerprint,
                        node.subject_id,
                        node.locus_key,
                    )
                    for node in rows
                ),
            )

    def list_nodes(
        self,
        *,
        subject_id: str | None = None,
        locus_key: str | None = None,
    ) -> tuple[CompilerNode, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if subject_id is not None:
            clauses.append("subject_id = ?")
            parameters.append(_clean_text(subject_id, field_name="subject_id"))
        if locus_key is not None:
            clauses.append("locus_key = ?")
            parameters.append(_clean_text(locus_key, field_name="locus_key"))

        sql = """
            SELECT node_id, category, kind, fingerprint, subject_id, locus_key
            FROM compiler_nodes
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"

        rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(
            CompilerNode(
                node_id=row[0],
                category=CompilerNodeCategory(row[1]),
                kind=_kind_from_storage(
                    category=CompilerNodeCategory(row[1]),
                    raw_kind=row[2],
                ),
                fingerprint=row[3],
                subject_id=row[4],
                locus_key=row[5],
            )
            for row in rows
        )

    def replace_dependencies(self, dependencies: Iterable[CompilerDependency]) -> None:
        rows = tuple(dependencies)
        with self._connection:
            self._connection.execute("DELETE FROM compiler_dependencies")
            if rows:
                self._connection.executemany(
                    """
                    INSERT INTO compiler_dependencies(upstream_node_id, downstream_node_id, role)
                    VALUES (?, ?, ?)
                    """,
                    (
                        (
                            dependency.upstream_node_id,
                            dependency.downstream_node_id,
                            dependency.role.value,
                        )
                        for dependency in rows
                    ),
                )

    def list_dependencies(
        self,
        *,
        upstream_node_id: str | None = None,
        downstream_node_id: str | None = None,
        role: DependencyRole | None = None,
    ) -> tuple[CompilerDependency, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if upstream_node_id is not None:
            clauses.append("upstream_node_id = ?")
            parameters.append(_clean_text(upstream_node_id, field_name="upstream_node_id"))
        if downstream_node_id is not None:
            clauses.append("downstream_node_id = ?")
            parameters.append(_clean_text(downstream_node_id, field_name="downstream_node_id"))
        if role is not None:
            clauses.append("role = ?")
            parameters.append(role.value)

        sql = """
            SELECT upstream_node_id, downstream_node_id, role
            FROM compiler_dependencies
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"

        rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(
            CompilerDependency(
                upstream_node_id=row[0],
                downstream_node_id=row[1],
                role=DependencyRole(row[2]),
            )
            for row in rows
        )

    def enqueue_dirty_nodes(
        self,
        dirty_nodes: Iterable[DirtyNode],
        *,
        queued_at: datetime,
        status: DirtyQueueStatus = DirtyQueueStatus.PENDING,
    ) -> None:
        queue_time = _validate_timestamp(queued_at, field_name="queued_at").isoformat()
        rows = tuple(dirty_nodes)
        if not rows:
            return

        with self._connection:
            self._connection.executemany(
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
                    (
                        dirty_node.node_id,
                        dirty_node.reasons[0].value,
                        dirty_node.subject_id,
                        dirty_node.locus_key,
                        _json_dumps(
                            [
                                {
                                    "reason": cause.reason.value,
                                    "changed_node_id": cause.changed_node_id,
                                    "changed_node_kind": cause.changed_node_kind,
                                    "dependency_path": list(cause.dependency_path),
                                }
                                for cause in dirty_node.causes
                            ]
                        ),
                        status.value,
                        queue_time,
                    )
                    for dirty_node in rows
                ),
            )

    def clear_dirty_nodes(
        self,
        *,
        statuses: Iterable[DirtyQueueStatus] = (
            DirtyQueueStatus.PENDING,
            DirtyQueueStatus.RUNNING,
        ),
    ) -> None:
        cleaned_statuses = tuple(dict.fromkeys(statuses))
        if not cleaned_statuses:
            return

        placeholders = ", ".join("?" for _ in cleaned_statuses)
        with self._connection:
            self._connection.execute(
                f"DELETE FROM compiler_dirty_queue WHERE status IN ({placeholders})",
                tuple(status.value for status in cleaned_statuses),
            )

    def set_dirty_node_status(
        self,
        node_ids: Iterable[str],
        *,
        status: DirtyQueueStatus,
        current_status: DirtyQueueStatus | None = None,
    ) -> None:
        cleaned_node_ids = tuple(
            dict.fromkeys(_clean_text(node_id, field_name="node_id") for node_id in node_ids)
        )
        if not cleaned_node_ids:
            return

        placeholders = ", ".join("?" for _ in cleaned_node_ids)
        sql = f"UPDATE compiler_dirty_queue SET status = ? WHERE node_id IN ({placeholders})"
        parameters: list[str] = [status.value, *cleaned_node_ids]
        if current_status is not None:
            sql += " AND status = ?"
            parameters.append(current_status.value)

        with self._connection:
            self._connection.execute(sql, parameters)

    def list_dirty_nodes(
        self,
        *,
        status: DirtyQueueStatus = DirtyQueueStatus.PENDING,
        subject_id: str | None = None,
        locus_key: str | None = None,
    ) -> tuple[DirtyNode, ...]:
        clauses = ["queue.status = ?"]
        parameters: list[str] = [status.value]
        if subject_id is not None:
            clauses.append("queue.subject_id = ?")
            parameters.append(_clean_text(subject_id, field_name="subject_id"))
        if locus_key is not None:
            clauses.append("queue.locus_key = ?")
            parameters.append(_clean_text(locus_key, field_name="locus_key"))

        sql = f"""
            SELECT
                queue.queue_id,
                queue.node_id,
                queue.reason,
                queue.subject_id,
                queue.locus_key,
                queue.causes_json,
                node.category,
                node.kind
            FROM compiler_dirty_queue AS queue
            JOIN compiler_nodes AS node ON node.node_id = queue.node_id
            WHERE {" AND ".join(clauses)}
            ORDER BY queue.queue_id
        """
        rows = self._connection.execute(sql, parameters).fetchall()

        node_state: dict[str, tuple[CompilerNodeCategory, StrEnum, str | None, str | None]] = {}
        causes_by_node_id: dict[str, set[DirtyCause]] = defaultdict(set)
        ordered_node_ids: list[str] = []

        for row in rows:
            node_id = row[1]
            if node_id not in node_state:
                ordered_node_ids.append(node_id)
                category = CompilerNodeCategory(row[6])
                node_state[node_id] = (
                    category,
                    _kind_from_storage(category=category, raw_kind=row[7]),
                    row[3],
                    row[4],
                )

            causes_by_node_id[node_id].update(
                self._load_dirty_causes(raw_json=row[5], fallback_reason=DirtyReason(row[2]))
            )

        return tuple(
            DirtyNode(
                node_id=node_id,
                category=node_state[node_id][0],
                kind=node_state[node_id][1],
                subject_id=node_state[node_id][2],
                locus_key=node_state[node_id][3],
                causes=tuple(causes_by_node_id[node_id]),
            )
            for node_id in ordered_node_ids
        )

    def list_affected_subject_ids(
        self,
        *,
        status: DirtyQueueStatus = DirtyQueueStatus.PENDING,
    ) -> tuple[str, ...]:
        rows = self._connection.execute(
            """
            SELECT DISTINCT subject_id
            FROM compiler_dirty_queue
            WHERE status = ? AND subject_id IS NOT NULL
            ORDER BY subject_id
            """,
            (status.value,),
        ).fetchall()
        return tuple(row[0] for row in rows)

    def list_affected_locus_keys(
        self,
        *,
        subject_id: str,
        status: DirtyQueueStatus = DirtyQueueStatus.PENDING,
    ) -> tuple[str, ...]:
        rows = self._connection.execute(
            """
            SELECT DISTINCT locus_key
            FROM compiler_dirty_queue
            WHERE status = ? AND subject_id = ? AND locus_key IS NOT NULL
            ORDER BY locus_key
            """,
            (status.value, _clean_text(subject_id, field_name="subject_id")),
        ).fetchall()
        return tuple(row[0] for row in rows)

    def read_rebuild_plan(
        self,
        *,
        status: DirtyQueueStatus = DirtyQueueStatus.PENDING,
        subject_id: str | None = None,
        locus_key: str | None = None,
    ) -> RebuildPlan:
        dirty_nodes = self.list_dirty_nodes(status=status, subject_id=subject_id, locus_key=locus_key)
        rebuild_order = _deterministic_topological_order(
            dirty_nodes=dirty_nodes,
            dependencies=self.list_dependencies(),
        )
        return RebuildPlan(
            changes=(),
            dirty_nodes=dirty_nodes,
            rebuild_order=rebuild_order,
            affected_subject_ids=tuple(
                sorted(
                    {
                        dirty_node.subject_id
                        for dirty_node in dirty_nodes
                        if dirty_node.subject_id is not None
                    }
                )
            ),
            affected_locus_keys=tuple(
                sorted(
                    {
                        dirty_node.locus_key
                        for dirty_node in dirty_nodes
                        if dirty_node.locus_key is not None
                    }
                )
            ),
        )

    def _load_dirty_causes(
        self,
        *,
        raw_json: str,
        fallback_reason: DirtyReason,
    ) -> tuple[DirtyCause, ...]:
        payload = json.loads(raw_json)
        causes: list[DirtyCause] = []
        for item in payload:
            changed_node_id = _clean_text(item["changed_node_id"], field_name="changed_node_id")
            dependency_path = tuple(
                _clean_text(node_id, field_name="dependency_path")
                for node_id in item["dependency_path"]
            )
            changed_node_kind = item.get("changed_node_kind") or self._node_kind_value(changed_node_id)
            reason_value = item.get("reason", fallback_reason.value)
            causes.append(
                DirtyCause(
                    reason=DirtyReason(reason_value),
                    changed_node_id=changed_node_id,
                    changed_node_kind=_clean_text(
                        changed_node_kind,
                        field_name="changed_node_kind",
                    ),
                    dependency_path=dependency_path,
                )
            )
        return tuple(causes)

    def _node_kind_value(self, node_id: str) -> str:
        row = self._connection.execute(
            """
            SELECT kind
            FROM compiler_nodes
            WHERE node_id = ?
            """,
            (_clean_text(node_id, field_name="node_id"),),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return row[0]


def _snapshot_artifact_kind_for_compiled_kind(
    artifact_kind: CompiledArtifactKind,
) -> SnapshotArtifactKind:
    return {
        CompiledArtifactKind.STATE_VIEW: SnapshotArtifactKind.STATE_VIEW,
        CompiledArtifactKind.TIMELINE_VIEW: SnapshotArtifactKind.TIMELINE_VIEW,
        CompiledArtifactKind.SET_VIEW: SnapshotArtifactKind.SET_VIEW,
        CompiledArtifactKind.PROFILE_VIEW: SnapshotArtifactKind.PROFILE_VIEW,
        CompiledArtifactKind.PROMPT_VIEW: SnapshotArtifactKind.PROMPT_VIEW,
        CompiledArtifactKind.EVIDENCE_VIEW: SnapshotArtifactKind.EVIDENCE_VIEW,
        CompiledArtifactKind.ANSWER_VIEW: SnapshotArtifactKind.ANSWER_VIEW,
        CompiledArtifactKind.VECTOR_INDEX_RECORD: SnapshotArtifactKind.VECTOR_INDEX,
    }[artifact_kind]


def _index_nodes(nodes: tuple[CompilerNode, ...] | list[CompilerNode]) -> dict[str, CompilerNode]:
    indexed: dict[str, CompilerNode] = {}
    for node in nodes:
        if node.node_id in indexed:
            raise ValueError(f"duplicate compiler node id: {node.node_id}")
        indexed[node.node_id] = node
    return indexed


class IncrementalRebuildPlanner:
    """Compute deterministic rebuild plans and stage compiled outputs off the active head."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        arbiter: MutationArbiter | None = None,
        snapshot_head_key: str = "current",
    ) -> None:
        self._connection = connection
        self._repository = CompilerStateRepository(connection)
        self._snapshots = SnapshotRepository(connection)
        self._arbiter = MutationArbiter(connection) if arbiter is None else arbiter
        self._snapshot_head_key = _clean_text(snapshot_head_key, field_name="snapshot_head_key")

    def plan_rebuild(
        self,
        *,
        nodes: tuple[CompilerNode, ...] | list[CompilerNode],
        dependencies: tuple[CompilerDependency, ...] | list[CompilerDependency],
        queued_at: datetime,
    ) -> RebuildPlan:
        current_nodes = tuple(nodes)
        current_dependencies = tuple(dependencies)
        previous_nodes = self._repository.list_nodes()
        changes = tuple(
            change
            for change in detect_fingerprint_changes(
                previous_nodes=previous_nodes,
                current_nodes=current_nodes,
            )
            if change.current_fingerprint is not None
        )
        plan = plan_incremental_rebuild(
            nodes=current_nodes,
            dependencies=current_dependencies,
            changes=changes,
        )

        self._repository.replace_nodes(current_nodes)
        self._repository.replace_dependencies(current_dependencies)
        self._repository.clear_dirty_nodes()
        self._repository.enqueue_dirty_nodes(plan.dirty_nodes, queued_at=queued_at)
        return plan

    def stage_rebuild(
        self,
        *,
        plan: RebuildPlan,
        candidate_snapshot_id: str,
        policy_stamp: str,
        published_at: datetime,
        rebuilt_artifacts: tuple[RebuildArtifact, ...] | list[RebuildArtifact],
    ) -> StagedRebuild:
        if not plan.dirty_nodes:
            raise ValueError("cannot stage a rebuild without dirty nodes")

        dirty_node_map = {
            dirty_node.node_id: dirty_node
            for dirty_node in plan.dirty_nodes
        }
        staged_artifacts = tuple(rebuilt_artifacts)
        if not staged_artifacts:
            raise ValueError("rebuilt_artifacts must be non-empty")

        for staged_artifact in staged_artifacts:
            dirty_node = dirty_node_map.get(staged_artifact.source_node_id)
            if dirty_node is None:
                raise ValueError(
                    f"{staged_artifact.source_node_id} is not part of the rebuild plan"
                )
            if dirty_node.category is not CompilerNodeCategory.COMPILED_ARTIFACT:
                raise ValueError("rebuilt artifacts must point at compiled_artifact nodes")
            if not isinstance(dirty_node.kind, CompiledArtifactKind):
                raise ValueError("compiled rebuild nodes must use CompiledArtifactKind")
            expected_kind = _snapshot_artifact_kind_for_compiled_kind(dirty_node.kind)
            if staged_artifact.artifact_ref.artifact_kind is not expected_kind:
                raise ValueError(
                    f"{staged_artifact.source_node_id} must stage a {expected_kind.value} artifact"
                )

        active_snapshot = self._snapshots.read_active_snapshot(head_key=self._snapshot_head_key)
        if active_snapshot is None:
            raise ValueError("staging a rebuild requires an active snapshot")

        superseded_ids = {
            artifact_id
            for staged_artifact in staged_artifacts
            for artifact_id in staged_artifact.supersedes_artifact_ids
        }
        candidate_artifact_refs = [
            artifact_ref
            for artifact_ref in active_snapshot.artifact_refs
            if artifact_ref.artifact_id not in superseded_ids
        ]
        candidate_artifact_refs.extend(
            staged_artifact.artifact_ref
            for staged_artifact in staged_artifacts
        )

        candidate_snapshot = MemorySnapshot(
            snapshot_id=_clean_text(
                candidate_snapshot_id,
                field_name="candidate_snapshot_id",
            ),
            policy_stamp=_clean_text(policy_stamp, field_name="policy_stamp"),
            parent_snapshot_id=active_snapshot.snapshot_id,
            created_by_transaction=TransactionKind.COMPILE_VIEWS,
            artifact_refs=tuple(dict.fromkeys(candidate_artifact_refs)),
        )
        self._snapshots.save_snapshot(candidate_snapshot)
        self._snapshots.upsert_head(
            SnapshotHead(
                head_key=self._snapshot_head_key,
                state=SnapshotHeadState.CANDIDATE,
                snapshot_id=candidate_snapshot.snapshot_id,
                based_on_snapshot_id=active_snapshot.snapshot_id,
            )
        )

        published_mutation = self._arbiter.publish(
            publication_kind=ArbiterPublicationKind.VIEW_PUBLICATION,
            transaction_kind=TransactionKind.COMPILE_VIEWS,
            phase=TransactionPhase.COMPILE_VIEWS,
            object_ids=tuple(
                dict.fromkeys(
                    staged_artifact.source_node_id
                    for staged_artifact in staged_artifacts
                )
            ),
            published_at=_validate_timestamp(
                published_at,
                field_name="published_at",
            ),
            payload_mode=EventPayloadMode.REFERENCE,
            reference_ids=(
                candidate_snapshot.snapshot_id,
                *tuple(
                    dict.fromkeys(
                        staged_artifact.artifact_ref.artifact_id
                        for staged_artifact in staged_artifacts
                    )
                ),
            ),
            reached_waterline=DurabilityWaterline.VIEWS_COMPILED,
        )
        self._repository.set_dirty_node_status(
            (dirty_node.node_id for dirty_node in plan.dirty_nodes),
            status=DirtyQueueStatus.DONE,
            current_status=DirtyQueueStatus.PENDING,
        )
        return StagedRebuild(
            plan=plan,
            active_snapshot_id=active_snapshot.snapshot_id,
            candidate_snapshot=candidate_snapshot,
            publication=published_mutation.publication,
            event=published_mutation.event,
        )


def detect_fingerprint_changes(
    *,
    previous_nodes: tuple[CompilerNode, ...] | list[CompilerNode],
    current_nodes: tuple[CompilerNode, ...] | list[CompilerNode],
) -> tuple[CompilerChange, ...]:
    previous_by_id = _index_nodes(previous_nodes)
    current_by_id = _index_nodes(current_nodes)
    changes: list[CompilerChange] = []

    for node_id in sorted(set(previous_by_id) | set(current_by_id)):
        previous = previous_by_id.get(node_id)
        current = current_by_id.get(node_id)
        reference = current or previous
        if reference is None:
            continue
        if previous and current and (
            previous.category is not current.category or previous.kind != current.kind
        ):
            raise ValueError("compiler node ids must not change category or kind across fingerprints")

        previous_fingerprint = previous.fingerprint if previous else None
        current_fingerprint = current.fingerprint if current else None
        if previous_fingerprint == current_fingerprint:
            continue

        changes.append(
            CompilerChange(
                node_id=node_id,
                category=reference.category,
                kind=reference.kind,
                reason=_dirty_reason_for(reference),
                previous_fingerprint=previous_fingerprint,
                current_fingerprint=current_fingerprint,
                subject_id=reference.subject_id,
                locus_key=reference.locus_key,
            )
        )

    return tuple(changes)


def plan_incremental_rebuild(
    *,
    nodes: tuple[CompilerNode, ...] | list[CompilerNode],
    dependencies: tuple[CompilerDependency, ...] | list[CompilerDependency],
    changes: tuple[CompilerChange, ...] | list[CompilerChange],
) -> RebuildPlan:
    node_by_id = _index_nodes(nodes)
    downstream_by_id: dict[str, list[str]] = defaultdict(list)
    dirty_causes_by_id: dict[str, set[DirtyCause]] = defaultdict(set)

    for dependency in dependencies:
        if dependency.upstream_node_id not in node_by_id:
            raise ValueError(f"unknown upstream node: {dependency.upstream_node_id}")
        if dependency.downstream_node_id not in node_by_id:
            raise ValueError(f"unknown downstream node: {dependency.downstream_node_id}")
        downstream_by_id[dependency.upstream_node_id].append(dependency.downstream_node_id)

    for downstream_ids in downstream_by_id.values():
        downstream_ids.sort(
            key=lambda node_id: _node_sort_key(node_by_id[node_id]),
        )

    queue: list[tuple[str, DirtyReason, tuple[str, ...], str]] = []
    for change in changes:
        if change.node_id not in node_by_id:
            raise ValueError(f"unknown changed node: {change.node_id}")

        changed_node = node_by_id[change.node_id]
        path = (change.node_id,)
        queue.append((change.node_id, change.reason, path, changed_node.kind.value))

        if changed_node.category not in {
            CompilerNodeCategory.SOURCE_INPUT,
            CompilerNodeCategory.UTILITY_STATE,
        }:
            dirty_causes_by_id[change.node_id].add(
                DirtyCause(
                    reason=change.reason,
                    changed_node_id=change.node_id,
                    changed_node_kind=changed_node.kind.value,
                    dependency_path=path,
                )
            )

    while queue:
        current_node_id, reason, path, changed_kind = queue.pop(0)
        for downstream_node_id in downstream_by_id.get(current_node_id, ()):
            if downstream_node_id in path:
                raise ValueError("compiler dependency graph must be acyclic")

            next_path = (*path, downstream_node_id)
            downstream_node = node_by_id[downstream_node_id]
            dirty_causes_by_id[downstream_node_id].add(
                DirtyCause(
                    reason=reason,
                    changed_node_id=path[0],
                    changed_node_kind=changed_kind,
                    dependency_path=next_path,
                )
            )
            queue.append((downstream_node_id, reason, next_path, changed_kind))

    dirty_nodes = tuple(
        DirtyNode(
            node_id=node.node_id,
            category=node.category,
            kind=node.kind,
            subject_id=node.subject_id,
            locus_key=node.locus_key,
            causes=tuple(dirty_causes_by_id[node.node_id]),
        )
        for node in sorted(
            (node_by_id[node_id] for node_id in dirty_causes_by_id),
            key=_node_sort_key,
        )
    )

    rebuild_order = _deterministic_topological_order(
        dirty_nodes=dirty_nodes,
        dependencies=tuple(dependencies),
    )
    affected_subject_ids = tuple(
        sorted({dirty_node.subject_id for dirty_node in dirty_nodes if dirty_node.subject_id is not None})
    )
    affected_locus_keys = tuple(
        sorted({dirty_node.locus_key for dirty_node in dirty_nodes if dirty_node.locus_key is not None})
    )

    return RebuildPlan(
        changes=tuple(changes),
        dirty_nodes=dirty_nodes,
        rebuild_order=rebuild_order,
        affected_subject_ids=affected_subject_ids,
        affected_locus_keys=affected_locus_keys,
    )


def _deterministic_topological_order(
    *,
    dirty_nodes: tuple[DirtyNode, ...],
    dependencies: tuple[CompilerDependency, ...],
) -> tuple[str, ...]:
    dirty_ids = {dirty_node.node_id for dirty_node in dirty_nodes}
    if not dirty_ids:
        return ()

    dirty_node_map = {
        dirty_node.node_id: dirty_node
        for dirty_node in dirty_nodes
    }
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in dirty_ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in dirty_ids}

    for dependency in dependencies:
        if dependency.upstream_node_id in dirty_ids and dependency.downstream_node_id in dirty_ids:
            predecessors[dependency.downstream_node_id].add(dependency.upstream_node_id)
            successors[dependency.upstream_node_id].add(dependency.downstream_node_id)

    ready = sorted(
        (node_id for node_id, node_predecessors in predecessors.items() if not node_predecessors),
        key=lambda node_id: _dirty_node_sort_key(dirty_node_map[node_id]),
    )
    order: list[str] = []

    while ready:
        current_node_id = ready.pop(0)
        order.append(current_node_id)
        for downstream_node_id in sorted(
            successors[current_node_id],
            key=lambda node_id: _dirty_node_sort_key(dirty_node_map[node_id]),
        ):
            predecessors[downstream_node_id].remove(current_node_id)
            if not predecessors[downstream_node_id]:
                ready.append(downstream_node_id)
                ready.sort(
                    key=lambda node_id: _dirty_node_sort_key(dirty_node_map[node_id])
                )

    if len(order) != len(dirty_ids):
        raise ValueError("compiler dependency graph must be acyclic")

    return tuple(order)


__all__ = [
    "CompiledArtifactKind",
    "CompilerChange",
    "CompilerDependency",
    "CompilerNode",
    "CompilerNodeCategory",
    "CompilerStateRepository",
    "DependencyRole",
    "DerivedArtifactKind",
    "DirtyCause",
    "DirtyNode",
    "DirtyQueueStatus",
    "DirtyReason",
    "IncrementalRebuildPlanner",
    "RebuildArtifact",
    "RebuildPlan",
    "SourceInputKind",
    "StagedRebuild",
    "UtilityStateKind",
    "detect_fingerprint_changes",
    "plan_incremental_rebuild",
]
