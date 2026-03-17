"""Forgetting, retraction, and erasure contract invariants for Continuity."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache


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


class ForgettingMode(StrEnum):
    SUPERSEDE = "supersede"
    SUPPRESS = "suppress"
    SEAL = "seal"
    EXPUNGE = "expunge"


class ForgettingTargetKind(StrEnum):
    CLAIM = "claim"
    LOCUS = "locus"
    SUBJECT = "subject"
    SESSION = "session"
    IMPORTED_ARTIFACT = "imported_artifact"
    COMPILED_VIEW = "compiled_view"


class ForgettingSurface(StrEnum):
    CLAIM_LEDGER = "claim_ledger"
    OBSERVATION_LOG = "observation_log"
    VECTOR_INDEX = "vector_index"
    SNAPSHOT_STORE = "snapshot_store"
    PREFETCH_CACHE = "prefetch_cache"
    REPLAY_ARTIFACTS = "replay_artifacts"
    ARCHIVE_TIER = "archive_tier"
    IMPORT_PIPELINE = "import_pipeline"
    DERIVATION_PIPELINE = "derivation_pipeline"
    TOMBSTONE_LEDGER = "tombstone_ledger"


class ArtifactResidency(StrEnum):
    RETAIN_CONTENT = "retain_content"
    HIDDEN_FROM_HOST = "hidden_from_host"
    ADMIN_METADATA_ONLY = "admin_metadata_only"
    TOMBSTONE_ONLY = "tombstone_only"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class ForgettingTarget:
    target_kind: ForgettingTargetKind
    target_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))


@dataclass(frozen=True, slots=True)
class ForgettingOperation:
    operation_id: str
    target: ForgettingTarget
    mode: ForgettingMode
    requested_by: str
    rationale: str
    policy_stamp: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _clean_text(self.operation_id, field_name="operation_id"))
        object.__setattr__(self, "requested_by", _clean_text(self.requested_by, field_name="requested_by"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )


@dataclass(frozen=True, slots=True)
class ForgettingRule:
    mode: ForgettingMode
    residencies: dict[ForgettingSurface, ArtifactResidency]
    blocks_resurrection_from: tuple[ForgettingSurface, ...] = ()
    host_reads_withdrawn: bool = False

    def __post_init__(self) -> None:
        expected_surfaces = set(ForgettingSurface)
        provided_surfaces = set(self.residencies)
        if provided_surfaces != expected_surfaces:
            missing = expected_surfaces - provided_surfaces
            extras = provided_surfaces - expected_surfaces
            raise ValueError(
                "residencies must define every forgetting surface "
                f"(missing={sorted(surface.value for surface in missing)}, "
                f"extra={sorted(surface.value for surface in extras)})"
            )
        object.__setattr__(
            self,
            "blocks_resurrection_from",
            tuple(dict.fromkeys(self.blocks_resurrection_from)),
        )

    def residency_for(self, surface: ForgettingSurface) -> ArtifactResidency:
        return self.residencies[surface]

    def blocks_resurrection(self, surface: ForgettingSurface) -> bool:
        return surface in self.blocks_resurrection_from


@dataclass(frozen=True, slots=True)
class ForgettingDecisionTrace:
    operation: ForgettingOperation
    rule: ForgettingRule

    def __post_init__(self) -> None:
        if self.operation.mode is not self.rule.mode:
            raise ValueError("operation mode must match forgetting rule mode")

    def residency_for(self, surface: ForgettingSurface) -> ArtifactResidency:
        return self.rule.residency_for(surface)

    def blocks_resurrection(self, surface: ForgettingSurface) -> bool:
        return self.rule.blocks_resurrection(surface)

    @property
    def retained_surfaces(self) -> tuple[ForgettingSurface, ...]:
        return tuple(
            surface
            for surface in ForgettingSurface
            if self.residency_for(surface) is not ArtifactResidency.REMOVED
        )

    @property
    def removed_surfaces(self) -> tuple[ForgettingSurface, ...]:
        return tuple(
            surface
            for surface in ForgettingSurface
            if self.residency_for(surface) is ArtifactResidency.REMOVED
        )


@dataclass(frozen=True, slots=True)
class ForgettingSurfaceEffect:
    operation_id: str
    surface: ForgettingSurface
    residency: ArtifactResidency
    blocks_resurrection: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _clean_text(self.operation_id, field_name="operation_id"))
        object.__setattr__(self, "blocks_resurrection", bool(self.blocks_resurrection))


@dataclass(frozen=True, slots=True)
class ForgettingTombstone:
    tombstone_id: str
    operation_id: str
    target: ForgettingTarget
    surface: ForgettingSurface
    content_fingerprint: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "tombstone_id", _clean_text(self.tombstone_id, field_name="tombstone_id"))
        object.__setattr__(self, "operation_id", _clean_text(self.operation_id, field_name="operation_id"))
        object.__setattr__(
            self,
            "content_fingerprint",
            _clean_text(self.content_fingerprint, field_name="content_fingerprint"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )


@dataclass(frozen=True, slots=True)
class ForgettingRecord:
    operation: ForgettingOperation
    surface_effects: tuple[ForgettingSurfaceEffect, ...]
    tombstones: tuple[ForgettingTombstone, ...] = ()

    def __post_init__(self) -> None:
        effects_by_surface = {effect.surface: effect for effect in self.surface_effects}
        if set(effects_by_surface) != set(ForgettingSurface):
            raise ValueError("surface_effects must cover every forgetting surface exactly once")
        if any(effect.operation_id != self.operation.operation_id for effect in self.surface_effects):
            raise ValueError("surface effects must belong to the same forgetting operation")
        if any(tombstone.operation_id != self.operation.operation_id for tombstone in self.tombstones):
            raise ValueError("tombstones must belong to the same forgetting operation")

        object.__setattr__(
            self,
            "surface_effects",
            tuple(effects_by_surface[surface] for surface in ForgettingSurface),
        )
        object.__setattr__(
            self,
            "tombstones",
            tuple(sorted(self.tombstones, key=lambda tombstone: (tombstone.recorded_at, tombstone.tombstone_id))),
        )

    def residency_for(self, surface: ForgettingSurface) -> ArtifactResidency:
        for effect in self.surface_effects:
            if effect.surface is surface:
                return effect.residency
        raise KeyError(surface)

    def blocks_resurrection(self, surface: ForgettingSurface) -> bool:
        for effect in self.surface_effects:
            if effect.surface is surface:
                return effect.blocks_resurrection
        raise KeyError(surface)

    @property
    def host_reads_withdrawn(self) -> bool:
        return forgetting_rule_for(self.operation.mode).host_reads_withdrawn

    @property
    def withdrawal_surfaces(self) -> tuple[ForgettingSurface, ...]:
        return tuple(
            surface
            for surface in ForgettingSurface
            if self.residency_for(surface) is not ArtifactResidency.RETAIN_CONTENT
        )

    @property
    def resurrection_guard_surfaces(self) -> tuple[ForgettingSurface, ...]:
        return tuple(
            surface
            for surface in ForgettingSurface
            if self.blocks_resurrection(surface)
        )


class ForgettingRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def record_operation(
        self,
        trace: ForgettingDecisionTrace,
        *,
        tombstones: tuple[ForgettingTombstone, ...] = (),
    ) -> None:
        for tombstone in tombstones:
            if tombstone.operation_id != trace.operation.operation_id:
                raise ValueError("tombstone operation_id must match the forgetting operation")
            if tombstone.target != trace.operation.target:
                raise ValueError("tombstone target must match the forgetting operation target")

        with self._connection:
            self._connection.execute(
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
                    trace.operation.operation_id,
                    trace.operation.target.target_kind.value,
                    trace.operation.target.target_id,
                    trace.operation.mode.value,
                    trace.operation.requested_by,
                    trace.operation.rationale,
                    trace.operation.policy_stamp,
                    trace.operation.recorded_at.isoformat(),
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO forgetting_surface_effects(
                    operation_id,
                    surface,
                    residency,
                    blocks_resurrection
                )
                VALUES (?, ?, ?, ?)
                """,
                tuple(
                    (
                        trace.operation.operation_id,
                        surface.value,
                        trace.residency_for(surface).value,
                        int(trace.blocks_resurrection(surface)),
                    )
                    for surface in ForgettingSurface
                ),
            )
            self._connection.executemany(
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
                tuple(
                    (
                        tombstone.tombstone_id,
                        tombstone.operation_id,
                        tombstone.target.target_kind.value,
                        tombstone.target.target_id,
                        tombstone.surface.value,
                        tombstone.content_fingerprint,
                        tombstone.recorded_at.isoformat(),
                    )
                    for tombstone in tombstones
                ),
            )

    def read_record(self, operation_id: str) -> ForgettingRecord | None:
        operation_row = self._connection.execute(
            """
            SELECT
                operation_id,
                target_kind,
                target_id,
                mode,
                requested_by,
                rationale,
                policy_stamp,
                recorded_at
            FROM forgetting_operations
            WHERE operation_id = ?
            """,
            (_clean_text(operation_id, field_name="operation_id"),),
        ).fetchone()
        if operation_row is None:
            return None
        return self._record_from_operation_row(operation_row)

    def list_operations(
        self,
        *,
        target: ForgettingTarget | None = None,
        mode: ForgettingMode | None = None,
        requested_by: str | None = None,
        limit: int | None = None,
    ) -> tuple[ForgettingRecord, ...]:
        conditions: list[str] = []
        params: list[str] = []
        if target is not None:
            conditions.extend(("target_kind = ?", "target_id = ?"))
            params.extend((target.target_kind.value, target.target_id))
        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode.value)
        if requested_by is not None:
            conditions.append("requested_by = ?")
            params.append(_clean_text(requested_by, field_name="requested_by"))
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {limit}"

        rows = self._connection.execute(
            f"""
            SELECT
                operation_id,
                target_kind,
                target_id,
                mode,
                requested_by,
                rationale,
                policy_stamp,
                recorded_at
            FROM forgetting_operations
            {where_clause}
            ORDER BY recorded_at DESC, operation_id DESC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return tuple(self._record_from_operation_row(row) for row in rows)

    def list_tombstones(
        self,
        *,
        target: ForgettingTarget | None = None,
        surface: ForgettingSurface | None = None,
        operation_id: str | None = None,
    ) -> tuple[ForgettingTombstone, ...]:
        conditions: list[str] = []
        params: list[str] = []
        if target is not None:
            conditions.extend(("target_kind = ?", "target_id = ?"))
            params.extend((target.target_kind.value, target.target_id))
        if surface is not None:
            conditions.append("surface = ?")
            params.append(surface.value)
        if operation_id is not None:
            conditions.append("operation_id = ?")
            params.append(_clean_text(operation_id, field_name="operation_id"))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        rows = self._connection.execute(
            f"""
            SELECT
                tombstone_id,
                operation_id,
                target_kind,
                target_id,
                surface,
                content_fingerprint,
                recorded_at
            FROM forgetting_tombstones
            {where_clause}
            ORDER BY recorded_at, tombstone_id
            """,
            tuple(params),
        ).fetchall()
        return tuple(self._tombstone_from_row(row) for row in rows)

    def current_record_for_target(self, target: ForgettingTarget) -> ForgettingRecord | None:
        records = self.list_operations(target=target, limit=1)
        if not records:
            return None
        return records[0]

    def surfaces_requiring_withdrawal(self, target: ForgettingTarget) -> tuple[ForgettingSurface, ...]:
        record = self.current_record_for_target(target)
        if record is None:
            return ()
        return record.withdrawal_surfaces

    def resurrection_guard_surfaces(self, target: ForgettingTarget) -> tuple[ForgettingSurface, ...]:
        record = self.current_record_for_target(target)
        if record is None:
            return ()
        return record.resurrection_guard_surfaces

    def ordinary_read_blocked(self, target: ForgettingTarget) -> bool:
        record = self.current_record_for_target(target)
        if record is None:
            return False
        return record.host_reads_withdrawn

    def _record_from_operation_row(self, row: tuple[object, ...]) -> ForgettingRecord:
        operation = ForgettingOperation(
            operation_id=row[0],
            target=ForgettingTarget(
                target_kind=ForgettingTargetKind(row[1]),
                target_id=row[2],
            ),
            mode=ForgettingMode(row[3]),
            requested_by=row[4],
            rationale=row[5],
            policy_stamp=row[6],
            recorded_at=_parse_timestamp(row[7], field_name="recorded_at"),
        )
        effect_rows = self._connection.execute(
            """
            SELECT operation_id, surface, residency, blocks_resurrection
            FROM forgetting_surface_effects
            WHERE operation_id = ?
            """,
            (operation.operation_id,),
        ).fetchall()
        tombstone_rows = self._connection.execute(
            """
            SELECT
                tombstone_id,
                operation_id,
                target_kind,
                target_id,
                surface,
                content_fingerprint,
                recorded_at
            FROM forgetting_tombstones
            WHERE operation_id = ?
            ORDER BY recorded_at, tombstone_id
            """,
            (operation.operation_id,),
        ).fetchall()
        return ForgettingRecord(
            operation=operation,
            surface_effects=tuple(
                ForgettingSurfaceEffect(
                    operation_id=effect_row[0],
                    surface=ForgettingSurface(effect_row[1]),
                    residency=ArtifactResidency(effect_row[2]),
                    blocks_resurrection=bool(effect_row[3]),
                )
                for effect_row in effect_rows
            ),
            tombstones=tuple(self._tombstone_from_row(row) for row in tombstone_rows),
        )

    def _tombstone_from_row(self, row: tuple[object, ...]) -> ForgettingTombstone:
        return ForgettingTombstone(
            tombstone_id=row[0],
            operation_id=row[1],
            target=ForgettingTarget(
                target_kind=ForgettingTargetKind(row[2]),
                target_id=row[3],
            ),
            surface=ForgettingSurface(row[4]),
            content_fingerprint=row[5],
            recorded_at=_parse_timestamp(row[6], field_name="recorded_at"),
        )


@lru_cache(maxsize=1)
def forgetting_rules() -> dict[ForgettingMode, ForgettingRule]:
    return {
        ForgettingMode.SUPERSEDE: ForgettingRule(
            mode=ForgettingMode.SUPERSEDE,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.REMOVED,
            },
        ),
        ForgettingMode.SUPPRESS: ForgettingRule(
            mode=ForgettingMode.SUPPRESS,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.REMOVED,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.REMOVED,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.REMOVED,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.HIDDEN_FROM_HOST,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.ADMIN_METADATA_ONLY,
            },
            host_reads_withdrawn=True,
        ),
        ForgettingMode.SEAL: ForgettingRule(
            mode=ForgettingMode.SEAL,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.REMOVED,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.REMOVED,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.REMOVED,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.REMOVED,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.REMOVED,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.ADMIN_METADATA_ONLY,
            },
            blocks_resurrection_from=(
                ForgettingSurface.SNAPSHOT_STORE,
                ForgettingSurface.PREFETCH_CACHE,
                ForgettingSurface.REPLAY_ARTIFACTS,
                ForgettingSurface.DERIVATION_PIPELINE,
            ),
            host_reads_withdrawn=True,
        ),
        ForgettingMode.EXPUNGE: ForgettingRule(
            mode=ForgettingMode.EXPUNGE,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.REMOVED,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.REMOVED,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.REMOVED,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.REMOVED,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.REMOVED,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.REMOVED,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.REMOVED,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.REMOVED,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.REMOVED,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.TOMBSTONE_ONLY,
            },
            blocks_resurrection_from=(
                ForgettingSurface.VECTOR_INDEX,
                ForgettingSurface.SNAPSHOT_STORE,
                ForgettingSurface.PREFETCH_CACHE,
                ForgettingSurface.REPLAY_ARTIFACTS,
                ForgettingSurface.ARCHIVE_TIER,
                ForgettingSurface.IMPORT_PIPELINE,
                ForgettingSurface.DERIVATION_PIPELINE,
            ),
            host_reads_withdrawn=True,
        ),
    }


def forgetting_rule_for(mode: ForgettingMode) -> ForgettingRule:
    return forgetting_rules()[mode]
