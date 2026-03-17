"""Async next-turn prefetch cache pinned to snapshot-consistent compiled views."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.compiler import CompilerNodeCategory, CompilerStateRepository
from continuity.snapshots import (
    SnapshotArtifactKind,
    SnapshotReadPin,
    SnapshotReadUse,
    SnapshotRepository,
)
from continuity.store.sqlite import SQLiteRepository
from continuity.tiers import TierStateRepository
from continuity.views import TierDefault, ViewKind, view_contract_for


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _validate_timestamp(value: datetime | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _dump_text_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(list(values))


def _load_text_tuple(value: str, *, field_name: str) -> tuple[str, ...]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError(f"{field_name} must be a JSON array")
    cleaned = tuple(_clean_text(str(item), field_name=field_name) for item in loaded)
    return tuple(dict.fromkeys(cleaned))


def prefetch_key_for(session_id: str) -> str:
    return f"prefetch:{_clean_text(session_id, field_name='session_id')}"


class PrefetchStatus(StrEnum):
    PENDING = "pending"
    WARM = "warm"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class PrefetchCacheEntry:
    prefetch_key: str
    session_id: str
    snapshot_id: str
    status: PrefetchStatus
    artifact_ids: tuple[str, ...]
    warmed_at: datetime | None = None
    invalidated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prefetch_key",
            _clean_text(self.prefetch_key, field_name="prefetch_key"),
        )
        object.__setattr__(self, "session_id", _clean_text(self.session_id, field_name="session_id"))
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(
            self,
            "artifact_ids",
            tuple(
                dict.fromkeys(
                    _clean_text(artifact_id, field_name="artifact_ids")
                    for artifact_id in self.artifact_ids
                )
            ),
        )
        object.__setattr__(
            self,
            "warmed_at",
            _validate_timestamp(self.warmed_at, field_name="warmed_at"),
        )
        object.__setattr__(
            self,
            "invalidated_at",
            _validate_timestamp(self.invalidated_at, field_name="invalidated_at"),
        )

        if self.prefetch_key != prefetch_key_for(self.session_id):
            raise ValueError("prefetch_key must match the canonical session-derived key")
        if self.status is PrefetchStatus.WARM and self.warmed_at is None:
            raise ValueError("warm cache entries require warmed_at")
        if self.status is PrefetchStatus.INVALIDATED and self.invalidated_at is None:
            raise ValueError("invalidated cache entries require invalidated_at")

    @property
    def read_pin(self) -> SnapshotReadPin:
        return SnapshotReadPin(
            snapshot_id=self.snapshot_id,
            read_use=SnapshotReadUse.PREFETCH,
            consumer_id=self.prefetch_key,
        )


class PrefetchRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def upsert(self, entry: PrefetchCacheEntry) -> None:
        with self._connection:
            self._connection.execute(
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
                ON CONFLICT(prefetch_key) DO UPDATE SET
                    snapshot_id = excluded.snapshot_id,
                    session_id = excluded.session_id,
                    status = excluded.status,
                    artifact_ids_json = excluded.artifact_ids_json,
                    warmed_at = excluded.warmed_at,
                    invalidated_at = excluded.invalidated_at
                """,
                (
                    entry.prefetch_key,
                    entry.snapshot_id,
                    entry.session_id,
                    entry.status.value,
                    _dump_text_tuple(entry.artifact_ids),
                    entry.warmed_at.isoformat() if entry.warmed_at is not None else None,
                    entry.invalidated_at.isoformat() if entry.invalidated_at is not None else None,
                ),
            )

    def read(
        self,
        *,
        session_id: str | None = None,
        prefetch_key: str | None = None,
    ) -> PrefetchCacheEntry | None:
        key = self._resolve_key(session_id=session_id, prefetch_key=prefetch_key)
        row = self._connection.execute(
            """
            SELECT prefetch_key, snapshot_id, session_id, status, artifact_ids_json, warmed_at, invalidated_at
            FROM prefetch_state
            WHERE prefetch_key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        return self._entry_from_row(row)

    def list(
        self,
        *,
        status: PrefetchStatus | None = None,
    ) -> tuple[PrefetchCacheEntry, ...]:
        sql = """
            SELECT prefetch_key, snapshot_id, session_id, status, artifact_ids_json, warmed_at, invalidated_at
            FROM prefetch_state
        """
        parameters: list[str] = []
        if status is not None:
            sql += " WHERE status = ?"
            parameters.append(status.value)
        sql += " ORDER BY rowid"
        rows = self._connection.execute(sql, tuple(parameters)).fetchall()
        return tuple(self._entry_from_row(row) for row in rows)

    def _resolve_key(
        self,
        *,
        session_id: str | None,
        prefetch_key: str | None,
    ) -> str:
        cleaned_key = _clean_optional_text(prefetch_key, field_name="prefetch_key")
        cleaned_session_id = _clean_optional_text(session_id, field_name="session_id")
        if cleaned_key is None and cleaned_session_id is None:
            raise ValueError("session_id or prefetch_key is required")
        if cleaned_key is not None and cleaned_session_id is not None:
            expected = prefetch_key_for(cleaned_session_id)
            if cleaned_key != expected:
                raise ValueError("prefetch_key does not match session_id")
        return cleaned_key or prefetch_key_for(cleaned_session_id or "")

    def _entry_from_row(self, row: tuple[object, ...]) -> PrefetchCacheEntry:
        warmed_at = datetime.fromisoformat(row[5]) if row[5] is not None else None
        invalidated_at = datetime.fromisoformat(row[6]) if row[6] is not None else None
        return PrefetchCacheEntry(
            prefetch_key=row[0],
            snapshot_id=row[1],
            session_id=row[2],
            status=PrefetchStatus(row[3]),
            artifact_ids=_load_text_tuple(row[4], field_name="artifact_ids"),
            warmed_at=warmed_at,
            invalidated_at=invalidated_at,
        )


class PrefetchRuntime:
    """Warm and serve next-turn cache entries without changing snapshot semantics."""

    _view_kind_by_artifact_kind = {
        SnapshotArtifactKind.STATE_VIEW: ViewKind.STATE,
        SnapshotArtifactKind.TIMELINE_VIEW: ViewKind.TIMELINE,
        SnapshotArtifactKind.SET_VIEW: ViewKind.SET,
        SnapshotArtifactKind.PROFILE_VIEW: ViewKind.PROFILE,
        SnapshotArtifactKind.PROMPT_VIEW: ViewKind.PROMPT,
        SnapshotArtifactKind.EVIDENCE_VIEW: ViewKind.EVIDENCE,
        SnapshotArtifactKind.ANSWER_VIEW: ViewKind.ANSWER,
    }

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        head_key: str = "current",
    ) -> None:
        self._connection = connection
        self._head_key = _clean_text(head_key, field_name="head_key")
        self._sessions = SQLiteRepository(connection)
        self._snapshots = SnapshotRepository(connection)
        self._prefetch = PrefetchRepository(connection)
        self._tiers = TierStateRepository(connection)
        self._compiler = CompilerStateRepository(connection)

    def warm_next_turn(
        self,
        *,
        session_id: str,
        warmed_at: datetime,
        target_snapshot_id: str | None = None,
    ) -> PrefetchCacheEntry:
        cleaned_session_id = _clean_text(session_id, field_name="session_id")
        session = self._sessions.read_session(cleaned_session_id)
        if session is None:
            raise KeyError(cleaned_session_id)

        snapshot = self._resolve_snapshot(target_snapshot_id=target_snapshot_id)
        artifact_ids = tuple(
            artifact_ref.artifact_id
            for artifact_ref in snapshot.artifact_refs
            if self._include_in_default_host_reads(
                artifact_kind=artifact_ref.artifact_kind,
                artifact_id=artifact_ref.artifact_id,
                policy_stamp=snapshot.policy_stamp,
            )
        )
        entry = PrefetchCacheEntry(
            prefetch_key=prefetch_key_for(session.session_id),
            session_id=session.session_id,
            snapshot_id=snapshot.snapshot_id,
            status=PrefetchStatus.WARM,
            artifact_ids=artifact_ids,
            warmed_at=warmed_at,
        )
        self._prefetch.upsert(entry)
        self._ensure_snapshot_pin(entry)
        stored = self._prefetch.read(session_id=session.session_id)
        assert stored is not None
        return stored

    def read_warm_cache(
        self,
        *,
        session_id: str,
        target_snapshot_id: str | None = None,
    ) -> PrefetchCacheEntry | None:
        entry = self._prefetch.read(session_id=session_id)
        if entry is None or entry.status is not PrefetchStatus.WARM:
            return None

        expected_snapshot_id = (
            _clean_text(target_snapshot_id, field_name="target_snapshot_id")
            if target_snapshot_id is not None
            else self._active_snapshot_id()
        )
        if expected_snapshot_id is None or entry.snapshot_id != expected_snapshot_id:
            return None
        return entry

    def invalidate_dirty_caches(
        self,
        *,
        invalidated_at: datetime,
    ) -> tuple[PrefetchCacheEntry, ...]:
        dirty_nodes = self._compiler.list_dirty_nodes()
        if not dirty_nodes:
            return ()

        upstream_dirty = any(
            dirty_node.category is not CompilerNodeCategory.COMPILED_ARTIFACT
            for dirty_node in dirty_nodes
        )
        dirty_artifact_ids = {
            dirty_node.node_id
            for dirty_node in dirty_nodes
            if dirty_node.category is CompilerNodeCategory.COMPILED_ARTIFACT
        }

        invalidated_entries: list[PrefetchCacheEntry] = []
        for entry in self._prefetch.list(status=PrefetchStatus.WARM):
            if not upstream_dirty and not dirty_artifact_ids.intersection(entry.artifact_ids):
                continue
            invalidated_entry = PrefetchCacheEntry(
                prefetch_key=entry.prefetch_key,
                session_id=entry.session_id,
                snapshot_id=entry.snapshot_id,
                status=PrefetchStatus.INVALIDATED,
                artifact_ids=entry.artifact_ids,
                warmed_at=entry.warmed_at,
                invalidated_at=invalidated_at,
            )
            self._prefetch.upsert(invalidated_entry)
            invalidated_entries.append(invalidated_entry)
        return tuple(invalidated_entries)

    def _resolve_snapshot(self, *, target_snapshot_id: str | None) -> object:
        if target_snapshot_id is not None:
            snapshot = self._snapshots.read_snapshot(
                _clean_text(target_snapshot_id, field_name="target_snapshot_id")
            )
        else:
            snapshot = self._snapshots.read_active_snapshot(head_key=self._head_key)
        if snapshot is None:
            raise ValueError("target snapshot is not available")
        return snapshot

    def _active_snapshot_id(self) -> str | None:
        snapshot = self._snapshots.read_active_snapshot(head_key=self._head_key)
        if snapshot is None:
            return None
        return snapshot.snapshot_id

    def _ensure_snapshot_pin(self, entry: PrefetchCacheEntry) -> None:
        if entry.read_pin in self._snapshots.list_read_pins(
            snapshot_id=entry.snapshot_id,
            read_use=SnapshotReadUse.PREFETCH,
        ):
            return
        self._snapshots.pin_read(entry.read_pin)

    def _include_in_default_host_reads(
        self,
        *,
        artifact_kind: SnapshotArtifactKind,
        artifact_id: str,
        policy_stamp: str,
    ) -> bool:
        view_kind = self._view_kind_by_artifact_kind.get(artifact_kind)
        if view_kind is None:
            return False

        retention = self._tiers.read_retention_metadata(
            target_kind="compiled_view",
            target_id=artifact_id,
            policy_stamp=_clean_text(policy_stamp, field_name="policy_stamp"),
        )
        if retention is not None:
            return retention.default_in_host_reads

        return any(
            tier_default in {TierDefault.HOT, TierDefault.WARM}
            for tier_default in view_contract_for(view_kind).tier_defaults
        )


__all__ = [
    "PrefetchCacheEntry",
    "PrefetchRepository",
    "PrefetchRuntime",
    "PrefetchStatus",
    "prefetch_key_for",
]
