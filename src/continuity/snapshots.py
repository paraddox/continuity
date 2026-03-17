"""Snapshot consistency invariants for Continuity."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.transactions import TransactionKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class SnapshotArtifactKind(StrEnum):
    STATE_VIEW = "state_view"
    TIMELINE_VIEW = "timeline_view"
    SET_VIEW = "set_view"
    PROFILE_VIEW = "profile_view"
    PROMPT_VIEW = "prompt_view"
    EVIDENCE_VIEW = "evidence_view"
    ANSWER_VIEW = "answer_view"
    VECTOR_INDEX = "vector_index"


class SnapshotHeadState(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"


class SnapshotReadUse(StrEnum):
    RETRIEVAL = "retrieval"
    PROMPT_ASSEMBLY = "prompt_assembly"
    ANSWER_QUERY = "answer_query"
    PREFETCH = "prefetch"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class SnapshotArtifactRef:
    artifact_kind: SnapshotArtifactKind
    artifact_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _clean_text(self.artifact_id, field_name="artifact_id"))


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    snapshot_id: str
    policy_stamp: str
    parent_snapshot_id: str | None
    created_by_transaction: TransactionKind
    artifact_refs: tuple[SnapshotArtifactRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        if self.parent_snapshot_id is not None:
            object.__setattr__(
                self,
                "parent_snapshot_id",
                _clean_text(self.parent_snapshot_id, field_name="parent_snapshot_id"),
            )
        object.__setattr__(self, "artifact_refs", tuple(dict.fromkeys(self.artifact_refs)))

        if not self.artifact_refs:
            raise ValueError("artifact_refs must be non-empty")
        if self.parent_snapshot_id == self.snapshot_id:
            raise ValueError("parent_snapshot_id must differ from snapshot_id")


@dataclass(frozen=True, slots=True)
class SnapshotHead:
    head_key: str
    state: SnapshotHeadState
    snapshot_id: str
    based_on_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_key", _clean_text(self.head_key, field_name="head_key"))
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        if self.based_on_snapshot_id is not None:
            object.__setattr__(
                self,
                "based_on_snapshot_id",
                _clean_text(self.based_on_snapshot_id, field_name="based_on_snapshot_id"),
            )

        if self.state is SnapshotHeadState.ACTIVE and self.based_on_snapshot_id is not None:
            raise ValueError("active heads cannot keep a based_on_snapshot_id")
        if self.state is SnapshotHeadState.CANDIDATE and self.based_on_snapshot_id is None:
            raise ValueError("candidate heads must declare the snapshot they are based on")


@dataclass(frozen=True, slots=True)
class SnapshotReadPin:
    snapshot_id: str
    read_use: SnapshotReadUse
    consumer_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(self, "consumer_id", _clean_text(self.consumer_id, field_name="consumer_id"))


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    from_snapshot_id: str
    to_snapshot_id: str
    added_artifacts: tuple[SnapshotArtifactRef, ...]
    removed_artifacts: tuple[SnapshotArtifactRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_snapshot_id",
            _clean_text(self.from_snapshot_id, field_name="from_snapshot_id"),
        )
        object.__setattr__(self, "to_snapshot_id", _clean_text(self.to_snapshot_id, field_name="to_snapshot_id"))
        object.__setattr__(self, "added_artifacts", tuple(dict.fromkeys(self.added_artifacts)))
        object.__setattr__(self, "removed_artifacts", tuple(dict.fromkeys(self.removed_artifacts)))


@dataclass(frozen=True, slots=True)
class SnapshotPromotion:
    previous_active_snapshot_id: str
    promoted_snapshot_id: str
    new_active_head: SnapshotHead

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_active_snapshot_id",
            _clean_text(
                self.previous_active_snapshot_id,
                field_name="previous_active_snapshot_id",
            ),
        )
        object.__setattr__(
            self,
            "promoted_snapshot_id",
            _clean_text(self.promoted_snapshot_id, field_name="promoted_snapshot_id"),
        )
        if self.new_active_head.state is not SnapshotHeadState.ACTIVE:
            raise ValueError("promotion must yield an active head")
        if self.new_active_head.snapshot_id != self.promoted_snapshot_id:
            raise ValueError("new_active_head must point at the promoted snapshot")


@dataclass(frozen=True, slots=True)
class SnapshotRollback:
    previous_active_snapshot_id: str
    rollback_to_snapshot_id: str
    reason: str
    new_active_head: SnapshotHead

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_active_snapshot_id",
            _clean_text(
                self.previous_active_snapshot_id,
                field_name="previous_active_snapshot_id",
            ),
        )
        object.__setattr__(
            self,
            "rollback_to_snapshot_id",
            _clean_text(self.rollback_to_snapshot_id, field_name="rollback_to_snapshot_id"),
        )
        object.__setattr__(self, "reason", _clean_text(self.reason, field_name="reason"))
        if self.new_active_head.state is not SnapshotHeadState.ACTIVE:
            raise ValueError("rollback must yield an active head")
        if self.new_active_head.snapshot_id != self.rollback_to_snapshot_id:
            raise ValueError("new_active_head must point at the rollback target")


@dataclass(frozen=True, slots=True)
class SnapshotPromotionRecord:
    promotion_id: str
    head_key: str
    previous_active_snapshot_id: str | None
    promoted_snapshot_id: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "promotion_id", _clean_text(self.promotion_id, field_name="promotion_id"))
        object.__setattr__(self, "head_key", _clean_text(self.head_key, field_name="head_key"))
        object.__setattr__(
            self,
            "previous_active_snapshot_id",
            _clean_optional_text(
                self.previous_active_snapshot_id,
                field_name="previous_active_snapshot_id",
            ),
        )
        object.__setattr__(
            self,
            "promoted_snapshot_id",
            _clean_text(self.promoted_snapshot_id, field_name="promoted_snapshot_id"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )


class SnapshotRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def save_snapshot(self, snapshot: MemorySnapshot) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id,
                    policy_stamp,
                    parent_snapshot_id,
                    created_by_transaction
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.policy_stamp,
                    snapshot.parent_snapshot_id,
                    snapshot.created_by_transaction.value,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO snapshot_artifacts(snapshot_id, artifact_kind, artifact_id)
                VALUES (?, ?, ?)
                """,
                (
                    (
                        snapshot.snapshot_id,
                        artifact_ref.artifact_kind.value,
                        artifact_ref.artifact_id,
                    )
                    for artifact_ref in snapshot.artifact_refs
                ),
            )

    def read_snapshot(self, snapshot_id: str) -> MemorySnapshot | None:
        cleaned_snapshot_id = _clean_text(snapshot_id, field_name="snapshot_id")
        row = self._connection.execute(
            """
            SELECT snapshot_id, policy_stamp, parent_snapshot_id, created_by_transaction
            FROM snapshots
            WHERE snapshot_id = ?
            """,
            (cleaned_snapshot_id,),
        ).fetchone()
        if row is None:
            return None

        artifact_rows = self._connection.execute(
            """
            SELECT artifact_kind, artifact_id
            FROM snapshot_artifacts
            WHERE snapshot_id = ?
            ORDER BY rowid
            """,
            (cleaned_snapshot_id,),
        ).fetchall()
        return MemorySnapshot(
            snapshot_id=row[0],
            policy_stamp=row[1],
            parent_snapshot_id=row[2],
            created_by_transaction=TransactionKind(row[3]),
            artifact_refs=tuple(
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind(artifact_row[0]),
                    artifact_id=artifact_row[1],
                )
                for artifact_row in artifact_rows
            ),
        )

    def list_snapshots(self) -> tuple[MemorySnapshot, ...]:
        rows = self._connection.execute(
            """
            SELECT snapshot_id
            FROM snapshots
            ORDER BY rowid
            """
        ).fetchall()
        return tuple(
            snapshot
            for row in rows
            if (snapshot := self.read_snapshot(row[0])) is not None
        )

    def upsert_head(self, head: SnapshotHead) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO snapshot_heads(head_key, state, snapshot_id, based_on_snapshot_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(head_key, state) DO UPDATE SET
                    snapshot_id = excluded.snapshot_id,
                    based_on_snapshot_id = excluded.based_on_snapshot_id
                """,
                (
                    head.head_key,
                    head.state.value,
                    head.snapshot_id,
                    head.based_on_snapshot_id,
                ),
            )

    def list_heads(
        self,
        *,
        head_key: str | None = None,
        state: SnapshotHeadState | None = None,
    ) -> tuple[SnapshotHead, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if head_key is not None:
            clauses.append("head_key = ?")
            parameters.append(_clean_text(head_key, field_name="head_key"))
        if state is not None:
            clauses.append("state = ?")
            parameters.append(state.value)

        sql = """
            SELECT head_key, state, snapshot_id, based_on_snapshot_id
            FROM snapshot_heads
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"

        rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(
            SnapshotHead(
                head_key=row[0],
                state=SnapshotHeadState(row[1]),
                snapshot_id=row[2],
                based_on_snapshot_id=row[3],
            )
            for row in rows
        )

    def read_active_snapshot(self, *, head_key: str) -> MemorySnapshot | None:
        return self._read_snapshot_for_head(head_key=head_key, state=SnapshotHeadState.ACTIVE)

    def read_candidate_snapshot(self, *, head_key: str) -> MemorySnapshot | None:
        return self._read_snapshot_for_head(head_key=head_key, state=SnapshotHeadState.CANDIDATE)

    def pin_read(self, pin: SnapshotReadPin) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO snapshot_read_pins(snapshot_id, read_use, consumer_id)
                VALUES (?, ?, ?)
                """,
                (
                    pin.snapshot_id,
                    pin.read_use.value,
                    pin.consumer_id,
                ),
            )

    def list_read_pins(
        self,
        *,
        snapshot_id: str | None = None,
        read_use: SnapshotReadUse | None = None,
    ) -> tuple[SnapshotReadPin, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if snapshot_id is not None:
            clauses.append("snapshot_id = ?")
            parameters.append(_clean_text(snapshot_id, field_name="snapshot_id"))
        if read_use is not None:
            clauses.append("read_use = ?")
            parameters.append(read_use.value)

        sql = """
            SELECT snapshot_id, read_use, consumer_id
            FROM snapshot_read_pins
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"

        rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(
            SnapshotReadPin(
                snapshot_id=row[0],
                read_use=SnapshotReadUse(row[1]),
                consumer_id=row[2],
            )
            for row in rows
        )

    def record_promotion(
        self,
        *,
        promotion_id: str,
        head_key: str,
        promotion: SnapshotPromotion,
        recorded_at: datetime,
    ) -> None:
        promotion_record = SnapshotPromotionRecord(
            promotion_id=promotion_id,
            head_key=head_key,
            previous_active_snapshot_id=promotion.previous_active_snapshot_id,
            promoted_snapshot_id=promotion.promoted_snapshot_id,
            recorded_at=recorded_at,
        )
        with self._connection:
            self._connection.execute(
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
                (
                    promotion_record.promotion_id,
                    promotion_record.head_key,
                    promotion_record.previous_active_snapshot_id,
                    promotion_record.promoted_snapshot_id,
                    promotion_record.recorded_at.isoformat(),
                ),
            )

    def list_promotions(
        self,
        *,
        head_key: str | None = None,
    ) -> tuple[SnapshotPromotionRecord, ...]:
        sql = """
            SELECT promotion_id, head_key, previous_active_snapshot_id, promoted_snapshot_id, recorded_at
            FROM snapshot_promotions
        """
        parameters: list[str] = []
        if head_key is not None:
            sql += " WHERE head_key = ?"
            parameters.append(_clean_text(head_key, field_name="head_key"))
        sql += " ORDER BY rowid"

        rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(
            SnapshotPromotionRecord(
                promotion_id=row[0],
                head_key=row[1],
                previous_active_snapshot_id=row[2],
                promoted_snapshot_id=row[3],
                recorded_at=datetime.fromisoformat(row[4]),
            )
            for row in rows
        )

    def _read_snapshot_for_head(
        self,
        *,
        head_key: str,
        state: SnapshotHeadState,
    ) -> MemorySnapshot | None:
        head_rows = self.list_heads(head_key=head_key, state=state)
        if not head_rows:
            return None
        return self.read_snapshot(head_rows[0].snapshot_id)


def diff_snapshots(from_snapshot: MemorySnapshot, to_snapshot: MemorySnapshot) -> SnapshotDiff:
    if from_snapshot.policy_stamp != to_snapshot.policy_stamp:
        raise ValueError("snapshot diffs require a shared policy stamp")

    from_refs = set(from_snapshot.artifact_refs)
    to_refs = set(to_snapshot.artifact_refs)

    return SnapshotDiff(
        from_snapshot_id=from_snapshot.snapshot_id,
        to_snapshot_id=to_snapshot.snapshot_id,
        added_artifacts=tuple(ref for ref in to_snapshot.artifact_refs if ref not in from_refs),
        removed_artifacts=tuple(ref for ref in from_snapshot.artifact_refs if ref not in to_refs),
    )


def promote_candidate_head(*, active_head: SnapshotHead, candidate_head: SnapshotHead) -> SnapshotPromotion:
    if active_head.state is not SnapshotHeadState.ACTIVE:
        raise ValueError("promotion requires an active head")
    if candidate_head.state is not SnapshotHeadState.CANDIDATE:
        raise ValueError("promotion requires a candidate head")
    if active_head.head_key != candidate_head.head_key:
        raise ValueError("promotion requires one shared head_key")
    if candidate_head.based_on_snapshot_id != active_head.snapshot_id:
        raise ValueError("candidate head must be based on the current active snapshot")

    return SnapshotPromotion(
        previous_active_snapshot_id=active_head.snapshot_id,
        promoted_snapshot_id=candidate_head.snapshot_id,
        new_active_head=SnapshotHead(
            head_key=active_head.head_key,
            state=SnapshotHeadState.ACTIVE,
            snapshot_id=candidate_head.snapshot_id,
        ),
    )


def rollback_active_head(
    *,
    active_head: SnapshotHead,
    rollback_to_snapshot_id: str,
    reason: str,
) -> SnapshotRollback:
    if active_head.state is not SnapshotHeadState.ACTIVE:
        raise ValueError("rollback requires an active head")

    cleaned_target = _clean_text(rollback_to_snapshot_id, field_name="rollback_to_snapshot_id")

    return SnapshotRollback(
        previous_active_snapshot_id=active_head.snapshot_id,
        rollback_to_snapshot_id=cleaned_target,
        reason=reason,
        new_active_head=SnapshotHead(
            head_key=active_head.head_key,
            state=SnapshotHeadState.ACTIVE,
            snapshot_id=cleaned_target,
        ),
    )
