"""Snapshot consistency invariants for Continuity."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from continuity.arbiter import ArbiterPublication, ArbiterPublicationKind, MutationArbiter
from continuity.events import EventPayloadMode
from continuity.transactions import DurabilityWaterline, TransactionKind, TransactionPhase

if TYPE_CHECKING:
    from continuity.events import SystemEvent


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


@dataclass(frozen=True, slots=True)
class MaterializedSnapshot:
    head_key: str
    base_snapshot: MemorySnapshot
    candidate_snapshot: MemorySnapshot
    candidate_head: SnapshotHead
    diff: SnapshotDiff

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_key", _clean_text(self.head_key, field_name="head_key"))
        if self.candidate_head.state is not SnapshotHeadState.CANDIDATE:
            raise ValueError("materialized snapshots require a candidate head")
        if self.candidate_head.head_key != self.head_key:
            raise ValueError("candidate_head must match the materialized head_key")
        if self.candidate_head.snapshot_id != self.candidate_snapshot.snapshot_id:
            raise ValueError("candidate_head must point at the materialized snapshot")
        if self.candidate_head.based_on_snapshot_id != self.base_snapshot.snapshot_id:
            raise ValueError("candidate_head must point at the base snapshot it was derived from")
        if self.diff.from_snapshot_id != self.base_snapshot.snapshot_id:
            raise ValueError("snapshot diff must start from the base snapshot")
        if self.diff.to_snapshot_id != self.candidate_snapshot.snapshot_id:
            raise ValueError("snapshot diff must end at the candidate snapshot")


@dataclass(frozen=True, slots=True)
class PublishedSnapshot:
    head_key: str
    materialized: MaterializedSnapshot
    promotion: SnapshotPromotion
    promotion_record: SnapshotPromotionRecord
    publication: ArbiterPublication
    event: SystemEvent

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_key", _clean_text(self.head_key, field_name="head_key"))
        if self.materialized.head_key != self.head_key:
            raise ValueError("materialized snapshot head_key must match the published head_key")
        if self.promotion.new_active_head.head_key != self.head_key:
            raise ValueError("published promotions must target the same head_key")
        if self.promotion.previous_active_snapshot_id != self.materialized.base_snapshot.snapshot_id:
            raise ValueError("published promotions must use the materialized base snapshot")
        if self.promotion.promoted_snapshot_id != self.materialized.candidate_snapshot.snapshot_id:
            raise ValueError("published promotions must promote the materialized candidate snapshot")
        if self.promotion_record.head_key != self.head_key:
            raise ValueError("promotion_record must match the published head_key")
        if self.promotion_record.promoted_snapshot_id != self.promotion.promoted_snapshot_id:
            raise ValueError("promotion_record must capture the promoted snapshot")
        if self.publication.publication_kind is not ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION:
            raise ValueError("published snapshots require a snapshot head promotion publication")
        if self.publication.phase is not TransactionPhase.PUBLISH_SNAPSHOT:
            raise ValueError("published snapshots must publish during publish_snapshot")
        if self.publication.reached_waterline is not DurabilityWaterline.SNAPSHOT_PUBLISHED:
            raise ValueError("published snapshots must reach snapshot_published")


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
            self._upsert_head_locked(head)

    def delete_head(self, *, head_key: str, state: SnapshotHeadState) -> None:
        with self._connection:
            self._delete_head_locked(head_key=head_key, state=state)

    def read_head(
        self,
        *,
        head_key: str,
        state: SnapshotHeadState,
    ) -> SnapshotHead | None:
        return self._read_head_locked(head_key=head_key, state=state)

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
            self._record_promotion_locked(promotion_record)

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
        head = self._read_head_locked(head_key=head_key, state=state)
        if head is None:
            return None
        return self.read_snapshot(head.snapshot_id)

    def _upsert_head_locked(self, head: SnapshotHead) -> None:
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

    def _delete_head_locked(
        self,
        *,
        head_key: str,
        state: SnapshotHeadState,
    ) -> None:
        self._connection.execute(
            """
            DELETE FROM snapshot_heads
            WHERE head_key = ? AND state = ?
            """,
            (
                _clean_text(head_key, field_name="head_key"),
                state.value,
            ),
        )

    def _read_head_locked(
        self,
        *,
        head_key: str,
        state: SnapshotHeadState,
    ) -> SnapshotHead | None:
        row = self._connection.execute(
            """
            SELECT head_key, state, snapshot_id, based_on_snapshot_id
            FROM snapshot_heads
            WHERE head_key = ? AND state = ?
            """,
            (
                _clean_text(head_key, field_name="head_key"),
                state.value,
            ),
        ).fetchone()
        if row is None:
            return None
        return SnapshotHead(
            head_key=row[0],
            state=SnapshotHeadState(row[1]),
            snapshot_id=row[2],
            based_on_snapshot_id=row[3],
        )

    def _record_promotion_locked(self, promotion_record: SnapshotPromotionRecord) -> None:
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


class SnapshotRuntime:
    """Materialize and publish coherent snapshot heads without leaking mixed read state."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        arbiter: MutationArbiter | None = None,
        head_key: str = "current",
    ) -> None:
        self._connection = connection
        self._snapshots = SnapshotRepository(connection)
        self._arbiter = MutationArbiter(connection) if arbiter is None else arbiter
        self._head_key = _clean_text(head_key, field_name="head_key")

    @property
    def arbiter(self) -> MutationArbiter:
        return self._arbiter

    def materialize_candidate(
        self,
        *,
        candidate_snapshot_id: str,
        policy_stamp: str,
        created_by_transaction: TransactionKind,
        added_artifacts: tuple[SnapshotArtifactRef, ...] | list[SnapshotArtifactRef],
        removed_artifact_ids: tuple[str, ...] | list[str] = (),
        base_snapshot_id: str | None = None,
    ) -> MaterializedSnapshot:
        base_snapshot = self._resolve_base_snapshot(base_snapshot_id=base_snapshot_id)
        cleaned_removed_ids = tuple(
            dict.fromkeys(
                _clean_text(artifact_id, field_name="removed_artifact_ids")
                for artifact_id in removed_artifact_ids
            )
        )
        cleaned_added_artifacts = tuple(added_artifacts)

        candidate_artifact_refs = [
            artifact_ref
            for artifact_ref in base_snapshot.artifact_refs
            if artifact_ref.artifact_id not in cleaned_removed_ids
        ]
        candidate_artifact_refs.extend(cleaned_added_artifacts)

        candidate_snapshot = MemorySnapshot(
            snapshot_id=_clean_text(candidate_snapshot_id, field_name="candidate_snapshot_id"),
            policy_stamp=_clean_text(policy_stamp, field_name="policy_stamp"),
            parent_snapshot_id=base_snapshot.snapshot_id,
            created_by_transaction=created_by_transaction,
            artifact_refs=tuple(dict.fromkeys(candidate_artifact_refs)),
        )
        candidate_head = SnapshotHead(
            head_key=self._head_key,
            state=SnapshotHeadState.CANDIDATE,
            snapshot_id=candidate_snapshot.snapshot_id,
            based_on_snapshot_id=base_snapshot.snapshot_id,
        )
        self._snapshots.save_snapshot(candidate_snapshot)
        self._snapshots.upsert_head(candidate_head)

        return MaterializedSnapshot(
            head_key=self._head_key,
            base_snapshot=base_snapshot,
            candidate_snapshot=candidate_snapshot,
            candidate_head=candidate_head,
            diff=diff_snapshots(base_snapshot, candidate_snapshot),
        )

    def promote_candidate(
        self,
        *,
        promotion_id: str,
        published_at: datetime,
    ) -> PublishedSnapshot:
        candidate_head = self._snapshots.read_head(
            head_key=self._head_key,
            state=SnapshotHeadState.CANDIDATE,
        )
        if candidate_head is None:
            raise ValueError("promotion requires an existing candidate head")

        base_head = self._resolve_active_head_for_candidate(candidate_head)
        candidate_snapshot = self._require_snapshot(candidate_head.snapshot_id)
        base_snapshot = self._require_snapshot(base_head.snapshot_id)
        materialized = MaterializedSnapshot(
            head_key=self._head_key,
            base_snapshot=base_snapshot,
            candidate_snapshot=candidate_snapshot,
            candidate_head=candidate_head,
            diff=diff_snapshots(base_snapshot, candidate_snapshot),
        )
        promotion = promote_candidate_head(
            active_head=base_head,
            candidate_head=candidate_head,
        )
        promotion_record = SnapshotPromotionRecord(
            promotion_id=_clean_text(promotion_id, field_name="promotion_id"),
            head_key=self._head_key,
            previous_active_snapshot_id=promotion.previous_active_snapshot_id,
            promoted_snapshot_id=promotion.promoted_snapshot_id,
            recorded_at=_validate_timestamp(published_at, field_name="published_at"),
        )

        published_mutation = self._arbiter.publish(
            publication_kind=ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,
            transaction_kind=TransactionKind.PUBLISH_SNAPSHOT,
            phase=TransactionPhase.PUBLISH_SNAPSHOT,
            object_ids=(
                promotion.previous_active_snapshot_id,
                promotion.promoted_snapshot_id,
            ),
            published_at=promotion_record.recorded_at,
            payload_mode=EventPayloadMode.REFERENCE,
            reference_ids=(promotion.promoted_snapshot_id,),
            snapshot_head_id=f"head:{self._head_key}",
            reached_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
            before_commit=lambda: self._commit_promotion_locked(
                candidate_head=candidate_head,
                expected_base_snapshot_id=base_snapshot.snapshot_id,
                active_head=promotion.new_active_head,
                promotion_record=promotion_record,
            ),
        )

        return PublishedSnapshot(
            head_key=self._head_key,
            materialized=materialized,
            promotion=promotion,
            promotion_record=promotion_record,
            publication=published_mutation.publication,
            event=published_mutation.event,
        )

    def _resolve_base_snapshot(self, *, base_snapshot_id: str | None) -> MemorySnapshot:
        if base_snapshot_id is None:
            active_snapshot = self._snapshots.read_active_snapshot(head_key=self._head_key)
            if active_snapshot is None:
                raise ValueError("materializing a candidate snapshot requires an active base snapshot")
            return active_snapshot

        return self._require_snapshot(base_snapshot_id)

    def _resolve_active_head_for_candidate(self, candidate_head: SnapshotHead) -> SnapshotHead:
        active_head = self._snapshots.read_head(
            head_key=self._head_key,
            state=SnapshotHeadState.ACTIVE,
        )
        if active_head is not None:
            return active_head

        assert candidate_head.based_on_snapshot_id is not None
        self._require_snapshot(candidate_head.based_on_snapshot_id)
        return SnapshotHead(
            head_key=self._head_key,
            state=SnapshotHeadState.ACTIVE,
            snapshot_id=candidate_head.based_on_snapshot_id,
        )

    def _commit_promotion_locked(
        self,
        *,
        candidate_head: SnapshotHead,
        expected_base_snapshot_id: str,
        active_head: SnapshotHead,
        promotion_record: SnapshotPromotionRecord,
    ) -> None:
        current_candidate = self._snapshots._read_head_locked(
            head_key=self._head_key,
            state=SnapshotHeadState.CANDIDATE,
        )
        if current_candidate != candidate_head:
            raise ValueError("candidate head changed before promotion")

        current_active = self._snapshots._read_head_locked(
            head_key=self._head_key,
            state=SnapshotHeadState.ACTIVE,
        )
        if current_active is None:
            if candidate_head.based_on_snapshot_id != expected_base_snapshot_id:
                raise ValueError("candidate head base snapshot changed before promotion")
            self._require_snapshot(expected_base_snapshot_id)
        elif current_active.snapshot_id != expected_base_snapshot_id:
            raise ValueError("active snapshot changed before promotion")

        self._snapshots._upsert_head_locked(active_head)
        self._snapshots._delete_head_locked(
            head_key=self._head_key,
            state=SnapshotHeadState.CANDIDATE,
        )
        self._snapshots._record_promotion_locked(promotion_record)

    def _require_snapshot(self, snapshot_id: str) -> MemorySnapshot:
        cleaned_snapshot_id = _clean_text(snapshot_id, field_name="snapshot_id")
        snapshot = self._snapshots.read_snapshot(cleaned_snapshot_id)
        if snapshot is None:
            raise ValueError(f"unknown snapshot: {cleaned_snapshot_id}")
        return snapshot


def diff_snapshots(from_snapshot: MemorySnapshot, to_snapshot: MemorySnapshot) -> SnapshotDiff:
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
