"""Resolution queue invariants for unresolved Continuity memory."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.store.claims import AdmissionOutcome


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


class ResolutionSource(StrEnum):
    NEEDS_CONFIRMATION = "needs_confirmation"
    NEEDS_FOLLOWUP = "needs_followup"
    OPEN_QUESTION = "open_question"
    STALE_ON_USE = "stale_on_use"
    CONFLICTED_LOCUS = "conflicted_locus"


class ResolutionPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    IMMEDIATE = "immediate"

    @property
    def score(self) -> int:
        return {
            ResolutionPriority.LOW: 1,
            ResolutionPriority.NORMAL: 2,
            ResolutionPriority.HIGH: 3,
            ResolutionPriority.IMMEDIATE: 4,
        }[self]


class ResolutionStatus(StrEnum):
    OPEN = "open"
    DEFERRED = "deferred"
    RESOLVED = "resolved"
    DROPPED = "dropped"


class ResolutionSurface(StrEnum):
    PROMPT_QUEUE = "prompt_queue"
    HOST_API = "host_api"
    INSPECTION = "inspection"


class ResolutionAction(StrEnum):
    CONFIRM = "confirm"
    CORRECT = "correct"
    DISCARD = "discard"
    KEEP_EPHEMERAL = "keep_ephemeral"
    PROMOTE_TO_DURABLE_CLAIM = "promote_to_durable_claim"


class ResolutionEffect(StrEnum):
    ADMISSION = "admission"
    BELIEF_REVISION = "belief_revision"
    OUTCOME_RECORDING = "outcome_recording"
    REPLAY_CAPTURE = "replay_capture"


@dataclass(frozen=True, slots=True)
class ResolutionQueueItem:
    item_id: str
    source: ResolutionSource
    priority: ResolutionPriority
    subject_id: str
    locus_key: str
    rationale: str
    created_at: datetime
    utility_boost: int = 0
    status: ResolutionStatus = ResolutionStatus.OPEN
    surfaces: tuple[ResolutionSurface, ...] = (ResolutionSurface.HOST_API,)
    deferred_until: datetime | None = None
    batch_key: str | None = None
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _clean_text(self.item_id, field_name="item_id"))
        object.__setattr__(self, "subject_id", _clean_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_text(self.locus_key, field_name="locus_key"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "created_at",
            _validate_timestamp(self.created_at, field_name="created_at"),
        )
        if self.deferred_until is not None:
            object.__setattr__(
                self,
                "deferred_until",
                _validate_timestamp(self.deferred_until, field_name="deferred_until"),
            )
        if not self.surfaces:
            raise ValueError("surfaces must be non-empty")
        if self.status is ResolutionStatus.DEFERRED and self.deferred_until is None:
            raise ValueError("deferred items require deferred_until")
        if self.status is not ResolutionStatus.DEFERRED and self.deferred_until is not None:
            raise ValueError("deferred_until is only valid for deferred items")
        if self.batch_key is not None:
            object.__setattr__(self, "batch_key", _clean_text(self.batch_key, field_name="batch_key"))
        if self.candidate_id is not None:
            object.__setattr__(
                self,
                "candidate_id",
                _clean_text(self.candidate_id, field_name="candidate_id"),
            )

    @property
    def publishes_claim(self) -> bool:
        return False

    @property
    def surfaces_in_prompt(self) -> bool:
        return ResolutionSurface.PROMPT_QUEUE in self.surfaces

    @property
    def surfaces_via_host_api(self) -> bool:
        return ResolutionSurface.HOST_API in self.surfaces

    def is_actionable(self, at_time: datetime) -> bool:
        current_time = _validate_timestamp(at_time, field_name="at_time")
        if self.status in {ResolutionStatus.RESOLVED, ResolutionStatus.DROPPED}:
            return False
        if self.status is ResolutionStatus.DEFERRED and self.deferred_until is not None:
            return current_time >= self.deferred_until
        return True

    def priority_key(self, at_time: datetime) -> tuple[int, int, int, datetime]:
        current_time = _validate_timestamp(at_time, field_name="at_time")
        if self.status is ResolutionStatus.RESOLVED:
            return (3, 0, 0, self.created_at)
        if self.status is ResolutionStatus.DROPPED:
            return (4, 0, 0, self.created_at)
        if (
            self.status is ResolutionStatus.DEFERRED
            and self.deferred_until is not None
            and current_time < self.deferred_until
        ):
            return (2, -self.priority.score, -self.utility_boost, self.deferred_until)
        return (0, -self.priority.score, -self.utility_boost, self.created_at)


@dataclass(frozen=True, slots=True)
class ResolutionRecord:
    item_id: str
    action: ResolutionAction
    rationale: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _clean_text(self.item_id, field_name="item_id"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )

    @property
    def resulting_admission_outcome(self) -> AdmissionOutcome:
        return {
            ResolutionAction.CONFIRM: AdmissionOutcome.DURABLE_CLAIM,
            ResolutionAction.CORRECT: AdmissionOutcome.NEEDS_CONFIRMATION,
            ResolutionAction.DISCARD: AdmissionOutcome.DISCARD,
            ResolutionAction.KEEP_EPHEMERAL: AdmissionOutcome.SESSION_EPHEMERAL,
            ResolutionAction.PROMOTE_TO_DURABLE_CLAIM: AdmissionOutcome.DURABLE_CLAIM,
        }[self.action]

    @property
    def resulting_status(self) -> ResolutionStatus:
        if self.action is ResolutionAction.DISCARD:
            return ResolutionStatus.DROPPED
        return ResolutionStatus.RESOLVED

    @property
    def effects(self) -> frozenset[ResolutionEffect]:
        effects = {
            ResolutionEffect.ADMISSION,
            ResolutionEffect.OUTCOME_RECORDING,
            ResolutionEffect.REPLAY_CAPTURE,
        }
        if self.action in {
            ResolutionAction.CONFIRM,
            ResolutionAction.CORRECT,
            ResolutionAction.PROMOTE_TO_DURABLE_CLAIM,
        }:
            effects.add(ResolutionEffect.BELIEF_REVISION)
        return frozenset(effects)


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


class ResolutionQueueRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def enqueue_item(self, item: ResolutionQueueItem) -> None:
        with self._connection:
            self._connection.execute(
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
                    item.item_id,
                    item.source.value,
                    item.priority.value,
                    item.subject_id,
                    item.locus_key,
                    item.rationale,
                    item.created_at.isoformat(),
                    item.utility_boost,
                    item.status.value,
                    json.dumps([surface.value for surface in item.surfaces]),
                    item.deferred_until.isoformat() if item.deferred_until is not None else None,
                    item.batch_key,
                    item.candidate_id,
                ),
            )

    def read_item(self, item_id: str) -> ResolutionQueueItem | None:
        row = self._connection.execute(
            """
            SELECT
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
            FROM resolution_queue_items
            WHERE item_id = ?
            """,
            (_clean_text(item_id, field_name="item_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._item_from_row(row)

    def list_items(
        self,
        *,
        subject_id: str | None = None,
        status: ResolutionStatus | None = None,
        surface: ResolutionSurface | None = None,
        batch_key: str | None = None,
        at_time: datetime | None = None,
        actionable_only: bool = False,
        limit: int | None = None,
    ) -> tuple[ResolutionQueueItem, ...]:
        conditions: list[str] = []
        params: list[str] = []

        if subject_id is not None:
            conditions.append("subject_id = ?")
            params.append(_clean_text(subject_id, field_name="subject_id"))
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        if batch_key is not None:
            conditions.append("batch_key = ?")
            params.append(_clean_text(batch_key, field_name="batch_key"))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        rows = self._connection.execute(
            f"""
            SELECT
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
            FROM resolution_queue_items
            {where_clause}
            """,
            tuple(params),
        ).fetchall()

        current_time = _validate_timestamp(
            at_time or datetime.now().astimezone(),
            field_name="at_time",
        )
        items = [self._item_from_row(row) for row in rows]
        if surface is not None:
            items = [item for item in items if surface in item.surfaces]
        if actionable_only:
            items = [item for item in items if item.is_actionable(current_time)]

        items.sort(key=lambda item: item.priority_key(current_time))
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            items = items[:limit]
        return tuple(items)

    def defer_item(self, item_id: str, *, until: datetime) -> None:
        deferred_until = _validate_timestamp(until, field_name="until")
        with self._connection:
            self._connection.execute(
                """
                UPDATE resolution_queue_items
                SET status = ?, deferred_until = ?
                WHERE item_id = ?
                """,
                (
                    ResolutionStatus.DEFERRED.value,
                    deferred_until.isoformat(),
                    _clean_text(item_id, field_name="item_id"),
                ),
            )

    def assign_batch(self, item_id: str, *, batch_key: str | None) -> None:
        with self._connection:
            self._connection.execute(
                """
                UPDATE resolution_queue_items
                SET batch_key = ?
                WHERE item_id = ?
                """,
                (
                    _clean_optional_text(batch_key, field_name="batch_key"),
                    _clean_text(item_id, field_name="item_id"),
                ),
            )

    def escalate_priority(
        self,
        item_id: str,
        *,
        priority: ResolutionPriority,
        utility_boost: int | None = None,
    ) -> None:
        params: list[object] = [priority.value]
        updates = ["priority = ?"]
        if utility_boost is not None:
            updates.append("utility_boost = ?")
            params.append(utility_boost)
        params.append(_clean_text(item_id, field_name="item_id"))

        with self._connection:
            self._connection.execute(
                f"""
                UPDATE resolution_queue_items
                SET {", ".join(updates)}
                WHERE item_id = ?
                """,
                tuple(params),
            )

    def record_resolution(self, record: ResolutionRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO resolution_actions(
                    item_id,
                    action,
                    rationale,
                    recorded_at,
                    resulting_admission_outcome,
                    effects_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.item_id,
                    record.action.value,
                    record.rationale,
                    record.recorded_at.isoformat(),
                    record.resulting_admission_outcome.value,
                    json.dumps(sorted(effect.value for effect in record.effects)),
                ),
            )
            self._connection.execute(
                """
                UPDATE resolution_queue_items
                SET status = ?, deferred_until = NULL
                WHERE item_id = ?
                """,
                (
                    record.resulting_status.value,
                    record.item_id,
                ),
            )

    def list_records(self, *, item_id: str) -> tuple[ResolutionRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT item_id, action, rationale, recorded_at
            FROM resolution_actions
            WHERE item_id = ?
            ORDER BY action_id
            """,
            (_clean_text(item_id, field_name="item_id"),),
        ).fetchall()
        return tuple(
            ResolutionRecord(
                item_id=row[0],
                action=ResolutionAction(row[1]),
                rationale=row[2],
                recorded_at=_validate_timestamp(
                    datetime.fromisoformat(row[3]),
                    field_name="recorded_at",
                ),
            )
            for row in rows
        )

    def _item_from_row(self, row: tuple[object, ...]) -> ResolutionQueueItem:
        return ResolutionQueueItem(
            item_id=row[0],  # type: ignore[arg-type]
            source=ResolutionSource(row[1]),  # type: ignore[arg-type]
            priority=ResolutionPriority(row[2]),  # type: ignore[arg-type]
            subject_id=row[3],  # type: ignore[arg-type]
            locus_key=row[4],  # type: ignore[arg-type]
            rationale=row[5],  # type: ignore[arg-type]
            created_at=_validate_timestamp(
                datetime.fromisoformat(row[6]),  # type: ignore[arg-type]
                field_name="created_at",
            ),
            utility_boost=row[7],  # type: ignore[arg-type]
            status=ResolutionStatus(row[8]),  # type: ignore[arg-type]
            surfaces=tuple(
                ResolutionSurface(surface)
                for surface in json.loads(row[9])  # type: ignore[arg-type]
            ),
            deferred_until=_parse_timestamp(row[10]),  # type: ignore[arg-type]
            batch_key=row[11],  # type: ignore[arg-type]
            candidate_id=row[12],  # type: ignore[arg-type]
        )
