"""System event journal invariants for authoritative Continuity publications."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.arbiter import ArbiterPublication, ArbiterPublicationKind
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    transaction_contract_for,
)


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _clean_deduped(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    cleaned = tuple(_clean_text(value, field_name=field_name) for value in values)
    deduped = tuple(dict.fromkeys(cleaned))
    if not deduped:
        raise ValueError(f"{field_name} must be non-empty")
    return deduped


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    return _validate_timestamp(datetime.fromisoformat(value), field_name=field_name)


def _dump_text_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(list(values))


def _load_text_tuple(
    value: str,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError(f"{field_name} must be a JSON array")
    if not loaded:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} must be non-empty")
    return _clean_deduped(tuple(loaded), field_name=field_name)


class SystemEventType(StrEnum):
    OBSERVATION_INGESTED = "observation_ingested"
    CLAIM_COMMITTED = "claim_committed"
    BELIEF_REVISED = "belief_revised"
    MEMORY_FORGOTTEN = "memory_forgotten"
    VIEW_COMPILED = "view_compiled"
    SNAPSHOT_PUBLISHED = "snapshot_published"
    OUTCOME_RECORDED = "outcome_recorded"

    @property
    def expected_publication_kind(self) -> ArbiterPublicationKind:
        return {
            SystemEventType.OBSERVATION_INGESTED: ArbiterPublicationKind.OBSERVATION_COMMIT,
            SystemEventType.CLAIM_COMMITTED: ArbiterPublicationKind.CLAIM_COMMIT,
            SystemEventType.BELIEF_REVISED: ArbiterPublicationKind.BELIEF_REVISION,
            SystemEventType.MEMORY_FORGOTTEN: ArbiterPublicationKind.FORGETTING_PUBLICATION,
            SystemEventType.VIEW_COMPILED: ArbiterPublicationKind.VIEW_PUBLICATION,
            SystemEventType.SNAPSHOT_PUBLISHED: ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION,
            SystemEventType.OUTCOME_RECORDED: ArbiterPublicationKind.OUTCOME_RECORDING,
        }[self]


class EventPayloadMode(StrEnum):
    INLINE = "inline"
    REFERENCE = "reference"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class SystemEvent:
    journal_position: int
    event_type: SystemEventType
    transaction_kind: TransactionKind
    arbiter_lane_position: int
    payload_mode: EventPayloadMode
    recorded_at: datetime
    object_ids: tuple[str, ...]
    inline_payload: tuple[str, ...] = ()
    reference_ids: tuple[str, ...] = ()
    waterline: DurabilityWaterline | None = None

    def __post_init__(self) -> None:
        if self.journal_position <= 0:
            raise ValueError("journal_position must be positive")
        if self.arbiter_lane_position <= 0:
            raise ValueError("arbiter_lane_position must be positive")

        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )
        object.__setattr__(
            self,
            "object_ids",
            _clean_deduped(self.object_ids, field_name="object_ids"),
        )
        object.__setattr__(
            self,
            "inline_payload",
            tuple(
                dict.fromkeys(
                    _clean_text(value, field_name="inline_payload")
                    for value in self.inline_payload
                )
            ),
        )
        object.__setattr__(
            self,
            "reference_ids",
            tuple(
                dict.fromkeys(
                    _clean_text(value, field_name="reference_ids")
                    for value in self.reference_ids
                )
            ),
        )

        if self.payload_mode is EventPayloadMode.INLINE:
            if not self.inline_payload or self.reference_ids:
                raise ValueError("inline payload mode requires only inline_payload")
        elif self.payload_mode is EventPayloadMode.REFERENCE:
            if self.inline_payload or not self.reference_ids:
                raise ValueError("reference payload mode requires only reference_ids")
        elif not self.inline_payload or not self.reference_ids:
            raise ValueError("mixed payload mode requires inline_payload and reference_ids")

        if self.waterline is not None and not transaction_contract_for(
            self.transaction_kind
        ).supports_waterline(self.waterline):
            raise ValueError(
                f"{self.transaction_kind.value} cannot reach {self.waterline.value}"
            )

    @classmethod
    def from_publication(
        cls,
        *,
        journal_position: int,
        event_type: SystemEventType,
        publication: ArbiterPublication,
        payload_mode: EventPayloadMode,
        recorded_at: datetime,
        object_ids: tuple[str, ...] = (),
        inline_payload: tuple[str, ...] = (),
        reference_ids: tuple[str, ...] = (),
    ) -> "SystemEvent":
        if publication.publication_kind is not event_type.expected_publication_kind:
            raise ValueError(
                f"{event_type.value} must link to {event_type.expected_publication_kind.value}"
            )
        return cls(
            journal_position=journal_position,
            event_type=event_type,
            transaction_kind=publication.transaction_kind,
            arbiter_lane_position=publication.lane_position,
            payload_mode=payload_mode,
            recorded_at=recorded_at,
            object_ids=object_ids or publication.object_ids,
            inline_payload=inline_payload,
            reference_ids=reference_ids,
            waterline=publication.reached_waterline,
        )

    @property
    def append_only_key(self) -> tuple[int, int]:
        return (self.journal_position, self.arbiter_lane_position)

    @property
    def has_inline_payload(self) -> bool:
        return bool(self.inline_payload)

    @property
    def has_reference_payload(self) -> bool:
        return bool(self.reference_ids)


class SystemEventJournal:
    """Append-only persistence and ordered lookup for authoritative system events."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    def next_position(self) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(journal_position), 0) + 1 AS next_journal_position
            FROM system_events
            """
        ).fetchone()
        return int(row["next_journal_position"])

    def append(self, event: SystemEvent) -> None:
        with self._connection:
            self._append_locked(event)

    def read_event(self, journal_position: int) -> SystemEvent | None:
        row = self._connection.execute(
            """
            SELECT
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
            FROM system_events
            WHERE journal_position = ?
            """,
            (journal_position,),
        ).fetchone()
        if row is None:
            return None
        return self._event_from_row(row)

    def list_events(
        self,
        *,
        event_type: SystemEventType | None = None,
        transaction_kind: TransactionKind | None = None,
        arbiter_lane_position: int | None = None,
        waterline: DurabilityWaterline | None = None,
        object_id: str | None = None,
        from_position: int | None = None,
        to_position: int | None = None,
        limit: int | None = None,
    ) -> tuple[SystemEvent, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        clauses: list[str] = []
        parameters: list[object] = []
        if event_type is not None:
            clauses.append("event_type = ?")
            parameters.append(event_type.value)
        if transaction_kind is not None:
            clauses.append("transaction_kind = ?")
            parameters.append(transaction_kind.value)
        if arbiter_lane_position is not None:
            clauses.append("arbiter_lane_position = ?")
            parameters.append(arbiter_lane_position)
        if waterline is not None:
            clauses.append("waterline = ?")
            parameters.append(waterline.value)
        if from_position is not None:
            clauses.append("journal_position >= ?")
            parameters.append(from_position)
        if to_position is not None:
            clauses.append("journal_position <= ?")
            parameters.append(to_position)

        query = """
            SELECT
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
            FROM system_events
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY journal_position, arbiter_lane_position"
        rows = self._connection.execute(query, parameters).fetchall()

        events = tuple(self._event_from_row(row) for row in rows)
        if object_id is not None:
            events = tuple(event for event in events if object_id in event.object_ids)
        if limit is not None:
            events = events[:limit]
        return events

    def reconstruct(
        self,
        *,
        object_id: str | None = None,
        arbiter_lane_position: int | None = None,
        from_position: int | None = None,
        to_position: int | None = None,
    ) -> tuple[SystemEvent, ...]:
        return self.list_events(
            object_id=object_id,
            arbiter_lane_position=arbiter_lane_position,
            from_position=from_position,
            to_position=to_position,
        )

    def _append_locked(self, event: SystemEvent) -> None:
        existing = self.read_event(event.journal_position)
        if existing is not None:
            if existing != event:
                raise ValueError("system events are append-only")
            return

        self._connection.execute(
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
                event.journal_position,
                event.event_type.value,
                event.transaction_kind.value,
                event.arbiter_lane_position,
                event.payload_mode.value,
                event.recorded_at.isoformat(),
                _dump_text_tuple(event.object_ids),
                _dump_text_tuple(event.inline_payload),
                _dump_text_tuple(event.reference_ids),
                event.waterline.value if event.waterline is not None else None,
            ),
        )

    def _event_from_row(self, row: sqlite3.Row) -> SystemEvent:
        waterline = (
            None if row["waterline"] is None else DurabilityWaterline(row["waterline"])
        )
        return SystemEvent(
            journal_position=row["journal_position"],
            event_type=SystemEventType(row["event_type"]),
            transaction_kind=TransactionKind(row["transaction_kind"]),
            arbiter_lane_position=row["arbiter_lane_position"],
            payload_mode=EventPayloadMode(row["payload_mode"]),
            recorded_at=_parse_timestamp(row["recorded_at"], field_name="recorded_at"),
            object_ids=_load_text_tuple(row["object_ids_json"], field_name="object_ids"),
            inline_payload=_load_text_tuple(
                row["inline_payload_json"],
                field_name="inline_payload",
                allow_empty=True,
            ),
            reference_ids=_load_text_tuple(
                row["reference_ids_json"],
                field_name="reference_ids",
                allow_empty=True,
            ),
            waterline=waterline,
        )
