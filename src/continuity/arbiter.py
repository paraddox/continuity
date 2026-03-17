"""Mutation arbiter invariants for serialized authoritative publication."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from collections.abc import Callable

from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
    transaction_contract_for,
)

if TYPE_CHECKING:
    from continuity.events import EventPayloadMode, SystemEvent, SystemEventJournal, SystemEventType


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


def _load_text_tuple(value: str, *, field_name: str) -> tuple[str, ...]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return _clean_deduped(tuple(loaded), field_name=field_name)


class OffLaneWork(StrEnum):
    EMBEDDING_GENERATION = "embedding_generation"
    CLAIM_DERIVATION = "claim_derivation"
    VIEW_COMPILATION = "view_compilation"
    PREFETCH_PREPARATION = "prefetch_preparation"


class ArbiterPublicationKind(StrEnum):
    OBSERVATION_COMMIT = "observation_commit"
    CLAIM_COMMIT = "claim_commit"
    BELIEF_REVISION = "belief_revision"
    FORGETTING_PUBLICATION = "forgetting_publication"
    VIEW_PUBLICATION = "view_publication"
    WORK_STATUS_TRANSITION = "work_status_transition"
    SNAPSHOT_HEAD_PROMOTION = "snapshot_head_promotion"
    DURABILITY_SIGNAL = "durability_signal"
    OUTCOME_RECORDING = "outcome_recording"


@dataclass(frozen=True, slots=True)
class ArbiterPublication:
    lane_position: int
    publication_kind: ArbiterPublicationKind
    transaction_kind: TransactionKind
    phase: TransactionPhase
    object_ids: tuple[str, ...]
    published_at: datetime
    snapshot_head_id: str | None = None
    reached_waterline: DurabilityWaterline | None = None

    def __post_init__(self) -> None:
        if self.lane_position <= 0:
            raise ValueError("lane_position must be positive")

        object.__setattr__(
            self,
            "object_ids",
            _clean_deduped(self.object_ids, field_name="object_ids"),
        )
        object.__setattr__(
            self,
            "published_at",
            _validate_timestamp(self.published_at, field_name="published_at"),
        )

        if self.snapshot_head_id is not None:
            object.__setattr__(
                self,
                "snapshot_head_id",
                _clean_text(self.snapshot_head_id, field_name="snapshot_head_id"),
            )

        contract = transaction_contract_for(self.transaction_kind)
        if self.phase not in contract.phases:
            raise ValueError(
                f"{self.phase.value} is not part of {self.transaction_kind.value}"
            )

        if self.reached_waterline is not None and not contract.supports_waterline(
            self.reached_waterline
        ):
            raise ValueError(
                f"{self.transaction_kind.value} cannot reach {self.reached_waterline.value}"
            )

        if self.publication_kind is ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION:
            if self.phase is not TransactionPhase.PUBLISH_SNAPSHOT:
                raise ValueError(
                    "snapshot head promotion must occur during publish_snapshot"
                )
            if self.snapshot_head_id is None:
                raise ValueError("snapshot head promotion requires snapshot_head_id")
            if self.reached_waterline is not DurabilityWaterline.SNAPSHOT_PUBLISHED:
                raise ValueError(
                    "snapshot head promotion must carry snapshot_published waterline"
                )
        elif self.snapshot_head_id is not None:
            raise ValueError(
                "snapshot_head_id is only valid for snapshot head promotion publications"
            )

        if (
            self.publication_kind is ArbiterPublicationKind.DURABILITY_SIGNAL
            and self.reached_waterline is None
        ):
            raise ValueError("durability signals require a reached_waterline")

    @property
    def replay_order_key(self) -> str:
        return f"arbiter:{self.lane_position}"

    @property
    def requires_journal_entry(self) -> bool:
        return True

    @property
    def is_snapshot_publication(self) -> bool:
        return self.publication_kind is ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION


@dataclass(frozen=True, slots=True)
class PublishedMutation:
    publication: ArbiterPublication
    event: "SystemEvent"


def _event_type_for_publication_kind(
    publication_kind: ArbiterPublicationKind,
) -> "SystemEventType":
    from continuity.events import SystemEventType

    mapping = {
        ArbiterPublicationKind.OBSERVATION_COMMIT: SystemEventType.OBSERVATION_INGESTED,
        ArbiterPublicationKind.CLAIM_COMMIT: SystemEventType.CLAIM_COMMITTED,
        ArbiterPublicationKind.BELIEF_REVISION: SystemEventType.BELIEF_REVISED,
        ArbiterPublicationKind.FORGETTING_PUBLICATION: SystemEventType.MEMORY_FORGOTTEN,
        ArbiterPublicationKind.VIEW_PUBLICATION: SystemEventType.VIEW_COMPILED,
        ArbiterPublicationKind.SNAPSHOT_HEAD_PROMOTION: SystemEventType.SNAPSHOT_PUBLISHED,
        ArbiterPublicationKind.OUTCOME_RECORDING: SystemEventType.OUTCOME_RECORDED,
    }
    try:
        return mapping[publication_kind]
    except KeyError as exc:
        raise ValueError(
            f"{publication_kind.value} has no supported v1 system event mapping"
        ) from exc


class MutationArbiter:
    """Serialize authoritative publications through one in-process commit lane."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        journal: "SystemEventJournal | None" = None,
    ) -> None:
        from continuity.events import SystemEventJournal

        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._journal = SystemEventJournal(connection) if journal is None else journal
        self._lane_lock = threading.Lock()

    @property
    def journal(self) -> "SystemEventJournal":
        return self._journal

    def publish(
        self,
        *,
        publication_kind: ArbiterPublicationKind,
        transaction_kind: TransactionKind,
        phase: TransactionPhase,
        object_ids: tuple[str, ...],
        published_at: datetime,
        payload_mode: "EventPayloadMode | None" = None,
        event_type: "SystemEventType | None" = None,
        recorded_at: datetime | None = None,
        event_object_ids: tuple[str, ...] = (),
        inline_payload: tuple[str, ...] = (),
        reference_ids: tuple[str, ...] = (),
        snapshot_head_id: str | None = None,
        reached_waterline: DurabilityWaterline | None = None,
        before_commit: Callable[[], None] | None = None,
    ) -> PublishedMutation:
        from continuity.events import EventPayloadMode, SystemEvent

        effective_payload_mode = (
            EventPayloadMode.REFERENCE if payload_mode is None else payload_mode
        )
        effective_recorded_at = published_at if recorded_at is None else recorded_at
        effective_reference_ids = reference_ids
        if (
            effective_payload_mode in {EventPayloadMode.REFERENCE, EventPayloadMode.MIXED}
            and not effective_reference_ids
        ):
            effective_reference_ids = tuple(object_ids)
        effective_event_type = (
            _event_type_for_publication_kind(publication_kind)
            if event_type is None
            else event_type
        )

        with self._lane_lock:
            publication = ArbiterPublication(
                lane_position=self.next_lane_position(),
                publication_kind=publication_kind,
                transaction_kind=transaction_kind,
                phase=phase,
                object_ids=object_ids,
                published_at=published_at,
                snapshot_head_id=snapshot_head_id,
                reached_waterline=reached_waterline,
            )
            event = SystemEvent.from_publication(
                journal_position=self._journal.next_position(),
                event_type=effective_event_type,
                publication=publication,
                payload_mode=effective_payload_mode,
                recorded_at=effective_recorded_at,
                object_ids=event_object_ids or publication.object_ids,
                inline_payload=inline_payload,
                reference_ids=effective_reference_ids,
            )
            with self._connection:
                if before_commit is not None:
                    before_commit()
                self._insert_publication_locked(publication)
                self._journal._append_locked(event)
        return PublishedMutation(publication=publication, event=event)

    def next_lane_position(self) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(lane_position), 0) + 1 AS next_lane_position
            FROM arbiter_publications
            """
        ).fetchone()
        return int(row["next_lane_position"])

    def read_publication(self, lane_position: int) -> ArbiterPublication | None:
        row = self._connection.execute(
            """
            SELECT
                lane_position,
                publication_kind,
                transaction_kind,
                phase,
                object_ids_json,
                published_at,
                snapshot_head_id,
                reached_waterline
            FROM arbiter_publications
            WHERE lane_position = ?
            """,
            (lane_position,),
        ).fetchone()
        if row is None:
            return None
        return self._publication_from_row(row)

    def list_publications(
        self,
        *,
        publication_kind: ArbiterPublicationKind | None = None,
        transaction_kind: TransactionKind | None = None,
        phase: TransactionPhase | None = None,
        reached_waterline: DurabilityWaterline | None = None,
        object_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[ArbiterPublication, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        clauses: list[str] = []
        parameters: list[object] = []
        if publication_kind is not None:
            clauses.append("publication_kind = ?")
            parameters.append(publication_kind.value)
        if transaction_kind is not None:
            clauses.append("transaction_kind = ?")
            parameters.append(transaction_kind.value)
        if phase is not None:
            clauses.append("phase = ?")
            parameters.append(phase.value)
        if reached_waterline is not None:
            clauses.append("reached_waterline = ?")
            parameters.append(reached_waterline.value)

        query = """
            SELECT
                lane_position,
                publication_kind,
                transaction_kind,
                phase,
                object_ids_json,
                published_at,
                snapshot_head_id,
                reached_waterline
            FROM arbiter_publications
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY lane_position"
        rows = self._connection.execute(query, parameters).fetchall()

        publications = tuple(self._publication_from_row(row) for row in rows)
        if object_id is not None:
            publications = tuple(
                publication
                for publication in publications
                if object_id in publication.object_ids
            )
        if limit is not None:
            publications = publications[:limit]
        return publications

    def _insert_publication_locked(self, publication: ArbiterPublication) -> None:
        existing = self.read_publication(publication.lane_position)
        if existing is not None:
            if existing != publication:
                raise ValueError("arbiter publications are append-only")
            return

        self._connection.execute(
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
                publication.lane_position,
                publication.publication_kind.value,
                publication.transaction_kind.value,
                publication.phase.value,
                _dump_text_tuple(publication.object_ids),
                publication.published_at.isoformat(),
                publication.snapshot_head_id,
                (
                    publication.reached_waterline.value
                    if publication.reached_waterline is not None
                    else None
                ),
            ),
        )

    def _publication_from_row(self, row: sqlite3.Row) -> ArbiterPublication:
        reached_waterline = (
            None
            if row["reached_waterline"] is None
            else DurabilityWaterline(row["reached_waterline"])
        )
        return ArbiterPublication(
            lane_position=row["lane_position"],
            publication_kind=ArbiterPublicationKind(row["publication_kind"]),
            transaction_kind=TransactionKind(row["transaction_kind"]),
            phase=TransactionPhase(row["phase"]),
            object_ids=_load_text_tuple(row["object_ids_json"], field_name="object_ids"),
            published_at=_parse_timestamp(row["published_at"], field_name="published_at"),
            snapshot_head_id=row["snapshot_head_id"],
            reached_waterline=reached_waterline,
        )
