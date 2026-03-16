"""System event journal invariants for authoritative Continuity publications."""

from __future__ import annotations

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
