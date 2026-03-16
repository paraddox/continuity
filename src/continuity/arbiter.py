"""Mutation arbiter invariants for serialized authoritative publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    TransactionPhase,
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
