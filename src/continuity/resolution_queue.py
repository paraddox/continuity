"""Resolution queue invariants for unresolved Continuity memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.store.claims import AdmissionOutcome


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


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

