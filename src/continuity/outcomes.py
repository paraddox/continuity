"""Outcome ledger invariants for downstream Continuity feedback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _optional_clean_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _dedupe_cleaned(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    cleaned_values = tuple(_clean_text(value, field_name=field_name) for value in values)
    return tuple(dict.fromkeys(cleaned_values))


class OutcomeLabel(StrEnum):
    PROMPT_INCLUDED = "prompt_included"
    ANSWER_CITED = "answer_cited"
    USER_CONFIRMED = "user_confirmed"
    USER_CORRECTED = "user_corrected"
    STALE_ON_USE = "stale_on_use"


class OutcomeTarget(StrEnum):
    CLAIM = "claim"
    LOCUS = "locus"
    COMPILED_VIEW = "compiled_view"
    ANSWER = "answer"
    RESOLUTION_QUEUE_ITEM = "resolution_queue_item"

    @property
    def requires_claim_provenance(self) -> bool:
        return self in {
            OutcomeTarget.CLAIM,
            OutcomeTarget.COMPILED_VIEW,
            OutcomeTarget.ANSWER,
        }


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    outcome_id: str
    label: OutcomeLabel
    target: OutcomeTarget
    target_id: str
    policy_stamp: str
    recorded_at: datetime
    rationale: str
    actor_subject_id: str | None = None
    claim_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    capture_for_replay: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome_id", _clean_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "actor_subject_id",
            _optional_clean_text(self.actor_subject_id, field_name="actor_subject_id"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )
        object.__setattr__(self, "claim_ids", _dedupe_cleaned(self.claim_ids, field_name="claim_ids"))
        object.__setattr__(
            self,
            "observation_ids",
            _dedupe_cleaned(self.observation_ids, field_name="observation_ids"),
        )

        if self.target.requires_claim_provenance and not self.claim_ids:
            raise ValueError("this outcome target requires claim provenance")

        if self.target is OutcomeTarget.CLAIM and self.target_id not in self.claim_ids:
            raise ValueError("claim outcomes must include the target claim id in claim_ids")

    @property
    def affects_memory_selection(self) -> bool:
        return self.label in {
            OutcomeLabel.PROMPT_INCLUDED,
            OutcomeLabel.ANSWER_CITED,
            OutcomeLabel.USER_CORRECTED,
            OutcomeLabel.STALE_ON_USE,
        }
