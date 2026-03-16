"""Utility ledger invariants compiled from explicit Continuity outcomes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeTarget
from continuity.policy import PolicyPack


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class UtilitySignal(StrEnum):
    PROMPT_INCLUSION = "prompt_inclusion"
    ANSWER_CITATION = "answer_citation"
    USER_CORRECTED = "user_corrected"
    STALE_ON_USE = "stale_on_use"


_OUTCOME_SIGNAL_MAP: dict[OutcomeLabel, tuple[UtilitySignal, ...]] = {
    OutcomeLabel.PROMPT_INCLUDED: (UtilitySignal.PROMPT_INCLUSION,),
    OutcomeLabel.ANSWER_CITED: (UtilitySignal.ANSWER_CITATION,),
    OutcomeLabel.USER_CONFIRMED: (),
    OutcomeLabel.USER_CORRECTED: (UtilitySignal.USER_CORRECTED,),
    OutcomeLabel.STALE_ON_USE: (UtilitySignal.STALE_ON_USE,),
}


@dataclass(frozen=True, slots=True)
class UtilityEvent:
    source_outcome_id: str
    signal: UtilitySignal
    target: OutcomeTarget
    target_id: str
    policy_stamp: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_outcome_id",
            _clean_text(self.source_outcome_id, field_name="source_outcome_id"),
        )
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )


@dataclass(frozen=True, slots=True)
class CompiledUtilityWeight:
    target: OutcomeTarget
    target_id: str
    policy_stamp: str
    weighted_score: int
    signal_counts: tuple[tuple[UtilitySignal, int], ...]
    source_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(
            self,
            "source_event_ids",
            tuple(dict.fromkeys(sorted(_clean_text(event_id, field_name="source_event_ids") for event_id in self.source_event_ids))),
        )
        object.__setattr__(
            self,
            "signal_counts",
            tuple(
                (signal, count)
                for signal, count in sorted(self.signal_counts, key=lambda item: item[0].value)
            ),
        )

    def signal_count_for(self, signal: UtilitySignal) -> int:
        for current_signal, count in self.signal_counts:
            if current_signal is signal:
                return count
        return 0


def utility_events_for_outcome(outcome: OutcomeRecord) -> tuple[UtilityEvent, ...]:
    return tuple(
        UtilityEvent(
            source_outcome_id=outcome.outcome_id,
            signal=signal,
            target=outcome.target,
            target_id=outcome.target_id,
            policy_stamp=outcome.policy_stamp,
            recorded_at=outcome.recorded_at,
        )
        for signal in _OUTCOME_SIGNAL_MAP[outcome.label]
    )


def compile_utility_weight(
    *,
    target: OutcomeTarget,
    target_id: str,
    policy: PolicyPack,
    events: tuple[UtilityEvent, ...],
) -> CompiledUtilityWeight:
    cleaned_target_id = _clean_text(target_id, field_name="target_id")
    signal_counts: Counter[UtilitySignal] = Counter()
    source_event_ids: set[str] = set()

    for event in sorted(events, key=lambda candidate: (candidate.source_outcome_id, candidate.signal.value)):
        if event.target is not target:
            raise ValueError("utility compilation requires one shared target kind")
        if event.target_id != cleaned_target_id:
            raise ValueError("utility compilation requires one shared target id")
        if event.policy_stamp != policy.policy_stamp:
            raise ValueError("utility compilation requires events from the same policy stamp")

        signal_counts[event.signal] += 1
        source_event_ids.add(event.source_outcome_id)

    weighted_score = sum(
        policy.utility_weight_for(signal.value) * count
        for signal, count in signal_counts.items()
    )

    return CompiledUtilityWeight(
        target=target,
        target_id=cleaned_target_id,
        policy_stamp=policy.policy_stamp,
        weighted_score=weighted_score,
        signal_counts=tuple(signal_counts.items()),
        source_event_ids=tuple(source_event_ids),
    )
