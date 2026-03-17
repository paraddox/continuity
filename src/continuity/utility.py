"""Utility ledger invariants compiled from explicit Continuity outcomes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
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


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    return _validate_timestamp(datetime.fromisoformat(value), field_name=field_name)


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


class UtilityRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def record_events(self, events: tuple[UtilityEvent, ...]) -> None:
        if not events:
            return

        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO utility_events(
                    event_id,
                    source_outcome_id,
                    signal,
                    target,
                    target_id,
                    policy_stamp,
                    recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    source_outcome_id = excluded.source_outcome_id,
                    signal = excluded.signal,
                    target = excluded.target,
                    target_id = excluded.target_id,
                    policy_stamp = excluded.policy_stamp,
                    recorded_at = excluded.recorded_at
                """,
                tuple(
                    (
                        self._event_id_for(event),
                        event.source_outcome_id,
                        event.signal.value,
                        event.target.value,
                        event.target_id,
                        event.policy_stamp,
                        event.recorded_at.isoformat(),
                    )
                    for event in events
                ),
            )

    def list_events(
        self,
        *,
        target: OutcomeTarget | None = None,
        target_id: str | None = None,
        signal: UtilitySignal | None = None,
        policy_stamp: str | None = None,
        source_outcome_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[UtilityEvent, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        conditions: list[str] = []
        params: list[str] = []
        if target is not None:
            conditions.append("target = ?")
            params.append(target.value)
        if target_id is not None:
            conditions.append("target_id = ?")
            params.append(_clean_text(target_id, field_name="target_id"))
        if signal is not None:
            conditions.append("signal = ?")
            params.append(signal.value)
        if policy_stamp is not None:
            conditions.append("policy_stamp = ?")
            params.append(_clean_text(policy_stamp, field_name="policy_stamp"))
        if source_outcome_id is not None:
            conditions.append("source_outcome_id = ?")
            params.append(_clean_text(source_outcome_id, field_name="source_outcome_id"))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {limit}"

        rows = self._connection.execute(
            f"""
            SELECT
                source_outcome_id,
                signal,
                target,
                target_id,
                policy_stamp,
                recorded_at
            FROM utility_events
            {where_clause}
            ORDER BY recorded_at DESC, event_id DESC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def write_compiled_weight(self, weight: CompiledUtilityWeight) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO compiled_utility_weights(
                    target,
                    target_id,
                    policy_stamp,
                    weighted_score,
                    signal_counts_json,
                    source_event_ids_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(target, target_id, policy_stamp) DO UPDATE SET
                    weighted_score = excluded.weighted_score,
                    signal_counts_json = excluded.signal_counts_json,
                    source_event_ids_json = excluded.source_event_ids_json
                """,
                (
                    weight.target.value,
                    weight.target_id,
                    weight.policy_stamp,
                    weight.weighted_score,
                    json.dumps(
                        {signal.value: count for signal, count in weight.signal_counts},
                        sort_keys=True,
                    ),
                    json.dumps(list(weight.source_event_ids)),
                ),
            )

    def read_compiled_weight(
        self,
        *,
        target: OutcomeTarget,
        target_id: str,
        policy_stamp: str,
    ) -> CompiledUtilityWeight | None:
        row = self._connection.execute(
            """
            SELECT
                target,
                target_id,
                policy_stamp,
                weighted_score,
                signal_counts_json,
                source_event_ids_json
            FROM compiled_utility_weights
            WHERE target = ? AND target_id = ? AND policy_stamp = ?
            """,
            (
                target.value,
                _clean_text(target_id, field_name="target_id"),
                _clean_text(policy_stamp, field_name="policy_stamp"),
            ),
        ).fetchone()
        if row is None:
            return None
        return self._weight_from_row(row)

    def list_compiled_weights(
        self,
        *,
        target: OutcomeTarget | None = None,
        target_id: str | None = None,
        policy_stamp: str | None = None,
        limit: int | None = None,
    ) -> tuple[CompiledUtilityWeight, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        conditions: list[str] = []
        params: list[str] = []
        if target is not None:
            conditions.append("target = ?")
            params.append(target.value)
        if target_id is not None:
            conditions.append("target_id = ?")
            params.append(_clean_text(target_id, field_name="target_id"))
        if policy_stamp is not None:
            conditions.append("policy_stamp = ?")
            params.append(_clean_text(policy_stamp, field_name="policy_stamp"))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {limit}"

        rows = self._connection.execute(
            f"""
            SELECT
                target,
                target_id,
                policy_stamp,
                weighted_score,
                signal_counts_json,
                source_event_ids_json
            FROM compiled_utility_weights
            {where_clause}
            ORDER BY weighted_score DESC, target_id, policy_stamp
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return tuple(self._weight_from_row(row) for row in rows)

    @staticmethod
    def _event_id_for(event: UtilityEvent) -> str:
        payload = "|".join(
            (
                event.source_outcome_id,
                event.signal.value,
                event.target.value,
                event.target_id,
                event.policy_stamp,
            )
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"utility-event:{digest}"

    @staticmethod
    def _event_from_row(row: tuple[object, ...]) -> UtilityEvent:
        return UtilityEvent(
            source_outcome_id=row[0],
            signal=UtilitySignal(row[1]),
            target=OutcomeTarget(row[2]),
            target_id=row[3],
            policy_stamp=row[4],
            recorded_at=_parse_timestamp(row[5], field_name="recorded_at"),
        )

    @staticmethod
    def _weight_from_row(row: tuple[object, ...]) -> CompiledUtilityWeight:
        signal_counts_payload = json.loads(row[4])
        return CompiledUtilityWeight(
            target=OutcomeTarget(row[0]),
            target_id=row[1],
            policy_stamp=row[2],
            weighted_score=row[3],
            signal_counts=tuple(
                (UtilitySignal(signal), int(count))
                for signal, count in signal_counts_payload.items()
            ),
            source_event_ids=tuple(json.loads(row[5])),
        )
