"""Admission gate invariants for Continuity."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.ontology import MemoryPartition
from continuity.store.claims import AdmissionDecision, AdmissionOutcome


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class AdmissionStrength(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def score(self) -> int:
        return {
            AdmissionStrength.LOW: 1,
            AdmissionStrength.MEDIUM: 2,
            AdmissionStrength.HIGH: 3,
        }[self]


@dataclass(frozen=True, slots=True)
class AdmissionThresholds:
    evidence: AdmissionStrength
    novelty: AdmissionStrength
    stability: AdmissionStrength
    salience: AdmissionStrength


@dataclass(frozen=True, slots=True)
class AdmissionAssessment:
    claim_type: str
    evidence: AdmissionStrength
    novelty: AdmissionStrength
    stability: AdmissionStrength
    salience: AdmissionStrength
    rationale: str
    utility_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_type", _clean_text(self.claim_type, field_name="claim_type"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        cleaned_signals = tuple(
            _clean_text(signal, field_name="utility_signals")
            for signal in self.utility_signals
        )
        object.__setattr__(self, "utility_signals", cleaned_signals)

    def shortfall_fields(self, thresholds: AdmissionThresholds) -> tuple[str, ...]:
        shortfalls: list[str] = []
        if self.evidence.score < thresholds.evidence.score:
            shortfalls.append("evidence")
        if self.novelty.score < thresholds.novelty.score:
            shortfalls.append("novelty")
        if self.stability.score < thresholds.stability.score:
            shortfalls.append("stability")
        if self.salience.score < thresholds.salience.score:
            shortfalls.append("salience")
        return tuple(shortfalls)

    def satisfies(self, thresholds: AdmissionThresholds) -> bool:
        return not self.shortfall_fields(thresholds)


@dataclass(frozen=True, slots=True)
class AdmissionWriteBudget:
    partition: MemoryPartition
    window_key: str
    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_key", _clean_text(self.window_key, field_name="window_key"))
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.used < 0:
            raise ValueError("used must be non-negative")
        if self.used > self.limit:
            raise ValueError("used must not exceed limit")

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def allows_durable_promotion(self, *, cost: int = 1) -> bool:
        if cost <= 0:
            raise ValueError("cost must be positive")
        return self.remaining >= cost


@dataclass(frozen=True, slots=True)
class AdmissionDecisionTrace:
    decision: AdmissionDecision
    claim_type: str
    policy_stamp: str
    assessment: AdmissionAssessment
    thresholds: AdmissionThresholds
    budget: AdmissionWriteBudget

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_type", _clean_text(self.claim_type, field_name="claim_type"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))

        if self.assessment.claim_type != self.claim_type:
            raise ValueError("assessment claim_type must match trace claim_type")

        if self.decision.outcome is AdmissionOutcome.DURABLE_CLAIM:
            if not self.assessment.satisfies(self.thresholds):
                raise ValueError("durable admission requires all explicit thresholds")
            if not self.budget.allows_durable_promotion():
                raise ValueError("durable admission requires remaining write budget")

    @property
    def shortfall_fields(self) -> tuple[str, ...]:
        return self.assessment.shortfall_fields(self.thresholds)

    @property
    def publishes_claim(self) -> bool:
        return self.decision.outcome is AdmissionOutcome.DURABLE_CLAIM

    @property
    def retains_candidate_context(self) -> bool:
        return self.decision.outcome in {
            AdmissionOutcome.SESSION_EPHEMERAL,
            AdmissionOutcome.PROMPT_ONLY,
            AdmissionOutcome.NEEDS_CONFIRMATION,
        }

    @property
    def requires_resolution_queue(self) -> bool:
        return self.decision.outcome is AdmissionOutcome.NEEDS_CONFIRMATION


class AdmissionRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def record_decision(self, trace: AdmissionDecisionTrace) -> None:
        current_budget = self.read_budget(
            partition=trace.budget.partition,
            window_key=trace.budget.window_key,
        )
        if current_budget is None:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO admission_write_budgets(
                        partition,
                        window_key,
                        limit_value,
                        used_value
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        trace.budget.partition.value,
                        trace.budget.window_key,
                        trace.budget.limit,
                        trace.budget.used,
                    ),
                )
        elif current_budget != trace.budget:
            raise ValueError("trace budget must match the current stored write budget")

        updated_used = trace.budget.used + int(trace.publishes_claim)

        with self._connection:
            self._connection.execute(
                """
                UPDATE admission_write_budgets
                SET limit_value = ?, used_value = ?
                WHERE partition = ? AND window_key = ?
                """,
                (
                    trace.budget.limit,
                    updated_used,
                    trace.budget.partition.value,
                    trace.budget.window_key,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO admission_decisions(
                    candidate_id,
                    outcome,
                    recorded_at,
                    rationale,
                    policy_stamp,
                    claim_type,
                    assessment_json,
                    thresholds_json,
                    budget_partition,
                    budget_window_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.decision.candidate_id,
                    trace.decision.outcome.value,
                    trace.decision.recorded_at.isoformat(),
                    trace.decision.rationale,
                    trace.policy_stamp,
                    trace.claim_type,
                    json.dumps(self._assessment_payload(trace)),
                    json.dumps(self._thresholds_payload(trace.thresholds)),
                    trace.budget.partition.value,
                    trace.budget.window_key,
                ),
            )

    def read_decision(self, candidate_id: str) -> AdmissionDecisionTrace | None:
        row = self._connection.execute(
            """
            SELECT
                candidate_id,
                outcome,
                recorded_at,
                rationale,
                policy_stamp,
                claim_type,
                assessment_json,
                thresholds_json,
                budget_partition,
                budget_window_key
            FROM admission_decisions
            WHERE candidate_id = ?
            """,
            (_clean_text(candidate_id, field_name="candidate_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._trace_from_row(row)

    def list_decisions(
        self,
        *,
        outcome: AdmissionOutcome | None = None,
        subject_id: str | None = None,
        claim_type: str | None = None,
        partition: MemoryPartition | None = None,
        window_key: str | None = None,
        limit: int | None = None,
    ) -> tuple[AdmissionDecisionTrace, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        conditions: list[str] = []
        params: list[str] = []
        if outcome is not None:
            conditions.append("admission_decisions.outcome = ?")
            params.append(outcome.value)
        if subject_id is not None:
            conditions.append("candidate_memories.subject_id = ?")
            params.append(_clean_text(subject_id, field_name="subject_id"))
        if claim_type is not None:
            conditions.append("admission_decisions.claim_type = ?")
            params.append(_clean_text(claim_type, field_name="claim_type"))
        if partition is not None:
            conditions.append("admission_decisions.budget_partition = ?")
            params.append(partition.value)
        if window_key is not None:
            conditions.append("admission_decisions.budget_window_key = ?")
            params.append(_clean_text(window_key, field_name="window_key"))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {limit}"

        rows = self._connection.execute(
            f"""
            SELECT
                admission_decisions.candidate_id,
                admission_decisions.outcome,
                admission_decisions.recorded_at,
                admission_decisions.rationale,
                admission_decisions.policy_stamp,
                admission_decisions.claim_type,
                admission_decisions.assessment_json,
                admission_decisions.thresholds_json,
                admission_decisions.budget_partition,
                admission_decisions.budget_window_key
            FROM admission_decisions
            JOIN candidate_memories
                ON candidate_memories.candidate_id = admission_decisions.candidate_id
            {where_clause}
            ORDER BY admission_decisions.recorded_at DESC, admission_decisions.candidate_id DESC
            {limit_clause}
            """,
            tuple(params),
        ).fetchall()
        return tuple(self._trace_from_row(row) for row in rows)

    def list_durable_promotions(
        self,
        *,
        subject_id: str | None = None,
        claim_type: str | None = None,
        partition: MemoryPartition | None = None,
        window_key: str | None = None,
        limit: int | None = None,
    ) -> tuple[AdmissionDecisionTrace, ...]:
        return self.list_decisions(
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            subject_id=subject_id,
            claim_type=claim_type,
            partition=partition,
            window_key=window_key,
            limit=limit,
        )

    def read_budget(
        self,
        *,
        partition: MemoryPartition,
        window_key: str,
    ) -> AdmissionWriteBudget | None:
        row = self._connection.execute(
            """
            SELECT partition, window_key, limit_value, used_value
            FROM admission_write_budgets
            WHERE partition = ? AND window_key = ?
            """,
            (
                partition.value,
                _clean_text(window_key, field_name="window_key"),
            ),
        ).fetchone()
        if row is None:
            return None
        return AdmissionWriteBudget(
            partition=MemoryPartition(row[0]),
            window_key=row[1],
            limit=row[2],
            used=row[3],
        )

    def _trace_from_row(self, row: tuple[object, ...]) -> AdmissionDecisionTrace:
        assessment_payload = json.loads(row[6])
        thresholds_payload = json.loads(row[7])
        budget_snapshot = assessment_payload.get("budget_snapshot")
        if budget_snapshot is None:
            budget_snapshot = self._budget_snapshot_from_current_row(
                partition_value=row[8],
                window_key=row[9],
            )

        return AdmissionDecisionTrace(
            decision=AdmissionDecision(
                candidate_id=row[0],
                outcome=AdmissionOutcome(row[1]),
                recorded_at=self._parse_timestamp(row[2]),
                rationale=row[3],
            ),
            claim_type=row[5],
            policy_stamp=row[4],
            assessment=AdmissionAssessment(
                claim_type=assessment_payload.get("claim_type", row[5]),
                evidence=AdmissionStrength(assessment_payload["evidence"]),
                novelty=AdmissionStrength(assessment_payload["novelty"]),
                stability=AdmissionStrength(assessment_payload["stability"]),
                salience=AdmissionStrength(assessment_payload["salience"]),
                rationale=assessment_payload.get("rationale", row[3]),
                utility_signals=tuple(assessment_payload.get("utility_signals", ())),
            ),
            thresholds=AdmissionThresholds(
                evidence=AdmissionStrength(thresholds_payload["evidence"]),
                novelty=AdmissionStrength(thresholds_payload["novelty"]),
                stability=AdmissionStrength(thresholds_payload["stability"]),
                salience=AdmissionStrength(thresholds_payload["salience"]),
            ),
            budget=AdmissionWriteBudget(
                partition=MemoryPartition(budget_snapshot["partition"]),
                window_key=budget_snapshot["window_key"],
                limit=budget_snapshot["limit"],
                used=budget_snapshot["used"],
            ),
        )

    def _budget_snapshot_from_current_row(
        self,
        *,
        partition_value: object,
        window_key: object,
    ) -> dict[str, object]:
        if partition_value is None or window_key is None:
            raise ValueError("stored admission decision is missing budget context")

        budget = self.read_budget(
            partition=MemoryPartition(partition_value),
            window_key=window_key,
        )
        if budget is None:
            raise ValueError("stored admission decision references a missing write budget")
        return {
            "partition": budget.partition.value,
            "window_key": budget.window_key,
            "limit": budget.limit,
            "used": budget.used,
        }

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        return datetime.fromisoformat(value)

    @staticmethod
    def _assessment_payload(trace: AdmissionDecisionTrace) -> dict[str, object]:
        return {
            "claim_type": trace.assessment.claim_type,
            "evidence": trace.assessment.evidence.value,
            "novelty": trace.assessment.novelty.value,
            "stability": trace.assessment.stability.value,
            "salience": trace.assessment.salience.value,
            "rationale": trace.assessment.rationale,
            "utility_signals": list(trace.assessment.utility_signals),
            "budget_snapshot": {
                "partition": trace.budget.partition.value,
                "window_key": trace.budget.window_key,
                "limit": trace.budget.limit,
                "used": trace.budget.used,
            },
        }

    @staticmethod
    def _thresholds_payload(thresholds: AdmissionThresholds) -> dict[str, str]:
        return {
            "evidence": thresholds.evidence.value,
            "novelty": thresholds.novelty.value,
            "stability": thresholds.stability.value,
            "salience": thresholds.salience.value,
        }
