"""Admission gate invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
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

