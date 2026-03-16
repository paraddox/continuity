"""Replay artifact and counterfactual replay invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.disclosure import DisclosureContext
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


def _optional_clean_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _clean_deduped(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_clean_text(value, field_name=field_name) for value in values))


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _clean_policy_fingerprint(value: tuple[str, str]) -> tuple[str, str]:
    if len(value) != 2:
        raise ValueError("policy_fingerprint must contain policy stamp and policy fingerprint id")
    return (
        _clean_text(value[0], field_name="policy_fingerprint"),
        _clean_text(value[1], field_name="policy_fingerprint"),
    )


class ReplayStep(StrEnum):
    RETRIEVAL = "retrieval"
    BELIEF = "belief"
    REASONING = "reasoning"


class ReplayMetric(StrEnum):
    CORRECTNESS = "correctness"
    FRESHNESS = "freshness"
    DISCLOSURE_SAFETY = "disclosure_safety"
    QUEUE_YIELD = "queue_yield"
    UTILITY_ALIGNMENT = "utility_alignment"


class ReplayMutationMode(StrEnum):
    READ_ONLY = "read_only"


@dataclass(frozen=True, slots=True)
class ReplayStrategy:
    step: ReplayStep
    strategy_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy_id",
            _clean_text(self.strategy_id, field_name="strategy_id"),
        )
        object.__setattr__(
            self,
            "fingerprint",
            _clean_text(self.fingerprint, field_name="fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class ReplayInputBundle:
    bundle_id: str
    surface: str
    snapshot_id: str
    journal_position: int
    arbiter_lane_position: int
    disclosure_context: DisclosureContext
    claim_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    compiled_view_ids: tuple[str, ...] = ()
    outcome_ids: tuple[str, ...] = ()
    reference_ids: tuple[str, ...] = ()
    query_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _clean_text(self.bundle_id, field_name="bundle_id"))
        object.__setattr__(self, "surface", _clean_text(self.surface, field_name="surface"))
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(
            self,
            "query_text",
            _optional_clean_text(self.query_text, field_name="query_text"),
        )
        object.__setattr__(self, "claim_ids", _clean_deduped(self.claim_ids, field_name="claim_ids"))
        object.__setattr__(
            self,
            "observation_ids",
            _clean_deduped(self.observation_ids, field_name="observation_ids"),
        )
        object.__setattr__(
            self,
            "compiled_view_ids",
            _clean_deduped(self.compiled_view_ids, field_name="compiled_view_ids"),
        )
        object.__setattr__(
            self,
            "outcome_ids",
            _clean_deduped(self.outcome_ids, field_name="outcome_ids"),
        )
        object.__setattr__(
            self,
            "reference_ids",
            _clean_deduped(self.reference_ids, field_name="reference_ids"),
        )

        if self.journal_position <= 0:
            raise ValueError("journal_position must be positive")
        if self.arbiter_lane_position <= 0:
            raise ValueError("arbiter_lane_position must be positive")

        if not any(
            (
                self.claim_ids,
                self.observation_ids,
                self.compiled_view_ids,
                self.outcome_ids,
                self.reference_ids,
                self.query_text is not None,
            )
        ):
            raise ValueError("replay inputs must carry stable references or query text")

    @property
    def deterministic_key(self) -> tuple[object, ...]:
        viewer = self.disclosure_context.viewer
        return (
            self.snapshot_id,
            self.journal_position,
            self.arbiter_lane_position,
            self.surface,
            self.disclosure_context.audience_principal.value,
            self.disclosure_context.channel.value,
            self.disclosure_context.purpose.value,
            viewer.viewer_kind.value,
            viewer.viewer_subject_id,
            viewer.active_user_id,
            viewer.active_peer_id,
            self.claim_ids,
            self.observation_ids,
            self.compiled_view_ids,
            self.outcome_ids,
            self.reference_ids,
            self.query_text,
        )


@dataclass(frozen=True, slots=True)
class ReplayRun:
    run_id: str
    input_bundle: ReplayInputBundle
    policy_fingerprint: tuple[str, str]
    strategies: tuple[ReplayStrategy, ...]
    output_refs: tuple[str, ...]
    metric_scores: dict[ReplayMetric, int]
    mutation_mode: ReplayMutationMode = ReplayMutationMode.READ_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _clean_text(self.run_id, field_name="run_id"))
        object.__setattr__(
            self,
            "policy_fingerprint",
            _clean_policy_fingerprint(self.policy_fingerprint),
        )
        object.__setattr__(
            self,
            "output_refs",
            _clean_deduped(self.output_refs, field_name="output_refs"),
        )

        if not self.output_refs:
            raise ValueError("output_refs must be non-empty")
        if not self.metric_scores:
            raise ValueError("metric_scores must be non-empty")

        strategy_map: dict[ReplayStep, ReplayStrategy] = {}
        for strategy in self.strategies:
            if strategy.step in strategy_map:
                raise ValueError(f"duplicate replay strategy for {strategy.step.value}")
            strategy_map[strategy.step] = strategy
        if set(strategy_map) != set(ReplayStep):
            raise ValueError("replay runs require one explicit strategy for each replay step")

        for metric, score in self.metric_scores.items():
            if not isinstance(metric, ReplayMetric):
                raise ValueError("metric_scores keys must be ReplayMetric values")
            if isinstance(score, bool) or not isinstance(score, int):
                raise ValueError("metric_scores values must be integers")

    def strategy_for(self, step: ReplayStep) -> ReplayStrategy:
        for strategy in self.strategies:
            if strategy.step is step:
                return strategy
        raise KeyError(f"missing strategy for {step.value}")


@dataclass(frozen=True, slots=True)
class ReplayArtifact:
    artifact_id: str
    version: str
    source_transaction: TransactionKind
    source_waterline: DurabilityWaterline
    captured_at: datetime
    baseline_run: ReplayRun
    source_object_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _clean_text(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "version", _clean_text(self.version, field_name="version"))
        object.__setattr__(
            self,
            "captured_at",
            _validate_timestamp(self.captured_at, field_name="captured_at"),
        )
        object.__setattr__(
            self,
            "source_object_ids",
            _clean_deduped(self.source_object_ids, field_name="source_object_ids"),
        )

        if not self.source_object_ids:
            raise ValueError("source_object_ids must be non-empty")
        if self.baseline_run.mutation_mode is not ReplayMutationMode.READ_ONLY:
            raise ValueError("replay artifacts must capture read-only replay runs")
        if not transaction_contract_for(self.source_transaction).supports_waterline(
            self.source_waterline
        ):
            raise ValueError(
                f"{self.source_transaction.value} cannot reach {self.source_waterline.value}"
            )

    @property
    def policy_fingerprint(self) -> tuple[str, str]:
        return self.baseline_run.policy_fingerprint


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    comparison_id: str
    baseline_run: ReplayRun
    candidate_run: ReplayRun
    compared_steps: tuple[ReplayStep, ...]
    rationale: str
    mutation_mode: ReplayMutationMode = ReplayMutationMode.READ_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "comparison_id",
            _clean_text(self.comparison_id, field_name="comparison_id"),
        )
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "compared_steps",
            tuple(dict.fromkeys(self.compared_steps)),
        )

        if not self.compared_steps:
            raise ValueError("compared_steps must be non-empty")
        if self.mutation_mode is not ReplayMutationMode.READ_ONLY:
            raise ValueError("counterfactual replay comparisons must stay read-only")
        if self.baseline_run.mutation_mode is not ReplayMutationMode.READ_ONLY:
            raise ValueError("baseline replay runs must stay read-only")
        if self.candidate_run.mutation_mode is not ReplayMutationMode.READ_ONLY:
            raise ValueError("candidate replay runs must stay read-only")
        if (
            self.baseline_run.input_bundle.deterministic_key
            != self.candidate_run.input_bundle.deterministic_key
        ):
            raise ValueError("counterfactual replay requires identical deterministic replay inputs")

    @property
    def policy_changed(self) -> bool:
        return self.baseline_run.policy_fingerprint != self.candidate_run.policy_fingerprint

    @property
    def changed_steps(self) -> frozenset[ReplayStep]:
        return frozenset(
            step
            for step in self.compared_steps
            if self.baseline_run.strategy_for(step).fingerprint
            != self.candidate_run.strategy_for(step).fingerprint
        )

    @property
    def mutates_authoritative_state(self) -> bool:
        return False
