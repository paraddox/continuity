"""Counterfactual replay runner over stored Continuity turn artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

from continuity.outcomes import OutcomeLabel, OutcomeRecord, OutcomeRepository, OutcomeTarget
from continuity.replay import (
    ReplayArtifact,
    ReplayMetric,
    ReplayRun,
    ReplayStep,
    ReplayStrategy,
)
from continuity.store.replay import ReplayComparisonRecord, ReplayRepository
from continuity.utility import CompiledUtilityWeight, UtilityRepository


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _normalize_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return deepcopy(dict(payload))


def _normalize_string_tuple(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_clean_text(value, field_name=field_name) for value in values))


class ReplayExecutionMode(StrEnum):
    RETRIEVAL_ONLY = "retrieval_only"
    BELIEF_ONLY = "belief_only"
    END_TO_END = "end_to_end"

    @property
    def allowed_steps(self) -> frozenset[ReplayStep]:
        if self is ReplayExecutionMode.RETRIEVAL_ONLY:
            return frozenset({ReplayStep.RETRIEVAL})
        if self is ReplayExecutionMode.BELIEF_ONLY:
            return frozenset({ReplayStep.BELIEF})
        return frozenset(ReplayStep)


@dataclass(frozen=True, slots=True)
class ReplayStageOverride:
    step: ReplayStep
    strategy_id: str
    fingerprint: str
    payload: Mapping[str, object]

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
        object.__setattr__(self, "payload", _normalize_payload(self.payload))


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    run_id: str
    comparison_id: str
    mode: ReplayExecutionMode
    rationale: str
    stage_overrides: tuple[ReplayStageOverride, ...] = ()
    policy_fingerprint: tuple[str, str] | None = None
    payload_overrides: Mapping[str, object] = field(default_factory=dict)
    output_refs: tuple[str, ...] = ()
    persist_comparison: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _clean_text(self.run_id, field_name="run_id"))
        object.__setattr__(
            self,
            "comparison_id",
            _clean_text(self.comparison_id, field_name="comparison_id"),
        )
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "payload_overrides",
            _normalize_payload(self.payload_overrides),
        )
        object.__setattr__(
            self,
            "output_refs",
            _normalize_string_tuple(self.output_refs, field_name="output_refs"),
        )

        seen_steps: set[ReplayStep] = set()
        for override in self.stage_overrides:
            if override.step in seen_steps:
                raise ValueError(f"duplicate stage override for {override.step.value}")
            seen_steps.add(override.step)

        invalid_steps = seen_steps - self.mode.allowed_steps
        if invalid_steps:
            invalid = ", ".join(step.value for step in sorted(invalid_steps, key=lambda step: step.value))
            raise ValueError(f"{self.mode.value} does not allow overrides for {invalid}")
        if not self.stage_overrides and not self.payload_overrides:
            raise ValueError("replay plans must override at least one stage or payload")

    @property
    def compared_steps(self) -> tuple[ReplayStep, ...]:
        if self.mode is ReplayExecutionMode.END_TO_END:
            return tuple(ReplayStep)
        return tuple(override.step for override in self.stage_overrides)


@dataclass(frozen=True, slots=True)
class ReplayEvaluationExpectation:
    expected_retrieval_view_keys: tuple[str, ...] = ()
    expected_selected_claim_ids: tuple[str, ...] = ()
    expected_answer_substrings: tuple[str, ...] = ()
    expected_disclosure_result: str | None = None
    forbidden_claim_ids: tuple[str, ...] = ()
    expected_queue_item_ids: tuple[str, ...] = ()
    expected_utility_event_kinds: tuple[str, ...] = ()
    fresh_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_retrieval_view_keys",
            _normalize_string_tuple(
                self.expected_retrieval_view_keys,
                field_name="expected_retrieval_view_keys",
            ),
        )
        object.__setattr__(
            self,
            "expected_selected_claim_ids",
            _normalize_string_tuple(
                self.expected_selected_claim_ids,
                field_name="expected_selected_claim_ids",
            ),
        )
        object.__setattr__(
            self,
            "expected_answer_substrings",
            _normalize_string_tuple(
                self.expected_answer_substrings,
                field_name="expected_answer_substrings",
            ),
        )
        object.__setattr__(
            self,
            "forbidden_claim_ids",
            _normalize_string_tuple(self.forbidden_claim_ids, field_name="forbidden_claim_ids"),
        )
        object.__setattr__(
            self,
            "expected_queue_item_ids",
            _normalize_string_tuple(
                self.expected_queue_item_ids,
                field_name="expected_queue_item_ids",
            ),
        )
        object.__setattr__(
            self,
            "expected_utility_event_kinds",
            _normalize_string_tuple(
                self.expected_utility_event_kinds,
                field_name="expected_utility_event_kinds",
            ),
        )
        object.__setattr__(
            self,
            "fresh_claim_ids",
            _normalize_string_tuple(self.fresh_claim_ids, field_name="fresh_claim_ids"),
        )
        if self.expected_disclosure_result is not None:
            object.__setattr__(
                self,
                "expected_disclosure_result",
                _clean_text(
                    self.expected_disclosure_result,
                    field_name="expected_disclosure_result",
                ),
            )


@dataclass(frozen=True, slots=True)
class ReplayEvaluation:
    artifact: ReplayArtifact
    candidate_run: ReplayRun
    candidate_payload: dict[str, object]
    comparison_record: ReplayComparisonRecord
    baseline_scores: dict[ReplayMetric, int]
    candidate_scores: dict[ReplayMetric, int]


@dataclass(frozen=True, slots=True)
class ReplayFixtureCase:
    fixture_id: str
    artifact_id: str
    plan: ReplayPlan
    expectation: ReplayEvaluationExpectation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_id",
            _clean_text(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(
            self,
            "artifact_id",
            _clean_text(self.artifact_id, field_name="artifact_id"),
        )


_PAYLOAD_KEY_BY_STEP: dict[ReplayStep, str] = {
    ReplayStep.RETRIEVAL: "retrieval",
    ReplayStep.BELIEF: "selection",
    ReplayStep.REASONING: "reasoning",
}


class ReplayRunner:
    """Replay stored turn artifacts without mutating authoritative memory state."""

    def __init__(
        self,
        *,
        replay_repository: ReplayRepository,
        outcome_repository: OutcomeRepository | None = None,
        utility_repository: UtilityRepository | None = None,
    ) -> None:
        self._replay = replay_repository
        self._outcomes = outcome_repository
        self._utility = utility_repository

    def evaluate_cases(
        self,
        cases: tuple[ReplayFixtureCase, ...],
    ) -> dict[str, ReplayEvaluation]:
        return {
            case.fixture_id: self.evaluate_artifact(
                artifact_id=case.artifact_id,
                plan=case.plan,
                expectation=case.expectation,
            )
            for case in cases
        }

    def evaluate_artifact(
        self,
        *,
        artifact_id: str,
        plan: ReplayPlan,
        expectation: ReplayEvaluationExpectation,
    ) -> ReplayEvaluation:
        artifact = self._artifact(artifact_id)
        baseline_payload = _normalize_payload(artifact.decision_payload)
        candidate_payload = _normalize_payload(artifact.decision_payload)

        strategies_by_step = {
            strategy.step: strategy
            for strategy in artifact.baseline_run.strategies
        }
        for override in plan.stage_overrides:
            candidate_payload[_PAYLOAD_KEY_BY_STEP[override.step]] = _normalize_payload(override.payload)
            strategies_by_step[override.step] = ReplayStrategy(
                step=override.step,
                strategy_id=override.strategy_id,
                fingerprint=override.fingerprint,
            )
        for key, value in plan.payload_overrides.items():
            candidate_payload[_clean_text(key, field_name="payload_overrides")] = deepcopy(value)

        baseline_scores = self._score_payload(
            baseline_payload,
            expectation=expectation,
            policy_stamp=artifact.policy_fingerprint[0],
            explicit_outcome_ids=artifact.baseline_run.input_bundle.outcome_ids,
        )
        candidate_policy_fingerprint = plan.policy_fingerprint or artifact.policy_fingerprint
        candidate_scores = self._score_payload(
            candidate_payload,
            expectation=expectation,
            policy_stamp=candidate_policy_fingerprint[0],
            explicit_outcome_ids=artifact.baseline_run.input_bundle.outcome_ids,
        )

        candidate_run = ReplayRun(
            run_id=plan.run_id,
            input_bundle=artifact.baseline_run.input_bundle,
            policy_fingerprint=candidate_policy_fingerprint,
            strategies=tuple(strategies_by_step[step] for step in ReplayStep),
            output_refs=plan.output_refs or artifact.baseline_run.output_refs,
            metric_scores=candidate_scores,
        )
        comparison_record = ReplayComparisonRecord(
            comparison=self._comparison_for(artifact=artifact, candidate_run=candidate_run, plan=plan),
            compared_at=artifact.captured_at,
            metric_deltas={
                metric: candidate_scores[metric] - baseline_scores[metric]
                for metric in ReplayMetric
            },
            notes=(
                f"mode={plan.mode.value}",
                f"artifact={artifact.artifact_id}",
            ),
        )
        if plan.persist_comparison:
            self._replay.record_comparison(comparison_record)

        return ReplayEvaluation(
            artifact=artifact,
            candidate_run=candidate_run,
            candidate_payload=candidate_payload,
            comparison_record=comparison_record,
            baseline_scores=baseline_scores,
            candidate_scores=candidate_scores,
        )

    def _artifact(self, artifact_id: str) -> ReplayArtifact:
        artifact = self._replay.read_artifact(artifact_id)
        if artifact is None:
            raise LookupError(_clean_text(artifact_id, field_name="artifact_id"))
        return artifact

    def _comparison_for(
        self,
        *,
        artifact: ReplayArtifact,
        candidate_run: ReplayRun,
        plan: ReplayPlan,
    ):
        from continuity.replay import ReplayComparison

        return ReplayComparison(
            comparison_id=plan.comparison_id,
            baseline_run=artifact.baseline_run,
            candidate_run=candidate_run,
            compared_steps=plan.compared_steps,
            rationale=plan.rationale,
        )

    def _score_payload(
        self,
        payload: Mapping[str, object],
        *,
        expectation: ReplayEvaluationExpectation,
        policy_stamp: str | None = None,
        explicit_outcome_ids: tuple[str, ...] = (),
    ) -> dict[ReplayMetric, int]:
        return {
            ReplayMetric.CORRECTNESS: self._correctness_score(payload, expectation=expectation),
            ReplayMetric.FRESHNESS: self._freshness_score(
                payload,
                expectation=expectation,
                explicit_outcome_ids=explicit_outcome_ids,
            ),
            ReplayMetric.DISCLOSURE_SAFETY: self._disclosure_safety_score(
                payload,
                expectation=expectation,
            ),
            ReplayMetric.QUEUE_YIELD: self._queue_yield_score(
                payload,
                expectation=expectation,
                explicit_outcome_ids=explicit_outcome_ids,
            ),
            ReplayMetric.UTILITY_ALIGNMENT: self._utility_alignment_score(
                payload,
                expectation=expectation,
                policy_stamp=policy_stamp,
            ),
        }

    def _correctness_score(
        self,
        payload: Mapping[str, object],
        *,
        expectation: ReplayEvaluationExpectation,
    ) -> int:
        checks: list[bool] = []

        retrieval = self._payload_mapping(payload, "retrieval")
        if expectation.expected_retrieval_view_keys:
            actual_view_keys = tuple(
                candidate.get("view_key")
                for candidate in self._payload_sequence(retrieval, "candidates")
                if isinstance(candidate, Mapping) and isinstance(candidate.get("view_key"), str)
            )
            checks.append(actual_view_keys[: len(expectation.expected_retrieval_view_keys)] == expectation.expected_retrieval_view_keys)

        selection = self._payload_mapping(payload, "selection")
        if expectation.expected_selected_claim_ids:
            selected_claim_ids = tuple(
                claim_id
                for claim_id in self._payload_sequence(selection, "selected_claim_ids")
                if isinstance(claim_id, str)
            )
            checks.append(selected_claim_ids == expectation.expected_selected_claim_ids)

        reasoning = self._payload_mapping(payload, "reasoning")
        response_text = reasoning.get("response_text")
        if expectation.expected_answer_substrings:
            if not isinstance(response_text, str):
                checks.append(False)
            else:
                checks.extend(
                    substring in response_text
                    for substring in expectation.expected_answer_substrings
                )

        return self._score_checks(checks)

    def _freshness_score(
        self,
        payload: Mapping[str, object],
        *,
        expectation: ReplayEvaluationExpectation,
        explicit_outcome_ids: tuple[str, ...],
    ) -> int:
        checks: list[bool] = []
        selection = self._payload_mapping(payload, "selection")
        selected_claim_ids = {
            claim_id
            for claim_id in self._payload_sequence(selection, "selected_claim_ids")
            if isinstance(claim_id, str)
        }
        checks.extend(
            claim_id in selected_claim_ids
            for claim_id in expectation.fresh_claim_ids
        )
        for outcome in self._relevant_outcomes(payload, explicit_outcome_ids=explicit_outcome_ids):
            if outcome.label is OutcomeLabel.STALE_ON_USE:
                checks.append(outcome.target_id not in selected_claim_ids)
        return self._score_checks(checks)

    def _disclosure_safety_score(
        self,
        payload: Mapping[str, object],
        *,
        expectation: ReplayEvaluationExpectation,
    ) -> int:
        checks: list[bool] = []
        disclosure = self._payload_mapping(payload, "disclosure")
        if expectation.expected_disclosure_result is not None:
            checks.append(disclosure.get("result") == expectation.expected_disclosure_result)

        if expectation.forbidden_claim_ids:
            hidden_claim_ids = {
                claim_id
                for claim_id in self._payload_sequence(disclosure, "hidden_claim_ids")
                if isinstance(claim_id, str)
            }
            withheld_claim_ids = {
                claim_id
                for claim_id in self._payload_sequence(disclosure, "withheld_claim_ids")
                if isinstance(claim_id, str)
            }
            selected_claim_ids = {
                claim_id
                for claim_id in self._payload_sequence(
                    self._payload_mapping(payload, "selection"),
                    "selected_claim_ids",
                )
                if isinstance(claim_id, str)
            }
            protected_claim_ids = hidden_claim_ids | withheld_claim_ids
            checks.extend(
                claim_id not in selected_claim_ids or claim_id in protected_claim_ids
                for claim_id in expectation.forbidden_claim_ids
            )

        return self._score_checks(checks)

    def _queue_yield_score(
        self,
        payload: Mapping[str, object],
        *,
        expectation: ReplayEvaluationExpectation,
        explicit_outcome_ids: tuple[str, ...],
    ) -> int:
        resolution_queue = self._payload_mapping(payload, "resolution_queue")
        actual_item_ids = tuple(
            item.get("item_id")
            for item in self._payload_sequence(resolution_queue, "surfaced_items")
            if isinstance(item, Mapping) and isinstance(item.get("item_id"), str)
        )
        checks = [
            item_id in actual_item_ids
            for item_id in expectation.expected_queue_item_ids
        ]
        for outcome in self._relevant_outcomes(payload, explicit_outcome_ids=explicit_outcome_ids):
            if outcome.target is OutcomeTarget.RESOLUTION_QUEUE_ITEM:
                checks.append(
                    outcome.label is OutcomeLabel.USER_CONFIRMED
                    and outcome.target_id in actual_item_ids
                )
        return self._score_checks(checks)

    def _utility_alignment_score(
        self,
        payload: Mapping[str, object],
        *,
        expectation: ReplayEvaluationExpectation,
        policy_stamp: str | None,
    ) -> int:
        checks: list[bool] = []
        actual_kinds = tuple(
            event.get("kind")
            for event in self._payload_sequence(payload, "utility_events")
            if isinstance(event, Mapping) and isinstance(event.get("kind"), str)
        )
        checks.extend(
            kind in actual_kinds
            for kind in expectation.expected_utility_event_kinds
        )
        if policy_stamp is not None:
            for weight in self._relevant_utility_weights(payload, policy_stamp=policy_stamp):
                checks.append(weight.weighted_score > 0)
        return self._score_checks(checks)

    def _relevant_outcomes(
        self,
        payload: Mapping[str, object],
        *,
        explicit_outcome_ids: tuple[str, ...],
    ) -> tuple[OutcomeRecord, ...]:
        if self._outcomes is None:
            return ()

        outcomes: dict[str, OutcomeRecord] = {}
        for outcome_id in explicit_outcome_ids:
            record = self._outcomes.read_record(outcome_id)
            if record is not None:
                outcomes[record.outcome_id] = record
        for claim_id in self._selected_claim_ids(payload):
            for record in self._outcomes.list_records(
                target=OutcomeTarget.CLAIM,
                target_id=claim_id,
            ):
                outcomes[record.outcome_id] = record
        for item_id in self._queue_item_ids(payload):
            for record in self._outcomes.list_records(
                target=OutcomeTarget.RESOLUTION_QUEUE_ITEM,
                target_id=item_id,
            ):
                outcomes[record.outcome_id] = record
        return tuple(
            outcomes[outcome_id]
            for outcome_id in sorted(outcomes)
        )

    def _relevant_utility_weights(
        self,
        payload: Mapping[str, object],
        *,
        policy_stamp: str,
    ) -> tuple[CompiledUtilityWeight, ...]:
        if self._utility is None:
            return ()

        weights: list[CompiledUtilityWeight] = []
        for claim_id in self._selected_claim_ids(payload):
            weight = self._utility.read_compiled_weight(
                target=OutcomeTarget.CLAIM,
                target_id=claim_id,
                policy_stamp=policy_stamp,
            )
            if weight is not None:
                weights.append(weight)
        for item_id in self._queue_item_ids(payload):
            weight = self._utility.read_compiled_weight(
                target=OutcomeTarget.RESOLUTION_QUEUE_ITEM,
                target_id=item_id,
                policy_stamp=policy_stamp,
            )
            if weight is not None:
                weights.append(weight)
        return tuple(weights)

    def _selected_claim_ids(self, payload: Mapping[str, object]) -> tuple[str, ...]:
        selection = self._payload_mapping(payload, "selection")
        return tuple(
            claim_id
            for claim_id in self._payload_sequence(selection, "selected_claim_ids")
            if isinstance(claim_id, str)
        )

    def _queue_item_ids(self, payload: Mapping[str, object]) -> tuple[str, ...]:
        resolution_queue = self._payload_mapping(payload, "resolution_queue")
        return tuple(
            item.get("item_id")
            for item in self._payload_sequence(resolution_queue, "surfaced_items")
            if isinstance(item, Mapping) and isinstance(item.get("item_id"), str)
        )

    @staticmethod
    def _payload_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
        return {}

    @staticmethod
    def _payload_sequence(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
        value = payload.get(key)
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return ()

    @staticmethod
    def _score_checks(checks: list[bool]) -> int:
        if not checks:
            return 0
        satisfied = sum(1 for check in checks if check)
        return int(round((satisfied / len(checks)) * 5))


__all__ = [
    "ReplayEvaluation",
    "ReplayEvaluationExpectation",
    "ReplayExecutionMode",
    "ReplayFixtureCase",
    "ReplayPlan",
    "ReplayRunner",
    "ReplayStageOverride",
]
