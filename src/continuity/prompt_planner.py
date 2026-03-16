"""Budgeted prompt planner contract for Continuity."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from continuity.epistemics import EpistemicStatus, PromptExposure, prompt_exposure_for_status
from continuity.views import ViewKind, view_contract_for


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _optional_clean_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _dedupe_cleaned(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    cleaned_values = tuple(_clean_text(value, field_name=field_name) for value in values)
    return tuple(dict.fromkeys(cleaned_values))


class PromptDisclosureAction(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    WITHHOLD = "withhold"


@dataclass(frozen=True, slots=True)
class PromptPlannerConfig:
    hard_token_budget: int
    soft_token_budgets: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hard_token_budget <= 0:
            raise ValueError("hard_token_budget must be positive")

        cleaned_budgets: dict[str, int] = {}
        for group, limit in self.soft_token_budgets.items():
            cleaned_group = _clean_text(group, field_name="soft_token_budgets")
            if limit < 0:
                raise ValueError("soft token budgets must be non-negative")
            cleaned_budgets[cleaned_group] = limit
        object.__setattr__(self, "soft_token_budgets", cleaned_budgets)


@dataclass(frozen=True, slots=True)
class PromptFragmentCandidate:
    fragment_id: str
    source_view: ViewKind
    text: str
    token_estimate: int
    priority_band: int
    claim_ids: tuple[str, ...]
    observation_ids: tuple[str, ...] = ()
    utility_weight: int = 0
    epistemic_status: EpistemicStatus = EpistemicStatus.SUPPORTED
    disclosure_action: PromptDisclosureAction = PromptDisclosureAction.ALLOW
    disclosure_reason: str | None = None
    soft_budget_group: str | None = None
    compressed_text: str | None = None
    compressed_token_estimate: int | None = None
    degradation_reason: str | None = None
    actual_token_usage: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_id", _clean_text(self.fragment_id, field_name="fragment_id"))
        object.__setattr__(self, "text", _clean_text(self.text, field_name="text"))
        object.__setattr__(self, "claim_ids", _dedupe_cleaned(self.claim_ids, field_name="claim_ids"))
        object.__setattr__(
            self,
            "observation_ids",
            _dedupe_cleaned(self.observation_ids, field_name="observation_ids"),
        )
        object.__setattr__(
            self,
            "soft_budget_group",
            _optional_clean_text(self.soft_budget_group, field_name="soft_budget_group"),
        )
        object.__setattr__(
            self,
            "disclosure_reason",
            _optional_clean_text(self.disclosure_reason, field_name="disclosure_reason"),
        )
        object.__setattr__(
            self,
            "compressed_text",
            _optional_clean_text(self.compressed_text, field_name="compressed_text"),
        )
        object.__setattr__(
            self,
            "degradation_reason",
            _optional_clean_text(self.degradation_reason, field_name="degradation_reason"),
        )

        if self.source_view not in view_contract_for(ViewKind.PROMPT).dependency_view_kinds:
            raise ValueError("prompt fragments must originate from prompt-eligible source views")
        if self.token_estimate <= 0:
            raise ValueError("token_estimate must be positive")
        if self.priority_band < 0:
            raise ValueError("priority_band must be non-negative")
        if not self.claim_ids:
            raise ValueError("prompt fragments require claim provenance")
        if self.actual_token_usage is not None and self.actual_token_usage <= 0:
            raise ValueError("actual_token_usage must be positive when provided")

        compressed_fields = (
            self.compressed_text,
            self.compressed_token_estimate,
            self.degradation_reason,
        )
        if any(value is not None for value in compressed_fields):
            if any(value is None for value in compressed_fields):
                raise ValueError("compressed prompt fragments require text, token estimate, and reason")
            if self.compressed_token_estimate is None or self.compressed_token_estimate <= 0:
                raise ValueError("compressed_token_estimate must be positive")
            if self.compressed_token_estimate >= self.token_estimate:
                raise ValueError("compressed_token_estimate must be smaller than token_estimate")

        if self.disclosure_action is not PromptDisclosureAction.ALLOW and self.disclosure_reason is None:
            raise ValueError("non-allow disclosure actions require an explicit reason")

    @property
    def can_degrade(self) -> bool:
        return self.compressed_text is not None


@dataclass(frozen=True, slots=True)
class PromptPlanFragment:
    fragment_id: str
    source_view: ViewKind
    text: str
    token_estimate: int
    claim_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    epistemic_action: PromptExposure
    disclosure_action: PromptDisclosureAction
    selection_reason: str
    disclosure_reason: str | None = None
    degradation_reason: str | None = None
    actual_token_usage: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_id", _clean_text(self.fragment_id, field_name="fragment_id"))
        object.__setattr__(self, "text", _clean_text(self.text, field_name="text"))
        object.__setattr__(self, "selection_reason", _clean_text(self.selection_reason, field_name="selection_reason"))
        object.__setattr__(
            self,
            "claim_ids",
            _dedupe_cleaned(self.claim_ids, field_name="claim_ids"),
        )
        object.__setattr__(
            self,
            "observation_ids",
            _dedupe_cleaned(self.observation_ids, field_name="observation_ids"),
        )
        object.__setattr__(
            self,
            "disclosure_reason",
            _optional_clean_text(self.disclosure_reason, field_name="disclosure_reason"),
        )
        object.__setattr__(
            self,
            "degradation_reason",
            _optional_clean_text(self.degradation_reason, field_name="degradation_reason"),
        )

        if self.token_estimate <= 0:
            raise ValueError("token_estimate must be positive")


@dataclass(frozen=True, slots=True)
class PromptPlanExclusion:
    fragment_id: str
    source_view: ViewKind
    reason: str
    claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_id", _clean_text(self.fragment_id, field_name="fragment_id"))
        object.__setattr__(self, "reason", _clean_text(self.reason, field_name="reason"))
        object.__setattr__(
            self,
            "claim_ids",
            _dedupe_cleaned(self.claim_ids, field_name="claim_ids"),
        )


@dataclass(frozen=True, slots=True)
class PromptPlan:
    policy_stamp: str
    hard_token_budget: int
    included_fragments: tuple[PromptPlanFragment, ...]
    excluded_fragments: tuple[PromptPlanExclusion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        if self.hard_token_budget <= 0:
            raise ValueError("hard_token_budget must be positive")

    @property
    def token_estimate(self) -> int:
        return sum(fragment.token_estimate for fragment in self.included_fragments)

    @property
    def actual_token_usage(self) -> int | None:
        if not self.included_fragments:
            return 0
        usages = tuple(fragment.actual_token_usage for fragment in self.included_fragments)
        if any(usage is None for usage in usages):
            return None
        return sum(usage for usage in usages if usage is not None)

    @property
    def fragment_ids_for_model(self) -> tuple[str, ...]:
        return tuple(fragment.fragment_id for fragment in self.included_fragments)

    @property
    def degradation_reasons(self) -> tuple[str, ...]:
        return tuple(
            fragment.degradation_reason
            for fragment in self.included_fragments
            if fragment.degradation_reason is not None
        )


def _candidate_sort_key(candidate: PromptFragmentCandidate) -> tuple[int, int, int, str]:
    return (candidate.priority_band, -candidate.utility_weight, candidate.token_estimate, candidate.fragment_id)


def _budget_failure_reason(
    *,
    candidate: PromptFragmentCandidate,
    config: PromptPlannerConfig,
    used_tokens: int,
    soft_budget_usage: dict[str, int],
    token_estimate: int,
) -> str | None:
    if candidate.soft_budget_group is not None and candidate.soft_budget_group in config.soft_token_budgets:
        soft_limit = config.soft_token_budgets[candidate.soft_budget_group]
        if soft_budget_usage.get(candidate.soft_budget_group, 0) + token_estimate > soft_limit:
            return f"soft_budget_exhausted:{candidate.soft_budget_group}"
    if used_tokens + token_estimate > config.hard_token_budget:
        return "hard_budget_exhausted"
    return None


def _selection_reason(
    *,
    epistemic_action: PromptExposure,
    disclosure_action: PromptDisclosureAction,
    degradation_reason: str | None,
) -> str:
    if degradation_reason is not None:
        return "selected_after_degradation"
    if disclosure_action is PromptDisclosureAction.REDACT:
        return "selected_after_disclosure_transform"
    if epistemic_action is PromptExposure.QUALIFY:
        return "selected_with_epistemic_qualification"
    return "selected_by_priority"


def plan_prompt_view(
    *,
    policy_stamp: str,
    config: PromptPlannerConfig,
    candidates: tuple[PromptFragmentCandidate, ...],
) -> PromptPlan:
    included: list[PromptPlanFragment] = []
    excluded: list[PromptPlanExclusion] = []
    used_tokens = 0
    soft_budget_usage: dict[str, int] = {}

    for candidate in sorted(candidates, key=_candidate_sort_key):
        epistemic_action = prompt_exposure_for_status(candidate.epistemic_status)
        if epistemic_action is PromptExposure.SUPPRESS:
            excluded.append(
                PromptPlanExclusion(
                    fragment_id=candidate.fragment_id,
                    source_view=candidate.source_view,
                    reason=f"suppressed_by_epistemic_status:{candidate.epistemic_status.value}",
                    claim_ids=candidate.claim_ids,
                )
            )
            continue

        if candidate.disclosure_action is PromptDisclosureAction.WITHHOLD:
            excluded.append(
                PromptPlanExclusion(
                    fragment_id=candidate.fragment_id,
                    source_view=candidate.source_view,
                    reason=candidate.disclosure_reason or "withheld",
                    claim_ids=candidate.claim_ids,
                )
            )
            continue

        selected_text = candidate.text
        selected_tokens = candidate.token_estimate
        selected_degradation_reason: str | None = None
        selected_actual_usage = candidate.actual_token_usage

        failure_reason = _budget_failure_reason(
            candidate=candidate,
            config=config,
            used_tokens=used_tokens,
            soft_budget_usage=soft_budget_usage,
            token_estimate=selected_tokens,
        )
        if failure_reason is not None:
            if candidate.can_degrade:
                compressed_failure_reason = _budget_failure_reason(
                    candidate=candidate,
                    config=config,
                    used_tokens=used_tokens,
                    soft_budget_usage=soft_budget_usage,
                    token_estimate=candidate.compressed_token_estimate or 0,
                )
                if compressed_failure_reason is None:
                    selected_text = candidate.compressed_text or candidate.text
                    selected_tokens = candidate.compressed_token_estimate or candidate.token_estimate
                    selected_degradation_reason = candidate.degradation_reason
                    selected_actual_usage = None
                else:
                    excluded.append(
                        PromptPlanExclusion(
                            fragment_id=candidate.fragment_id,
                            source_view=candidate.source_view,
                            reason=compressed_failure_reason,
                            claim_ids=candidate.claim_ids,
                        )
                    )
                    continue
            else:
                excluded.append(
                    PromptPlanExclusion(
                        fragment_id=candidate.fragment_id,
                        source_view=candidate.source_view,
                        reason=failure_reason,
                        claim_ids=candidate.claim_ids,
                    )
                )
                continue

        included.append(
            PromptPlanFragment(
                fragment_id=candidate.fragment_id,
                source_view=candidate.source_view,
                text=selected_text,
                token_estimate=selected_tokens,
                claim_ids=candidate.claim_ids,
                observation_ids=candidate.observation_ids,
                epistemic_action=epistemic_action,
                disclosure_action=candidate.disclosure_action,
                selection_reason=_selection_reason(
                    epistemic_action=epistemic_action,
                    disclosure_action=candidate.disclosure_action,
                    degradation_reason=selected_degradation_reason,
                ),
                disclosure_reason=candidate.disclosure_reason,
                degradation_reason=selected_degradation_reason,
                actual_token_usage=selected_actual_usage,
            )
        )
        used_tokens += selected_tokens
        if candidate.soft_budget_group is not None:
            soft_budget_usage[candidate.soft_budget_group] = (
                soft_budget_usage.get(candidate.soft_budget_group, 0) + selected_tokens
            )

    return PromptPlan(
        policy_stamp=policy_stamp,
        hard_token_budget=config.hard_token_budget,
        included_fragments=tuple(included),
        excluded_fragments=tuple(excluded),
    )
