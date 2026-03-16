"""Belief revision and epistemic-status invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from continuity.store.claims import AggregationMode, Claim, ClaimRelationKind, MemoryLocus


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _claim_sort_key(claim: Claim) -> tuple[datetime, datetime, str]:
    return (claim.learned_at, claim.observed_at, claim.claim_id)


def _claims_share_locus(claims: tuple[Claim, ...]) -> None:
    first_locus = claims[0].locus
    expected = (first_locus.address, first_locus.aggregation_mode)
    for claim in claims[1:]:
        current = (claim.locus.address, claim.locus.aggregation_mode)
        if current != expected:
            raise ValueError("belief projection requires one shared locus and aggregation mode")


def _superseded_claim_ids(claims: tuple[Claim, ...]) -> set[str]:
    superseded: set[str] = set()
    for claim in claims:
        for relation in claim.relations:
            if relation.kind in {ClaimRelationKind.SUPERSEDES, ClaimRelationKind.CORRECTS}:
                superseded.add(relation.related_claim_id)
    return superseded


def _claim_is_stale(claim: Claim, *, as_of: datetime) -> bool:
    return claim.valid_to is not None and claim.valid_to < as_of


def _claims_conflict(claims: tuple[Claim, ...]) -> bool:
    active_ids = {claim.claim_id for claim in claims}
    for claim in claims:
        for relation in claim.relations:
            if relation.kind is ClaimRelationKind.CONTRADICTS and relation.related_claim_id in active_ids:
                return True
    return False


class EpistemicStatus(StrEnum):
    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    TENTATIVE = "tentative"
    CONFLICTED = "conflicted"
    STALE = "stale"
    NEEDS_CONFIRMATION = "needs_confirmation"


class EpistemicTarget(StrEnum):
    CLAIM = "claim"
    LOCUS_RESOLUTION = "locus_resolution"
    COMPILED_VIEW = "compiled_view"
    ANSWER = "answer"


class AnswerMode(StrEnum):
    ASSERT = "assert"
    QUALIFY = "qualify"
    ABSTAIN = "abstain"
    ASK_CONFIRMATION = "ask_confirmation"


class PromptExposure(StrEnum):
    INCLUDE = "include"
    QUALIFY = "qualify"
    SUPPRESS = "suppress"


def answer_mode_for_status(status: EpistemicStatus) -> AnswerMode:
    if status is EpistemicStatus.SUPPORTED:
        return AnswerMode.ASSERT
    if status in {EpistemicStatus.TENTATIVE, EpistemicStatus.STALE}:
        return AnswerMode.QUALIFY
    if status is EpistemicStatus.NEEDS_CONFIRMATION:
        return AnswerMode.ASK_CONFIRMATION
    return AnswerMode.ABSTAIN


def prompt_exposure_for_status(status: EpistemicStatus) -> PromptExposure:
    if status is EpistemicStatus.SUPPORTED:
        return PromptExposure.INCLUDE
    if status in {EpistemicStatus.TENTATIVE, EpistemicStatus.STALE}:
        return PromptExposure.QUALIFY
    return PromptExposure.SUPPRESS


@dataclass(frozen=True, slots=True)
class EpistemicAssessment:
    status: EpistemicStatus
    target: EpistemicTarget
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))

    @property
    def answer_mode(self) -> AnswerMode:
        return answer_mode_for_status(self.status)

    @property
    def prompt_exposure(self) -> PromptExposure:
        return prompt_exposure_for_status(self.status)


@dataclass(frozen=True, slots=True)
class BeliefProjection:
    locus: MemoryLocus
    active_claim_ids: tuple[str, ...]
    historical_claim_ids: tuple[str, ...]
    epistemic: EpistemicAssessment

    def __post_init__(self) -> None:
        history = tuple(dict.fromkeys(self.historical_claim_ids))
        active = tuple(dict.fromkeys(self.active_claim_ids))
        if any(claim_id not in history for claim_id in active):
            raise ValueError("active claims must remain part of the historical claim set")

        object.__setattr__(self, "historical_claim_ids", history)
        object.__setattr__(self, "active_claim_ids", active)

    @property
    def retrieval_order(self) -> tuple[str, ...]:
        historical_tail = tuple(
            claim_id
            for claim_id in self.historical_claim_ids
            if claim_id not in self.active_claim_ids
        )
        return self.active_claim_ids + historical_tail


def _projection(
    *,
    locus: MemoryLocus,
    active_claims: tuple[Claim, ...],
    historical_claims: tuple[Claim, ...],
    status: EpistemicStatus,
    rationale: str,
) -> BeliefProjection:
    return BeliefProjection(
        locus=locus,
        active_claim_ids=tuple(claim.claim_id for claim in active_claims),
        historical_claim_ids=tuple(claim.claim_id for claim in historical_claims),
        epistemic=EpistemicAssessment(
            status=status,
            target=EpistemicTarget.LOCUS_RESOLUTION,
            rationale=rationale,
        ),
    )


def _resolve_singular_locus(
    claims: tuple[Claim, ...],
    *,
    as_of: datetime,
) -> tuple[tuple[Claim, ...], EpistemicStatus, str]:
    current_claims = tuple(claim for claim in claims if not _claim_is_stale(claim, as_of=as_of))
    if _claims_conflict(current_claims):
        return (), EpistemicStatus.CONFLICTED, "contradictory claims remain unresolved"
    if current_claims:
        return (current_claims[0],), EpistemicStatus.SUPPORTED, "latest supported claim wins"
    if claims:
        return (claims[0],), EpistemicStatus.STALE, "only stale claims remain for the locus"
    return (), EpistemicStatus.UNKNOWN, "no claims remain after revision"


def _resolve_set_union_locus(
    claims: tuple[Claim, ...],
    *,
    as_of: datetime,
) -> tuple[tuple[Claim, ...], EpistemicStatus, str]:
    current_claims = tuple(claim for claim in claims if not _claim_is_stale(claim, as_of=as_of))
    if _claims_conflict(current_claims):
        return (), EpistemicStatus.CONFLICTED, "set-union locus contains contradictory current claims"
    if current_claims:
        return current_claims, EpistemicStatus.SUPPORTED, "non-conflicting current claims remain active"
    if claims:
        return (claims[0],), EpistemicStatus.STALE, "only stale claims remain for the locus"
    return (), EpistemicStatus.UNKNOWN, "no claims remain after revision"


def _resolve_timeline_locus(
    claims: tuple[Claim, ...],
    *,
    as_of: datetime,
) -> tuple[tuple[Claim, ...], EpistemicStatus, str]:
    ordered = tuple(sorted(claims, key=lambda claim: (claim.valid_from or claim.observed_at, claim.claim_id)))
    if not ordered:
        return (), EpistemicStatus.UNKNOWN, "timeline locus has no claims"
    if all(_claim_is_stale(claim, as_of=as_of) for claim in ordered):
        return ordered, EpistemicStatus.STALE, "timeline retains only stale history"
    return ordered, EpistemicStatus.SUPPORTED, "timeline preserves ordered claim history"


def resolve_locus_belief(
    claims: tuple[Claim, ...],
    *,
    as_of: datetime,
) -> BeliefProjection:
    if not claims:
        raise ValueError("belief projection requires at least one claim")

    _validate_timestamp(as_of, field_name="as_of")
    _claims_share_locus(claims)

    ordered_history = tuple(sorted(claims, key=_claim_sort_key, reverse=True))
    surviving_claims = tuple(
        claim
        for claim in ordered_history
        if claim.claim_id not in _superseded_claim_ids(ordered_history)
    )
    locus = ordered_history[0].locus

    if locus.aggregation_mode in {AggregationMode.LATEST_WINS, AggregationMode.STATE_MACHINE}:
        active_claims, status, rationale = _resolve_singular_locus(surviving_claims, as_of=as_of)
    elif locus.aggregation_mode is AggregationMode.SET_UNION:
        active_claims, status, rationale = _resolve_set_union_locus(surviving_claims, as_of=as_of)
    elif locus.aggregation_mode is AggregationMode.TIMELINE:
        active_claims, status, rationale = _resolve_timeline_locus(surviving_claims, as_of=as_of)
    else:
        raise ValueError(f"unsupported aggregation mode: {locus.aggregation_mode}")

    return _projection(
        locus=locus,
        active_claims=active_claims,
        historical_claims=ordered_history,
        status=status,
        rationale=rationale,
    )
