"""Retrieval runtime and compiled-view assembly for Continuity."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from continuity.config import normalize_recall_mode
from continuity.disclosure import (
    DisclosureAction,
    DisclosureContext,
    disclosure_policy_for,
    evaluate_disclosure,
)
from continuity.epistemics import EpistemicStatus, resolve_locus_belief
from continuity.forgetting import ForgettingTarget, ForgettingTargetKind
from continuity.index.zvec_index import (
    EmbeddingClientProtocol,
    IndexSearchHit,
    IndexSourceKind,
    VectorBackendProtocol,
    ZvecIndex,
)
from continuity.ontology import MemoryPartition
from continuity.outcomes import OutcomeTarget
from continuity.policy import PolicyPack, get_policy_pack
from continuity.prompt_planner import (
    PromptDisclosureAction,
    PromptFragmentCandidate,
    PromptPlannerConfig,
    plan_prompt_view,
)
from continuity.resolution_queue import ResolutionSurface
from continuity.service import (
    ResolvedServiceRequest,
    ServiceExecutor,
    ServiceOperation,
    ServiceResponse,
)
from continuity.store.belief_revision import BeliefStateRepository, StoredBeliefState
from continuity.store.claims import (
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    MemoryLocus,
    Observation,
    Subject,
    SubjectKind,
)
from continuity.store.sqlite import SQLiteRepository
from continuity.utility import CompiledUtilityWeight
from continuity.views import CompiledView, ViewKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _optional_clean_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _claim_type_root(claim_type: str) -> str:
    return _clean_text(claim_type, field_name="claim_type").split(".", 1)[0]


def _stringify_value(value: Any) -> str:
    if isinstance(value, Mapping):
        if len(value) == 1:
            return str(next(iter(value.values())))
        return ", ".join(
            f"{str(key).replace('_', ' ')}={_stringify_value(nested)}"
            for key, nested in sorted(value.items())
        )
    if isinstance(value, tuple | list):
        return ", ".join(_stringify_value(item) for item in value)
    return str(value)


def _token_estimate(text: str) -> int:
    return max(1, len(text.split()))


def _transport(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return tuple(_transport(item) for item in value)
    if isinstance(value, list):
        return tuple(_transport(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _transport(nested) for key, nested in value.items()}
    if dataclass_is_instance(value):
        payload = {
            key: _transport(nested)
            for key, nested in value.__dict__.items()
        }
        return payload
    return str(value)


def dataclass_is_instance(value: object) -> bool:
    return hasattr(value, "__dataclass_fields__")


class SubjectResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class SubjectResolution:
    reference_text: str
    status: SubjectResolutionStatus
    subject_id: str | None = None
    matched_by: str | None = None
    candidate_subject_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssembledView:
    compiled_view: CompiledView
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class SearchResult:
    view: AssembledView
    score: float
    source_kind: IndexSourceKind
    record_id: str
    excerpt: str


class ContinuityContextBuilder:
    """Builds deterministic retrieval views from local SQLite plus zvec state."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        embedding_client: EmbeddingClientProtocol,
        vector_backend: VectorBackendProtocol,
        policy_name: str = "hermes_v1",
        prompt_hard_token_budget: int | None = None,
        prompt_soft_token_budgets: Mapping[str, int] | None = None,
    ) -> None:
        self._connection = connection
        self._repository = SQLiteRepository(connection)
        self._beliefs = BeliefStateRepository(connection)
        self._policy = get_policy_pack(_clean_text(policy_name, field_name="policy_name"))
        self._index = ZvecIndex(
            connection=connection,
            embedding_client=embedding_client,
            backend=vector_backend,
            policy_stamp=self._policy.policy_stamp,
        )
        self._prompt_config = PromptPlannerConfig(
            hard_token_budget=prompt_hard_token_budget or self._policy.prompt_budget_tokens,
            soft_token_budgets=dict(prompt_soft_token_budgets or {}),
        )

    def resolve_subject_reference(
        self,
        reference_text: str,
        *,
        subject_kind: SubjectKind | None = None,
    ) -> SubjectResolution:
        cleaned_reference = _clean_text(reference_text, field_name="reference_text")
        matches = tuple(
            subject
            for subject in self._repository.list_subjects(kind=subject_kind)
            if subject.matches_name(cleaned_reference)
        )
        if not matches:
            return SubjectResolution(
                reference_text=cleaned_reference,
                status=SubjectResolutionStatus.MISSING,
            )
        if len(matches) > 1:
            return SubjectResolution(
                reference_text=cleaned_reference,
                status=SubjectResolutionStatus.AMBIGUOUS,
                candidate_subject_ids=tuple(subject.subject_id for subject in matches),
            )

        subject = matches[0]
        matched_by = "canonical_name"
        if subject.canonical_name.strip().casefold() != cleaned_reference.casefold():
            matched_by = "alias"
        return SubjectResolution(
            reference_text=cleaned_reference,
            status=SubjectResolutionStatus.RESOLVED,
            subject_id=subject.subject_id,
            matched_by=matched_by,
            candidate_subject_ids=(subject.subject_id,),
        )

    def search(
        self,
        *,
        query_text: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
        limit: int | None = None,
        subject_id: str | None = None,
        view_kinds: tuple[ViewKind, ...] = (),
    ) -> tuple[SearchResult, ...]:
        cleaned_limit = 5 if limit is None else limit
        if cleaned_limit <= 0:
            return ()

        snapshot_id = self._resolve_snapshot_id(target_snapshot_id)
        self._index.rebuild_from_sqlite()
        cleaned_query = _clean_text(query_text, field_name="query_text")
        resolved_subject_id = _optional_clean_text(subject_id, field_name="subject_id")
        raw_hits = (
            *self._index.search(
                cleaned_query,
                topk=max(cleaned_limit * 4, cleaned_limit),
                subject_id=resolved_subject_id,
                source_kinds=(IndexSourceKind.BELIEF_STATE, IndexSourceKind.CLAIM),
            ),
            *self._index.search(
                cleaned_query,
                topk=max(cleaned_limit * 2, cleaned_limit),
                subject_id=resolved_subject_id,
                source_kinds=(IndexSourceKind.OBSERVATION, IndexSourceKind.COMPILED_VIEW),
            ),
        )
        allowed_view_kinds = frozenset(view_kinds)
        results: list[SearchResult] = []
        seen_view_keys: set[str] = set()

        for hit in raw_hits:
            if hit.record.source_kind is IndexSourceKind.SESSION_MESSAGE:
                continue
            if hit.score <= 0.0:
                continue
            try:
                view = self._view_for_search_hit(
                    hit,
                    disclosure_context=disclosure_context,
                    snapshot_id=snapshot_id,
                )
            except LookupError:
                continue
            if view is None:
                continue
            if allowed_view_kinds and view.compiled_view.kind not in allowed_view_kinds:
                continue
            if view.compiled_view.view_key in seen_view_keys:
                continue
            seen_view_keys.add(view.compiled_view.view_key)
            results.append(
                SearchResult(
                    view=view,
                    score=hit.score,
                    source_kind=hit.record.source_kind,
                    record_id=hit.record.record_id,
                    excerpt=hit.record.document_text,
                )
            )
        ordered_results = tuple(
            sorted(
                results,
                key=lambda result: (
                    self._search_priority(result.view.compiled_view.kind),
                    -result.score,
                    result.view.compiled_view.view_key,
                ),
            )
        )
        return ordered_results[:cleaned_limit]

    def build_state_view(
        self,
        *,
        subject_id: str,
        locus_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> AssembledView:
        projection, visible_claims = self._visible_projection(
            subject_id=subject_id,
            locus_key=locus_key,
            disclosure_context=disclosure_context,
        )
        if projection.projection.locus.aggregation_mode not in {
            AggregationMode.LATEST_WINS,
            AggregationMode.STATE_MACHINE,
        }:
            raise ValueError("state_view requires a singular locus aggregation mode")

        active_claims = tuple(
            claim
            for claim in visible_claims
            if claim.claim_id in projection.projection.active_claim_ids
        ) or visible_claims[:1]
        active_values = tuple(_stringify_value(claim.value) for claim in active_claims)
        subject = self._subject(subject_id)

        return AssembledView(
            compiled_view=CompiledView(
                kind=ViewKind.STATE,
                view_key=self._state_view_key(subject_id, locus_key),
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                claim_ids=tuple(claim.claim_id for claim in visible_claims),
                epistemic_status=projection.projection.epistemic.status,
            ),
            payload={
                "subject_id": subject_id,
                "subject_name": subject.canonical_name,
                "locus_key": locus_key,
                "summary": f"{subject.canonical_name} currently prefers {active_values[0]}.",
                "active_values": active_values,
                "active_claim_ids": tuple(claim.claim_id for claim in active_claims),
                "historical_claim_ids": tuple(
                    claim_id
                    for claim_id in projection.projection.historical_claim_ids
                    if claim_id in {claim.claim_id for claim in visible_claims}
                ),
                "epistemic_status": projection.projection.epistemic.status,
                "rationale": projection.projection.epistemic.rationale,
            },
        )

    def build_set_view(
        self,
        *,
        subject_id: str,
        locus_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> AssembledView:
        projection, visible_claims = self._visible_projection(
            subject_id=subject_id,
            locus_key=locus_key,
            disclosure_context=disclosure_context,
        )
        if projection.projection.locus.aggregation_mode is not AggregationMode.SET_UNION:
            raise ValueError("set_view requires a set-union locus")

        active_claims = tuple(
            claim
            for claim in visible_claims
            if claim.claim_id in projection.projection.active_claim_ids
        ) or visible_claims
        subject = self._subject(subject_id)
        items = tuple(_stringify_value(claim.value) for claim in active_claims)

        return AssembledView(
            compiled_view=CompiledView(
                kind=ViewKind.SET,
                view_key=self._set_view_key(subject_id, locus_key),
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                claim_ids=tuple(claim.claim_id for claim in visible_claims),
                epistemic_status=projection.projection.epistemic.status,
            ),
            payload={
                "subject_id": subject_id,
                "subject_name": subject.canonical_name,
                "locus_key": locus_key,
                "items": items,
                "summary": f"{subject.canonical_name} currently has {', '.join(items)}.",
                "active_claim_ids": tuple(claim.claim_id for claim in active_claims),
                "epistemic_status": projection.projection.epistemic.status,
            },
        )

    def build_timeline_view(
        self,
        *,
        subject_id: str,
        locus_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> AssembledView:
        projection, visible_claims = self._visible_projection(
            subject_id=subject_id,
            locus_key=locus_key,
            disclosure_context=disclosure_context,
        )

        ordered_claims = tuple(
            sorted(
                visible_claims,
                key=lambda claim: (
                    claim.valid_from or claim.observed_at,
                    claim.learned_at,
                    claim.claim_id,
                ),
            )
        )
        observation_ids = tuple(
            dict.fromkeys(
                observation_id
                for claim in ordered_claims
                for observation_id in claim.provenance.observation_ids
                if self._observation_visible(observation_id)
            )
        )

        return AssembledView(
            compiled_view=CompiledView(
                kind=ViewKind.TIMELINE,
                view_key=self._timeline_view_key(subject_id, locus_key),
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                claim_ids=tuple(claim.claim_id for claim in ordered_claims),
                observation_ids=observation_ids,
                epistemic_status=projection.projection.epistemic.status,
            ),
            payload={
                "subject_id": subject_id,
                "locus_key": locus_key,
                "entries": tuple(
                    {
                        "claim_id": claim.claim_id,
                        "value": _stringify_value(claim.value),
                        "observed_at": claim.observed_at.isoformat(),
                        "learned_at": claim.learned_at.isoformat(),
                    }
                    for claim in ordered_claims
                ),
                "epistemic_status": projection.projection.epistemic.status,
            },
        )

    def build_profile_view(
        self,
        *,
        subject_id: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> AssembledView:
        subject = self._subject(subject_id)
        states = self._beliefs.list_states(subject_id=subject_id, policy_stamp=self._policy.policy_stamp)
        assembled_entries: list[tuple[int, str, AssembledView]] = []
        for state in states:
            locus = state.projection.locus
            if locus.aggregation_mode is AggregationMode.TIMELINE:
                continue
            try:
                if locus.aggregation_mode is AggregationMode.SET_UNION:
                    view = self.build_set_view(
                        subject_id=subject_id,
                        locus_key=locus.locus_key,
                        disclosure_context=disclosure_context,
                        target_snapshot_id=target_snapshot_id,
                    )
                else:
                    view = self.build_state_view(
                        subject_id=subject_id,
                        locus_key=locus.locus_key,
                        disclosure_context=disclosure_context,
                        target_snapshot_id=target_snapshot_id,
                    )
            except LookupError:
                continue
            assembled_entries.append(
                (
                    self._retrieval_rank_for_claim_ids(view.compiled_view.claim_ids),
                    locus.locus_key,
                    view,
                )
            )

        if not assembled_entries:
            raise LookupError(f"no visible profile entries remain for {subject_id}")

        ordered_views = tuple(
            view
            for _, _, view in sorted(assembled_entries, key=lambda item: (item[0], item[1]))
        )
        claim_ids = tuple(
            dict.fromkeys(
                claim_id
                for view in ordered_views
                for claim_id in view.compiled_view.claim_ids
            )
        )

        return AssembledView(
            compiled_view=CompiledView(
                kind=ViewKind.PROFILE,
                view_key=self._profile_view_key(subject_id),
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                claim_ids=claim_ids,
                epistemic_status=EpistemicStatus.SUPPORTED,
            ),
            payload={
                "subject_id": subject_id,
                "subject_name": subject.canonical_name,
                "locus_keys": tuple(view.payload["locus_key"] for view in ordered_views),
                "entries": tuple(
                    {
                        "view_kind": view.compiled_view.kind.value,
                        "locus_key": view.payload["locus_key"],
                        "summary": view.payload["summary"],
                    }
                    for view in ordered_views
                ),
            },
        )

    def build_prompt_view(
        self,
        *,
        session_id: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
        recall_mode: str = "hybrid",
    ) -> AssembledView:
        normalized_recall_mode = normalize_recall_mode(recall_mode)
        if normalized_recall_mode == "tools":
            raise ValueError("tools recall mode does not inject prompt context")

        session_subject_id = disclosure_context.viewer.active_user_id
        if session_subject_id is None:
            raise ValueError("prompt assembly requires an active_user_id")

        state_views = self._state_like_views_for_subject(
            subject_id=session_subject_id,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )
        profile_view = self.build_profile_view(
            subject_id=session_subject_id,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )
        evidence_view = self.build_evidence_view(
            target_kind="claim",
            target_id=state_views[0].compiled_view.claim_ids[0],
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )
        timeline_views = tuple(
            view
            for view in self._timeline_views_for_subject(
                subject_id=session_subject_id,
                disclosure_context=disclosure_context,
                target_snapshot_id=target_snapshot_id,
            )
        )

        candidates = (
            self._prompt_candidate_for_view(state_views[0]),
            self._prompt_candidate_for_view(profile_view),
            self._prompt_candidate_for_view(evidence_view),
            *tuple(self._prompt_candidate_for_view(view) for view in timeline_views),
        )
        prompt_plan = plan_prompt_view(
            policy_stamp=self._policy.policy_stamp,
            config=self._prompt_config,
            candidates=candidates,
        )
        claim_ids = tuple(
            dict.fromkeys(
                claim_id
                for fragment in prompt_plan.included_fragments
                for claim_id in fragment.claim_ids
            )
        )
        if not claim_ids:
            raise LookupError("prompt assembly requires at least one claim-backed fragment")

        auxiliary_prompt_memory = self._auxiliary_prompt_memory(
            session_id=session_id,
            subject_id=session_subject_id,
        )
        follow_ups = self._follow_up_payloads(
            subject_id=session_subject_id,
            surface=ResolutionSurface.PROMPT_QUEUE,
        )

        return AssembledView(
            compiled_view=CompiledView(
                kind=ViewKind.PROMPT,
                view_key=self._prompt_view_key(session_id),
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                claim_ids=claim_ids,
                epistemic_status=EpistemicStatus.SUPPORTED,
            ),
            payload={
                "session_id": session_id,
                "recall_mode": normalized_recall_mode,
                "prompt_plan": {
                    "included_fragment_ids": tuple(
                        fragment.fragment_id for fragment in prompt_plan.included_fragments
                    ),
                    "excluded_fragments": {
                        fragment.fragment_id: fragment.reason
                        for fragment in prompt_plan.excluded_fragments
                    },
                    "degradation_reasons": {
                        fragment.fragment_id: fragment.degradation_reason
                        for fragment in prompt_plan.included_fragments
                        if fragment.degradation_reason is not None
                    },
                    "token_estimate": prompt_plan.token_estimate,
                    "model_text": "\n".join(
                        fragment.text for fragment in prompt_plan.included_fragments
                    ),
                },
                "auxiliary_prompt_memory": auxiliary_prompt_memory,
                "follow_ups": follow_ups,
            },
        )

    def build_evidence_view(
        self,
        *,
        target_kind: str,
        target_id: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> AssembledView:
        cleaned_target_kind = _clean_text(target_kind, field_name="target_kind")
        cleaned_target_id = _clean_text(target_id, field_name="target_id")
        claims: tuple[Claim, ...]
        observation_ids: tuple[str, ...]

        if cleaned_target_kind == "claim":
            claim = self._repository.read_claim(cleaned_target_id)
            if claim is None:
                raise LookupError(cleaned_target_id)
            if not self._claim_visible(claim, disclosure_context):
                raise LookupError(cleaned_target_id)
            claims = (claim,)
            observation_ids = tuple(
                observation_id
                for observation_id in claim.provenance.observation_ids
                if self._observation_visible(observation_id)
            )
        elif cleaned_target_kind == "observation":
            observation_ids = (cleaned_target_id,)
            claims = tuple(
                claim
                for claim in self._repository.list_claims()
                if cleaned_target_id in claim.provenance.observation_ids
                and self._claim_visible(claim, disclosure_context)
            )
            if not claims:
                raise LookupError(cleaned_target_id)
        elif cleaned_target_kind == "view":
            raise LookupError("compiled view evidence resolution is not implemented yet")
        else:
            raise ValueError(f"unsupported evidence target_kind: {cleaned_target_kind}")

        observations = tuple(
            self._repository.read_observation(observation_id)
            for observation_id in observation_ids
        )
        visible_observations = tuple(
            observation
            for observation in observations
            if observation is not None
        )
        if not visible_observations:
            raise LookupError(cleaned_target_id)

        return AssembledView(
            compiled_view=CompiledView(
                kind=ViewKind.EVIDENCE,
                view_key=self._evidence_view_key(cleaned_target_kind, cleaned_target_id),
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                claim_ids=tuple(dict.fromkeys(claim.claim_id for claim in claims)),
                observation_ids=tuple(
                    dict.fromkeys(observation.observation_id for observation in visible_observations)
                ),
                epistemic_status=EpistemicStatus.SUPPORTED,
            ),
            payload={
                "target_kind": cleaned_target_kind,
                "target_id": cleaned_target_id,
                "claims": tuple(
                    {
                        "claim_id": claim.claim_id,
                        "text": self._claim_summary(claim),
                    }
                    for claim in claims
                ),
                "observations": tuple(
                    {
                        "observation_id": observation.observation_id,
                        "content": observation.content,
                    }
                    for observation in visible_observations
                ),
            },
        )

    def list_memory_follow_ups(
        self,
        *,
        subject_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[dict[str, str], ...]:
        return self._follow_up_payloads(
            subject_id=subject_id,
            surface=ResolutionSurface.HOST_API,
            limit=limit,
        )

    def service_executors(self) -> dict[ServiceOperation, ServiceExecutor]:
        return {
            ServiceOperation.SEARCH: self._execute_search,
            ServiceOperation.GET_STATE_VIEW: self._execute_state_view,
            ServiceOperation.GET_TIMELINE_VIEW: self._execute_timeline_view,
            ServiceOperation.GET_PROFILE_VIEW: self._execute_profile_view,
            ServiceOperation.GET_PROMPT_VIEW: self._execute_prompt_view,
            ServiceOperation.LIST_MEMORY_FOLLOW_UPS: self._execute_follow_ups,
            ServiceOperation.RESOLVE_SUBJECT: self._execute_resolve_subject,
            ServiceOperation.INSPECT_EVIDENCE: self._execute_evidence_view,
        }

    def _execute_search(self, request: ResolvedServiceRequest) -> ServiceResponse:
        payload = request.request.payload
        view_kinds = tuple(ViewKind(value) for value in payload.get("view_kinds", ()))
        results = self.search(
            query_text=str(payload["query_text"]),
            disclosure_context=request.request.disclosure_context,
            target_snapshot_id=request.request.target_snapshot_id,
            limit=int(payload.get("limit", 5)),
            subject_id=payload.get("subject_id"),
            view_kinds=view_kinds,
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={
                "snapshot_id": self._resolve_snapshot_id(request.request.target_snapshot_id),
                "results": tuple(self._search_payload(result) for result in results),
            },
        )

    def _execute_state_view(self, request: ResolvedServiceRequest) -> ServiceResponse:
        subject_id, locus_key = self._parse_subject_locus_view_key(
            str(request.request.payload["view_key"]),
            expected_prefix="state",
        )
        view = self.build_state_view(
            subject_id=subject_id,
            locus_key=locus_key,
            disclosure_context=request.request.disclosure_context,
            target_snapshot_id=request.request.target_snapshot_id,
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"view": self._view_payload(view)},
        )

    def _execute_timeline_view(self, request: ResolvedServiceRequest) -> ServiceResponse:
        subject_id, locus_key = self._parse_subject_locus_view_key(
            str(request.request.payload["view_key"]),
            expected_prefix="timeline",
        )
        view = self.build_timeline_view(
            subject_id=subject_id,
            locus_key=locus_key,
            disclosure_context=request.request.disclosure_context,
            target_snapshot_id=request.request.target_snapshot_id,
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"view": self._view_payload(view)},
        )

    def _execute_profile_view(self, request: ResolvedServiceRequest) -> ServiceResponse:
        subject_id = self._parse_subject_view_key(
            str(request.request.payload["view_key"]),
            expected_prefix="profile",
        )
        view = self.build_profile_view(
            subject_id=subject_id,
            disclosure_context=request.request.disclosure_context,
            target_snapshot_id=request.request.target_snapshot_id,
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"view": self._view_payload(view)},
        )

    def _execute_prompt_view(self, request: ResolvedServiceRequest) -> ServiceResponse:
        session_id = self._parse_prompt_view_key(str(request.request.payload["view_key"]))
        view = self.build_prompt_view(
            session_id=session_id,
            disclosure_context=request.request.disclosure_context,
            target_snapshot_id=request.request.target_snapshot_id,
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"view": self._view_payload(view)},
        )

    def _execute_follow_ups(self, request: ResolvedServiceRequest) -> ServiceResponse:
        payload = request.request.payload
        items = self.list_memory_follow_ups(
            subject_id=payload.get("subject_id"),
            limit=payload.get("limit"),
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"items": items},
        )

    def _execute_resolve_subject(self, request: ResolvedServiceRequest) -> ServiceResponse:
        payload = request.request.payload
        subject_kind = payload.get("subject_kind")
        resolved = self.resolve_subject_reference(
            str(payload["reference_text"]),
            subject_kind=None if subject_kind is None else SubjectKind(str(subject_kind)),
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={
                "reference_text": resolved.reference_text,
                "status": resolved.status.value,
                "subject_id": resolved.subject_id,
                "matched_by": resolved.matched_by,
                "candidate_subject_ids": resolved.candidate_subject_ids,
            },
        )

    def _execute_evidence_view(self, request: ResolvedServiceRequest) -> ServiceResponse:
        payload = request.request.payload
        view = self.build_evidence_view(
            target_kind=str(payload["target_kind"]),
            target_id=str(payload["target_id"]),
            disclosure_context=request.request.disclosure_context
            or DisclosureContext(  # pragma: no cover - defensive fallback
                viewer=None,  # type: ignore[arg-type]
                audience_principal=None,  # type: ignore[arg-type]
                channel=None,  # type: ignore[arg-type]
                purpose=None,  # type: ignore[arg-type]
                policy_stamp=self._policy.policy_stamp,
            ),
            target_snapshot_id=request.request.target_snapshot_id,
        )
        return ServiceResponse(
            operation=request.request.operation,
            payload={"view": self._view_payload(view)},
        )

    def _visible_projection(
        self,
        *,
        subject_id: str,
        locus_key: str,
        disclosure_context: DisclosureContext,
    ) -> tuple[StoredBeliefState, tuple[Claim, ...]]:
        projection = self._belief_for(subject_id=subject_id, locus_key=locus_key)
        claims_by_id = {
            claim.claim_id: claim
            for claim in self._repository.list_claims(subject_id=subject_id, locus_key=locus_key)
        }
        visible_claims = tuple(
            claim
            for claim_id in projection.projection.retrieval_order
            for claim in (claims_by_id.get(claim_id),)
            if claim is not None and self._claim_visible(claim, disclosure_context)
        )
        if not visible_claims:
            raise LookupError(f"no visible claims remain for {subject_id} {locus_key}")
        visible_projection = resolve_locus_belief(
            visible_claims,
            as_of=projection.as_of,
        )
        return (
            StoredBeliefState(
                belief_id=projection.belief_id,
                policy_stamp=projection.policy_stamp,
                projection=visible_projection,
                as_of=projection.as_of,
            ),
            visible_claims,
        )

    def _belief_for(self, *, subject_id: str, locus_key: str) -> StoredBeliefState:
        state = self._beliefs.read_current_state(
            subject_id=subject_id,
            locus_key=locus_key,
            policy_stamp=self._policy.policy_stamp,
        )
        if state is not None:
            return state

        claims = self._repository.list_claims(subject_id=subject_id, locus_key=locus_key)
        if not claims:
            raise LookupError(f"no claims exist for {subject_id} {locus_key}")
        projection = resolve_locus_belief(
            claims,
            as_of=datetime.now(timezone.utc),
        )
        return StoredBeliefState(
            belief_id=f"belief:{subject_id}:{locus_key}:dynamic",
            policy_stamp=self._policy.policy_stamp,
            projection=projection,
            as_of=datetime.now(timezone.utc),
        )

    def _subject(self, subject_id: str) -> Subject:
        subject = self._repository.read_subject(_clean_text(subject_id, field_name="subject_id"))
        if subject is None:
            raise LookupError(subject_id)
        return subject

    def _claim_visible(self, claim: Claim, disclosure_context: DisclosureContext) -> bool:
        if self._ordinary_read_blocked(
            ForgettingTarget(ForgettingTargetKind.CLAIM, claim.claim_id)
        ):
            return False
        if self._ordinary_read_blocked(
            ForgettingTarget(
                ForgettingTargetKind.LOCUS,
                self._locus_target_id(claim.locus),
            )
        ):
            return False
        if self._ordinary_read_blocked(
            ForgettingTarget(ForgettingTargetKind.SUBJECT, claim.subject_id)
        ):
            return False

        disclosure = evaluate_disclosure(
            disclosure_policy_for(claim.disclosure_policy),
            disclosure_context,
        )
        return disclosure.exposes_content

    def _observation_visible(self, observation_id: str) -> bool:
        return not self._ordinary_read_blocked(
            ForgettingTarget(ForgettingTargetKind.COMPILED_VIEW, observation_id)
        )

    def _ordinary_read_blocked(self, target: ForgettingTarget) -> bool:
        record = self._repository.current_forgetting_record(target)
        return bool(record and record.host_reads_withdrawn)

    def _view_for_search_hit(
        self,
        hit: IndexSearchHit,
        *,
        disclosure_context: DisclosureContext,
        snapshot_id: str,
    ) -> AssembledView | None:
        if hit.record.source_kind is IndexSourceKind.BELIEF_STATE:
            state = hit.source
            source_claim_ids = (
                *state.projection.active_claim_ids,
                *state.projection.historical_claim_ids,
            )
            if any(
                (
                    claim := self._repository.read_claim(claim_id)
                ) is not None
                and not self._claim_visible(claim, disclosure_context)
                for claim_id in source_claim_ids
            ):
                return None
            locus = state.projection.locus
            return self._view_for_locus(
                subject_id=locus.subject_id,
                locus_key=locus.locus_key,
                aggregation_mode=locus.aggregation_mode,
                disclosure_context=disclosure_context,
                target_snapshot_id=snapshot_id,
            )

        if hit.record.source_kind is IndexSourceKind.CLAIM:
            claim = hit.source
            if not self._claim_visible(claim, disclosure_context):
                return None
            belief, _ = self._visible_projection(
                subject_id=claim.subject_id,
                locus_key=claim.locus.locus_key,
                disclosure_context=disclosure_context,
            )
            if claim.claim_id in belief.projection.active_claim_ids:
                return self._view_for_locus(
                    subject_id=claim.subject_id,
                    locus_key=claim.locus.locus_key,
                    aggregation_mode=belief.projection.locus.aggregation_mode,
                    disclosure_context=disclosure_context,
                    target_snapshot_id=snapshot_id,
                )
            return self.build_timeline_view(
                subject_id=claim.subject_id,
                locus_key=claim.locus.locus_key,
                disclosure_context=disclosure_context,
                target_snapshot_id=snapshot_id,
            )

        if hit.record.source_kind is IndexSourceKind.OBSERVATION:
            try:
                return self.build_evidence_view(
                    target_kind="observation",
                    target_id=hit.record.source_id,
                    disclosure_context=disclosure_context,
                    target_snapshot_id=snapshot_id,
                )
            except LookupError:
                return None

        if hit.record.source_kind is IndexSourceKind.COMPILED_VIEW:
            stored_view = hit.source
            return AssembledView(
                compiled_view=stored_view.view,
                payload=stored_view.payload,
            )

        return None

    def _view_for_locus(
        self,
        *,
        subject_id: str,
        locus_key: str,
        aggregation_mode: AggregationMode,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str,
    ) -> AssembledView:
        if aggregation_mode is AggregationMode.SET_UNION:
            return self.build_set_view(
                subject_id=subject_id,
                locus_key=locus_key,
                disclosure_context=disclosure_context,
                target_snapshot_id=target_snapshot_id,
            )
        if aggregation_mode is AggregationMode.TIMELINE:
            return self.build_timeline_view(
                subject_id=subject_id,
                locus_key=locus_key,
                disclosure_context=disclosure_context,
                target_snapshot_id=target_snapshot_id,
            )
        return self.build_state_view(
            subject_id=subject_id,
            locus_key=locus_key,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def _state_like_views_for_subject(
        self,
        *,
        subject_id: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None,
    ) -> tuple[AssembledView, ...]:
        views: list[tuple[int, str, AssembledView]] = []
        for state in self._beliefs.list_states(subject_id=subject_id, policy_stamp=self._policy.policy_stamp):
            locus = state.projection.locus
            if locus.aggregation_mode is AggregationMode.TIMELINE:
                continue
            try:
                view = self._view_for_locus(
                    subject_id=subject_id,
                    locus_key=locus.locus_key,
                    aggregation_mode=locus.aggregation_mode,
                    disclosure_context=disclosure_context,
                    target_snapshot_id=self._resolve_snapshot_id(target_snapshot_id),
                )
            except LookupError:
                continue
            views.append(
                (
                    self._retrieval_rank_for_claim_ids(view.compiled_view.claim_ids),
                    locus.locus_key,
                    view,
                )
            )
        return tuple(
            view
            for _, _, view in sorted(views, key=lambda item: (item[0], item[1]))
        )

    def _timeline_views_for_subject(
        self,
        *,
        subject_id: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None,
    ) -> tuple[AssembledView, ...]:
        views: list[tuple[int, str, AssembledView]] = []
        for state in self._beliefs.list_states(subject_id=subject_id, policy_stamp=self._policy.policy_stamp):
            locus = state.projection.locus
            if locus.aggregation_mode is not AggregationMode.TIMELINE:
                continue
            try:
                view = self.build_timeline_view(
                    subject_id=subject_id,
                    locus_key=locus.locus_key,
                    disclosure_context=disclosure_context,
                    target_snapshot_id=target_snapshot_id,
                )
            except LookupError:
                continue
            views.append(
                (
                    self._retrieval_rank_for_claim_ids(view.compiled_view.claim_ids),
                    locus.locus_key,
                    view,
                )
            )
        return tuple(
            view
            for _, _, view in sorted(views, key=lambda item: (item[0], item[1]))
        )

    def _prompt_candidate_for_view(self, view: AssembledView) -> PromptFragmentCandidate:
        text = self._prompt_text_for_view(view)
        token_estimate = {
            ViewKind.STATE: 6,
            ViewKind.SET: 5,
            ViewKind.PROFILE: 5,
            ViewKind.EVIDENCE: 5,
            ViewKind.TIMELINE: 8,
        }.get(view.compiled_view.kind, _token_estimate(text))

        compressed_text = None
        compressed_token_estimate = None
        degradation_reason = None
        soft_budget_group = None
        if view.compiled_view.kind is ViewKind.EVIDENCE:
            compressed_text = "Evidence summary"
            compressed_token_estimate = 2
            degradation_reason = "compressed evidence"
            soft_budget_group = "evidence"
        elif view.compiled_view.kind is ViewKind.TIMELINE:
            compressed_text = "Collapsed timeline"
            compressed_token_estimate = 4
            degradation_reason = "collapsed timeline"

        utility_weight = max(
            (
                weight.weighted_score
                for claim_id in view.compiled_view.claim_ids
                for weight in (self._claim_weight(claim_id),)
                if weight is not None
            ),
            default=0,
        )

        return PromptFragmentCandidate(
            fragment_id=view.compiled_view.view_key,
            source_view=view.compiled_view.kind,
            text=text,
            token_estimate=token_estimate,
            priority_band=self._prompt_priority_band(view.compiled_view.kind, view.compiled_view.claim_ids),
            claim_ids=view.compiled_view.claim_ids,
            observation_ids=view.compiled_view.observation_ids,
            utility_weight=utility_weight,
            epistemic_status=view.compiled_view.epistemic_status,
            disclosure_action=PromptDisclosureAction.ALLOW,
            soft_budget_group=soft_budget_group,
            compressed_text=compressed_text,
            compressed_token_estimate=compressed_token_estimate,
            degradation_reason=degradation_reason,
        )

    def _prompt_text_for_view(self, view: AssembledView) -> str:
        if view.compiled_view.kind is ViewKind.STATE:
            return str(view.payload["summary"])
        if view.compiled_view.kind is ViewKind.SET:
            return f"Current set: {', '.join(view.payload['items'])}"
        if view.compiled_view.kind is ViewKind.PROFILE:
            return " ".join(entry["summary"] for entry in view.payload["entries"])
        if view.compiled_view.kind is ViewKind.EVIDENCE:
            return " ".join(observation["content"] for observation in view.payload["observations"])
        if view.compiled_view.kind is ViewKind.TIMELINE:
            return " ".join(entry["value"] for entry in view.payload["entries"])
        return json.dumps(view.payload, sort_keys=True)

    def _prompt_priority_band(self, kind: ViewKind, claim_ids: tuple[str, ...]) -> int:
        base = {
            ViewKind.STATE: 10,
            ViewKind.SET: 12,
            ViewKind.PROFILE: 20,
            ViewKind.EVIDENCE: 30,
            ViewKind.TIMELINE: 40,
        }[kind]
        return base + self._retrieval_rank_for_claim_ids(claim_ids)

    def _auxiliary_prompt_memory(
        self,
        *,
        session_id: str,
        subject_id: str,
    ) -> tuple[dict[str, str], ...]:
        entries: list[dict[str, str]] = []
        for candidate in self._repository.list_candidate_memories(subject_id=subject_id):
            trace = self._repository.read_admission_trace(candidate.candidate_id)
            if trace is None:
                continue
            if trace.decision.outcome is not AdmissionOutcome.PROMPT_ONLY:
                continue
            entries.append(
                {
                    "entry_kind": "prompt_only",
                    "entry_id": candidate.candidate_id,
                    "text": _stringify_value(candidate.value.get("note", candidate.value)),
                }
            )
        for buffer_record in self._repository.list_session_buffers(session_id=session_id):
            if buffer_record.buffer_kind != "session_ephemeral":
                continue
            text = buffer_record.payload.get("text")
            if not isinstance(text, str):
                continue
            entries.append(
                {
                    "entry_kind": "session_ephemeral",
                    "entry_id": buffer_record.buffer_key,
                    "text": text,
                }
            )
        return tuple(entries)

    def _follow_up_payloads(
        self,
        *,
        subject_id: str | None,
        surface: ResolutionSurface,
        limit: int | None = None,
    ) -> tuple[dict[str, str], ...]:
        items = self._repository.list_resolution_items(
            subject_id=subject_id,
            surface=surface,
            at_time=datetime.now(timezone.utc),
            actionable_only=True,
            limit=limit,
        )
        return tuple(
            {
                "item_id": item.item_id,
                "subject_id": item.subject_id,
                "locus_key": item.locus_key,
                "rationale": item.rationale,
            }
            for item in items
        )

    def _claim_weight(self, claim_id: str) -> CompiledUtilityWeight | None:
        return self._repository.read_compiled_utility_weight(
            target=OutcomeTarget.CLAIM,
            target_id=claim_id,
            policy_stamp=self._policy.policy_stamp,
        )

    def _retrieval_rank_for_claim_ids(self, claim_ids: tuple[str, ...]) -> int:
        ranks: list[int] = []
        for claim_id in claim_ids:
            claim = self._repository.read_claim(claim_id)
            if claim is None:
                continue
            ranks.append(
                self._policy.retrieval_rank_for(_claim_type_root(claim.claim_type))
            )
        return min(ranks) if ranks else 999

    @staticmethod
    def _search_priority(kind: ViewKind) -> int:
        return {
            ViewKind.STATE: 1,
            ViewKind.SET: 2,
            ViewKind.PROFILE: 3,
            ViewKind.TIMELINE: 4,
            ViewKind.EVIDENCE: 5,
            ViewKind.PROMPT: 6,
            ViewKind.ANSWER: 7,
        }[kind]

    def _claim_summary(self, claim: Claim) -> str:
        subject_name = self._subject(claim.subject_id).canonical_name
        value_text = _stringify_value(claim.value)
        root = _claim_type_root(claim.claim_type)
        if root == "preference":
            return f"{subject_name} prefers {value_text}"
        if root == "biography":
            return f"{subject_name} knows {value_text}"
        if root == "task_state":
            return f"{subject_name} status: {value_text}"
        return f"{subject_name} {claim.locus.locus_key}: {value_text}"

    def _view_payload(self, view: AssembledView) -> dict[str, object]:
        return {
            "view_kind": view.compiled_view.kind.value,
            "view_key": view.compiled_view.view_key,
            "snapshot_id": view.compiled_view.snapshot_id,
            "policy_stamp": view.compiled_view.policy_stamp,
            "epistemic_status": _transport(view.compiled_view.epistemic_status),
            "claim_ids": view.compiled_view.claim_ids,
            "observation_ids": view.compiled_view.observation_ids,
            "payload": _transport(view.payload),
        }

    def _search_payload(self, result: SearchResult) -> dict[str, object]:
        return {
            "view_kind": result.view.compiled_view.kind.value,
            "view_key": result.view.compiled_view.view_key,
            "score": round(result.score, 6),
            "source_kind": result.source_kind.value,
            "record_id": result.record_id,
            "excerpt": result.excerpt,
            "claim_ids": result.view.compiled_view.claim_ids,
            "observation_ids": result.view.compiled_view.observation_ids,
            "payload": _transport(result.view.payload),
        }

    def _resolve_snapshot_id(self, target_snapshot_id: str | None) -> str:
        if target_snapshot_id is not None:
            return _clean_text(target_snapshot_id, field_name="target_snapshot_id")
        row = self._connection.execute(
            """
            SELECT snapshot_id
            FROM snapshots
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise LookupError("no snapshots exist")
        return row["snapshot_id"]

    @staticmethod
    def _state_view_key(subject_id: str, locus_key: str) -> str:
        return f"state:{subject_id}:{locus_key}"

    @staticmethod
    def _set_view_key(subject_id: str, locus_key: str) -> str:
        return f"set:{subject_id}:{locus_key}"

    @staticmethod
    def _timeline_view_key(subject_id: str, locus_key: str) -> str:
        return f"timeline:{subject_id}:{locus_key}"

    @staticmethod
    def _profile_view_key(subject_id: str) -> str:
        return f"profile:{subject_id}"

    @staticmethod
    def _prompt_view_key(session_id: str) -> str:
        return f"prompt:{session_id}"

    @staticmethod
    def _evidence_view_key(target_kind: str, target_id: str) -> str:
        return f"evidence:{target_kind}:{target_id}"

    @staticmethod
    def _locus_target_id(locus: MemoryLocus) -> str:
        return f"locus:{locus.subject_id}:{locus.locus_key}"

    @staticmethod
    def _parse_subject_locus_view_key(view_key: str, *, expected_prefix: str) -> tuple[str, str]:
        parts = _clean_text(view_key, field_name="view_key").split(":")
        if len(parts) < 5 or parts[0] != expected_prefix or parts[1] != "subject":
            raise ValueError(f"invalid {expected_prefix} view_key: {view_key}")
        return ":".join(parts[1:4]), ":".join(parts[4:])

    @staticmethod
    def _parse_subject_view_key(view_key: str, *, expected_prefix: str) -> str:
        parts = _clean_text(view_key, field_name="view_key").split(":")
        if len(parts) < 4 or parts[0] != expected_prefix or parts[1] != "subject":
            raise ValueError(f"invalid {expected_prefix} view_key: {view_key}")
        return ":".join(parts[1:4])

    @staticmethod
    def _parse_prompt_view_key(view_key: str) -> str:
        cleaned = _clean_text(view_key, field_name="view_key")
        prefix = "prompt:"
        if not cleaned.startswith(prefix):
            raise ValueError(f"invalid prompt view_key: {view_key}")
        return cleaned[len(prefix):]
