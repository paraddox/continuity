"""Claim derivation pipeline for schema-hard reasoning writes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.compiler import (
    CompilerDependency,
    CompilerNode,
    CompilerNodeCategory,
    CompilerStateRepository,
    DependencyRole,
    SourceInputKind,
)
from continuity.forgetting import ForgettingSurface
from continuity.index.zvec_index import EmbeddingClientProtocol, VectorBackendProtocol, ZvecIndex
from continuity.ontology import EvidenceKind
from continuity.outcomes import OutcomeTarget
from continuity.policy import PolicyPack, get_policy_pack
from continuity.session_manager import SessionManager
from continuity.snapshots import (
    MemorySnapshot,
    SnapshotArtifactKind,
    SnapshotArtifactRef,
    SnapshotHead,
    SnapshotHeadState,
    SnapshotRepository,
)
from continuity.store.belief_revision import BeliefRevisionEngine, BeliefStateRepository, StoredBeliefState
from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimScope,
    MemoryLocus,
    Observation,
    Subject,
    SubjectKind,
)
from continuity.store.sqlite import SQLiteRepository
from continuity.tiers import TierAssignment, TierStateRepository, hermes_v1_tier_policy, initial_tier_for_claim_type
from continuity.transactions import TransactionKind
from continuity.utility import CompiledUtilityWeight
from continuity.views import CompiledView, ViewKind

from .base import (
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningAdapter,
    ReasoningMessage,
    StructuredOutputSchema,
    publish_authoritative_mutation,
    validate_structured_output,
)
from .codex_adapter import CodexAdapterConfig


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _json_dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _hash_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}:{digest.hexdigest()[:16]}"


def _stringify_value(value: object) -> str:
    if isinstance(value, Mapping):
        if len(value) == 1:
            return _stringify_value(next(iter(value.values())))
        return ", ".join(
            f"{key}={_stringify_value(nested)}"
            for key, nested in sorted(value.items())
        )
    if isinstance(value, tuple | list):
        return ", ".join(_stringify_value(item) for item in value)
    return str(value)


def fingerprint_candidate_content(
    *,
    claim_type: str,
    subject_id: str,
    locus_key: str,
    value: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _json_dumps(
            {
                "claim_type": _clean_text(claim_type, field_name="claim_type"),
                "subject_id": _clean_text(subject_id, field_name="subject_id"),
                "locus_key": _clean_text(locus_key, field_name="locus_key"),
                "value": dict(value),
            }
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ClaimDerivationCandidate:
    claim_type: str
    subject_ref: str
    scope: ClaimScope
    locus_key: str
    value: dict[str, object]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_type", _clean_text(self.claim_type, field_name="claim_type"))
        object.__setattr__(self, "subject_ref", _clean_text(self.subject_ref, field_name="subject_ref"))
        object.__setattr__(self, "locus_key", _clean_text(self.locus_key, field_name="locus_key"))
        object.__setattr__(self, "value", dict(self.value))
        if not self.value:
            raise ValueError("value must be a non-empty object")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(_clean_text(reference, field_name="evidence_refs") for reference in self.evidence_refs),
        )
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")


@dataclass(frozen=True, slots=True)
class ClaimDerivationEnvelope:
    candidates: tuple[ClaimDerivationCandidate, ...]


class ClaimDerivationEnvelopeSchema(StructuredOutputSchema[ClaimDerivationEnvelope]):
    name = "claim_derivation_envelope.v1"

    def validate(self, payload: object) -> ClaimDerivationEnvelope:
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates must be a list")

        candidates: list[ClaimDerivationCandidate] = []
        for index, raw_candidate in enumerate(raw_candidates):
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(f"candidate {index} must be an object")
            raw_scope = raw_candidate.get("scope")
            raw_value = raw_candidate.get("value")
            raw_evidence_refs = raw_candidate.get("evidence_refs")
            if not isinstance(raw_scope, str):
                raise ValueError(f"candidate {index} scope must be a string")
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"candidate {index} value must be an object")
            if not isinstance(raw_evidence_refs, list) or not all(
                isinstance(reference, str) for reference in raw_evidence_refs
            ):
                raise ValueError(f"candidate {index} evidence_refs must be a list of strings")

            candidates.append(
                ClaimDerivationCandidate(
                    claim_type=str(raw_candidate.get("claim_type", "")),
                    subject_ref=str(raw_candidate.get("subject_ref", "")),
                    scope=ClaimScope(raw_scope),
                    locus_key=str(raw_candidate.get("locus_key", "")),
                    value={str(key): value for key, value in raw_value.items()},
                    evidence_refs=tuple(raw_evidence_refs),
                )
            )

        return ClaimDerivationEnvelope(candidates=tuple(candidates))


class AdmissionStrategy(Protocol):
    def evaluate(
        self,
        *,
        candidate: ClaimDerivationCandidate,
        subject: Subject,
        evidence: tuple[Observation, ...],
        policy: PolicyPack,
        window_key: str,
        recorded_at: datetime,
        budget: AdmissionWriteBudget,
        blocked_by_tombstone: bool,
    ) -> AdmissionDecisionTrace: ...


class DefaultAdmissionStrategy:
    _DURABLE_THRESHOLDS = AdmissionThresholds(
        evidence=AdmissionStrength.MEDIUM,
        novelty=AdmissionStrength.MEDIUM,
        stability=AdmissionStrength.MEDIUM,
        salience=AdmissionStrength.LOW,
    )

    _NON_DURABLE_THRESHOLDS = AdmissionThresholds(
        evidence=AdmissionStrength.LOW,
        novelty=AdmissionStrength.LOW,
        stability=AdmissionStrength.LOW,
        salience=AdmissionStrength.LOW,
    )

    def evaluate(
        self,
        *,
        candidate: ClaimDerivationCandidate,
        subject: Subject,
        evidence: tuple[Observation, ...],
        policy: PolicyPack,
        window_key: str,
        recorded_at: datetime,
        budget: AdmissionWriteBudget,
        blocked_by_tombstone: bool,
    ) -> AdmissionDecisionTrace:
        spec = policy.memory_class_spec_for(candidate.claim_type)
        if blocked_by_tombstone:
            return AdmissionDecisionTrace(
                decision=AdmissionDecision(
                    candidate_id="candidate:pending",
                    outcome=AdmissionOutcome.DISCARD,
                    recorded_at=recorded_at,
                    rationale="candidate content is blocked by a derivation-pipeline tombstone",
                ),
                claim_type=candidate.claim_type,
                policy_stamp=policy.policy_stamp,
                assessment=AdmissionAssessment(
                    claim_type=candidate.claim_type,
                    evidence=AdmissionStrength.HIGH,
                    novelty=AdmissionStrength.HIGH,
                    stability=AdmissionStrength.HIGH,
                    salience=AdmissionStrength.HIGH,
                    rationale="blocked by forgetting tombstone",
                ),
                thresholds=self._NON_DURABLE_THRESHOLDS,
                budget=budget,
            )

        default_outcome = policy.default_admission_outcome_for(candidate.claim_type)
        if default_outcome is AdmissionOutcome.DURABLE_CLAIM and not budget.allows_durable_promotion():
            default_outcome = AdmissionOutcome.DISCARD

        assessment = AdmissionAssessment(
            claim_type=candidate.claim_type,
            evidence=AdmissionStrength.HIGH if evidence else AdmissionStrength.LOW,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH if len(evidence) == 1 else AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.MEDIUM,
            rationale=(
                f"derived {candidate.claim_type} for {subject.subject_id} from "
                f"{len(evidence)} observation(s)"
            ),
            utility_signals=("prompt_inclusion",),
        )
        thresholds = (
            self._DURABLE_THRESHOLDS
            if default_outcome is AdmissionOutcome.DURABLE_CLAIM
            else self._NON_DURABLE_THRESHOLDS
        )
        return AdmissionDecisionTrace(
            decision=AdmissionDecision(
                candidate_id="candidate:pending",
                outcome=default_outcome,
                recorded_at=recorded_at,
                rationale=assessment.rationale,
            ),
            claim_type=spec.claim_type,
            policy_stamp=policy.policy_stamp,
            assessment=assessment,
            thresholds=thresholds,
            budget=budget,
        )


@dataclass(frozen=True, slots=True)
class ResolvedCandidate:
    candidate: ClaimDerivationCandidate
    subject: Subject
    evidence: tuple[Observation, ...]
    evidence_kind: EvidenceKind
    content_fingerprint: str


@dataclass(frozen=True, slots=True)
class ClaimDerivationResult:
    derivation_run_id: str | None
    decision_traces: tuple[AdmissionDecisionTrace, ...]
    claim_ids: tuple[str, ...]
    buffered_candidate_ids: tuple[str, ...]
    compiled_view_ids: tuple[str, ...]
    active_snapshot_id: str | None
    staged_snapshot_id: str | None
    compiler_dependencies: tuple[CompilerDependency, ...]
    claim_utility_target: OutcomeTarget = OutcomeTarget.CLAIM


_VIEW_TO_SNAPSHOT_ARTIFACT = {
    ViewKind.STATE: SnapshotArtifactKind.STATE_VIEW,
    ViewKind.TIMELINE: SnapshotArtifactKind.TIMELINE_VIEW,
    ViewKind.SET: SnapshotArtifactKind.SET_VIEW,
    ViewKind.PROFILE: SnapshotArtifactKind.PROFILE_VIEW,
    ViewKind.PROMPT: SnapshotArtifactKind.PROMPT_VIEW,
    ViewKind.EVIDENCE: SnapshotArtifactKind.EVIDENCE_VIEW,
    ViewKind.ANSWER: SnapshotArtifactKind.ANSWER_VIEW,
}


class ClaimDerivationPipeline:
    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        adapter: ReasoningAdapter,
        embedding_client: EmbeddingClientProtocol,
        vector_backend: VectorBackendProtocol,
        session_manager: SessionManager | None = None,
        policy_name: str = "hermes_v1",
        adapter_config: CodexAdapterConfig | None = None,
        admission_strategy: AdmissionStrategy | None = None,
        snapshot_head_key: str = "current",
    ) -> None:
        self._connection = connection
        self._repository = SQLiteRepository(connection)
        self._beliefs = BeliefStateRepository(connection)
        self._belief_revision = BeliefRevisionEngine(connection)
        self._compiler = CompilerStateRepository(connection)
        self._snapshots = SnapshotRepository(connection)
        self._tiers = TierStateRepository(connection)
        self._adapter = adapter
        self._index = ZvecIndex(
            connection=connection,
            embedding_client=embedding_client,
            backend=vector_backend,
            policy_stamp=get_policy_pack(policy_name).policy_stamp,
        )
        self._session_manager = session_manager
        self._policy = get_policy_pack(policy_name)
        self._adapter_config = adapter_config or CodexAdapterConfig()
        self._admission_strategy = admission_strategy or DefaultAdmissionStrategy()
        self._snapshot_head_key = _clean_text(snapshot_head_key, field_name="snapshot_head_key")

    def derive_from_observations(
        self,
        *,
        observation_ids: Sequence[str],
        session_id: str | None,
        source_transaction_kind: TransactionKind,
        run_at: datetime,
    ) -> ClaimDerivationResult:
        observations = self._load_observations(observation_ids)
        request = ClaimDerivationRequest(
            observations=self._reasoning_messages_for(observations),
        )
        validated = validate_structured_output(
            self._adapter.derive_claims(request),
            ClaimDerivationEnvelopeSchema(),
        )
        return publish_authoritative_mutation(
            validated,
            lambda envelope: self._publish_validated_envelope(
                envelope=envelope,
                observations=observations,
                session_id=session_id,
                source_transaction_kind=source_transaction_kind,
                run_at=run_at,
            ),
        )

    def _publish_validated_envelope(
        self,
        *,
        envelope: ClaimDerivationEnvelope,
        observations: tuple[Observation, ...],
        session_id: str | None,
        source_transaction_kind: TransactionKind,
        run_at: datetime,
    ) -> ClaimDerivationResult:
        resolved_candidates = tuple(
            self._resolve_candidate(candidate, observations=observations)
            for candidate in envelope.candidates
        )
        active_snapshot_id = self._ensure_active_snapshot()
        derivation_run_id = None
        decision_traces: list[AdmissionDecisionTrace] = []
        durable_claims: list[Claim] = []
        buffered_candidate_ids: list[str] = []

        if resolved_candidates:
            derivation_run_id = self._save_derivation_run(
                observations=observations,
                source_transaction_kind=source_transaction_kind,
                run_at=run_at,
            )

        for index, resolved in enumerate(resolved_candidates):
            candidate_id = _hash_id(
                "candidate",
                derivation_run_id or "no-run",
                str(index),
                resolved.subject.subject_id,
                resolved.candidate.locus_key,
                _json_dumps(resolved.candidate.value),
            )
            candidate = CandidateMemory(
                candidate_id=candidate_id,
                claim_type=resolved.candidate.claim_type,
                subject_id=resolved.subject.subject_id,
                scope=resolved.candidate.scope,
                value=resolved.candidate.value,
                source_observation_ids=tuple(
                    observation.observation_id for observation in resolved.evidence
                ),
            )
            self._repository.save_candidate_memory(candidate, created_at=run_at)
            budget = self._current_budget_for(candidate, session_id=session_id, fallback_session_id=observations[0].session_id)
            trace = self._admission_strategy.evaluate(
                candidate=resolved.candidate,
                subject=resolved.subject,
                evidence=resolved.evidence,
                policy=self._policy,
                window_key=budget.window_key,
                recorded_at=run_at,
                budget=budget,
                blocked_by_tombstone=self._is_blocked_by_tombstone(resolved),
            )
            trace = AdmissionDecisionTrace(
                decision=AdmissionDecision(
                    candidate_id=candidate_id,
                    outcome=trace.decision.outcome,
                    recorded_at=trace.decision.recorded_at,
                    rationale=trace.decision.rationale,
                ),
                claim_type=trace.claim_type,
                policy_stamp=trace.policy_stamp,
                assessment=trace.assessment,
                thresholds=trace.thresholds,
                budget=budget,
            )
            self._repository.admissions.record_decision(trace)
            decision_traces.append(trace)

            if trace.publishes_claim:
                durable_claims.append(
                    self._claim_from_candidate(
                        candidate=candidate,
                        trace=trace,
                        resolved=resolved,
                        derivation_run_id=derivation_run_id,
                        learned_at=run_at,
                    )
                )
                continue

            if trace.retains_candidate_context and self._session_manager is not None:
                self._session_manager.record_non_durable_memory(
                    session_id=session_id or observations[0].session_id,
                    candidate=candidate,
                    trace=trace,
                    updated_at=run_at,
                )
                buffered_candidate_ids.append(candidate_id)

        for claim in durable_claims:
            self._repository.save_claim(claim)

        claim_ids = tuple(claim.claim_id for claim in durable_claims)
        compiled_view_ids: tuple[str, ...] = ()
        staged_snapshot_id: str | None = None
        compiler_dependencies: tuple[CompilerDependency, ...] = ()

        if durable_claims:
            affected_states = self._revise_beliefs(durable_claims=tuple(durable_claims), run_at=run_at)
            compiled_view_ids = self._stage_compiled_views(
                durable_claims=tuple(durable_claims),
                affected_states=affected_states,
                active_snapshot_id=active_snapshot_id,
                run_at=run_at,
            )
            staged_snapshot_id = self._snapshots.read_candidate_snapshot(head_key=self._snapshot_head_key).snapshot_id
            self._initialize_utility_and_tiers(
                durable_claims=tuple(durable_claims),
                compiled_view_ids=compiled_view_ids,
                run_at=run_at,
            )
            compiler_dependencies = self._refresh_index_and_compiler_state(
                durable_claims=tuple(durable_claims),
                compiled_view_ids=compiled_view_ids,
            )

        return ClaimDerivationResult(
            derivation_run_id=derivation_run_id,
            decision_traces=tuple(decision_traces),
            claim_ids=claim_ids,
            buffered_candidate_ids=tuple(buffered_candidate_ids),
            compiled_view_ids=compiled_view_ids,
            active_snapshot_id=active_snapshot_id,
            staged_snapshot_id=staged_snapshot_id,
            compiler_dependencies=compiler_dependencies,
        )

    def _load_observations(self, observation_ids: Sequence[str]) -> tuple[Observation, ...]:
        loaded: list[Observation] = []
        for observation_id in observation_ids:
            observation = self._repository.read_observation(
                _clean_text(observation_id, field_name="observation_ids")
            )
            if observation is None:
                raise ValueError(f"unknown observation: {observation_id}")
            loaded.append(observation)
        if not loaded:
            raise ValueError("observation_ids must be non-empty")
        return tuple(loaded)

    def _reasoning_messages_for(
        self,
        observations: Sequence[Observation],
    ) -> tuple[ReasoningMessage, ...]:
        messages: list[ReasoningMessage] = []
        for index, observation in enumerate(observations):
            subject = self._repository.read_subject(observation.author_subject_id)
            role = "assistant" if subject is not None and subject.kind is SubjectKind.ASSISTANT else "user"
            messages.append(
                ReasoningMessage(
                    role=role,
                    content=(
                        f"observation:{index} "
                        f"author={observation.author_subject_id} "
                        f"content={observation.content}"
                    ),
                )
            )
        return tuple(messages)

    def _resolve_candidate(
        self,
        candidate: ClaimDerivationCandidate,
        *,
        observations: Sequence[Observation],
    ) -> ResolvedCandidate:
        observation_map = {
            f"observation:{index}": observation
            for index, observation in enumerate(observations)
        }
        evidence = tuple(
            observation_map[self._clean_reference(reference)]
            for reference in candidate.evidence_refs
        )
        if len({observation.author_subject_id for observation in evidence}) > 1 and candidate.subject_ref.endswith(".author"):
            raise ValueError("author-scoped subject_ref requires one shared evidence author")

        subject = self._resolve_subject(
            subject_ref=candidate.subject_ref,
            observations=observation_map,
        )
        spec = self._policy.memory_class_spec_for(candidate.claim_type)
        if not spec.supports_subject_kind(subject.kind):
            raise ValueError(
                f"{candidate.claim_type} does not support subject kind {subject.kind.value}"
            )
        if not spec.supports_scope(candidate.scope):
            raise ValueError(
                f"{candidate.claim_type} does not support scope {candidate.scope.value}"
            )
        if not spec.supports_locus_key(candidate.locus_key):
            raise ValueError(
                f"{candidate.locus_key} is outside the allowed locus prefix for {candidate.claim_type}"
            )

        evidence_kind = self._evidence_kind_for(subject=subject, spec=spec, evidence=evidence)
        if not spec.supports_evidence(evidence_kind):
            raise ValueError(
                f"{candidate.claim_type} does not support evidence kind {evidence_kind.value}"
            )
        if self._repository.read_disclosure_policy(spec.default_disclosure_policy) is None:
            raise ValueError(
                f"missing disclosure policy required for {candidate.claim_type}: "
                f"{spec.default_disclosure_policy}"
            )

        return ResolvedCandidate(
            candidate=candidate,
            subject=subject,
            evidence=evidence,
            evidence_kind=evidence_kind,
            content_fingerprint=fingerprint_candidate_content(
                claim_type=candidate.claim_type,
                subject_id=subject.subject_id,
                locus_key=candidate.locus_key,
                value=candidate.value,
            ),
        )

    def _clean_reference(self, value: str) -> str:
        cleaned = _clean_text(value, field_name="reference")
        if cleaned.endswith(".author"):
            return cleaned[:-7]
        return cleaned

    def _resolve_subject(
        self,
        *,
        subject_ref: str,
        observations: Mapping[str, Observation],
    ) -> Subject:
        cleaned = _clean_text(subject_ref, field_name="subject_ref")
        if cleaned.endswith(".author"):
            observation_key = cleaned[:-7]
            observation = observations.get(observation_key)
            if observation is None:
                raise ValueError(f"unknown observation reference in subject_ref: {subject_ref}")
            subject = self._repository.read_subject(observation.author_subject_id)
        else:
            subject = self._repository.read_subject(cleaned)

        if subject is None:
            raise ValueError(f"unknown subject_ref: {subject_ref}")
        return subject

    def _evidence_kind_for(
        self,
        *,
        subject: Subject,
        spec: Any,
        evidence: Sequence[Observation],
    ) -> EvidenceKind:
        if all(observation.source_kind == "session_message" for observation in evidence):
            if (
                subject.kind is SubjectKind.USER
                and spec.supports_evidence(EvidenceKind.EXPLICIT_USER_STATEMENT)
            ):
                return EvidenceKind.EXPLICIT_USER_STATEMENT
            if (
                subject.kind is SubjectKind.ASSISTANT
                and spec.supports_evidence(EvidenceKind.EXPLICIT_ASSISTANT_STATEMENT)
            ):
                return EvidenceKind.EXPLICIT_ASSISTANT_STATEMENT
            if (
                subject.kind is SubjectKind.PEER
                and spec.supports_evidence(EvidenceKind.EXPLICIT_PEER_STATEMENT)
            ):
                return EvidenceKind.EXPLICIT_PEER_STATEMENT
            if spec.supports_evidence(EvidenceKind.SESSION_TRANSCRIPT):
                return EvidenceKind.SESSION_TRANSCRIPT
        if spec.supports_evidence(EvidenceKind.HOST_IMPORT):
            return EvidenceKind.HOST_IMPORT
        if spec.supports_evidence(EvidenceKind.TOOL_RESULT):
            return EvidenceKind.TOOL_RESULT
        return EvidenceKind.HOST_IMPORT

    def _current_budget_for(
        self,
        candidate: CandidateMemory,
        *,
        session_id: str | None,
        fallback_session_id: str,
    ) -> AdmissionWriteBudget:
        spec = self._policy.memory_class_spec_for(candidate.claim_type)
        window_key = f"session:{session_id or fallback_session_id}"
        stored_budget = self._repository.admissions.read_budget(
            partition=spec.partition,
            window_key=window_key,
        )
        if stored_budget is not None:
            return stored_budget
        return AdmissionWriteBudget(
            partition=spec.partition,
            window_key=window_key,
            limit=self._policy.write_budget_for_partition(spec.partition),
            used=0,
        )

    def _is_blocked_by_tombstone(self, resolved: ResolvedCandidate) -> bool:
        return any(
            tombstone.content_fingerprint == resolved.content_fingerprint
            for tombstone in self._repository.forgetting.list_tombstones(
                surface=ForgettingSurface.DERIVATION_PIPELINE
            )
        )

    def _save_derivation_run(
        self,
        *,
        observations: Sequence[Observation],
        source_transaction_kind: TransactionKind,
        run_at: datetime,
    ) -> str:
        derivation_run_id = _hash_id(
            "derivation",
            self._policy.policy_stamp,
            self._adapter_config.fingerprint,
            run_at.isoformat(),
            *[observation.observation_id for observation in observations],
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO derivation_runs(
                    derivation_run_id,
                    adapter_name,
                    adapter_version,
                    policy_stamp,
                    schema_name,
                    source_transaction_kind,
                    created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    derivation_run_id,
                    self._adapter_config.strategy_id,
                    self._adapter_config.fingerprint,
                    self._policy.policy_stamp,
                    ClaimDerivationEnvelopeSchema.name,
                    source_transaction_kind.value,
                    run_at.isoformat(),
                    _json_dumps(
                        {
                            "observation_ids": [observation.observation_id for observation in observations],
                        }
                    ),
                ),
            )
        return derivation_run_id

    def _claim_from_candidate(
        self,
        *,
        candidate: CandidateMemory,
        trace: AdmissionDecisionTrace,
        resolved: ResolvedCandidate,
        derivation_run_id: str | None,
        learned_at: datetime,
    ) -> Claim:
        spec = self._policy.memory_class_spec_for(candidate.claim_type)
        observed_at = min(observation.observed_at for observation in resolved.evidence)
        return Claim.from_candidate(
            claim_id=_hash_id(
                "claim",
                candidate.subject_id,
                candidate.claim_type,
                resolved.candidate.locus_key,
                _json_dumps(candidate.value),
                learned_at.isoformat(),
            ),
            candidate=candidate,
            admission=trace.decision,
            locus=MemoryLocus(
                subject_id=resolved.subject.subject_id,
                locus_key=resolved.candidate.locus_key,
                scope=resolved.candidate.scope,
                default_disclosure_policy=spec.default_disclosure_policy,
                conflict_set_key=resolved.candidate.locus_key.replace("/", "."),
                aggregation_mode=spec.default_aggregation_mode,
            ),
            provenance=ClaimProvenance(
                observation_ids=tuple(observation.observation_id for observation in resolved.evidence),
                derivation_run_id=derivation_run_id,
            ),
            disclosure_policy=spec.default_disclosure_policy,
            observed_at=observed_at,
            learned_at=learned_at,
            valid_from=observed_at,
        )

    def _revise_beliefs(
        self,
        *,
        durable_claims: tuple[Claim, ...],
        run_at: datetime,
    ) -> tuple[StoredBeliefState, ...]:
        affected_subject_ids = tuple(
            dict.fromkeys(claim.subject_id for claim in durable_claims)
        )
        states: list[StoredBeliefState] = []
        for subject_id in affected_subject_ids:
            states.extend(
                self._belief_revision.revise_subject(
                    subject_id=subject_id,
                    as_of=run_at,
                    policy_name=self._policy.policy_name,
                )
            )
        return tuple(states)

    def _stage_compiled_views(
        self,
        *,
        durable_claims: tuple[Claim, ...],
        affected_states: tuple[StoredBeliefState, ...],
        active_snapshot_id: str,
        run_at: datetime,
    ) -> tuple[str, ...]:
        affected_addresses = {
            (claim.subject_id, claim.locus.locus_key)
            for claim in durable_claims
        }
        candidate_snapshot_id = _hash_id(
            "snapshot:candidate",
            active_snapshot_id,
            run_at.isoformat(),
            *sorted(claim.claim_id for claim in durable_claims),
        )
        planned_views: list[tuple[str, CompiledView, dict[str, object]]] = []
        for state in affected_states:
            if (state.subject_id, state.locus_key) not in affected_addresses:
                continue
            claim_group = tuple(
                claim
                for claim in self._repository.list_claims(subject_id=state.subject_id, locus_key=state.locus_key)
            )
            if not claim_group:
                continue
            planned_views.extend(
                self._planned_views_for_state(
                    state=state,
                    claim_group=claim_group,
                    snapshot_id=candidate_snapshot_id,
                    run_at=run_at,
                )
            )

        active_snapshot = self._snapshots.read_active_snapshot(head_key=self._snapshot_head_key)
        artifact_refs = list(() if active_snapshot is None else active_snapshot.artifact_refs)
        artifact_refs.extend(
            SnapshotArtifactRef(
                artifact_kind=_VIEW_TO_SNAPSHOT_ARTIFACT[compiled_view.kind],
                artifact_id=compiled_view_id,
            )
            for compiled_view_id, compiled_view, _ in planned_views
        )
        artifact_refs.append(
            SnapshotArtifactRef(
                artifact_kind=SnapshotArtifactKind.VECTOR_INDEX,
                artifact_id=f"vector:index:{candidate_snapshot_id}",
            )
        )
        self._snapshots.save_snapshot(
            MemorySnapshot(
                snapshot_id=candidate_snapshot_id,
                policy_stamp=self._policy.policy_stamp,
                parent_snapshot_id=active_snapshot_id,
                created_by_transaction=TransactionKind.COMPILE_VIEWS,
                artifact_refs=tuple(dict.fromkeys(artifact_refs)),
            )
        )
        self._snapshots.upsert_head(
            SnapshotHead(
                head_key=self._snapshot_head_key,
                state=SnapshotHeadState.CANDIDATE,
                snapshot_id=candidate_snapshot_id,
                based_on_snapshot_id=active_snapshot_id,
            )
        )
        self._persist_planned_views(planned_views, run_at=run_at)
        return tuple(compiled_view_id for compiled_view_id, _, _ in planned_views)

    def _planned_views_for_state(
        self,
        *,
        state: StoredBeliefState,
        claim_group: tuple[Claim, ...],
        snapshot_id: str,
        run_at: datetime,
    ) -> tuple[tuple[str, CompiledView, dict[str, object]], ...]:
        compiled_views: list[tuple[str, CompiledView, dict[str, object]]] = []
        active_claims = tuple(
            claim
            for claim in claim_group
            if claim.claim_id in state.projection.active_claim_ids
        ) or claim_group[:1]
        observation_ids = tuple(
            dict.fromkeys(
                observation_id
                for claim in claim_group
                for observation_id in claim.provenance.observation_ids
            )
        )

        if state.projection.locus.aggregation_mode in {
            AggregationMode.LATEST_WINS,
            AggregationMode.STATE_MACHINE,
        }:
            compiled_views.append(
                self._compiled_view_record(
                    kind=ViewKind.STATE,
                    subject_id=state.subject_id,
                    locus_key=state.locus_key,
                    claim_ids=tuple(claim.claim_id for claim in claim_group),
                    observation_ids=(),
                    epistemic_status=state.projection.epistemic.status,
                    payload={
                        "subject_id": state.subject_id,
                        "locus_key": state.locus_key,
                        "active_values": tuple(_stringify_value(claim.value) for claim in active_claims),
                    },
                    snapshot_id=snapshot_id,
                    run_at=run_at,
                )
            )
        elif state.projection.locus.aggregation_mode is AggregationMode.SET_UNION:
            compiled_views.append(
                self._compiled_view_record(
                    kind=ViewKind.SET,
                    subject_id=state.subject_id,
                    locus_key=state.locus_key,
                    claim_ids=tuple(claim.claim_id for claim in claim_group),
                    observation_ids=(),
                    epistemic_status=state.projection.epistemic.status,
                    payload={
                        "subject_id": state.subject_id,
                        "locus_key": state.locus_key,
                        "items": tuple(_stringify_value(claim.value) for claim in active_claims),
                    },
                    snapshot_id=snapshot_id,
                    run_at=run_at,
                )
            )

        compiled_views.append(
            self._compiled_view_record(
                kind=ViewKind.TIMELINE,
                subject_id=state.subject_id,
                locus_key=state.locus_key,
                claim_ids=tuple(claim.claim_id for claim in claim_group),
                observation_ids=observation_ids,
                epistemic_status=state.projection.epistemic.status,
                payload={
                    "subject_id": state.subject_id,
                    "locus_key": state.locus_key,
                    "entries": tuple(
                        {
                            "claim_id": claim.claim_id,
                            "value": _stringify_value(claim.value),
                            "learned_at": claim.learned_at.isoformat(),
                        }
                        for claim in claim_group
                    ),
                },
                snapshot_id=snapshot_id,
                run_at=run_at,
            )
        )
        compiled_views.append(
            self._compiled_view_record(
                kind=ViewKind.EVIDENCE,
                subject_id=state.subject_id,
                locus_key=state.locus_key,
                claim_ids=tuple(claim.claim_id for claim in claim_group),
                observation_ids=observation_ids,
                epistemic_status=state.projection.epistemic.status,
                payload={
                    "subject_id": state.subject_id,
                    "locus_key": state.locus_key,
                    "observation_ids": observation_ids,
                },
                snapshot_id=snapshot_id,
                run_at=run_at,
            )
        )
        return tuple(compiled_views)

    def _persist_planned_views(
        self,
        planned_views: Sequence[tuple[str, CompiledView, dict[str, object]]],
        *,
        run_at: datetime,
    ) -> None:
        with self._connection:
            for compiled_view_id, compiled_view, payload in planned_views:
                self._connection.execute(
                    """
                    INSERT INTO compiled_views(
                        compiled_view_id,
                        kind,
                        view_key,
                        policy_stamp,
                        snapshot_id,
                        epistemic_status,
                        payload_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        compiled_view_id,
                        compiled_view.kind.value,
                        compiled_view.view_key,
                        compiled_view.policy_stamp,
                        compiled_view.snapshot_id,
                        compiled_view.epistemic_status.value,
                        _json_dumps(payload),
                        run_at.isoformat(),
                    ),
                )
                self._connection.executemany(
                    """
                    INSERT INTO compiled_view_claims(compiled_view_id, claim_id)
                    VALUES (?, ?)
                    """,
                    tuple((compiled_view_id, claim_id) for claim_id in compiled_view.claim_ids),
                )
                self._connection.executemany(
                    """
                    INSERT INTO compiled_view_observations(compiled_view_id, observation_id)
                    VALUES (?, ?)
                    """,
                    tuple((compiled_view_id, observation_id) for observation_id in compiled_view.observation_ids),
                )

    def _compiled_view_record(
        self,
        *,
        kind: ViewKind,
        subject_id: str,
        locus_key: str,
        claim_ids: tuple[str, ...],
        observation_ids: tuple[str, ...],
        epistemic_status: object,
        payload: dict[str, object],
        snapshot_id: str,
        run_at: datetime,
    ) -> tuple[str, CompiledView, dict[str, object]]:
        compiled_view_id = _hash_id(
            "compiled-view",
            kind.value,
            subject_id,
            locus_key,
            run_at.isoformat(),
        )
        view_key = f"{kind.value}:{subject_id}:{locus_key}"
        return (
            compiled_view_id,
            CompiledView(
                kind=kind,
                view_key=view_key,
                policy_stamp=self._policy.policy_stamp,
                snapshot_id=snapshot_id,
                claim_ids=claim_ids,
                observation_ids=observation_ids,
                epistemic_status=epistemic_status,
            ),
            payload,
        )

    def _compiled_view_kind(self, compiled_view_id: str) -> ViewKind:
        row = self._connection.execute(
            """
            SELECT kind
            FROM compiled_views
            WHERE compiled_view_id = ?
            """,
            (compiled_view_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown compiled view id: {compiled_view_id}")
        return ViewKind(row[0])

    def _initialize_utility_and_tiers(
        self,
        *,
        durable_claims: tuple[Claim, ...],
        compiled_view_ids: tuple[str, ...],
        run_at: datetime,
    ) -> None:
        for claim in durable_claims:
            self._repository.utility.write_compiled_weight(
                CompiledUtilityWeight(
                    target=OutcomeTarget.CLAIM,
                    target_id=claim.claim_id,
                    policy_stamp=self._policy.policy_stamp,
                    weighted_score=0,
                    signal_counts=(),
                    source_event_ids=(),
                )
            )
            self._tiers.upsert_assignment(
                TierAssignment(
                    target_kind="claim",
                    target_id=claim.claim_id,
                    policy_stamp=self._policy.policy_stamp,
                    tier=initial_tier_for_claim_type(
                        claim.claim_type,
                        admission_outcome=claim.admission.outcome,
                        policy=self._policy,
                    ),
                    rationale=f"initial tier for {claim.claim_type}",
                    assigned_at=run_at,
                )
            )

        tier_policy = hermes_v1_tier_policy()
        for compiled_view_id in compiled_view_ids:
            view_kind = self._compiled_view_kind(compiled_view_id)
            self._repository.utility.write_compiled_weight(
                CompiledUtilityWeight(
                    target=OutcomeTarget.COMPILED_VIEW,
                    target_id=compiled_view_id,
                    policy_stamp=self._policy.policy_stamp,
                    weighted_score=0,
                    signal_counts=(),
                    source_event_ids=(),
                )
            )
            self._tiers.upsert_assignment(
                TierAssignment(
                    target_kind="compiled_view",
                    target_id=compiled_view_id,
                    policy_stamp=self._policy.policy_stamp,
                    tier=tier_policy.view_tiers[view_kind][0],
                    rationale=f"initial tier for {view_kind.value}",
                    assigned_at=run_at,
                )
            )

    def _refresh_index_and_compiler_state(
        self,
        *,
        durable_claims: tuple[Claim, ...],
        compiled_view_ids: tuple[str, ...],
    ) -> tuple[CompilerDependency, ...]:
        rebuild = self._index.rebuild_from_sqlite()
        policy_node = CompilerNode(
            node_id=f"policy:{self._policy.policy_stamp}",
            category=CompilerNodeCategory.SOURCE_INPUT,
            kind=SourceInputKind.POLICY_PACK,
            fingerprint=self._policy.policy_stamp,
        )
        adapter_node = CompilerNode(
            node_id=f"adapter:{self._adapter_config.fingerprint}",
            category=CompilerNodeCategory.SOURCE_INPUT,
            kind=SourceInputKind.REASONING_ADAPTER,
            fingerprint=self._adapter_config.fingerprint,
        )

        dependencies: dict[tuple[str, str, DependencyRole], CompilerDependency] = {
            (
                dependency.upstream_node_id,
                dependency.downstream_node_id,
                dependency.role,
            ): dependency
            for dependency in rebuild.compiler_dependencies
        }
        claim_ids = {claim.claim_id for claim in durable_claims}
        for claim in durable_claims:
            claim_node_id = f"claim:{claim.claim_id}"
            dependencies[(policy_node.node_id, claim_node_id, DependencyRole.POLICY)] = CompilerDependency(
                upstream_node_id=policy_node.node_id,
                downstream_node_id=claim_node_id,
                role=DependencyRole.POLICY,
            )
            dependencies[(adapter_node.node_id, claim_node_id, DependencyRole.POLICY)] = CompilerDependency(
                upstream_node_id=adapter_node.node_id,
                downstream_node_id=claim_node_id,
                role=DependencyRole.POLICY,
            )
            for observation_id in claim.provenance.observation_ids:
                dependencies[
                    (f"observation:{observation_id}", claim_node_id, DependencyRole.PROVENANCE)
                ] = CompilerDependency(
                    upstream_node_id=f"observation:{observation_id}",
                    downstream_node_id=claim_node_id,
                    role=DependencyRole.PROVENANCE,
                )
        for compiled_view_id in compiled_view_ids:
            row = self._connection.execute(
                """
                SELECT claim_id
                FROM compiled_view_claims
                WHERE compiled_view_id = ?
                """,
                (compiled_view_id,),
            ).fetchall()
            for claim_row in row:
                if claim_row[0] not in claim_ids:
                    continue
                dependencies[
                    (f"claim:{claim_row[0]}", f"view:{compiled_view_id}", DependencyRole.PROJECTION)
                ] = CompilerDependency(
                    upstream_node_id=f"claim:{claim_row[0]}",
                    downstream_node_id=f"view:{compiled_view_id}",
                    role=DependencyRole.PROJECTION,
                )

        self._compiler.upsert_nodes((*rebuild.compiler_nodes, policy_node, adapter_node))
        self._compiler.replace_dependencies(dependencies.values())
        return tuple(
            sorted(
                dependencies.values(),
                key=lambda dependency: (
                    dependency.upstream_node_id,
                    dependency.downstream_node_id,
                    dependency.role.value,
                ),
            )
        )

    def _ensure_active_snapshot(self) -> str:
        active_snapshot = self._snapshots.read_active_snapshot(head_key=self._snapshot_head_key)
        if active_snapshot is not None:
            return active_snapshot.snapshot_id

        bootstrap_snapshot = MemorySnapshot(
            snapshot_id="snapshot:bootstrap",
            policy_stamp=self._policy.policy_stamp,
            parent_snapshot_id=None,
            created_by_transaction=TransactionKind.PUBLISH_SNAPSHOT,
            artifact_refs=(
                SnapshotArtifactRef(
                    artifact_kind=SnapshotArtifactKind.VECTOR_INDEX,
                    artifact_id="vector:index:bootstrap",
                ),
            ),
        )
        self._snapshots.save_snapshot(bootstrap_snapshot)
        self._snapshots.upsert_head(
            SnapshotHead(
                head_key=self._snapshot_head_key,
                state=SnapshotHeadState.ACTIVE,
                snapshot_id=bootstrap_snapshot.snapshot_id,
            )
        )
        return bootstrap_snapshot.snapshot_id
