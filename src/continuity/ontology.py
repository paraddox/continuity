"""Typed memory ontology invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from continuity.store.claims import AdmissionOutcome, AggregationMode, ClaimScope, SubjectKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class MemoryClass(StrEnum):
    PREFERENCE = "preference"
    BIOGRAPHY = "biography"
    RELATIONSHIP = "relationship"
    TASK_STATE = "task_state"
    PROJECT_FACT = "project_fact"
    INSTRUCTION = "instruction"
    COMMITMENT = "commitment"
    OPEN_QUESTION = "open_question"
    EPHEMERAL_CONTEXT = "ephemeral_context"
    ASSISTANT_SELF_MODEL = "assistant_self_model"


class MemoryPartition(StrEnum):
    USER_MEMORY = "user_memory"
    ASSISTANT_MEMORY = "assistant_memory"
    SHARED_CONTEXT = "shared_context"
    EPHEMERAL_STATE = "ephemeral_state"


class EvidenceKind(StrEnum):
    EXPLICIT_USER_STATEMENT = "explicit_user_statement"
    EXPLICIT_ASSISTANT_STATEMENT = "explicit_assistant_statement"
    EXPLICIT_PEER_STATEMENT = "explicit_peer_statement"
    OBSERVED_BEHAVIOR = "observed_behavior"
    SESSION_TRANSCRIPT = "session_transcript"
    HOST_IMPORT = "host_import"
    TOOL_RESULT = "tool_result"


class DecayMode(StrEnum):
    STABLE_UNTIL_SUPERSEDED = "stable_until_superseded"
    REQUIRES_REFRESH = "requires_refresh"
    PROGRESSIVE_STATE = "progressive_state"
    OPEN_UNTIL_RESOLVED = "open_until_resolved"
    SESSION_ONLY = "session_only"
    SELF_REVIEW = "self_review"


class PromptRenderStyle(StrEnum):
    PROFILE_FACT = "profile_fact"
    RELATIONSHIP_NOTE = "relationship_note"
    TASK_STATUS = "task_status"
    REFERENCE_FACT = "reference_fact"
    INSTRUCTION_BLOCK = "instruction_block"
    COMMITMENT_TRACKER = "commitment_tracker"
    QUESTION_QUEUE = "question_queue"
    SESSION_NOTE = "session_note"
    SELF_CHECK = "self_check"


@dataclass(frozen=True, slots=True)
class MemoryClassSpec:
    memory_class: MemoryClass
    partition: MemoryPartition
    allowed_subject_kinds: frozenset[SubjectKind]
    allowed_scopes: frozenset[ClaimScope]
    allowed_evidence: frozenset[EvidenceKind]
    locus_prefix: str
    default_admission_outcome: AdmissionOutcome
    supports_durable_promotion: bool
    default_aggregation_mode: AggregationMode
    decay_mode: DecayMode
    prompt_render_style: PromptRenderStyle
    retrieval_priority: int
    default_disclosure_policy: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "locus_prefix", _clean_text(self.locus_prefix, field_name="locus_prefix"))
        object.__setattr__(
            self,
            "default_disclosure_policy",
            _clean_text(
                self.default_disclosure_policy,
                field_name="default_disclosure_policy",
            ),
        )
        if not self.locus_prefix.endswith("/"):
            raise ValueError("locus_prefix must end with '/'")
        if not self.allowed_subject_kinds:
            raise ValueError("allowed_subject_kinds must be non-empty")
        if not self.allowed_scopes:
            raise ValueError("allowed_scopes must be non-empty")
        if not self.allowed_evidence:
            raise ValueError("allowed_evidence must be non-empty")
        if self.retrieval_priority < 0:
            raise ValueError("retrieval_priority must be non-negative")
        if (
            self.default_admission_outcome is AdmissionOutcome.DURABLE_CLAIM
            and not self.supports_durable_promotion
        ):
            raise ValueError("durable defaults must support durable promotion")

    @property
    def claim_type(self) -> str:
        return self.memory_class.value

    def supports_subject_kind(self, subject_kind: SubjectKind) -> bool:
        return subject_kind in self.allowed_subject_kinds

    def supports_scope(self, scope: ClaimScope) -> bool:
        return scope in self.allowed_scopes

    def supports_evidence(self, evidence: EvidenceKind) -> bool:
        return evidence in self.allowed_evidence

    def supports_locus_key(self, locus_key: str) -> bool:
        return locus_key.startswith(self.locus_prefix)


@dataclass(frozen=True, slots=True)
class MemoryOntology:
    ontology_id: str
    version: str
    memory_classes: tuple[MemoryClassSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ontology_id", _clean_text(self.ontology_id, field_name="ontology_id"))
        object.__setattr__(self, "version", _clean_text(self.version, field_name="version"))

        if not self.memory_classes:
            raise ValueError("memory_classes must be non-empty")

        seen_types: set[MemoryClass] = set()
        for spec in self.memory_classes:
            if spec.memory_class in seen_types:
                raise ValueError("memory_classes must remain unique")
            seen_types.add(spec.memory_class)

    def types(self) -> tuple[MemoryClassSpec, ...]:
        return self.memory_classes

    def class_for_claim_type(self, claim_type: str) -> MemoryClassSpec:
        cleaned = _clean_text(claim_type, field_name="claim_type")
        for spec in self.memory_classes:
            if spec.claim_type == cleaned:
                return spec
        raise KeyError(f"unknown claim type: {cleaned}")


@lru_cache(maxsize=1)
def hermes_v1_ontology() -> MemoryOntology:
    return MemoryOntology(
        ontology_id="hermes_core_v1",
        version="1.0.0",
        memory_classes=(
            MemoryClassSpec(
                memory_class=MemoryClass.PREFERENCE,
                partition=MemoryPartition.USER_MEMORY,
                allowed_subject_kinds=frozenset({SubjectKind.USER, SubjectKind.PEER}),
                allowed_scopes=frozenset({ClaimScope.USER, ClaimScope.PEER}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.EXPLICIT_PEER_STATEMENT,
                        EvidenceKind.OBSERVED_BEHAVIOR,
                        EvidenceKind.HOST_IMPORT,
                    }
                ),
                locus_prefix="preference/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.LATEST_WINS,
                decay_mode=DecayMode.STABLE_UNTIL_SUPERSEDED,
                prompt_render_style=PromptRenderStyle.PROFILE_FACT,
                retrieval_priority=20,
                default_disclosure_policy="current_peer",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.BIOGRAPHY,
                partition=MemoryPartition.USER_MEMORY,
                allowed_subject_kinds=frozenset({SubjectKind.USER, SubjectKind.PEER}),
                allowed_scopes=frozenset({ClaimScope.USER, ClaimScope.PEER}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.EXPLICIT_PEER_STATEMENT,
                        EvidenceKind.HOST_IMPORT,
                    }
                ),
                locus_prefix="biography/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.SET_UNION,
                decay_mode=DecayMode.STABLE_UNTIL_SUPERSEDED,
                prompt_render_style=PromptRenderStyle.PROFILE_FACT,
                retrieval_priority=30,
                default_disclosure_policy="current_peer",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.RELATIONSHIP,
                partition=MemoryPartition.USER_MEMORY,
                allowed_subject_kinds=frozenset({SubjectKind.USER, SubjectKind.PEER}),
                allowed_scopes=frozenset({ClaimScope.USER, ClaimScope.PEER, ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.EXPLICIT_PEER_STATEMENT,
                        EvidenceKind.OBSERVED_BEHAVIOR,
                    }
                ),
                locus_prefix="relationship/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.LATEST_WINS,
                decay_mode=DecayMode.REQUIRES_REFRESH,
                prompt_render_style=PromptRenderStyle.RELATIONSHIP_NOTE,
                retrieval_priority=35,
                default_disclosure_policy="current_peer",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.TASK_STATE,
                partition=MemoryPartition.SHARED_CONTEXT,
                allowed_subject_kinds=frozenset(
                    {
                        SubjectKind.USER,
                        SubjectKind.PEER,
                        SubjectKind.PROJECT,
                        SubjectKind.REPO,
                        SubjectKind.FILE,
                    }
                ),
                allowed_scopes=frozenset({ClaimScope.SESSION, ClaimScope.PEER, ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.SESSION_TRANSCRIPT,
                        EvidenceKind.TOOL_RESULT,
                        EvidenceKind.HOST_IMPORT,
                    }
                ),
                locus_prefix="task/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.STATE_MACHINE,
                decay_mode=DecayMode.PROGRESSIVE_STATE,
                prompt_render_style=PromptRenderStyle.TASK_STATUS,
                retrieval_priority=40,
                default_disclosure_policy="shared_session",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.PROJECT_FACT,
                partition=MemoryPartition.SHARED_CONTEXT,
                allowed_subject_kinds=frozenset(
                    {SubjectKind.PROJECT, SubjectKind.REPO, SubjectKind.FILE}
                ),
                allowed_scopes=frozenset({ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.SESSION_TRANSCRIPT,
                        EvidenceKind.TOOL_RESULT,
                        EvidenceKind.HOST_IMPORT,
                    }
                ),
                locus_prefix="project/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.SET_UNION,
                decay_mode=DecayMode.STABLE_UNTIL_SUPERSEDED,
                prompt_render_style=PromptRenderStyle.REFERENCE_FACT,
                retrieval_priority=50,
                default_disclosure_policy="shared_session",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.INSTRUCTION,
                partition=MemoryPartition.SHARED_CONTEXT,
                allowed_subject_kinds=frozenset(
                    {SubjectKind.USER, SubjectKind.PROJECT, SubjectKind.REPO}
                ),
                allowed_scopes=frozenset({ClaimScope.USER, ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.SESSION_TRANSCRIPT,
                        EvidenceKind.HOST_IMPORT,
                    }
                ),
                locus_prefix="instruction/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.SET_UNION,
                decay_mode=DecayMode.STABLE_UNTIL_SUPERSEDED,
                prompt_render_style=PromptRenderStyle.INSTRUCTION_BLOCK,
                retrieval_priority=5,
                default_disclosure_policy="host_internal",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.COMMITMENT,
                partition=MemoryPartition.SHARED_CONTEXT,
                allowed_subject_kinds=frozenset(
                    {SubjectKind.USER, SubjectKind.PEER, SubjectKind.PROJECT}
                ),
                allowed_scopes=frozenset({ClaimScope.SESSION, ClaimScope.PEER, ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.EXPLICIT_PEER_STATEMENT,
                        EvidenceKind.SESSION_TRANSCRIPT,
                    }
                ),
                locus_prefix="commitment/",
                default_admission_outcome=AdmissionOutcome.DURABLE_CLAIM,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.TIMELINE,
                decay_mode=DecayMode.PROGRESSIVE_STATE,
                prompt_render_style=PromptRenderStyle.COMMITMENT_TRACKER,
                retrieval_priority=15,
                default_disclosure_policy="shared_session",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.OPEN_QUESTION,
                partition=MemoryPartition.SHARED_CONTEXT,
                allowed_subject_kinds=frozenset(
                    {SubjectKind.USER, SubjectKind.PEER, SubjectKind.PROJECT, SubjectKind.REPO}
                ),
                allowed_scopes=frozenset({ClaimScope.SESSION, ClaimScope.PEER, ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_USER_STATEMENT,
                        EvidenceKind.SESSION_TRANSCRIPT,
                        EvidenceKind.TOOL_RESULT,
                    }
                ),
                locus_prefix="question/",
                default_admission_outcome=AdmissionOutcome.NEEDS_CONFIRMATION,
                supports_durable_promotion=True,
                default_aggregation_mode=AggregationMode.TIMELINE,
                decay_mode=DecayMode.OPEN_UNTIL_RESOLVED,
                prompt_render_style=PromptRenderStyle.QUESTION_QUEUE,
                retrieval_priority=25,
                default_disclosure_policy="shared_session",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.EPHEMERAL_CONTEXT,
                partition=MemoryPartition.EPHEMERAL_STATE,
                allowed_subject_kinds=frozenset(set(SubjectKind)),
                allowed_scopes=frozenset({ClaimScope.SESSION, ClaimScope.SHARED}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.SESSION_TRANSCRIPT,
                        EvidenceKind.TOOL_RESULT,
                    }
                ),
                locus_prefix="session/",
                default_admission_outcome=AdmissionOutcome.PROMPT_ONLY,
                supports_durable_promotion=False,
                default_aggregation_mode=AggregationMode.TIMELINE,
                decay_mode=DecayMode.SESSION_ONLY,
                prompt_render_style=PromptRenderStyle.SESSION_NOTE,
                retrieval_priority=90,
                default_disclosure_policy="shared_session",
            ),
            MemoryClassSpec(
                memory_class=MemoryClass.ASSISTANT_SELF_MODEL,
                partition=MemoryPartition.ASSISTANT_MEMORY,
                allowed_subject_kinds=frozenset({SubjectKind.ASSISTANT}),
                allowed_scopes=frozenset({ClaimScope.ASSISTANT}),
                allowed_evidence=frozenset(
                    {
                        EvidenceKind.EXPLICIT_ASSISTANT_STATEMENT,
                        EvidenceKind.SESSION_TRANSCRIPT,
                        EvidenceKind.HOST_IMPORT,
                    }
                ),
                locus_prefix="self/",
                default_admission_outcome=AdmissionOutcome.SESSION_EPHEMERAL,
                supports_durable_promotion=False,
                default_aggregation_mode=AggregationMode.LATEST_WINS,
                decay_mode=DecayMode.SELF_REVIEW,
                prompt_render_style=PromptRenderStyle.SELF_CHECK,
                retrieval_priority=95,
                default_disclosure_policy="assistant_internal",
            ),
        ),
    )
