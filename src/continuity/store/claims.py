"""Canonical subject, observation, claim, and locus invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _normalize_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _validate_identifier_tuple(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must be non-empty")

    cleaned: list[str] = []
    for value in values:
        cleaned.append(_clean_text(value, field_name=field_name))

    return tuple(cleaned)


class SubjectKind(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    PEER = "peer"
    PROJECT = "project"
    REPO = "repo"
    FILE = "file"


class ClaimScope(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    PEER = "peer"
    SESSION = "session"
    SHARED = "shared"


class AdmissionOutcome(StrEnum):
    DISCARD = "discard"
    SESSION_EPHEMERAL = "session_ephemeral"
    PROMPT_ONLY = "prompt_only"
    NEEDS_CONFIRMATION = "needs_confirmation"
    DURABLE_CLAIM = "durable_claim"


class AggregationMode(StrEnum):
    LATEST_WINS = "latest_wins"
    SET_UNION = "set_union"
    TIMELINE = "timeline"
    STATE_MACHINE = "state_machine"


class ClaimRelationKind(StrEnum):
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    CORRECTS = "corrects"


@dataclass(frozen=True, slots=True)
class SubjectAlias:
    alias: str
    alias_type: str
    source_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", _clean_text(self.alias, field_name="alias"))
        object.__setattr__(self, "alias_type", _clean_text(self.alias_type, field_name="alias_type"))
        object.__setattr__(
            self,
            "source_observation_ids",
            _validate_identifier_tuple(
                self.source_observation_ids,
                field_name="source_observation_ids",
            ),
        )

    @property
    def normalized_alias(self) -> str:
        return _normalize_name(self.alias)


@dataclass(frozen=True, slots=True)
class SubjectMergeRecord:
    survivor_subject_id: str
    merged_subject_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "survivor_subject_id",
            _clean_text(self.survivor_subject_id, field_name="survivor_subject_id"),
        )
        object.__setattr__(
            self,
            "merged_subject_ids",
            _validate_identifier_tuple(self.merged_subject_ids, field_name="merged_subject_ids"),
        )
        object.__setattr__(
            self,
            "source_observation_ids",
            _validate_identifier_tuple(
                self.source_observation_ids,
                field_name="source_observation_ids",
            ),
        )
        if self.survivor_subject_id in self.merged_subject_ids:
            raise ValueError("merged_subject_ids must exclude the surviving subject")


@dataclass(frozen=True, slots=True)
class SubjectSplitRecord:
    source_subject_id: str
    child_subject_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_subject_id",
            _clean_text(self.source_subject_id, field_name="source_subject_id"),
        )
        object.__setattr__(
            self,
            "child_subject_ids",
            _validate_identifier_tuple(self.child_subject_ids, field_name="child_subject_ids"),
        )
        object.__setattr__(
            self,
            "source_observation_ids",
            _validate_identifier_tuple(
                self.source_observation_ids,
                field_name="source_observation_ids",
            ),
        )
        if self.source_subject_id in self.child_subject_ids:
            raise ValueError("child_subject_ids must exclude the source subject")


@dataclass(frozen=True, slots=True)
class Subject:
    subject_id: str
    kind: SubjectKind
    canonical_name: str
    aliases: tuple[SubjectAlias, ...] = ()
    merges: tuple[SubjectMergeRecord, ...] = ()
    splits: tuple[SubjectSplitRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _clean_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(
            self,
            "canonical_name",
            _clean_text(self.canonical_name, field_name="canonical_name"),
        )

        seen_aliases: set[str] = set()
        for alias in self.aliases:
            if alias.normalized_alias in seen_aliases:
                raise ValueError("subject aliases must remain unique after normalization")
            seen_aliases.add(alias.normalized_alias)

        for merge in self.merges:
            if merge.survivor_subject_id != self.subject_id:
                raise ValueError("subject merge history must target the canonical subject")

        for split in self.splits:
            if split.source_subject_id != self.subject_id:
                raise ValueError("subject split history must originate from the canonical subject")

    @property
    def normalized_names(self) -> frozenset[str]:
        names = {_normalize_name(self.canonical_name)}
        names.update(alias.normalized_alias for alias in self.aliases)
        return frozenset(names)

    def matches_name(self, candidate: str) -> bool:
        return _normalize_name(candidate) in self.normalized_names


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    source_kind: str
    session_id: str
    author_subject_id: str
    content: str
    observed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _clean_text(self.observation_id, field_name="observation_id"),
        )
        object.__setattr__(self, "source_kind", _clean_text(self.source_kind, field_name="source_kind"))
        object.__setattr__(self, "session_id", _clean_text(self.session_id, field_name="session_id"))
        object.__setattr__(
            self,
            "author_subject_id",
            _clean_text(self.author_subject_id, field_name="author_subject_id"),
        )
        object.__setattr__(self, "content", _clean_text(self.content, field_name="content"))
        object.__setattr__(
            self,
            "observed_at",
            _validate_timestamp(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    candidate_id: str
    claim_type: str
    subject_id: str
    scope: ClaimScope
    value: Any
    source_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _clean_text(self.candidate_id, field_name="candidate_id"),
        )
        object.__setattr__(self, "claim_type", _clean_text(self.claim_type, field_name="claim_type"))
        object.__setattr__(self, "subject_id", _clean_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(
            self,
            "source_observation_ids",
            _validate_identifier_tuple(
                self.source_observation_ids,
                field_name="source_observation_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    candidate_id: str
    outcome: AdmissionOutcome
    recorded_at: datetime
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _clean_text(self.candidate_id, field_name="candidate_id"),
        )
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )


@dataclass(frozen=True, slots=True)
class ClaimProvenance:
    observation_ids: tuple[str, ...]
    derivation_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_ids",
            _validate_identifier_tuple(self.observation_ids, field_name="observation_ids"),
        )
        if self.derivation_run_id is not None:
            object.__setattr__(
                self,
                "derivation_run_id",
                _clean_text(self.derivation_run_id, field_name="derivation_run_id"),
            )


@dataclass(frozen=True, slots=True)
class MemoryLocus:
    subject_id: str
    locus_key: str
    scope: ClaimScope
    default_disclosure_policy: str
    conflict_set_key: str
    aggregation_mode: AggregationMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _clean_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_text(self.locus_key, field_name="locus_key"))
        object.__setattr__(
            self,
            "default_disclosure_policy",
            _clean_text(
                self.default_disclosure_policy,
                field_name="default_disclosure_policy",
            ),
        )
        object.__setattr__(
            self,
            "conflict_set_key",
            _clean_text(self.conflict_set_key, field_name="conflict_set_key"),
        )

    @property
    def address(self) -> tuple[str, str]:
        return (self.subject_id, self.locus_key)


@dataclass(frozen=True, slots=True)
class ClaimRelation:
    kind: ClaimRelationKind
    related_claim_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "related_claim_id",
            _clean_text(self.related_claim_id, field_name="related_claim_id"),
        )


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    claim_type: str
    subject_id: str
    locus: MemoryLocus
    scope: ClaimScope
    disclosure_policy: str
    value: Any
    provenance: ClaimProvenance
    admission: AdmissionDecision
    observed_at: datetime
    learned_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    relations: tuple[ClaimRelation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _clean_text(self.claim_id, field_name="claim_id"))
        object.__setattr__(self, "claim_type", _clean_text(self.claim_type, field_name="claim_type"))
        object.__setattr__(self, "subject_id", _clean_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(
            self,
            "disclosure_policy",
            _clean_text(self.disclosure_policy, field_name="disclosure_policy"),
        )
        object.__setattr__(
            self,
            "observed_at",
            _validate_timestamp(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(
            self,
            "learned_at",
            _validate_timestamp(self.learned_at, field_name="learned_at"),
        )

        if self.valid_from is not None:
            object.__setattr__(
                self,
                "valid_from",
                _validate_timestamp(self.valid_from, field_name="valid_from"),
            )
        if self.valid_to is not None:
            object.__setattr__(
                self,
                "valid_to",
                _validate_timestamp(self.valid_to, field_name="valid_to"),
            )

        if self.subject_id != self.locus.subject_id:
            raise ValueError("claim subject_id must match locus subject_id")
        if self.admission.outcome is not AdmissionOutcome.DURABLE_CLAIM:
            raise ValueError("claims require explicit durable admission")
        if self.learned_at < self.observed_at:
            raise ValueError("learned_at cannot precede observed_at")
        if self.valid_from is not None and self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot precede valid_from")

    @classmethod
    def from_candidate(
        cls,
        *,
        claim_id: str,
        candidate: CandidateMemory,
        admission: AdmissionDecision,
        locus: MemoryLocus,
        provenance: ClaimProvenance,
        disclosure_policy: str,
        observed_at: datetime,
        learned_at: datetime,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        relations: tuple[ClaimRelation, ...] = (),
    ) -> Claim:
        if admission.candidate_id != candidate.candidate_id:
            raise ValueError("admission must target the candidate being promoted")
        if admission.outcome is not AdmissionOutcome.DURABLE_CLAIM:
            raise ValueError("only durable admissions may publish claims")
        if candidate.subject_id != locus.subject_id:
            raise ValueError("candidate subject_id must match locus subject_id")
        if not set(provenance.observation_ids).issubset(set(candidate.source_observation_ids)):
            raise ValueError("claim provenance must resolve to the candidate source observations")

        return cls(
            claim_id=claim_id,
            claim_type=candidate.claim_type,
            subject_id=candidate.subject_id,
            locus=locus,
            scope=candidate.scope,
            disclosure_policy=disclosure_policy,
            value=candidate.value,
            provenance=provenance,
            admission=admission,
            observed_at=observed_at,
            learned_at=learned_at,
            valid_from=valid_from,
            valid_to=valid_to,
            relations=relations,
        )


@dataclass(frozen=True, slots=True)
class HostMemoryArtifact:
    artifact_id: str
    artifact_kind: str
    claim_ids: tuple[str, ...]
    observation_ids: tuple[str, ...] = ()
    durable_root: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _clean_text(self.artifact_id, field_name="artifact_id"),
        )
        object.__setattr__(
            self,
            "artifact_kind",
            _clean_text(self.artifact_kind, field_name="artifact_kind"),
        )
        object.__setattr__(
            self,
            "claim_ids",
            _validate_identifier_tuple(self.claim_ids, field_name="claim_ids"),
        )
        if self.observation_ids:
            object.__setattr__(
                self,
                "observation_ids",
                _validate_identifier_tuple(
                    self.observation_ids,
                    field_name="observation_ids",
                ),
            )
        if self.durable_root:
            raise ValueError("compiled artifacts may not become durable roots outside the claim ledger")
