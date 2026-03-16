"""Forgetting, retraction, and erasure contract invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class ForgettingMode(StrEnum):
    SUPERSEDE = "supersede"
    SUPPRESS = "suppress"
    SEAL = "seal"
    EXPUNGE = "expunge"


class ForgettingTargetKind(StrEnum):
    CLAIM = "claim"
    LOCUS = "locus"
    SUBJECT = "subject"
    SESSION = "session"
    IMPORTED_ARTIFACT = "imported_artifact"
    COMPILED_VIEW = "compiled_view"


class ForgettingSurface(StrEnum):
    CLAIM_LEDGER = "claim_ledger"
    OBSERVATION_LOG = "observation_log"
    VECTOR_INDEX = "vector_index"
    SNAPSHOT_STORE = "snapshot_store"
    PREFETCH_CACHE = "prefetch_cache"
    REPLAY_ARTIFACTS = "replay_artifacts"
    ARCHIVE_TIER = "archive_tier"
    IMPORT_PIPELINE = "import_pipeline"
    DERIVATION_PIPELINE = "derivation_pipeline"
    TOMBSTONE_LEDGER = "tombstone_ledger"


class ArtifactResidency(StrEnum):
    RETAIN_CONTENT = "retain_content"
    HIDDEN_FROM_HOST = "hidden_from_host"
    ADMIN_METADATA_ONLY = "admin_metadata_only"
    TOMBSTONE_ONLY = "tombstone_only"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class ForgettingTarget:
    target_kind: ForgettingTargetKind
    target_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))


@dataclass(frozen=True, slots=True)
class ForgettingOperation:
    operation_id: str
    target: ForgettingTarget
    mode: ForgettingMode
    requested_by: str
    rationale: str
    policy_stamp: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _clean_text(self.operation_id, field_name="operation_id"))
        object.__setattr__(self, "requested_by", _clean_text(self.requested_by, field_name="requested_by"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(
            self,
            "recorded_at",
            _validate_timestamp(self.recorded_at, field_name="recorded_at"),
        )


@dataclass(frozen=True, slots=True)
class ForgettingRule:
    mode: ForgettingMode
    residencies: dict[ForgettingSurface, ArtifactResidency]
    blocks_resurrection_from: tuple[ForgettingSurface, ...] = ()
    host_reads_withdrawn: bool = False

    def __post_init__(self) -> None:
        expected_surfaces = set(ForgettingSurface)
        provided_surfaces = set(self.residencies)
        if provided_surfaces != expected_surfaces:
            missing = expected_surfaces - provided_surfaces
            extras = provided_surfaces - expected_surfaces
            raise ValueError(
                "residencies must define every forgetting surface "
                f"(missing={sorted(surface.value for surface in missing)}, "
                f"extra={sorted(surface.value for surface in extras)})"
            )
        object.__setattr__(
            self,
            "blocks_resurrection_from",
            tuple(dict.fromkeys(self.blocks_resurrection_from)),
        )

    def residency_for(self, surface: ForgettingSurface) -> ArtifactResidency:
        return self.residencies[surface]

    def blocks_resurrection(self, surface: ForgettingSurface) -> bool:
        return surface in self.blocks_resurrection_from


@dataclass(frozen=True, slots=True)
class ForgettingDecisionTrace:
    operation: ForgettingOperation
    rule: ForgettingRule

    def __post_init__(self) -> None:
        if self.operation.mode is not self.rule.mode:
            raise ValueError("operation mode must match forgetting rule mode")

    def residency_for(self, surface: ForgettingSurface) -> ArtifactResidency:
        return self.rule.residency_for(surface)

    def blocks_resurrection(self, surface: ForgettingSurface) -> bool:
        return self.rule.blocks_resurrection(surface)

    @property
    def retained_surfaces(self) -> tuple[ForgettingSurface, ...]:
        return tuple(
            surface
            for surface in ForgettingSurface
            if self.residency_for(surface) is not ArtifactResidency.REMOVED
        )

    @property
    def removed_surfaces(self) -> tuple[ForgettingSurface, ...]:
        return tuple(
            surface
            for surface in ForgettingSurface
            if self.residency_for(surface) is ArtifactResidency.REMOVED
        )


@lru_cache(maxsize=1)
def forgetting_rules() -> dict[ForgettingMode, ForgettingRule]:
    return {
        ForgettingMode.SUPERSEDE: ForgettingRule(
            mode=ForgettingMode.SUPERSEDE,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.REMOVED,
            },
        ),
        ForgettingMode.SUPPRESS: ForgettingRule(
            mode=ForgettingMode.SUPPRESS,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.REMOVED,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.REMOVED,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.REMOVED,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.RETAIN_CONTENT,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.HIDDEN_FROM_HOST,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.ADMIN_METADATA_ONLY,
            },
            host_reads_withdrawn=True,
        ),
        ForgettingMode.SEAL: ForgettingRule(
            mode=ForgettingMode.SEAL,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.REMOVED,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.REMOVED,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.REMOVED,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.REMOVED,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.ADMIN_METADATA_ONLY,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.REMOVED,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.ADMIN_METADATA_ONLY,
            },
            blocks_resurrection_from=(
                ForgettingSurface.SNAPSHOT_STORE,
                ForgettingSurface.PREFETCH_CACHE,
                ForgettingSurface.REPLAY_ARTIFACTS,
                ForgettingSurface.DERIVATION_PIPELINE,
            ),
            host_reads_withdrawn=True,
        ),
        ForgettingMode.EXPUNGE: ForgettingRule(
            mode=ForgettingMode.EXPUNGE,
            residencies={
                ForgettingSurface.CLAIM_LEDGER: ArtifactResidency.REMOVED,
                ForgettingSurface.OBSERVATION_LOG: ArtifactResidency.REMOVED,
                ForgettingSurface.VECTOR_INDEX: ArtifactResidency.REMOVED,
                ForgettingSurface.SNAPSHOT_STORE: ArtifactResidency.REMOVED,
                ForgettingSurface.PREFETCH_CACHE: ArtifactResidency.REMOVED,
                ForgettingSurface.REPLAY_ARTIFACTS: ArtifactResidency.REMOVED,
                ForgettingSurface.ARCHIVE_TIER: ArtifactResidency.REMOVED,
                ForgettingSurface.IMPORT_PIPELINE: ArtifactResidency.REMOVED,
                ForgettingSurface.DERIVATION_PIPELINE: ArtifactResidency.REMOVED,
                ForgettingSurface.TOMBSTONE_LEDGER: ArtifactResidency.TOMBSTONE_ONLY,
            },
            blocks_resurrection_from=(
                ForgettingSurface.VECTOR_INDEX,
                ForgettingSurface.SNAPSHOT_STORE,
                ForgettingSurface.PREFETCH_CACHE,
                ForgettingSurface.REPLAY_ARTIFACTS,
                ForgettingSurface.ARCHIVE_TIER,
                ForgettingSurface.IMPORT_PIPELINE,
                ForgettingSurface.DERIVATION_PIPELINE,
            ),
            host_reads_withdrawn=True,
        ),
    }


def forgetting_rule_for(mode: ForgettingMode) -> ForgettingRule:
    return forgetting_rules()[mode]
