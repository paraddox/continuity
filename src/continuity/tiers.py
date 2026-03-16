"""Generational tiering invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from continuity.ontology import MemoryClass
from continuity.policy import PolicyPack, hermes_v1_policy_pack
from continuity.store.claims import AdmissionOutcome
from continuity.views import TierDefault, ViewKind, view_contracts


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class MemoryTier(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    FROZEN = "frozen"


class RetrievalBias(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    RECALL_ONLY = "recall_only"
    AUDIT_ONLY = "audit_only"


class RebuildUrgency(StrEnum):
    IMMEDIATE = "immediate"
    SOON = "soon"
    BACKGROUND = "background"
    ARCHIVAL = "archival"


class SnapshotResidency(StrEnum):
    ACTIVE = "active"
    RECALLABLE = "recallable"
    ARCHIVAL_ONLY = "archival_only"


class ArchivalArtifactKind(StrEnum):
    REPLAY_RECORD = "replay_record"
    SNAPSHOT_HISTORY = "snapshot_history"
    EVALUATION_RESULT = "evaluation_result"


@dataclass(frozen=True, slots=True)
class TierRule:
    tier: MemoryTier
    retrieval_bias: RetrievalBias
    rebuild_urgency: RebuildUrgency
    snapshot_residency: SnapshotResidency
    default_in_host_reads: bool
    expunge_guarded: bool
    promotion_targets: tuple[MemoryTier, ...] = ()
    demotion_targets: tuple[MemoryTier, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "promotion_targets", tuple(dict.fromkeys(self.promotion_targets)))
        object.__setattr__(self, "demotion_targets", tuple(dict.fromkeys(self.demotion_targets)))
        if self.tier in self.promotion_targets:
            raise ValueError("promotion_targets cannot contain the current tier")
        if self.tier in self.demotion_targets:
            raise ValueError("demotion_targets cannot contain the current tier")


@dataclass(frozen=True, slots=True)
class ClaimTierRule:
    claim_type: str
    initial_tier: MemoryTier
    allowed_tiers: tuple[MemoryTier, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_type", _clean_text(self.claim_type, field_name="claim_type"))
        object.__setattr__(self, "allowed_tiers", tuple(dict.fromkeys(self.allowed_tiers)))
        if not self.allowed_tiers:
            raise ValueError("allowed_tiers must be non-empty")
        if self.initial_tier not in self.allowed_tiers:
            raise ValueError("initial_tier must be present in allowed_tiers")


@dataclass(frozen=True, slots=True)
class TierPolicy:
    policy_stamp: str
    claim_tiers: dict[str, ClaimTierRule]
    view_tiers: dict[ViewKind, tuple[MemoryTier, ...]]
    archival_tiers: dict[ArchivalArtifactKind, MemoryTier]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        if not self.claim_tiers:
            raise ValueError("claim_tiers must be non-empty")
        if set(self.view_tiers) != set(ViewKind):
            raise ValueError("view_tiers must cover every compiled view kind")
        if set(self.archival_tiers) != set(ArchivalArtifactKind):
            raise ValueError("archival_tiers must cover every archival artifact kind")


@lru_cache(maxsize=1)
def tier_rules() -> dict[MemoryTier, TierRule]:
    return {
        MemoryTier.HOT: TierRule(
            tier=MemoryTier.HOT,
            retrieval_bias=RetrievalBias.PRIMARY,
            rebuild_urgency=RebuildUrgency.IMMEDIATE,
            snapshot_residency=SnapshotResidency.ACTIVE,
            default_in_host_reads=True,
            expunge_guarded=False,
            demotion_targets=(MemoryTier.WARM, MemoryTier.COLD),
        ),
        MemoryTier.WARM: TierRule(
            tier=MemoryTier.WARM,
            retrieval_bias=RetrievalBias.SECONDARY,
            rebuild_urgency=RebuildUrgency.SOON,
            snapshot_residency=SnapshotResidency.ACTIVE,
            default_in_host_reads=True,
            expunge_guarded=False,
            promotion_targets=(MemoryTier.HOT,),
            demotion_targets=(MemoryTier.COLD,),
        ),
        MemoryTier.COLD: TierRule(
            tier=MemoryTier.COLD,
            retrieval_bias=RetrievalBias.RECALL_ONLY,
            rebuild_urgency=RebuildUrgency.BACKGROUND,
            snapshot_residency=SnapshotResidency.RECALLABLE,
            default_in_host_reads=False,
            expunge_guarded=True,
            promotion_targets=(MemoryTier.WARM, MemoryTier.HOT),
            demotion_targets=(MemoryTier.FROZEN,),
        ),
        MemoryTier.FROZEN: TierRule(
            tier=MemoryTier.FROZEN,
            retrieval_bias=RetrievalBias.AUDIT_ONLY,
            rebuild_urgency=RebuildUrgency.ARCHIVAL,
            snapshot_residency=SnapshotResidency.ARCHIVAL_ONLY,
            default_in_host_reads=False,
            expunge_guarded=True,
            promotion_targets=(MemoryTier.COLD,),
        ),
    }


def tier_rule_for(tier: MemoryTier) -> TierRule:
    return tier_rules()[tier]


_CLAIM_TIER_DEFAULTS: dict[MemoryClass, MemoryTier] = {
    MemoryClass.PREFERENCE: MemoryTier.WARM,
    MemoryClass.BIOGRAPHY: MemoryTier.WARM,
    MemoryClass.RELATIONSHIP: MemoryTier.WARM,
    MemoryClass.TASK_STATE: MemoryTier.HOT,
    MemoryClass.PROJECT_FACT: MemoryTier.WARM,
    MemoryClass.INSTRUCTION: MemoryTier.HOT,
    MemoryClass.COMMITMENT: MemoryTier.HOT,
    MemoryClass.OPEN_QUESTION: MemoryTier.HOT,
}

_VIEW_TIER_MAP: dict[TierDefault, MemoryTier] = {
    TierDefault.HOT: MemoryTier.HOT,
    TierDefault.WARM: MemoryTier.WARM,
    TierDefault.COLD: MemoryTier.COLD,
}


@lru_cache(maxsize=1)
def hermes_v1_tier_policy() -> TierPolicy:
    policy = hermes_v1_policy_pack()
    claim_tiers: dict[str, ClaimTierRule] = {}

    for spec in policy.ontology.types():
        if spec.supports_durable_promotion:
            try:
                initial_tier = _CLAIM_TIER_DEFAULTS[spec.memory_class]
            except KeyError as exc:
                raise ValueError(f"missing tier default for durable claim type: {spec.claim_type}") from exc
            claim_tiers[spec.claim_type] = ClaimTierRule(
                claim_type=spec.claim_type,
                initial_tier=initial_tier,
                allowed_tiers=(MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD),
            )
        elif spec.memory_class in _CLAIM_TIER_DEFAULTS:
            raise ValueError(f"non-durable claim type must not have a tier default: {spec.claim_type}")

    view_tiers = {
        kind: tuple(_VIEW_TIER_MAP[tier] for tier in contract.tier_defaults)
        for kind, contract in view_contracts().items()
    }

    archival_tiers = {
        ArchivalArtifactKind.REPLAY_RECORD: MemoryTier.FROZEN,
        ArchivalArtifactKind.SNAPSHOT_HISTORY: MemoryTier.FROZEN,
        ArchivalArtifactKind.EVALUATION_RESULT: MemoryTier.FROZEN,
    }

    return TierPolicy(
        policy_stamp=policy.policy_stamp,
        claim_tiers=claim_tiers,
        view_tiers=view_tiers,
        archival_tiers=archival_tiers,
    )


def initial_tier_for_claim_type(
    claim_type: str,
    *,
    admission_outcome: AdmissionOutcome,
    policy: PolicyPack,
) -> MemoryTier:
    spec = policy.memory_class_spec_for(claim_type)
    if not spec.supports_durable_promotion:
        raise ValueError("tiering only applies to durable claim types")
    if admission_outcome is not AdmissionOutcome.DURABLE_CLAIM:
        raise ValueError("tiering begins only after durable admission")

    return hermes_v1_tier_policy().claim_tiers[spec.claim_type].initial_tier
