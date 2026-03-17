"""Generational tiering invariants for Continuity."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
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


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


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


@dataclass(frozen=True, slots=True)
class TierAssignment:
    target_kind: str
    target_id: str
    policy_stamp: str
    tier: MemoryTier
    rationale: str
    assigned_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_kind", _clean_text(self.target_kind, field_name="target_kind"))
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(self, "assigned_at", _validate_timestamp(self.assigned_at, field_name="assigned_at"))


@dataclass(frozen=True, slots=True)
class TierTransition:
    transition_id: str
    target_kind: str
    target_id: str
    policy_stamp: str
    from_tier: MemoryTier
    to_tier: MemoryTier
    rationale: str
    transitioned_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition_id", _clean_text(self.transition_id, field_name="transition_id"))
        object.__setattr__(self, "target_kind", _clean_text(self.target_kind, field_name="target_kind"))
        object.__setattr__(self, "target_id", _clean_text(self.target_id, field_name="target_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "rationale", _clean_text(self.rationale, field_name="rationale"))
        object.__setattr__(
            self,
            "transitioned_at",
            _validate_timestamp(self.transitioned_at, field_name="transitioned_at"),
        )
        if self.from_tier is self.to_tier:
            raise ValueError("tier transitions must change the tier")


@dataclass(frozen=True, slots=True)
class RetentionMetadata:
    target_kind: str
    target_id: str
    policy_stamp: str
    tier: MemoryTier
    rationale: str
    assigned_at: datetime
    retrieval_bias: RetrievalBias
    rebuild_urgency: RebuildUrgency
    snapshot_residency: SnapshotResidency
    default_in_host_reads: bool
    expunge_guarded: bool


class TierStateRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def upsert_assignment(self, assignment: TierAssignment) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO tier_assignments(
                    target_kind,
                    target_id,
                    policy_stamp,
                    tier,
                    rationale,
                    assigned_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_kind, target_id, policy_stamp) DO UPDATE SET
                    tier = excluded.tier,
                    rationale = excluded.rationale,
                    assigned_at = excluded.assigned_at
                """,
                (
                    assignment.target_kind,
                    assignment.target_id,
                    assignment.policy_stamp,
                    assignment.tier.value,
                    assignment.rationale,
                    assignment.assigned_at.isoformat(),
                ),
            )

    def read_assignment(
        self,
        *,
        target_kind: str,
        target_id: str,
        policy_stamp: str,
    ) -> TierAssignment | None:
        row = self._connection.execute(
            """
            SELECT target_kind, target_id, policy_stamp, tier, rationale, assigned_at
            FROM tier_assignments
            WHERE target_kind = ? AND target_id = ? AND policy_stamp = ?
            """,
            (
                _clean_text(target_kind, field_name="target_kind"),
                _clean_text(target_id, field_name="target_id"),
                _clean_text(policy_stamp, field_name="policy_stamp"),
            ),
        ).fetchone()
        if row is None:
            return None
        return self._assignment_from_row(row)

    def list_assignments(
        self,
        *,
        target_kind: str | None = None,
        policy_stamp: str | None = None,
        tiers: tuple[MemoryTier, ...] | None = None,
        limit: int | None = None,
    ) -> tuple[TierAssignment, ...]:
        conditions: list[str] = []
        parameters: list[object] = []

        if target_kind is not None:
            conditions.append("target_kind = ?")
            parameters.append(_clean_text(target_kind, field_name="target_kind"))
        if policy_stamp is not None:
            conditions.append("policy_stamp = ?")
            parameters.append(_clean_text(policy_stamp, field_name="policy_stamp"))
        if tiers is not None:
            cleaned_tiers = tuple(dict.fromkeys(tiers))
            if not cleaned_tiers:
                return ()
            placeholders = ", ".join("?" for _ in cleaned_tiers)
            conditions.append(f"tier IN ({placeholders})")
            parameters.extend(tier.value for tier in cleaned_tiers)

        sql = """
            SELECT target_kind, target_id, policy_stamp, tier, rationale, assigned_at
            FROM tier_assignments
        """
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY rowid"

        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)

        rows = self._connection.execute(sql, tuple(parameters)).fetchall()
        return tuple(self._assignment_from_row(row) for row in rows)

    def record_transition(self, transition: TierTransition) -> None:
        current_assignment = self.read_assignment(
            target_kind=transition.target_kind,
            target_id=transition.target_id,
            policy_stamp=transition.policy_stamp,
        )
        if current_assignment is None:
            raise ValueError("cannot record a tier transition without an existing assignment")
        if current_assignment.tier is not transition.from_tier:
            raise ValueError("tier transition must start from the current stored tier")

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO tier_transitions(
                    transition_id,
                    target_kind,
                    target_id,
                    policy_stamp,
                    from_tier,
                    to_tier,
                    rationale,
                    transitioned_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition.transition_id,
                    transition.target_kind,
                    transition.target_id,
                    transition.policy_stamp,
                    transition.from_tier.value,
                    transition.to_tier.value,
                    transition.rationale,
                    transition.transitioned_at.isoformat(),
                ),
            )
            self._connection.execute(
                """
                UPDATE tier_assignments
                SET tier = ?, rationale = ?, assigned_at = ?
                WHERE target_kind = ? AND target_id = ? AND policy_stamp = ?
                """,
                (
                    transition.to_tier.value,
                    transition.rationale,
                    transition.transitioned_at.isoformat(),
                    transition.target_kind,
                    transition.target_id,
                    transition.policy_stamp,
                ),
            )

    def list_transitions(
        self,
        *,
        target_kind: str | None = None,
        target_id: str | None = None,
        policy_stamp: str | None = None,
        limit: int | None = None,
    ) -> tuple[TierTransition, ...]:
        conditions: list[str] = []
        parameters: list[object] = []

        if target_kind is not None:
            conditions.append("target_kind = ?")
            parameters.append(_clean_text(target_kind, field_name="target_kind"))
        if target_id is not None:
            conditions.append("target_id = ?")
            parameters.append(_clean_text(target_id, field_name="target_id"))
        if policy_stamp is not None:
            conditions.append("policy_stamp = ?")
            parameters.append(_clean_text(policy_stamp, field_name="policy_stamp"))

        sql = """
            SELECT
                transition_id,
                target_kind,
                target_id,
                policy_stamp,
                from_tier,
                to_tier,
                rationale,
                transitioned_at
            FROM tier_transitions
        """
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY rowid"

        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)

        rows = self._connection.execute(sql, tuple(parameters)).fetchall()
        return tuple(self._transition_from_row(row) for row in rows)

    def read_retention_metadata(
        self,
        *,
        target_kind: str,
        target_id: str,
        policy_stamp: str,
    ) -> RetentionMetadata | None:
        assignment = self.read_assignment(
            target_kind=target_kind,
            target_id=target_id,
            policy_stamp=policy_stamp,
        )
        if assignment is None:
            return None
        return self._retention_metadata_from_assignment(assignment)

    def list_retention_metadata(
        self,
        *,
        target_kind: str | None = None,
        policy_stamp: str | None = None,
        tiers: tuple[MemoryTier, ...] | None = None,
        limit: int | None = None,
    ) -> tuple[RetentionMetadata, ...]:
        return tuple(
            self._retention_metadata_from_assignment(assignment)
            for assignment in self.list_assignments(
                target_kind=target_kind,
                policy_stamp=policy_stamp,
                tiers=tiers,
                limit=limit,
            )
        )

    def _assignment_from_row(self, row: tuple[object, ...]) -> TierAssignment:
        return TierAssignment(
            target_kind=row[0],
            target_id=row[1],
            policy_stamp=row[2],
            tier=MemoryTier(row[3]),
            rationale=row[4],
            assigned_at=_validate_timestamp(datetime.fromisoformat(row[5]), field_name="assigned_at"),
        )

    def _transition_from_row(self, row: tuple[object, ...]) -> TierTransition:
        return TierTransition(
            transition_id=row[0],
            target_kind=row[1],
            target_id=row[2],
            policy_stamp=row[3],
            from_tier=MemoryTier(row[4]),
            to_tier=MemoryTier(row[5]),
            rationale=row[6],
            transitioned_at=_validate_timestamp(
                datetime.fromisoformat(row[7]),
                field_name="transitioned_at",
            ),
        )

    def _retention_metadata_from_assignment(self, assignment: TierAssignment) -> RetentionMetadata:
        rule = tier_rule_for(assignment.tier)
        return RetentionMetadata(
            target_kind=assignment.target_kind,
            target_id=assignment.target_id,
            policy_stamp=assignment.policy_stamp,
            tier=assignment.tier,
            rationale=assignment.rationale,
            assigned_at=assignment.assigned_at,
            retrieval_bias=rule.retrieval_bias,
            rebuild_urgency=rule.rebuild_urgency,
            snapshot_residency=rule.snapshot_residency,
            default_in_host_reads=rule.default_in_host_reads,
            expunge_guarded=rule.expunge_guarded,
        )


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
