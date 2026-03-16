"""Snapshot consistency invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from continuity.transactions import TransactionKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class SnapshotArtifactKind(StrEnum):
    STATE_VIEW = "state_view"
    TIMELINE_VIEW = "timeline_view"
    SET_VIEW = "set_view"
    PROFILE_VIEW = "profile_view"
    PROMPT_VIEW = "prompt_view"
    EVIDENCE_VIEW = "evidence_view"
    ANSWER_VIEW = "answer_view"
    VECTOR_INDEX = "vector_index"


class SnapshotHeadState(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"


class SnapshotReadUse(StrEnum):
    RETRIEVAL = "retrieval"
    PROMPT_ASSEMBLY = "prompt_assembly"
    ANSWER_QUERY = "answer_query"
    PREFETCH = "prefetch"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class SnapshotArtifactRef:
    artifact_kind: SnapshotArtifactKind
    artifact_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _clean_text(self.artifact_id, field_name="artifact_id"))


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    snapshot_id: str
    policy_stamp: str
    parent_snapshot_id: str | None
    created_by_transaction: TransactionKind
    artifact_refs: tuple[SnapshotArtifactRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        if self.parent_snapshot_id is not None:
            object.__setattr__(
                self,
                "parent_snapshot_id",
                _clean_text(self.parent_snapshot_id, field_name="parent_snapshot_id"),
            )
        object.__setattr__(self, "artifact_refs", tuple(dict.fromkeys(self.artifact_refs)))

        if not self.artifact_refs:
            raise ValueError("artifact_refs must be non-empty")
        if self.parent_snapshot_id == self.snapshot_id:
            raise ValueError("parent_snapshot_id must differ from snapshot_id")


@dataclass(frozen=True, slots=True)
class SnapshotHead:
    head_key: str
    state: SnapshotHeadState
    snapshot_id: str
    based_on_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_key", _clean_text(self.head_key, field_name="head_key"))
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        if self.based_on_snapshot_id is not None:
            object.__setattr__(
                self,
                "based_on_snapshot_id",
                _clean_text(self.based_on_snapshot_id, field_name="based_on_snapshot_id"),
            )

        if self.state is SnapshotHeadState.ACTIVE and self.based_on_snapshot_id is not None:
            raise ValueError("active heads cannot keep a based_on_snapshot_id")
        if self.state is SnapshotHeadState.CANDIDATE and self.based_on_snapshot_id is None:
            raise ValueError("candidate heads must declare the snapshot they are based on")


@dataclass(frozen=True, slots=True)
class SnapshotReadPin:
    snapshot_id: str
    read_use: SnapshotReadUse
    consumer_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(self, "consumer_id", _clean_text(self.consumer_id, field_name="consumer_id"))


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    from_snapshot_id: str
    to_snapshot_id: str
    added_artifacts: tuple[SnapshotArtifactRef, ...]
    removed_artifacts: tuple[SnapshotArtifactRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_snapshot_id",
            _clean_text(self.from_snapshot_id, field_name="from_snapshot_id"),
        )
        object.__setattr__(self, "to_snapshot_id", _clean_text(self.to_snapshot_id, field_name="to_snapshot_id"))
        object.__setattr__(self, "added_artifacts", tuple(dict.fromkeys(self.added_artifacts)))
        object.__setattr__(self, "removed_artifacts", tuple(dict.fromkeys(self.removed_artifacts)))


@dataclass(frozen=True, slots=True)
class SnapshotPromotion:
    previous_active_snapshot_id: str
    promoted_snapshot_id: str
    new_active_head: SnapshotHead

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_active_snapshot_id",
            _clean_text(
                self.previous_active_snapshot_id,
                field_name="previous_active_snapshot_id",
            ),
        )
        object.__setattr__(
            self,
            "promoted_snapshot_id",
            _clean_text(self.promoted_snapshot_id, field_name="promoted_snapshot_id"),
        )
        if self.new_active_head.state is not SnapshotHeadState.ACTIVE:
            raise ValueError("promotion must yield an active head")
        if self.new_active_head.snapshot_id != self.promoted_snapshot_id:
            raise ValueError("new_active_head must point at the promoted snapshot")


@dataclass(frozen=True, slots=True)
class SnapshotRollback:
    previous_active_snapshot_id: str
    rollback_to_snapshot_id: str
    reason: str
    new_active_head: SnapshotHead

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_active_snapshot_id",
            _clean_text(
                self.previous_active_snapshot_id,
                field_name="previous_active_snapshot_id",
            ),
        )
        object.__setattr__(
            self,
            "rollback_to_snapshot_id",
            _clean_text(self.rollback_to_snapshot_id, field_name="rollback_to_snapshot_id"),
        )
        object.__setattr__(self, "reason", _clean_text(self.reason, field_name="reason"))
        if self.new_active_head.state is not SnapshotHeadState.ACTIVE:
            raise ValueError("rollback must yield an active head")
        if self.new_active_head.snapshot_id != self.rollback_to_snapshot_id:
            raise ValueError("new_active_head must point at the rollback target")


def diff_snapshots(from_snapshot: MemorySnapshot, to_snapshot: MemorySnapshot) -> SnapshotDiff:
    if from_snapshot.policy_stamp != to_snapshot.policy_stamp:
        raise ValueError("snapshot diffs require a shared policy stamp")

    from_refs = set(from_snapshot.artifact_refs)
    to_refs = set(to_snapshot.artifact_refs)

    return SnapshotDiff(
        from_snapshot_id=from_snapshot.snapshot_id,
        to_snapshot_id=to_snapshot.snapshot_id,
        added_artifacts=tuple(ref for ref in to_snapshot.artifact_refs if ref not in from_refs),
        removed_artifacts=tuple(ref for ref in from_snapshot.artifact_refs if ref not in to_refs),
    )


def promote_candidate_head(*, active_head: SnapshotHead, candidate_head: SnapshotHead) -> SnapshotPromotion:
    if active_head.state is not SnapshotHeadState.ACTIVE:
        raise ValueError("promotion requires an active head")
    if candidate_head.state is not SnapshotHeadState.CANDIDATE:
        raise ValueError("promotion requires a candidate head")
    if active_head.head_key != candidate_head.head_key:
        raise ValueError("promotion requires one shared head_key")
    if candidate_head.based_on_snapshot_id != active_head.snapshot_id:
        raise ValueError("candidate head must be based on the current active snapshot")

    return SnapshotPromotion(
        previous_active_snapshot_id=active_head.snapshot_id,
        promoted_snapshot_id=candidate_head.snapshot_id,
        new_active_head=SnapshotHead(
            head_key=active_head.head_key,
            state=SnapshotHeadState.ACTIVE,
            snapshot_id=candidate_head.snapshot_id,
        ),
    )


def rollback_active_head(
    *,
    active_head: SnapshotHead,
    rollback_to_snapshot_id: str,
    reason: str,
) -> SnapshotRollback:
    if active_head.state is not SnapshotHeadState.ACTIVE:
        raise ValueError("rollback requires an active head")

    cleaned_target = _clean_text(rollback_to_snapshot_id, field_name="rollback_to_snapshot_id")

    return SnapshotRollback(
        previous_active_snapshot_id=active_head.snapshot_id,
        rollback_to_snapshot_id=cleaned_target,
        reason=reason,
        new_active_head=SnapshotHead(
            head_key=active_head.head_key,
            state=SnapshotHeadState.ACTIVE,
            snapshot_id=cleaned_target,
        ),
    )
