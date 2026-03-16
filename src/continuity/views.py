"""Compiled view algebra invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from continuity.epistemics import EpistemicStatus


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _dedupe_cleaned(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    cleaned_values = tuple(_clean_text(value, field_name=field_name) for value in values)
    return tuple(dict.fromkeys(cleaned_values))


class ViewKind(StrEnum):
    STATE = "state"
    TIMELINE = "timeline"
    SET = "set"
    PROFILE = "profile"
    PROMPT = "prompt"
    EVIDENCE = "evidence"
    ANSWER = "answer"


class ProvenanceSurface(StrEnum):
    CLAIM_IDS = "claim_ids"
    CLAIMS_AND_OBSERVATIONS = "claims_and_observations"


class SnapshotBinding(StrEnum):
    LOCUS_SNAPSHOT = "locus_snapshot"
    SUBJECT_SNAPSHOT = "subject_snapshot"
    SESSION_SNAPSHOT = "session_snapshot"
    TARGET_SNAPSHOT = "target_snapshot"
    QUERY_SNAPSHOT = "query_snapshot"


class TierDefault(StrEnum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class ViewContract:
    kind: ViewKind
    input_boundaries: tuple[str, ...]
    dependency_view_kinds: tuple[ViewKind, ...]
    snapshot_binding: SnapshotBinding
    provenance_surface: ProvenanceSurface
    disclosure_purposes: tuple[str, ...]
    tier_defaults: tuple[TierDefault, ...]
    deterministic: bool = True
    cacheable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_boundaries",
            _dedupe_cleaned(self.input_boundaries, field_name="input_boundaries"),
        )
        object.__setattr__(
            self,
            "disclosure_purposes",
            _dedupe_cleaned(self.disclosure_purposes, field_name="disclosure_purposes"),
        )
        object.__setattr__(
            self,
            "dependency_view_kinds",
            tuple(dict.fromkeys(self.dependency_view_kinds)),
        )
        object.__setattr__(
            self,
            "tier_defaults",
            tuple(dict.fromkeys(self.tier_defaults)),
        )

        if not self.input_boundaries:
            raise ValueError("input_boundaries must be non-empty")
        if not self.disclosure_purposes:
            raise ValueError("disclosure_purposes must be non-empty")
        if not self.tier_defaults:
            raise ValueError("tier_defaults must be non-empty")


@dataclass(frozen=True, slots=True)
class CompiledView:
    kind: ViewKind
    view_key: str
    policy_stamp: str
    snapshot_id: str
    claim_ids: tuple[str, ...]
    epistemic_status: EpistemicStatus
    observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "view_key", _clean_text(self.view_key, field_name="view_key"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "snapshot_id", _clean_text(self.snapshot_id, field_name="snapshot_id"))
        object.__setattr__(self, "claim_ids", _dedupe_cleaned(self.claim_ids, field_name="claim_ids"))
        object.__setattr__(
            self,
            "observation_ids",
            _dedupe_cleaned(self.observation_ids, field_name="observation_ids"),
        )

        if not self.claim_ids:
            raise ValueError("compiled views require claim provenance")

        if (
            view_contract_for(self.kind).provenance_surface is ProvenanceSurface.CLAIMS_AND_OBSERVATIONS
            and not self.observation_ids
        ):
            raise ValueError("this view kind requires observation provenance")


@lru_cache(maxsize=1)
def view_contracts() -> dict[ViewKind, ViewContract]:
    contracts = {
        ViewKind.STATE: ViewContract(
            kind=ViewKind.STATE,
            input_boundaries=("belief_projection", "claim_ledger"),
            dependency_view_kinds=(),
            snapshot_binding=SnapshotBinding.LOCUS_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIM_IDS,
            disclosure_purposes=("prompt", "answer", "profile", "search"),
            tier_defaults=(TierDefault.HOT, TierDefault.WARM),
        ),
        ViewKind.TIMELINE: ViewContract(
            kind=ViewKind.TIMELINE,
            input_boundaries=("claim_ledger", "revision_history"),
            dependency_view_kinds=(),
            snapshot_binding=SnapshotBinding.LOCUS_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIMS_AND_OBSERVATIONS,
            disclosure_purposes=("prompt", "answer", "evidence", "replay"),
            tier_defaults=(TierDefault.WARM, TierDefault.COLD),
        ),
        ViewKind.SET: ViewContract(
            kind=ViewKind.SET,
            input_boundaries=("belief_projection", "claim_ledger"),
            dependency_view_kinds=(),
            snapshot_binding=SnapshotBinding.LOCUS_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIM_IDS,
            disclosure_purposes=("prompt", "answer", "profile", "search"),
            tier_defaults=(TierDefault.HOT, TierDefault.WARM),
        ),
        ViewKind.PROFILE: ViewContract(
            kind=ViewKind.PROFILE,
            input_boundaries=("subject_projection", "active_loci"),
            dependency_view_kinds=(ViewKind.STATE, ViewKind.SET),
            snapshot_binding=SnapshotBinding.SUBJECT_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIM_IDS,
            disclosure_purposes=("profile", "prompt", "answer"),
            tier_defaults=(TierDefault.HOT, TierDefault.WARM),
        ),
        ViewKind.PROMPT: ViewContract(
            kind=ViewKind.PROMPT,
            input_boundaries=("session_context", "peer_context", "policy_pack"),
            dependency_view_kinds=(
                ViewKind.STATE,
                ViewKind.TIMELINE,
                ViewKind.SET,
                ViewKind.PROFILE,
                ViewKind.EVIDENCE,
            ),
            snapshot_binding=SnapshotBinding.SESSION_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIM_IDS,
            disclosure_purposes=("prompt",),
            tier_defaults=(TierDefault.HOT, TierDefault.WARM),
        ),
        ViewKind.EVIDENCE: ViewContract(
            kind=ViewKind.EVIDENCE,
            input_boundaries=("claim_ledger", "observation_log"),
            dependency_view_kinds=(),
            snapshot_binding=SnapshotBinding.TARGET_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIMS_AND_OBSERVATIONS,
            disclosure_purposes=("evidence", "answer", "replay"),
            tier_defaults=(TierDefault.WARM, TierDefault.COLD),
        ),
        ViewKind.ANSWER: ViewContract(
            kind=ViewKind.ANSWER,
            input_boundaries=("query_scope", "policy_pack", "snapshot_head"),
            dependency_view_kinds=(
                ViewKind.STATE,
                ViewKind.TIMELINE,
                ViewKind.SET,
                ViewKind.PROFILE,
                ViewKind.EVIDENCE,
            ),
            snapshot_binding=SnapshotBinding.QUERY_SNAPSHOT,
            provenance_surface=ProvenanceSurface.CLAIM_IDS,
            disclosure_purposes=("answer",),
            tier_defaults=(TierDefault.HOT, TierDefault.WARM),
        ),
    }
    return contracts


def view_contract_for(kind: ViewKind) -> ViewContract:
    return view_contracts()[kind]
