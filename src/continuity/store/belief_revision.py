"""Belief-state persistence and revision over stored claims."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from continuity.epistemics import (
    BeliefProjection,
    EpistemicAssessment,
    EpistemicStatus,
    EpistemicTarget,
    resolve_locus_belief,
)
from continuity.ontology import DecayMode
from continuity.policy import PolicyPack, get_policy_pack
from continuity.store.claims import AggregationMode, Claim, ClaimScope, MemoryLocus
from continuity.store.sqlite import SQLiteRepository


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    return _validate_timestamp(datetime.fromisoformat(value), field_name=field_name)


def _load_json_identifiers(value: str, *, field_name: str) -> tuple[str, ...]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError(f"{field_name} must decode to a JSON list")
    return tuple(_clean_text(item, field_name=field_name) for item in loaded)


@dataclass(frozen=True, slots=True)
class StoredBeliefState:
    belief_id: str
    policy_stamp: str
    projection: BeliefProjection
    as_of: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief_id", _clean_text(self.belief_id, field_name="belief_id"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "as_of", _validate_timestamp(self.as_of, field_name="as_of"))

    @property
    def subject_id(self) -> str:
        return self.projection.locus.subject_id

    @property
    def locus_key(self) -> str:
        return self.projection.locus.locus_key


class BeliefStateRepository:
    """Append-only storage and lookup helpers for belief projections."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    def record_state(self, state: StoredBeliefState) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO belief_states(
                    belief_id,
                    subject_id,
                    locus_id,
                    policy_stamp,
                    epistemic_status,
                    rationale,
                    active_claim_ids_json,
                    historical_claim_ids_json,
                    as_of
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(belief_id) DO UPDATE SET
                    subject_id = excluded.subject_id,
                    locus_id = excluded.locus_id,
                    policy_stamp = excluded.policy_stamp,
                    epistemic_status = excluded.epistemic_status,
                    rationale = excluded.rationale,
                    active_claim_ids_json = excluded.active_claim_ids_json,
                    historical_claim_ids_json = excluded.historical_claim_ids_json,
                    as_of = excluded.as_of
                """,
                (
                    state.belief_id,
                    state.subject_id,
                    self._locus_id_for(state.projection.locus),
                    state.policy_stamp,
                    state.projection.epistemic.status.value,
                    state.projection.epistemic.rationale,
                    json.dumps(list(state.projection.active_claim_ids)),
                    json.dumps(list(state.projection.historical_claim_ids)),
                    state.as_of.isoformat(),
                ),
            )

    def read_state(self, belief_id: str) -> StoredBeliefState | None:
        row = self._connection.execute(
            """
            SELECT
                belief_states.belief_id,
                belief_states.subject_id,
                belief_states.policy_stamp,
                belief_states.epistemic_status,
                belief_states.rationale,
                belief_states.active_claim_ids_json,
                belief_states.historical_claim_ids_json,
                belief_states.as_of,
                memory_loci.locus_key,
                memory_loci.scope,
                memory_loci.default_disclosure_policy_id,
                memory_loci.conflict_set_key,
                memory_loci.aggregation_mode
            FROM belief_states
            JOIN memory_loci ON memory_loci.locus_id = belief_states.locus_id
            WHERE belief_states.belief_id = ?
            """,
            (_clean_text(belief_id, field_name="belief_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._state_from_row(row)

    def read_current_state(
        self,
        *,
        subject_id: str,
        locus_key: str,
        policy_stamp: str,
    ) -> StoredBeliefState | None:
        row = self._connection.execute(
            """
            SELECT
                belief_states.belief_id,
                belief_states.subject_id,
                belief_states.policy_stamp,
                belief_states.epistemic_status,
                belief_states.rationale,
                belief_states.active_claim_ids_json,
                belief_states.historical_claim_ids_json,
                belief_states.as_of,
                memory_loci.locus_key,
                memory_loci.scope,
                memory_loci.default_disclosure_policy_id,
                memory_loci.conflict_set_key,
                memory_loci.aggregation_mode
            FROM belief_states
            JOIN memory_loci ON memory_loci.locus_id = belief_states.locus_id
            WHERE belief_states.subject_id = ?
              AND memory_loci.locus_key = ?
              AND belief_states.policy_stamp = ?
            ORDER BY belief_states.as_of DESC, belief_states.belief_id DESC
            LIMIT 1
            """,
            (
                _clean_text(subject_id, field_name="subject_id"),
                _clean_text(locus_key, field_name="locus_key"),
                _clean_text(policy_stamp, field_name="policy_stamp"),
            ),
        ).fetchone()
        if row is None:
            return None
        return self._state_from_row(row)

    def list_states(
        self,
        *,
        subject_id: str | None = None,
        locus_key: str | None = None,
        policy_stamp: str | None = None,
    ) -> tuple[StoredBeliefState, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if subject_id is not None:
            clauses.append("belief_states.subject_id = ?")
            parameters.append(_clean_text(subject_id, field_name="subject_id"))
        if locus_key is not None:
            clauses.append("memory_loci.locus_key = ?")
            parameters.append(_clean_text(locus_key, field_name="locus_key"))
        if policy_stamp is not None:
            clauses.append("belief_states.policy_stamp = ?")
            parameters.append(_clean_text(policy_stamp, field_name="policy_stamp"))
        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        rows = self._connection.execute(
            f"""
            SELECT
                belief_states.belief_id,
                belief_states.subject_id,
                belief_states.policy_stamp,
                belief_states.epistemic_status,
                belief_states.rationale,
                belief_states.active_claim_ids_json,
                belief_states.historical_claim_ids_json,
                belief_states.as_of,
                memory_loci.locus_key,
                memory_loci.scope,
                memory_loci.default_disclosure_policy_id,
                memory_loci.conflict_set_key,
                memory_loci.aggregation_mode
            FROM belief_states
            JOIN memory_loci ON memory_loci.locus_id = belief_states.locus_id
            {where}
            ORDER BY belief_states.as_of DESC, belief_states.belief_id DESC
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(self._state_from_row(row) for row in rows)

    def _state_from_row(self, row: sqlite3.Row) -> StoredBeliefState:
        locus = MemoryLocus(
            subject_id=row["subject_id"],
            locus_key=row["locus_key"],
            scope=ClaimScope(row["scope"]),
            default_disclosure_policy=row["default_disclosure_policy_id"],
            conflict_set_key=row["conflict_set_key"],
            aggregation_mode=AggregationMode(row["aggregation_mode"]),
        )
        projection = BeliefProjection(
            locus=locus,
            active_claim_ids=_load_json_identifiers(
                row["active_claim_ids_json"],
                field_name="active_claim_ids_json",
            ),
            historical_claim_ids=_load_json_identifiers(
                row["historical_claim_ids_json"],
                field_name="historical_claim_ids_json",
            ),
            epistemic=EpistemicAssessment(
                status=EpistemicStatus(row["epistemic_status"]),
                target=EpistemicTarget.LOCUS_RESOLUTION,
                rationale=row["rationale"],
            ),
        )
        return StoredBeliefState(
            belief_id=row["belief_id"],
            policy_stamp=row["policy_stamp"],
            projection=projection,
            as_of=_parse_timestamp(row["as_of"], field_name="as_of"),
        )

    @staticmethod
    def _locus_id_for(locus: MemoryLocus) -> str:
        return f"locus:{locus.subject_id}:{locus.locus_key}"


class BeliefRevisionEngine:
    """Resolve stored claims into durable belief projections."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._claims = SQLiteRepository(connection)
        self._beliefs = BeliefStateRepository(connection)

    def revise_subject(
        self,
        *,
        subject_id: str,
        as_of: datetime,
        policy_name: str = "hermes_v1",
    ) -> tuple[StoredBeliefState, ...]:
        claims = self._claims.list_claims(subject_id=subject_id)
        return self._revise_claim_groups(
            claim_groups=self._group_claims_by_locus(claims),
            as_of=as_of,
            policy=self._policy_for(policy_name),
        )

    def revise_all(
        self,
        *,
        as_of: datetime,
        policy_name: str = "hermes_v1",
    ) -> tuple[StoredBeliefState, ...]:
        claims = self._claims.list_claims()
        return self._revise_claim_groups(
            claim_groups=self._group_claims_by_locus(claims),
            as_of=as_of,
            policy=self._policy_for(policy_name),
        )

    def _revise_claim_groups(
        self,
        *,
        claim_groups: tuple[tuple[Claim, ...], ...],
        as_of: datetime,
        policy: PolicyPack,
    ) -> tuple[StoredBeliefState, ...]:
        revision_time = _validate_timestamp(as_of, field_name="as_of")
        states: list[StoredBeliefState] = []
        for claim_group in claim_groups:
            if not claim_group:
                continue
            projection = resolve_locus_belief(claim_group, as_of=revision_time)
            claim_type = self._single_claim_type_for(claim_group)
            projection = self._apply_type_decay_policy(
                projection=projection,
                claim_type=claim_type,
                policy=policy,
            )
            state = StoredBeliefState(
                belief_id=self._belief_id_for(
                    projection=projection,
                    claim_type=claim_type,
                    policy_stamp=policy.policy_stamp,
                    as_of=revision_time,
                ),
                policy_stamp=policy.policy_stamp,
                projection=projection,
                as_of=revision_time,
            )
            self._beliefs.record_state(state)
            states.append(state)
        return tuple(
            sorted(
                states,
                key=lambda state: (state.subject_id, state.locus_key, state.belief_id),
            )
        )

    @staticmethod
    def _group_claims_by_locus(claims: Iterable[Claim]) -> tuple[tuple[Claim, ...], ...]:
        groups: dict[tuple[str, str], list[Claim]] = {}
        for claim in claims:
            groups.setdefault(claim.locus.address, []).append(claim)
        return tuple(
            tuple(group)
            for _, group in sorted(groups.items(), key=lambda item: item[0])
        )

    @staticmethod
    def _single_claim_type_for(claims: tuple[Claim, ...]) -> str:
        claim_types = {claim.claim_type for claim in claims}
        if len(claim_types) != 1:
            raise ValueError("belief revision requires one claim type per locus")
        return next(iter(claim_types))

    @staticmethod
    def _policy_for(policy_name: str) -> PolicyPack:
        return get_policy_pack(_clean_text(policy_name, field_name="policy_name"))

    @staticmethod
    def _apply_type_decay_policy(
        *,
        projection: BeliefProjection,
        claim_type: str,
        policy: PolicyPack,
    ) -> BeliefProjection:
        if projection.epistemic.status is not EpistemicStatus.STALE:
            return projection

        decay_mode = policy.memory_class_spec_for(claim_type).decay_mode
        if decay_mode not in {DecayMode.REQUIRES_REFRESH, DecayMode.SESSION_ONLY}:
            return projection

        return BeliefProjection(
            locus=projection.locus,
            active_claim_ids=(),
            historical_claim_ids=projection.historical_claim_ids,
            epistemic=EpistemicAssessment(
                status=EpistemicStatus.STALE,
                target=EpistemicTarget.LOCUS_RESOLUTION,
                rationale=(
                    f"{projection.epistemic.rationale}; "
                    f"{claim_type} decay mode {decay_mode.value} removes stale claims from current belief"
                ),
            ),
        )

    @staticmethod
    def _belief_id_for(
        *,
        projection: BeliefProjection,
        claim_type: str,
        policy_stamp: str,
        as_of: datetime,
    ) -> str:
        digest = hashlib.sha256()
        for part in (
            projection.locus.subject_id,
            projection.locus.locus_key,
            claim_type,
            policy_stamp,
            as_of.isoformat(),
            projection.epistemic.status.value,
            ",".join(projection.active_claim_ids),
            ",".join(projection.historical_claim_ids),
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return (
            "belief:"
            f"{projection.locus.subject_id}:"
            f"{projection.locus.locus_key}:"
            f"{digest.hexdigest()[:16]}"
        )
