"""SQLite-backed repository primitives for Continuity durable state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from continuity.admission import AdmissionDecisionTrace, AdmissionRepository
from continuity.forgetting import ForgettingRecord, ForgettingRepository, ForgettingTarget
from continuity.outcomes import OutcomeLabel, OutcomeRepository, OutcomeRecord, OutcomeTarget
from continuity.resolution_queue import ResolutionQueueRepository, ResolutionQueueItem, ResolutionSurface
from continuity.store.replay import ReplayRepository
from continuity.store.claims import (
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimRelation,
    ClaimRelationKind,
    ClaimScope,
    MemoryLocus,
    Observation,
    Subject,
    SubjectAlias,
    SubjectKind,
    SubjectMergeRecord,
    SubjectSplitRecord,
)
from continuity.utility import CompiledUtilityWeight, UtilityRepository


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _parse_timestamp(value: str | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _validate_timestamp(datetime.fromisoformat(value), field_name=field_name)


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _load_json_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("expected a JSON object payload")
    return loaded


def _load_json_list(value: str) -> list[Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError("expected a JSON array payload")
    return loaded


def _make_locus_id(subject_id: str, locus_key: str) -> str:
    return f"locus:{subject_id}:{locus_key}"


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    host_namespace: str
    session_name: str
    recall_mode: str
    write_frequency: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _clean_text(self.session_id, field_name="session_id"))
        object.__setattr__(
            self,
            "host_namespace",
            _clean_text(self.host_namespace, field_name="host_namespace"),
        )
        object.__setattr__(
            self,
            "session_name",
            _clean_text(self.session_name, field_name="session_name"),
        )
        object.__setattr__(self, "recall_mode", _clean_text(self.recall_mode, field_name="recall_mode"))
        object.__setattr__(
            self,
            "write_frequency",
            _clean_text(self.write_frequency, field_name="write_frequency"),
        )
        object.__setattr__(self, "created_at", _validate_timestamp(self.created_at, field_name="created_at"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SessionMessageRecord:
    message_id: str
    session_id: str
    role: str
    author_subject_id: str
    content: str
    observed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _clean_text(self.message_id, field_name="message_id"))
        object.__setattr__(self, "session_id", _clean_text(self.session_id, field_name="session_id"))
        object.__setattr__(self, "role", _clean_text(self.role, field_name="role"))
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
class SessionBufferRecord:
    buffer_key: str
    session_id: str
    buffer_kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "buffer_key", _clean_text(self.buffer_key, field_name="buffer_key"))
        object.__setattr__(self, "session_id", _clean_text(self.session_id, field_name="session_id"))
        object.__setattr__(self, "buffer_kind", _clean_text(self.buffer_kind, field_name="buffer_kind"))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(
            self,
            "updated_at",
            _validate_timestamp(self.updated_at, field_name="updated_at"),
        )


@dataclass(frozen=True, slots=True)
class StoredDisclosurePolicy:
    policy_id: str
    audience_principal: str
    channel: str
    purpose: str
    exposure_mode: str
    redaction_mode: str
    capture_for_replay: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _clean_text(self.policy_id, field_name="policy_id"))
        object.__setattr__(
            self,
            "audience_principal",
            _clean_text(self.audience_principal, field_name="audience_principal"),
        )
        object.__setattr__(self, "channel", _clean_text(self.channel, field_name="channel"))
        object.__setattr__(self, "purpose", _clean_text(self.purpose, field_name="purpose"))
        object.__setattr__(
            self,
            "exposure_mode",
            _clean_text(self.exposure_mode, field_name="exposure_mode"),
        )
        object.__setattr__(
            self,
            "redaction_mode",
            _clean_text(self.redaction_mode, field_name="redaction_mode"),
        )
        object.__setattr__(self, "capture_for_replay", bool(self.capture_for_replay))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class StoredMemoryLocus:
    locus_id: str
    locus: MemoryLocus

    def __post_init__(self) -> None:
        object.__setattr__(self, "locus_id", _clean_text(self.locus_id, field_name="locus_id"))


class SQLiteRepository:
    """Explicit CRUD and lookup helpers over the canonical SQLite schema."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self.admissions = AdmissionRepository(connection)
        self.forgetting = ForgettingRepository(connection)
        self.outcomes = OutcomeRepository(connection)
        self.replay = ReplayRepository(connection)
        self.resolution_queue = ResolutionQueueRepository(connection)
        self.utility = UtilityRepository(connection)

    def save_disclosure_policy(self, policy: StoredDisclosurePolicy) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO disclosure_policies(
                    policy_id,
                    audience_principal,
                    channel,
                    purpose,
                    exposure_mode,
                    redaction_mode,
                    capture_for_replay,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(policy_id) DO UPDATE SET
                    audience_principal = excluded.audience_principal,
                    channel = excluded.channel,
                    purpose = excluded.purpose,
                    exposure_mode = excluded.exposure_mode,
                    redaction_mode = excluded.redaction_mode,
                    capture_for_replay = excluded.capture_for_replay,
                    metadata_json = excluded.metadata_json
                """,
                (
                    policy.policy_id,
                    policy.audience_principal,
                    policy.channel,
                    policy.purpose,
                    policy.exposure_mode,
                    policy.redaction_mode,
                    int(policy.capture_for_replay),
                    _dump_json(policy.metadata),
                ),
            )

    def read_disclosure_policy(self, policy_id: str) -> StoredDisclosurePolicy | None:
        row = self._connection.execute(
            """
            SELECT
                policy_id,
                audience_principal,
                channel,
                purpose,
                exposure_mode,
                redaction_mode,
                capture_for_replay,
                metadata_json
            FROM disclosure_policies
            WHERE policy_id = ?
            """,
            (_clean_text(policy_id, field_name="policy_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._disclosure_policy_from_row(row)

    def save_subject(
        self,
        subject: Subject,
        *,
        created_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        created = _validate_timestamp(created_at, field_name="created_at")
        subject_metadata = {} if metadata is None else dict(metadata)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO subjects(subject_id, kind, canonical_name, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(subject_id) DO UPDATE SET
                    kind = excluded.kind,
                    canonical_name = excluded.canonical_name,
                    metadata_json = excluded.metadata_json
                """,
                (
                    subject.subject_id,
                    subject.kind.value,
                    subject.canonical_name,
                    created.isoformat(),
                    _dump_json(subject_metadata),
                ),
            )
            self._replace_subject_collections(subject, recorded_at=created)

    def read_subject(self, subject_id: str) -> Subject | None:
        row = self._connection.execute(
            """
            SELECT subject_id, kind, canonical_name
            FROM subjects
            WHERE subject_id = ?
            """,
            (_clean_text(subject_id, field_name="subject_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._subject_from_row(row)

    def list_subjects(self, *, kind: SubjectKind | None = None) -> tuple[Subject, ...]:
        if kind is None:
            rows = self._connection.execute(
                """
                SELECT subject_id, kind, canonical_name
                FROM subjects
                ORDER BY subject_id
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT subject_id, kind, canonical_name
                FROM subjects
                WHERE kind = ?
                ORDER BY subject_id
                """,
                (kind.value,),
            ).fetchall()
        return tuple(self._subject_from_row(row) for row in rows)

    def resolve_subject(self, candidate: str, *, kind: SubjectKind | None = None) -> Subject | None:
        matches = tuple(subject for subject in self.list_subjects(kind=kind) if subject.matches_name(candidate))
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"subject resolution is ambiguous for {candidate!r}")
        return matches[0]

    def save_session(self, session: SessionRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions(
                    session_id,
                    host_namespace,
                    session_name,
                    recall_mode,
                    write_frequency,
                    created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    host_namespace = excluded.host_namespace,
                    session_name = excluded.session_name,
                    recall_mode = excluded.recall_mode,
                    write_frequency = excluded.write_frequency,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session.session_id,
                    session.host_namespace,
                    session.session_name,
                    session.recall_mode,
                    session.write_frequency,
                    session.created_at.isoformat(),
                    _dump_json(session.metadata),
                ),
            )

    def read_session(self, session_id: str) -> SessionRecord | None:
        row = self._connection.execute(
            """
            SELECT
                session_id,
                host_namespace,
                session_name,
                recall_mode,
                write_frequency,
                created_at,
                metadata_json
            FROM sessions
            WHERE session_id = ?
            """,
            (_clean_text(session_id, field_name="session_id"),),
        ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            host_namespace=row["host_namespace"],
            session_name=row["session_name"],
            recall_mode=row["recall_mode"],
            write_frequency=row["write_frequency"],
            created_at=_parse_timestamp(row["created_at"], field_name="created_at"),
            metadata=_load_json_object(row["metadata_json"]),
        )

    def save_message(self, message: SessionMessageRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO session_messages(
                    message_id,
                    session_id,
                    role,
                    author_subject_id,
                    content,
                    observed_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    role = excluded.role,
                    author_subject_id = excluded.author_subject_id,
                    content = excluded.content,
                    observed_at = excluded.observed_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.role,
                    message.author_subject_id,
                    message.content,
                    message.observed_at.isoformat(),
                    _dump_json(message.metadata),
                ),
            )

    def list_messages(self, *, session_id: str) -> tuple[SessionMessageRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                message_id,
                session_id,
                role,
                author_subject_id,
                content,
                observed_at,
                metadata_json
            FROM session_messages
            WHERE session_id = ?
            ORDER BY observed_at, message_id
            """,
            (_clean_text(session_id, field_name="session_id"),),
        ).fetchall()
        return tuple(
            SessionMessageRecord(
                message_id=row["message_id"],
                session_id=row["session_id"],
                role=row["role"],
                author_subject_id=row["author_subject_id"],
                content=row["content"],
                observed_at=_parse_timestamp(row["observed_at"], field_name="observed_at"),
                metadata=_load_json_object(row["metadata_json"]),
            )
            for row in rows
        )

    def save_session_buffer(self, buffer: SessionBufferRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO session_ephemeral_buffers(
                    buffer_key,
                    session_id,
                    buffer_kind,
                    payload_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(buffer_key) DO UPDATE SET
                    session_id = excluded.session_id,
                    buffer_kind = excluded.buffer_kind,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    buffer.buffer_key,
                    buffer.session_id,
                    buffer.buffer_kind,
                    _dump_json(buffer.payload),
                    buffer.updated_at.isoformat(),
                ),
            )

    def read_session_buffer(self, buffer_key: str) -> SessionBufferRecord | None:
        row = self._connection.execute(
            """
            SELECT
                buffer_key,
                session_id,
                buffer_kind,
                payload_json,
                updated_at
            FROM session_ephemeral_buffers
            WHERE buffer_key = ?
            """,
            (_clean_text(buffer_key, field_name="buffer_key"),),
        ).fetchone()
        if row is None:
            return None
        return SessionBufferRecord(
            buffer_key=row["buffer_key"],
            session_id=row["session_id"],
            buffer_kind=row["buffer_kind"],
            payload=_load_json_object(row["payload_json"]),
            updated_at=_parse_timestamp(row["updated_at"], field_name="updated_at"),
        )

    def list_session_buffers(
        self,
        *,
        session_id: str,
        buffer_kind: str | None = None,
    ) -> tuple[SessionBufferRecord, ...]:
        parameters: list[str] = [_clean_text(session_id, field_name="session_id")]
        where = ["session_id = ?"]
        if buffer_kind is not None:
            where.append("buffer_kind = ?")
            parameters.append(_clean_text(buffer_kind, field_name="buffer_kind"))
        rows = self._connection.execute(
            f"""
            SELECT
                buffer_key,
                session_id,
                buffer_kind,
                payload_json,
                updated_at
            FROM session_ephemeral_buffers
            WHERE {" AND ".join(where)}
            ORDER BY updated_at, buffer_key
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(
            SessionBufferRecord(
                buffer_key=row["buffer_key"],
                session_id=row["session_id"],
                buffer_kind=row["buffer_kind"],
                payload=_load_json_object(row["payload_json"]),
                updated_at=_parse_timestamp(row["updated_at"], field_name="updated_at"),
            )
            for row in rows
        )

    def delete_session_buffer(self, buffer_key: str) -> None:
        with self._connection:
            self._connection.execute(
                """
                DELETE FROM session_ephemeral_buffers
                WHERE buffer_key = ?
                """,
                (_clean_text(buffer_key, field_name="buffer_key"),),
            )

    def save_observation(self, observation: Observation, *, message_id: str | None = None) -> None:
        cleaned_message_id = None if message_id is None else _clean_text(message_id, field_name="message_id")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO observations(
                    observation_id,
                    source_kind,
                    session_id,
                    author_subject_id,
                    message_id,
                    content,
                    observed_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET
                    source_kind = excluded.source_kind,
                    session_id = excluded.session_id,
                    author_subject_id = excluded.author_subject_id,
                    message_id = excluded.message_id,
                    content = excluded.content,
                    observed_at = excluded.observed_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    observation.observation_id,
                    observation.source_kind,
                    observation.session_id,
                    observation.author_subject_id,
                    cleaned_message_id,
                    observation.content,
                    observation.observed_at.isoformat(),
                    _dump_json(observation.metadata),
                ),
            )

    def list_observations(
        self,
        *,
        session_id: str | None = None,
        author_subject_id: str | None = None,
    ) -> tuple[Observation, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            parameters.append(_clean_text(session_id, field_name="session_id"))
        if author_subject_id is not None:
            clauses.append("author_subject_id = ?")
            parameters.append(_clean_text(author_subject_id, field_name="author_subject_id"))
        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        rows = self._connection.execute(
            f"""
            SELECT
                observation_id,
                source_kind,
                session_id,
                author_subject_id,
                content,
                observed_at,
                metadata_json
            FROM observations
            {where}
            ORDER BY observed_at, observation_id
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(
            Observation(
                observation_id=row["observation_id"],
                source_kind=row["source_kind"],
                session_id=row["session_id"],
                author_subject_id=row["author_subject_id"],
                content=row["content"],
                observed_at=_parse_timestamp(row["observed_at"], field_name="observed_at"),
                metadata=_load_json_object(row["metadata_json"]),
            )
            for row in rows
        )

    def read_observation(self, observation_id: str) -> Observation | None:
        row = self._connection.execute(
            """
            SELECT
                observation_id,
                source_kind,
                session_id,
                author_subject_id,
                content,
                observed_at,
                metadata_json
            FROM observations
            WHERE observation_id = ?
            """,
            (_clean_text(observation_id, field_name="observation_id"),),
        ).fetchone()
        if row is None:
            return None
        return Observation(
            observation_id=row["observation_id"],
            source_kind=row["source_kind"],
            session_id=row["session_id"],
            author_subject_id=row["author_subject_id"],
            content=row["content"],
            observed_at=_parse_timestamp(row["observed_at"], field_name="observed_at"),
            metadata=_load_json_object(row["metadata_json"]),
        )

    def save_candidate_memory(self, candidate: CandidateMemory, *, created_at: datetime) -> None:
        created = _validate_timestamp(created_at, field_name="created_at")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO candidate_memories(
                    candidate_id,
                    claim_type,
                    subject_id,
                    scope,
                    value_json,
                    source_observation_ids_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    claim_type = excluded.claim_type,
                    subject_id = excluded.subject_id,
                    scope = excluded.scope,
                    value_json = excluded.value_json,
                    source_observation_ids_json = excluded.source_observation_ids_json,
                    created_at = excluded.created_at
                """,
                (
                    candidate.candidate_id,
                    candidate.claim_type,
                    candidate.subject_id,
                    candidate.scope.value,
                    _dump_json(candidate.value),
                    _dump_json(list(candidate.source_observation_ids)),
                    created.isoformat(),
                ),
            )

    def read_candidate_memory(self, candidate_id: str) -> CandidateMemory | None:
        row = self._connection.execute(
            """
            SELECT
                candidate_id,
                claim_type,
                subject_id,
                scope,
                value_json,
                source_observation_ids_json
            FROM candidate_memories
            WHERE candidate_id = ?
            """,
            (_clean_text(candidate_id, field_name="candidate_id"),),
        ).fetchone()
        if row is None:
            return None
        return CandidateMemory(
            candidate_id=row["candidate_id"],
            claim_type=row["claim_type"],
            subject_id=row["subject_id"],
            scope=ClaimScope(row["scope"]),
            value=json.loads(row["value_json"]),
            source_observation_ids=tuple(_load_json_list(row["source_observation_ids_json"])),
        )

    def list_candidate_memories(
        self,
        *,
        subject_id: str | None = None,
        scope: ClaimScope | None = None,
    ) -> tuple[CandidateMemory, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if subject_id is not None:
            clauses.append("subject_id = ?")
            parameters.append(_clean_text(subject_id, field_name="subject_id"))
        if scope is not None:
            clauses.append("scope = ?")
            parameters.append(scope.value)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        rows = self._connection.execute(
            f"""
            SELECT
                candidate_id,
                claim_type,
                subject_id,
                scope,
                value_json,
                source_observation_ids_json
            FROM candidate_memories
            {where}
            ORDER BY created_at, candidate_id
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(
            CandidateMemory(
                candidate_id=row["candidate_id"],
                claim_type=row["claim_type"],
                subject_id=row["subject_id"],
                scope=ClaimScope(row["scope"]),
                value=json.loads(row["value_json"]),
                source_observation_ids=tuple(_load_json_list(row["source_observation_ids_json"])),
            )
            for row in rows
        )

    def save_memory_locus(self, locus: MemoryLocus) -> StoredMemoryLocus:
        stored = StoredMemoryLocus(
            locus_id=_make_locus_id(locus.subject_id, locus.locus_key),
            locus=locus,
        )
        with self._connection:
            self._upsert_memory_locus(stored)
        return stored

    def read_memory_locus(self, subject_id: str, locus_key: str) -> StoredMemoryLocus | None:
        row = self._connection.execute(
            """
            SELECT
                locus_id,
                subject_id,
                locus_key,
                scope,
                default_disclosure_policy_id,
                conflict_set_key,
                aggregation_mode
            FROM memory_loci
            WHERE subject_id = ? AND locus_key = ?
            """,
            (
                _clean_text(subject_id, field_name="subject_id"),
                _clean_text(locus_key, field_name="locus_key"),
            ),
        ).fetchone()
        if row is None:
            return None
        return StoredMemoryLocus(
            locus_id=row["locus_id"],
            locus=MemoryLocus(
                subject_id=row["subject_id"],
                locus_key=row["locus_key"],
                scope=ClaimScope(row["scope"]),
                default_disclosure_policy=row["default_disclosure_policy_id"],
                conflict_set_key=row["conflict_set_key"],
                aggregation_mode=AggregationMode(row["aggregation_mode"]),
            ),
        )

    def save_claim(self, claim: Claim) -> None:
        stored_locus = StoredMemoryLocus(
            locus_id=_make_locus_id(claim.locus.subject_id, claim.locus.locus_key),
            locus=claim.locus,
        )
        with self._connection:
            self._upsert_memory_locus(stored_locus)
            self._connection.execute(
                """
                INSERT INTO claims(
                    claim_id,
                    claim_type,
                    subject_id,
                    locus_id,
                    scope,
                    disclosure_policy_id,
                    value_json,
                    candidate_id,
                    admission_candidate_id,
                    observed_at,
                    learned_at,
                    valid_from,
                    valid_to,
                    derivation_run_id,
                    confidence_json,
                    support_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    claim_type = excluded.claim_type,
                    subject_id = excluded.subject_id,
                    locus_id = excluded.locus_id,
                    scope = excluded.scope,
                    disclosure_policy_id = excluded.disclosure_policy_id,
                    value_json = excluded.value_json,
                    candidate_id = excluded.candidate_id,
                    admission_candidate_id = excluded.admission_candidate_id,
                    observed_at = excluded.observed_at,
                    learned_at = excluded.learned_at,
                    valid_from = excluded.valid_from,
                    valid_to = excluded.valid_to,
                    derivation_run_id = excluded.derivation_run_id
                """,
                (
                    claim.claim_id,
                    claim.claim_type,
                    claim.subject_id,
                    stored_locus.locus_id,
                    claim.scope.value,
                    claim.disclosure_policy,
                    _dump_json(claim.value),
                    claim.admission.candidate_id,
                    claim.admission.candidate_id,
                    claim.observed_at.isoformat(),
                    claim.learned_at.isoformat(),
                    None if claim.valid_from is None else claim.valid_from.isoformat(),
                    None if claim.valid_to is None else claim.valid_to.isoformat(),
                    claim.provenance.derivation_run_id,
                    _dump_json({}),
                    _dump_json({}),
                ),
            )
            self._connection.execute(
                "DELETE FROM claim_sources WHERE claim_id = ?",
                (claim.claim_id,),
            )
            self._connection.executemany(
                """
                INSERT INTO claim_sources(claim_id, observation_id, source_rank)
                VALUES (?, ?, ?)
                """,
                (
                    (claim.claim_id, observation_id, rank)
                    for rank, observation_id in enumerate(claim.provenance.observation_ids)
                ),
            )
            self._connection.execute(
                "DELETE FROM claim_relations WHERE claim_id = ?",
                (claim.claim_id,),
            )
            self._connection.executemany(
                """
                INSERT INTO claim_relations(claim_id, relation_kind, related_claim_id)
                VALUES (?, ?, ?)
                """,
                (
                    (claim.claim_id, relation.kind.value, relation.related_claim_id)
                    for relation in claim.relations
                ),
            )

    def read_claim(self, claim_id: str) -> Claim | None:
        row = self._connection.execute(
            """
            SELECT
                claims.claim_id,
                claims.claim_type,
                claims.subject_id,
                claims.scope,
                claims.disclosure_policy_id,
                claims.value_json,
                claims.admission_candidate_id,
                claims.observed_at,
                claims.learned_at,
                claims.valid_from,
                claims.valid_to,
                claims.derivation_run_id,
                memory_loci.subject_id AS locus_subject_id,
                memory_loci.locus_key,
                memory_loci.scope AS locus_scope,
                memory_loci.default_disclosure_policy_id,
                memory_loci.conflict_set_key,
                memory_loci.aggregation_mode
            FROM claims
            JOIN memory_loci ON memory_loci.locus_id = claims.locus_id
            WHERE claims.claim_id = ?
            """,
            (_clean_text(claim_id, field_name="claim_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._claim_from_row(row)

    def list_claims(
        self,
        *,
        subject_id: str | None = None,
        locus_key: str | None = None,
    ) -> tuple[Claim, ...]:
        clauses: list[str] = []
        parameters: list[str] = []
        if subject_id is not None:
            clauses.append("claims.subject_id = ?")
            parameters.append(_clean_text(subject_id, field_name="subject_id"))
        if locus_key is not None:
            clauses.append("memory_loci.locus_key = ?")
            parameters.append(_clean_text(locus_key, field_name="locus_key"))
        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        rows = self._connection.execute(
            f"""
            SELECT
                claims.claim_id,
                claims.claim_type,
                claims.subject_id,
                claims.scope,
                claims.disclosure_policy_id,
                claims.value_json,
                claims.admission_candidate_id,
                claims.observed_at,
                claims.learned_at,
                claims.valid_from,
                claims.valid_to,
                claims.derivation_run_id,
                memory_loci.subject_id AS locus_subject_id,
                memory_loci.locus_key,
                memory_loci.scope AS locus_scope,
                memory_loci.default_disclosure_policy_id,
                memory_loci.conflict_set_key,
                memory_loci.aggregation_mode
            FROM claims
            JOIN memory_loci ON memory_loci.locus_id = claims.locus_id
            {where}
            ORDER BY claims.learned_at DESC, claims.claim_id
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(self._claim_from_row(row) for row in rows)

    def read_admission_trace(self, candidate_id: str) -> AdmissionDecisionTrace | None:
        return self.admissions.read_decision(candidate_id)

    def list_resolution_items(
        self,
        *,
        subject_id: str | None = None,
        surface: ResolutionSurface | None = None,
        at_time: datetime | None = None,
        actionable_only: bool = False,
        limit: int | None = None,
    ) -> tuple[ResolutionQueueItem, ...]:
        return self.resolution_queue.list_items(
            subject_id=subject_id,
            surface=surface,
            at_time=at_time,
            actionable_only=actionable_only,
            limit=limit,
        )

    def current_forgetting_record(self, target: ForgettingTarget) -> ForgettingRecord | None:
        return self.forgetting.current_record_for_target(target)

    def read_outcome_record(self, outcome_id: str) -> OutcomeRecord | None:
        return self.outcomes.read_record(outcome_id)

    def list_outcome_records(
        self,
        *,
        target: OutcomeTarget | None = None,
        target_id: str | None = None,
        label: str | None = None,
        actor_subject_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[OutcomeRecord, ...]:
        outcome_label = None if label is None else OutcomeLabel(label)
        return self.outcomes.list_records(
            target=target,
            target_id=target_id,
            label=outcome_label,
            actor_subject_id=actor_subject_id,
            limit=limit,
        )

    def read_compiled_utility_weight(
        self,
        *,
        target: OutcomeTarget,
        target_id: str,
        policy_stamp: str,
    ) -> CompiledUtilityWeight | None:
        return self.utility.read_compiled_weight(
            target=target,
            target_id=target_id,
            policy_stamp=policy_stamp,
        )

    def _replace_subject_collections(self, subject: Subject, *, recorded_at: datetime) -> None:
        self._connection.execute(
            "DELETE FROM subject_aliases WHERE subject_id = ?",
            (subject.subject_id,),
        )
        self._connection.executemany(
            """
            INSERT INTO subject_aliases(
                subject_id,
                alias,
                normalized_alias,
                alias_type,
                source_observation_ids_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    subject.subject_id,
                    alias.alias,
                    alias.normalized_alias,
                    alias.alias_type,
                    _dump_json(list(alias.source_observation_ids)),
                )
                for alias in subject.aliases
            ),
        )
        self._connection.execute(
            "DELETE FROM subject_merges WHERE survivor_subject_id = ?",
            (subject.subject_id,),
        )
        self._connection.executemany(
            """
            INSERT INTO subject_merges(
                survivor_subject_id,
                merged_subject_id,
                source_observation_ids_json,
                recorded_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                (
                    merge.survivor_subject_id,
                    merged_subject_id,
                    _dump_json(list(merge.source_observation_ids)),
                    recorded_at.isoformat(),
                )
                for merge in subject.merges
                for merged_subject_id in merge.merged_subject_ids
            ),
        )
        self._connection.execute(
            "DELETE FROM subject_splits WHERE source_subject_id = ?",
            (subject.subject_id,),
        )
        self._connection.executemany(
            """
            INSERT INTO subject_splits(
                source_subject_id,
                child_subject_id,
                source_observation_ids_json,
                recorded_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                (
                    split.source_subject_id,
                    child_subject_id,
                    _dump_json(list(split.source_observation_ids)),
                    recorded_at.isoformat(),
                )
                for split in subject.splits
                for child_subject_id in split.child_subject_ids
            ),
        )

    def _subject_from_row(self, row: sqlite3.Row) -> Subject:
        subject_id = row["subject_id"]
        alias_rows = self._connection.execute(
            """
            SELECT alias, alias_type, source_observation_ids_json
            FROM subject_aliases
            WHERE subject_id = ?
            ORDER BY normalized_alias
            """,
            (subject_id,),
        ).fetchall()
        merge_rows = self._connection.execute(
            """
            SELECT merged_subject_id, source_observation_ids_json
            FROM subject_merges
            WHERE survivor_subject_id = ?
            ORDER BY merged_subject_id
            """,
            (subject_id,),
        ).fetchall()
        split_rows = self._connection.execute(
            """
            SELECT child_subject_id, source_observation_ids_json
            FROM subject_splits
            WHERE source_subject_id = ?
            ORDER BY child_subject_id
            """,
            (subject_id,),
        ).fetchall()
        merges_by_sources: dict[tuple[str, ...], list[str]] = {}
        for merge_row in merge_rows:
            key = tuple(_load_json_list(merge_row["source_observation_ids_json"]))
            merges_by_sources.setdefault(key, []).append(merge_row["merged_subject_id"])
        splits_by_sources: dict[tuple[str, ...], list[str]] = {}
        for split_row in split_rows:
            key = tuple(_load_json_list(split_row["source_observation_ids_json"]))
            splits_by_sources.setdefault(key, []).append(split_row["child_subject_id"])
        return Subject(
            subject_id=subject_id,
            kind=SubjectKind(row["kind"]),
            canonical_name=row["canonical_name"],
            aliases=tuple(
                SubjectAlias(
                    alias=alias_row["alias"],
                    alias_type=alias_row["alias_type"],
                    source_observation_ids=tuple(_load_json_list(alias_row["source_observation_ids_json"])),
                )
                for alias_row in alias_rows
            ),
            merges=tuple(
                SubjectMergeRecord(
                    survivor_subject_id=subject_id,
                    merged_subject_ids=tuple(merged_subject_ids),
                    source_observation_ids=source_ids,
                )
                for source_ids, merged_subject_ids in merges_by_sources.items()
            ),
            splits=tuple(
                SubjectSplitRecord(
                    source_subject_id=subject_id,
                    child_subject_ids=tuple(child_subject_ids),
                    source_observation_ids=source_ids,
                )
                for source_ids, child_subject_ids in splits_by_sources.items()
            ),
        )

    def _upsert_memory_locus(self, stored_locus: StoredMemoryLocus) -> None:
        self._connection.execute(
            """
            INSERT INTO memory_loci(
                locus_id,
                subject_id,
                locus_key,
                scope,
                default_disclosure_policy_id,
                conflict_set_key,
                aggregation_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(locus_id) DO UPDATE SET
                subject_id = excluded.subject_id,
                locus_key = excluded.locus_key,
                scope = excluded.scope,
                default_disclosure_policy_id = excluded.default_disclosure_policy_id,
                conflict_set_key = excluded.conflict_set_key,
                aggregation_mode = excluded.aggregation_mode
            """,
            (
                stored_locus.locus_id,
                stored_locus.locus.subject_id,
                stored_locus.locus.locus_key,
                stored_locus.locus.scope.value,
                stored_locus.locus.default_disclosure_policy,
                stored_locus.locus.conflict_set_key,
                stored_locus.locus.aggregation_mode.value,
            ),
        )

    def _claim_from_row(self, row: sqlite3.Row) -> Claim:
        admission_trace = self.read_admission_trace(row["admission_candidate_id"])
        if admission_trace is None:
            raise LookupError(f"missing admission trace for {row['admission_candidate_id']}")
        source_rows = self._connection.execute(
            """
            SELECT observation_id
            FROM claim_sources
            WHERE claim_id = ?
            ORDER BY source_rank, observation_id
            """,
            (row["claim_id"],),
        ).fetchall()
        relation_rows = self._connection.execute(
            """
            SELECT relation_kind, related_claim_id
            FROM claim_relations
            WHERE claim_id = ?
            ORDER BY relation_kind, related_claim_id
            """,
            (row["claim_id"],),
        ).fetchall()
        return Claim(
            claim_id=row["claim_id"],
            claim_type=row["claim_type"],
            subject_id=row["subject_id"],
            locus=MemoryLocus(
                subject_id=row["locus_subject_id"],
                locus_key=row["locus_key"],
                scope=ClaimScope(row["locus_scope"]),
                default_disclosure_policy=row["default_disclosure_policy_id"],
                conflict_set_key=row["conflict_set_key"],
                aggregation_mode=AggregationMode(row["aggregation_mode"]),
            ),
            scope=ClaimScope(row["scope"]),
            disclosure_policy=row["disclosure_policy_id"],
            value=json.loads(row["value_json"]),
            provenance=ClaimProvenance(
                observation_ids=tuple(source_row["observation_id"] for source_row in source_rows),
                derivation_run_id=row["derivation_run_id"],
            ),
            admission=admission_trace.decision,
            observed_at=_parse_timestamp(row["observed_at"], field_name="observed_at"),
            learned_at=_parse_timestamp(row["learned_at"], field_name="learned_at"),
            valid_from=_parse_timestamp(row["valid_from"], field_name="valid_from"),
            valid_to=_parse_timestamp(row["valid_to"], field_name="valid_to"),
            relations=tuple(
                ClaimRelation(
                    kind=ClaimRelationKind(relation_row["relation_kind"]),
                    related_claim_id=relation_row["related_claim_id"],
                )
                for relation_row in relation_rows
            ),
        )

    def _disclosure_policy_from_row(self, row: sqlite3.Row) -> StoredDisclosurePolicy:
        return StoredDisclosurePolicy(
            policy_id=row["policy_id"],
            audience_principal=row["audience_principal"],
            channel=row["channel"],
            purpose=row["purpose"],
            exposure_mode=row["exposure_mode"],
            redaction_mode=row["redaction_mode"],
            capture_for_replay=bool(row["capture_for_replay"]),
            metadata=_load_json_object(row["metadata_json"]),
        )


__all__ = [
    "SessionMessageRecord",
    "SessionRecord",
    "SQLiteRepository",
    "StoredDisclosurePolicy",
    "StoredMemoryLocus",
]
