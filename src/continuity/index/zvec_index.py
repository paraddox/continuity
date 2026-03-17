"""zvec-backed indexing and rebuild flow for Continuity retrieval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import sqrt
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from continuity.compiler import (
    CompiledArtifactKind,
    CompilerDependency,
    CompilerNode,
    CompilerNodeCategory,
    DependencyRole,
    DerivedArtifactKind,
    SourceInputKind,
)
from continuity.store.belief_revision import BeliefStateRepository, StoredBeliefState
from continuity.store.claims import Claim, Observation
from continuity.store.sqlite import SQLiteRepository, SessionMessageRecord
from continuity.views import CompiledView, ViewKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _validate_topk(value: int) -> int:
    if value <= 0:
        raise ValueError("topk must be positive")
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _hash_fingerprint(payload: Any) -> str:
    serialized = _json_dumps(payload).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _zvec_document_id(record_id: str) -> str:
    cleaned_record_id = _clean_text(record_id, field_name="record_id")
    digest = hashlib.sha256(cleaned_record_id.encode("utf-8")).hexdigest()
    return f"record_{digest[:56]}"


def _zvec_result_record_id(result: Any) -> str:
    fields = getattr(result, "fields", None)
    if not isinstance(fields, Mapping):
        raise RuntimeError("zvec query result is missing fields metadata")
    record_id = fields.get("record_id")
    if not isinstance(record_id, str):
        raise RuntimeError("zvec query result is missing the authoritative record_id field")
    return _clean_text(record_id, field_name="record_id")


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_item * right_item for left_item, right_item in zip(left, right, strict=True))


def _magnitude(vector: Sequence[float]) -> float:
    return sqrt(sum(item * item for item in vector))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_magnitude = _magnitude(left)
    right_magnitude = _magnitude(right)
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0
    return _dot(left, right) / (left_magnitude * right_magnitude)


def _ensure_json_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(value)


class IndexSourceKind(StrEnum):
    SESSION_MESSAGE = "session_message"
    OBSERVATION = "observation"
    CLAIM = "claim"
    BELIEF_STATE = "belief_state"
    COMPILED_VIEW = "compiled_view"


@dataclass(frozen=True, slots=True)
class VectorIndexRecord:
    record_id: str
    source_kind: IndexSourceKind
    source_id: str
    subject_id: str | None
    locus_key: str | None
    policy_stamp: str
    document_text: str
    embedding_model: str
    embedding_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _clean_text(self.record_id, field_name="record_id"))
        object.__setattr__(self, "source_id", _clean_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "document_text", _clean_text(self.document_text, field_name="document_text"))
        object.__setattr__(self, "embedding_model", _clean_text(self.embedding_model, field_name="embedding_model"))
        object.__setattr__(
            self,
            "embedding_fingerprint",
            _clean_text(self.embedding_fingerprint, field_name="embedding_fingerprint"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class StoredCompiledView:
    compiled_view_id: str
    view: CompiledView
    payload: dict[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compiled_view_id",
            _clean_text(self.compiled_view_id, field_name="compiled_view_id"),
        )
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True, slots=True)
class DraftIndexEntry:
    record_id: str
    source_kind: IndexSourceKind
    source_id: str
    subject_id: str | None
    locus_key: str | None
    policy_stamp: str
    document_text: str
    metadata: dict[str, Any]
    source_node: CompilerNode
    extra_dependency_nodes: tuple[CompilerNode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _clean_text(self.record_id, field_name="record_id"))
        object.__setattr__(self, "source_id", _clean_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))
        object.__setattr__(self, "document_text", _clean_text(self.document_text, field_name="document_text"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "extra_dependency_nodes", tuple(self.extra_dependency_nodes))


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    record: VectorIndexRecord
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("vector documents require at least one dimension")
        object.__setattr__(self, "vector", tuple(float(item) for item in self.vector))


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    record_id: str
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _clean_text(self.record_id, field_name="record_id"))
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class IndexSearchHit:
    record: VectorIndexRecord
    score: float
    source: Any
    source_node: CompilerNode

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class IndexRebuildResult:
    records: tuple[VectorIndexRecord, ...]
    compiler_nodes: tuple[CompilerNode, ...]
    compiler_dependencies: tuple[CompilerDependency, ...]
    deleted_record_ids: tuple[str, ...]


@runtime_checkable
class EmbeddingBatchProtocol(Protocol):
    model: str
    fingerprint: str
    embeddings: tuple[tuple[float, ...], ...]


@runtime_checkable
class EmbeddingClientProtocol(Protocol):
    def embed(self, inputs: str | Sequence[str]) -> EmbeddingBatchProtocol: ...


@runtime_checkable
class VectorBackendProtocol(Protocol):
    def upsert_documents(self, documents: Iterable[IndexedDocument]) -> None: ...

    def delete_documents(self, document_ids: Iterable[str]) -> None: ...

    def query(
        self,
        *,
        query_vector: Sequence[float],
        topk: int,
        subject_id: str | None = None,
        source_kinds: Sequence[IndexSourceKind] | None = None,
    ) -> tuple[ScoredDocument, ...]: ...


class InMemoryZvecBackend:
    """Pure-Python backend used for local tests and fallback-free unit coverage."""

    def __init__(self) -> None:
        self._documents: dict[str, IndexedDocument] = {}

    def upsert_documents(self, documents: Iterable[IndexedDocument]) -> None:
        for document in documents:
            self._documents[document.record.record_id] = document

    def delete_documents(self, document_ids: Iterable[str]) -> None:
        for document_id in document_ids:
            self._documents.pop(_clean_text(document_id, field_name="document_id"), None)

    def query(
        self,
        *,
        query_vector: Sequence[float],
        topk: int,
        subject_id: str | None = None,
        source_kinds: Sequence[IndexSourceKind] | None = None,
    ) -> tuple[ScoredDocument, ...]:
        cleaned_topk = _validate_topk(topk)
        source_kind_filter = None if source_kinds is None else set(source_kinds)
        scored: list[ScoredDocument] = []
        for document in self._documents.values():
            record = document.record
            if subject_id is not None and record.subject_id != subject_id:
                continue
            if source_kind_filter is not None and record.source_kind not in source_kind_filter:
                continue
            scored.append(
                ScoredDocument(
                    record_id=record.record_id,
                    score=_cosine_similarity(query_vector, document.vector),
                )
            )
        ranked = sorted(scored, key=lambda item: (-item.score, item.record_id))
        return tuple(ranked[:cleaned_topk])


class ZvecBackend:
    """Thin wrapper over the optional third-party zvec Python package."""

    def __init__(self, *, collection_path: str, dimensions: int, collection_name: str = "continuity_index") -> None:
        cleaned_path = _clean_text(collection_path, field_name="collection_path")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        try:
            import zvec  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only with real zvec installs
            raise RuntimeError("Install the 'zvec' Python package to use the real zvec backend") from exc

        self._zvec = zvec
        if hasattr(self._zvec, "init"):
            self._zvec.init()
        if Path(cleaned_path).exists():
            self._collection = self._zvec.open(cleaned_path)
        else:
            self._collection = self._zvec.create_and_open(
                path=cleaned_path,
                schema=self._zvec.CollectionSchema(
                    name=_clean_text(collection_name, field_name="collection_name"),
                    fields=[
                        self._zvec.FieldSchema("record_id", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("source_kind", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("source_id", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("subject_id", self._zvec.DataType.STRING, nullable=True),
                        self._zvec.FieldSchema("locus_key", self._zvec.DataType.STRING, nullable=True),
                        self._zvec.FieldSchema("policy_stamp", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("document_text", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("embedding_model", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("embedding_fingerprint", self._zvec.DataType.STRING),
                        self._zvec.FieldSchema("metadata_json", self._zvec.DataType.STRING),
                    ],
                    vectors=[
                        self._zvec.VectorSchema(
                            "embedding",
                            self._zvec.DataType.VECTOR_FP32,
                            dimensions,
                        )
                    ],
                ),
            )

    def upsert_documents(self, documents: Iterable[IndexedDocument]) -> None:  # pragma: no cover - real backend path
        docs = tuple(documents)
        if not docs:
            return
        statuses = self._collection.upsert(
            [
                self._zvec.Doc(
                    id=_zvec_document_id(document.record.record_id),
                    vectors={"embedding": list(document.vector)},
                    fields={
                        "record_id": document.record.record_id,
                        "source_kind": document.record.source_kind.value,
                        "source_id": document.record.source_id,
                        "subject_id": document.record.subject_id,
                        "locus_key": document.record.locus_key,
                        "policy_stamp": document.record.policy_stamp,
                        "document_text": document.record.document_text,
                        "embedding_model": document.record.embedding_model,
                        "embedding_fingerprint": document.record.embedding_fingerprint,
                        "metadata_json": _json_dumps(document.record.metadata),
                    },
                )
                for document in docs
            ]
        )
        self._assert_statuses(statuses)
        self._collection.flush()

    def delete_documents(self, document_ids: Iterable[str]) -> None:  # pragma: no cover - real backend path
        ids = tuple(_zvec_document_id(document_id) for document_id in document_ids)
        if not ids:
            return
        statuses = self._collection.delete(list(ids))
        self._assert_statuses(statuses)
        self._collection.flush()

    def query(
        self,
        *,
        query_vector: Sequence[float],
        topk: int,
        subject_id: str | None = None,
        source_kinds: Sequence[IndexSourceKind] | None = None,
    ) -> tuple[ScoredDocument, ...]:  # pragma: no cover - real backend path
        filter_expr = _build_zvec_filter(subject_id=subject_id, source_kinds=source_kinds)
        raw_results = self._collection.query(
            vectors=self._zvec.VectorQuery("embedding", vector=list(query_vector)),
            topk=_validate_topk(topk),
            filter=filter_expr,
            output_fields=("record_id", "source_kind", "source_id"),
            include_vector=False,
        )
        return tuple(
            ScoredDocument(
                record_id=_zvec_result_record_id(result),
                score=float(result.score),
            )
            for result in raw_results
        )

    def _assert_statuses(self, statuses: Any) -> None:
        if isinstance(statuses, (list, tuple)):
            items = tuple(statuses)
        else:
            items = (statuses,)
        for status in items:
            ok_method = getattr(status, "ok", None)
            if callable(ok_method):
                if not ok_method():
                    message = getattr(status, "message", "zvec operation failed")
                    raise RuntimeError(str(message))


def _build_zvec_filter(
    *,
    subject_id: str | None,
    source_kinds: Sequence[IndexSourceKind] | None,
) -> str | None:
    clauses: list[str] = []
    if subject_id is not None:
        cleaned_subject_id = _clean_text(subject_id, field_name="subject_id").replace("'", "\\'")
        clauses.append(f"subject_id = '{cleaned_subject_id}'")
    if source_kinds:
        source_kind_clauses = " OR ".join(f"source_kind = '{kind.value}'" for kind in source_kinds)
        clauses.append(f"({source_kind_clauses})")
    if not clauses:
        return None
    return " AND ".join(clauses)


class VectorIndexStore:
    """Direct SQLite persistence helpers for vector index records and compiled view reads."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    def list_messages(self) -> tuple[SessionMessageRecord, ...]:
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
            ORDER BY observed_at, message_id
            """
        ).fetchall()
        return tuple(self._message_from_row(row) for row in rows)

    def read_message(self, message_id: str) -> SessionMessageRecord | None:
        row = self._connection.execute(
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
            WHERE message_id = ?
            """,
            (_clean_text(message_id, field_name="message_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._message_from_row(row)

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
            metadata=_ensure_json_object(_json_loads(row["metadata_json"]), field_name="metadata_json"),
        )

    def list_compiled_views(self) -> tuple[StoredCompiledView, ...]:
        rows = self._connection.execute(
            """
            SELECT compiled_view_id
            FROM compiled_views
            ORDER BY created_at, compiled_view_id
            """
        ).fetchall()
        loaded: list[StoredCompiledView] = []
        for row in rows:
            stored_view = self.read_compiled_view(row["compiled_view_id"])
            if stored_view is not None:
                loaded.append(stored_view)
        return tuple(loaded)

    def read_compiled_view(self, compiled_view_id: str) -> StoredCompiledView | None:
        row = self._connection.execute(
            """
            SELECT
                compiled_view_id,
                kind,
                view_key,
                policy_stamp,
                snapshot_id,
                epistemic_status,
                payload_json,
                created_at
            FROM compiled_views
            WHERE compiled_view_id = ?
            """,
            (_clean_text(compiled_view_id, field_name="compiled_view_id"),),
        ).fetchone()
        if row is None:
            return None

        claim_rows = self._connection.execute(
            """
            SELECT claim_id
            FROM compiled_view_claims
            WHERE compiled_view_id = ?
            ORDER BY claim_id
            """,
            (compiled_view_id,),
        ).fetchall()
        observation_rows = self._connection.execute(
            """
            SELECT observation_id
            FROM compiled_view_observations
            WHERE compiled_view_id = ?
            ORDER BY observation_id
            """,
            (compiled_view_id,),
        ).fetchall()

        return StoredCompiledView(
            compiled_view_id=row["compiled_view_id"],
            view=CompiledView(
                kind=ViewKind(row["kind"]),
                view_key=row["view_key"],
                policy_stamp=row["policy_stamp"],
                snapshot_id=row["snapshot_id"],
                claim_ids=tuple(item["claim_id"] for item in claim_rows),
                observation_ids=tuple(item["observation_id"] for item in observation_rows),
                epistemic_status=row["epistemic_status"],
            ),
            payload=_ensure_json_object(_json_loads(row["payload_json"]), field_name="payload_json"),
            created_at=_parse_timestamp(row["created_at"], field_name="created_at"),
        )

    def replace_records(self, records: Iterable[VectorIndexRecord]) -> tuple[str, ...]:
        rows = tuple(records)
        existing_ids = {
            row["record_id"]
            for row in self._connection.execute(
                """
                SELECT record_id
                FROM vector_index_records
                """
            ).fetchall()
        }
        next_ids = {record.record_id for record in rows}
        deleted_ids = tuple(sorted(existing_ids - next_ids))
        with self._connection:
            if rows:
                self._connection.executemany(
                    """
                    INSERT INTO vector_index_records(
                        record_id,
                        source_kind,
                        source_id,
                        subject_id,
                        locus_key,
                        policy_stamp,
                        document_text,
                        embedding_model,
                        embedding_fingerprint,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        source_kind = excluded.source_kind,
                        source_id = excluded.source_id,
                        subject_id = excluded.subject_id,
                        locus_key = excluded.locus_key,
                        policy_stamp = excluded.policy_stamp,
                        document_text = excluded.document_text,
                        embedding_model = excluded.embedding_model,
                        embedding_fingerprint = excluded.embedding_fingerprint,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        (
                            record.record_id,
                            record.source_kind.value,
                            record.source_id,
                            record.subject_id,
                            record.locus_key,
                            record.policy_stamp,
                            record.document_text,
                            record.embedding_model,
                            record.embedding_fingerprint,
                            _json_dumps(record.metadata),
                        )
                        for record in rows
                    ),
                )
            if deleted_ids:
                self._connection.executemany(
                    """
                    DELETE FROM vector_index_records
                    WHERE record_id = ?
                    """,
                    ((record_id,) for record_id in deleted_ids),
                )
        return deleted_ids

    def list_records(self) -> tuple[VectorIndexRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT
                record_id,
                source_kind,
                source_id,
                subject_id,
                locus_key,
                policy_stamp,
                document_text,
                embedding_model,
                embedding_fingerprint,
                metadata_json
            FROM vector_index_records
            ORDER BY record_id
            """
        ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def read_record(self, record_id: str) -> VectorIndexRecord | None:
        row = self._connection.execute(
            """
            SELECT
                record_id,
                source_kind,
                source_id,
                subject_id,
                locus_key,
                policy_stamp,
                document_text,
                embedding_model,
                embedding_fingerprint,
                metadata_json
            FROM vector_index_records
            WHERE record_id = ?
            """,
            (_clean_text(record_id, field_name="record_id"),),
        ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def _message_from_row(self, row: sqlite3.Row) -> SessionMessageRecord:
        return SessionMessageRecord(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=row["role"],
            author_subject_id=row["author_subject_id"],
            content=row["content"],
            observed_at=_parse_timestamp(row["observed_at"], field_name="observed_at"),
            metadata=_ensure_json_object(_json_loads(row["metadata_json"]), field_name="metadata_json"),
        )

    def _record_from_row(self, row: sqlite3.Row) -> VectorIndexRecord:
        return VectorIndexRecord(
            record_id=row["record_id"],
            source_kind=IndexSourceKind(row["source_kind"]),
            source_id=row["source_id"],
            subject_id=row["subject_id"],
            locus_key=row["locus_key"],
            policy_stamp=row["policy_stamp"],
            document_text=row["document_text"],
            embedding_model=row["embedding_model"],
            embedding_fingerprint=row["embedding_fingerprint"],
            metadata=_ensure_json_object(_json_loads(row["metadata_json"]), field_name="metadata_json"),
        )


_VIEW_KIND_TO_COMPILER_KIND: dict[ViewKind, CompiledArtifactKind] = {
    ViewKind.STATE: CompiledArtifactKind.STATE_VIEW,
    ViewKind.TIMELINE: CompiledArtifactKind.TIMELINE_VIEW,
    ViewKind.SET: CompiledArtifactKind.SET_VIEW,
    ViewKind.PROFILE: CompiledArtifactKind.PROFILE_VIEW,
    ViewKind.PROMPT: CompiledArtifactKind.PROMPT_VIEW,
    ViewKind.EVIDENCE: CompiledArtifactKind.EVIDENCE_VIEW,
    ViewKind.ANSWER: CompiledArtifactKind.ANSWER_VIEW,
}


class ZvecIndex:
    """Builds and queries the rebuildable vector index over SQLite-backed memory state."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        embedding_client: EmbeddingClientProtocol,
        backend: VectorBackendProtocol,
        policy_stamp: str,
    ) -> None:
        self._connection = connection
        self._store = VectorIndexStore(connection)
        self._embedding_client = embedding_client
        self._backend = backend
        self._policy_stamp = _clean_text(policy_stamp, field_name="policy_stamp")

    def list_records(self) -> tuple[VectorIndexRecord, ...]:
        return self._store.list_records()

    def rebuild_from_sqlite(self) -> IndexRebuildResult:
        drafts = self._load_draft_entries()
        existing_records = self._store.list_records()
        existing_ids = {record.record_id for record in existing_records}
        if not drafts:
            deleted_ids = tuple(sorted(existing_ids))
            if deleted_ids:
                self._backend.delete_documents(deleted_ids)
                self._store.replace_records(())
            return IndexRebuildResult(
                records=(),
                compiler_nodes=(),
                compiler_dependencies=(),
                deleted_record_ids=deleted_ids,
            )

        batch = self._embedding_client.embed(tuple(draft.document_text for draft in drafts))
        if len(batch.embeddings) != len(drafts):
            raise ValueError("embedding client must return one embedding per indexed document")

        records = tuple(
            VectorIndexRecord(
                record_id=draft.record_id,
                source_kind=draft.source_kind,
                source_id=draft.source_id,
                subject_id=draft.subject_id,
                locus_key=draft.locus_key,
                policy_stamp=draft.policy_stamp,
                document_text=draft.document_text,
                embedding_model=batch.model,
                embedding_fingerprint=batch.fingerprint,
                metadata=draft.metadata,
            )
            for draft in drafts
        )
        documents = tuple(
            IndexedDocument(record=record, vector=batch.embeddings[index])
            for index, record in enumerate(records)
        )
        deleted_record_ids = self._store.replace_records(records)
        if deleted_record_ids:
            self._backend.delete_documents(deleted_record_ids)
        self._backend.upsert_documents(documents)

        compiler_nodes, compiler_dependencies = self._compiler_artifacts_for(records=records, drafts=drafts)
        return IndexRebuildResult(
            records=records,
            compiler_nodes=compiler_nodes,
            compiler_dependencies=compiler_dependencies,
            deleted_record_ids=deleted_record_ids,
        )

    def search(
        self,
        query_text: str,
        *,
        topk: int = 5,
        subject_id: str | None = None,
        source_kinds: Sequence[IndexSourceKind] | None = None,
    ) -> tuple[IndexSearchHit, ...]:
        batch = self._embedding_client.embed(_clean_text(query_text, field_name="query_text"))
        if len(batch.embeddings) != 1:
            raise ValueError("query embedding must return exactly one vector")
        scored_documents = self._backend.query(
            query_vector=batch.embeddings[0],
            topk=_validate_topk(topk),
            subject_id=subject_id,
            source_kinds=source_kinds,
        )
        hits: list[IndexSearchHit] = []
        for scored in scored_documents:
            record = self._store.read_record(scored.record_id)
            if record is None:
                continue
            source = self._load_source(record)
            hits.append(
                IndexSearchHit(
                    record=record,
                    score=scored.score,
                    source=source,
                    source_node=self._source_node_for_loaded_source(record.source_kind, source),
                )
            )
        return tuple(hits)

    def _load_draft_entries(self) -> tuple[DraftIndexEntry, ...]:
        repository = SQLiteRepository(self._connection)
        beliefs = BeliefStateRepository(self._connection)
        claims = repository.list_claims()
        claim_by_id = {claim.claim_id: claim for claim in claims}
        observations = repository.list_observations()
        observation_by_id = {observation.observation_id: observation for observation in observations}

        drafts: list[DraftIndexEntry] = []
        drafts.extend(self._drafts_for_messages(self._store.list_messages()))
        drafts.extend(self._drafts_for_observations(observations))
        drafts.extend(self._drafts_for_claims(claims))
        drafts.extend(self._drafts_for_beliefs(beliefs.list_states(), claim_by_id=claim_by_id))
        drafts.extend(
            self._drafts_for_compiled_views(
                self._store.list_compiled_views(),
                claim_by_id=claim_by_id,
                observation_by_id=observation_by_id,
            )
        )
        return tuple(sorted(drafts, key=lambda draft: draft.record_id))

    def _drafts_for_messages(self, messages: Sequence[SessionMessageRecord]) -> list[DraftIndexEntry]:
        drafts: list[DraftIndexEntry] = []
        for message in messages:
            source_node = self._message_node(message)
            drafts.append(
                DraftIndexEntry(
                    record_id=f"vector:message:{message.message_id}",
                    source_kind=IndexSourceKind.SESSION_MESSAGE,
                    source_id=message.message_id,
                    subject_id=message.author_subject_id,
                    locus_key=None,
                    policy_stamp=self._policy_stamp,
                    document_text=f"{message.role} message {message.content}",
                    metadata={
                        "index_view": "raw_history",
                        "session_id": message.session_id,
                        "role": message.role,
                    },
                    source_node=source_node,
                )
            )
        return drafts

    def _drafts_for_observations(self, observations: Sequence[Observation]) -> list[DraftIndexEntry]:
        drafts: list[DraftIndexEntry] = []
        for observation in observations:
            source_node = self._observation_node(observation)
            drafts.append(
                DraftIndexEntry(
                    record_id=f"vector:observation:{observation.observation_id}",
                    source_kind=IndexSourceKind.OBSERVATION,
                    source_id=observation.observation_id,
                    subject_id=observation.author_subject_id,
                    locus_key=observation.metadata.get("locus_key"),
                    policy_stamp=self._policy_stamp,
                    document_text=f"observation {observation.source_kind} {observation.content}",
                    metadata={
                        "index_view": "raw_history",
                        "session_id": observation.session_id,
                        "author_subject_id": observation.author_subject_id,
                        "source_kind": observation.source_kind,
                    },
                    source_node=source_node,
                )
            )
        return drafts

    def _drafts_for_claims(self, claims: Sequence[Claim]) -> list[DraftIndexEntry]:
        drafts: list[DraftIndexEntry] = []
        for claim in claims:
            source_node = self._claim_node(claim)
            drafts.append(
                DraftIndexEntry(
                    record_id=f"vector:claim:{claim.claim_id}",
                    source_kind=IndexSourceKind.CLAIM,
                    source_id=claim.claim_id,
                    subject_id=claim.subject_id,
                    locus_key=claim.locus.locus_key,
                    policy_stamp=self._policy_stamp,
                    document_text=self._claim_text(claim),
                    metadata={
                        "index_view": "claim",
                        "claim_type": claim.claim_type,
                        "observation_ids": list(claim.provenance.observation_ids),
                    },
                    source_node=source_node,
                )
            )
        return drafts

    def _drafts_for_beliefs(
        self,
        beliefs: Sequence[StoredBeliefState],
        *,
        claim_by_id: dict[str, Claim],
    ) -> list[DraftIndexEntry]:
        drafts: list[DraftIndexEntry] = []
        for belief in beliefs:
            source_node = self._belief_node(belief)
            claim_nodes: list[CompilerNode] = []
            active_texts: list[str] = []
            historical_texts: list[str] = []
            for claim_id in belief.projection.active_claim_ids:
                claim = claim_by_id[claim_id]
                claim_nodes.append(self._claim_node(claim))
                active_texts.append(self._claim_text(claim))
            for claim_id in belief.projection.historical_claim_ids:
                claim = claim_by_id[claim_id]
                if claim_id not in belief.projection.active_claim_ids:
                    claim_nodes.append(self._claim_node(claim))
                historical_texts.append(self._claim_text(claim))
            drafts.append(
                DraftIndexEntry(
                    record_id=f"vector:belief:{belief.belief_id}",
                    source_kind=IndexSourceKind.BELIEF_STATE,
                    source_id=belief.belief_id,
                    subject_id=belief.subject_id,
                    locus_key=belief.locus_key,
                    policy_stamp=belief.policy_stamp,
                    document_text=" ".join(
                        (
                            "belief state",
                            belief.subject_id,
                            belief.locus_key,
                            belief.projection.epistemic.status.value,
                            "active",
                            " ".join(active_texts),
                            "history",
                            " ".join(historical_texts),
                        )
                    ).strip(),
                    metadata={
                        "index_view": "belief_state",
                        "epistemic_status": belief.projection.epistemic.status.value,
                        "active_claim_ids": list(belief.projection.active_claim_ids),
                        "historical_claim_ids": list(belief.projection.historical_claim_ids),
                    },
                    source_node=source_node,
                    extra_dependency_nodes=tuple(dict.fromkeys(claim_nodes)),
                )
            )
        return drafts

    def _drafts_for_compiled_views(
        self,
        compiled_views: Sequence[StoredCompiledView],
        *,
        claim_by_id: dict[str, Claim],
        observation_by_id: dict[str, Observation],
    ) -> list[DraftIndexEntry]:
        drafts: list[DraftIndexEntry] = []
        for stored_view in compiled_views:
            subject_id = None
            locus_key = None
            claim_texts: list[str] = []
            observation_texts: list[str] = []
            if stored_view.view.claim_ids:
                first_claim = claim_by_id[stored_view.view.claim_ids[0]]
                subject_id = first_claim.subject_id
                locus_key = first_claim.locus.locus_key
            for claim_id in stored_view.view.claim_ids:
                claim_texts.append(self._claim_text(claim_by_id[claim_id]))
            for observation_id in stored_view.view.observation_ids:
                observation_texts.append(observation_by_id[observation_id].content)
            drafts.append(
                DraftIndexEntry(
                    record_id=f"vector:view:{stored_view.compiled_view_id}",
                    source_kind=IndexSourceKind.COMPILED_VIEW,
                    source_id=stored_view.compiled_view_id,
                    subject_id=subject_id,
                    locus_key=locus_key,
                    policy_stamp=stored_view.view.policy_stamp,
                    document_text=" ".join(
                        (
                            "compiled view",
                            stored_view.view.kind.value,
                            stored_view.view.view_key,
                            _json_dumps(stored_view.payload),
                            " ".join(claim_texts),
                            " ".join(observation_texts),
                        )
                    ).strip(),
                    metadata={
                        "index_view": "compiled_view",
                        "view_kind": stored_view.view.kind.value,
                        "snapshot_id": stored_view.view.snapshot_id,
                    },
                    source_node=self._compiled_view_node(stored_view),
                )
            )
        return drafts

    def _compiler_artifacts_for(
        self,
        *,
        records: Sequence[VectorIndexRecord],
        drafts: Sequence[DraftIndexEntry],
    ) -> tuple[tuple[CompilerNode, ...], tuple[CompilerDependency, ...]]:
        nodes_by_id: dict[str, CompilerNode] = {}
        dependencies: dict[tuple[str, str, DependencyRole], CompilerDependency] = {}

        for record, draft in zip(records, drafts, strict=True):
            for source_node in (draft.source_node, *draft.extra_dependency_nodes):
                nodes_by_id[source_node.node_id] = source_node

            vector_node = CompilerNode(
                node_id=record.record_id,
                category=CompilerNodeCategory.COMPILED_ARTIFACT,
                kind=CompiledArtifactKind.VECTOR_INDEX_RECORD,
                fingerprint=_hash_fingerprint(
                    {
                        "record_id": record.record_id,
                        "document_text": record.document_text,
                        "embedding_fingerprint": record.embedding_fingerprint,
                        "source_kind": record.source_kind.value,
                        "source_id": record.source_id,
                        "metadata": record.metadata,
                    }
                ),
                subject_id=record.subject_id,
                locus_key=record.locus_key,
            )
            nodes_by_id[vector_node.node_id] = vector_node

            for source_node in (draft.source_node, *draft.extra_dependency_nodes):
                key = (source_node.node_id, vector_node.node_id, DependencyRole.INDEX)
                dependencies[key] = CompilerDependency(
                    upstream_node_id=source_node.node_id,
                    downstream_node_id=vector_node.node_id,
                    role=DependencyRole.INDEX,
                )

        return (
            tuple(sorted(nodes_by_id.values(), key=lambda node: node.node_id)),
            tuple(
                sorted(
                    dependencies.values(),
                    key=lambda dependency: (
                        dependency.upstream_node_id,
                        dependency.downstream_node_id,
                        dependency.role.value,
                    ),
                )
            ),
        )

    def _load_source(self, record: VectorIndexRecord) -> Any:
        repository = SQLiteRepository(self._connection)
        beliefs = BeliefStateRepository(self._connection)
        if record.source_kind is IndexSourceKind.SESSION_MESSAGE:
            source = self._store.read_message(record.source_id)
        elif record.source_kind is IndexSourceKind.OBSERVATION:
            source = self._store.read_observation(record.source_id)
        elif record.source_kind is IndexSourceKind.CLAIM:
            source = repository.read_claim(record.source_id)
        elif record.source_kind is IndexSourceKind.BELIEF_STATE:
            source = beliefs.read_state(record.source_id)
        elif record.source_kind is IndexSourceKind.COMPILED_VIEW:
            source = self._store.read_compiled_view(record.source_id)
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported source kind: {record.source_kind}")
        if source is None:
            raise KeyError(record.source_id)
        return source

    def _source_node_for_loaded_source(self, source_kind: IndexSourceKind, source: Any) -> CompilerNode:
        if source_kind is IndexSourceKind.SESSION_MESSAGE:
            return self._message_node(source)
        if source_kind is IndexSourceKind.OBSERVATION:
            return self._observation_node(source)
        if source_kind is IndexSourceKind.CLAIM:
            return self._claim_node(source)
        if source_kind is IndexSourceKind.BELIEF_STATE:
            return self._belief_node(source)
        if source_kind is IndexSourceKind.COMPILED_VIEW:
            return self._compiled_view_node(source)
        raise ValueError(f"unsupported source kind: {source_kind}")

    def _message_node(self, message: SessionMessageRecord) -> CompilerNode:
        return CompilerNode(
            node_id=f"message:{message.message_id}",
            category=CompilerNodeCategory.SOURCE_INPUT,
            kind=SourceInputKind.OBSERVATION,
            fingerprint=_hash_fingerprint(
                {
                    "message_id": message.message_id,
                    "session_id": message.session_id,
                    "role": message.role,
                    "author_subject_id": message.author_subject_id,
                    "content": message.content,
                    "observed_at": message.observed_at.isoformat(),
                    "metadata": message.metadata,
                }
            ),
            subject_id=message.author_subject_id,
            locus_key=None,
        )

    def _observation_node(self, observation: Observation) -> CompilerNode:
        return CompilerNode(
            node_id=f"observation:{observation.observation_id}",
            category=CompilerNodeCategory.SOURCE_INPUT,
            kind=SourceInputKind.OBSERVATION,
            fingerprint=_hash_fingerprint(
                {
                    "observation_id": observation.observation_id,
                    "session_id": observation.session_id,
                    "author_subject_id": observation.author_subject_id,
                    "source_kind": observation.source_kind,
                    "content": observation.content,
                    "observed_at": observation.observed_at.isoformat(),
                    "metadata": observation.metadata,
                }
            ),
            subject_id=observation.author_subject_id,
            locus_key=observation.metadata.get("locus_key"),
        )

    def _claim_node(self, claim: Claim) -> CompilerNode:
        return CompilerNode(
            node_id=f"claim:{claim.claim_id}",
            category=CompilerNodeCategory.DERIVED_IR,
            kind=DerivedArtifactKind.CLAIM,
            fingerprint=_hash_fingerprint(
                {
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "subject_id": claim.subject_id,
                    "locus_key": claim.locus.locus_key,
                    "value": claim.value,
                    "observed_at": claim.observed_at.isoformat(),
                    "learned_at": claim.learned_at.isoformat(),
                    "valid_from": None if claim.valid_from is None else claim.valid_from.isoformat(),
                    "valid_to": None if claim.valid_to is None else claim.valid_to.isoformat(),
                    "relations": [
                        {
                            "kind": relation.kind.value,
                            "related_claim_id": relation.related_claim_id,
                        }
                        for relation in claim.relations
                    ],
                }
            ),
            subject_id=claim.subject_id,
            locus_key=claim.locus.locus_key,
        )

    def _belief_node(self, belief: StoredBeliefState) -> CompilerNode:
        return CompilerNode(
            node_id=f"locus:{belief.subject_id}:{belief.locus_key}",
            category=CompilerNodeCategory.DERIVED_IR,
            kind=DerivedArtifactKind.LOCUS,
            fingerprint=_hash_fingerprint(
                {
                    "belief_id": belief.belief_id,
                    "policy_stamp": belief.policy_stamp,
                    "subject_id": belief.subject_id,
                    "locus_key": belief.locus_key,
                    "active_claim_ids": list(belief.projection.active_claim_ids),
                    "historical_claim_ids": list(belief.projection.historical_claim_ids),
                    "epistemic_status": belief.projection.epistemic.status.value,
                    "as_of": belief.as_of.isoformat(),
                }
            ),
            subject_id=belief.subject_id,
            locus_key=belief.locus_key,
        )

    def _compiled_view_node(self, stored_view: StoredCompiledView) -> CompilerNode:
        return CompilerNode(
            node_id=f"view:{stored_view.compiled_view_id}",
            category=CompilerNodeCategory.COMPILED_ARTIFACT,
            kind=_VIEW_KIND_TO_COMPILER_KIND[stored_view.view.kind],
            fingerprint=_hash_fingerprint(
                {
                    "compiled_view_id": stored_view.compiled_view_id,
                    "kind": stored_view.view.kind.value,
                    "view_key": stored_view.view.view_key,
                    "policy_stamp": stored_view.view.policy_stamp,
                    "snapshot_id": stored_view.view.snapshot_id,
                    "claim_ids": list(stored_view.view.claim_ids),
                    "observation_ids": list(stored_view.view.observation_ids),
                    "payload": stored_view.payload,
                    "created_at": stored_view.created_at.isoformat(),
                }
            ),
            subject_id=_clean_optional_text(self._compiled_view_subject_id(stored_view), field_name="subject_id"),
            locus_key=_clean_optional_text(self._compiled_view_locus_key(stored_view), field_name="locus_key"),
        )

    def _compiled_view_subject_id(self, stored_view: StoredCompiledView) -> str | None:
        repository = SQLiteRepository(self._connection)
        if not stored_view.view.claim_ids:
            return None
        first_claim = repository.read_claim(stored_view.view.claim_ids[0])
        if first_claim is None:
            return None
        return first_claim.subject_id

    def _compiled_view_locus_key(self, stored_view: StoredCompiledView) -> str | None:
        repository = SQLiteRepository(self._connection)
        if not stored_view.view.claim_ids:
            return None
        first_claim = repository.read_claim(stored_view.view.claim_ids[0])
        if first_claim is None:
            return None
        return first_claim.locus.locus_key

    def _claim_text(self, claim: Claim) -> str:
        return " ".join(
            (
                "claim",
                claim.claim_type,
                claim.subject_id,
                claim.locus.locus_key,
                _json_dumps(claim.value),
            )
        )


__all__ = [
    "IndexRebuildResult",
    "IndexSearchHit",
    "IndexSourceKind",
    "InMemoryZvecBackend",
    "VectorIndexRecord",
    "ZvecBackend",
    "ZvecIndex",
]
