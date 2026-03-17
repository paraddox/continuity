"""Legacy history and memory-file import helpers for Continuity."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuity.admission import (
    AdmissionAssessment,
    AdmissionDecisionTrace,
    AdmissionStrength,
    AdmissionThresholds,
    AdmissionWriteBudget,
)
from continuity.forgetting import ForgettingSurface, ForgettingTarget, ForgettingTargetKind
from continuity.ontology import MemoryPartition
from continuity.policy import get_policy_pack
from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimScope,
    MemoryLocus,
    Observation,
    Subject,
    SubjectAlias,
    SubjectKind,
)
from continuity.store.sqlite import (
    ImportRunRecord,
    MigrationArtifactRecord,
    SQLiteRepository,
    SessionMessageRecord,
    SessionRecord,
    StoredDisclosurePolicy,
)


_BRACKETED_TRANSCRIPT_LINE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+(?P<role>[a-zA-Z_]+):\s*(?P<content>.*)$"
)
_PLAIN_TRANSCRIPT_LINE = re.compile(
    r"^(?P<timestamp>\S+)\s+(?P<role>[a-zA-Z_]+):\s*(?P<content>.*)$"
)
_CONTEXT_BLOCK = re.compile(r"<context>.*?</context>", re.IGNORECASE | re.DOTALL)


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_timestamp(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _normalize_timestamp(raw: str) -> datetime:
    text = _clean_text(raw, field_name="timestamp")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _slug(value: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    )
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or "seed"


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_hexdigest(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            payload = part
        else:
            payload = str(part).encode("utf-8")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{_hash_hexdigest(*parts)[:24]}"


def _entry_source_ref(entry: "MigrationEntry") -> str:
    original_file = entry.metadata.get("original_file")
    if isinstance(original_file, str) and original_file.strip():
        return original_file.strip()
    return entry.upload_name


def _entry_source_kind(source_kind: str, entry: "MigrationEntry") -> str:
    source = entry.metadata.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return source_kind


def _strip_memory_wrapper(content: str) -> str:
    text = content.strip()
    text = re.sub(r"</?prior_memory_file>", "", text, flags=re.IGNORECASE)
    text = _CONTEXT_BLOCK.sub("", text)
    return text.strip()


def _iter_transcript_messages(content: str) -> tuple[tuple[datetime, str, str], ...]:
    messages: list[tuple[datetime, str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<") or line.startswith("</"):
            continue
        match = _BRACKETED_TRANSCRIPT_LINE.match(line) or _PLAIN_TRANSCRIPT_LINE.match(line)
        if match is None:
            continue
        messages.append(
            (
                _normalize_timestamp(match.group("timestamp")),
                match.group("role").strip().lower(),
                match.group("content").strip(),
            )
        )
    return tuple(messages)


@dataclass(frozen=True, slots=True)
class MigrationEntry:
    upload_name: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "upload_name", _clean_text(self.upload_name, field_name="upload_name"))
        object.__setattr__(self, "content", _clean_text(self.content, field_name="content"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class MigrationArtifactIdentity:
    artifact_id: str
    content_fingerprint: str
    source_kind: str
    source_ref: str


@dataclass(frozen=True, slots=True)
class LegacyImportResult:
    import_run_id: str
    artifact_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    blocked_artifact_ids: tuple[str, ...]
    seeded_subject_ids: tuple[str, ...]


def artifact_identity_for_entry(
    *,
    session_id: str,
    source_kind: str,
    entry: MigrationEntry,
) -> MigrationArtifactIdentity:
    cleaned_session_id = _clean_text(session_id, field_name="session_id")
    cleaned_source_kind = _clean_text(source_kind, field_name="source_kind")
    source_ref = _entry_source_ref(entry)
    artifact_source_kind = _entry_source_kind(cleaned_source_kind, entry)
    fingerprint = "sha256:" + _hash_hexdigest(
        artifact_source_kind,
        source_ref,
        entry.upload_name,
        entry.content.strip(),
        _json_dumps(entry.metadata),
    )
    artifact_id = _hash_id(
        "artifact",
        cleaned_session_id,
        cleaned_source_kind,
        source_ref,
        fingerprint,
    )
    return MigrationArtifactIdentity(
        artifact_id=artifact_id,
        content_fingerprint=fingerprint,
        source_kind=artifact_source_kind,
        source_ref=source_ref,
    )


def import_legacy_history(
    repository: SQLiteRepository,
    *,
    session_id: str,
    source_kind: str,
    entries: tuple[MigrationEntry, ...],
    imported_at: datetime,
    host_namespace: str = "hermes",
    session_name: str | None = None,
    policy_name: str = "hermes_v1",
    user_subject_id: str = "subject:user:self",
    assistant_subject_id: str = "subject:assistant:hermes",
    user_canonical_name: str = "User",
    assistant_canonical_name: str = "Hermes",
) -> LegacyImportResult:
    cleaned_session_id = _clean_text(session_id, field_name="session_id")
    cleaned_source_kind = _clean_text(source_kind, field_name="source_kind")
    imported = _validate_timestamp(imported_at, field_name="imported_at")
    if not entries:
        raise ValueError("entries must be non-empty")

    policy = get_policy_pack(policy_name)
    identities = tuple(
        artifact_identity_for_entry(
            session_id=cleaned_session_id,
            source_kind=cleaned_source_kind,
            entry=entry,
        )
        for entry in entries
    )
    import_run_id = _hash_id(
        "import",
        cleaned_session_id,
        cleaned_source_kind,
        *(f"{identity.artifact_id}:{identity.content_fingerprint}" for identity in identities),
    )

    _ensure_default_disclosure_policies(repository)
    _ensure_session(
        repository,
        session_id=cleaned_session_id,
        host_namespace=host_namespace,
        session_name=session_name,
        imported_at=imported,
    )

    repository.save_import_run(
        ImportRunRecord(
            import_run_id=import_run_id,
            source_kind=cleaned_source_kind,
            source_ref=cleaned_session_id,
            policy_stamp=policy.policy_stamp,
            started_at=imported,
            finished_at=imported,
            metadata={"entry_count": len(entries)},
        )
    )

    artifact_ids: list[str] = []
    observation_ids: list[str] = []
    candidate_ids: list[str] = []
    claim_ids: list[str] = []
    blocked_artifact_ids: list[str] = []
    seeded_subject_ids: list[str] = []

    _ensure_subject(repository, subject_id=user_subject_id, kind=SubjectKind.USER, canonical_name=user_canonical_name, recorded_at=imported)
    _ensure_subject(
        repository,
        subject_id=assistant_subject_id,
        kind=SubjectKind.ASSISTANT,
        canonical_name=assistant_canonical_name,
        recorded_at=imported,
    )
    seeded_subject_ids.extend((user_subject_id, assistant_subject_id))

    for entry, identity in zip(entries, identities, strict=True):
        target = ForgettingTarget(
            target_kind=ForgettingTargetKind.IMPORTED_ARTIFACT,
            target_id=identity.artifact_id,
        )
        if _artifact_is_blocked(repository, target=target, content_fingerprint=identity.content_fingerprint):
            blocked_artifact_ids.append(identity.artifact_id)
            continue

        artifact_ids.append(identity.artifact_id)
        repository.save_migration_artifact(
            MigrationArtifactRecord(
                artifact_id=identity.artifact_id,
                import_run_id=import_run_id,
                source_kind=identity.source_kind,
                source_ref=identity.source_ref,
                subject_id=_artifact_subject_id(
                    entry,
                    user_subject_id=user_subject_id,
                    assistant_subject_id=assistant_subject_id,
                ),
                imported_at=imported,
                content_fingerprint=identity.content_fingerprint,
                metadata={"upload_name": entry.upload_name, **entry.metadata},
            )
        )

        if identity.source_kind == "local_jsonl":
            seeded = _import_transcript_entry(
                repository,
                entry=entry,
                identity=identity,
                session_id=cleaned_session_id,
                imported_at=imported,
                import_run_id=import_run_id,
                user_subject_id=user_subject_id,
                assistant_subject_id=assistant_subject_id,
                user_canonical_name=user_canonical_name,
                assistant_canonical_name=assistant_canonical_name,
            )
            observation_ids.extend(seeded.observation_ids)
            seeded_subject_ids.extend(seeded.seeded_subject_ids)
            continue

        seeded = _import_memory_file_entry(
            repository,
            entry=entry,
            identity=identity,
            session_id=cleaned_session_id,
            imported_at=imported,
            import_run_id=import_run_id,
            policy_stamp=policy.policy_stamp,
            user_subject_id=user_subject_id,
            assistant_subject_id=assistant_subject_id,
            user_canonical_name=user_canonical_name,
            assistant_canonical_name=assistant_canonical_name,
            write_budget_for_partition=policy.write_budget_for_partition,
        )
        observation_ids.extend(seeded.observation_ids)
        candidate_ids.extend(seeded.candidate_ids)
        claim_ids.extend(seeded.claim_ids)
        seeded_subject_ids.extend(seeded.seeded_subject_ids)

    return LegacyImportResult(
        import_run_id=import_run_id,
        artifact_ids=tuple(dict.fromkeys(artifact_ids)),
        observation_ids=tuple(dict.fromkeys(observation_ids)),
        candidate_ids=tuple(dict.fromkeys(candidate_ids)),
        claim_ids=tuple(dict.fromkeys(claim_ids)),
        blocked_artifact_ids=tuple(dict.fromkeys(blocked_artifact_ids)),
        seeded_subject_ids=tuple(dict.fromkeys(seeded_subject_ids)),
    )


@dataclass(frozen=True, slots=True)
class _EntryImportSummary:
    observation_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    seeded_subject_ids: tuple[str, ...] = ()


def _artifact_subject_id(
    entry: MigrationEntry,
    *,
    user_subject_id: str,
    assistant_subject_id: str,
) -> str | None:
    target_peer = str(entry.metadata.get("target_peer", "")).strip().lower()
    if target_peer == "user":
        return user_subject_id
    if target_peer == "ai":
        return assistant_subject_id
    if entry.metadata.get("source") == "local_jsonl":
        return user_subject_id
    return None


def _artifact_is_blocked(
    repository: SQLiteRepository,
    *,
    target: ForgettingTarget,
    content_fingerprint: str,
) -> bool:
    guard_surfaces = repository.forgetting.resurrection_guard_surfaces(target)
    if ForgettingSurface.IMPORT_PIPELINE not in guard_surfaces:
        return False
    tombstones = repository.forgetting.list_tombstones(
        target=target,
        surface=ForgettingSurface.IMPORT_PIPELINE,
    )
    return any(tombstone.content_fingerprint == content_fingerprint for tombstone in tombstones)


def _ensure_default_disclosure_policies(repository: SQLiteRepository) -> None:
    for policy_id, principal, channel, purpose in (
        ("assistant_internal", "assistant_internal", "prompt", "prompt"),
        ("current_user", "current_user", "search", "search"),
        ("current_peer", "current_peer", "search", "search"),
        ("shared_session", "shared_session", "prompt", "prompt"),
        ("host_internal", "assistant_internal", "prompt", "prompt"),
    ):
        if repository.read_disclosure_policy(policy_id) is not None:
            continue
        repository.save_disclosure_policy(
            StoredDisclosurePolicy(
                policy_id=policy_id,
                audience_principal=principal,
                channel=channel,
                purpose=purpose,
                exposure_mode="direct",
                redaction_mode="none",
                capture_for_replay=True,
            )
        )


def _ensure_session(
    repository: SQLiteRepository,
    *,
    session_id: str,
    host_namespace: str,
    session_name: str | None,
    imported_at: datetime,
) -> None:
    if repository.read_session(session_id) is not None:
        return
    repository.save_session(
        SessionRecord(
            session_id=session_id,
            host_namespace=_clean_text(host_namespace, field_name="host_namespace"),
            session_name=_clean_text(session_name or session_id, field_name="session_name"),
            recall_mode="hybrid",
            write_frequency="turn",
            created_at=imported_at,
            metadata={"source": "legacy_import"},
        )
    )


def _ensure_subject(
    repository: SQLiteRepository,
    *,
    subject_id: str,
    kind: SubjectKind,
    canonical_name: str,
    recorded_at: datetime,
) -> None:
    if repository.read_subject(subject_id) is not None:
        return
    repository.save_subject(
        Subject(
            subject_id=subject_id,
            kind=kind,
            canonical_name=_clean_text(canonical_name, field_name="canonical_name"),
        ),
        created_at=recorded_at,
    )


def _append_aliases(
    repository: SQLiteRepository,
    *,
    subject_id: str,
    aliases: tuple[str, ...],
    alias_type: str,
    source_observation_id: str,
    recorded_at: datetime,
) -> None:
    existing = repository.read_subject(subject_id)
    if existing is None:
        raise ValueError(f"unknown subject for alias seeding: {subject_id}")

    alias_map = {alias.normalized_alias: alias for alias in existing.aliases}
    for alias in aliases:
        cleaned = alias.strip()
        if not cleaned:
            continue
        normalized = " ".join(cleaned.split()).casefold()
        if normalized in alias_map:
            continue
        alias_map[normalized] = SubjectAlias(
            alias=cleaned,
            alias_type=alias_type,
            source_observation_ids=(source_observation_id,),
        )

    repository.save_subject(
        Subject(
            subject_id=existing.subject_id,
            kind=existing.kind,
            canonical_name=existing.canonical_name,
            aliases=tuple(alias_map.values()),
            merges=existing.merges,
            splits=existing.splits,
        ),
        created_at=recorded_at,
    )


def _import_transcript_entry(
    repository: SQLiteRepository,
    *,
    entry: MigrationEntry,
    identity: MigrationArtifactIdentity,
    session_id: str,
    imported_at: datetime,
    import_run_id: str,
    user_subject_id: str,
    assistant_subject_id: str,
    user_canonical_name: str,
    assistant_canonical_name: str,
) -> _EntryImportSummary:
    observations: list[str] = []
    seeded_subjects: list[str] = []
    messages = _iter_transcript_messages(entry.content)

    for index, (observed_at, role, content) in enumerate(messages):
        subject_id, subject_kind, canonical_name, aliases = _subject_for_role(
            role,
            user_subject_id=user_subject_id,
            assistant_subject_id=assistant_subject_id,
            user_canonical_name=user_canonical_name,
            assistant_canonical_name=assistant_canonical_name,
        )
        _ensure_subject(
            repository,
            subject_id=subject_id,
            kind=subject_kind,
            canonical_name=canonical_name,
            recorded_at=imported_at,
        )
        seeded_subjects.append(subject_id)
        observation_id = _hash_id("observation", identity.artifact_id, index)
        message_id = _hash_id("message", identity.artifact_id, index)

        repository.save_message(
            SessionMessageRecord(
                message_id=message_id,
                session_id=session_id,
                role="assistant" if subject_kind is SubjectKind.ASSISTANT else "user",
                author_subject_id=subject_id,
                content=content,
                observed_at=observed_at,
                metadata={
                    "import_run_id": import_run_id,
                    "artifact_id": identity.artifact_id,
                    "source_ref": identity.source_ref,
                    "upload_name": entry.upload_name,
                },
            )
        )
        repository.save_observation(
            Observation(
                observation_id=observation_id,
                source_kind="history_import",
                session_id=session_id,
                author_subject_id=subject_id,
                content=content,
                observed_at=observed_at,
                metadata={
                    "import_run_id": import_run_id,
                    "artifact_id": identity.artifact_id,
                    "source_ref": identity.source_ref,
                    "upload_name": entry.upload_name,
                    "line_index": index,
                },
            ),
            message_id=message_id,
        )
        _append_aliases(
            repository,
            subject_id=subject_id,
            aliases=aliases,
            alias_type="legacy_role",
            source_observation_id=observation_id,
            recorded_at=imported_at,
        )
        observations.append(observation_id)

    return _EntryImportSummary(
        observation_ids=tuple(observations),
        seeded_subject_ids=tuple(seeded_subjects),
    )


def _subject_for_role(
    role: str,
    *,
    user_subject_id: str,
    assistant_subject_id: str,
    user_canonical_name: str,
    assistant_canonical_name: str,
) -> tuple[str, SubjectKind, str, tuple[str, ...]]:
    cleaned_role = _clean_text(role, field_name="role").lower()
    if cleaned_role == "user":
        return user_subject_id, SubjectKind.USER, user_canonical_name, ("user",)
    if cleaned_role == "assistant":
        return assistant_subject_id, SubjectKind.ASSISTANT, assistant_canonical_name, ("assistant",)
    return f"subject:peer:{_slug(cleaned_role)}", SubjectKind.PEER, cleaned_role.title(), (cleaned_role,)


def _import_memory_file_entry(
    repository: SQLiteRepository,
    *,
    entry: MigrationEntry,
    identity: MigrationArtifactIdentity,
    session_id: str,
    imported_at: datetime,
    import_run_id: str,
    policy_stamp: str,
    user_subject_id: str,
    assistant_subject_id: str,
    user_canonical_name: str,
    assistant_canonical_name: str,
    write_budget_for_partition: Any,
) -> _EntryImportSummary:
    target_peer = str(entry.metadata.get("target_peer", "")).strip().lower()
    if target_peer not in {"user", "ai"}:
        raise ValueError("memory file imports require metadata.target_peer of 'user' or 'ai'")

    source_ref = identity.source_ref
    body = _strip_memory_wrapper(entry.content)
    observation_id = _hash_id("observation", identity.artifact_id, "file")
    source_file = Path(source_ref).name

    if target_peer == "user":
        subject_id = user_subject_id
        subject_kind = SubjectKind.USER
        canonical_name = user_canonical_name
        aliases = ("user", "self")
        claim_type = "biography"
        scope = ClaimScope.USER
        locus_suffix = {
            "MEMORY.md": "consolidated_memory",
            "USER.md": "user_profile",
        }.get(source_file, _slug(Path(source_file).stem))
        locus_key = f"biography/{locus_suffix}"
    else:
        subject_id = assistant_subject_id
        subject_kind = SubjectKind.ASSISTANT
        canonical_name = assistant_canonical_name
        aliases = ("assistant", "ai")
        claim_type = "assistant_self_model"
        scope = ClaimScope.ASSISTANT
        locus_key = f"self/{_slug(Path(source_file).stem)}"

    _ensure_subject(
        repository,
        subject_id=subject_id,
        kind=subject_kind,
        canonical_name=canonical_name,
        recorded_at=imported_at,
    )
    repository.save_observation(
        Observation(
            observation_id=observation_id,
            source_kind="memory_file_import",
            session_id=session_id,
            author_subject_id=subject_id,
            content=body,
            observed_at=imported_at,
            metadata={
                "import_run_id": import_run_id,
                "artifact_id": identity.artifact_id,
                "source_ref": source_ref,
                "upload_name": entry.upload_name,
                "target_peer": target_peer,
            },
        )
    )
    _append_aliases(
        repository,
        subject_id=subject_id,
        aliases=aliases,
        alias_type="legacy_seed",
        source_observation_id=observation_id,
        recorded_at=imported_at,
    )

    candidate = CandidateMemory(
        candidate_id=_hash_id("candidate", identity.artifact_id, claim_type, locus_key),
        claim_type=claim_type,
        subject_id=subject_id,
        scope=scope,
        value={"source_file": source_file, "content": body},
        source_observation_ids=(observation_id,),
    )
    if repository.read_candidate_memory(candidate.candidate_id) is None:
        repository.save_candidate_memory(candidate, created_at=imported_at)

    trace = repository.admissions.read_decision(candidate.candidate_id)
    if trace is None:
        trace = _build_admission_trace(
            candidate=candidate,
            claim_type=claim_type,
            policy_stamp=policy_stamp,
            source_ref=source_ref,
            budget_limit=int(write_budget_for_partition(_partition_for_claim_type(claim_type))),
            repository=repository,
            imported_at=imported_at,
        )
        repository.admissions.record_decision(trace)

    claim_ids: tuple[str, ...] = ()
    if trace.publishes_claim:
        claim = Claim.from_candidate(
            claim_id=_hash_id("claim", candidate.candidate_id, locus_key),
            candidate=candidate,
            admission=trace.decision,
            locus=MemoryLocus(
                subject_id=subject_id,
                locus_key=locus_key,
                scope=scope,
                default_disclosure_policy=_default_disclosure_policy_for_claim_type(claim_type),
                conflict_set_key=f"{claim_type}:{locus_key}",
                aggregation_mode=_aggregation_mode_for_claim_type(claim_type),
            ),
            provenance=ClaimProvenance(observation_ids=(observation_id,)),
            disclosure_policy=_default_disclosure_policy_for_claim_type(claim_type),
            observed_at=imported_at,
            learned_at=imported_at,
        )
        if repository.read_claim(claim.claim_id) is None:
            repository.save_claim(claim)
        claim_ids = (claim.claim_id,)

    return _EntryImportSummary(
        observation_ids=(observation_id,),
        candidate_ids=(candidate.candidate_id,),
        claim_ids=claim_ids,
        seeded_subject_ids=(subject_id,),
    )


def _build_admission_trace(
    *,
    candidate: CandidateMemory,
    claim_type: str,
    policy_stamp: str,
    source_ref: str,
    budget_limit: int,
    repository: SQLiteRepository,
    imported_at: datetime,
) -> AdmissionDecisionTrace:
    partition = _partition_for_claim_type(claim_type)
    budget_window_key = f"import:{candidate.subject_id}:{partition.value}"
    stored_budget = repository.admissions.read_budget(
        partition=partition,
        window_key=budget_window_key,
    )
    budget = AdmissionWriteBudget(
        partition=partition,
        window_key=budget_window_key,
        limit=budget_limit,
        used=0 if stored_budget is None else stored_budget.used,
    )
    durable = claim_type != "assistant_self_model" and budget.allows_durable_promotion()
    outcome = AdmissionOutcome.DURABLE_CLAIM if durable else AdmissionOutcome.SESSION_EPHEMERAL
    return AdmissionDecisionTrace(
        decision=AdmissionDecision(
            candidate_id=candidate.candidate_id,
            outcome=outcome,
            recorded_at=imported_at,
            rationale=f"imported from {source_ref}",
        ),
        claim_type=claim_type,
        policy_stamp=policy_stamp,
        assessment=AdmissionAssessment(
            claim_type=claim_type,
            evidence=AdmissionStrength.HIGH,
            novelty=AdmissionStrength.HIGH,
            stability=AdmissionStrength.HIGH if durable else AdmissionStrength.LOW,
            salience=AdmissionStrength.MEDIUM,
            rationale=f"host import seeded from {source_ref}",
            utility_signals=("prompt_inclusion",),
        ),
        thresholds=AdmissionThresholds(
            evidence=AdmissionStrength.MEDIUM,
            novelty=AdmissionStrength.MEDIUM,
            stability=AdmissionStrength.MEDIUM,
            salience=AdmissionStrength.LOW,
        ),
        budget=budget,
    )


def _partition_for_claim_type(claim_type: str) -> MemoryPartition:
    if claim_type == "assistant_self_model":
        return MemoryPartition.ASSISTANT_MEMORY
    return MemoryPartition.USER_MEMORY


def _default_disclosure_policy_for_claim_type(claim_type: str) -> str:
    if claim_type == "assistant_self_model":
        return "assistant_internal"
    return "current_peer"


def _aggregation_mode_for_claim_type(claim_type: str) -> AggregationMode:
    if claim_type == "assistant_self_model":
        return AggregationMode.LATEST_WINS
    return AggregationMode.SET_UNION
