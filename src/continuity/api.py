"""Deployment boundary for the Continuity host service contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TypeVar

from continuity.disclosure import DisclosureContext
from continuity.forgetting import ForgettingMode, ForgettingTargetKind
from continuity.outcomes import OutcomeLabel, OutcomeTarget
from continuity.resolution_queue import ResolutionAction
from continuity.service import (
    ContinuityServiceFacade,
    SERVICE_CONTRACT_VERSION,
    ServiceOperation,
    ServiceRequest,
    ServiceResponse,
)
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    write_frequency_policy_for,
)
from continuity.views import ViewKind


EnumT = TypeVar("EnumT", bound=StrEnum)


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _optional_clean_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _dedupe_cleaned(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_clean_text(value, field_name=field_name) for value in values))


def _dedupe_enum_tuple(values: tuple[EnumT, ...]) -> tuple[EnumT, ...]:
    return tuple(dict.fromkeys(values))


def _clean_optional_limit(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _payload_fields(**fields: object) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, tuple) and not value:
            continue
        payload[key] = value
    return payload


class DeploymentMode(StrEnum):
    EMBEDDED = "embedded"
    DAEMON = "daemon"


class TransportAdapter(StrEnum):
    IN_PROCESS = "in_process"
    UNIX_DOMAIN_SOCKET = "unix_domain_socket"


@dataclass(frozen=True, slots=True)
class SqliteOwnership:
    owner_role: str
    one_owner_process_per_store: bool
    serialized_commit_lane: bool
    in_process_worker_threads_only: bool
    multi_process_write_coordination: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_role", _clean_text(self.owner_role, field_name="owner_role"))
        if not self.one_owner_process_per_store:
            raise ValueError("v1 ownership contract requires one owner process per store")
        if not self.serialized_commit_lane:
            raise ValueError("v1 ownership contract requires a serialized commit lane")
        if self.multi_process_write_coordination:
            raise ValueError("v1 must not require multi-process write coordination")


@dataclass(frozen=True, slots=True)
class DeploymentBoundary:
    mode: DeploymentMode
    contract_version: str
    transport_adapter: TransportAdapter
    local_only: bool
    hosted_service_assumptions: bool
    sqlite_ownership: SqliteOwnership
    shared_transaction_kinds: tuple[TransactionKind, ...]
    shared_durability_waterlines: tuple[DurabilityWaterline, ...]
    shared_semantics: tuple[str, ...]
    engine_responsibilities: tuple[str, ...]
    shell_responsibilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_version",
            _clean_text(self.contract_version, field_name="contract_version"),
        )
        object.__setattr__(
            self,
            "shared_transaction_kinds",
            _dedupe_enum_tuple(self.shared_transaction_kinds),
        )
        object.__setattr__(
            self,
            "shared_durability_waterlines",
            _dedupe_enum_tuple(self.shared_durability_waterlines),
        )
        object.__setattr__(
            self,
            "shared_semantics",
            _dedupe_cleaned(self.shared_semantics, field_name="shared_semantics"),
        )
        object.__setattr__(
            self,
            "engine_responsibilities",
            _dedupe_cleaned(
                self.engine_responsibilities,
                field_name="engine_responsibilities",
            ),
        )
        object.__setattr__(
            self,
            "shell_responsibilities",
            _dedupe_cleaned(
                self.shell_responsibilities,
                field_name="shell_responsibilities",
            ),
        )

        if not self.local_only:
            raise ValueError("deployment modes must remain local-only")
        if self.hosted_service_assumptions:
            raise ValueError("hosted service assumptions are out of scope for v1")
        if not self.shared_transaction_kinds:
            raise ValueError("shared_transaction_kinds must be non-empty")
        if not self.shared_durability_waterlines:
            raise ValueError("shared_durability_waterlines must be non-empty")
        if not self.shared_semantics:
            raise ValueError("shared_semantics must be non-empty")
        if not self.engine_responsibilities:
            raise ValueError("engine_responsibilities must be non-empty")
        if not self.shell_responsibilities:
            raise ValueError("shell_responsibilities must be non-empty")

        if self.mode is DeploymentMode.EMBEDDED and self.transport_adapter is not TransportAdapter.IN_PROCESS:
            raise ValueError("embedded mode must use the in-process transport adapter")
        if self.mode is DeploymentMode.DAEMON and self.transport_adapter is not TransportAdapter.UNIX_DOMAIN_SOCKET:
            raise ValueError("daemon mode must use Unix domain sockets")


@lru_cache(maxsize=1)
def deployment_boundaries() -> dict[DeploymentMode, DeploymentBoundary]:
    shared_transaction_kinds = tuple(TransactionKind)
    shared_durability_waterlines = tuple(DurabilityWaterline)
    shared_semantics = (
        "transaction_entrypoints",
        "durability_waterlines",
        "snapshot_consistency",
        "replay_artifacts",
        "disclosure_decisions",
    )
    engine_responsibilities = (
        "execute_canonical_service_contract",
        "enforce_transaction_pipeline",
        "publish_snapshot_consistent_reads",
        "capture_replay_artifacts",
    )

    return {
        DeploymentMode.EMBEDDED: DeploymentBoundary(
            mode=DeploymentMode.EMBEDDED,
            contract_version=SERVICE_CONTRACT_VERSION,
            transport_adapter=TransportAdapter.IN_PROCESS,
            local_only=True,
            hosted_service_assumptions=False,
            sqlite_ownership=SqliteOwnership(
                owner_role="hermes_process",
                one_owner_process_per_store=True,
                serialized_commit_lane=True,
                in_process_worker_threads_only=True,
                multi_process_write_coordination=False,
            ),
            shared_transaction_kinds=shared_transaction_kinds,
            shared_durability_waterlines=shared_durability_waterlines,
            shared_semantics=shared_semantics,
            engine_responsibilities=engine_responsibilities,
            shell_responsibilities=(
                "hermes_managed_lifecycle",
                "direct_in_process_calls",
                "process_local_threads",
            ),
        ),
        DeploymentMode.DAEMON: DeploymentBoundary(
            mode=DeploymentMode.DAEMON,
            contract_version=SERVICE_CONTRACT_VERSION,
            transport_adapter=TransportAdapter.UNIX_DOMAIN_SOCKET,
            local_only=True,
            hosted_service_assumptions=False,
            sqlite_ownership=SqliteOwnership(
                owner_role="daemon_process",
                one_owner_process_per_store=True,
                serialized_commit_lane=True,
                in_process_worker_threads_only=True,
                multi_process_write_coordination=False,
            ),
            shared_transaction_kinds=shared_transaction_kinds,
            shared_durability_waterlines=shared_durability_waterlines,
            shared_semantics=shared_semantics,
            engine_responsibilities=engine_responsibilities,
            shell_responsibilities=(
                "daemon_managed_lifecycle",
                "unix_domain_sockets",
                "local_only_transport",
            ),
        ),
    }


def deployment_boundary_for(mode: DeploymentMode) -> DeploymentBoundary:
    return deployment_boundaries()[mode]


def _copy_payload_mapping(
    value: Mapping[str, object] | None,
    *,
    field_name: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    copied = dict(value)
    if not copied:
        raise ValueError(f"{field_name} must be non-empty when provided")
    return copied


def _save_turn_waterline(write_frequency: str | int | None) -> DurabilityWaterline | None:
    if write_frequency is None:
        return None
    return write_frequency_policy_for(write_frequency).awaited_waterline


def _forget_waterline(mode: ForgettingMode) -> DurabilityWaterline:
    if mode is ForgettingMode.SUPERSEDE:
        return DurabilityWaterline.VIEWS_COMPILED
    return DurabilityWaterline.SNAPSHOT_PUBLISHED


class ContinuityMutationApi:
    """Typed control and mutating surface over the transport-neutral facade."""

    def __init__(self, facade: ContinuityServiceFacade) -> None:
        self._facade = facade

    def initialize(
        self,
        *,
        request_id: str,
        host_namespace: str,
        session_id: str | None = None,
        session_name: str | None = None,
        recall_mode: str | None = None,
        write_frequency: str | int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            host_namespace=_clean_text(host_namespace, field_name="host_namespace"),
            session_id=_optional_clean_text(session_id, field_name="session_id"),
            session_name=_optional_clean_text(session_name, field_name="session_name"),
            recall_mode=_optional_clean_text(recall_mode, field_name="recall_mode"),
            write_frequency=write_frequency,
            metadata=_copy_payload_mapping(metadata, field_name="metadata"),
        )
        return self._execute(
            operation=ServiceOperation.INITIALIZE,
            request_id=request_id,
            payload=payload,
        )

    def save_turn(
        self,
        *,
        request_id: str,
        session_id: str,
        turn_id: str,
        messages: tuple[Mapping[str, object], ...],
        write_frequency: str | int | None = None,
        metadata: Mapping[str, object] | None = None,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            session_id=_clean_text(session_id, field_name="session_id"),
            turn_id=_clean_text(turn_id, field_name="turn_id"),
            messages=messages,
            write_frequency=write_frequency,
            metadata=_copy_payload_mapping(metadata, field_name="metadata"),
        )
        return self._execute(
            operation=ServiceOperation.SAVE_TURN,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline or _save_turn_waterline(write_frequency),
        )

    def write_conclusion(
        self,
        *,
        request_id: str,
        session_id: str,
        subject_id: str,
        locus_key: str,
        conclusion: str,
        metadata: Mapping[str, object] | None = None,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            session_id=_clean_text(session_id, field_name="session_id"),
            subject_id=_clean_text(subject_id, field_name="subject_id"),
            locus_key=_clean_text(locus_key, field_name="locus_key"),
            conclusion=_clean_text(conclusion, field_name="conclusion"),
            metadata=_copy_payload_mapping(metadata, field_name="metadata"),
        )
        return self._execute(
            operation=ServiceOperation.WRITE_CONCLUSION,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline,
        )

    def forget_memory(
        self,
        *,
        request_id: str,
        target_id: str,
        target_kind: ForgettingTargetKind,
        mode: ForgettingMode,
        requested_by: str,
        rationale: str,
        policy_stamp: str,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            target_id=_clean_text(target_id, field_name="target_id"),
            target_kind=target_kind,
            mode=mode,
            requested_by=_clean_text(requested_by, field_name="requested_by"),
            rationale=_clean_text(rationale, field_name="rationale"),
            policy_stamp=_clean_text(policy_stamp, field_name="policy_stamp"),
        )
        return self._execute(
            operation=ServiceOperation.FORGET_MEMORY,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline or _forget_waterline(mode),
        )

    def resolve_memory_follow_up(
        self,
        *,
        request_id: str,
        item_id: str,
        action: ResolutionAction,
        rationale: str,
        metadata: Mapping[str, object] | None = None,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            item_id=_clean_text(item_id, field_name="item_id"),
            action=action,
            rationale=_clean_text(rationale, field_name="rationale"),
            metadata=_copy_payload_mapping(metadata, field_name="metadata"),
        )
        return self._execute(
            operation=ServiceOperation.RESOLVE_MEMORY_FOLLOW_UP,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline,
        )

    def import_history(
        self,
        *,
        request_id: str,
        session_id: str,
        source_kind: str,
        entries: tuple[Mapping[str, object], ...],
        metadata: Mapping[str, object] | None = None,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            session_id=_clean_text(session_id, field_name="session_id"),
            source_kind=_clean_text(source_kind, field_name="source_kind"),
            entries=entries,
            metadata=_copy_payload_mapping(metadata, field_name="metadata"),
        )
        return self._execute(
            operation=ServiceOperation.IMPORT_HISTORY,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline,
        )

    def publish_snapshot(
        self,
        *,
        request_id: str,
        snapshot_id: str,
        reason: str | None = None,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            snapshot_id=_clean_text(snapshot_id, field_name="snapshot_id"),
            reason=_optional_clean_text(reason, field_name="reason"),
        )
        return self._execute(
            operation=ServiceOperation.PUBLISH_SNAPSHOT,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline,
        )

    def record_outcome(
        self,
        *,
        request_id: str,
        outcome_label: OutcomeLabel,
        target_kind: OutcomeTarget,
        target_id: str,
        policy_stamp: str,
        rationale: str,
        actor_subject_id: str | None = None,
        claim_ids: tuple[str, ...] = (),
        observation_ids: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            outcome_label=outcome_label,
            target_kind=target_kind,
            target_id=_clean_text(target_id, field_name="target_id"),
            policy_stamp=_clean_text(policy_stamp, field_name="policy_stamp"),
            rationale=_clean_text(rationale, field_name="rationale"),
            actor_subject_id=_optional_clean_text(
                actor_subject_id,
                field_name="actor_subject_id",
            ),
            claim_ids=_dedupe_cleaned(claim_ids, field_name="claim_ids"),
            observation_ids=_dedupe_cleaned(
                observation_ids,
                field_name="observation_ids",
            ),
            metadata=_copy_payload_mapping(metadata, field_name="metadata"),
        )
        return self._execute(
            operation=ServiceOperation.RECORD_OUTCOME,
            request_id=request_id,
            payload=payload,
            minimum_waterline=minimum_waterline,
        )

    def _execute(
        self,
        *,
        operation: ServiceOperation,
        request_id: str,
        payload: Mapping[str, object],
        minimum_waterline: DurabilityWaterline | None = None,
    ) -> ServiceResponse:
        return self._facade.execute(
            ServiceRequest(
                operation=operation,
                request_id=request_id,
                payload=payload,
                minimum_waterline=minimum_waterline,
            )
        )


class ContinuityReadApi:
    """Typed read-side and inspection surface over the transport-neutral facade."""

    def __init__(self, facade: ContinuityServiceFacade) -> None:
        self._facade = facade

    def search(
        self,
        *,
        request_id: str,
        query_text: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
        limit: int | None = None,
        subject_id: str | None = None,
        view_kinds: tuple[ViewKind, ...] = (),
    ) -> ServiceResponse:
        payload = _payload_fields(
            query_text=_clean_text(query_text, field_name="query_text"),
            limit=_clean_optional_limit(limit, field_name="limit"),
            subject_id=_optional_clean_text(subject_id, field_name="subject_id"),
            view_kinds=_dedupe_enum_tuple(view_kinds),
        )
        return self._execute(
            operation=ServiceOperation.SEARCH,
            request_id=request_id,
            payload=payload,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def get_state_view(
        self,
        *,
        request_id: str,
        view_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._read_view(
            operation=ServiceOperation.GET_STATE_VIEW,
            request_id=request_id,
            view_key=view_key,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def get_timeline_view(
        self,
        *,
        request_id: str,
        view_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._read_view(
            operation=ServiceOperation.GET_TIMELINE_VIEW,
            request_id=request_id,
            view_key=view_key,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def get_profile_view(
        self,
        *,
        request_id: str,
        view_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._read_view(
            operation=ServiceOperation.GET_PROFILE_VIEW,
            request_id=request_id,
            view_key=view_key,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def get_prompt_view(
        self,
        *,
        request_id: str,
        view_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._read_view(
            operation=ServiceOperation.GET_PROMPT_VIEW,
            request_id=request_id,
            view_key=view_key,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def answer_memory_question(
        self,
        *,
        request_id: str,
        question: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None = None,
        subject_id: str | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            question=_clean_text(question, field_name="question"),
            subject_id=_optional_clean_text(subject_id, field_name="subject_id"),
        )
        return self._execute(
            operation=ServiceOperation.ANSWER_MEMORY_QUESTION,
            request_id=request_id,
            payload=payload,
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def list_memory_follow_ups(
        self,
        *,
        request_id: str,
        subject_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            subject_id=_optional_clean_text(subject_id, field_name="subject_id"),
            status=_optional_clean_text(status, field_name="status"),
            limit=_clean_optional_limit(limit, field_name="limit"),
        )
        return self._execute(
            operation=ServiceOperation.LIST_MEMORY_FOLLOW_UPS,
            request_id=request_id,
            payload=payload,
        )

    def resolve_subject(
        self,
        *,
        request_id: str,
        reference_text: str,
        subject_kind: str | None = None,
    ) -> ServiceResponse:
        payload = _payload_fields(
            reference_text=_clean_text(reference_text, field_name="reference_text"),
            subject_kind=_optional_clean_text(subject_kind, field_name="subject_kind"),
        )
        return self._execute(
            operation=ServiceOperation.RESOLVE_SUBJECT,
            request_id=request_id,
            payload=payload,
        )

    def inspect_evidence(
        self,
        *,
        request_id: str,
        target_id: str,
        target_kind: str,
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_EVIDENCE,
            request_id=request_id,
            payload=_payload_fields(
                target_id=_clean_text(target_id, field_name="target_id"),
                target_kind=_clean_text(target_kind, field_name="target_kind"),
            ),
            target_snapshot_id=target_snapshot_id,
        )

    def inspect_admission(
        self,
        *,
        request_id: str,
        candidate_id: str | None = None,
        outcome: str | None = None,
        limit: int | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_ADMISSION,
            request_id=request_id,
            payload=_payload_fields(
                candidate_id=_optional_clean_text(candidate_id, field_name="candidate_id"),
                outcome=_optional_clean_text(outcome, field_name="outcome"),
                limit=_clean_optional_limit(limit, field_name="limit"),
            ),
        )

    def inspect_resolution_queue(
        self,
        *,
        request_id: str,
        status: str | None = None,
        session_id: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_RESOLUTION_QUEUE,
            request_id=request_id,
            payload=_payload_fields(
                status=_optional_clean_text(status, field_name="status"),
                session_id=_optional_clean_text(session_id, field_name="session_id"),
            ),
        )

    def inspect_disclosure(
        self,
        *,
        request_id: str,
        target_id: str | None = None,
        target_kind: str | None = None,
        policy_name: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_DISCLOSURE,
            request_id=request_id,
            payload=_payload_fields(
                target_id=_optional_clean_text(target_id, field_name="target_id"),
                target_kind=_optional_clean_text(target_kind, field_name="target_kind"),
                policy_name=_optional_clean_text(policy_name, field_name="policy_name"),
            ),
        )

    def inspect_forgetting(
        self,
        *,
        request_id: str,
        target_id: str | None = None,
        target_kind: str | None = None,
        mode: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_FORGETTING,
            request_id=request_id,
            payload=_payload_fields(
                target_id=_optional_clean_text(target_id, field_name="target_id"),
                target_kind=_optional_clean_text(target_kind, field_name="target_kind"),
                mode=_optional_clean_text(mode, field_name="mode"),
            ),
        )

    def inspect_epistemic_status(
        self,
        *,
        request_id: str,
        claim_id: str | None = None,
        view_key: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_EPISTEMIC_STATUS,
            request_id=request_id,
            payload=_payload_fields(
                claim_id=_optional_clean_text(claim_id, field_name="claim_id"),
                view_key=_optional_clean_text(view_key, field_name="view_key"),
            ),
        )

    def inspect_outcomes(
        self,
        *,
        request_id: str,
        target_id: str | None = None,
        target_kind: str | None = None,
        label: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_OUTCOMES,
            request_id=request_id,
            payload=_payload_fields(
                target_id=_optional_clean_text(target_id, field_name="target_id"),
                target_kind=_optional_clean_text(target_kind, field_name="target_kind"),
                label=_optional_clean_text(label, field_name="label"),
            ),
        )

    def inspect_utility(
        self,
        *,
        request_id: str,
        target_id: str | None = None,
        target_kind: str | None = None,
        policy_stamp: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_UTILITY,
            request_id=request_id,
            payload=_payload_fields(
                target_id=_optional_clean_text(target_id, field_name="target_id"),
                target_kind=_optional_clean_text(target_kind, field_name="target_kind"),
                policy_stamp=_optional_clean_text(policy_stamp, field_name="policy_stamp"),
            ),
        )

    def inspect_turn_decision(
        self,
        *,
        request_id: str,
        artifact_id: str | None = None,
        run_id: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_TURN_DECISION,
            request_id=request_id,
            payload=_payload_fields(
                artifact_id=_optional_clean_text(artifact_id, field_name="artifact_id"),
                run_id=_optional_clean_text(run_id, field_name="run_id"),
            ),
        )

    def inspect_policy(
        self,
        *,
        request_id: str,
        policy_stamp: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_POLICY,
            request_id=request_id,
            payload=_payload_fields(
                policy_stamp=_optional_clean_text(policy_stamp, field_name="policy_stamp"),
            ),
        )

    def inspect_compiler(
        self,
        *,
        request_id: str,
        node_id: str | None = None,
        dirty_only: bool = False,
        limit: int | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_COMPILER,
            request_id=request_id,
            payload=_payload_fields(
                node_id=_optional_clean_text(node_id, field_name="node_id"),
                dirty_only=dirty_only if dirty_only else None,
                limit=_clean_optional_limit(limit, field_name="limit"),
            ),
        )

    def inspect_snapshot(
        self,
        *,
        request_id: str,
        snapshot_id: str | None = None,
        include_diff_from: str | None = None,
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_SNAPSHOT,
            request_id=request_id,
            payload=_payload_fields(
                snapshot_id=_optional_clean_text(snapshot_id, field_name="snapshot_id"),
                include_diff_from=_optional_clean_text(
                    include_diff_from,
                    field_name="include_diff_from",
                ),
            ),
        )

    def inspect_tiers(
        self,
        *,
        request_id: str,
        target_kind: str | None = None,
        target_id: str | None = None,
        policy_stamp: str | None = None,
        tiers: tuple[str, ...] = (),
    ) -> ServiceResponse:
        return self._inspect(
            operation=ServiceOperation.INSPECT_TIERS,
            request_id=request_id,
            payload=_payload_fields(
                target_kind=_optional_clean_text(target_kind, field_name="target_kind"),
                target_id=_optional_clean_text(target_id, field_name="target_id"),
                policy_stamp=_optional_clean_text(policy_stamp, field_name="policy_stamp"),
                tiers=_dedupe_cleaned(tiers, field_name="tiers") if tiers else (),
            ),
        )

    def _read_view(
        self,
        *,
        operation: ServiceOperation,
        request_id: str,
        view_key: str,
        disclosure_context: DisclosureContext,
        target_snapshot_id: str | None,
    ) -> ServiceResponse:
        return self._execute(
            operation=operation,
            request_id=request_id,
            payload={"view_key": _clean_text(view_key, field_name="view_key")},
            disclosure_context=disclosure_context,
            target_snapshot_id=target_snapshot_id,
        )

    def _inspect(
        self,
        *,
        operation: ServiceOperation,
        request_id: str,
        payload: Mapping[str, object],
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._execute(
            operation=operation,
            request_id=request_id,
            payload=payload,
            target_snapshot_id=target_snapshot_id,
        )

    def _execute(
        self,
        *,
        operation: ServiceOperation,
        request_id: str,
        payload: Mapping[str, object],
        disclosure_context: DisclosureContext | None = None,
        target_snapshot_id: str | None = None,
    ) -> ServiceResponse:
        return self._facade.execute(
            ServiceRequest(
                operation=operation,
                request_id=request_id,
                payload=payload,
                disclosure_context=disclosure_context,
                target_snapshot_id=target_snapshot_id,
            )
        )
