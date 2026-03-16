"""Transport-neutral host service contract for Continuity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

from continuity.disclosure import DisclosureContext
from continuity.transactions import (
    DurabilityWaterline,
    TransactionKind,
    transaction_contract_for,
)
from continuity.views import ViewKind


SERVICE_CONTRACT_VERSION = "continuity_service_v1"


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


def _dedupe_enum_tuple[T: StrEnum](values: tuple[T, ...]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def _normalize_transport_value(value: object, *, field_name: str) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return tuple(
            _normalize_transport_value(item, field_name=field_name)
            for item in value
        )
    if isinstance(value, list):
        return [
            _normalize_transport_value(item, field_name=field_name)
            for item in value
        ]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            normalized[_clean_text(key, field_name=field_name)] = _normalize_transport_value(
                nested_value,
                field_name=field_name,
            )
        return normalized
    raise TypeError(f"{field_name} must contain transport-neutral data")


def _normalize_transport_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_transport_value(dict(payload), field_name="payload")
    assert isinstance(normalized, dict)
    return normalized


class OperationFamily(StrEnum):
    CONTROL = "control"
    MUTATION = "mutation"
    READ = "read"
    INSPECTION = "inspection"


class InspectionTarget(StrEnum):
    EVIDENCE = "evidence"
    ADMISSION = "admission"
    RESOLUTION_QUEUE = "resolution_queue"
    DISCLOSURE = "disclosure"
    FORGETTING = "forgetting"
    EPISTEMIC_STATUS = "epistemic_status"
    OUTCOMES = "outcomes"
    UTILITY = "utility"
    TURN_DECISION = "turn_decision"
    POLICY = "policy"
    COMPILER = "compiler"
    SNAPSHOT = "snapshot"
    TIERS = "tiers"


class ServiceOperation(StrEnum):
    INITIALIZE = "initialize"
    SAVE_TURN = "save_turn"
    SEARCH = "search"
    GET_STATE_VIEW = "get_state_view"
    GET_TIMELINE_VIEW = "get_timeline_view"
    GET_PROFILE_VIEW = "get_profile_view"
    GET_PROMPT_VIEW = "get_prompt_view"
    ANSWER_MEMORY_QUESTION = "answer_memory_question"
    FORGET_MEMORY = "forget_memory"
    WRITE_CONCLUSION = "write_conclusion"
    LIST_MEMORY_FOLLOW_UPS = "list_memory_follow_ups"
    RESOLVE_MEMORY_FOLLOW_UP = "resolve_memory_follow_up"
    IMPORT_HISTORY = "import_history"
    PUBLISH_SNAPSHOT = "publish_snapshot"
    RESOLVE_SUBJECT = "resolve_subject"
    INSPECT_EVIDENCE = "inspect_evidence"
    INSPECT_ADMISSION = "inspect_admission"
    INSPECT_RESOLUTION_QUEUE = "inspect_resolution_queue"
    INSPECT_DISCLOSURE = "inspect_disclosure"
    INSPECT_FORGETTING = "inspect_forgetting"
    INSPECT_EPISTEMIC_STATUS = "inspect_epistemic_status"
    RECORD_OUTCOME = "record_outcome"
    INSPECT_OUTCOMES = "inspect_outcomes"
    INSPECT_UTILITY = "inspect_utility"
    INSPECT_TURN_DECISION = "inspect_turn_decision"
    INSPECT_POLICY = "inspect_policy"
    INSPECT_COMPILER = "inspect_compiler"
    INSPECT_SNAPSHOT = "inspect_snapshot"
    INSPECT_TIERS = "inspect_tiers"


@dataclass(frozen=True, slots=True)
class ServiceOperationContract:
    operation: ServiceOperation
    family: OperationFamily
    transaction_kind: TransactionKind | None = None
    view_kind: ViewKind | None = None
    inspection_target: InspectionTarget | None = None
    requires_disclosure_context: bool = False
    default_minimum_waterline: DurabilityWaterline | None = None

    def __post_init__(self) -> None:
        if self.view_kind is not None and self.inspection_target is not None:
            raise ValueError("operations may target a compiled view or inspection surface, not both")

        if self.default_minimum_waterline is not None:
            if self.transaction_kind is None:
                raise ValueError("default_minimum_waterline requires a transaction_kind")
            transaction_contract = transaction_contract_for(self.transaction_kind)
            if not transaction_contract.supports_waterline(self.default_minimum_waterline):
                raise ValueError(
                    f"{self.transaction_kind.value} cannot satisfy "
                    f"{self.default_minimum_waterline.value}"
                )


@dataclass(frozen=True, slots=True)
class ServiceRequest:
    operation: ServiceOperation
    request_id: str
    payload: Mapping[str, object] = field(default_factory=dict)
    disclosure_context: DisclosureContext | None = None
    target_snapshot_id: str | None = None
    minimum_waterline: DurabilityWaterline | None = None
    contract_version: str = SERVICE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _clean_text(self.request_id, field_name="request_id"))
        object.__setattr__(
            self,
            "contract_version",
            _clean_text(self.contract_version, field_name="contract_version"),
        )
        object.__setattr__(
            self,
            "target_snapshot_id",
            _optional_clean_text(self.target_snapshot_id, field_name="target_snapshot_id"),
        )
        object.__setattr__(self, "payload", _normalize_transport_mapping(self.payload))

        contract = service_contract_for(self.operation)
        if contract.requires_disclosure_context and self.disclosure_context is None:
            raise ValueError(f"{self.operation.value} requires disclosure_context")

        if self.minimum_waterline is not None:
            if contract.transaction_kind is None:
                raise ValueError(f"{self.operation.value} does not declare a transaction waterline")
            if self.minimum_waterline is DurabilityWaterline.PREFETCH_WARMED:
                raise ValueError("prefetch_warmed is reserved for the explicit prefetch transaction")
            if not transaction_contract_for(contract.transaction_kind).supports_waterline(
                self.minimum_waterline
            ):
                raise ValueError(
                    f"{contract.transaction_kind.value} cannot satisfy "
                    f"{self.minimum_waterline.value}"
                )


@dataclass(frozen=True, slots=True)
class ResolvedServiceRequest:
    request: ServiceRequest
    contract: ServiceOperationContract
    effective_minimum_waterline: DurabilityWaterline | None = None

    def __post_init__(self) -> None:
        if self.contract.operation is not self.request.operation:
            raise ValueError("resolved requests must keep one service operation")

        if self.effective_minimum_waterline is not None:
            if self.contract.transaction_kind is None:
                raise ValueError("effective_minimum_waterline requires a mutating service operation")
            if not transaction_contract_for(self.contract.transaction_kind).supports_waterline(
                self.effective_minimum_waterline
            ):
                raise ValueError(
                    f"{self.contract.transaction_kind.value} cannot satisfy "
                    f"{self.effective_minimum_waterline.value}"
                )


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    operation: ServiceOperation
    payload: Mapping[str, object]
    reached_waterline: DurabilityWaterline | None = None
    active_snapshot_id: str | None = None
    replay_artifact_ids: tuple[str, ...] = ()
    contract_version: str = SERVICE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _normalize_transport_mapping(self.payload))
        object.__setattr__(
            self,
            "active_snapshot_id",
            _optional_clean_text(self.active_snapshot_id, field_name="active_snapshot_id"),
        )
        object.__setattr__(
            self,
            "replay_artifact_ids",
            _dedupe_cleaned(self.replay_artifact_ids, field_name="replay_artifact_ids"),
        )
        object.__setattr__(
            self,
            "contract_version",
            _clean_text(self.contract_version, field_name="contract_version"),
        )

        contract = service_contract_for(self.operation)
        if self.reached_waterline is not None:
            if contract.transaction_kind is None:
                raise ValueError(f"{self.operation.value} does not report durability waterlines")
            if self.reached_waterline is DurabilityWaterline.PREFETCH_WARMED:
                raise ValueError("prefetch_warmed is reserved for the explicit prefetch transaction")
            if not transaction_contract_for(contract.transaction_kind).supports_waterline(
                self.reached_waterline
            ):
                raise ValueError(
                    f"{contract.transaction_kind.value} cannot report "
                    f"{self.reached_waterline.value}"
                )


ServiceExecutor = Callable[[ResolvedServiceRequest], ServiceResponse]


class ContinuityServiceFacade:
    def __init__(self, executors: Mapping[ServiceOperation, ServiceExecutor]) -> None:
        normalized_executors: dict[ServiceOperation, ServiceExecutor] = {}
        for operation, executor in executors.items():
            if not isinstance(operation, ServiceOperation):
                raise TypeError("service executors must be keyed by ServiceOperation")
            if not callable(executor):
                raise TypeError(f"{operation.value} executor must be callable")
            normalized_executors[operation] = executor
        self._executors = normalized_executors

    def supported_operations(self) -> tuple[ServiceOperation, ...]:
        return tuple(operation for operation in ServiceOperation if operation in self._executors)

    def contracts(self) -> tuple[ServiceOperationContract, ...]:
        return tuple(service_contract_for(operation) for operation in self.supported_operations())

    def resolve_request(self, request: ServiceRequest) -> ResolvedServiceRequest:
        if request.contract_version != SERVICE_CONTRACT_VERSION:
            raise ValueError(f"unsupported service contract version: {request.contract_version}")

        contract = service_contract_for(request.operation)
        effective_minimum_waterline = request.minimum_waterline or contract.default_minimum_waterline
        return ResolvedServiceRequest(
            request=request,
            contract=contract,
            effective_minimum_waterline=effective_minimum_waterline,
        )

    def execute(self, request: ServiceRequest) -> ServiceResponse:
        executor = self._executors.get(request.operation)
        if executor is None:
            raise NotImplementedError(f"{request.operation.value} is not configured")

        resolved_request = self.resolve_request(request)
        response = executor(resolved_request)
        if not isinstance(response, ServiceResponse):
            raise TypeError("service executors must return ServiceResponse objects")
        if response.operation is not request.operation:
            raise ValueError("service executors must respond with the same operation they received")

        required_waterline = resolved_request.effective_minimum_waterline
        if required_waterline is not None:
            if response.reached_waterline is None:
                raise ValueError(f"{request.operation.value} must report a reached waterline")
            if response.reached_waterline.rank < required_waterline.rank:
                raise ValueError(
                    f"{request.operation.value} must reach at least {required_waterline.value}"
                )

        return response


@lru_cache(maxsize=1)
def service_contracts() -> dict[ServiceOperation, ServiceOperationContract]:
    return {
        ServiceOperation.INITIALIZE: ServiceOperationContract(
            operation=ServiceOperation.INITIALIZE,
            family=OperationFamily.CONTROL,
        ),
        ServiceOperation.SAVE_TURN: ServiceOperationContract(
            operation=ServiceOperation.SAVE_TURN,
            family=OperationFamily.MUTATION,
            transaction_kind=TransactionKind.INGEST_TURN,
            default_minimum_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
        ),
        ServiceOperation.SEARCH: ServiceOperationContract(
            operation=ServiceOperation.SEARCH,
            family=OperationFamily.READ,
            requires_disclosure_context=True,
        ),
        ServiceOperation.GET_STATE_VIEW: ServiceOperationContract(
            operation=ServiceOperation.GET_STATE_VIEW,
            family=OperationFamily.READ,
            view_kind=ViewKind.STATE,
            requires_disclosure_context=True,
        ),
        ServiceOperation.GET_TIMELINE_VIEW: ServiceOperationContract(
            operation=ServiceOperation.GET_TIMELINE_VIEW,
            family=OperationFamily.READ,
            view_kind=ViewKind.TIMELINE,
            requires_disclosure_context=True,
        ),
        ServiceOperation.GET_PROFILE_VIEW: ServiceOperationContract(
            operation=ServiceOperation.GET_PROFILE_VIEW,
            family=OperationFamily.READ,
            view_kind=ViewKind.PROFILE,
            requires_disclosure_context=True,
        ),
        ServiceOperation.GET_PROMPT_VIEW: ServiceOperationContract(
            operation=ServiceOperation.GET_PROMPT_VIEW,
            family=OperationFamily.READ,
            view_kind=ViewKind.PROMPT,
            requires_disclosure_context=True,
        ),
        ServiceOperation.ANSWER_MEMORY_QUESTION: ServiceOperationContract(
            operation=ServiceOperation.ANSWER_MEMORY_QUESTION,
            family=OperationFamily.READ,
            view_kind=ViewKind.ANSWER,
            requires_disclosure_context=True,
        ),
        ServiceOperation.FORGET_MEMORY: ServiceOperationContract(
            operation=ServiceOperation.FORGET_MEMORY,
            family=OperationFamily.MUTATION,
            transaction_kind=TransactionKind.FORGET_MEMORY,
            default_minimum_waterline=DurabilityWaterline.VIEWS_COMPILED,
        ),
        ServiceOperation.WRITE_CONCLUSION: ServiceOperationContract(
            operation=ServiceOperation.WRITE_CONCLUSION,
            family=OperationFamily.MUTATION,
            transaction_kind=TransactionKind.WRITE_CONCLUSION,
            default_minimum_waterline=DurabilityWaterline.VIEWS_COMPILED,
        ),
        ServiceOperation.LIST_MEMORY_FOLLOW_UPS: ServiceOperationContract(
            operation=ServiceOperation.LIST_MEMORY_FOLLOW_UPS,
            family=OperationFamily.READ,
        ),
        ServiceOperation.RESOLVE_MEMORY_FOLLOW_UP: ServiceOperationContract(
            operation=ServiceOperation.RESOLVE_MEMORY_FOLLOW_UP,
            family=OperationFamily.MUTATION,
        ),
        ServiceOperation.IMPORT_HISTORY: ServiceOperationContract(
            operation=ServiceOperation.IMPORT_HISTORY,
            family=OperationFamily.MUTATION,
            transaction_kind=TransactionKind.IMPORT_HISTORY,
            default_minimum_waterline=DurabilityWaterline.VIEWS_COMPILED,
        ),
        ServiceOperation.PUBLISH_SNAPSHOT: ServiceOperationContract(
            operation=ServiceOperation.PUBLISH_SNAPSHOT,
            family=OperationFamily.MUTATION,
            transaction_kind=TransactionKind.PUBLISH_SNAPSHOT,
            default_minimum_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
        ),
        ServiceOperation.RESOLVE_SUBJECT: ServiceOperationContract(
            operation=ServiceOperation.RESOLVE_SUBJECT,
            family=OperationFamily.READ,
        ),
        ServiceOperation.INSPECT_EVIDENCE: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_EVIDENCE,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.EVIDENCE,
        ),
        ServiceOperation.INSPECT_ADMISSION: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_ADMISSION,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.ADMISSION,
        ),
        ServiceOperation.INSPECT_RESOLUTION_QUEUE: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_RESOLUTION_QUEUE,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.RESOLUTION_QUEUE,
        ),
        ServiceOperation.INSPECT_DISCLOSURE: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_DISCLOSURE,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.DISCLOSURE,
        ),
        ServiceOperation.INSPECT_FORGETTING: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_FORGETTING,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.FORGETTING,
        ),
        ServiceOperation.INSPECT_EPISTEMIC_STATUS: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_EPISTEMIC_STATUS,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.EPISTEMIC_STATUS,
        ),
        ServiceOperation.RECORD_OUTCOME: ServiceOperationContract(
            operation=ServiceOperation.RECORD_OUTCOME,
            family=OperationFamily.MUTATION,
        ),
        ServiceOperation.INSPECT_OUTCOMES: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_OUTCOMES,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.OUTCOMES,
        ),
        ServiceOperation.INSPECT_UTILITY: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_UTILITY,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.UTILITY,
        ),
        ServiceOperation.INSPECT_TURN_DECISION: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_TURN_DECISION,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.TURN_DECISION,
        ),
        ServiceOperation.INSPECT_POLICY: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_POLICY,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.POLICY,
        ),
        ServiceOperation.INSPECT_COMPILER: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_COMPILER,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.COMPILER,
        ),
        ServiceOperation.INSPECT_SNAPSHOT: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_SNAPSHOT,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.SNAPSHOT,
        ),
        ServiceOperation.INSPECT_TIERS: ServiceOperationContract(
            operation=ServiceOperation.INSPECT_TIERS,
            family=OperationFamily.INSPECTION,
            inspection_target=InspectionTarget.TIERS,
        ),
    }


def service_contract_for(operation: ServiceOperation) -> ServiceOperationContract:
    return service_contracts()[operation]
