"""Deployment boundary for the Continuity host service contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from continuity.service import SERVICE_CONTRACT_VERSION
from continuity.transactions import DurabilityWaterline, TransactionKind


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _dedupe_cleaned(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_clean_text(value, field_name=field_name) for value in values))


def _dedupe_enum_tuple[T: StrEnum](values: tuple[T, ...]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


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
