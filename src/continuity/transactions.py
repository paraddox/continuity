"""Transaction pipeline and durability contract invariants for Continuity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

from continuity.forgetting import ForgettingMode


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _dedupe_phases(phases: tuple["TransactionPhase", ...]) -> tuple["TransactionPhase", ...]:
    return tuple(dict.fromkeys(phases))


class TransactionKind(StrEnum):
    INGEST_TURN = "ingest_turn"
    WRITE_CONCLUSION = "write_conclusion"
    FORGET_MEMORY = "forget_memory"
    IMPORT_HISTORY = "import_history"
    COMPILE_VIEWS = "compile_views"
    PUBLISH_SNAPSHOT = "publish_snapshot"
    PREFETCH_NEXT_TURN = "prefetch_next_turn"


class TransactionPhase(StrEnum):
    NORMALIZE_OBSERVATIONS = "normalize_observations"
    COMMIT_OBSERVATIONS = "commit_observations"
    RESOLVE_SUBJECTS = "resolve_subjects"
    DERIVE_CANDIDATES = "derive_candidates"
    RUN_ADMISSION = "run_admission"
    RECORD_NON_DURABLE_CONTEXT = "record_non_durable_context"
    ASSIGN_LOCI = "assign_loci"
    COMMIT_CLAIMS = "commit_claims"
    RESOLVE_FORGETTING = "resolve_forgetting"
    REVISE_BELIEFS = "revise_beliefs"
    COMPILE_VIEWS = "compile_views"
    REFRESH_UTILITY = "refresh_utility"
    CAPTURE_REPLAY = "capture_replay"
    PUBLISH_SNAPSHOT = "publish_snapshot"
    PREFETCH = "prefetch"


class DurabilityWaterline(StrEnum):
    OBSERVATION_COMMITTED = "observation_committed"
    CLAIM_COMMITTED = "claim_committed"
    VIEWS_COMPILED = "views_compiled"
    SNAPSHOT_PUBLISHED = "snapshot_published"
    PREFETCH_WARMED = "prefetch_warmed"

    @property
    def rank(self) -> int:
        return {
            DurabilityWaterline.OBSERVATION_COMMITTED: 1,
            DurabilityWaterline.CLAIM_COMMITTED: 2,
            DurabilityWaterline.VIEWS_COMPILED: 3,
            DurabilityWaterline.SNAPSHOT_PUBLISHED: 4,
            DurabilityWaterline.PREFETCH_WARMED: 5,
        }[self]


class PrefetchBehavior(StrEnum):
    NONE = "none"
    ENQUEUE = "enqueue"
    WARM_ONLY = "warm_only"


class WriteFrequencySchedule(StrEnum):
    PER_TURN = "per_turn"
    SESSION_END = "session_end"
    BATCHED_TURNS = "batched_turns"


class HostOperation(StrEnum):
    SAVE_TURN = "save_turn"
    WRITE_CONCLUSION = "write_conclusion"
    FORGET_MEMORY = "forget_memory"
    IMPORT_HISTORY = "import_history"
    PUBLISH_SNAPSHOT = "publish_snapshot"
    PREFETCH_NEXT_TURN = "prefetch_next_turn"
    READ_PROMPT = "read_prompt"

    @property
    def transaction_kind(self) -> TransactionKind:
        return {
            HostOperation.SAVE_TURN: TransactionKind.INGEST_TURN,
            HostOperation.WRITE_CONCLUSION: TransactionKind.WRITE_CONCLUSION,
            HostOperation.FORGET_MEMORY: TransactionKind.FORGET_MEMORY,
            HostOperation.IMPORT_HISTORY: TransactionKind.IMPORT_HISTORY,
            HostOperation.PUBLISH_SNAPSHOT: TransactionKind.PUBLISH_SNAPSHOT,
            HostOperation.PREFETCH_NEXT_TURN: TransactionKind.PREFETCH_NEXT_TURN,
            HostOperation.READ_PROMPT: TransactionKind.PUBLISH_SNAPSHOT,
        }[self]


@dataclass(frozen=True, slots=True)
class TransactionContract:
    kind: TransactionKind
    phases: tuple[TransactionPhase, ...]
    may_publish_snapshot: bool = False
    prefetch_behavior: PrefetchBehavior = PrefetchBehavior.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", _dedupe_phases(self.phases))
        if not self.phases:
            raise ValueError("phases must be non-empty")

        if self.may_publish_snapshot != (TransactionPhase.PUBLISH_SNAPSHOT in self.phases):
            raise ValueError("may_publish_snapshot must reflect whether publish_snapshot is present")

        if self.prefetch_behavior is PrefetchBehavior.NONE and TransactionPhase.PREFETCH in self.phases:
            raise ValueError("prefetch phases require an explicit prefetch behavior")
        if self.prefetch_behavior is not PrefetchBehavior.NONE and TransactionPhase.PREFETCH not in self.phases:
            raise ValueError("prefetch behavior requires a prefetch phase")

    def supports_waterline(self, waterline: DurabilityWaterline) -> bool:
        return waterline in self.reachable_waterlines

    @property
    def reachable_waterlines(self) -> tuple[DurabilityWaterline, ...]:
        reached: list[DurabilityWaterline] = []
        if TransactionPhase.COMMIT_OBSERVATIONS in self.phases:
            reached.append(DurabilityWaterline.OBSERVATION_COMMITTED)
        if (
            TransactionPhase.COMMIT_CLAIMS in self.phases
            and TransactionPhase.REVISE_BELIEFS in self.phases
        ):
            reached.append(DurabilityWaterline.CLAIM_COMMITTED)
        if TransactionPhase.COMPILE_VIEWS in self.phases:
            reached.append(DurabilityWaterline.VIEWS_COMPILED)
        if TransactionPhase.PUBLISH_SNAPSHOT in self.phases:
            reached.append(DurabilityWaterline.SNAPSHOT_PUBLISHED)
        if TransactionPhase.PREFETCH in self.phases:
            reached.append(DurabilityWaterline.PREFETCH_WARMED)
        return tuple(reached)


@dataclass(frozen=True, slots=True)
class WriteFrequencyPolicy:
    raw_value: str | int
    schedule: WriteFrequencySchedule
    trigger_transaction: TransactionKind
    awaited_waterline: DurabilityWaterline
    flush_on: str
    batch_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "flush_on", _clean_text(self.flush_on, field_name="flush_on"))
        if self.schedule is WriteFrequencySchedule.BATCHED_TURNS:
            if self.batch_size is None or self.batch_size <= 0:
                raise ValueError("batched turn schedules require a positive batch_size")
        elif self.batch_size is not None:
            raise ValueError("batch_size is only valid for batched turn schedules")


@dataclass(frozen=True, slots=True)
class HostOperationContract:
    operation: HostOperation
    transaction_kind: TransactionKind
    minimum_waterline: DurabilityWaterline
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cleaned_notes = tuple(_clean_text(note, field_name="notes") for note in self.notes)
        object.__setattr__(self, "notes", tuple(dict.fromkeys(cleaned_notes)))

        contract = transaction_contract_for(self.transaction_kind)
        if not contract.supports_waterline(self.minimum_waterline):
            raise ValueError(
                f"{self.transaction_kind.value} cannot satisfy {self.minimum_waterline.value}"
            )


@dataclass(slots=True)
class TransactionExecutionContext:
    kind: TransactionKind
    payload: dict[str, object] = field(default_factory=dict)
    phase_outputs: dict[TransactionPhase, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransactionPhaseExecution:
    phase: TransactionPhase
    output: object = None
    reached_waterline: DurabilityWaterline | None = None


@dataclass(frozen=True, slots=True)
class TransactionExecution:
    kind: TransactionKind
    requested_waterline: DurabilityWaterline | None
    reached_waterline: DurabilityWaterline | None
    phase_executions: tuple[TransactionPhaseExecution, ...]
    deferred_phases: tuple[TransactionPhase, ...] = ()

    @property
    def executed_phases(self) -> tuple[TransactionPhase, ...]:
        return tuple(execution.phase for execution in self.phase_executions)

    def phase_execution_for(self, phase: TransactionPhase) -> TransactionPhaseExecution:
        for execution in self.phase_executions:
            if execution.phase is phase:
                return execution
        raise KeyError(f"{phase.value} was not executed")


TransactionPhaseHandler = Callable[[TransactionExecutionContext], object]


class TransactionRunner:
    def __init__(
        self,
        phase_handlers: Mapping[TransactionPhase, TransactionPhaseHandler] | None = None,
    ) -> None:
        normalized_handlers: dict[TransactionPhase, TransactionPhaseHandler] = {}
        for phase, handler in (phase_handlers or {}).items():
            if not isinstance(phase, TransactionPhase):
                raise TypeError("transaction phase handlers must be keyed by TransactionPhase")
            if not callable(handler):
                raise TypeError(f"{phase.value} handler must be callable")
            normalized_handlers[phase] = handler
        self._phase_handlers = normalized_handlers

    def run(
        self,
        kind: TransactionKind,
        *,
        payload: Mapping[str, object] | None = None,
        requested_waterline: DurabilityWaterline | None = None,
    ) -> TransactionExecution:
        contract = transaction_contract_for(kind)
        if requested_waterline is not None and not contract.supports_waterline(requested_waterline):
            raise ValueError(f"{kind.value} cannot satisfy {requested_waterline.value}")

        context = TransactionExecutionContext(kind=kind, payload=dict(payload or {}))
        phase_executions: list[TransactionPhaseExecution] = []
        reached_waterline: DurabilityWaterline | None = None
        deferred_phases: tuple[TransactionPhase, ...] = ()

        for index, phase in enumerate(contract.phases):
            output = self._execute_phase(phase, context)
            phase_waterline = _waterline_reached_after_phase(contract, phase)
            if output is not None:
                context.phase_outputs[phase] = output
            phase_executions.append(
                TransactionPhaseExecution(
                    phase=phase,
                    output=output,
                    reached_waterline=phase_waterline,
                )
            )
            if phase_waterline is not None:
                reached_waterline = phase_waterline
            if (
                requested_waterline is not None
                and reached_waterline is not None
                and reached_waterline.rank >= requested_waterline.rank
            ):
                deferred_phases = tuple(contract.phases[index + 1 :])
                break

        return TransactionExecution(
            kind=kind,
            requested_waterline=requested_waterline,
            reached_waterline=reached_waterline,
            phase_executions=tuple(phase_executions),
            deferred_phases=deferred_phases,
        )

    def _execute_phase(
        self,
        phase: TransactionPhase,
        context: TransactionExecutionContext,
    ) -> object:
        handler = self._phase_handlers.get(phase)
        if handler is None:
            return None
        return handler(context)


@lru_cache(maxsize=1)
def transaction_contracts() -> dict[TransactionKind, TransactionContract]:
    return {
        TransactionKind.INGEST_TURN: TransactionContract(
            kind=TransactionKind.INGEST_TURN,
            phases=(
                TransactionPhase.NORMALIZE_OBSERVATIONS,
                TransactionPhase.COMMIT_OBSERVATIONS,
                TransactionPhase.RESOLVE_SUBJECTS,
                TransactionPhase.DERIVE_CANDIDATES,
                TransactionPhase.RUN_ADMISSION,
                TransactionPhase.RECORD_NON_DURABLE_CONTEXT,
                TransactionPhase.ASSIGN_LOCI,
                TransactionPhase.COMMIT_CLAIMS,
                TransactionPhase.REVISE_BELIEFS,
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
                TransactionPhase.PREFETCH,
            ),
            may_publish_snapshot=True,
            prefetch_behavior=PrefetchBehavior.ENQUEUE,
        ),
        TransactionKind.WRITE_CONCLUSION: TransactionContract(
            kind=TransactionKind.WRITE_CONCLUSION,
            phases=(
                TransactionPhase.NORMALIZE_OBSERVATIONS,
                TransactionPhase.COMMIT_OBSERVATIONS,
                TransactionPhase.RESOLVE_SUBJECTS,
                TransactionPhase.DERIVE_CANDIDATES,
                TransactionPhase.RUN_ADMISSION,
                TransactionPhase.ASSIGN_LOCI,
                TransactionPhase.COMMIT_CLAIMS,
                TransactionPhase.REVISE_BELIEFS,
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
                TransactionPhase.PREFETCH,
            ),
            may_publish_snapshot=True,
            prefetch_behavior=PrefetchBehavior.ENQUEUE,
        ),
        TransactionKind.FORGET_MEMORY: TransactionContract(
            kind=TransactionKind.FORGET_MEMORY,
            phases=(
                TransactionPhase.RESOLVE_FORGETTING,
                TransactionPhase.COMMIT_CLAIMS,
                TransactionPhase.REVISE_BELIEFS,
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
            ),
            may_publish_snapshot=True,
        ),
        TransactionKind.IMPORT_HISTORY: TransactionContract(
            kind=TransactionKind.IMPORT_HISTORY,
            phases=(
                TransactionPhase.NORMALIZE_OBSERVATIONS,
                TransactionPhase.COMMIT_OBSERVATIONS,
                TransactionPhase.RESOLVE_SUBJECTS,
                TransactionPhase.DERIVE_CANDIDATES,
                TransactionPhase.RUN_ADMISSION,
                TransactionPhase.ASSIGN_LOCI,
                TransactionPhase.COMMIT_CLAIMS,
                TransactionPhase.REVISE_BELIEFS,
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
            ),
            may_publish_snapshot=True,
        ),
        TransactionKind.COMPILE_VIEWS: TransactionContract(
            kind=TransactionKind.COMPILE_VIEWS,
            phases=(
                TransactionPhase.COMPILE_VIEWS,
                TransactionPhase.REFRESH_UTILITY,
                TransactionPhase.CAPTURE_REPLAY,
                TransactionPhase.PUBLISH_SNAPSHOT,
            ),
            may_publish_snapshot=True,
        ),
        TransactionKind.PUBLISH_SNAPSHOT: TransactionContract(
            kind=TransactionKind.PUBLISH_SNAPSHOT,
            phases=(TransactionPhase.PUBLISH_SNAPSHOT,),
            may_publish_snapshot=True,
        ),
        TransactionKind.PREFETCH_NEXT_TURN: TransactionContract(
            kind=TransactionKind.PREFETCH_NEXT_TURN,
            phases=(TransactionPhase.PREFETCH,),
            prefetch_behavior=PrefetchBehavior.WARM_ONLY,
        ),
    }


def transaction_contract_for(kind: TransactionKind) -> TransactionContract:
    return transaction_contracts()[kind]


def _waterline_reached_after_phase(
    contract: TransactionContract,
    phase: TransactionPhase,
) -> DurabilityWaterline | None:
    if phase is TransactionPhase.COMMIT_OBSERVATIONS:
        return DurabilityWaterline.OBSERVATION_COMMITTED
    if (
        phase is TransactionPhase.REVISE_BELIEFS
        and TransactionPhase.COMMIT_CLAIMS in contract.phases
    ):
        return DurabilityWaterline.CLAIM_COMMITTED
    if phase is TransactionPhase.COMPILE_VIEWS:
        return DurabilityWaterline.VIEWS_COMPILED
    if phase is TransactionPhase.PUBLISH_SNAPSHOT:
        return DurabilityWaterline.SNAPSHOT_PUBLISHED
    if phase is TransactionPhase.PREFETCH:
        return DurabilityWaterline.PREFETCH_WARMED
    return None


def write_frequency_policy_for(value: str | int) -> WriteFrequencyPolicy:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError("batched turn frequency must be positive")
        return WriteFrequencyPolicy(
            raw_value=value,
            schedule=WriteFrequencySchedule.BATCHED_TURNS,
            trigger_transaction=TransactionKind.INGEST_TURN,
            awaited_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
            flush_on="turn_threshold",
            batch_size=value,
        )

    normalized = str(value).strip().lower()
    if normalized.isdigit():
        return write_frequency_policy_for(int(normalized))

    if normalized == "async":
        return WriteFrequencyPolicy(
            raw_value="async",
            schedule=WriteFrequencySchedule.PER_TURN,
            trigger_transaction=TransactionKind.INGEST_TURN,
            awaited_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
            flush_on="background",
        )
    if normalized == "turn":
        return WriteFrequencyPolicy(
            raw_value="turn",
            schedule=WriteFrequencySchedule.PER_TURN,
            trigger_transaction=TransactionKind.INGEST_TURN,
            awaited_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
            flush_on="same_turn",
        )
    if normalized == "session":
        return WriteFrequencyPolicy(
            raw_value="session",
            schedule=WriteFrequencySchedule.SESSION_END,
            trigger_transaction=TransactionKind.INGEST_TURN,
            awaited_waterline=DurabilityWaterline.OBSERVATION_COMMITTED,
            flush_on="session_end",
        )
    raise ValueError(f"unsupported write frequency: {value!r}")


def minimum_forgetting_waterline(mode: ForgettingMode) -> DurabilityWaterline:
    if mode is ForgettingMode.SUPERSEDE:
        return DurabilityWaterline.VIEWS_COMPILED
    return DurabilityWaterline.SNAPSHOT_PUBLISHED


def host_operation_contract_for(
    operation: HostOperation,
    *,
    write_frequency: str | int = "async",
    forgetting_mode: ForgettingMode = ForgettingMode.SUPPRESS,
) -> HostOperationContract:
    if operation is HostOperation.SAVE_TURN:
        policy = write_frequency_policy_for(write_frequency)
        return HostOperationContract(
            operation=operation,
            transaction_kind=policy.trigger_transaction,
            minimum_waterline=policy.awaited_waterline,
            notes=(f"flush_on={policy.flush_on}",),
        )

    if operation is HostOperation.WRITE_CONCLUSION:
        return HostOperationContract(
            operation=operation,
            transaction_kind=operation.transaction_kind,
            minimum_waterline=DurabilityWaterline.VIEWS_COMPILED,
        )

    if operation is HostOperation.FORGET_MEMORY:
        return HostOperationContract(
            operation=operation,
            transaction_kind=operation.transaction_kind,
            minimum_waterline=minimum_forgetting_waterline(forgetting_mode),
        )

    if operation is HostOperation.IMPORT_HISTORY:
        return HostOperationContract(
            operation=operation,
            transaction_kind=operation.transaction_kind,
            minimum_waterline=DurabilityWaterline.VIEWS_COMPILED,
        )

    if operation is HostOperation.PUBLISH_SNAPSHOT:
        return HostOperationContract(
            operation=operation,
            transaction_kind=operation.transaction_kind,
            minimum_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )

    if operation is HostOperation.PREFETCH_NEXT_TURN:
        return HostOperationContract(
            operation=operation,
            transaction_kind=operation.transaction_kind,
            minimum_waterline=DurabilityWaterline.PREFETCH_WARMED,
        )

    if operation is HostOperation.READ_PROMPT:
        return HostOperationContract(
            operation=operation,
            transaction_kind=operation.transaction_kind,
            minimum_waterline=DurabilityWaterline.SNAPSHOT_PUBLISHED,
        )

    raise ValueError(f"unsupported host operation: {operation!r}")
