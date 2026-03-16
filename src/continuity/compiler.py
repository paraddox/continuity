"""Incremental compiler dependency and invalidation model for Continuity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _clean_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


class CompilerNodeCategory(StrEnum):
    SOURCE_INPUT = "source_input"
    DERIVED_IR = "derived_ir"
    UTILITY_STATE = "utility_state"
    COMPILED_ARTIFACT = "compiled_artifact"


class SourceInputKind(StrEnum):
    OBSERVATION = "observation"
    IMPORTED_ARTIFACT = "imported_artifact"
    ADMISSION_POLICY = "admission_policy"
    ADMISSION_DECISION = "admission_decision"
    SUBJECT_BINDING = "subject_binding"
    FORGETTING_OPERATION = "forgetting_operation"
    RESOLUTION_ACTION = "resolution_action"
    UTILITY_EVENT = "utility_event"
    POLICY_PACK = "policy_pack"
    REASONING_ADAPTER = "reasoning_adapter"
    PROMPT_RULE = "prompt_rule"


class DerivedArtifactKind(StrEnum):
    SUBJECT = "subject"
    CLAIM = "claim"
    LOCUS = "locus"
    CLAIM_RELATION = "claim_relation"


class UtilityStateKind(StrEnum):
    COMPILED_WEIGHT = "compiled_weight"


class CompiledArtifactKind(StrEnum):
    STATE_VIEW = "state_view"
    TIMELINE_VIEW = "timeline_view"
    SET_VIEW = "set_view"
    PROFILE_VIEW = "profile_view"
    PROMPT_VIEW = "prompt_view"
    EVIDENCE_VIEW = "evidence_view"
    ANSWER_VIEW = "answer_view"
    VECTOR_INDEX_RECORD = "vector_index_record"


class DependencyRole(StrEnum):
    CONTENT = "content"
    PROJECTION = "projection"
    MEMBERSHIP = "membership"
    POLICY = "policy"
    UTILITY = "utility"
    PROVENANCE = "provenance"
    INDEX = "index"


class DirtyReason(StrEnum):
    SOURCE_EDITED = "source_edited"
    ADMISSION_POLICY_CHANGED = "admission_policy_changed"
    CLAIM_CORRECTED = "claim_corrected"
    SUBJECT_IDENTITY_CHANGED = "subject_identity_changed"
    LOCUS_MEMBERSHIP_CHANGED = "locus_membership_changed"
    FORGETTING_CHANGED = "forgetting_changed"
    RESOLUTION_CHANGED = "resolution_changed"
    UTILITY_INPUT_CHANGED = "utility_input_changed"
    POLICY_UPGRADED = "policy_upgraded"
    ADAPTER_CHANGED = "adapter_changed"


_CATEGORY_KINDS: dict[CompilerNodeCategory, tuple[type[StrEnum], ...]] = {
    CompilerNodeCategory.SOURCE_INPUT: (SourceInputKind,),
    CompilerNodeCategory.DERIVED_IR: (DerivedArtifactKind,),
    CompilerNodeCategory.UTILITY_STATE: (UtilityStateKind,),
    CompilerNodeCategory.COMPILED_ARTIFACT: (CompiledArtifactKind,),
}


def _node_sort_key(node: "CompilerNode") -> tuple[str, str, str, str, str]:
    return (
        node.category.value,
        node.subject_id or "",
        node.locus_key or "",
        node.kind.value,
        node.node_id,
    )


def _dirty_node_sort_key(dirty_node: "DirtyNode") -> tuple[str, str, str, str, str]:
    return (
        dirty_node.category.value,
        dirty_node.subject_id or "",
        dirty_node.locus_key or "",
        dirty_node.kind.value,
        dirty_node.node_id,
    )


def _dirty_reason_for(node: "CompilerNode") -> DirtyReason:
    if node.category is CompilerNodeCategory.SOURCE_INPUT:
        if node.kind in {SourceInputKind.OBSERVATION, SourceInputKind.IMPORTED_ARTIFACT}:  # type: ignore[arg-type]
            return DirtyReason.SOURCE_EDITED
        if node.kind in {SourceInputKind.ADMISSION_POLICY, SourceInputKind.ADMISSION_DECISION}:  # type: ignore[arg-type]
            return DirtyReason.ADMISSION_POLICY_CHANGED
        if node.kind is SourceInputKind.SUBJECT_BINDING:  # type: ignore[comparison-overlap]
            return DirtyReason.SUBJECT_IDENTITY_CHANGED
        if node.kind is SourceInputKind.FORGETTING_OPERATION:  # type: ignore[comparison-overlap]
            return DirtyReason.FORGETTING_CHANGED
        if node.kind is SourceInputKind.RESOLUTION_ACTION:  # type: ignore[comparison-overlap]
            return DirtyReason.RESOLUTION_CHANGED
        if node.kind is SourceInputKind.UTILITY_EVENT:  # type: ignore[comparison-overlap]
            return DirtyReason.UTILITY_INPUT_CHANGED
        if node.kind in {SourceInputKind.POLICY_PACK, SourceInputKind.PROMPT_RULE}:  # type: ignore[arg-type]
            return DirtyReason.POLICY_UPGRADED
        if node.kind is SourceInputKind.REASONING_ADAPTER:  # type: ignore[comparison-overlap]
            return DirtyReason.ADAPTER_CHANGED
    if node.category is CompilerNodeCategory.DERIVED_IR:
        if node.kind is DerivedArtifactKind.CLAIM:  # type: ignore[comparison-overlap]
            return DirtyReason.CLAIM_CORRECTED
        if node.kind is DerivedArtifactKind.SUBJECT:  # type: ignore[comparison-overlap]
            return DirtyReason.SUBJECT_IDENTITY_CHANGED
        if node.kind is DerivedArtifactKind.LOCUS:  # type: ignore[comparison-overlap]
            return DirtyReason.LOCUS_MEMBERSHIP_CHANGED
        return DirtyReason.CLAIM_CORRECTED
    if node.category is CompilerNodeCategory.UTILITY_STATE:
        return DirtyReason.UTILITY_INPUT_CHANGED
    if node.kind in {CompiledArtifactKind.PROMPT_VIEW, CompiledArtifactKind.ANSWER_VIEW}:  # type: ignore[arg-type]
        return DirtyReason.POLICY_UPGRADED
    return DirtyReason.CLAIM_CORRECTED


@dataclass(frozen=True, slots=True)
class CompilerNode:
    node_id: str
    category: CompilerNodeCategory
    kind: StrEnum
    fingerprint: str
    subject_id: str | None = None
    locus_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_text(self.node_id, field_name="node_id"))
        object.__setattr__(self, "fingerprint", _clean_text(self.fingerprint, field_name="fingerprint"))
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))

        allowed_kinds = _CATEGORY_KINDS[self.category]
        if not isinstance(self.kind, allowed_kinds):
            raise ValueError(f"{self.category.value} nodes must use the correct kind enum")


@dataclass(frozen=True, slots=True)
class CompilerDependency:
    upstream_node_id: str
    downstream_node_id: str
    role: DependencyRole

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "upstream_node_id",
            _clean_text(self.upstream_node_id, field_name="upstream_node_id"),
        )
        object.__setattr__(
            self,
            "downstream_node_id",
            _clean_text(self.downstream_node_id, field_name="downstream_node_id"),
        )
        if self.upstream_node_id == self.downstream_node_id:
            raise ValueError("compiler dependencies must connect distinct nodes")


@dataclass(frozen=True, slots=True)
class CompilerChange:
    node_id: str
    category: CompilerNodeCategory
    kind: StrEnum
    reason: DirtyReason
    previous_fingerprint: str | None
    current_fingerprint: str | None
    subject_id: str | None = None
    locus_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_text(self.node_id, field_name="node_id"))
        object.__setattr__(
            self,
            "previous_fingerprint",
            _clean_optional_text(self.previous_fingerprint, field_name="previous_fingerprint"),
        )
        object.__setattr__(
            self,
            "current_fingerprint",
            _clean_optional_text(self.current_fingerprint, field_name="current_fingerprint"),
        )
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))
        if self.previous_fingerprint == self.current_fingerprint:
            raise ValueError("compiler changes require a fingerprint delta")


@dataclass(frozen=True, slots=True)
class DirtyCause:
    reason: DirtyReason
    changed_node_id: str
    changed_node_kind: str
    dependency_path: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "changed_node_id",
            _clean_text(self.changed_node_id, field_name="changed_node_id"),
        )
        object.__setattr__(
            self,
            "changed_node_kind",
            _clean_text(self.changed_node_kind, field_name="changed_node_kind"),
        )
        object.__setattr__(
            self,
            "dependency_path",
            tuple(_clean_text(node_id, field_name="dependency_path") for node_id in self.dependency_path),
        )
        if not self.dependency_path:
            raise ValueError("dirty causes require a dependency path")


@dataclass(frozen=True, slots=True)
class DirtyNode:
    node_id: str
    category: CompilerNodeCategory
    kind: StrEnum
    subject_id: str | None
    locus_key: str | None
    causes: tuple[DirtyCause, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_text(self.node_id, field_name="node_id"))
        object.__setattr__(self, "subject_id", _clean_optional_text(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "locus_key", _clean_optional_text(self.locus_key, field_name="locus_key"))
        sorted_causes = tuple(
            sorted(
                set(self.causes),
                key=lambda cause: (cause.reason.value, cause.changed_node_id, cause.dependency_path),
            )
        )
        object.__setattr__(self, "causes", sorted_causes)
        if not self.causes:
            raise ValueError("dirty nodes require at least one dirty cause")

    @property
    def reasons(self) -> tuple[DirtyReason, ...]:
        return tuple(dict.fromkeys(cause.reason for cause in self.causes))

    @property
    def cause_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(dict.fromkeys(cause.dependency_path for cause in self.causes))


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    changes: tuple[CompilerChange, ...]
    dirty_nodes: tuple[DirtyNode, ...]
    rebuild_order: tuple[str, ...]
    affected_subject_ids: tuple[str, ...]
    affected_locus_keys: tuple[str, ...]

    def dirty_node(self, node_id: str) -> DirtyNode:
        cleaned_node_id = _clean_text(node_id, field_name="node_id")
        for dirty_node in self.dirty_nodes:
            if dirty_node.node_id == cleaned_node_id:
                return dirty_node
        raise KeyError(cleaned_node_id)


def _index_nodes(nodes: tuple[CompilerNode, ...] | list[CompilerNode]) -> dict[str, CompilerNode]:
    indexed: dict[str, CompilerNode] = {}
    for node in nodes:
        if node.node_id in indexed:
            raise ValueError(f"duplicate compiler node id: {node.node_id}")
        indexed[node.node_id] = node
    return indexed


def detect_fingerprint_changes(
    *,
    previous_nodes: tuple[CompilerNode, ...] | list[CompilerNode],
    current_nodes: tuple[CompilerNode, ...] | list[CompilerNode],
) -> tuple[CompilerChange, ...]:
    previous_by_id = _index_nodes(previous_nodes)
    current_by_id = _index_nodes(current_nodes)
    changes: list[CompilerChange] = []

    for node_id in sorted(set(previous_by_id) | set(current_by_id)):
        previous = previous_by_id.get(node_id)
        current = current_by_id.get(node_id)
        reference = current or previous
        if reference is None:
            continue
        if previous and current and (
            previous.category is not current.category or previous.kind != current.kind
        ):
            raise ValueError("compiler node ids must not change category or kind across fingerprints")

        previous_fingerprint = previous.fingerprint if previous else None
        current_fingerprint = current.fingerprint if current else None
        if previous_fingerprint == current_fingerprint:
            continue

        changes.append(
            CompilerChange(
                node_id=node_id,
                category=reference.category,
                kind=reference.kind,
                reason=_dirty_reason_for(reference),
                previous_fingerprint=previous_fingerprint,
                current_fingerprint=current_fingerprint,
                subject_id=reference.subject_id,
                locus_key=reference.locus_key,
            )
        )

    return tuple(changes)


def plan_incremental_rebuild(
    *,
    nodes: tuple[CompilerNode, ...] | list[CompilerNode],
    dependencies: tuple[CompilerDependency, ...] | list[CompilerDependency],
    changes: tuple[CompilerChange, ...] | list[CompilerChange],
) -> RebuildPlan:
    node_by_id = _index_nodes(nodes)
    downstream_by_id: dict[str, list[str]] = defaultdict(list)
    dirty_causes_by_id: dict[str, set[DirtyCause]] = defaultdict(set)

    for dependency in dependencies:
        if dependency.upstream_node_id not in node_by_id:
            raise ValueError(f"unknown upstream node: {dependency.upstream_node_id}")
        if dependency.downstream_node_id not in node_by_id:
            raise ValueError(f"unknown downstream node: {dependency.downstream_node_id}")
        downstream_by_id[dependency.upstream_node_id].append(dependency.downstream_node_id)

    for downstream_ids in downstream_by_id.values():
        downstream_ids.sort(
            key=lambda node_id: _node_sort_key(node_by_id[node_id]),
        )

    queue: list[tuple[str, DirtyReason, tuple[str, ...], str]] = []
    for change in changes:
        if change.node_id not in node_by_id:
            raise ValueError(f"unknown changed node: {change.node_id}")

        changed_node = node_by_id[change.node_id]
        path = (change.node_id,)
        queue.append((change.node_id, change.reason, path, changed_node.kind.value))

        if changed_node.category not in {
            CompilerNodeCategory.SOURCE_INPUT,
            CompilerNodeCategory.UTILITY_STATE,
        }:
            dirty_causes_by_id[change.node_id].add(
                DirtyCause(
                    reason=change.reason,
                    changed_node_id=change.node_id,
                    changed_node_kind=changed_node.kind.value,
                    dependency_path=path,
                )
            )

    while queue:
        current_node_id, reason, path, changed_kind = queue.pop(0)
        for downstream_node_id in downstream_by_id.get(current_node_id, ()):
            if downstream_node_id in path:
                raise ValueError("compiler dependency graph must be acyclic")

            next_path = (*path, downstream_node_id)
            downstream_node = node_by_id[downstream_node_id]
            dirty_causes_by_id[downstream_node_id].add(
                DirtyCause(
                    reason=reason,
                    changed_node_id=path[0],
                    changed_node_kind=changed_kind,
                    dependency_path=next_path,
                )
            )
            queue.append((downstream_node_id, reason, next_path, changed_kind))

    dirty_nodes = tuple(
        DirtyNode(
            node_id=node.node_id,
            category=node.category,
            kind=node.kind,
            subject_id=node.subject_id,
            locus_key=node.locus_key,
            causes=tuple(dirty_causes_by_id[node.node_id]),
        )
        for node in sorted(
            (node_by_id[node_id] for node_id in dirty_causes_by_id),
            key=_node_sort_key,
        )
    )

    rebuild_order = _deterministic_topological_order(
        dirty_nodes=dirty_nodes,
        dependencies=tuple(dependencies),
    )
    affected_subject_ids = tuple(
        sorted({dirty_node.subject_id for dirty_node in dirty_nodes if dirty_node.subject_id is not None})
    )
    affected_locus_keys = tuple(
        sorted({dirty_node.locus_key for dirty_node in dirty_nodes if dirty_node.locus_key is not None})
    )

    return RebuildPlan(
        changes=tuple(changes),
        dirty_nodes=dirty_nodes,
        rebuild_order=rebuild_order,
        affected_subject_ids=affected_subject_ids,
        affected_locus_keys=affected_locus_keys,
    )


def _deterministic_topological_order(
    *,
    dirty_nodes: tuple[DirtyNode, ...],
    dependencies: tuple[CompilerDependency, ...],
) -> tuple[str, ...]:
    dirty_ids = {dirty_node.node_id for dirty_node in dirty_nodes}
    if not dirty_ids:
        return ()

    dirty_node_map = {
        dirty_node.node_id: dirty_node
        for dirty_node in dirty_nodes
    }
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in dirty_ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in dirty_ids}

    for dependency in dependencies:
        if dependency.upstream_node_id in dirty_ids and dependency.downstream_node_id in dirty_ids:
            predecessors[dependency.downstream_node_id].add(dependency.upstream_node_id)
            successors[dependency.upstream_node_id].add(dependency.downstream_node_id)

    ready = sorted(
        (node_id for node_id, node_predecessors in predecessors.items() if not node_predecessors),
        key=lambda node_id: _dirty_node_sort_key(dirty_node_map[node_id]),
    )
    order: list[str] = []

    while ready:
        current_node_id = ready.pop(0)
        order.append(current_node_id)
        for downstream_node_id in sorted(
            successors[current_node_id],
            key=lambda node_id: _dirty_node_sort_key(dirty_node_map[node_id]),
        ):
            predecessors[downstream_node_id].remove(current_node_id)
            if not predecessors[downstream_node_id]:
                ready.append(downstream_node_id)
                ready.sort(
                    key=lambda node_id: _dirty_node_sort_key(dirty_node_map[node_id])
                )

    if len(order) != len(dirty_ids):
        raise ValueError("compiler dependency graph must be acyclic")

    return tuple(order)


__all__ = [
    "CompiledArtifactKind",
    "CompilerChange",
    "CompilerDependency",
    "CompilerNode",
    "CompilerNodeCategory",
    "DependencyRole",
    "DerivedArtifactKind",
    "DirtyCause",
    "DirtyNode",
    "DirtyReason",
    "RebuildPlan",
    "SourceInputKind",
    "UtilityStateKind",
    "detect_fingerprint_changes",
    "plan_incremental_rebuild",
]
