"""Versioned memory policy-pack invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from continuity.ontology import (
    MemoryClass,
    MemoryClassSpec,
    MemoryOntology,
    MemoryPartition,
    PromptRenderStyle,
    hermes_v1_ontology,
)
from continuity.store.claims import AdmissionOutcome


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


@dataclass(frozen=True, slots=True)
class PolicyPack:
    policy_name: str
    version: str
    ontology: MemoryOntology
    write_budgets: dict[MemoryPartition, int]
    utility_weights: dict[str, int]
    prompt_budget_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_name", _clean_text(self.policy_name, field_name="policy_name"))
        object.__setattr__(self, "version", _clean_text(self.version, field_name="version"))

        missing_partitions = set(MemoryPartition) - set(self.write_budgets)
        if missing_partitions:
            raise ValueError("write_budgets must cover every memory partition")
        if any(budget < 0 for budget in self.write_budgets.values()):
            raise ValueError("write budgets must be non-negative")
        if not self.utility_weights:
            raise ValueError("utility_weights must be non-empty")
        if self.prompt_budget_tokens <= 0:
            raise ValueError("prompt_budget_tokens must be positive")

    @property
    def policy_stamp(self) -> str:
        return f"{self.policy_name}@{self.version}"

    def enabled_memory_classes(self) -> tuple[MemoryClass, ...]:
        return tuple(spec.memory_class for spec in self.ontology.types())

    def memory_class_spec_for(self, claim_type: str) -> MemoryClassSpec:
        return self.ontology.class_for_claim_type(claim_type)

    def default_admission_outcome_for(self, claim_type: str) -> AdmissionOutcome:
        return self.memory_class_spec_for(claim_type).default_admission_outcome

    def prompt_render_style_for(self, claim_type: str) -> PromptRenderStyle:
        return self.memory_class_spec_for(claim_type).prompt_render_style

    def retrieval_rank_for(self, claim_type: str) -> int:
        spec = self.memory_class_spec_for(claim_type)
        ordered = sorted(self.ontology.types(), key=lambda candidate: candidate.retrieval_priority)
        for index, candidate in enumerate(ordered):
            if candidate.memory_class is spec.memory_class:
                return index
        raise KeyError(f"claim type not enabled by policy: {claim_type}")

    def write_budget_for_partition(self, partition: MemoryPartition) -> int:
        return self.write_budgets[partition]

    def utility_weight_for(self, signal: str) -> int:
        cleaned = _clean_text(signal, field_name="signal")
        return self.utility_weights[cleaned]

    def replay_fingerprint(self) -> tuple[str, str]:
        return (self.policy_stamp, self.ontology.ontology_id)


@lru_cache(maxsize=1)
def hermes_v1_policy_pack() -> PolicyPack:
    return PolicyPack(
        policy_name="hermes_v1",
        version="1.0.0",
        ontology=hermes_v1_ontology(),
        write_budgets={
            MemoryPartition.USER_MEMORY: 8,
            MemoryPartition.SHARED_CONTEXT: 6,
            MemoryPartition.ASSISTANT_MEMORY: 2,
            MemoryPartition.EPHEMERAL_STATE: 0,
        },
        utility_weights={
            "prompt_inclusion": 3,
            "answer_citation": 5,
            "user_corrected": -6,
            "stale_on_use": -4,
        },
        prompt_budget_tokens=1200,
    )


def get_policy_pack(policy_name: str) -> PolicyPack:
    cleaned = _clean_text(policy_name, field_name="policy_name")
    if cleaned == "hermes_v1":
        return hermes_v1_policy_pack()
    raise KeyError(f"unknown policy pack: {cleaned}")
