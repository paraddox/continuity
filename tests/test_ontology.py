#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.ontology import (
    DecayMode,
    EvidenceKind,
    MemoryClass,
    MemoryPartition,
    PromptRenderStyle,
    hermes_v1_ontology,
)
from continuity.store.claims import AdmissionOutcome, ClaimScope, SubjectKind


class OntologyInvariantTests(unittest.TestCase):
    def test_hermes_v1_ontology_is_small_explicit_and_hermes_driven(self) -> None:
        ontology = hermes_v1_ontology()

        self.assertEqual(
            tuple(memory_class.value for memory_class in MemoryClass),
            (
                "preference",
                "biography",
                "relationship",
                "task_state",
                "project_fact",
                "instruction",
                "commitment",
                "open_question",
                "ephemeral_context",
                "assistant_self_model",
            ),
        )
        self.assertEqual(tuple(spec.memory_class for spec in ontology.types()), tuple(MemoryClass))

    def test_preference_rules_pin_subject_scope_evidence_and_locus_shape(self) -> None:
        ontology = hermes_v1_ontology()
        preference = ontology.class_for_claim_type("preference")

        self.assertEqual(preference.partition, MemoryPartition.USER_MEMORY)
        self.assertEqual(preference.allowed_subject_kinds, frozenset({SubjectKind.USER, SubjectKind.PEER}))
        self.assertEqual(preference.allowed_scopes, frozenset({ClaimScope.USER, ClaimScope.PEER}))
        self.assertEqual(preference.default_admission_outcome, AdmissionOutcome.DURABLE_CLAIM)
        self.assertEqual(preference.prompt_render_style, PromptRenderStyle.PROFILE_FACT)
        self.assertEqual(preference.decay_mode, DecayMode.STABLE_UNTIL_SUPERSEDED)
        self.assertEqual(preference.locus_prefix, "preference/")
        self.assertTrue(preference.supports_subject_kind(SubjectKind.USER))
        self.assertTrue(preference.supports_scope(ClaimScope.PEER))
        self.assertTrue(preference.supports_evidence(EvidenceKind.EXPLICIT_USER_STATEMENT))
        self.assertTrue(preference.supports_locus_key("preference/coffee"))
        self.assertFalse(preference.supports_locus_key("project/build"))

    def test_open_questions_are_confirmation_gated_not_durable_by_default(self) -> None:
        question = hermes_v1_ontology().class_for_claim_type("open_question")

        self.assertEqual(question.partition, MemoryPartition.SHARED_CONTEXT)
        self.assertEqual(question.default_admission_outcome, AdmissionOutcome.NEEDS_CONFIRMATION)
        self.assertTrue(question.supports_durable_promotion)
        self.assertEqual(question.decay_mode, DecayMode.OPEN_UNTIL_RESOLVED)
        self.assertEqual(question.prompt_render_style, PromptRenderStyle.QUESTION_QUEUE)

    def test_ephemeral_context_stays_prompt_only_and_session_scoped(self) -> None:
        context = hermes_v1_ontology().class_for_claim_type("ephemeral_context")

        self.assertEqual(context.partition, MemoryPartition.EPHEMERAL_STATE)
        self.assertEqual(context.default_admission_outcome, AdmissionOutcome.PROMPT_ONLY)
        self.assertFalse(context.supports_durable_promotion)
        self.assertEqual(context.allowed_scopes, frozenset({ClaimScope.SESSION, ClaimScope.SHARED}))
        self.assertEqual(context.decay_mode, DecayMode.SESSION_ONLY)
        self.assertEqual(context.prompt_render_style, PromptRenderStyle.SESSION_NOTE)

    def test_assistant_self_model_stays_partitioned_from_user_memory(self) -> None:
        self_model = hermes_v1_ontology().class_for_claim_type("assistant_self_model")

        self.assertEqual(self_model.partition, MemoryPartition.ASSISTANT_MEMORY)
        self.assertEqual(self_model.allowed_subject_kinds, frozenset({SubjectKind.ASSISTANT}))
        self.assertEqual(self_model.default_disclosure_policy, "assistant_internal")
        self.assertEqual(self_model.default_admission_outcome, AdmissionOutcome.SESSION_EPHEMERAL)

    def test_unknown_claim_type_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            hermes_v1_ontology().class_for_claim_type("unknown_type")


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_typed_memory_ontology(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("typed memory ontology", text)
        self.assertIn("preference", text)
        self.assertIn("biography", text)
        self.assertIn("open_question", text)
        self.assertIn("assistant_self_model", text)
        self.assertIn("ephemeral_context", text)


if __name__ == "__main__":
    unittest.main()
