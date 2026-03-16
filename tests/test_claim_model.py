#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.store.claims import (
    AdmissionDecision,
    AdmissionOutcome,
    AggregationMode,
    CandidateMemory,
    Claim,
    ClaimProvenance,
    ClaimRelation,
    ClaimRelationKind,
    ClaimScope,
    HostMemoryArtifact,
    MemoryLocus,
    Observation,
    Subject,
    SubjectAlias,
    SubjectKind,
    SubjectMergeRecord,
    SubjectSplitRecord,
)


def sample_time() -> datetime:
    return datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


def sample_locus() -> MemoryLocus:
    return MemoryLocus(
        subject_id="subject:user:alice",
        locus_key="preference/coffee",
        scope=ClaimScope.USER,
        default_disclosure_policy="assistant_internal",
        conflict_set_key="preference/coffee",
        aggregation_mode=AggregationMode.LATEST_WINS,
    )


class SubjectInvariantTests(unittest.TestCase):
    def test_subject_aliases_are_normalized_and_auditable(self) -> None:
        subject = Subject(
            subject_id="subject:user:alice",
            kind=SubjectKind.USER,
            canonical_name="Alice Example",
            aliases=(
                SubjectAlias(
                    alias=" alice ",
                    alias_type="handle",
                    source_observation_ids=("obs-1",),
                ),
                SubjectAlias(
                    alias="Alice Example",
                    alias_type="display_name",
                    source_observation_ids=("obs-2",),
                ),
            ),
        )

        self.assertTrue(subject.matches_name("ALICE"))
        self.assertTrue(subject.matches_name("alice example"))
        self.assertEqual(
            subject.normalized_names,
            frozenset({"alice", "alice example"}),
        )

    def test_subject_rejects_duplicate_aliases_after_normalization(self) -> None:
        with self.assertRaises(ValueError):
            Subject(
                subject_id="subject:user:alice",
                kind=SubjectKind.USER,
                canonical_name="Alice Example",
                aliases=(
                    SubjectAlias(
                        alias="Alice",
                        alias_type="nickname",
                        source_observation_ids=("obs-1",),
                    ),
                    SubjectAlias(
                        alias=" alice ",
                        alias_type="handle",
                        source_observation_ids=("obs-2",),
                    ),
                ),
            )

    def test_subject_merge_history_excludes_the_surviving_subject(self) -> None:
        merge = SubjectMergeRecord(
            survivor_subject_id="subject:user:alice",
            merged_subject_ids=("subject:user:alice-smith",),
            source_observation_ids=("obs-merge",),
        )

        self.assertEqual(merge.survivor_subject_id, "subject:user:alice")
        self.assertEqual(merge.merged_subject_ids, ("subject:user:alice-smith",))

        with self.assertRaises(ValueError):
            SubjectMergeRecord(
                survivor_subject_id="subject:user:alice",
                merged_subject_ids=("subject:user:alice",),
                source_observation_ids=("obs-merge",),
            )

    def test_subject_split_history_excludes_the_source_subject(self) -> None:
        split = SubjectSplitRecord(
            source_subject_id="subject:user:alice",
            child_subject_ids=("subject:user:alice-work", "subject:user:alice-home"),
            source_observation_ids=("obs-split",),
        )

        self.assertEqual(split.source_subject_id, "subject:user:alice")
        self.assertEqual(
            split.child_subject_ids,
            ("subject:user:alice-work", "subject:user:alice-home"),
        )

        with self.assertRaises(ValueError):
            SubjectSplitRecord(
                source_subject_id="subject:user:alice",
                child_subject_ids=("subject:user:alice",),
                source_observation_ids=("obs-split",),
            )


class ObservationInvariantTests(unittest.TestCase):
    def test_observations_are_immutable_normalized_source_records(self) -> None:
        observation = Observation(
            observation_id="obs-1",
            source_kind="message",
            session_id="telegram:123456",
            author_subject_id="subject:user:alice",
            content="Alice prefers espresso.",
            observed_at=sample_time(),
        )

        self.assertEqual(observation.source_kind, "message")
        self.assertEqual(observation.author_subject_id, "subject:user:alice")

        with self.assertRaises(FrozenInstanceError):
            observation.content = "Alice prefers tea"  # type: ignore[misc]


class LocusInvariantTests(unittest.TestCase):
    def test_locus_address_is_subject_and_stable_key(self) -> None:
        locus = sample_locus()

        self.assertEqual(locus.address, ("subject:user:alice", "preference/coffee"))
        self.assertEqual(locus.conflict_set_key, "preference/coffee")
        self.assertEqual(locus.scope, ClaimScope.USER)

    def test_supported_aggregation_modes_are_explicit(self) -> None:
        self.assertEqual(
            {mode.value for mode in AggregationMode},
            {"latest_wins", "set_union", "timeline", "state_machine"},
        )


class ClaimInvariantTests(unittest.TestCase):
    def test_claim_requires_explicit_durable_admission(self) -> None:
        candidate = CandidateMemory(
            candidate_id="candidate-1",
            claim_type="preference",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"drink": "espresso"},
            source_observation_ids=("obs-1",),
        )
        admission = AdmissionDecision(
            candidate_id="candidate-1",
            outcome=AdmissionOutcome.PROMPT_ONLY,
            recorded_at=sample_time(),
            rationale="session-only signal",
        )

        with self.assertRaises(ValueError):
            Claim.from_candidate(
                claim_id="claim-1",
                candidate=candidate,
                admission=admission,
                locus=sample_locus(),
                provenance=ClaimProvenance(
                    observation_ids=("obs-1",),
                    derivation_run_id="run-1",
                ),
                disclosure_policy="assistant_internal",
                observed_at=sample_time(),
                learned_at=sample_time(),
            )

    def test_claims_are_append_only_typed_scoped_and_provenance_linked(self) -> None:
        candidate = CandidateMemory(
            candidate_id="candidate-1",
            claim_type="preference",
            subject_id="subject:user:alice",
            scope=ClaimScope.USER,
            value={"drink": "espresso"},
            source_observation_ids=("obs-1",),
        )
        claim = Claim.from_candidate(
            claim_id="claim-1",
            candidate=candidate,
            admission=AdmissionDecision(
                candidate_id="candidate-1",
                outcome=AdmissionOutcome.DURABLE_CLAIM,
                recorded_at=sample_time(),
                rationale="explicit user statement",
            ),
            locus=sample_locus(),
            provenance=ClaimProvenance(
                observation_ids=("obs-1",),
                derivation_run_id="run-1",
            ),
            disclosure_policy="assistant_internal",
            observed_at=sample_time(),
            learned_at=sample_time(),
            valid_from=sample_time(),
            valid_to=None,
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.SUPPORTS,
                    related_claim_id="claim-0",
                ),
            ),
        )

        self.assertEqual(claim.claim_type, "preference")
        self.assertEqual(claim.scope, ClaimScope.USER)
        self.assertEqual(claim.subject_id, "subject:user:alice")
        self.assertEqual(claim.locus.address, ("subject:user:alice", "preference/coffee"))
        self.assertEqual(claim.provenance.observation_ids, ("obs-1",))
        self.assertEqual(claim.provenance.derivation_run_id, "run-1")
        self.assertEqual(claim.relations[0].kind, ClaimRelationKind.SUPPORTS)

        with self.assertRaises(FrozenInstanceError):
            claim.value = {"drink": "tea"}  # type: ignore[misc]

    def test_claim_relation_kinds_cover_the_core_revision_edges(self) -> None:
        self.assertEqual(
            {kind.value for kind in ClaimRelationKind},
            {"supports", "supersedes", "contradicts", "corrects"},
        )


class HostArtifactInvariantTests(unittest.TestCase):
    def test_host_visible_artifact_requires_claim_provenance(self) -> None:
        artifact = HostMemoryArtifact(
            artifact_id="profile:subject:user:alice",
            artifact_kind="profile_view",
            claim_ids=("claim-1",),
            observation_ids=("obs-1",),
        )

        self.assertEqual(artifact.claim_ids, ("claim-1",))
        self.assertEqual(artifact.observation_ids, ("obs-1",))

        with self.assertRaises(ValueError):
            HostMemoryArtifact(
                artifact_id="profile:subject:user:alice",
                artifact_kind="profile_view",
                claim_ids=(),
                observation_ids=("obs-1",),
            )

    def test_compiled_artifacts_cannot_be_declared_as_durable_roots(self) -> None:
        with self.assertRaises(ValueError):
            HostMemoryArtifact(
                artifact_id="answer:1",
                artifact_kind="answer_view",
                claim_ids=("claim-1",),
                observation_ids=("obs-1",),
                durable_root=True,
            )


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_freezes_core_claim_ledger_invariants(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8")
        normalized = text.lower()

        self.assertIn("typed claims are the only durable derived primitive", normalized)
        self.assertIn("subject graph", normalized)
        self.assertIn("immutable observations", normalized)
        self.assertIn("memory loci", normalized)
        self.assertIn("admission", normalized)
        self.assertIn("claim provenance", normalized)
        self.assertIn("compiled views", normalized)


if __name__ == "__main__":
    unittest.main()
