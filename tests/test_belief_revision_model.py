#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DOC_PATH = ROOT_DIR / "docs" / "architecture.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from continuity.epistemics import EpistemicStatus, resolve_locus_belief
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
    MemoryLocus,
)


def sample_time(offset_minutes: int = 0) -> datetime:
    base = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def sample_locus(*, aggregation_mode: AggregationMode = AggregationMode.LATEST_WINS) -> MemoryLocus:
    return MemoryLocus(
        subject_id="subject:user:alice",
        locus_key="preference/coffee",
        scope=ClaimScope.USER,
        default_disclosure_policy="assistant_internal",
        conflict_set_key="preference/coffee",
        aggregation_mode=aggregation_mode,
    )


def make_claim(
    claim_id: str,
    *,
    learned_at: datetime,
    relations: tuple[ClaimRelation, ...] = (),
    valid_to: datetime | None = None,
    aggregation_mode: AggregationMode = AggregationMode.LATEST_WINS,
    drink: str = "espresso",
) -> Claim:
    candidate = CandidateMemory(
        candidate_id=f"candidate:{claim_id}",
        claim_type="preference",
        subject_id="subject:user:alice",
        scope=ClaimScope.USER,
        value={"drink": drink},
        source_observation_ids=(f"obs:{claim_id}",),
    )
    return Claim.from_candidate(
        claim_id=claim_id,
        candidate=candidate,
        admission=AdmissionDecision(
            candidate_id=f"candidate:{claim_id}",
            outcome=AdmissionOutcome.DURABLE_CLAIM,
            recorded_at=learned_at,
            rationale="explicit user statement",
        ),
        locus=sample_locus(aggregation_mode=aggregation_mode),
        provenance=ClaimProvenance(
            observation_ids=(f"obs:{claim_id}",),
            derivation_run_id=f"run:{claim_id}",
        ),
        disclosure_policy="assistant_internal",
        observed_at=learned_at,
        learned_at=learned_at,
        valid_from=learned_at,
        valid_to=valid_to,
        relations=relations,
    )


class BeliefRevisionInvariantTests(unittest.TestCase):
    def test_corrections_supersede_older_current_beliefs_without_erasing_history(self) -> None:
        original = make_claim("claim-1", learned_at=sample_time(0), drink="espresso")
        corrected = make_claim(
            "claim-2",
            learned_at=sample_time(10),
            drink="tea",
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.CORRECTS,
                    related_claim_id="claim-1",
                ),
            ),
        )

        projection = resolve_locus_belief((original, corrected), as_of=sample_time(20))

        self.assertEqual(projection.active_claim_ids, ("claim-2",))
        self.assertEqual(set(projection.historical_claim_ids), {"claim-1", "claim-2"})
        self.assertEqual(projection.epistemic.status, EpistemicStatus.SUPPORTED)

    def test_contradictions_do_not_silently_coexist_as_current_truth(self) -> None:
        first = make_claim(
            "claim-1",
            learned_at=sample_time(0),
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.CONTRADICTS,
                    related_claim_id="claim-2",
                ),
            ),
            drink="espresso",
        )
        second = make_claim("claim-2", learned_at=sample_time(1), drink="tea")

        projection = resolve_locus_belief((first, second), as_of=sample_time(20))

        self.assertEqual(projection.active_claim_ids, ())
        self.assertEqual(projection.epistemic.status, EpistemicStatus.CONFLICTED)
        self.assertEqual(set(projection.historical_claim_ids), {"claim-1", "claim-2"})

    def test_stale_beliefs_decay_without_dropping_historical_evidence(self) -> None:
        stale_claim = make_claim(
            "claim-1",
            learned_at=sample_time(0),
            valid_to=sample_time(5),
        )

        projection = resolve_locus_belief((stale_claim,), as_of=sample_time(20))

        self.assertEqual(projection.active_claim_ids, ("claim-1",))
        self.assertEqual(projection.epistemic.status, EpistemicStatus.STALE)
        self.assertEqual(projection.historical_claim_ids, ("claim-1",))

    def test_set_union_loci_can_keep_multiple_non_conflicting_current_claims(self) -> None:
        first = make_claim(
            "claim-1",
            learned_at=sample_time(0),
            aggregation_mode=AggregationMode.SET_UNION,
            drink="espresso",
        )
        second = make_claim(
            "claim-2",
            learned_at=sample_time(5),
            aggregation_mode=AggregationMode.SET_UNION,
            drink="cappuccino",
        )

        projection = resolve_locus_belief((first, second), as_of=sample_time(20))

        self.assertEqual(projection.active_claim_ids, ("claim-2", "claim-1"))
        self.assertEqual(projection.epistemic.status, EpistemicStatus.SUPPORTED)

    def test_retrieval_prefers_current_beliefs_before_historical_claims(self) -> None:
        original = make_claim("claim-1", learned_at=sample_time(0), drink="espresso")
        corrected = make_claim(
            "claim-2",
            learned_at=sample_time(10),
            drink="tea",
            relations=(
                ClaimRelation(
                    kind=ClaimRelationKind.CORRECTS,
                    related_claim_id="claim-1",
                ),
            ),
        )

        projection = resolve_locus_belief((original, corrected), as_of=sample_time(20))

        self.assertEqual(projection.retrieval_order, ("claim-2", "claim-1"))


class ArchitectureDocTests(unittest.TestCase):
    def test_architecture_doc_mentions_belief_revision_invariants(self) -> None:
        self.assertTrue(DOC_PATH.exists(), f"missing architecture doc: {DOC_PATH}")

        text = DOC_PATH.read_text(encoding="utf-8").lower()

        self.assertIn("beliefs are projections over claims", text)
        self.assertIn("current belief", text)
        self.assertIn("historical claim history", text)
        self.assertIn("retrieval should prefer active beliefs", text)
        self.assertIn("aggregation mode", text)


if __name__ == "__main__":
    unittest.main()
