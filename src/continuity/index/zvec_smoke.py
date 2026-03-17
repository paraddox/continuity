"""Real-backend smoke test for the optional zvec retrieval bootstrap."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from continuity.index.zvec_index import IndexedDocument, VectorIndexRecord, ZvecBackend
from continuity.index.zvec_index import IndexSourceKind


class BackendFactory(Protocol):
    def __call__(
        self,
        *,
        collection_path: str,
        dimensions: int,
        collection_name: str = "continuity_smoke",
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class ZvecSmokeResult:
    collection_path: str
    indexed_record_ids: tuple[str, ...]
    top_hit_record_id: str
    remaining_record_ids: tuple[str, ...]


def _build_documents() -> tuple[IndexedDocument, ...]:
    return (
        IndexedDocument(
            record=VectorIndexRecord(
                record_id="vector:smoke:alpha",
                source_kind=IndexSourceKind.CLAIM,
                source_id="smoke-claim-alpha",
                subject_id="subject:smoke:alpha",
                locus_key="smoke/favorite_drink",
                policy_stamp="smoke@1",
                document_text="alpha espresso preference",
                embedding_model="smoke-fixture",
                embedding_fingerprint="smoke-fixture@1",
                metadata={"smoke": True, "label": "alpha"},
            ),
            vector=(1.0, 0.0, 0.0, 0.0),
        ),
        IndexedDocument(
            record=VectorIndexRecord(
                record_id="vector:smoke:beta",
                source_kind=IndexSourceKind.CLAIM,
                source_id="smoke-claim-beta",
                subject_id="subject:smoke:beta",
                locus_key="smoke/favorite_drink",
                policy_stamp="smoke@1",
                document_text="beta tea preference",
                embedding_model="smoke-fixture",
                embedding_fingerprint="smoke-fixture@1",
                metadata={"smoke": True, "label": "beta"},
            ),
            vector=(0.0, 1.0, 0.0, 0.0),
        ),
    )


def run_zvec_backend_smoke(
    *,
    collection_path: str | Path,
    backend_factory: BackendFactory = ZvecBackend,
) -> ZvecSmokeResult:
    collection_path_value = Path(collection_path).expanduser()
    if collection_path_value.exists():
        raise ValueError("collection_path must not already exist")
    collection_path_value.parent.mkdir(parents=True, exist_ok=True)
    normalized_path = str(collection_path_value)
    backend = backend_factory(
        collection_path=normalized_path,
        dimensions=4,
        collection_name="continuity_smoke",
    )
    documents = _build_documents()

    backend.upsert_documents(documents)

    top_hits = backend.query(query_vector=(1.0, 0.0, 0.0, 0.0), topk=1)
    if not top_hits:
        raise RuntimeError("zvec smoke query returned no hits after upsert")
    top_hit_record_id = top_hits[0].record_id
    if top_hit_record_id != "vector:smoke:alpha":
        raise RuntimeError(f"zvec smoke query returned unexpected top hit: {top_hit_record_id}")

    backend.delete_documents(("vector:smoke:beta",))
    remaining_hits = backend.query(query_vector=(1.0, 0.0, 0.0, 0.0), topk=10)
    remaining_record_ids = tuple(hit.record_id for hit in remaining_hits)
    if "vector:smoke:beta" in remaining_record_ids:
        raise RuntimeError("zvec smoke delete did not remove vector:smoke:beta")

    return ZvecSmokeResult(
        collection_path=normalized_path,
        indexed_record_ids=tuple(document.record.record_id for document in documents),
        top_hit_record_id=top_hit_record_id,
        remaining_record_ids=remaining_record_ids,
    )


def main(argv: list[str] | None = None, *, backend_factory: BackendFactory = ZvecBackend) -> int:
    parser = argparse.ArgumentParser(description="Run a real-backend zvec smoke test for Continuity.")
    parser.add_argument(
        "--collection-path",
        help="Directory for the zvec collection. Defaults to a temporary directory.",
    )
    args = parser.parse_args(argv)

    if args.collection_path:
        result = run_zvec_backend_smoke(
            collection_path=args.collection_path,
            backend_factory=backend_factory,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="continuity-zvec-smoke-") as temp_dir:
            result = run_zvec_backend_smoke(
                collection_path=Path(temp_dir) / "collection",
                backend_factory=backend_factory,
            )

    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
