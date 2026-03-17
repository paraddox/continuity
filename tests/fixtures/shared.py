from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIXTURE_FAMILIES = ("core_engine", "hermes_parity", "service_contract")
PARITY_TARGET = "internal_hermes_embedded_patch_v1"


class FixtureBundleError(ValueError):
    """Raised when a normalized fixture bundle cannot be assembled safely."""


@dataclass(frozen=True)
class FixtureProducer:
    corpus: str
    manifest: str


@dataclass(frozen=True)
class FixtureRecord:
    id: str
    category: str
    title: str
    summary: str
    fixture_families: tuple[str, ...]
    normalized_input: dict[str, Any]
    expected_behavior: dict[str, Any]
    provenance: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FixtureBundle:
    bundle_version: str
    fixture_family: str
    producer: FixtureProducer
    parity_target: str
    fixtures: tuple[FixtureRecord, ...]

    def fixture(self, fixture_id: str) -> FixtureRecord:
        for fixture in self.fixtures:
            if fixture.id == fixture_id:
                return fixture
        raise FixtureBundleError(f"unknown fixture id: {fixture_id}")


def load_fixture_bundle(fixture_family: str, *, root_dir: Path) -> FixtureBundle:
    fixture_family = _validate_fixture_family_name(fixture_family)
    fixtures_dir = Path(root_dir) / "tests" / "fixtures"
    corpus_path = fixtures_dir / "hermes_corpus.json"
    manifest_path = fixtures_dir / fixture_family / "manifest.json"

    corpus = _load_json_object(corpus_path)
    manifest = _load_json_object(manifest_path)
    bundle_version = _require_string(corpus, "generated_at", source=corpus_path)
    manifest_family = _require_string(manifest, "fixture_family", source=manifest_path)
    manifest_ids = _require_string_list(manifest, "fixture_ids", source=manifest_path)

    if manifest_family != fixture_family:
        raise FixtureBundleError(
            f"manifest family mismatch in {manifest_path}: expected {fixture_family}, got {manifest_family}"
        )

    corpus_fixtures = _require_list(corpus, "fixtures", source=corpus_path)
    fixture_index: dict[str, FixtureRecord] = {}

    for payload in corpus_fixtures:
        fixture = _parse_fixture_record(payload, source=corpus_path)
        if fixture.id in fixture_index:
            raise FixtureBundleError(f"duplicate fixture id in {corpus_path}: {fixture.id}")
        fixture_index[fixture.id] = fixture

    ordered_fixtures: list[FixtureRecord] = []
    seen_manifest_ids: set[str] = set()
    for fixture_id in manifest_ids:
        if fixture_id in seen_manifest_ids:
            raise FixtureBundleError(f"duplicate manifest fixture id in {manifest_path}: {fixture_id}")
        seen_manifest_ids.add(fixture_id)

        fixture = fixture_index.get(fixture_id)
        if fixture is None:
            raise FixtureBundleError(
                f"manifest {manifest_path} references unknown fixture id: {fixture_id}"
            )
        if fixture_family not in fixture.fixture_families:
            raise FixtureBundleError(
                f"fixture {fixture_id} does not declare fixture family {fixture_family}"
            )
        ordered_fixtures.append(fixture)

    return FixtureBundle(
        bundle_version=bundle_version,
        fixture_family=fixture_family,
        producer=FixtureProducer(
            corpus=_relative_fixture_path(corpus_path, root_dir=root_dir),
            manifest=_relative_fixture_path(manifest_path, root_dir=root_dir),
        ),
        parity_target=PARITY_TARGET,
        fixtures=tuple(ordered_fixtures),
    )


def load_fixture_bundles(*, root_dir: Path) -> dict[str, FixtureBundle]:
    return {
        fixture_family: load_fixture_bundle(fixture_family, root_dir=root_dir)
        for fixture_family in FIXTURE_FAMILIES
    }


def _parse_fixture_record(payload: object, *, source: Path) -> FixtureRecord:
    if not isinstance(payload, dict):
        raise FixtureBundleError(f"fixture entries in {source} must be objects")

    fixture_families = tuple(_require_string_list(payload, "fixture_families", source=source))
    for fixture_family in fixture_families:
        _validate_fixture_family_name(fixture_family)

    normalized_input = payload.get("normalized_input")
    if not isinstance(normalized_input, dict):
        raise FixtureBundleError(f"{source} fixture normalized_input must be an object")

    expected_behavior = payload.get("expected_behavior")
    if not isinstance(expected_behavior, dict):
        raise FixtureBundleError(f"{source} fixture expected_behavior must be an object")

    provenance = payload.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise FixtureBundleError(f"{source} fixture provenance must be a non-empty list")
    for entry in provenance:
        if not isinstance(entry, dict):
            raise FixtureBundleError(f"{source} fixture provenance entries must be objects")

    return FixtureRecord(
        id=_require_string(payload, "id", source=source),
        category=_require_string(payload, "category", source=source),
        title=_require_string(payload, "title", source=source),
        summary=_require_string(payload, "summary", source=source),
        fixture_families=fixture_families,
        normalized_input=normalized_input,
        expected_behavior=expected_behavior,
        provenance=tuple(provenance),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FixtureBundleError(f"missing fixture file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureBundleError(f"invalid json in fixture file: {path}") from exc

    if not isinstance(payload, dict):
        raise FixtureBundleError(f"fixture file must contain a top-level object: {path}")
    return payload


def _relative_fixture_path(path: Path, *, root_dir: Path) -> str:
    return path.relative_to(root_dir).as_posix()


def _require_list(payload: dict[str, Any], key: str, *, source: Path) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise FixtureBundleError(f"{source} field {key} must be a list")
    return value


def _require_string(payload: dict[str, Any], key: str, *, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FixtureBundleError(f"{source} field {key} must be a non-empty string")
    return value


def _require_string_list(payload: dict[str, Any], key: str, *, source: Path) -> list[str]:
    values = _require_list(payload, key, source=source)
    if not values:
        raise FixtureBundleError(f"{source} field {key} must not be empty")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise FixtureBundleError(f"{source} field {key} must contain non-empty strings")
    return values


def _validate_fixture_family_name(fixture_family: str) -> str:
    if fixture_family not in FIXTURE_FAMILIES:
        raise FixtureBundleError(f"unknown fixture family: {fixture_family}")
    return fixture_family
