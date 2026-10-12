"""Operations may only grow: the committed per-operation schema baseline.

A client and a host from different releases agree on every call that uses
nothing new, because requests leave default-valued fields off the wire and
results decode leniently. That holds only while every change to an operation
is additive, so this lock compares the live catalog with
``operation_schema_baseline.json`` and refuses anything else:

* a removed operation, or a changed ``family``, ``safety``, ``destructive`` or
  ``context_required``;
* a removed or renamed field, a changed type or default, or a new required
  field, in a request or a result, however deeply nested;
* a new member in a result ``Enum`` or ``Literal`` (an older client cannot
  decode it), or a removed member anywhere;
* a new or removed error category.

New operations and new defaulted request or result fields are allowed. Any
other change needs a new operation name or a protocol version bump. After an
allowed change, regenerate the baseline so the new shape is locked too::

    uv run python -m tests.characterization.test_operation_schema_evolution
"""

# ruff: noqa: S101 - pytest architecture tests use assertions intentionally.

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from potpie.runtime.operations import operation_schema_document

BASELINE = Path(__file__).with_name("operation_schema_baseline.json")
REGENERATE = "uv run python -m tests.characterization.test_operation_schema_evolution"

Json = Any


def current_document() -> dict[str, Json]:
    """The live catalog's schema document, as plain JSON values."""

    return json.loads(json.dumps(operation_schema_document()))


def load_baseline() -> dict[str, Json]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def render(document: Mapping[str, Json]) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def evolution_violations(old: Mapping[str, Json], new: Mapping[str, Json]) -> list[str]:
    """Every way ``new`` is not an additive evolution of ``old``."""

    violations: list[str] = []
    old_categories = set(old["error_categories"])
    new_categories = set(new["error_categories"])
    for category in sorted(new_categories - old_categories):
        violations.append(f"error category {category!r} was added")
    for category in sorted(old_categories - new_categories):
        violations.append(f"error category {category!r} was removed")
    if new["protocol_version"] != old["protocol_version"]:
        # A protocol version bump is the sanctioned way to break operations;
        # the baseline is then regenerated as a whole.
        return violations

    comparer = _Comparer(old["types"], new["types"], violations)
    for name, before in old["operations"].items():
        after = new["operations"].get(name)
        if after is None:
            violations.append(f"operation {name!r} was removed")
            continue
        for key in ("family", "safety", "destructive", "context_required"):
            if before[key] != after[key]:
                violations.append(
                    f"operation {name!r} changed {key}: "
                    f"{before[key]!r} -> {after[key]!r}"
                )
        comparer.node(before["request"], after["request"], f"{name}.request", "request")
        comparer.node(before["result"], after["result"], f"{name}.result", "result")
    return violations


class _Comparer:
    def __init__(
        self,
        old_types: Mapping[str, Json],
        new_types: Mapping[str, Json],
        violations: list[str],
    ) -> None:
        self._old_types = old_types
        self._new_types = new_types
        self._violations = violations
        self._seen: set[tuple[str, str, str]] = set()

    def node(self, old: Json, new: Json, path: str, position: str) -> None:
        if old.get("kind") == "ref" and new.get("kind") == "ref":
            key = (old["name"], new["name"], position)
            if key in self._seen:
                return
            self._seen.add(key)
        old = self._resolve(old, self._old_types)
        new = self._resolve(new, self._new_types)
        kind = old.get("kind")
        if kind != new.get("kind"):
            self._fail(f"{path} changed type: {kind} -> {new.get('kind')}")
            return
        if kind == "object":
            self._object(old, new, path, position)
        elif kind in {"literal", "enum"}:
            self._members(old["values"], new["values"], path, position)
        elif kind in {"array"}:
            self.node(old["items"], new["items"], f"{path}[]", position)
        elif kind in {"tuple", "union"}:
            key = "items" if kind == "tuple" else "options"
            if len(old[key]) != len(new[key]):
                self._fail(f"{path} changed type: {kind} members changed")
                return
            for index, (before, after) in enumerate(zip(old[key], new[key])):
                self.node(before, after, f"{path}<{index}>", position)
        elif kind == "map":
            self.node(old["keys"], new["keys"], f"{path}{{key}}", position)
            self.node(old["values"], new["values"], f"{path}{{}}", position)
        elif kind in {"opaque", "unresolved"}:
            if old.get("name") != new.get("name"):
                self._fail(f"{path} changed type: {old['name']} -> {new['name']}")

    def _object(self, old: Json, new: Json, path: str, position: str) -> None:
        after = {item["name"]: item for item in new["fields"]}
        before_names = set()
        for field in old["fields"]:
            name = field["name"]
            before_names.add(name)
            field_path = f"{path}.{name}"
            replacement = after.get(name)
            if replacement is None:
                self._fail(f"{field_path} was removed or renamed")
                continue
            if "required" in replacement and "required" not in field:
                self._fail(f"{field_path} became required")
            elif "required" not in field and _default(field) != _default(replacement):
                self._fail(
                    f"{field_path} changed default: "
                    f"{_default(field)!r} -> {_default(replacement)!r}"
                )
            self.node(field["schema"], replacement["schema"], field_path, position)
        for name, field in after.items():
            if name not in before_names and "required" in field:
                self._fail(f"{path}.{name} was added as a required field")

    def _members(
        self, old: list[Json], new: list[Json], path: str, position: str
    ) -> None:
        old_values = [json.dumps(value, sort_keys=True) for value in old]
        new_values = [json.dumps(value, sort_keys=True) for value in new]
        for value in old_values:
            if value not in new_values:
                self._fail(f"{path} lost member {value}")
        if position == "result":
            for value in new_values:
                if value not in old_values:
                    self._fail(f"{path} gained result member {value}")

    @staticmethod
    def _resolve(node: Json, types: Mapping[str, Json]) -> Json:
        if node.get("kind") == "ref":
            return types[node["name"]]
        return node

    def _fail(self, message: str) -> None:
        if message not in self._violations:
            self._violations.append(message)


def _default(field: Json) -> Json:
    if "default" in field:
        return field["default"]
    return {"computed_by": field.get("default_factory")}


# --- the live catalog against the committed baseline ------------------------


def test_operations_evolve_additively_from_the_baseline() -> None:
    violations = evolution_violations(load_baseline(), current_document())

    assert violations == [], (
        "operation changes must be additive (new operations, new defaulted "
        "fields); use a new operation name or a protocol version bump for "
        "anything else:\n  " + "\n  ".join(violations)
    )


def test_the_baseline_records_the_current_catalog() -> None:
    assert render(load_baseline()) == render(current_document()), (
        f"the operation catalog grew; lock the new shape in with: {REGENERATE}"
    )


# --- the rules themselves ----------------------------------------------------


def _document() -> dict[str, Json]:
    return {
        "format": 1,
        "protocol_version": 2,
        "error_categories": ["domain", "protocol"],
        "operations": {
            "lookup": {
                "family": "engine",
                "safety": "shared_context_read",
                "destructive": False,
                "context_required": True,
                "resource_type": None,
                "resource_identity_fields": [],
                "request": {"kind": "ref", "name": "LookupRequest"},
                "result": {"kind": "ref", "name": "LookupResult"},
            }
        },
        "types": {
            "LookupRequest": {
                "kind": "object",
                "fields": [
                    {"name": "query", "schema": {"kind": "string"}, "default": ""},
                    {
                        "name": "mode",
                        "schema": {"kind": "literal", "values": ["fast", "deep"]},
                        "default": "fast",
                    },
                ],
            },
            "LookupResult": {
                "kind": "object",
                "fields": [
                    {
                        "name": "items",
                        "schema": {
                            "kind": "array",
                            "items": {"kind": "ref", "name": "Item"},
                        },
                        "default": [],
                    },
                ],
            },
            "Item": {
                "kind": "object",
                "fields": [
                    {"name": "key", "schema": {"kind": "string"}, "required": True},
                    {
                        "name": "status",
                        "schema": {"kind": "enum", "values": ["ok", "stale"]},
                        "default": "ok",
                    },
                ],
            },
        },
    }


def _fields(document: dict[str, Json], type_name: str) -> list[Json]:
    return document["types"][type_name]["fields"]


def test_an_identical_document_is_additive() -> None:
    assert evolution_violations(_document(), _document()) == []


def test_new_operations_and_new_defaulted_fields_are_allowed() -> None:
    new = _document()
    new["operations"]["lookup_more"] = copy.deepcopy(new["operations"]["lookup"])
    _fields(new, "LookupRequest").append(
        {"name": "limit", "schema": {"kind": "integer"}, "default": 12}
    )
    _fields(new, "Item").append(
        {"name": "score", "schema": {"kind": "number"}, "default": 0.0}
    )
    # A request literal may grow: an older host refuses the new value itself.
    _fields(new, "LookupRequest")[1]["schema"]["values"].append("exhaustive")

    assert evolution_violations(_document(), new) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda doc: _fields(doc, "LookupRequest").pop(0),
            "lookup.request.query was removed or renamed",
        ),
        (
            lambda doc: _fields(doc, "LookupRequest")[0].update(name="text"),
            "lookup.request.query was removed or renamed",
        ),
        (
            lambda doc: _fields(doc, "LookupRequest")[0].update(default="*"),
            "lookup.request.query changed default: '' -> '*'",
        ),
        (
            lambda doc: _fields(doc, "LookupRequest")[0].update(
                schema={"kind": "integer"}
            ),
            "lookup.request.query changed type: string -> integer",
        ),
        (
            lambda doc: _fields(doc, "LookupRequest").append(
                {"name": "pot", "schema": {"kind": "string"}, "required": True}
            ),
            "lookup.request.pot was added as a required field",
        ),
        (
            lambda doc: _fields(doc, "Item")[1]["schema"]["values"].append("gone"),
            'lookup.result.items[].status gained result member "gone"',
        ),
        (
            lambda doc: _fields(doc, "LookupRequest")[1]["schema"]["values"].remove(
                "deep"
            ),
            'lookup.request.mode lost member "deep"',
        ),
        (
            lambda doc: doc["operations"].pop("lookup"),
            "operation 'lookup' was removed",
        ),
        (
            lambda doc: doc["operations"]["lookup"].update(
                safety="exclusive_context_mutation"
            ),
            "operation 'lookup' changed safety: "
            "'shared_context_read' -> 'exclusive_context_mutation'",
        ),
        (
            lambda doc: doc["operations"]["lookup"].update(destructive=True),
            "operation 'lookup' changed destructive: False -> True",
        ),
        (
            lambda doc: doc["operations"]["lookup"].update(context_required=False),
            "operation 'lookup' changed context_required: True -> False",
        ),
        (
            lambda doc: doc["error_categories"].append("quota"),
            "error category 'quota' was added",
        ),
    ],
)
def test_breaking_changes_are_refused(change, expected: str) -> None:
    new = _document()
    change(new)

    assert expected in evolution_violations(_document(), new)


def test_a_protocol_version_bump_is_the_sanctioned_break() -> None:
    new = _document()
    new["protocol_version"] = 3
    new["operations"].pop("lookup")

    assert evolution_violations(_document(), new) == []


def _regenerate(argv: list[str]) -> int:
    current = current_document()
    if BASELINE.exists() and "--force" not in argv:
        violations = evolution_violations(load_baseline(), current)
        if violations:
            sys.stderr.write(
                "refusing to record a non-additive change (pass --force only "
                "together with a protocol version bump):\n  "
                + "\n  ".join(violations)
                + "\n"
            )
            return 1
    BASELINE.write_text(render(current), encoding="utf-8")
    sys.stdout.write(f"wrote {BASELINE.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_regenerate(sys.argv[1:]))
