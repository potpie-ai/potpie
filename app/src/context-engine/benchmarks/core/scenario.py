"""Scenario YAML loader and schema.

Scenarios live under ``use_cases/<use_case>/scenarios/*.yaml`` and conform
to the schema documented in ``benchmarks/README.md``. This module is
intentionally schema-strict: an invalid scenario fails at load time so
authors get the error before a benchmark run begins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

USE_CASES = frozenset({"feature", "debugging", "review", "operations", "onboarding"})
TIERS = frozenset({"quick", "extended"})

_RELATIVE_OFFSET_RE = re.compile(r"^([+-]?\d+)([smhd])$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")


@dataclass(frozen=True)
class IngestStep:
    event: str  # e.g. "github/pr_merged__1042.json"
    at: str  # ISO8601 or relative offset like "-60d", "+2h"


@dataclass(frozen=True)
class EntityAssertion:
    label: str
    key_pattern: str | None = None
    where: dict[str, Any] = field(default_factory=dict)
    min_count: int = 1


@dataclass(frozen=True)
class EdgeAssertion:
    from_label: str
    to_label: str
    type: str
    min_count: int = 1


@dataclass(frozen=True)
class ReconciliationAssertion:
    soft_downgrades_max: int = 0
    failed_events_max: int = 0


@dataclass(frozen=True)
class PostIngestAssertions:
    graph_must_contain_entities: tuple[EntityAssertion, ...] = ()
    graph_must_contain_edges: tuple[EdgeAssertion, ...] = ()
    no_orphan_entities: bool = False
    reconciliation: ReconciliationAssertion = field(default_factory=ReconciliationAssertion)


@dataclass(frozen=True)
class QuerySpec:
    intent: str
    scope: dict[str, Any] = field(default_factory=dict)
    include: tuple[str, ...] = ()
    mode: str = "fast"
    source_policy: str = "references_only"


@dataclass(frozen=True)
class RetrievalAssertions:
    required_includes_used: tuple[str, ...] = ()
    source_refs_min: int = 0
    must_cite_event_id: str | None = None
    forbid_in_answer: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgeCriterion:
    name: str
    weight: int
    pass_threshold: int  # 1..5
    prompt: str


@dataclass(frozen=True)
class JudgeRubric:
    pass_score: int
    criteria: tuple[JudgeCriterion, ...]


@dataclass(frozen=True)
class AxisWeights:
    ingestion: float = 0.30
    retrieval: float = 0.30
    synthesis: float = 0.40

    def normalized(self) -> "AxisWeights":
        total = self.ingestion + self.retrieval + self.synthesis
        if total <= 0:
            raise ValueError("axis_weights must sum to a positive number")
        return AxisWeights(self.ingestion / total, self.retrieval / total, self.synthesis / total)


@dataclass(frozen=True)
class Scenario:
    id: str
    use_case: str
    tier: str
    description: str
    ingest: tuple[IngestStep, ...]
    post_ingest_assertions: PostIngestAssertions
    query: QuerySpec
    retrieval_assertions: RetrievalAssertions
    judge: JudgeRubric
    axis_weights: AxisWeights
    source_path: Path  # Where the YAML lives, for error messages.


class ScenarioLoadError(ValueError):
    """Raised with file context when a scenario fails to parse."""

    def __init__(self, source: Path, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source


def _require(d: dict[str, Any], key: str, source: Path, context: str = "") -> Any:
    if key not in d:
        where = f" in {context}" if context else ""
        raise ScenarioLoadError(source, f"missing required field '{key}'{where}")
    return d[key]


def _validate_offset(at: str, source: Path) -> str:
    if _RELATIVE_OFFSET_RE.match(at):
        return at
    # ISO 8601 — accept anything pyyaml passes through; the replay layer
    # will fail loudly if it can't parse.
    if "T" in at and ":" in at:
        return at
    raise ScenarioLoadError(source, f"invalid 'at' value '{at}'; use ISO8601 or relative offset like -60d")


def _parse_ingest(raw: list[dict[str, Any]], source: Path) -> tuple[IngestStep, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScenarioLoadError(source, "'ingest' must be a non-empty list")
    out: list[IngestStep] = []
    for i, step in enumerate(raw):
        if not isinstance(step, dict):
            raise ScenarioLoadError(source, f"ingest[{i}] must be a mapping")
        event = _require(step, "event", source, f"ingest[{i}]")
        at = _require(step, "at", source, f"ingest[{i}]")
        out.append(IngestStep(event=str(event), at=_validate_offset(str(at), source)))
    return tuple(out)


def _parse_post_ingest(raw: dict[str, Any] | None, source: Path) -> PostIngestAssertions:
    if not raw:
        return PostIngestAssertions()
    entities = tuple(
        EntityAssertion(
            label=_require(e, "label", source, "post_ingest_assertions.graph_must_contain_entities"),
            key_pattern=e.get("key_pattern"),
            where=dict(e.get("where") or {}),
            min_count=int(e.get("min_count", 1)),
        )
        for e in (raw.get("graph_must_contain_entities") or [])
    )
    edges = tuple(
        EdgeAssertion(
            from_label=_require(e, "from_label", source, "post_ingest_assertions.graph_must_contain_edges"),
            to_label=_require(e, "to_label", source, "post_ingest_assertions.graph_must_contain_edges"),
            type=_require(e, "type", source, "post_ingest_assertions.graph_must_contain_edges"),
            min_count=int(e.get("min_count", 1)),
        )
        for e in (raw.get("graph_must_contain_edges") or [])
    )
    reco_raw = raw.get("reconciliation") or {}
    reconciliation = ReconciliationAssertion(
        soft_downgrades_max=int(reco_raw.get("soft_downgrades_max", 0)),
        failed_events_max=int(reco_raw.get("failed_events_max", 0)),
    )
    return PostIngestAssertions(
        graph_must_contain_entities=entities,
        graph_must_contain_edges=edges,
        no_orphan_entities=bool(raw.get("no_orphan_entities", False)),
        reconciliation=reconciliation,
    )


def _parse_query(raw: dict[str, Any], source: Path) -> QuerySpec:
    return QuerySpec(
        intent=_require(raw, "intent", source, "query"),
        scope=dict(raw.get("scope") or {}),
        include=tuple(raw.get("include") or ()),
        mode=str(raw.get("mode", "fast")),
        source_policy=str(raw.get("source_policy", "references_only")),
    )


def _parse_retrieval(raw: dict[str, Any] | None) -> RetrievalAssertions:
    if not raw:
        return RetrievalAssertions()
    return RetrievalAssertions(
        required_includes_used=tuple(raw.get("required_includes_used") or ()),
        source_refs_min=int(raw.get("source_refs_min", 0)),
        must_cite_event_id=raw.get("must_cite_event_id"),
        forbid_in_answer=tuple(raw.get("forbid_in_answer") or ()),
    )


def _parse_judge(raw: dict[str, Any], source: Path) -> JudgeRubric:
    pass_score = int(raw.get("pass_score", 75))
    criteria_raw = _require(raw, "criteria", source, "judge")
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise ScenarioLoadError(source, "judge.criteria must be a non-empty list")
    criteria = tuple(
        JudgeCriterion(
            name=_require(c, "name", source, "judge.criteria"),
            weight=int(_require(c, "weight", source, "judge.criteria")),
            pass_threshold=int(_require(c, "pass_threshold", source, "judge.criteria")),
            prompt=_require(c, "prompt", source, "judge.criteria"),
        )
        for c in criteria_raw
    )
    weight_sum = sum(c.weight for c in criteria)
    if weight_sum != 100:
        raise ScenarioLoadError(source, f"judge.criteria weights must sum to 100, got {weight_sum}")
    return JudgeRubric(pass_score=pass_score, criteria=criteria)


def load_scenario(path: Path) -> Scenario:
    """Parse a single YAML file into a Scenario.

    Raises ``ScenarioLoadError`` (a ``ValueError``) on schema violations.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ScenarioLoadError(path, "top-level YAML must be a mapping")

    scenario_id = _require(raw, "id", path)
    if not _ID_RE.match(scenario_id):
        raise ScenarioLoadError(path, f"id '{scenario_id}' must match {_ID_RE.pattern}")

    use_case = _require(raw, "use_case", path)
    if use_case not in USE_CASES:
        raise ScenarioLoadError(path, f"use_case '{use_case}' must be one of {sorted(USE_CASES)}")

    tier = raw.get("tier", "quick")
    if tier not in TIERS:
        raise ScenarioLoadError(path, f"tier '{tier}' must be one of {sorted(TIERS)}")

    weights_raw = raw.get("axis_weights") or {}
    axis_weights = AxisWeights(
        ingestion=float(weights_raw.get("ingestion", AxisWeights.ingestion)),
        retrieval=float(weights_raw.get("retrieval", AxisWeights.retrieval)),
        synthesis=float(weights_raw.get("synthesis", AxisWeights.synthesis)),
    ).normalized()

    return Scenario(
        id=scenario_id,
        use_case=use_case,
        tier=tier,
        description=str(raw.get("description") or "").strip(),
        ingest=_parse_ingest(_require(raw, "ingest", path), path),
        post_ingest_assertions=_parse_post_ingest(raw.get("post_ingest_assertions"), path),
        query=_parse_query(_require(raw, "query", path), path),
        retrieval_assertions=_parse_retrieval(raw.get("retrieval_assertions")),
        judge=_parse_judge(_require(raw, "judge", path), path),
        axis_weights=axis_weights,
        source_path=path,
    )


def discover_scenarios(root: Path) -> list[Scenario]:
    """Walk ``use_cases/*/scenarios/*.yaml`` and load all scenarios.

    Errors are collected and re-raised as a single aggregate so a typo in
    one scenario doesn't hide problems in five others.
    """
    use_cases_dir = root / "use_cases"
    if not use_cases_dir.exists():
        return []

    scenarios: list[Scenario] = []
    errors: list[str] = []
    seen_ids: dict[str, Path] = {}

    for yaml_path in sorted(use_cases_dir.glob("*/scenarios/*.yaml")):
        try:
            scenario = load_scenario(yaml_path)
        except ScenarioLoadError as exc:
            errors.append(str(exc))
            continue

        # Sanity-check use_case matches the directory.
        directory_use_case = yaml_path.parent.parent.name
        if scenario.use_case != directory_use_case:
            errors.append(
                f"{yaml_path}: declared use_case '{scenario.use_case}' "
                f"does not match directory '{directory_use_case}'"
            )
            continue

        if scenario.id in seen_ids:
            errors.append(
                f"{yaml_path}: duplicate scenario id '{scenario.id}' "
                f"(also in {seen_ids[scenario.id]})"
            )
            continue
        seen_ids[scenario.id] = yaml_path
        scenarios.append(scenario)

    if errors:
        raise ScenarioLoadError(use_cases_dir, "scenario discovery failed:\n  " + "\n  ".join(errors))
    return scenarios
