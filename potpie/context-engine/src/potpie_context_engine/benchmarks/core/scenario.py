"""Scenario YAML loader and schema.

Scenarios live under ``use_cases/<USE_CASE>/scenarios/*.yaml`` and conform
to the schema documented in ``benchmarks/README.md`` and the bench plan
(``docs/context-graph/bench-plan.md``).

The schema is intentionally strict — an invalid scenario fails at load
time so authors get the error before a benchmark run begins. Every new
field added in this rewrite (dimensions, difficulty, source_mix,
distractor_events, graph_must_not_contain, temporal, judge
criterion.dimensions) is optional unless declared otherwise, so the
existing OPS-218 scenario continues to load after its relocation to
``use_cases/BUG/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Knowledge-dimension use cases (see bench-plan §2).
USE_CASES = frozenset({"PREF", "INFRA", "TIME", "BUG", "COMBO"})

# Dimensions a scenario can exercise. COMBO is a *modifier* — composite
# scenarios live under ``use_cases/COMBO/`` and declare which dimensions
# they span via ``dimensions:``.
DIMENSIONS = frozenset({"PREF", "INFRA", "TIME", "BUG"})

DIFFICULTIES = ("easy", "medium", "hard", "adversarial")
SOURCE_MIXES = ("single", "dual", "full", "adversarial")
TIERS = frozenset({"quick", "extended"})

# Per-use-case axis-weight defaults (bench-plan §3.2). A scenario can
# override via its own ``axis_weights:`` block, but defaults flow from
# use_case so we don't restate weights in every file.
DEFAULT_AXIS_WEIGHTS: dict[str, tuple[float, float, float]] = {
    # use_case -> (ingestion, retrieval, synthesis)
    "PREF": (0.20, 0.40, 0.40),
    "INFRA": (0.30, 0.40, 0.30),
    "TIME": (0.40, 0.30, 0.30),
    "BUG": (0.25, 0.35, 0.40),
    "COMBO": (0.30, 0.35, 0.35),
}

# Difficulty-adjusted default ``pass_score``. The same scenario *answer*
# satisfies different bars on different difficulty rungs — a 70 on
# `adversarial` is great; a 70 on `easy` is mediocre. Scenarios that
# explicitly set ``judge.pass_score`` win; otherwise the loader picks
# from this table by difficulty.
DEFAULT_PASS_SCORE_BY_DIFFICULTY: dict[str, int] = {
    "easy": 75,
    "medium": 65,
    "hard": 55,
    "adversarial": 45,
}

_RELATIVE_OFFSET_RE = re.compile(r"^([+-]?\d+)([smhd])$")
_RELATIVE_OFFSET_RANGE_RE = re.compile(r"^([+-]?\d+)([smhd])\.\.([+-]?\d+)([smhd])$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")


@dataclass(frozen=True)
class IngestStep:
    event: str  # e.g. "github/pr_merged__1042.json"
    at: str  # ISO8601 or relative offset like "-60d", "+2h"
    tags: tuple[str, ...] = ()  # author hint, e.g. ("signal",) — not enforced.


@dataclass(frozen=True)
class DistractorStep:
    """A noise event (or batch) injected into the ingestion timeline.

    Distractors are events the engine should ingest *and then not surface*
    on retrieval. They are the negative class for the precision sub-axis.

    ``count`` expands a single fixture pattern into N events; if the
    pattern contains a wildcard the matching files are enumerated
    instead. ``at`` may be a single offset or a range like ``-21d..-7d``,
    in which case the events are spread uniformly across the range.
    """

    event: str  # path or glob pattern, e.g. "linear/issue_create__noise_*.json"
    at: str
    count: int = 1
    shape: str | None = None  # author hint, e.g. "noise/random"


@dataclass(frozen=True)
class SeedStep:
    """A canonical-universe seed event (see bench-plan §5.1).

    Seed events are ingested at scenario start *before* signal/distractor
    events. They establish the Acme universe (services, team, repos,
    ADRs, runbooks) so the engine has the same baseline shape across
    every scenario.
    """

    event: str  # universe-relative path, e.g. "universe/acme/services.yaml"
    at: str = "-365d"


@dataclass(frozen=True)
class EntityAssertion:
    label: str
    key_pattern: str | None = None
    where: dict[str, Any] = field(default_factory=dict)
    min_count: int = 1
    # Dimensions this assertion attributes credit to. Empty means
    # "scenario-wide" (all the scenario's declared dimensions). Used by
    # the future per-dimension ingestion roll-up in by_dimension; today
    # the runner still broadcasts ingestion across declared dimensions
    # — this field is a forward-compat hook.
    dimensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class NegativeEntityAssertion:
    """A label / key pattern that the graph *must not* contain.

    Feeds the ingestion precision sub-axis: noise events were ingested
    but their entities should not have survived (or, if they did, they
    should be filtered out by the reconciliation agent's de-duplication
    / scoring).
    """

    label: str
    key_pattern: str | None = None
    where: dict[str, Any] = field(default_factory=dict)
    max_count: int = 0


@dataclass(frozen=True)
class EdgeAssertion:
    from_label: str
    to_label: str
    type: str
    min_count: int = 1
    dimensions: tuple[str, ...] = ()  # see EntityAssertion.dimensions


@dataclass(frozen=True)
class ReconciliationAssertion:
    soft_downgrades_max: int = 0
    failed_events_max: int = 0


@dataclass(frozen=True)
class PostIngestAssertions:
    graph_must_contain_entities: tuple[EntityAssertion, ...] = ()
    graph_must_contain_edges: tuple[EdgeAssertion, ...] = ()
    graph_must_not_contain: tuple[NegativeEntityAssertion, ...] = ()
    no_orphan_entities: bool = False
    reconciliation: ReconciliationAssertion = field(
        default_factory=ReconciliationAssertion
    )
    # Opt-in sub-axis gates. When set, the ingestion axis FAILS if the
    # sub-axis drops below the floor, regardless of the primary structural
    # score (bench-plan v3 §3.3, second-pass decision). Defaults of ``None``
    # mean "report-only, don't gate" — back-compat with existing scenarios.
    coverage_floor: int | None = None
    precision_floor: int | None = None


@dataclass(frozen=True)
class QuerySpec:
    intent: str
    scope: dict[str, Any] = field(default_factory=dict)
    include: tuple[str, ...] = ()
    mode: str = "fast"
    source_policy: str = "references_only"


@dataclass(frozen=True)
class TemporalAssertion:
    """Optional temporal constraints used by TIME scenarios."""

    must_order_correctly: bool = False
    window_from: str | None = None
    window_to: str | None = None
    out_of_window_refs_max: int = 0


@dataclass(frozen=True)
class RetrievalAssertions:
    required_includes_used: tuple[str, ...] = ()
    source_refs_min: int = 0
    must_cite_event_ids: tuple[str, ...] = ()
    must_not_cite_event_ids: tuple[str, ...] = ()
    forbid_in_answer: tuple[str, ...] = ()
    temporal: TemporalAssertion | None = None
    # Opt-in sub-axis gates (see PostIngestAssertions for the same idea
    # on the ingestion axis). Defaults of ``None`` mean report-only.
    coverage_floor: int | None = None
    precision_floor: int | None = None


@dataclass(frozen=True)
class JudgeCriterion:
    name: str
    weight: int
    pass_threshold: int  # 1..5
    prompt: str
    # Dimensions this criterion attributes credit to (composite scenarios).
    # Empty means the criterion attributes to the scenario's primary
    # dimensions (i.e. its declared ``dimensions:`` or its use_case).
    dimensions: tuple[str, ...] = ()


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
        return AxisWeights(
            self.ingestion / total, self.retrieval / total, self.synthesis / total
        )


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

    # Bench-plan v3 additions (all optional; defaults preserve old behaviour).
    dimensions: tuple[str, ...] = ()
    difficulty: str = "easy"
    source_mix: str = "single"
    universe: str | None = None
    seed: tuple[SeedStep, ...] = ()
    distractor_events: tuple[DistractorStep, ...] = ()
    # Curated 5-scenario subset for ``benchmarks run-light``. Set on
    # exactly one easy/medium scenario per dimension so a light run
    # exercises PREF/INFRA/TIME/BUG/COMBO end-to-end in seconds (with
    # ``--concurrency 5`` + invariant judging).
    light: bool = False

    @property
    def effective_dimensions(self) -> tuple[str, ...]:
        """Dimensions this scenario contributes credit to.

        For PREF/INFRA/TIME/BUG, the use_case *is* the single dimension.
        For COMBO, the scenario's explicit ``dimensions:`` list wins.
        """
        if self.use_case in DIMENSIONS:
            return (self.use_case,)
        return self.dimensions or ()


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
    if _RELATIVE_OFFSET_RANGE_RE.match(at):
        return at
    # ISO 8601 — accept anything pyyaml passes through; the replay layer
    # will fail loudly if it can't parse.
    if "T" in at and ":" in at:
        return at
    raise ScenarioLoadError(
        source,
        f"invalid 'at' value '{at}'; use ISO8601, relative offset like -60d, "
        "or a range like -21d..-7d",
    )


def _parse_ingest(raw: list[dict[str, Any]], source: Path) -> tuple[IngestStep, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScenarioLoadError(source, "'ingest' must be a non-empty list")
    out: list[IngestStep] = []
    for i, step in enumerate(raw):
        if not isinstance(step, dict):
            raise ScenarioLoadError(source, f"ingest[{i}] must be a mapping")
        event = _require(step, "event", source, f"ingest[{i}]")
        at = _require(step, "at", source, f"ingest[{i}]")
        tags = tuple(str(t) for t in (step.get("tags") or ()))
        out.append(
            IngestStep(
                event=str(event), at=_validate_offset(str(at), source), tags=tags
            )
        )
    return tuple(out)


def _parse_distractors(raw: Any, source: Path) -> tuple[DistractorStep, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ScenarioLoadError(source, "'distractor_events' must be a list")
    out: list[DistractorStep] = []
    for i, step in enumerate(raw):
        if not isinstance(step, dict):
            raise ScenarioLoadError(source, f"distractor_events[{i}] must be a mapping")
        event = _require(step, "event", source, f"distractor_events[{i}]")
        at = _require(step, "at", source, f"distractor_events[{i}]")
        count = int(step.get("count", 1))
        if count < 1:
            raise ScenarioLoadError(
                source, f"distractor_events[{i}].count must be >= 1, got {count}"
            )
        out.append(
            DistractorStep(
                event=str(event),
                at=_validate_offset(str(at), source),
                count=count,
                shape=str(step.get("shape")) if step.get("shape") else None,
            )
        )
    return tuple(out)


def _parse_seed(raw: Any, source: Path) -> tuple[SeedStep, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ScenarioLoadError(source, "'seed' must be a list")
    out: list[SeedStep] = []
    for i, step in enumerate(raw):
        if not isinstance(step, dict):
            raise ScenarioLoadError(source, f"seed[{i}] must be a mapping")
        event = _require(step, "event", source, f"seed[{i}]")
        at = str(step.get("at", "-365d"))
        out.append(SeedStep(event=str(event), at=_validate_offset(at, source)))
    return tuple(out)


def _parse_negative_entities(
    raw: Any, source: Path
) -> tuple[NegativeEntityAssertion, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ScenarioLoadError(source, "'graph_must_not_contain' must be a list")
    return tuple(
        NegativeEntityAssertion(
            label=_require(
                e, "label", source, "post_ingest_assertions.graph_must_not_contain"
            ),
            key_pattern=e.get("key_pattern"),
            where=dict(e.get("where") or {}),
            max_count=int(e.get("max_count", 0)),
        )
        for e in raw
    )


def _parse_post_ingest(
    raw: dict[str, Any] | None, source: Path
) -> PostIngestAssertions:
    if not raw:
        return PostIngestAssertions()
    entities = tuple(
        EntityAssertion(
            label=_require(
                e, "label", source, "post_ingest_assertions.graph_must_contain_entities"
            ),
            key_pattern=e.get("key_pattern"),
            where=dict(e.get("where") or {}),
            min_count=int(e.get("min_count", 1)),
            dimensions=_parse_dimension_list(
                e.get("dimensions"), source, "entity assertion"
            ),
        )
        for e in (raw.get("graph_must_contain_entities") or [])
    )
    edges = tuple(
        EdgeAssertion(
            from_label=_require(
                e,
                "from_label",
                source,
                "post_ingest_assertions.graph_must_contain_edges",
            ),
            to_label=_require(
                e, "to_label", source, "post_ingest_assertions.graph_must_contain_edges"
            ),
            type=_require(
                e, "type", source, "post_ingest_assertions.graph_must_contain_edges"
            ),
            min_count=int(e.get("min_count", 1)),
            dimensions=_parse_dimension_list(
                e.get("dimensions"), source, "edge assertion"
            ),
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
        graph_must_not_contain=_parse_negative_entities(
            raw.get("graph_must_not_contain"), source
        ),
        no_orphan_entities=bool(raw.get("no_orphan_entities", False)),
        reconciliation=reconciliation,
        coverage_floor=_optional_int(raw.get("coverage_floor")),
        precision_floor=_optional_int(raw.get("precision_floor")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_dimension_list(raw: Any, source: Path, context: str) -> tuple[str, ...]:
    """Parse and validate a ``dimensions: [...]`` field on an assertion."""
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ScenarioLoadError(source, f"'dimensions' on {context} must be a list")
    out = tuple(str(d) for d in raw)
    for d in out:
        if d not in DIMENSIONS:
            raise ScenarioLoadError(
                source,
                f"dimension '{d}' on {context} must be one of {sorted(DIMENSIONS)}",
            )
    return out


def _parse_query(raw: dict[str, Any], source: Path) -> QuerySpec:
    return QuerySpec(
        intent=_require(raw, "intent", source, "query"),
        scope=dict(raw.get("scope") or {}),
        include=tuple(raw.get("include") or ()),
        mode=str(raw.get("mode", "fast")),
        source_policy=str(raw.get("source_policy", "references_only")),
    )


def _parse_temporal(raw: Any, source: Path) -> TemporalAssertion | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ScenarioLoadError(
            source, "'retrieval_assertions.temporal' must be a mapping"
        )
    window = raw.get("window") or {}
    return TemporalAssertion(
        must_order_correctly=bool(raw.get("must_order_correctly", False)),
        window_from=str(window["from"])
        if isinstance(window, dict) and "from" in window
        else None,
        window_to=str(window["to"])
        if isinstance(window, dict) and "to" in window
        else None,
        out_of_window_refs_max=int(raw.get("out_of_window_refs_max", 0)),
    )


def _parse_retrieval(raw: dict[str, Any] | None, source: Path) -> RetrievalAssertions:
    if not raw:
        return RetrievalAssertions()

    # ``must_cite_event_id`` accepts either a single string (back-compat) or a list.
    raw_must = raw.get("must_cite_event_id")
    if isinstance(raw_must, str):
        must_cite = (raw_must,)
    elif isinstance(raw_must, list):
        must_cite = tuple(str(x) for x in raw_must)
    else:
        must_cite = ()

    raw_must_not = raw.get("must_not_cite_event_id")
    if isinstance(raw_must_not, str):
        must_not_cite = (raw_must_not,)
    elif isinstance(raw_must_not, list):
        must_not_cite = tuple(str(x) for x in raw_must_not)
    else:
        must_not_cite = ()

    return RetrievalAssertions(
        required_includes_used=tuple(raw.get("required_includes_used") or ()),
        source_refs_min=int(raw.get("source_refs_min", 0)),
        must_cite_event_ids=must_cite,
        must_not_cite_event_ids=must_not_cite,
        forbid_in_answer=tuple(raw.get("forbid_in_answer") or ()),
        temporal=_parse_temporal(raw.get("temporal"), source),
        coverage_floor=_optional_int(raw.get("coverage_floor")),
        precision_floor=_optional_int(raw.get("precision_floor")),
    )


def _parse_judge(raw: dict[str, Any], source: Path, *, difficulty: str) -> JudgeRubric:
    if "pass_score" in raw:
        pass_score = int(raw["pass_score"])
    else:
        # Difficulty-adjusted default so the same answer satisfies a
        # different bar on different rungs (bench-plan v3 §3.3).
        pass_score = DEFAULT_PASS_SCORE_BY_DIFFICULTY.get(difficulty, 70)
    criteria_raw = _require(raw, "criteria", source, "judge")
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise ScenarioLoadError(source, "judge.criteria must be a non-empty list")
    criteria_list: list[JudgeCriterion] = []
    for c in criteria_raw:
        dims = tuple(str(d) for d in (c.get("dimensions") or ()))
        for d in dims:
            if d not in DIMENSIONS:
                raise ScenarioLoadError(
                    source,
                    f"judge.criteria '{c.get('name')}' has invalid dimension '{d}'; "
                    f"must be one of {sorted(DIMENSIONS)}",
                )
        criteria_list.append(
            JudgeCriterion(
                name=_require(c, "name", source, "judge.criteria"),
                weight=int(_require(c, "weight", source, "judge.criteria")),
                pass_threshold=int(
                    _require(c, "pass_threshold", source, "judge.criteria")
                ),
                prompt=_require(c, "prompt", source, "judge.criteria"),
                dimensions=dims,
            )
        )
    criteria = tuple(criteria_list)
    weight_sum = sum(c.weight for c in criteria)
    if weight_sum != 100:
        raise ScenarioLoadError(
            source, f"judge.criteria weights must sum to 100, got {weight_sum}"
        )
    return JudgeRubric(pass_score=pass_score, criteria=criteria)


def _default_weights_for(use_case: str) -> AxisWeights:
    ing, ret, syn = DEFAULT_AXIS_WEIGHTS.get(use_case, (0.30, 0.30, 0.40))
    return AxisWeights(ingestion=ing, retrieval=ret, synthesis=syn)


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
        raise ScenarioLoadError(
            path, f"use_case '{use_case}' must be one of {sorted(USE_CASES)}"
        )

    tier = raw.get("tier", "quick")
    if tier not in TIERS:
        raise ScenarioLoadError(path, f"tier '{tier}' must be one of {sorted(TIERS)}")

    difficulty = str(raw.get("difficulty", "easy"))
    if difficulty not in DIFFICULTIES:
        raise ScenarioLoadError(
            path, f"difficulty '{difficulty}' must be one of {list(DIFFICULTIES)}"
        )

    source_mix = str(raw.get("source_mix", "single"))
    if source_mix not in SOURCE_MIXES:
        raise ScenarioLoadError(
            path, f"source_mix '{source_mix}' must be one of {list(SOURCE_MIXES)}"
        )

    dimensions_raw = raw.get("dimensions") or []
    if not isinstance(dimensions_raw, list):
        raise ScenarioLoadError(path, "'dimensions' must be a list of dimension codes")
    dimensions = tuple(str(d) for d in dimensions_raw)
    for d in dimensions:
        if d not in DIMENSIONS:
            raise ScenarioLoadError(
                path, f"dimension '{d}' must be one of {sorted(DIMENSIONS)}"
            )

    if use_case == "COMBO" and len(dimensions) < 2:
        raise ScenarioLoadError(
            path,
            "COMBO scenarios must declare 'dimensions:' with at least 2 entries",
        )

    weights_raw = raw.get("axis_weights")
    if weights_raw:
        axis_weights = AxisWeights(
            ingestion=float(weights_raw.get("ingestion", AxisWeights.ingestion)),
            retrieval=float(weights_raw.get("retrieval", AxisWeights.retrieval)),
            synthesis=float(weights_raw.get("synthesis", AxisWeights.synthesis)),
        ).normalized()
    else:
        # Defaults are authored to sum to 1.0 exactly; skip normalize so
        # equality checks against the table don't see float dust.
        axis_weights = _default_weights_for(use_case)

    return Scenario(
        id=scenario_id,
        use_case=use_case,
        tier=tier,
        description=str(raw.get("description") or "").strip(),
        ingest=_parse_ingest(_require(raw, "ingest", path), path),
        post_ingest_assertions=_parse_post_ingest(
            raw.get("post_ingest_assertions"), path
        ),
        query=_parse_query(_require(raw, "query", path), path),
        retrieval_assertions=_parse_retrieval(raw.get("retrieval_assertions"), path),
        judge=_parse_judge(_require(raw, "judge", path), path, difficulty=difficulty),
        axis_weights=axis_weights,
        source_path=path,
        dimensions=dimensions,
        difficulty=difficulty,
        source_mix=source_mix,
        universe=(str(raw["universe"]) if raw.get("universe") else None),
        seed=_parse_seed(raw.get("seed"), path),
        distractor_events=_parse_distractors(raw.get("distractor_events"), path),
        light=bool(raw.get("light", False)),
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
        raise ScenarioLoadError(
            use_cases_dir, "scenario discovery failed:\n  " + "\n  ".join(errors)
        )
    return scenarios
