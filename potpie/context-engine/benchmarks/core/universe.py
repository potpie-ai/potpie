"""Canonical-universe seed loading.

A *universe* is a stable synthetic organisation that scenarios share.
The bench's canonical universe is "Acme Corp" (bench-plan §5.1):
5 backend services, 1 frontend, 1 data service, 3 environments, ~20
personas, a small set of ADRs and runbooks.

The seed is a directory tree under
``benchmarks/universe/<name>/raw_events/...`` that contains
fixture envelopes in exactly the same shape as scenario-specific
fixtures. When a scenario declares ``universe: acme``, the runner
ingests every file in that tree at ``-365d`` before signal events.

Keeping the seed in the same envelope shape — instead of a bespoke
"universe descriptor" — means the engine receives canonical events for
every Acme entity, exercising the real reconciliation path. There is
no shortcut into the graph: even seed data is reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.core.scenario import SeedStep


@dataclass(frozen=True)
class Universe:
    name: str
    root: Path  # absolute path to the universe directory
    seeds: tuple[SeedStep, ...]


def discover_universe(benchmarks_root: Path, name: str) -> Universe:
    """Walk a universe fixture tree and return ordered seed steps.

    Seed envelopes live alongside other fixtures so the existing
    fixture loader handles them without special-casing — concretely,
    at ``benchmarks/fixtures/raw_events/universe/<name>/...``. The
    universe dir name is just the path prefix; ``load_envelope`` takes
    care of the rest.

    Files are sorted by name so the order is deterministic. Authors
    use digit-prefixed names like ``00-architecture.json``,
    ``10-team.json`` to control ingestion order without resorting to
    nested directories. Events come pre-anchored at ``-365d``; the
    scenario can override per-file via an explicit ``seed:`` entry.
    """
    fixtures_universe = benchmarks_root / "fixtures" / "raw_events" / "universe" / name
    if not fixtures_universe.exists():
        raise FileNotFoundError(
            f"universe '{name}' has no fixture tree at {fixtures_universe}"
        )

    seeds: list[SeedStep] = []
    for path in sorted(fixtures_universe.rglob("*.json")):
        rel = path.relative_to(fixtures_universe)
        seeds.append(SeedStep(event=f"universe/{name}/{rel.as_posix()}", at="-365d"))
    return Universe(name=name, root=fixtures_universe, seeds=tuple(seeds))


def resolve_seeds_for_scenario(
    benchmarks_root: Path,
    universe_name: str | None,
    explicit_seeds: tuple[SeedStep, ...],
) -> tuple[SeedStep, ...]:
    """Combine universe-derived seeds with any explicit ``seed:`` entries.

    Explicit seeds override universe seeds with the same ``event``
    fixture path (so a scenario can pin a specific seed to a non-default
    timestamp).
    """
    if not universe_name and not explicit_seeds:
        return ()
    if not universe_name:
        return explicit_seeds
    universe = discover_universe(benchmarks_root, universe_name)
    overrides = {s.event: s for s in explicit_seeds}
    merged = [overrides.get(s.event, s) for s in universe.seeds]
    # Append any explicit seeds that don't shadow a universe entry.
    seen = {s.event for s in merged}
    for s in explicit_seeds:
        if s.event not in seen:
            merged.append(s)
    return tuple(merged)
