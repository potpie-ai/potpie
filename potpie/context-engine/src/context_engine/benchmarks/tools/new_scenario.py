"""Scaffold a new scenario YAML from a per-use-case template.

Hand-authoring a 50–150 line scenario YAML from scratch is the main
authoring tax on ramping the corpus. This tool emits a syntactically
correct, schema-valid stub with sensible defaults so the author edits
content rather than boilerplate. The output passes ``discover_scenarios``
on day one — runs end-to-end against the bench harness (it will fail
all assertions, of course, since the content is placeholder).

Usage::

    python -m context_engine.benchmarks.tools.new_scenario \\
        --use-case PREF \\
        --difficulty medium \\
        --id pref_pydantic_over_dataclass \\
        [--source-mix dual] [--universe acme] [--out path/to/file.yaml]

By default the file lands in this package's
``benchmarks/use_cases/<USE_CASE>/scenarios/<id>.yaml`` tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Use-case-tailored rubric templates. Each criterion sums to 100 weight
# (the loader rejects otherwise). Authors typically keep these and just
# rewrite the prompts to match their specific scenario.
_RUBRICS: dict[str, list[dict]] = {
    "PREF": [
        {
            "name": "cites_correct_preference",
            "weight": 30,
            "pass_threshold": 4,
            "prompt": "Does the answer name the specific rule and source? Vague references fail.",
        },
        {
            "name": "applies_preference_concretely",
            "weight": 30,
            "pass_threshold": 3,
            "prompt": "Does the answer translate the rule into a concrete next action?",
        },
        {
            "name": "no_invented_rules",
            "weight": 20,
            "pass_threshold": 5,
            "prompt": "Are all cited rules backed by a real source in the supporting data?",
        },
        {
            "name": "surfaces_unprompted_context",
            "weight": 20,
            "pass_threshold": 3,
            "prompt": "Does the answer also surface a related preference the user did not ask for?",
        },
    ],
    "INFRA": [
        {
            "name": "topology_correct",
            "weight": 35,
            "pass_threshold": 3,
            "prompt": "Does the answer state the correct service-to-service and service-to-storage deps?",
        },
        {
            "name": "environment_distinction",
            "weight": 25,
            "pass_threshold": 3,
            "prompt": "Does the answer distinguish prod/staging/dev where it matters?",
        },
        {
            "name": "owner_attribution",
            "weight": 20,
            "pass_threshold": 4,
            "prompt": "Does the answer name the correct owner team / lead from CODEOWNERS?",
        },
        {
            "name": "no_invented_services",
            "weight": 20,
            "pass_threshold": 5,
            "prompt": "Are all named services drawn from the canonical Acme list?",
        },
    ],
    "TIME": [
        {
            "name": "correct_chronology",
            "weight": 30,
            "pass_threshold": 3,
            "prompt": "Are the surfaced changes presented in chronological order (or dated)?",
        },
        {
            "name": "correct_window_bounds",
            "weight": 30,
            "pass_threshold": 4,
            "prompt": "Does the answer stay inside the declared time window? Distractor events outside the window must not appear.",
        },
        {
            "name": "change_attribution",
            "weight": 25,
            "pass_threshold": 3,
            "prompt": "Does the answer name PR numbers, authors, and approximate merge dates?",
        },
        {
            "name": "links_change_to_effect",
            "weight": 15,
            "pass_threshold": 2,
            "prompt": "Does the answer briefly describe what each change does functionally?",
        },
    ],
    "BUG": [
        {
            "name": "surfaces_prior_incident",
            "weight": 30,
            "pass_threshold": 4,
            "prompt": "Does the answer name the prior incident by identifier? Vague mentions do not count.",
        },
        {
            "name": "identifies_recurrence_pattern",
            "weight": 20,
            "pass_threshold": 3,
            "prompt": "Does the answer connect today's symptoms to the prior root-cause class?",
        },
        {
            "name": "cites_decision_or_policy",
            "weight": 20,
            "pass_threshold": 3,
            "prompt": "Does the answer cite the postmortem decision and note any current violation?",
        },
        {
            "name": "actionable_first_steps",
            "weight": 15,
            "pass_threshold": 3,
            "prompt": "Does the answer give concrete first actions tied to the prior fix?",
        },
        {
            "name": "no_hallucination",
            "weight": 15,
            "pass_threshold": 5,
            "prompt": "Are all factual claims grounded in the retrieved facts? Invented PRs/services fail.",
        },
    ],
    "COMBO": [
        {
            "name": "primary_dimension_a",
            "weight": 35,
            "pass_threshold": 3,
            "prompt": "Does the answer satisfy the primary criterion of the first declared dimension?",
        },
        {
            "name": "primary_dimension_b",
            "weight": 35,
            "pass_threshold": 3,
            "prompt": "Does the answer satisfy the primary criterion of the second declared dimension?",
        },
        {
            "name": "no_hallucination",
            "weight": 15,
            "pass_threshold": 5,
            "prompt": "Are all factual claims grounded in the retrieved facts?",
        },
        {
            "name": "integrates_dimensions",
            "weight": 15,
            "pass_threshold": 3,
            "prompt": "Does the answer feel like one coherent plan rather than two glued lists?",
        },
    ],
}


_INTENT_BY_USE_CASE = {
    "PREF": "feature",
    "INFRA": "debugging",
    "TIME": "review",
    "BUG": "debugging",
    "COMBO": "feature",
}


def _render_rubric(use_case: str) -> str:
    """Render the rubric criteria block for a use case.

    For non-COMBO use cases each criterion is tagged with the parent
    use case so reports' ``by_dimension`` rolls up correctly. COMBO
    scenarios get placeholder ``dimensions: [PREF]`` on the two
    primary criteria so the author edits one line per criterion to
    match the actual exercise.
    """
    rubric = _RUBRICS[use_case]
    lines = []
    for c in rubric:
        lines.append(f"    - name: {c['name']}")
        lines.append(f"      weight: {c['weight']}")
        lines.append(f"      pass_threshold: {c['pass_threshold']}")
        if use_case == "COMBO" and c["name"] in {
            "primary_dimension_a",
            "primary_dimension_b",
        }:
            # Author fills in the actual dimension at edit time.
            lines.append("      dimensions: [PREF]")
        elif use_case in {"PREF", "INFRA", "TIME", "BUG"}:
            lines.append(f"      dimensions: [{use_case}]")
        prompt = c["prompt"].replace("\n", "\n        ")
        lines.append("      prompt: |")
        lines.append(f"        {prompt}")
    return "\n".join(lines)


def render_stub(
    *,
    scenario_id: str,
    use_case: str,
    difficulty: str,
    source_mix: str,
    universe: str | None,
    dimensions: list[str] | None,
) -> str:
    rubric_block = _render_rubric(use_case)
    intent = _INTENT_BY_USE_CASE[use_case]
    dims_line = ""
    if use_case == "COMBO":
        dims = dimensions or ["PREF", "INFRA"]
        dims_line = f"dimensions: {dims}\n"

    universe_block = f"universe: {universe}\n" if universe else ""

    return f"""\
id: {scenario_id}
use_case: {use_case}
tier: quick
difficulty: {difficulty}
source_mix: {source_mix}
{universe_block}{dims_line}
description: |
  TODO — one paragraph describing the agent task and why it is hard.
  Reference specific persons (bench-user-N), services (checkout-api,
  inventory-svc, ...), and the canonical universe entities. State what
  the engine should surface and what it should NOT surface.

ingest:
  # TODO — replace with real signal envelopes. At least one is required.
  - {{ event: linear/REPLACE_ME.json, at: "-7d", tags: [signal] }}

# distractor_events:                # uncomment to inject noise (precision side)
#   - {{ event: linear/noise_*.json, at: "-21d..-7d", count: 12 }}

post_ingest_assertions:
  graph_must_contain_entities:
    # TODO — at least one structural assertion the engine must satisfy.
    - {{ label: Entity, min_count: 1 }}
  reconciliation:
    soft_downgrades_max: 3
    failed_events_max: 0

query:
  intent: {intent}
  scope: {{}}                       # TODO — narrow the scope (e.g. service, repo, window)
  include: [docs, owners]           # TODO — pick the include keys this scenario needs
  mode: balanced
  source_policy: summary

retrieval_assertions:
  source_refs_min: 1
  must_cite_event_id:
    - linear/REPLACE_ME.json
  # must_not_cite_event_id:         # uncomment to mark distractors that must not appear
  #   - linear/noise_*.json

judge:
  # pass_score is omitted on purpose — the loader picks
  # easy=75 / medium=65 / hard=55 / adversarial=45.
  criteria:
{rubric_block}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        "new_scenario",
        description="Scaffold a new bench scenario YAML.",
    )
    parser.add_argument(
        "--use-case", required=True, choices=("PREF", "INFRA", "TIME", "BUG", "COMBO")
    )
    parser.add_argument(
        "--id", required=True, help="Scenario id (lowercase snake_case)."
    )
    parser.add_argument(
        "--difficulty",
        default="easy",
        choices=("easy", "medium", "hard", "adversarial"),
    )
    parser.add_argument(
        "--source-mix",
        default="single",
        choices=("single", "dual", "full", "adversarial"),
    )
    parser.add_argument(
        "--universe", default="acme", help="Universe to seed (or '' for none)."
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated dimensions for COMBO scenarios (e.g. PREF,INFRA). Ignored otherwise.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output path. Default: this package's "
            "benchmarks/use_cases/<USE_CASE>/scenarios/<id>.yaml"
        ),
    )
    args = parser.parse_args(argv)

    use_case = args.use_case
    if use_case == "COMBO":
        if not args.dimensions:
            print(
                "error: --dimensions is required for COMBO scenarios (e.g. --dimensions PREF,INFRA)",
                file=sys.stderr,
            )
            return 2
        dims = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        if len(dims) < 2:
            print(
                "error: COMBO scenarios must declare >= 2 dimensions", file=sys.stderr
            )
            return 2
    else:
        dims = None

    universe = args.universe or None
    body = render_stub(
        scenario_id=args.id,
        use_case=use_case,
        difficulty=args.difficulty,
        source_mix=args.source_mix,
        universe=universe,
        dimensions=dims,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        bench_root = Path(__file__).resolve().parents[1]
        out_path = bench_root / "use_cases" / use_case / "scenarios" / f"{args.id}.yaml"

    if out_path.exists():
        print(
            f"error: {out_path} already exists — refusing to overwrite", file=sys.stderr
        )
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
