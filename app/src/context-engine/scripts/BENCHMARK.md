# Context Graph Benchmark

A comprehensive benchmark harness for the Potpie context graph, implemented as the `benchmarks` package under `app/src/context-engine/benchmarks/`.

## Modes

| Mode | Description | Use When |
|------|-------------|----------|
| `mock` | In-process mock intelligence provider | Fastest baseline; no I/O |
| `http-e2e` | In-process FastAPI test client | Router-level latency without network |
| `api` | Live Potpie server at `POTPIE_API_URL` | Real-world end-to-end benchmarking |

## Quick Start

```bash
# From the context-engine package root
cd app/src/context-engine

# Mock mode (fastest, no server)
uv run python -m benchmarks.cli mock

# HTTP-E2E mode (in-process router)
uv run python -m benchmarks.cli http-e2e

# API mode (live server; needs POTPIE_API_KEY)
uv run python -m benchmarks.cli api

# Or via the convenience wrapper
uv run python scripts/benchmark_context_engine.py mock
```

## Scenarios

The benchmark exercises the full context graph surface:

### Seeding
- **Episodes** — seed episodic memory (decisions, workflows, debugging, runbooks, preferences)
- **Records** — seed structured records (decision, fix, workflow)
- **PR bundles** — ingest pull request fixtures as episodes (or live PRs with `--ingest-pr-live`)

### Query Scenarios
- **Agent context recipes** — `feature`, `debugging`, `review`, `operations` intents
- **Semantic search** — semantic retrieval with varying limits
- **Readiness / quality checks** — status and recipe validation

Each scenario carries weighted assertions:

| Assertion | Weight | Description |
|-----------|--------|-------------|
| `must_contain` | 1.0 | Keyword must appear in flattened response |
| `required_paths` | 1.0 | JSON path must exist in response |
| `coverage` | 0.75 | Coverage sub-field must match expected value |
| `quality` | 0.5 | Quality sub-field must match expected value |
| `min_source_refs` | 1.0 | At least N source references |
| `max_fallbacks` | 0.75 | At most N fallback items |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `benchmarks/data/context_graph_benchmark_dataset.json` | Dataset JSON path |
| `--report` | `.tmp/context-graph-benchmark-report.json` | Output path for JSON report |
| `--baseline` | — | Previous report for regression checks |
| `--pot-id` | auto | Pot UUID. API mode auto-detects from server list if omitted |
| `--repo-name` | — | Scope filter for queries and seeding |
| `--iterations` / `-i` | 3 | Iterations per scenario |
| `--concurrency` / `-c` | 4 | Max concurrent scenario requests |
| `--no-seed` | — | Skip fixture seeding |
| `--ingest-pr-live` | — | Call `/ingest-pr` for PR fixtures instead of seeding episodes |
| `--print-json` | — | Print full JSON report to stdout |

## Example: Compare API vs Mock

```bash
uv run python -m benchmarks.cli api  --iterations 5 --report .tmp/bench-api.json
uv run python -m benchmarks.cli mock --iterations 5 --report .tmp/bench-mock.json

# Diff key metrics
uv run python -c "
import json
for path, label in [('.tmp/bench-api.json','API'),('.tmp/bench-mock.json','Mock')]:
    d = json.load(open(path))
    print(f'--- {label} ---')
    for s in d['scenarios']:
        lat = s.get('latency_ms',{})
        ratio = s['score']/s['max_score'] if s['max_score'] else 0
        print(f\"{s['id']:<35} score={ratio:>6.1%} p95={lat.get('p95'):>8.2f}ms errs={s['errors']}\")
"
```

## Interpreting Results

The report JSON contains:

```json
{
  "dataset": {"name": "...", "version": "..."},
  "target": {"mode": "api", "pot_id": "...", "iterations": 3, "concurrency": 4},
  "summary": {
    "ok": true,
    "score": 0.9184,
    "grade": "excellent",
    "scenario_count": 7,
    "error_count": 0,
    "pass_score": 0.72
  },
  "scenarios": [
    {
      "id": "agent_feature_recipe",
      "feature": "agent_context_port",
      "grade": "excellent",
      "score": 5.5,
      "max_score": 5.5,
      "errors": 0,
      "latency_ms": {"min": 120, "p50": 145, "p95": 180, "max": 200},
      "assertions": [...],
      "response_summary": {...}
    }
  ],
  "regressions": []
}
```

### Grades

| Grade | Threshold | Meaning |
|-------|-----------|---------|
| `excellent` | ≥ 90 % | All critical assertions pass |
| `good` | ≥ 75 % | Acceptable for production |
| `watch` | ≥ 60 % | Needs attention |
| `regressed` | < 60 % | Blocking issue |

### Regressions

When `--baseline` is provided, the benchmark flags:
- **score_drop** — scenario score dropped more than `thresholds.max_score_drop` (default 5 %)
- **latency_p95** — p95 latency grew more than `thresholds.max_p95_latency_ratio` (default 3×)

## Known Limitations

- **HTTP-E2E record paths**: The in-process router does not wire a live reconciliation agent, so `seed_record` returns skipped. Episodes and queries work fully.
- **API seed errors**: If the target pot does not contain the configured `repo_name`, record seeding returns `repo_not_in_pot`. This is expected and does not block the benchmark.
- **LLM variance**: API mode uses live LLM extraction; exact keyword matches (`must_contain`) may vary between runs. The dataset is tuned for stable extraction, but occasional variance is normal.
