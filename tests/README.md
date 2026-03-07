# Test Suite

All tests live under this directory. See `docs/test-suite-and-parsing-plan.md` for the full plan.

## One command (recommended)

From the project root, one command runs the full suite in order (unit → integration → real_parse → optional stress):

```bash
./scripts/run_tests.sh
# or
uv run python scripts/run_tests.py
```

- **Modular:** The runner uses pytest discovery and markers only. New tests under `tests/unit/` or `tests/integration-tests/` are picked up automatically; no changes to the script are needed.
- **Phases:** Unit → Integration (excluding stress/real_parse) → Real parse (optional) → Stress (optional). First failure stops the run.
- **Environment (optional):**
  - `SKIP_REAL_PARSE=1` — Skip real_parse phase (e.g. CI without Neo4j).
  - `RUN_STRESS=1` — Include stress tests after the main phases.

**Subset runs (same script):**

```bash
./scripts/run_tests.sh --unit-only
./scripts/run_tests.sh --integration-only
./scripts/run_tests.sh --real-parse-only
./scripts/run_tests.sh --stress-only
```

**Pass extra args to pytest:** Use `--` to separate script options from pytest:

```bash
./scripts/run_tests.sh -- -x -k "test_parsing"
./scripts/run_tests.sh --unit-only -- -x
```

**CI / automation:** Use the same script. Example without Neo4j:

```bash
SKIP_REAL_PARSE=1 ./scripts/run_tests.sh
```

To add new tests later: add test files under `tests/unit/` or `tests/integration-tests/` with the appropriate marker (`unit`, `integration`, or `real_parse`). No changes to `scripts/run_tests.py` or `scripts/run_tests.sh` are required.

## Output and debugging

- **Phase banners:** When you run the full suite, each phase (Unit, Integration, etc.) is wrapped in clear `====` banners and ends with a `Phase «…» finished: OK/FAILED` line so you can quickly see which phase failed.
- **Short summary (`-ra`):** Pytest is run with `-ra`, so you get a compact summary of all outcomes (failed, skipped, xfailed, etc.) at the end of each phase, not only when something fails.
- **Slow tests (`--durations=5`):** The 5 slowest tests are listed at the end of each run so you can spot slow or flaky tests.
- **Verbose (`-v`):** Each test is printed with its full node id (e.g. `tests/unit/parsing/test_parsing_schema.py::TestParsingRequest::test_valid_with_repo_name_only`), making it easy to re-run a single test:  
  `uv run pytest tests/unit/parsing/test_parsing_schema.py::TestParsingRequest::test_valid_with_repo_name_only -v`

To focus on a single failure, use `-x` (stop on first failure) and/or `-k 'test_name_pattern'`:

```bash
./scripts/run_tests.sh -- -x -k "test_parse"
```

## Structure

- **`unit/`** — Unit tests (schema, utils, helpers with mocks). Marked with `@pytest.mark.unit`.
- **`integration-tests/`** — Integration tests (API, DB, mocked external services).

## Running tests (direct pytest)

From the project root:

```bash
# All tests (requires Postgres; see .env POSTGRES_SERVER)
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/unit/ -v -m unit

# Parsing unit tests only
uv run pytest tests/unit/parsing/ -v -m unit

# Auth unit tests only
uv run pytest tests/unit/auth/ -v -m unit

# Integration tests (exclude stress and real_parse)
uv run pytest tests/integration-tests/ -v -m "not stress and not real_parse and not github_live"

# Real parse test (requires Postgres + Neo4j)
uv run pytest tests/integration-tests/parsing/test_real_parse.py -v -m real_parse
```

**Note:** Tests that need the database (e.g. `db_session`, auth models, `ParseHelper` with DB) depend on `setup_test_database`, which requires Postgres. If `POSTGRES_SERVER` is not set or Postgres is unreachable, those tests are **skipped** (no failure). Pure unit tests (e.g. parsing schema, content_hash, repo_name_normalizer, encoding_detector, validator) run without Postgres.

## Real Parse Test

The `test_real_parse.py` test runs `ParsingService.parse_directory` with:
- **Real Postgres** — requires `POSTGRES_SERVER` env var
- **Real Neo4j** — requires `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` env vars
- **Local test repo** — a small Python project created as a fixture (no GitHub clone)

This test catches regressions in core parsing logic that mocked tests miss. If Postgres or Neo4j is unavailable, the test is **skipped**.

## Markers

- `unit` — Unit tests
- `integration` — Integration tests
- `asyncio` — Async tests (asyncio_mode=auto in pyproject)
- `github_live` — Tests that call the live GitHub API (require `GH_TOKEN_LIST`; use `@pytest.mark.usefixtures("require_github_tokens")` in the test module)
- `stress` — Stress tests (exclude with `-m "not stress"`)
- `real_parse` — Real parsing tests requiring Postgres + Neo4j (exclude with `-m "not real_parse"`)
