# Test Suite and Parsing Tests — Comprehensive Plan

This document describes the overall test suite structure, migration steps, and a detailed plan for unit and integration tests for the parsing module. **No code changes are specified**—only structure, scope, and test-case planning.

---

## Part 1: Overall Test Suite Structure

### 1.1 Goals

- **Single root test tree:** All tests live under `tests/` at the repository root.
- **Single conftest:** One `tests/conftest.py` provides all shared fixtures (DB, client, mocks). No module-level conftest files.
- **Clear separation:** Unit tests (fast, isolated, no real external services) vs integration tests (real DB, mocked external services).
- **Uniform DB:** All tests use the same Postgres test database from the root conftest (no in-memory SQLite for auth or parsing).
- **Discoverable and runnable:** Pytest markers (`unit`, `integration`, etc.) and consistent paths so CI and developers can run subsets (e.g. `pytest -m unit`, `pytest tests/unit/parsing/`).

### 1.2 Directory Layout (Target State)

```
tests/
├── conftest.py                    # Single source of fixtures (DB, client, auth, mocks)
├── unit/
│   ├── auth/                      # Migrated from app/modules/auth/tests/
│   │   ├── test_models.py
│   │   ├── test_unified_auth_service.py
│   │   └── test_auth_router.py
│   └── parsing/
│       ├── test_parsing_schema.py
│       ├── test_parsing_validator.py
│       ├── test_content_hash.py
│       ├── test_repo_name_normalizer.py
│       ├── test_encoding_detector.py
│       └── test_parsing_helper.py
└── integration_tests/             # Or keep existing name: integration-tests
    ├── parsing/
    │   ├── test_parsing_router.py
    │   └── test_parsing_service.py  # Optional: service with mocked Neo4j/helper
    ├── code_provider/
    ├── conversations/
    └── stress/
```

### 1.3 Pytest Configuration

- **testpaths:** Include both unit and integration roots (e.g. `tests/unit`, `tests/integration_tests`) or a single `tests/` so that `pytest` discovers everything. Alternatively keep explicit paths and run via `pytest tests/unit/` or `pytest tests/integration_tests/`.
- **Markers:** Ensure `unit`, `integration`, `asyncio`, and any existing markers (`github_live`, `stress`) are registered in `pyproject.toml`. All new parsing tests should use `@pytest.mark.unit` or `@pytest.mark.integration` as appropriate.
- **asyncio_mode:** Remain `auto` for async tests.

### 1.4 Migration Steps (Existing Tests)

1. **Create `tests/unit/`** (and `tests/unit/auth/`, `tests/unit/parsing/`) if they do not exist.
2. **Move auth tests** from `app/modules/auth/tests/` to `tests/unit/auth/`:
   - Move `test_models.py`, `test_unified_auth_service.py`, `test_auth_router.py` (and any other `test_*.py`).
   - Do **not** move or keep `app/modules/auth/tests/conftest.py`; delete it after migration.
3. **Add auth fixtures to root conftest:** In `tests/conftest.py`, add fixtures that auth tests currently rely on: `test_user`, `test_user_with_github`, `test_user_with_multiple_providers`, `pending_link`, `org_sso_config`. Implement these using the existing `db_session` (Postgres) and the same model classes. Ensure fixture names match so auth tests do not need code changes beyond imports (if any).
4. **Update imports in auth tests:** Change any imports that assume the auth package layout (e.g. relative or package-relative) to use the app layout (e.g. `from app.modules.auth...`). Ensure tests run with project root in `sys.path` (already the case if pytest is run from repo root).
5. **Run auth tests:** Execute `pytest tests/unit/auth/ -v` and fix any failures due to fixture differences (e.g. SQLite vs Postgres column types or behavior).
6. **Optional:** Add a `tests/integration_tests/auth/` and move any auth tests that are better classified as integration (e.g. full router with client) there, or keep them under `unit` if they only use DB and mocks.
7. **Remove** `app/modules/auth/tests/` directory and update `pyproject.toml` `testpaths` if it currently points to `app/modules/auth/tests`.

### 1.5 Conventions

- **Naming:** Test files `test_<module_or_feature>.py`; test classes `Test<Component>`; test functions `test_<scenario>_<expected>`.
- **Isolation:** Each test should be independent; use fixtures for setup. No reliance on execution order.
- **Markers:** Tag every test with `@pytest.mark.unit` or `@pytest.mark.integration`. Use `@pytest.mark.asyncio` for async tests (or rely on `asyncio_mode = "auto"`).
- **Fixtures:** Only define fixtures in `tests/conftest.py` unless a subdirectory truly needs a local conftest (avoid for now to keep “single conftest”).
- **External boundaries:** In integration tests, mock only external systems: Celery, Neo4j, GitHub API, RepoManager (when testing controller/service). Use real DB and real app dependencies (e.g. ProjectService, AuthService) where appropriate.

### 1.6 Test automation (one command)

A single entry point runs the full suite and is used by developers and CI:

- **Script:** `scripts/run_tests.py` (or `./scripts/run_tests.sh`). Runs phases: unit → integration (excluding stress/real_parse) → real_parse (optional) → stress (optional).
- **Modular:** Uses pytest discovery and markers only; no test file paths in the script. New tests under `tests/unit/` or `tests/integration-tests/` are picked up automatically.
- **Env:** `SKIP_REAL_PARSE=1` to skip real_parse (e.g. CI without Neo4j); `RUN_STRESS=1` to include stress.
- **Subsets:** `--unit-only`, `--integration-only`, `--real-parse-only`, `--stress-only`. See `tests/README.md`.

---

## Part 2: Parsing Unit Tests — Detailed Plan

### 2.1 Scope of “Unit” for Parsing

- **In scope:** Schema validation, validator decorator, pure utils (content_hash, repo_name_normalizer, encoding_detector), and ParseHelper methods that can be tested in isolation with temp dirs or mocks (e.g. `get_directory_size`, `detect_repo_language`, `is_text_file`, `check_commit_status` with mocked ProjectService/GitHub).
- **Out of scope for unit:** Full ParsingService.parse_directory (too many dependencies; cover in integration with mocks), Celery task execution (integration), real Neo4j/Git/RepoManager (integration or E2E).

### 2.2 test_parsing_schema.py

| Test case | Description |
|-----------|-------------|
| ParsingRequest valid with repo_name only | Instantiate with `repo_name="owner/repo"`; no error; attribute access works. |
| ParsingRequest valid with repo_path only | Instantiate with `repo_path="/some/path"`; no error. |
| ParsingRequest valid with both repo_name and repo_path | Both set; no error. |
| ParsingRequest invalid when both missing | Neither repo_name nor repo_path (or both None); expect `ValueError`. |
| ParsingRequest invalid with empty strings | `repo_name=""`, `repo_path=None` — confirm behavior (e.g. ValueError or downstream handling). |
| RepoDetails construction | Valid RepoDetails with required fields; optional repo_path and commit_id. |
| ParsingStatusRequest required repo_name | Must have repo_name; optional commit_id and branch_name. |

### 2.3 test_parsing_validator.py

| Test case | Description |
|-----------|-------------|
| Validator allows request when dev mode and repo_path | Set env `isDevelopmentMode=enabled`; call wrapped handler with repo_details.repo_path set; no 403. |
| Validator returns 403 when repo_path and non-dev | Set env `isDevelopmentMode` not enabled; repo_details.repo_path set; expect HTTPException 403 with message about development mode. |
| Validator returns 403 when default user and repo_name | Set env `defaultUsername` to a value; pass user dict with user_id matching it and repo_details.repo_name set; expect 403 “Cannot parse remote repository without auth token”. (Requires decorator to receive user_id from `user`; document as “after fix” if validator is currently broken.) |
| Validator allows request when user_id not default | user_id different from defaultUsername; repo_name set; no 403. |
| Validator passes through when repo_details or user missing | Call with kwargs missing repo_details or user; wrapped handler is invoked (no 403). |

**Note:** Current validator reads `user_id` from `kwargs`; controller passes `user` dict. Plan should note that either (a) validator is fixed to derive user_id from `user`, or (b) tests document current (broken) behavior until fixed.

### 2.4 test_content_hash.py

| Test case | Description |
|-----------|-------------|
| generate_content_hash deterministic | Same input (and optional node_type) yields same hash. |
| generate_content_hash with node_type | Include node_type; hash differs from same content without node_type. |
| generate_content_hash whitespace normalized | Content with different whitespace but same normalized form yields same hash. |
| generate_content_hash empty string | Empty string produces valid hex hash. |
| generate_content_hash None input | None or non-string input raises (e.g. AttributeError/TypeError). |
| has_unresolved_references exact phrase | Text containing exact substring “Code replaced for brevity. See node_id” returns True. |
| has_unresolved_references partial or variant | Text with “node_id” only or different wording returns False. |
| is_content_cacheable short content | Content shorter than min_length returns False. |
| is_content_cacheable with unresolved refs | Content with unresolved references returns False. |
| is_content_cacheable repetitive lines | Content with &lt; 30% unique lines returns False. |
| is_content_cacheable long unique no refs | Long content, no refs, sufficient unique lines returns True. |
| is_content_cacheable boundary 30% unique | Exactly 30% unique lines; assert expected True/False per implementation. |

### 2.5 test_repo_name_normalizer.py

| Test case | Description |
|-----------|-------------|
| normalize_repo_name None or empty | Returns input as-is. |
| normalize_repo_name no slash | e.g. `"noproject"` returns as-is. |
| normalize_repo_name github style | `"owner/repo"` with provider github returns as-is. |
| normalize_repo_name gitbucket root with username | `"root/repo"` with CODE_PROVIDER=gitbucket and GITBUCKET_USERNAME set returns `"<username>/repo"`. |
| normalize_repo_name gitbucket root no username | GITBUCKET_USERNAME unset; `"root/repo"` returns as-is. |
| get_actual_repo_name_for_lookup github | Returns repo_name as-is for github. |
| get_actual_repo_name_for_lookup gitbucket username format | Repo name already `"username/repo"` with GITBUCKET_USERNAME set; returns as-is. |
| get_actual_repo_name_for_lookup gitbucket root format | `"root/repo"` returns as-is. |
| get_actual_repo_name_for_lookup gitbucket fallback to root | Username set but repo_name not username-prefixed; assert correct fallback (e.g. `"root/repo"`). |

Use `monkeypatch` or env context for CODE_PROVIDER and GITBUCKET_USERNAME where needed.

### 2.6 test_encoding_detector.py

| Test case | Description |
|-----------|-------------|
| detect_encoding utf8 file | Temp file with UTF-8 content; returns `"utf-8"`. |
| detect_encoding utf16 file | Temp file with UTF-16 content; returns appropriate encoding from list. |
| detect_encoding file not found | Non-existent path; returns None or raises; document behavior. |
| detect_encoding directory path | Path is a directory; returns None (or raises). |
| read_file success | Temp file; returns (content, encoding). |
| read_file file not found | Returns (None, None) or equivalent. |
| is_text_file success | Temp text file; returns True. |
| is_text_file binary | Temp binary file; returns False. |

### 2.7 test_parsing_helper.py

| Test case | Description |
|-----------|-------------|
| get_directory_size empty dir | Temp empty directory; returns 0. |
| get_directory_size with files | Temp dir with known-size files; returns sum (symlinks excluded). |
| get_directory_size with symlink | Dir with symlink to file; symlink size not counted. |
| detect_repo_language directory missing | Non-existent path; returns `"other"`. |
| detect_repo_language not a directory | Path is file; returns `"other"`. |
| detect_repo_language python only | Temp dir with only .py files; returns `"python"`. |
| detect_repo_language no supported files | Dir with only unsupported extensions; returns `"other"`. |
| detect_repo_language mixed | Dir with .py and .js; returns predominant per implementation. |
| is_text_file utf8 | Temp UTF-8 file; returns True. |
| is_text_file binary | Temp binary; returns False. |
| check_commit_status no project | Mock ProjectService.get_project_from_db_by_id to return None; expect False. |
| check_commit_status pinned commit match | Project has commit_id; requested_commit_id same; expect True. |
| check_commit_status pinned commit mismatch | Requested != stored; expect False. |
| check_commit_status branch-based match | No requested_commit_id; mock GitHub to return same commit as stored; expect True. |
| check_commit_status branch-based mismatch | Mock GitHub to return different commit; expect False. |
| check_commit_status GitHub exception | Mock get_repo/get_branch to raise; expect False. |

For ParseHelper tests that need DB or GitHub, use fixtures from root conftest (db_session) and mocks (e.g. patch ProjectService or CodeProviderService). Prefer unit-style mocks rather than real DB for pure “logic” coverage of check_commit_status.

---

## Part 3: Parsing Integration Tests — Plan

### 3.1 test_parsing_router.py (HTTP API)

| Test case | Description |
|-----------|-------------|
| POST /parse valid repo_name | Mock Celery process_parsing.delay; valid body with repo_name; expect 200 and task submitted; project created or existing returned. |
| POST /parse invalid body | Missing both repo_name and repo_path; expect 422. |
| POST /parse repo_path without dev mode | isDevelopmentMode not enabled; repo_path in body; expect 403. |
| GET /parsing-status/{project_id} found | Project exists and user owns it; expect 200 with status and latest. |
| GET /parsing-status/{project_id} not found | Unknown project_id or no access; expect 404. |
| POST /parsing-status by repo | Valid repo_name (+ branch/commit); project exists; expect 200 with project_id, status, latest. |
| POST /parsing-status by repo not found | No project for given repo/branch/commit; expect 404. |

Use root conftest `client`, `db_session`, and fixtures (e.g. project, user). Mock Celery and optionally Neo4j/GitHub where needed so tests do not require real brokers or external APIs.

### 3.2 test_parsing_service.py (Optional)

| Test case | Description |
|-----------|-------------|
| parse_directory project already INFERRING | Project status INFERRING; expect early return with message, no clone/analyze. |
| parse_directory project READY and commit matches | Mock check_commit_status True; expect early return, no clone. |
| parse_directory cleanup_graph failure | Mock CodeGraphService.cleanup_graph to raise; expect status ERROR and 500 or ParsingServiceError. |
| parse_directory setup_project_directory returns None | Mock ParseHelper.setup_project_directory to return (None, id); expect 500 or ParsingServiceError. |
| analyze_directory invalid extracted_dir type | Pass non-string; expect ValueError. |
| analyze_directory directory missing | Path does not exist; expect FileNotFoundError. |
| analyze_directory project not in DB | get_project_from_db_by_id returns None; expect 404 or ParsingServiceError. |
| analyze_directory language other | Mock or set up so language is "other"; expect ParsingFailedError and status ERROR. |

Heavy mocking of ParseHelper, CodeGraphService, InferenceService, and ProjectService so that no real Neo4j/Git/RepoManager is required.

---

## Part 4: Edge Cases and Gaps to Cover (Checklist)

Use this as a checklist when implementing tests; not every item need be a separate test if covered by an existing case.

### 4.1 Concurrency and races ✅ IMPLEMENTED

- [x] Double-submit same repo+branch: only one project or idempotent behavior.
  - `test_parsing_router.py::TestConcurrencyAndRaces::test_double_submit_same_repo_idempotent`
- [ ] Project status race (INFERRING set by another task): controller or task handles gracefully.
  - Covered by `test_parse_directory_project_inferring_early_return`

### 4.2 Input validation ✅ IMPLEMENTED

- [x] `repo_name` whitespace-only.
  - `test_parsing_schema.py::TestInputEdgeCases::test_repo_name_whitespace_only`
  - `test_parsing_router.py::TestInputValidationEdgeCases::test_post_parse_whitespace_only_repo_name`
- [x] `commit_id` empty or invalid.
  - `test_parsing_schema.py::TestInputEdgeCases::test_empty_commit_id_string`
  - `test_parsing_router.py::TestInputValidationEdgeCases::test_post_parse_empty_commit_id`
- [x] `repo_name` with multiple slashes or no slash (expect 404 or validation).
  - `test_parsing_schema.py::TestInputEdgeCases::test_repo_name_no_slash`
  - `test_parsing_schema.py::TestInputEdgeCases::test_repo_name_multiple_slashes`
  - `test_parsing_router.py::TestInputValidationEdgeCases::test_post_parse_repo_name_no_slash`
  - `test_parsing_router.py::TestInputValidationEdgeCases::test_post_parse_repo_name_multiple_slashes`
- [x] Path auto-detection when repo_name looks like a path (e.g. `/tmp/repo`, `./repo`).
  - `test_parsing_router.py::TestInputValidationEdgeCases::test_post_parse_path_like_repo_name`
  - `test_parsing_router.py::TestInputValidationEdgeCases::test_post_parse_relative_path_repo_name`
- [x] Additional edge cases:
  - `test_parsing_schema.py::TestInputEdgeCases::test_repo_name_leading_trailing_spaces`
  - `test_parsing_schema.py::TestInputEdgeCases::test_very_long_repo_name`
  - `test_parsing_schema.py::TestInputEdgeCases::test_special_characters_in_repo_name`
  - `test_parsing_schema.py::TestInputEdgeCases::test_unicode_in_repo_name`

### 4.3 RepoManager / worktree

- [ ] Worktree creation failure (e.g. mock subprocess failure).
- [ ] Worktree path missing but bare repo exists (recreate path).
- [ ] RepoManager disabled vs enabled (clone path differences).

### 4.4 Neo4j and graph ✅ IMPLEMENTED

- [x] Neo4j connection failure → project status ERROR.
  - `test_parsing_service.py::TestNeo4jFailures::test_neo4j_connection_failure_on_cleanup`
- [x] cleanup_graph partial failure.
  - `test_parsing_service.py::TestParseDirectory::test_parse_directory_cleanup_graph_failure`
- [ ] Empty directory → 0 nodes, no crash.
- [ ] duplicate_graph with mocked driver (and optionally APOC).

### 4.5 Inference and cache ✅ IMPLEMENTED

- [x] LLM/provider failure or timeout → status ERROR.
  - `test_parsing_service.py::TestNeo4jFailures::test_inference_service_failure`
- [ ] Embedding model load failure.
- [ ] InferenceCacheService init failure → inference proceeds without cache.

### 4.6 ProjectService ✅ IMPLEMENTED

- [x] register_project with existing project_id but different user_id → 403.
  - `test_parsing_service.py::TestProjectServiceOwnership::test_register_project_different_user_403`
- [x] get_project_from_db with branch_name=None, commit_id=None.
  - `test_parsing_service.py::TestProjectServiceOwnership::test_get_project_with_no_branch_no_commit`
- [x] update_project with non-existent project_id.
  - `test_parsing_service.py::TestProjectServiceOwnership::test_update_nonexistent_project`

### 4.7 Celery and session

- [ ] Task receives valid payload; session from BaseTask works.
- [ ] Transient failure (no retry) — document or test.

### 4.8 Type and contract bugs

- [ ] fetch_parsing_status_by_repo: controller type hint AsyncSession vs sync Session from router; ensure test passes and document fix if needed.

---

## Part 5: Implementation Order (Recommended)

1. **Phase 1 — Structure**
   - Create `tests/unit/` and `tests/unit/auth/`, `tests/unit/parsing/`.
   - Add auth fixtures to root conftest.
   - Move auth tests and remove auth conftest; fix imports and run auth tests.

2. **Phase 2 — Parsing unit (no DB)**
   - test_parsing_schema.py
   - test_content_hash.py
   - test_repo_name_normalizer.py
   - test_encoding_detector.py
   - test_parsing_validator.py

3. **Phase 3 — Parsing unit (DB/mocks)**
   - test_parsing_helper.py (get_directory_size, detect_repo_language, is_text_file, check_commit_status with mocks).

4. **Phase 4 — Parsing integration**
   - test_parsing_router.py (POST /parse, GET/POST parsing-status).
   - Optionally test_parsing_service.py with heavy mocks.

5. **Phase 5 — Edge cases**
   - Add tests from Part 4 checklist as needed; prioritize concurrency, validator fix, Neo4j/LLM failure, and ProjectService ownership.

6. **Phase 6 — Pytest and CI**
   - Update testpaths/markers in pyproject.toml.
   - Ensure `pytest tests/unit/`, `pytest tests/integration_tests/`, and `pytest -m unit` / `pytest -m integration` work as expected.

---

## Part 6: Out of Scope (No Code in This Plan)

- Actual code changes to application or tests (this plan is structure and test-case specification only).
- E2E tests with real Neo4j/Celery/GitHub (optional future work).
- Performance or stress tests for parsing (covered by existing stress test patterns if needed).
- Changes to validator implementation (only test current and, if fixed, new behavior).

---

*Document version: 1.0. No code changes are part of this plan.*
