# Testing Structure

This directory contains all tests for the Potpie application, organized into two main categories:

## Directory Structure

```text
tests/
├── conftest.py              # Root config (minimal, markers only)
├── unit/                    # Unit tests (no external dependencies)
│   ├── conftest.py         # Minimal conftest for unit tests
│   └── parsing/
│       └── utils/
│           ├── test_content_hash.py
│           └── test_encoding_detector.py
└── integration/             # Integration tests (requires services)
    ├── conftest.py         # Full conftest with DB, Redis, etc.
    └── [integration tests]
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose:** Fast, isolated tests that verify individual functions/classes without external dependencies.

**Requirements:** None - can run without PostgreSQL, Redis, Neo4j, or GitHub tokens.

**Location:** All unit tests go in `tests/unit/` subdirectories.

**Marker:** Use `@pytest.mark.unit` decorator.

**Running:**
```bash
# Run only unit tests
pytest tests/unit/ -v

# Run all tests marked as unit
pytest -m unit -v
```

**Example:**
```python
import pytest
from app.modules.parsing.utils.content_hash import generate_content_hash

@pytest.mark.unit
def test_content_hash():
    result = generate_content_hash("test")
    assert result is not None
```

### Integration Tests (`tests/integration/`)

**Purpose:** Tests that verify interactions between components and external services.

**Requirements:** 
- PostgreSQL (via `POSTGRES_SERVER` env var) - **Required**
- GitHub tokens for live API tests (via `GH_TOKEN_LIST` env var) - **Optional** (only for tests marked `@pytest.mark.github_live`)
- Private repo name (via `PRIVATE_TEST_REPO_NAME` env var) - **Optional** (only for private repo tests)

**Note:** Redis and Neo4j are mocked in integration tests and do not require actual services.

**Location:** All integration tests go in `tests/integration/` subdirectories.

**Marker:** Use `@pytest.mark.integration` decorator.

**Running:**
```bash
# Run only integration tests
pytest tests/integration/ -v

# Run all tests marked as integration
pytest -m integration -v
```

## Why This Structure?

Previously, **all** tests required a full environment setup because `tests/conftest.py` imported the FastAPI app at the module level. This meant even simple unit tests like `assert 1+1==2` would fail if PostgreSQL wasn't running.

By separating the test infrastructure:

1. **Faster Development:** Unit tests run in milliseconds without service startup.
2. **Lower Barrier:** Contributors can write/run unit tests without complex setup.
3. **Better CI:** CI can run unit tests quickly on every commit, integration tests on PR.
4. **Clearer Purpose:** Test organization reflects testing philosophy.

## Writing Tests

### For Unit Tests:

1. Place test file in `tests/unit/` matching the module structure
2. Import only the specific function/class being tested
3. Don't use database fixtures, Redis, or external services
4. Mark with `@pytest.mark.unit`
5. Test should be deterministic and fast (< 100ms)

### For Integration Tests:

1. Place test file in `tests/integration/`
2. Use fixtures from `tests/integration/conftest.py` (db_session, github_service, etc.)
3. Mark with `@pytest.mark.integration`
4. Ensure `.env` has all required environment variables

## Migration Guide

If you have existing tests:

1. **Determine if test is unit or integration:**
   - Uses `db_session`, `async_db_session`, `github_service_with_fake_redis`, or `client`? → Integration
   - Imports only pure functions, no external deps? → Unit

2. **Move to appropriate directory:**
   - Unit → `tests/unit/[module_path]/test_[name].py`
   - Integration → `tests/integration/test_[name].py`

3. **Add appropriate marker:**
   - Unit → `@pytest.mark.unit`
   - Integration → `@pytest.mark.integration`

## Running Tests

```bash
# All tests
pytest tests/ -v

# Only unit tests (fast, no services needed)
pytest tests/unit/ -v
pytest -m unit -v

# Only integration tests (requires services)
pytest tests/integration/ -v
pytest -m integration -v

# Specific test file
pytest tests/unit/parsing/utils/test_content_hash.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## CI/CD Recommendations

```yaml
# Fast checks on every push
unit-tests:
  script:
    - pytest tests/unit/ -v --cov=app

# Full validation on PR
integration-tests:
  script:
    - docker-compose up -d postgres
    - pytest tests/integration/ -v
```

## Troubleshooting

**Problem:** Unit tests fail with database errors

**Solution:** Ensure you're running from `tests/unit/` directory or using `-m unit` marker. Check that the test doesn't import app fixtures.

**Problem:** Integration tests can't find fixtures

**Solution:** Ensure `tests/integration/conftest.py` exists and contains necessary fixtures. Check `.env` has all required variables.

**Problem:** "No tests collected"

**Solution:** Ensure test files start with `test_` and functions start with `test_`. Check that `__init__.py` doesn't exist in test directories (pytest doesn't need them).
