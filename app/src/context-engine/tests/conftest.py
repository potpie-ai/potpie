"""Auto-mark engine tests by directory.

The root CI runner (``scripts/run_tests.py``) selects tests with ``-m unit``
and ``-m integration``. Most engine test files do not carry an explicit
``pytestmark``; their phase is implied by location (``tests/unit/`` vs
``tests/integration/``). This conftest applies the matching marker at
collection time so the root runner discovers them with no per-file edits.

Files that already declare ``pytestmark`` are left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
_UNIT_DIR = _TESTS_ROOT / "unit"
_INTEGRATION_DIR = _TESTS_ROOT / "integration"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            path.relative_to(_UNIT_DIR)
            phase = "unit"
        except ValueError:
            try:
                path.relative_to(_INTEGRATION_DIR)
                phase = "integration"
            except ValueError:
                continue
        item.add_marker(getattr(pytest.mark, phase))
