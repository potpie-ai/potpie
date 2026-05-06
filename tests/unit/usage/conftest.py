"""Enable BILLING_ENABLED for the usage test package.

Production now opts-out of dodo/stripe-potpie by default. These tests still
exercise the integration logic, so we flip the flag on for their scope.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BILLING_ENABLED", "true")
