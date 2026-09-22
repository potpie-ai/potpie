from __future__ import annotations

import pytest

from potpie_context_core.repository_identity import normalize_repo_ref


@pytest.mark.parametrize(
    "value",
    (
        "git@github.com:Acme/Shop.git",
        "https://github.com/acme/shop",
        "ssh://git@github.com/ACME/SHOP.git",
        "github.com/acme/shop.git",
        "github.com/acme/shop.git/",
        "repo:github.com/acme/shop",
    ),
)
def test_remote_spellings_normalize_to_one_identity(value: str) -> None:
    assert normalize_repo_ref(value) == "github.com/acme/shop"
